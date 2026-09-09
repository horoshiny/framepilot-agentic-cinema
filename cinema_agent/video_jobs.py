import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable
from uuid import uuid4

from .schemas import (
    VideoApproval,
    VideoApprovalRequest,
    VideoAllowanceStatus,
    ControlledAuthorizationStatus,
    VideoFailure,
    VideoJob,
    VideoJobKind,
    VideoJobRequest,
    VideoOutput,
)
from .video_provider import (
    MockVideoProvider,
    ProviderError,
    ProviderOperation,
    VideoProvider,
    VideoProviderRequest,
    VertexVeoProvider,
    build_veo_prompt,
    decode_image_data_url,
    effective_veo_model,
    SUPPORTED_STORYBOARD_MIME_TYPES,
    validate_veo_model,
)
from .video_ledger import VideoLedger


def video_cache_key(
    kind: VideoJobKind,
    scene_key: str,
    source_signature: str,
    replacement_for_job_id: str | None = None,
) -> str:
    payload = json.dumps(
        {
            "namespace": "video-job",
            "kind": kind,
            "scene_key": scene_key,
            "source_signature": source_signature,
            **(
                {"replacement_for_job_id": replacement_for_job_id}
                if replacement_for_job_id
                else {}
            ),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
    return "video-job:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


class VideoJobError(Exception):
    status_code = 400
    code = "video_job_error"


class ApprovalRequired(VideoJobError):
    code = "approval_required"


class ApprovalInvalid(VideoJobError):
    code = "approval_invalid"


class GenerationDisabled(VideoJobError):
    status_code = 503
    code = "generation_disabled"


class GenerationLimitReached(VideoJobError):
    status_code = 429
    code = "generation_limit_reached"


class ReplacementCreditUnavailable(VideoJobError):
    status_code = 409
    code = "replacement_credit_unavailable"


class ControlledAuthorizationUnavailable(VideoJobError):
    status_code = 409
    code = "controlled_authorization_unavailable"


class ActiveJobExists(VideoJobError):
    status_code = 409
    code = "active_job_exists"


class DuplicateSceneKind(VideoJobError):
    status_code = 409
    code = "duplicate_scene_kind"


class JobNotFound(VideoJobError):
    status_code = 404
    code = "job_not_found"


class RevisionPrerequisiteFailed(VideoJobError):
    code = "revision_prerequisite_failed"


class ProviderConfigurationError(VideoJobError):
    status_code = 503
    code = "provider_configuration_error"


class InvalidImage(VideoJobError):
    code = "invalid_image"


class StoryboardImageMissing(VideoJobError):
    code = "storyboard_image_missing"


SUBMISSION_UNKNOWN_MESSAGE = "Submission status uncertain. No retry sent; allowance remains reserved."


VIDEO_FAILURE_CODES = {
    "approval_required",
    "generation_disabled",
    "generation_limit_reached",
    "active_job_exists",
    "duplicate_scene_kind",
    "provider_submission_failed",
    "submission_unknown",
    "provider_failed",
    "generation_timeout",
    "video_retrieval_failed",
    "invalid_image",
}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_nonnegative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return max(0.1, float(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def _failure(
    code: str,
    message: str,
    *,
    provider_error: ProviderError | None = None,
) -> VideoFailure:
    return VideoFailure(
        code=code,
        message=message,
        provider_error_code=provider_error.provider_error_code if provider_error else None,
        provider_error_status=provider_error.provider_error_status if provider_error else None,
        provider_error_category=provider_error.provider_error_category if provider_error else None,
        provider_error_message=provider_error.provider_error_message if provider_error else None,
    )


@dataclass(frozen=True)
class _ApprovalRecord:
    client_id: str
    fingerprint: str
    expires_at: float


@dataclass(frozen=True)
class _ImageRecord:
    client_id: str
    image_bytes: bytes
    mime_type: str


class MockVideoJobService:
    """Provider-backed job lifecycle with a deterministic, process-local store."""

    def __init__(
        self,
        *,
        provider: VideoProvider | None = None,
        generation_limit: int | None = None,
        global_limit: int | None = None,
        timeout_seconds: float | None = None,
        approval_ttl_seconds: float | None = None,
        poll_interval_seconds: float | None = None,
        now: Callable[[], float] | None = None,
        ledger: VideoLedger | None = None,
        ledger_path: str | None = None,
    ):
        self.provider_name = os.getenv("VIDEO_GENERATION_PROVIDER", "mock").strip().lower()
        self.provider = provider
        self.generation_limit = generation_limit or _env_int("VIDEO_GENERATION_LIMIT", 2)
        self.global_limit = global_limit or _env_int("VIDEO_GENERATION_GLOBAL_LIMIT", 25)
        self.real_global_limit = _env_nonnegative_int("MAX_REAL_VEO_GENERATIONS_GLOBAL", 1)
        self.real_per_ip_limit = _env_nonnegative_int("MAX_REAL_VEO_GENERATIONS_PER_IP", 1)
        self.real_director_global_limit = _env_nonnegative_int(
            "MAX_REAL_DIRECTORS_CUT_GENERATIONS_GLOBAL", 0
        )
        self.timeout_seconds = timeout_seconds or _env_float(
            "VIDEO_GENERATION_TIMEOUT_SECONDS",
            600,
        )
        self.approval_ttl_seconds = approval_ttl_seconds or _env_float("VIDEO_APPROVAL_TTL_SECONDS", 600)
        self.poll_interval_seconds = poll_interval_seconds or _env_float("VIDEO_POLL_INTERVAL_SECONDS", 1)
        self._now = now or time.monotonic
        self._jobs: dict[str, VideoJob] = {}
        self._owners: dict[str, str] = {}
        self._fingerprints: dict[str, str] = {}
        self._operations: dict[str, ProviderOperation] = {}
        self._next_poll_at: dict[str, float] = {}
        self._video_bytes: dict[str, tuple[bytes, str]] = {}
        self._video_paths: dict[str, str] = {}
        self._output_references: dict[str, str] = {}
        self._critique_contexts: dict[str, dict] = {}
        self._approvals: dict[str, _ApprovalRecord] = {}
        self._images: dict[str, _ImageRecord] = {}
        self._dedupe: dict[tuple[str, str], str] = {}
        self._scene_kinds: dict[tuple[str, str, VideoJobKind], str] = {}
        self._client_counts: dict[str, int] = {}
        self._global_count = 0
        self._lock = RLock()
        self.ledger = ledger or VideoLedger(
            ledger_path,
            global_limit=self.real_global_limit,
            per_ip_limit=self.real_per_ip_limit,
            director_cut_global_limit=self.real_director_global_limit,
        )
        self._restore_durable_jobs()

    @property
    def kill_switch_enabled(self) -> bool:
        return _env_bool("VIDEO_GENERATION_KILL_SWITCH", False)

    @property
    def veo_allowed(self) -> bool:
        return os.getenv("ALLOW_VEO_GENERATION", "false") == "true"

    def allowance_status(self, client_id: str, client_ip: str | None = None) -> VideoAllowanceStatus:
        snapshot = self.ledger.allowance(client_id, client_ip)
        return VideoAllowanceStatus(**snapshot.__dict__)

    def create_controlled_test_authorization(
        self,
        *,
        client_id: str,
        scene_key: str,
        source_signature: str,
        model: str,
    ) -> ControlledAuthorizationStatus:
        provider = self._ensure_provider_available()
        provider_model = self._validate_provider_model(provider)
        if provider.name != "vertex" or model != provider_model:
            raise ControlledAuthorizationUnavailable(
                "The controlled test authorization is available only for the active Vertex model."
            )
        try:
            authorization = self.ledger.create_controlled_test_authorization(
                client_id=client_id,
                scene_key=scene_key,
                source_signature=source_signature,
                model=model,
                authorization_id=f"test-auth-{uuid4().hex}",
                created_at=self._now(),
            )
        except ValueError as exc:
            raise ControlledAuthorizationUnavailable(str(exc)) from exc
        return self._controlled_status(authorization)

    def controlled_test_authorization_status(
        self,
        *,
        client_id: str,
        scene_key: str,
        source_signature: str,
        authorization_id: str | None,
    ) -> ControlledAuthorizationStatus:
        if not authorization_id or self.provider_name != "vertex":
            return ControlledAuthorizationStatus()
        authorization = self.ledger.controlled_test_authorization(
            authorization_id,
            client_id=client_id,
            scene_key=scene_key,
            source_signature=source_signature,
            model=effective_veo_model(),
        )
        return self._controlled_status(authorization)

    @staticmethod
    def _controlled_status(authorization) -> ControlledAuthorizationStatus:
        if authorization is None:
            return ControlledAuthorizationStatus()
        return ControlledAuthorizationStatus(
            authorization_id=authorization.authorization_id,
            available=authorization.state == "available",
            scene_key=authorization.scene_key,
            source_signature=authorization.source_signature,
            model=authorization.model,
        )

    def request_approval(
        self,
        request: VideoApprovalRequest,
        client_id: str,
        client_ip: str | None = None,
    ) -> VideoApproval:
        provider = self._ensure_provider_available()
        provider_model = self._validate_provider_model(provider)
        with self._lock:
            self._validate_image_handle(
                request.image_handle,
                client_id,
                required=provider.name == "vertex",
            )
            if request.kind == "director_cut":
                self._require_approved_first_cut(request.first_cut_job_id, client_id)
            if request.replacement_for_job_id:
                if request.kind != "first_cut" or provider.name != "vertex":
                    raise ReplacementCreditUnavailable(
                        "The authorized replacement is available only for a Vertex First Cut."
                    )
                try:
                    self.ledger.validate_replacement_authorization(
                        request.replacement_for_job_id,
                        client_id,
                        require_available=True,
                    )
                except ValueError as exc:
                    raise ReplacementCreditUnavailable(str(exc)) from exc
            if request.controlled_authorization_id:
                if (
                    request.kind != "first_cut"
                    or provider.name != "vertex"
                    or request.replacement_for_job_id
                    or not self.ledger.controlled_test_authorization(
                        request.controlled_authorization_id,
                        client_id=client_id,
                        scene_key=request.scene_key,
                        source_signature=request.source_signature,
                        model=provider_model,
                        require_available=True,
                    )
                ):
                    raise ControlledAuthorizationUnavailable(
                        "The controlled authorization is unavailable for this client or scene."
                    )
            approval_id = f"approval-{uuid4().hex}"
            expires_at = self._now() + self.approval_ttl_seconds
            self._approvals[approval_id] = _ApprovalRecord(
                client_id=client_id,
                fingerprint=self._request_fingerprint(request),
                expires_at=expires_at,
            )
            return VideoApproval(
                kind=request.kind,
                scene_key=request.scene_key,
                source_signature=request.source_signature,
                first_cut_job_id=request.first_cut_job_id,
                replacement_for_job_id=request.replacement_for_job_id,
                approval_id=approval_id,
                expires_at=expires_at,
                model=provider_model,
                controlled_authorization_id=request.controlled_authorization_id,
            )

    def register_image(self, image_data_url: str | None, client_id: str) -> str | None:
        """Keep decoded upload bytes only in process memory, behind a client-bound handle."""
        image_bytes, mime_type = decode_image_data_url(image_data_url)
        if image_bytes is None or mime_type is None:
            return None
        handle = f"image-{uuid4().hex}"
        with self._lock:
            self._images[handle] = _ImageRecord(client_id, image_bytes, mime_type)
        return handle

    def image_available(self, image_handle: str, client_id: str) -> bool:
        with self._lock:
            record = self._images.get(image_handle)
            return bool(record and record.client_id == client_id)

    def create(
        self,
        request: VideoJobRequest,
        client_id: str,
        client_ip: str | None = None,
    ) -> VideoJob:
        client_ip = client_ip or client_id
        if not request.approved:
            raise ApprovalRequired("Explicit approval is required before a video job is created.")
        fingerprint = self._request_fingerprint(request)
        replacement_for_job_id = request.replacement_for_job_id
        cache_key = video_cache_key(
            request.kind,
            request.scene_key,
            request.source_signature,
            replacement_for_job_id,
        )
        cached = self._cached_job(client_id, cache_key, fingerprint)
        if cached is not None:
            return cached.model_copy(update={"deduplicated": True})
        provider = self._ensure_provider_available()
        provider_model = self._validate_provider_model(provider)
        if replacement_for_job_id:
            if request.kind != "first_cut" or provider.name != "vertex":
                raise ReplacementCreditUnavailable(
                    "The authorized replacement is available only for a Vertex First Cut."
                )
            try:
                self.ledger.validate_replacement_authorization(
                    replacement_for_job_id,
                    client_id,
                    require_available=True,
                )
            except ValueError as exc:
                raise ReplacementCreditUnavailable(str(exc)) from exc
        if request.controlled_authorization_id:
            if (
                request.kind != "first_cut"
                or provider.name != "vertex"
                or replacement_for_job_id
                or not self.ledger.controlled_test_authorization(
                    request.controlled_authorization_id,
                    client_id=client_id,
                    scene_key=request.scene_key,
                    source_signature=request.source_signature,
                    model=provider_model,
                    require_available=True,
                )
            ):
                raise ControlledAuthorizationUnavailable(
                    "The controlled authorization is unavailable for this client or scene."
                )
        if provider.name == "vertex":
            self._preflight_vertex_provider(provider)
        provider_request = self._provider_request(request, client_id)
        with self._lock:
            cached = self._cached_job(client_id, cache_key, fingerprint)
            if cached is not None:
                return cached.model_copy(update={"deduplicated": True})
            self._consume_approval(request, client_id, fingerprint)
            if request.kind == "director_cut":
                self._require_approved_first_cut(request.first_cut_job_id, client_id)
            scene_key = (client_id, request.scene_key, request.kind)
            if not replacement_for_job_id and scene_key in self._scene_kinds:
                raise DuplicateSceneKind("This scene already has a job for that cut.")
            if provider.name == "vertex":
                job_id = f"video-{uuid4().hex}"
                critique_context = self._critique_context(request, client_id)
                admission = self.ledger.begin_submission(
                    job_id=job_id,
                    client_id=client_id,
                    client_ip=client_ip,
                    kind=request.kind,
                    scene_key=request.scene_key,
                    source_signature=request.source_signature,
                    provider=provider.name,
                    model=provider_model,
                    created_at=self._now(),
                    request_fingerprint=fingerprint,
                    cache_key=cache_key,
                    critique_context_json=json.dumps(
                        {
                            "screenplay": critique_context["screenplay"],
                            "creative_intent": critique_context["creative_intent"],
                            "shot_plan": critique_context["shot_plan"],
                        },
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                    storyboard_bytes=critique_context["storyboard_bytes"],
                    storyboard_mime_type=critique_context["storyboard_mime_type"],
                    replacement_for_job_id=replacement_for_job_id,
                    controlled_authorization_id=request.controlled_authorization_id,
                )
                if not admission.allowed:
                    if admission.reason == "cached" and admission.existing_job is not None:
                        cached_job = self._job_from_row(admission.existing_job)
                        self._remember_job(cached_job, admission.existing_job)
                        return cached_job.model_copy(update={"deduplicated": True})
                    if admission.reason == "controlled_authorization_unavailable":
                        raise ControlledAuthorizationUnavailable(
                            "The controlled authorization was already reserved or consumed."
                        )
                    self._raise_ledger_admission(admission.reason)
            else:
                if self._has_active_job(client_id):
                    raise ActiveJobExists("Only one video job may be active for this client.")
                if (
                    self._client_counts.get(client_ip, 0) >= self.generation_limit
                    or self._global_count >= self.global_limit
                ):
                    raise GenerationLimitReached("Video generation limit reached for this client or project.")
                job_id = f"video-{uuid4().hex}"
                critique_context = self._critique_context(request, client_id)

            now = self._now()
            job = VideoJob(
                job_id=job_id,
                kind=request.kind,
                status="queued",
                scene_key=request.scene_key,
                source_signature=request.source_signature,
                provider=provider.name,
                created_at=now,
                updated_at=now,
                model=provider_model,
                replacement_for_job_id=replacement_for_job_id,
            )
            self._jobs[job.job_id] = job
            self._owners[job.job_id] = client_id
            self._fingerprints[job.job_id] = fingerprint
            self._dedupe[(client_id, cache_key)] = job.job_id
            if not replacement_for_job_id:
                self._scene_kinds[scene_key] = job.job_id
            self._critique_contexts[job.job_id] = critique_context
            if provider.name != "vertex":
                self._client_counts[client_ip] = self._client_counts.get(client_ip, 0) + 1
                self._global_count += 1
            try:
                operation = provider.submit(provider_request)
            except ProviderError as exc:
                if provider.name == "vertex":
                    if exc.uncertain or exc.code == "submission_unknown":
                        operation_id = exc.operation_id
                        if operation_id:
                            self._operations[job.job_id] = ProviderOperation(operation_id)
                            self._next_poll_at[job.job_id] = self._now()
                        self.ledger.mark_submission_unknown(
                            job.job_id,
                            operation_id,
                            "submission_unknown",
                            SUBMISSION_UNKNOWN_MESSAGE,
                            self._now(),
                        )
                        return self._update(
                            job,
                            status="submission_unknown",
                            error=_failure("submission_unknown", SUBMISSION_UNKNOWN_MESSAGE),
                        )
                    release_code = exc.code if exc.code in VIDEO_FAILURE_CODES else "provider_failed"
                    self.ledger.release_submission(job.job_id, release_code, str(exc), self._now())
                return self._fail(job, exc.code, str(exc), provider_error=exc)
            except Exception:
                if provider.name == "vertex":
                    self.ledger.mark_submission_unknown(
                        job.job_id,
                        None,
                        "submission_unknown",
                        SUBMISSION_UNKNOWN_MESSAGE,
                        self._now(),
                    )
                    return self._update(
                        job,
                        status="submission_unknown",
                        error=_failure("submission_unknown", SUBMISSION_UNKNOWN_MESSAGE),
                    )
                raise
            self._operations[job.job_id] = operation
            self._next_poll_at[job.job_id] = now
            if provider.name == "vertex":
                self.ledger.accept_submission(job.job_id, operation.operation_id, self._now())
            return job

    def get(self, job_id: str, client_id: str) -> VideoJob:
        with self._lock:
            job = self._owned_job(job_id, client_id)
            if job.status in {"completed", "failed"}:
                return job
            operation = self._operations.get(job_id)
            if job.status == "submission_unknown" and operation is None:
                return job
            now = self._now()
            if job.status != "submission_unknown" and now - job.created_at >= self.timeout_seconds:
                return self._fail(job, "generation_timeout", "Video generation exceeded its deadline.")
            if now < self._next_poll_at.get(job_id, now):
                return job
            if operation is None:
                if job.status == "submission_unknown":
                    return job
                return self._fail(job, "provider_failed", "Video generation operation is unavailable.")
            try:
                poll = self._ensure_provider().poll(operation)
            except ProviderError as exc:
                if job.status == "submission_unknown":
                    self._next_poll_at[job_id] = now + self.poll_interval_seconds
                    return self._update(
                        job,
                        status="submission_unknown",
                        error=_failure("submission_unknown", SUBMISSION_UNKNOWN_MESSAGE),
                    )
                return self._fail(job, exc.code, str(exc))
            self._next_poll_at[job_id] = now + self.poll_interval_seconds
            if job.status == "submission_unknown":
                self.ledger.accept_submission(job.job_id, operation.operation_id, self._now())
            if poll.status == "failed":
                error = poll.error or ProviderError("provider_failed", "Video generation failed.")
                return self._fail(job, error.code, str(error), provider_error=error)
            if poll.status != "completed":
                return self._update(job, status="generating")
            if poll.output_reference:
                self._output_references[job_id] = poll.output_reference
                job = self._update(job)
            try:
                video = self._ensure_provider().retrieve(operation)
            except ProviderError as exc:
                return self._fail(job, exc.code, str(exc), provider_error=exc)
            if video.video_bytes is not None:
                self._video_bytes[job_id] = (video.video_bytes, video.mime_type)
                if self.ledger.path != ":memory:":
                    video_directory = os.path.join(os.path.dirname(self.ledger.path), "video_outputs")
                    os.makedirs(video_directory, exist_ok=True)
                    video_path = os.path.join(video_directory, f"{job_id}.mp4")
                    with open(video_path, "wb") as video_file:
                        video_file.write(video.video_bytes)
                    self._video_paths[job_id] = video_path
                url = f"/api/video-jobs/{job_id}/video"
            else:
                url = video.url or "/static/demo-generation.mp4"
            output = VideoOutput(
                url=url,
                label="Demo generation" if self.provider_name == "mock" else "Vertex AI Veo",
                source="bundled_fixture" if self.provider_name == "mock" else "vertex_veo",
                kind=job.kind,
            )
            return self._update(job, status="completed", output=output)

    def latest_completed(self, client_id: str, *, kind: VideoJobKind = "first_cut") -> VideoJob | None:
        with self._lock:
            row = self.ledger.latest_completed_job(client_id, kind=kind)
            if not row:
                return None
            job = self._job_from_row(row)
            self._remember_job(job, row)
            if row.get("video_path"):
                self._video_paths[job.job_id] = row["video_path"]
            if row.get("output_reference"):
                self._output_references[job.job_id] = row["output_reference"]
            if row.get("critique_context_json"):
                self._critique_contexts[job.job_id] = {
                    **json.loads(row["critique_context_json"]),
                    "storyboard_bytes": row.get("storyboard_bytes"),
                    "storyboard_mime_type": row.get("storyboard_mime_type"),
                }
            return job.model_copy(update={"scene_snapshot": self._scene_snapshot(row, job)})

    def video_bytes(self, job_id: str, client_id: str) -> tuple[bytes, str]:
        with self._lock:
            job = self._owned_job(job_id, client_id)
            if job.status != "completed":
                raise JobNotFound("Video output is not available for this client.")
            if job_id not in self._video_bytes:
                video_path = self._video_paths.get(job_id)
                if not video_path and job.provider == "mock":
                    video_path = str(Path(__file__).parent.parent / "static" / "demo-generation.mp4")
                if not video_path or not os.path.exists(video_path):
                    raise JobNotFound("Video output is not available for this client.")
                with open(video_path, "rb") as video_file:
                    self._video_bytes[job_id] = (video_file.read(), "video/mp4")
            return self._video_bytes[job_id]

    def storyboard_bytes(self, job_id: str, client_id: str) -> tuple[bytes, str]:
        with self._lock:
            row = self.ledger.job_for_client(job_id, client_id)
            if not row or row.get("status") != "completed":
                raise JobNotFound("Storyboard image was not found for this client.")
            image_bytes = row.get("storyboard_bytes")
            mime_type = row.get("storyboard_mime_type")
            if not image_bytes or not mime_type:
                raise JobNotFound("Storyboard image was not stored for this client.")
            return image_bytes, mime_type

    def get_critique_context(self, job_id: str, client_id: str) -> dict | None:
        with self._lock:
            self._owned_job(job_id, client_id)
            context = self._critique_contexts.get(job_id)
            return dict(context) if context else None

    def approve_revision(self, job_id: str, client_id: str) -> VideoJob:
        with self._lock:
            job = self._owned_job(job_id, client_id)
            job = self._advance_if_needed(job)
            if job.kind != "first_cut" or job.status != "completed":
                raise RevisionPrerequisiteFailed("Revision approval requires a completed First Cut.")
            return self._update(job, revision_approved=True)

    def _cached_job(self, client_id: str, cache_key: str, fingerprint: str) -> VideoJob | None:
        existing_id = self._dedupe.get((client_id, cache_key))
        if existing_id and self._fingerprints.get(existing_id) == fingerprint:
            return self._jobs[existing_id]
        if self.provider_name != "vertex":
            return None
        row = self.ledger.cached_job(client_id, cache_key)
        if not row or row["request_fingerprint"] != fingerprint:
            return None
        job = self._job_from_row(row)
        self._remember_job(job, row)
        return job

    def _raise_ledger_admission(self, reason: str | None) -> None:
        if reason == "duplicate_scene_kind":
            raise DuplicateSceneKind("This scene already has a job for that cut.")
        if reason == "active_job":
            raise ActiveJobExists("Only one video job may be active for this client.")
        if reason == "replacement_unavailable":
            raise ReplacementCreditUnavailable(
                "The authorized replacement is unavailable for this client or job."
            )
        raise GenerationLimitReached("Real Veo generation allowance is exhausted.")

    def _restore_durable_jobs(self) -> None:
        if self.provider_name != "vertex":
            return
        for row in self.ledger.load_jobs():
            if (
                row["status"] == "failed"
                and row.get("error_code") == "generation_timeout"
                and row.get("operation_id")
            ):
                self.ledger.resume_accepted_operation(
                    row["job_id"],
                    time.monotonic(),
                )
                row["status"] = "generating"
                row["error_code"] = None
                row["error_message"] = None
                row["provider_error_code"] = None
                row["provider_error_status"] = None
                row["provider_error_category"] = None
                row["provider_error_message"] = None
            if row["status"] == "submitting":
                self.ledger.mark_submission_unknown(
                    row["job_id"],
                    row["operation_id"],
                    "submission_unknown",
                    SUBMISSION_UNKNOWN_MESSAGE,
                    time.monotonic(),
                )
                row["status"] = "submission_unknown"
                row["error_code"] = "submission_unknown"
                row["error_message"] = SUBMISSION_UNKNOWN_MESSAGE
            job = self._job_from_row(row)
            self._remember_job(job, row)
            if row.get("critique_context_json"):
                self._critique_contexts[job.job_id] = {
                    **json.loads(row["critique_context_json"]),
                    "storyboard_bytes": row.get("storyboard_bytes"),
                    "storyboard_mime_type": row.get("storyboard_mime_type"),
                }
            if row.get("video_path"):
                self._video_paths[job.job_id] = row["video_path"]
            if row.get("output_reference"):
                self._output_references[job.job_id] = row["output_reference"]
            if row["operation_id"] and job.status in {"queued", "generating", "submission_unknown"}:
                self._operations[job.job_id] = ProviderOperation(row["operation_id"])
                self._next_poll_at[job.job_id] = self._now()

    def _job_from_row(self, row: dict) -> VideoJob:
        output = json.loads(row["output_json"]) if row.get("output_json") else None
        return VideoJob(
            job_id=row["job_id"],
            kind=row["kind"],
            status=row["status"] if row["status"] != "submitting" else "failed",
            scene_key=row["scene_key"],
            source_signature=row["source_signature"],
            provider=row["provider"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            model=row.get("model"),
            replacement_for_job_id=row.get("replacement_for_job_id"),
            output=output,
            error=(
                _failure(
                    row["error_code"],
                    row["error_message"],
                    provider_error=ProviderError(
                        row["error_code"],
                        row["error_message"],
                        provider_error_code=row.get("provider_error_code"),
                        provider_error_status=row.get("provider_error_status"),
                        provider_error_category=row.get("provider_error_category"),
                        provider_error_message=row.get("provider_error_message"),
                    )
                    if any(
                        row.get(field)
                        is not None
                        for field in (
                            "provider_error_code",
                            "provider_error_status",
                            "provider_error_category",
                            "provider_error_message",
                        )
                    )
                    else None,
                )
                if row.get("error_code") and row.get("error_message")
                else None
            ),
            revision_approved=bool(row["revision_approved"]),
        )

    @staticmethod
    def _scene_snapshot(row: dict, job: VideoJob) -> dict | None:
        raw_context = row.get("critique_context_json")
        if not raw_context:
            return None
        try:
            context = json.loads(raw_context)
        except (TypeError, ValueError):
            return None
        if not isinstance(context, dict):
            return None
        return {
            "scene_key": job.scene_key,
            "source_signature": job.source_signature,
            "screenplay": context.get("screenplay"),
            "creative_intent": context.get("creative_intent"),
            "direction_response": context.get("direction_response"),
            "shot_plan": context.get("shot_plan"),
            "storyboard_url": (
                f"/api/video-jobs/{job.job_id}/storyboard"
                if row.get("storyboard_bytes")
                else None
            ),
        }

    def _remember_job(self, job: VideoJob, row: dict | None = None) -> None:
        row = row or {}
        self._jobs[job.job_id] = job
        self._owners[job.job_id] = row.get("client_id", self._owners.get(job.job_id, ""))
        self._fingerprints[job.job_id] = row.get(
            "request_fingerprint", self._fingerprints.get(job.job_id, "")
        )
        cache_key = row.get("cache_key")
        if cache_key:
            self._dedupe[(self._owners[job.job_id], cache_key)] = job.job_id
        if not job.replacement_for_job_id:
            self._scene_kinds[(self._owners[job.job_id], job.scene_key, job.kind)] = job.job_id

    def _ensure_provider_available(self) -> VideoProvider:
        if self.kill_switch_enabled:
            raise GenerationDisabled("Video generation is disabled by the generation kill switch.")
        if self.provider_name not in {"mock", "vertex"}:
            raise ProviderConfigurationError("VIDEO_GENERATION_PROVIDER must be mock or vertex.")
        if self.provider_name == "vertex" and not self.veo_allowed:
            raise GenerationDisabled("Vertex Veo generation requires ALLOW_VEO_GENERATION=true.")
        return self._ensure_provider()

    def _ensure_provider(self) -> VideoProvider:
        if self.provider is None:
            self.provider = MockVideoProvider() if self.provider_name == "mock" else VertexVeoProvider()
        return self.provider

    @staticmethod
    def _preflight_vertex_provider(provider: VideoProvider) -> None:
        preflight = getattr(provider, "preflight", None)
        if not callable(preflight):
            return
        try:
            preflight()
        except ProviderError as exc:
            raise ProviderConfigurationError(str(exc)) from exc
        except Exception as exc:
            raise ProviderConfigurationError("Vertex output storage configuration is invalid.") from exc

    @staticmethod
    def _validate_provider_model(provider: VideoProvider) -> str | None:
        if getattr(provider, "name", None) != "vertex":
            return None
        model = getattr(provider, "model", None)
        try:
            return validate_veo_model(model or effective_veo_model())
        except ProviderError as exc:
            raise ProviderConfigurationError(str(exc)) from exc

    def _provider_request(self, request: VideoJobRequest, client_id: str) -> VideoProviderRequest:
        image_record = self._images.get(request.image_handle) if request.image_handle else None
        if self.provider_name == "vertex" and not request.image_handle:
            raise StoryboardImageMissing(
                "A storyboard image is required for Vertex First Cut generation."
            )
        if request.image_handle and image_record is None:
            raise InvalidImage("Storyboard image is unavailable.")
        if image_record and image_record.client_id != client_id:
            raise InvalidImage("Storyboard image is not available for this client.")
        if self.provider_name == "vertex":
            if image_record is None or not image_record.image_bytes:
                raise InvalidImage("Storyboard image data is invalid.")
            if image_record.mime_type not in SUPPORTED_STORYBOARD_MIME_TYPES:
                raise InvalidImage("Storyboard image format is not supported.")
        image_bytes = image_record.image_bytes if image_record else None
        image_mime_type = image_record.mime_type if image_record else None
        duration = min(8, max(4, round(request.shot_plan.shot.duration_seconds / 2) * 2))
        return VideoProviderRequest(
            kind=request.kind,
            prompt=build_veo_prompt(
                request.screenplay,
                request.creative_intent,
                request.shot_plan,
                request.kind,
                request.video_critique,
            ),
            image_bytes=image_bytes,
            image_mime_type=image_mime_type,
            duration_seconds=duration,
        )

    def _critique_context(self, request: VideoJobRequest, client_id: str) -> dict:
        image_record = self._images.get(request.image_handle) if request.image_handle else None
        return {
            "screenplay": request.screenplay,
            "creative_intent": request.creative_intent,
            "shot_plan": request.shot_plan.model_dump(),
            "storyboard_bytes": image_record.image_bytes if image_record and image_record.client_id == client_id else None,
            "storyboard_mime_type": image_record.mime_type
            if image_record and image_record.client_id == client_id
            else None,
        }

    def _request_fingerprint(self, request: VideoApprovalRequest | VideoJobRequest) -> str:
        payload = request.model_dump(exclude={"approved", "approval_id"})
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _validate_image_handle(
        self,
        image_handle: str | None,
        client_id: str,
        *,
        required: bool = False,
    ) -> None:
        if not image_handle:
            if required:
                raise StoryboardImageMissing(
                    "A storyboard image is required for Vertex First Cut generation."
                )
            return
        record = self._images.get(image_handle)
        if record is None:
            raise InvalidImage("Storyboard image is unavailable.")
        if record.client_id != client_id:
            raise InvalidImage("Storyboard image is not available for this client.")
        if not record.image_bytes:
            raise InvalidImage("Storyboard image data is invalid.")
        if record.mime_type not in SUPPORTED_STORYBOARD_MIME_TYPES:
            raise InvalidImage("Storyboard image format is not supported.")

    def _consume_approval(self, request: VideoJobRequest, client_id: str, fingerprint: str) -> None:
        if not request.approval_id:
            raise ApprovalRequired("A server-issued approval is required before a video job is created.")
        approval = self._approvals.get(request.approval_id)
        if (
            approval is None
            or approval.client_id != client_id
            or approval.expires_at <= self._now()
            or approval.fingerprint != fingerprint
        ):
            raise ApprovalInvalid("Approval does not match this exact video job.")
        del self._approvals[request.approval_id]

    def _require_approved_first_cut(self, job_id: str | None, client_id: str) -> None:
        if not job_id:
            raise RevisionPrerequisiteFailed(
                "Director’s Cut requires a completed First Cut and explicit revision approval."
            )
        first_cut = self._jobs.get(job_id)
        if (
            not first_cut
            or not self._owns(first_cut, client_id)
            or first_cut.kind != "first_cut"
            or self._advance_if_needed(first_cut).status != "completed"
            or not first_cut.revision_approved
        ):
            raise RevisionPrerequisiteFailed(
                "Director’s Cut requires a completed First Cut and explicit revision approval."
            )

    def _advance_if_needed(self, job: VideoJob) -> VideoJob:
        return self.get(job.job_id, self._owners[job.job_id]) if job.status not in {"completed", "failed"} else job

    def _has_active_job(self, client_id: str) -> bool:
        return any(
            self._owns(job, client_id) and job.status in {"queued", "generating"}
            for job in self._jobs.values()
        )

    def _owned_job(self, job_id: str, client_id: str) -> VideoJob:
        job = self._jobs.get(job_id)
        if not job or not self._owns(job, client_id):
            raise JobNotFound("Video job was not found for this client.")
        return job

    def _owns(self, job: VideoJob, client_id: str) -> bool:
        return self._owners.get(job.job_id) == client_id

    def _update(self, job: VideoJob, **changes) -> VideoJob:
        updated = job.model_copy(update={"updated_at": self._now(), **changes})
        self._jobs[job.job_id] = updated
        if self.provider_name == "vertex":
            self.ledger.update_job(
                job.job_id,
                status=updated.status,
                updated_at=updated.updated_at,
                revision_approved=updated.revision_approved,
                output=updated.output.model_dump() if updated.output else None,
                video_path=self._video_paths.get(job.job_id),
                output_reference=self._output_references.get(job.job_id),
                error_code=updated.error.code if updated.error else None,
                error_message=updated.error.message if updated.error else None,
                provider_error_code=updated.error.provider_error_code if updated.error else None,
                provider_error_status=updated.error.provider_error_status if updated.error else None,
                provider_error_category=updated.error.provider_error_category if updated.error else None,
                provider_error_message=updated.error.provider_error_message if updated.error else None,
            )
        return updated

    def _fail(
        self,
        job: VideoJob,
        code: str,
        message: str,
        *,
        provider_error: ProviderError | None = None,
    ) -> VideoJob:
        safe_code = code if code in VIDEO_FAILURE_CODES else "provider_failed"
        return self._update(
            job,
            status="failed",
            error=_failure(safe_code, message, provider_error=provider_error),
        )