import base64
import logging
import os
import re
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit
from uuid import uuid4

from .schemas import ShotPlan, VideoCritique, VideoJobKind


logger = logging.getLogger(__name__)
SUPPORTED_STORYBOARD_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
DEFAULT_VEO_MODEL = "veo-3.1-generate-001"
RETIRED_VEO_MODELS = frozenset({"veo-3.0-generate-001"})
VEO3_MODEL_PREFIX = "veo-3"
_SAFE_LOG_FIELD = re.compile(r"^[A-Za-z0-9_.:/-]{1,120}$")
_DATA_URL = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+", re.IGNORECASE)
_BEARER = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|api[-_ ]?key|token|secret|password|private[-_ ]?key)"
    r"\s*[:=]\s*\S+"
)
_LONG_PAYLOAD = re.compile(r"[A-Za-z0-9+/=_-]{160,}")


class ProviderError(Exception):
    """Sanitized provider failure; never contains credentials or image data."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        uncertain: bool = False,
        operation_id: str | None = None,
        provider_error_type: str | None = None,
        provider_error_code: int | None = None,
        provider_error_status: str | None = None,
        provider_error_category: str | None = None,
        provider_error_message: str | None = None,
    ):
        self.code = code
        self.uncertain = uncertain
        self.operation_id = operation_id
        self.provider_error_type = provider_error_type
        self.provider_error_code = provider_error_code
        self.provider_error_status = provider_error_status
        self.provider_error_category = provider_error_category
        self.provider_error_message = provider_error_message
        super().__init__(message)


@dataclass(frozen=True)
class ProviderErrorMetadata:
    """Bounded provider metadata safe to persist and expose to the UI."""

    error_type: str
    code: int | None
    status: str | None
    category: str | None
    message: str | None
    detail_types: tuple[str, ...] = ()
    detail_fields: tuple[str, ...] = ()


def _is_uncertain_submission_error(error: Exception) -> bool:
    error_name = type(error).__name__.lower()
    error_message = str(error).lower()
    return (
        isinstance(error, TimeoutError)
        or "timeout" in error_name
        or "timed out" in error_message
        or "deadline exceeded" in error_message
    )


@dataclass(frozen=True)
class VideoProviderRequest:
    kind: VideoJobKind
    prompt: str
    image_bytes: bytes | None
    image_mime_type: str | None
    duration_seconds: int
    aspect_ratio: str = "16:9"


@dataclass(frozen=True)
class ProviderOperation:
    operation_id: str


@dataclass(frozen=True)
class ProviderPoll:
    status: str
    error: ProviderError | None = None
    output_reference: str | None = None


@dataclass(frozen=True)
class ProviderVideo:
    url: str | None = None
    video_bytes: bytes | None = None
    mime_type: str = "video/mp4"


class VideoProvider(Protocol):
    name: str

    def submit(self, request: VideoProviderRequest) -> ProviderOperation:
        ...

    def poll(self, operation: ProviderOperation) -> ProviderPoll:
        ...

    def retrieve(self, operation: ProviderOperation) -> ProviderVideo:
        ...


def effective_veo_model(model: str | None = None) -> str:
    """Return the configured production model without exposing secret values."""
    configured = model if model is not None else os.getenv("VEO_MODEL")
    return configured.strip() if configured and configured.strip() else DEFAULT_VEO_MODEL


def validate_veo_model(model: str | None = None) -> str:
    effective_model = effective_veo_model(model)
    if effective_model in RETIRED_VEO_MODELS:
        raise ProviderError(
            "provider_configuration_failed",
            (
                f"Configured Veo model {effective_model} is retired. "
                f"Set VEO_MODEL to {DEFAULT_VEO_MODEL} or another supported model."
            ),
        )
    return effective_model


def validate_prompt_enhancement(model: str | None, enhance_prompt: bool | None = None) -> None:
    """Reject the unsupported explicit false setting for Veo 3 and 3.1."""
    effective_model = effective_veo_model(model)
    if effective_model.startswith(VEO3_MODEL_PREFIX) and enhance_prompt is False:
        raise ProviderError(
            "provider_configuration_failed",
            "Veo 3 prompt enhancement must use the provider default.",
        )


def veo_estimate_label(model: str | None = None) -> str | None:
    """Return a numeric estimate label only when deployment configuration provides one."""
    effective_model = effective_veo_model(model)
    configured = os.getenv("VEO_ESTIMATED_COST_INR", "").strip()
    try:
        amount = float(configured)
    except (TypeError, ValueError):
        amount = 0.0
    if amount > 0:
        return f"Approximate {effective_model} estimate: ₹{amount:,.2f}"
    return None


def decode_image_data_url(image_data_url: str | None) -> tuple[bytes | None, str | None]:
    if not image_data_url:
        return None, None
    if "," not in image_data_url:
        raise ProviderError("invalid_image", "Storyboard image data is invalid.")
    header, encoded = image_data_url.split(",", 1)
    mime_type = header.split(";")[0].replace("data:", "").lower()
    if mime_type not in SUPPORTED_STORYBOARD_MIME_TYPES:
        raise ProviderError("invalid_image", "Storyboard image format is not supported.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ProviderError("invalid_image", "Storyboard image data is invalid.") from exc
    if not image_bytes:
        raise ProviderError("invalid_image", "Storyboard image data is invalid.")
    if len(image_bytes) > 8 * 1024 * 1024:
        raise ProviderError("invalid_image", "Storyboard image must be smaller than 8 MB.")
    return image_bytes, mime_type


def _candidate_lines(plan: ShotPlan) -> list[str]:
    motion = plan.motion_plan
    candidates = [
        *motion.movable_characters,
        *motion.movable_objects,
        *motion.environmental_motion,
        motion.camera_movement,
    ]
    lines = [
        f"- {candidate.label}: {candidate.action}; "
        f"visual evidence: {candidate.visual_evidence}; "
        f"screenplay evidence: {candidate.screenplay_evidence}; "
        f"intent evidence: {candidate.intent_evidence}; "
        f"grounding source: {candidate.grounding_source}; "
        f"support: {candidate.support}; confidence: {candidate.confidence:.2f}"
        for candidate in candidates
    ]
    lines.extend(
        f"- REJECTED UNSUPPORTED: {candidate.label}: {candidate.action}; "
        f"screenplay evidence: {candidate.screenplay_evidence}; "
        f"intent evidence: {candidate.intent_evidence}; "
        f"grounding source: {candidate.grounding_source}"
        for candidate in motion.unsupported_actions
    )
    return lines


def build_veo_prompt(
    screenplay: str,
    creative_intent: str,
    shot_plan: ShotPlan,
    kind: VideoJobKind,
    video_critique: VideoCritique | None = None,
) -> str:
    """Build a scene-specific continuity prompt from the approved structured plan."""
    motion = shot_plan.motion_plan
    depth = "; ".join(f"{key}: {value}" for key, value in shot_plan.depth_notes.items())
    candidates = "\n".join(_candidate_lines(shot_plan)) or "- No semantic movement approved."
    relationships = "\n".join(
        f"- {item.source_id} {item.relation} {item.target_id}"
        + f": {item.action or 'no specific action'}"
        for item in shot_plan.relationships
    ) or "- No validated entity relationships reported."
    preserved = ", ".join(motion.preserved_elements)
    prohibited = ", ".join(motion.prohibited_changes) or "No additions beyond the supplied frame and plan."
    critique_lines: list[str] = []
    if kind == "director_cut" and video_critique and video_critique.status == "available":
        groups = (
            ("SUCCESSFUL ELEMENT TO PRESERVE", video_critique.requested_actions_achieved),
            ("SUPPORTED REVISION FINDING", video_critique.requested_actions_missing),
            ("SUPPORTED REVISION FINDING", video_critique.character_object_consistency),
            ("SUPPORTED REVISION FINDING", video_critique.camera_movement_pacing),
            ("SUPPORTED REVISION FINDING", video_critique.visible_artifacts),
            ("SUPPORTED REVISION FINDING", video_critique.emotional_intent_alignment),
        )
        for label, findings in groups:
            for finding in findings:
                if finding.support == "supported":
                    critique_lines.append(f"- {label}: {finding.finding}")
        for recommendation in video_critique.revision_recommendations:
            if recommendation.support == "supported":
                critique_lines.append(
                    f"- SUPPORTED REVISION RECOMMENDATION: {recommendation.recommendation}"
                )
    critique_context = "\n".join(critique_lines) or "- No supported video-critique finding; do not force a change."
    return (
        f"Create a restrained {kind.replace('_', ' ')} cinematic video for this specific scene.\n"
        f"SCREENPLAY:\n{screenplay}\n\n"
        f"CREATIVE INTENT: {creative_intent}\n"
        f"SCENE SUMMARY: {shot_plan.scene_summary}\n"
        f"VISIBLE FOCAL SUBJECT: {shot_plan.focal_subject}\n"
        f"ENVIRONMENT / ARCHITECTURE / LIGHTING NOTES: {depth}\n"
        f"APPROVED CAMERA: {shot_plan.shot.camera_motion}; "
        f"zoom {shot_plan.shot.zoom_start:.2f} to {shot_plan.shot.zoom_end:.2f}; "
        f"pan ({shot_plan.shot.pan_x:.1f}, {shot_plan.shot.pan_y:.1f})\n"
        "GROUNDING KEY: visual is confirmed by the storyboard; screenplay is explicitly named "
        "in the screenplay but may require approval because the storyboard does not confirm it; "
        "visual_and_screenplay has both sources; unsupported is excluded and prohibited.\n"
        f"APPROVED MOTION CANDIDATES:\n{candidates}\n"
        f"VALIDATED ENTITY RELATIONSHIPS:\n{relationships}\n"
        f"GROUNDING CONFLICTS REQUIRING REVIEW:\n"
        f"{chr(10).join(f'- {item}' for item in motion.grounding_conflicts) or '- None reported.'}\n"
        f"PRESERVE EXACTLY: {preserved}\n"
        f"PROHIBITED ADDITIONS OR CHANGES: {prohibited}\n"
        f"SUPPORTED VIDEO CRITIQUE FOR THIS DIRECTOR'S CUT:\n{critique_context}\n"
        "Preserve every visible narrative agentive entity, ordinary object, environment, architecture, composition, "
        "lighting relationship, identity, and continuity from the attached storyboard. "
        "Execute the bounded screenplay candidates exactly after approval, even when "
        "their visual_evidence says the local pass could not identify them. "
        "Do not invent a subject or semantic action not present in the approved evidence. "
        "Do not add cuts, camera angles, objects, text, logos, or scene transitions. "
        "Preserve successful elements and target only supported critique findings; do not guarantee "
        "that the revision will improve the result."
    )


class MockVideoProvider:
    name = "mock"

    def __init__(self, fixture_url: str = "/static/demo-generation.mp4"):
        self.fixture_url = fixture_url
        self._polls: dict[str, int] = {}

    def submit(self, request: VideoProviderRequest) -> ProviderOperation:
        operation = ProviderOperation(f"mock-op-{uuid4().hex}")
        self._polls[operation.operation_id] = 0
        return operation

    def poll(self, operation: ProviderOperation) -> ProviderPoll:
        count = self._polls.get(operation.operation_id, 0) + 1
        self._polls[operation.operation_id] = count
        return ProviderPoll("generating" if count == 1 else "completed")

    def retrieve(self, operation: ProviderOperation) -> ProviderVideo:
        return ProviderVideo(url=self.fixture_url)


class VertexVeoProvider:
    name = "vertex"

    def __init__(
        self,
        *,
        client_factory=None,
        model: str | None = None,
        storage_session_factory=None,
    ):
        self._client_factory = client_factory
        self.model = effective_veo_model(model)
        self._storage_session_factory = storage_session_factory
        self._client = None
        self._operations: dict[str, object] = {}

    @property
    def client(self):
        if self._client is None:
            if self._client_factory is None:
                from .vertex import _client

                self._client_factory = _client
            self._client = self._client_factory()
        return self._client

    def preflight(self) -> str:
        validate_veo_model(self.model)
        validate_prompt_enhancement(self.model)
        output_gcs_uri = _configured_output_gcs_uri()
        if not output_gcs_uri:
            raise ProviderError(
                "provider_configuration_failed",
                "VEO_OUTPUT_GCS_BUCKET and VEO_OUTPUT_GCS_PREFIX are required for Vertex Veo generation.",
            )
        return output_gcs_uri

    def submit(self, request: VideoProviderRequest) -> ProviderOperation:
        from google.genai import types

        image = (
            types.Image(image_bytes=request.image_bytes, mime_type=request.image_mime_type)
            if request.image_bytes
            else None
        )
        try:
            config_kwargs = {
                "number_of_videos": 1,
                "duration_seconds": request.duration_seconds,
                "aspect_ratio": request.aspect_ratio,
                "generate_audio": False,
            }
            config_kwargs["output_gcs_uri"] = self.preflight()
            operation = self.client.models.generate_videos(
                model=self.model,
                prompt=request.prompt,
                image=image,
                config=types.GenerateVideosConfig(**config_kwargs),
            )
        except Exception as exc:
            _log_submission_failure(self, request, exc)
            uncertain = _is_uncertain_submission_error(exc)
            raise ProviderError(
                "submission_unknown" if uncertain else "provider_submission_failed",
                (
                    "Submission status uncertain. No retry sent; allowance remains reserved."
                    if uncertain
                    else "Vertex video submission failed."
                ),
                uncertain=uncertain,
                operation_id=getattr(exc, "operation_id", None),
            ) from exc
        operation_id = getattr(operation, "name", None) or f"vertex-op-{uuid4().hex}"
        self._operations[operation_id] = operation
        return ProviderOperation(operation_id)

    def poll(self, operation: ProviderOperation) -> ProviderPoll:
        current = self._operations.get(operation.operation_id) or operation.operation_id
        try:
            if isinstance(current, str):
                from google.genai import types

                current = types.GenerateVideosOperation(name=current)
            current = self.client.operations.get(current)
        except Exception as exc:
            raise ProviderError("provider_failed", "Vertex video polling failed.") from exc
        self._operations[operation.operation_id] = current
        if not getattr(current, "done", False):
            return ProviderPoll("generating")
        provider_error = getattr(current, "error", None)
        if provider_error:
            metadata = _provider_error_metadata(provider_error)
            _log_operation_failure(operation, metadata)
            return ProviderPoll(
                "failed",
                ProviderError(
                    "provider_failed",
                    "Vertex video generation failed.",
                    operation_id=operation.operation_id,
                    provider_error_type=metadata.error_type,
                    provider_error_code=metadata.code,
                    provider_error_status=metadata.status,
                    provider_error_category=metadata.category,
                    provider_error_message=metadata.message,
                ),
            )
        return ProviderPoll("completed", output_reference=_video_uri(current))

    def retrieve(self, operation: ProviderOperation) -> ProviderVideo:
        current = self._operations.get(operation.operation_id)
        response = getattr(current, "response", None) or getattr(current, "result", None)
        generated_videos = getattr(response, "generated_videos", None) if response else None
        if not generated_videos:
            raise ProviderError("video_retrieval_failed", "Vertex returned no video output.")
        video = getattr(generated_videos[0], "video", None)
        if video is None:
            raise ProviderError("video_retrieval_failed", "Vertex returned no video output.")
        video_bytes = getattr(video, "video_bytes", None)
        if isinstance(video_bytes, bytes) and video_bytes:
            return ProviderVideo(video_bytes=video_bytes, mime_type="video/mp4")
        video_uri = _video_uri(current)
        if not video_uri:
            raise ProviderError("video_retrieval_failed", "Vertex returned an invalid video output.")
        try:
            video_bytes = self._download_gcs_video(video_uri)
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("video_retrieval_failed", "Vertex video retrieval failed.") from exc
        if not isinstance(video_bytes, bytes) or not video_bytes:
            raise ProviderError("video_retrieval_failed", "Vertex returned an invalid video output.")
        return ProviderVideo(video_bytes=video_bytes, mime_type="video/mp4")

    def _download_gcs_video(self, video_uri: str) -> bytes:
        bucket, object_name = _approved_gcs_object(video_uri)
        try:
            session = self._storage_session()
            response = session.get(
                (
                    "https://storage.googleapis.com/storage/v1/b/"
                    f"{quote(bucket, safe='')}/o/{quote(object_name, safe='')}?alt=media"
                ),
                timeout=30,
            )
            response.raise_for_status()
            return response.content
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError("video_retrieval_failed", "Vertex video retrieval failed.") from exc

    def _storage_session(self):
        if self._storage_session_factory is None:
            from google.auth import default
            from google.auth.transport.requests import AuthorizedSession

            from .vertex import _service_account_credentials

            credentials = _service_account_credentials()
            if credentials is None:
                credentials, _ = default(
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
            self._storage_session_factory = lambda: AuthorizedSession(credentials)
        return self._storage_session_factory()


def _safe_log_field(value: object, default: str = "unknown") -> str:
    if isinstance(value, bool) or value is None:
        return default
    text = str(value).strip()
    return text if _SAFE_LOG_FIELD.fullmatch(text) else default


def _safe_log_message(value: object) -> str:
    if value is None:
        return "No provider message recorded"
    text = " ".join(str(value).split())
    text = _DATA_URL.sub("[redacted image data]", text)
    text = _BEARER.sub("Bearer [redacted]", text)
    text = _SECRET_ASSIGNMENT.sub(r"\1=[redacted]", text)
    text = _LONG_PAYLOAD.sub("[redacted long payload]", text)
    return text[:500] or "No provider message recorded"


def _safe_metadata_name(value: object, default: str = "unknown") -> str:
    if value is None or isinstance(value, bool):
        return default
    text = " ".join(str(value).split()).strip()
    if not text or len(text) > 120 or not re.fullmatch(r"[A-Za-z0-9_@.$:/-]+", text):
        return default
    return text


def _model_dump_without_none(value: object) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    dumper = getattr(value, "model_dump", None)
    if not callable(dumper):
        return {}
    try:
        dumped = dumper(exclude_none=True)
    except (TypeError, ValueError):
        try:
            dumped = dumper()
        except Exception:
            return {}
    return dict(dumped) if isinstance(dumped, Mapping) else {}


def _provider_error_sources(error: object) -> tuple[dict[str, Any], dict[str, Any]]:
    dumped = _model_dump_without_none(error)
    attributes: dict[str, Any] = {}
    for name in (
        "code",
        "status",
        "google_status",
        "reason",
        "category",
        "message",
        "details",
    ):
        try:
            value = getattr(error, name, None)
        except Exception:
            value = None
        if value is not None:
            attributes[name] = value
    return attributes, dumped


def _source_value(
    attributes: Mapping[str, Any],
    dumped: Mapping[str, Any],
    *names: str,
) -> object:
    for source in (attributes, dumped):
        for name in names:
            if name in source and source[name] is not None:
                return source[name]
    return None


def _detail_mappings(value: object) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if not isinstance(value, (list, tuple)):
        return []
    mappings = []
    for item in value[:16]:
        mapping = _model_dump_without_none(item)
        if mapping:
            mappings.append(mapping)
    return mappings


def _provider_error_category(
    *,
    explicit: object,
    reason: object,
    status: str | None,
    message: str | None,
    details: list[dict[str, Any]],
) -> str:
    for candidate in (explicit, reason):
        safe = _safe_metadata_name(candidate, "")
        if safe:
            return safe
    for detail in details:
        for candidate in (detail.get("reason"), detail.get("category")):
            safe = _safe_metadata_name(candidate, "")
            if safe:
                return safe
    detail_text = " ".join(
        _safe_log_message(value)
        for detail in details
        for key in ("reason", "category", "status", "type")
        for value in (detail.get(key),)
        if isinstance(value, (str, int, float))
    )
    searchable = " ".join(filter(None, (status, message, detail_text))).lower()
    categories = (
        ("safety_filtering", ("safety", "responsible ai", "content filter", "blocked")),
        ("quota_capacity", ("quota", "resource exhausted", "capacity", "rate limit")),
        ("unsupported_configuration", ("invalid argument", "invalid_argument", "unsupported", "not found")),
        ("input_image_rejection", ("image", "mime", "storyboard")),
        ("output_bucket_permission", ("bucket", "storage", "permission", "access denied")),
        ("internal_provider_failure", ("internal", "backend", "unknown")),
    )
    for category, markers in categories:
        if any(marker in searchable for marker in markers):
            return category
    return "unknown"


def _provider_error_metadata(error: object) -> ProviderErrorMetadata:
    attributes, dumped = _provider_error_sources(error)
    details = _detail_mappings(_source_value(attributes, dumped, "details"))
    detail_types: list[str] = []
    detail_fields: list[str] = []
    for detail in details:
        detail_type = _safe_metadata_name(
            detail.get("@type") or detail.get("type") or detail.get("kind"),
            "",
        )
        if detail_type and detail_type not in detail_types:
            detail_types.append(detail_type)
        for key in detail:
            safe_key = _safe_metadata_name(key, "")
            if safe_key and safe_key not in detail_fields:
                detail_fields.append(safe_key)

    raw_code = _source_value(attributes, dumped, "code", "error_code")
    code: int | None = None
    if isinstance(raw_code, int) and not isinstance(raw_code, bool):
        code = raw_code
    elif isinstance(raw_code, str) and re.fullmatch(r"-?[0-9]{1,6}", raw_code.strip()):
        code = int(raw_code.strip())

    raw_message = _source_value(attributes, dumped, "message", "error_message")
    message = _safe_log_message(raw_message) if isinstance(raw_message, str) else None
    raw_status = _source_value(attributes, dumped, "status", "google_status")
    status = _safe_metadata_name(raw_status, "") or None
    reason = _source_value(attributes, dumped, "reason")
    category = _provider_error_category(
        explicit=_source_value(attributes, dumped, "category"),
        reason=reason,
        status=status,
        message=message,
        details=details,
    )
    return ProviderErrorMetadata(
        error_type=_safe_metadata_name(type(error).__name__),
        code=code,
        status=status,
        category=category,
        message=message,
        detail_types=tuple(detail_types[:8]),
        detail_fields=tuple(detail_fields[:16]),
    )


def _log_operation_failure(
    operation: ProviderOperation,
    metadata: ProviderErrorMetadata,
) -> None:
    operation_suffix = _safe_metadata_name(operation.operation_id[-16:], "unknown")
    logger.warning(
        "Vertex Veo operation failed: error_type=%s provider_error_code=%s "
        "provider_error_status=%s provider_error_category=%s "
        "provider_error_message=%s detail_types=%s detail_fields=%s operation_suffix=%s",
        metadata.error_type,
        metadata.code if metadata.code is not None else "unknown",
        metadata.status or "unknown",
        metadata.category or "unknown",
        metadata.message or "No provider message recorded",
        ",".join(metadata.detail_types) or "none",
        ",".join(metadata.detail_fields) or "none",
        operation_suffix,
    )


def _log_submission_failure(
    provider: VertexVeoProvider,
    request: VideoProviderRequest,
    error: Exception,
) -> None:
    """Log bounded provider diagnostics without prompt or image payloads."""
    logger.warning(
        "Vertex Veo submission failed: exception_type=%s http_status=%s "
        "google_status=%s error_code=%s message=%s model=%s region=%s "
        "image_supplied=%s image_mime_type=%s duration_seconds=%s aspect_ratio=%s "
        "output_count=%s generate_audio=%s",
        type(error).__name__,
        _safe_log_field(getattr(error, "code", None)),
        _safe_log_field(getattr(error, "status", None)),
        _safe_log_field(getattr(error, "reason", None)),
        _safe_log_message(getattr(error, "message", None) or error),
        _safe_log_field(provider.model),
        _safe_log_field(os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")),
        bool(request.image_bytes),
        _safe_log_field(request.image_mime_type, "none") if request.image_bytes else "none",
        request.duration_seconds,
        _safe_log_field(request.aspect_ratio),
        1,
        False,
    )


def _configured_output_gcs_uri() -> str | None:
    bucket = os.getenv("VEO_OUTPUT_GCS_BUCKET", "").strip()
    prefix = os.getenv("VEO_OUTPUT_GCS_PREFIX", "").strip().strip("/")
    if not bucket and not prefix:
        return None
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]", bucket)
        or not prefix
        or "\\" in prefix
        or any(part in {"", ".", ".."} for part in prefix.split("/"))
    ):
        raise ProviderError(
            "provider_configuration_failed",
            "VEO_OUTPUT_GCS_BUCKET and VEO_OUTPUT_GCS_PREFIX must identify an approved bucket and prefix.",
        )
    return f"gs://{bucket}/{prefix}/"


def _video_uri(operation: object) -> str | None:
    response = getattr(operation, "response", None) or getattr(operation, "result", None)
    generated_videos = getattr(response, "generated_videos", None) if response else None
    if not generated_videos:
        return None
    video = getattr(generated_videos[0], "video", None)
    uri = getattr(video, "uri", None) if video is not None else None
    return uri if isinstance(uri, str) and uri else None


def _approved_gcs_object(video_uri: str) -> tuple[str, str]:
    bucket = os.getenv("VEO_OUTPUT_GCS_BUCKET", "").strip()
    prefix = os.getenv("VEO_OUTPUT_GCS_PREFIX", "").strip().strip("/")
    if not bucket or not prefix:
        raise ProviderError(
            "video_retrieval_failed",
            "Approved Vertex output GCS bucket and prefix are not configured.",
        )
    parsed = urlsplit(video_uri)
    if (
        parsed.scheme != "gs"
        or parsed.netloc != bucket
        or parsed.query
        or parsed.fragment
        or parsed.username
        or parsed.password
        or parsed.port
    ):
        raise ProviderError("video_retrieval_failed", "Vertex returned an unapproved video URI.")
    object_name = unquote(parsed.path.lstrip("/"))
    approved_prefix = f"{prefix}/"
    if (
        not object_name
        or "\\" in object_name
        or any(part in {"", ".", ".."} for part in object_name.split("/"))
        or not object_name.startswith(approved_prefix)
    ):
        raise ProviderError("video_retrieval_failed", "Vertex returned an unapproved video URI.")
    return bucket, object_name

