"""Offline-safe critique of completed video jobs."""

from __future__ import annotations

import hashlib
import json
import os
from threading import RLock
from typing import Any, Protocol

from .rate_limit import VertexRateLimiter
from .schemas import (
    ShotPlan,
    VideoCritique,
    VideoCritiquePayload,
    VideoEvidence,
)
from .vertex import _client


VIDEO_CRITIQUE_PROMPT_VERSION = "video-critique-v1"
VIDEO_CRITIQUE_UNAVAILABLE = "Video critique unavailable."
DEMO_CRITIQUE_NOTICE = "Demo critique—not an analysis of your video."
MAX_VIDEO_BYTES = 16 * 1024 * 1024
MAX_STORYBOARD_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_CHARS = 24_000
MAX_OUTPUT_TOKENS = 2_200


class VideoCritiqueProvider(Protocol):
    name: str

    def analyse(
        self,
        *,
        video_bytes: bytes,
        video_mime_type: str,
        storyboard_bytes: bytes | None,
        storyboard_mime_type: str | None,
        screenplay: str,
        creative_intent: str,
        shot_plan: ShotPlan,
    ) -> VideoCritiquePayload:
        ...


def _env_nonnegative_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _observation(topic: str) -> dict[str, Any]:
    return {
        "finding": f"Demo mode does not inspect whether {topic} is present in the supplied footage.",
        "support": "not_observed",
        "evidence": [],
        "uncertainty": "No visual inference is performed in demo mode.",
    }


class MockVideoCritiqueProvider:
    """Deterministic fixture-backed provider; it never interprets video pixels."""

    name = "mock"

    def analyse(
        self,
        *,
        video_bytes: bytes,
        video_mime_type: str,
        storyboard_bytes: bytes | None,
        storyboard_mime_type: str | None,
        screenplay: str,
        creative_intent: str,
        shot_plan: ShotPlan,
    ) -> VideoCritiquePayload:
        if not video_bytes:
            raise ValueError("Completed video bytes are empty.")
        return VideoCritiquePayload(
            requested_actions_achieved=[_observation("the requested actions")],
            requested_actions_missing=[_observation("the requested actions")],
            character_object_consistency=[_observation("character and object identity")],
            camera_movement_pacing=[_observation("camera movement and pacing")],
            visible_artifacts=[_observation("visible artifacts")],
            emotional_intent_alignment=[_observation("emotional intent alignment")],
            revision_recommendations=[],
            uncertainty=(
                "This is a deterministic demo response. It is not evidence about the supplied "
                "video and should not be used as a footage assessment."
            ),
        )


VIDEO_CRITIQUE_SYSTEM_PROMPT = """You are FramePilot's video critic. Evaluate only the supplied
completed video, using the screenplay, original storyboard, and approved motion plan as comparison
context. Do not infer or claim footage that is absent or unreadable. Separate what is visibly
supported from what is missing, uncertain, or not observable.

Return only the requested JSON schema. Cover every category. Each concrete visual finding should
include one or more timestamped evidence items when a timestamp can be established. If a finding
cannot be timestamped or verified, use support=uncertain or support=not_observed and explain the
uncertainty. Recommendations must be specific, conservative, and grounded in supported findings.
Do not guarantee that a revision will improve the result."""


class GeminiVideoCritiqueProvider:
    name = "vertex"

    def __init__(self, *, client_factory=None, model: str | None = None):
        self._client_factory = client_factory or _client
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def analyse(
        self,
        *,
        video_bytes: bytes,
        video_mime_type: str,
        storyboard_bytes: bytes | None,
        storyboard_mime_type: str | None,
        screenplay: str,
        creative_intent: str,
        shot_plan: ShotPlan,
    ) -> VideoCritiquePayload:
        if not video_bytes:
            raise ValueError("Completed video bytes are empty.")
        if len(video_bytes) > MAX_VIDEO_BYTES:
            raise ValueError("Completed video is too large to analyse.")
        if storyboard_bytes and len(storyboard_bytes) > MAX_STORYBOARD_BYTES:
            raise ValueError("Storyboard image is too large to analyse.")

        plan_json = json.dumps(shot_plan.model_dump(), ensure_ascii=False, separators=(",", ":"))
        context = (
            f"SCREENPLAY:\n{screenplay[:8_000]}\n\n"
            f"CREATIVE INTENT:\n{creative_intent[:500]}\n\n"
            f"APPROVED MOTION PLAN:\n{plan_json[:12_000]}\n\n"
            "The next video part is the completed First Cut. The optional image part is the "
            "original storyboard and must be used only as comparison context."
        )[:MAX_CONTEXT_CHARS]
        contents: list[Any] = [
            context,
            types_part(video_bytes, video_mime_type),
        ]
        if storyboard_bytes and storyboard_mime_type:
            contents.append(types_part(storyboard_bytes, storyboard_mime_type))

        with self._client_factory() as client:
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=generate_content_config(),
            )
        return VideoCritiquePayload.model_validate_json(response.text)


def types_part(data: bytes, mime_type: str):
    from google.genai import types

    return types.Part.from_bytes(data=data, mime_type=mime_type)


def generate_content_config():
    from google.genai import types

    return types.GenerateContentConfig(
        system_instruction=VIDEO_CRITIQUE_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=VideoCritiquePayload,
        temperature=0.2,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )


def _content_hash(video_bytes: bytes) -> str:
    return hashlib.sha256(video_bytes).hexdigest()


def _source_plan_hash(context: dict[str, Any]) -> str:
    payload = {
        "screenplay": context.get("screenplay", ""),
        "creative_intent": context.get("creative_intent", ""),
        "shot_plan": context.get("shot_plan", {}),
        "storyboard_hash": (
            hashlib.sha256(context["storyboard_bytes"]).hexdigest()
            if context.get("storyboard_bytes")
            else None
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_timestamps(payload: VideoCritiquePayload, max_duration: float) -> None:
    groups = (
        payload.requested_actions_achieved,
        payload.requested_actions_missing,
        payload.character_object_consistency,
        payload.camera_movement_pacing,
        payload.visible_artifacts,
        payload.emotional_intent_alignment,
        payload.revision_recommendations,
    )
    for group in groups:
        for item in group:
            for evidence in item.evidence:
                if evidence.timestamp_seconds is not None and evidence.timestamp_seconds > max_duration:
                    raise ValueError("Video critique evidence timestamp exceeds the approved shot duration.")


def _unavailable(model: str | None = None) -> VideoCritique:
    return VideoCritique(
        status="unavailable",
        source="unavailable",
        label=VIDEO_CRITIQUE_UNAVAILABLE,
        message=VIDEO_CRITIQUE_UNAVAILABLE,
        uncertainty="No reliable video critique result is available.",
        model=model,
    )


class VideoCritiqueService:
    """Explicitly invoked critique orchestration with cache and cost guards."""

    def __init__(
        self,
        *,
        provider: VideoCritiqueProvider | None = None,
        vertex_limiter: VertexRateLimiter | None = None,
        critique_cap: int | None = None,
    ):
        provider_name = os.getenv("VIDEO_CRITIQUE_PROVIDER", "mock").strip().lower()
        self.provider_name = getattr(provider, "name", provider_name)
        self.provider = provider or (
            MockVideoCritiqueProvider()
            if self.provider_name == "mock"
            else GeminiVideoCritiqueProvider()
        )
        self.vertex_limiter = vertex_limiter or VertexRateLimiter()
        self.critique_cap = (
            critique_cap
            if critique_cap is not None
            else _env_nonnegative_int("VIDEO_CRITIQUE_GLOBAL_LIMIT", 5)
        )
        self._successful_live_critiques = 0
        self._cache: dict[str, VideoCritique] = {}
        self._job_results: dict[str, VideoCritique] = {}
        self._lock = RLock()

    @property
    def live_allowed(self) -> bool:
        return os.getenv("ALLOW_VIDEO_CRITIQUE", "false") == "true"

    def for_job(self, job_id: str) -> VideoCritique | None:
        with self._lock:
            return self._job_results.get(job_id)

    def analyse_first_cut(
        self,
        video_jobs,
        job_id: str,
        client_id: str,
        rate_limit_key: str | None = None,
    ) -> VideoCritique:
        try:
            job = video_jobs.get(job_id, client_id)
            if job.kind != "first_cut" or job.status != "completed":
                return _unavailable(self._provider_model())
            video_bytes, video_mime_type = video_jobs.video_bytes(job_id, client_id)
            context = video_jobs.get_critique_context(job_id, client_id)
            if not context:
                return _unavailable(self._provider_model())
        except Exception:
            return _unavailable(self._provider_model())

        storyboard_bytes = context.get("storyboard_bytes")
        if storyboard_bytes and len(storyboard_bytes) > MAX_STORYBOARD_BYTES:
            return _unavailable(self._provider_model())
        if len(video_bytes) > MAX_VIDEO_BYTES:
            return _unavailable(self._provider_model())

        content_hash = _content_hash(video_bytes)
        source_plan_hash = _source_plan_hash(context)
        cache_key = ":".join(
            [content_hash, source_plan_hash, self._provider_model() or "", VIDEO_CRITIQUE_PROMPT_VERSION]
        )
        with self._lock:
            cached = self._cache.get(cache_key)
            if cached:
                self._job_results[job_id] = cached
                return cached
            if self.provider_name == "vertex" and not self.live_allowed:
                return _unavailable(self._provider_model())
            if self.provider_name == "vertex" and self._successful_live_critiques >= self.critique_cap:
                return _unavailable(self._provider_model())

            try:
                if self.provider_name == "vertex":
                    allowed, payload = self.vertex_limiter.run(
                        rate_limit_key or client_id,
                        lambda: self.provider.analyse(
                            video_bytes=video_bytes,
                            video_mime_type=video_mime_type,
                            storyboard_bytes=storyboard_bytes,
                            storyboard_mime_type=context.get("storyboard_mime_type"),
                            screenplay=context.get("screenplay", ""),
                            creative_intent=context.get("creative_intent", ""),
                            shot_plan=ShotPlan.model_validate(context["shot_plan"]),
                        ),
                    )
                    if not allowed:
                        return _unavailable(self._provider_model())
                else:
                    payload = self.provider.analyse(
                        video_bytes=video_bytes,
                        video_mime_type=video_mime_type,
                        storyboard_bytes=storyboard_bytes,
                        storyboard_mime_type=context.get("storyboard_mime_type"),
                        screenplay=context.get("screenplay", ""),
                        creative_intent=context.get("creative_intent", ""),
                        shot_plan=ShotPlan.model_validate(context["shot_plan"]),
                    )
                payload = VideoCritiquePayload.model_validate(payload)
                max_duration = float(ShotPlan.model_validate(context["shot_plan"]).shot.duration_seconds)
                _validate_timestamps(payload, max_duration)
            except Exception:
                return _unavailable(self._provider_model())

            result = VideoCritique(
                **payload.model_dump(),
                status="available",
                source="gemini_video" if self.provider_name == "vertex" else "demo_fixture",
                label="Gemini video critique" if self.provider_name == "vertex" else DEMO_CRITIQUE_NOTICE,
                message="Video critique complete.",
                notice=DEMO_CRITIQUE_NOTICE if self.provider_name == "mock" else None,
                content_hash=content_hash,
                source_plan_hash=source_plan_hash,
                model=self._provider_model(),
            )
            self._cache[cache_key] = result
            self._job_results[job_id] = result
            if self.provider_name == "vertex":
                self._successful_live_critiques += 1
            return result

    def _provider_model(self) -> str | None:
        return getattr(self.provider, "model", None) or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")