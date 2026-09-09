from pathlib import Path
import hashlib
import json
import logging
import re
import secrets

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from cinema_agent.demo import demo_critique, demo_plan
from cinema_agent.cache import AnalysisCache, analysis_cache_key
from cinema_agent.camera_grammar import ShotDiversityGuard, signature_text
from cinema_agent.depth import heuristic_depth_layout
from cinema_agent.router import route_scene_action
from cinema_agent.rate_limit import VertexRateLimiter
from cinema_agent.schemas import DirectRequest, DirectResponse
from cinema_agent.schemas import (
    ControlledAuthorizationRequest,
    ControlledAuthorizationStatus,
    VideoAllowanceStatus,
    VideoApproval,
    VideoApprovalRequest,
    VideoCritique,
    VideoJob,
    VideoJobRequest,
)
from cinema_agent.tools import runtime_mode
from cinema_agent.video_jobs import MockVideoJobService, VideoJobError
from cinema_agent.video_critique import VideoCritiqueService
from cinema_agent.video_provider import ProviderError, effective_veo_model, veo_estimate_label
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
STATIC = ROOT / "static"

app = FastAPI(title="FramePilot", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
vertex_request_limiter = VertexRateLimiter()
analysis_cache = AnalysisCache()
shot_diversity_guard = ShotDiversityGuard()
video_job_service = MockVideoJobService()
video_critique_service = VideoCritiqueService(vertex_limiter=vertex_request_limiter)
SESSION_COOKIE_NAME = "framepilot_session"
SESSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 30
logger = logging.getLogger(__name__)

_SAFE_VERTEX_FIELD = re.compile(r"^[A-Za-z0-9_.:/-]{1,96}$")
_VERTEX_DATA_URL = re.compile(r"data:[^;\s]+;base64,[A-Za-z0-9+/=_-]+", re.IGNORECASE)
_VERTEX_BEARER = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_VERTEX_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(authorization|api[-_ ]?key|token|secret|password|private[-_ ]?key)"
    r"\s*[:=]\s*\S+"
)
_VERTEX_LONG_PAYLOAD = re.compile(r"[A-Za-z0-9+/=_-]{160,}")


def _safe_vertex_field(value, default="unknown"):
    if isinstance(value, bool) or value is None:
        return default
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    return text if _SAFE_VERTEX_FIELD.fullmatch(text) else default


def _safe_vertex_message(value):
    if value is None:
        return "No provider message recorded"
    text = " ".join(str(value).split())
    text = _VERTEX_DATA_URL.sub("[redacted image data]", text)
    text = _VERTEX_BEARER.sub("Bearer [redacted]", text)
    text = _VERTEX_SECRET_ASSIGNMENT.sub(r"\1=[redacted]", text)
    text = _VERTEX_LONG_PAYLOAD.sub("[redacted long payload]", text)
    return text[:500] or "No provider message recorded"


def _vertex_error_category(code, status, message):
    normalized_status = str(status or "").upper()
    normalized_message = str(message or "").lower()
    if code in {401} or normalized_status in {"UNAUTHENTICATED", "UNAUTHENTICATED_ERROR"}:
        return "authentication"
    if code in {403} or normalized_status == "PERMISSION_DENIED":
        if any(term in normalized_message for term in ("quota", "rate", "access", "resource exhausted")):
            return "quota_or_access_restriction"
        return "missing_iam_permission"
    if code in {413} or "too large" in normalized_message or "payload" in normalized_message:
        return "request_size_or_mime"
    if code in {400} or normalized_status in {"INVALID_ARGUMENT", "FAILED_PRECONDITION"}:
        return "invalid_request_or_schema"
    if code in {404} or normalized_status in {"NOT_FOUND", "UNIMPLEMENTED"}:
        return "model_region_or_endpoint"
    if code in {429} or normalized_status == "RESOURCE_EXHAUSTED":
        return "quota_or_access_restriction"
    if isinstance(code, int) and code >= 500:
        return "provider_unavailable"
    return "client_error_unclassified"


def _safe_vertex_error_details(error):
    code = getattr(error, "code", None)
    status = getattr(error, "status", None)
    message = getattr(error, "message", None)
    validation_errors = _safe_validation_errors(error)
    if (
        isinstance(error, ValueError)
        and str(error) == "SceneAnalysis has no recoverable semantic entities"
    ):
        validation_errors = ["scene_analysis:no_recoverable_entities"]
    if validation_errors:
        message = "SceneAnalysis response failed local validation"
    if message is None and not hasattr(error, "message"):
        message = str(error)
    safe_code = _safe_vertex_field(code)
    safe_status = _safe_vertex_field(status)
    safe_message = _safe_vertex_message(message)
    numeric_code = code if isinstance(code, int) and not isinstance(code, bool) else None
    return {
        "error_type": type(error).__name__,
        "http_status": safe_code,
        "google_status": safe_status,
        "category": "response_validation"
        if validation_errors
        else _vertex_error_category(numeric_code, status, message),
        "message": safe_message,
        "validation_errors": validation_errors,
    }


def _safe_validation_errors(error):
    errors_method = getattr(error, "errors", None)
    if not callable(errors_method):
        return []
    try:
        raw_errors = errors_method()
    except Exception:
        return []
    if not isinstance(raw_errors, list):
        return []
    sanitized = []
    for item in raw_errors[:20]:
        if not isinstance(item, dict):
            continue
        location = item.get("loc", ())
        if not isinstance(location, (tuple, list)):
            location = (location,)
        path = ".".join(str(part) for part in location) or "$"
        safe_path = _safe_vertex_field(path)
        safe_type = _safe_vertex_field(item.get("type"), "unknown")
        sanitized.append(f"{safe_path}:{safe_type}")
    if len(raw_errors) > 20:
        sanitized.append(f"...+{len(raw_errors) - 20}")
    return sanitized


def _log_vertex_failure(error):
    details = _safe_vertex_error_details(error)
    log_message = (
        "Vertex multimodal request failed: error_type=%s http_status=%s "
        "google_status=%s category=%s message=%s"
    )
    log_args = (
        details["error_type"],
        details["http_status"],
        details["google_status"],
        details["category"],
        details["message"],
    )
    if details["validation_errors"]:
        logger.warning(
            log_message + " validation_errors=%s",
            *log_args,
            ",".join(details["validation_errors"]),
        )
    else:
        logger.warning(log_message, *log_args)
    return details


@app.middleware("http")
async def issue_session_cookie(request: Request, call_next):
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        session_id = secrets.token_urlsafe(32)
        request.state.framepilot_session_id = session_id
        response = await call_next(request)
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=SESSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
        )
        return response
    request.state.framepilot_session_id = session_id
    return await call_next(request)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "mode": runtime_mode(),
        "video_provider": video_job_service.provider_name,
        "video_model": effective_veo_model() if video_job_service.provider_name == "vertex" else None,
        "video_estimate": (
            veo_estimate_label()
            if video_job_service.provider_name == "vertex"
            else None
        ),
    }


@app.get("/api/video-allowance", response_model=VideoAllowanceStatus)
def video_allowance(http_request: Request):
    return video_job_service.allowance_status(
        _session_id(http_request),
        _request_ip(http_request),
    )


@app.post(
    "/api/video-test-authorization",
    response_model=ControlledAuthorizationStatus,
)
def activate_video_test_authorization(
    request: ControlledAuthorizationRequest,
    http_request: Request,
):
    try:
        return video_job_service.activate_controlled_test_authorization(
            request,
            client_id=_session_id(http_request),
        )
    except VideoJobError as error:
        raise _video_error(error) from error


@app.get(
    "/api/video-test-authorization",
    response_model=ControlledAuthorizationStatus,
)
def video_test_authorization(
    scene_key: str,
    source_signature: str,
    authorization_id: str | None = None,
    http_request: Request = None,
):
    return video_job_service.controlled_test_authorization_status(
        client_id=_session_id(http_request),
        scene_key=scene_key,
        source_signature=source_signature,
        authorization_id=authorization_id,
    )


@app.get("/api/storyboard-images/{image_handle}")
def storyboard_image_availability(image_handle: str, http_request: Request):
    if not video_job_service.image_available(image_handle, _session_id(http_request)):
        raise HTTPException(status_code=404, detail="Storyboard image is no longer available for this session.")
    return {"available": True}


def _session_id(http_request: Request) -> str:
    return http_request.state.framepilot_session_id


def _request_ip(http_request: Request) -> str:
    return http_request.client.host if http_request.client else "unknown"


def _direct_scene_identity(plan, critique, image_handle: str | None) -> tuple[str, str]:
    scene_payload = {
        "summary": plan.scene_summary,
        "intent": plan.emotional_intent,
        "image": bool(image_handle),
    }
    source_payload = {
        "shot": plan.shot.model_dump(mode="json"),
        "motion_plan": plan.motion_plan.model_dump(mode="json"),
        "revision": critique.revision.model_dump(mode="json"),
    }

    def digest(payload: dict) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:24]

    return f"scene-{digest(scene_payload)}", f"source-{digest(source_payload)}"


def _video_error(error: VideoJobError) -> HTTPException:
    return HTTPException(
        status_code=error.status_code,
        detail={"code": error.code, "message": str(error)},
    )


@app.post("/api/video-jobs", response_model=VideoJob)
def create_video_job(request: VideoJobRequest, http_request: Request):
    try:
        return video_job_service.create(
            request,
            _session_id(http_request),
            _request_ip(http_request),
        )
    except VideoJobError as error:
        raise _video_error(error) from error


@app.post("/api/video-jobs/approval", response_model=VideoApproval)
def request_video_approval(request: VideoApprovalRequest, http_request: Request):
    try:
        if request.kind == "director_cut" and request.first_cut_job_id:
            saved_critique = video_critique_service.for_job(request.first_cut_job_id)
            if saved_critique is not None:
                request = request.model_copy(update={"video_critique": saved_critique})
        return video_job_service.request_approval(request, _session_id(http_request))
    except VideoJobError as error:
        raise _video_error(error) from error


@app.get("/api/video-jobs/{job_id}/video")
def get_video_output(job_id: str, http_request: Request):
    try:
        content, media_type = video_job_service.video_bytes(job_id, _session_id(http_request))
        return Response(content=content, media_type=media_type)
    except VideoJobError as error:
        raise _video_error(error) from error


@app.get("/api/video-jobs/{job_id}/storyboard")
def get_storyboard_output(job_id: str, http_request: Request):
    try:
        content, media_type = video_job_service.storyboard_bytes(
            job_id,
            _session_id(http_request),
        )
        return Response(content=content, media_type=media_type)
    except VideoJobError as error:
        raise _video_error(error) from error


@app.get("/api/video-jobs/latest", response_model=VideoJob | None)
def get_latest_video_job(http_request: Request):
    try:
        return video_job_service.latest_completed(_session_id(http_request), kind="first_cut")
    except VideoJobError as error:
        raise _video_error(error) from error


@app.get("/api/video-jobs/{job_id}", response_model=VideoJob)
def get_video_job(job_id: str, http_request: Request):
    try:
        job = video_job_service.get(job_id, _session_id(http_request))
        critique = video_critique_service.for_job(job_id)
        return job.model_copy(update={"video_critique": critique})
    except VideoJobError as error:
        raise _video_error(error) from error


@app.post("/api/video-jobs/{job_id}/approve-revision", response_model=VideoJob)
def approve_video_revision(job_id: str, http_request: Request):
    try:
        return video_job_service.approve_revision(job_id, _session_id(http_request))
    except VideoJobError as error:
        raise _video_error(error) from error


@app.post("/api/video-jobs/{job_id}/critique", response_model=VideoCritique)
def critique_video_job(job_id: str, http_request: Request):
    return video_critique_service.analyse_first_cut(
        video_job_service,
        job_id,
        _session_id(http_request),
        _request_ip(http_request),
    )


@app.post("/api/direct", response_model=DirectResponse)
def direct_scene(request: DirectRequest, http_request: Request):
    mode = runtime_mode()
    routing = route_scene_action(request.screenplay, request.mood)
    fallback = None
    depth_source = "heuristic"
    analysis_source = "deterministic_fallback"
    cache_hit = False
    client_ip = _request_ip(http_request)
    try:
        image_handle = video_job_service.register_image(
            request.image_data_url,
            _session_id(http_request),
        )
    except ProviderError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if mode == "vertex":
        try:
            from cinema_agent.vertex import generate_critique, generate_shot_plan

            def direct_with_vertex():
                plan = generate_shot_plan(request.screenplay, request.mood, request.image_data_url)
                image_aware = bool(request.image_data_url and plan.depth_layout)
                if plan.depth_layout is None:
                    plan = plan.model_copy(update={"depth_layout": heuristic_depth_layout()})
                return (
                    plan,
                    generate_critique(request.screenplay, plan),
                    "image_aware" if image_aware else "heuristic",
                )

            cache_key = (
                analysis_cache_key(request.image_data_url, request.screenplay, request.mood)
                if request.image_data_url
                else None
            )
            cached_result = analysis_cache.get(cache_key) if cache_key else None
            if cached_result:
                plan, critique, depth_source = cached_result
                cache_hit = True
                analysis_source = "vertex_multimodal"
            else:
                allowed, vertex_result = vertex_request_limiter.run(client_ip, direct_with_vertex)
                if allowed:
                    plan, critique, depth_source = vertex_result
                    analysis_source = "vertex_multimodal"
                    if cache_key:
                        analysis_cache.set(cache_key, vertex_result)
                else:
                    mode = "rate_limited_fallback"
                    plan = demo_plan(request.screenplay, request.mood, request.image_data_url)
                    critique = demo_critique(plan)
        except Exception as exc:
            # The demo remains available if credits, billing, IAM, quota, or model access
            # are not ready. The UI makes this fallback visible instead of hiding it.
            failure_details = _log_vertex_failure(exc)
            mode = "demo"
            if failure_details["validation_errors"]:
                fallback = "Vertex response failed local validation; deterministic fallback used."
            else:
                fallback = f"Vertex call unavailable: {type(exc).__name__}"
            plan = demo_plan(request.screenplay, request.mood, request.image_data_url)
            critique = demo_critique(plan)
    else:
        plan = demo_plan(request.screenplay, request.mood, request.image_data_url)
        critique = demo_critique(plan)

    original_shot = plan.shot
    compiled = shot_diversity_guard.compile(
        original_shot,
        request.screenplay,
        request.mood,
        client_ip,
    )
    plan = plan.model_copy(update={"shot": compiled.shot})

    revision_changed = critique.revision.model_dump() != original_shot.model_dump()
    compiled_revision = shot_diversity_guard.compile(
        critique.revision,
        request.screenplay,
        request.mood,
        client_ip,
        force_distinct=revision_changed,
    )
    critique = critique.model_copy(update={"revision": compiled_revision.shot})

    activity = [
        {"step": "ROUTE", "detail": f"{routing.classification}: {routing.rationale}"},
        {
            "step": "ANALYSE",
            "detail": f"Mapped {plan.focal_subject} · intent: {plan.emotional_intent}",
        },
        {
            "step": "CAMERA",
            "detail": f"{compiled.grammar.upper()} grammar: {signature_text(compiled.signature)}",
        },
        {
            "step": "DIRECT",
            "detail": f"Chose {compiled.shot.camera_motion.replace('_', ' ')} with "
            f"{', '.join(compiled.shot.atmosphere) if compiled.shot.atmosphere else 'clean atmosphere'}",
        },
        {
            "step": "ANIMATE",
            "detail": "Compiled browser motion across foreground, midground and background",
        },
        {"step": "CRITIQUE", "detail": critique.diagnosis},
        {
            "step": "REVISE",
            "detail": f"{compiled_revision.grammar.upper()} revision · {critique.revision_rationale}",
        },
    ]
    if fallback:
        activity.insert(0, {"step": "FALLBACK", "detail": fallback})
    if mode == "rate_limited_fallback":
        activity.insert(
            1,
            {
                "step": "RATE LIMIT",
                "detail": "Vertex usage limit reached; showing the deterministic local 2.5D result.",
            },
        )
    if cache_hit:
        activity.insert(
            1,
            {
                "step": "CACHE",
                "detail": "Reused the matching image-aware direction; no Vertex request was made.",
            },
        )
    if compiled.adjustment:
        activity.insert(3, {"step": "DIVERSITY", "detail": compiled.adjustment})
    if compiled_revision.adjustment:
        activity.insert(4, {"step": "REVISION", "detail": compiled_revision.adjustment})
    scene_key, source_signature = _direct_scene_identity(plan, critique, image_handle)
    video_job_service.remember_direct_context(
        client_id=_session_id(http_request),
        scene_key=scene_key,
        source_signature=source_signature,
        analysis_source=analysis_source,
        image_handle=image_handle,
    )
    return DirectResponse(
        mode=mode,
        analysis_source=analysis_source,
        scene_key=scene_key,
        source_signature=source_signature,
        image_handle=image_handle,
        depth_source=depth_source,
        plan=plan,
        critique=critique,
        routing=routing,
        activity=activity,
        camera_grammar=compiled.grammar,
        shot_signature=compiled.signature,
        diversity_adjustment=compiled.adjustment,
        revision_camera_grammar=compiled_revision.grammar,
        revision_shot_signature=compiled_revision.signature,
        revision_diversity_adjustment=compiled_revision.adjustment,
    )
