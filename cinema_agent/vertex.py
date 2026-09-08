import base64
import json
import logging
import math
import os
import re

from google import genai
from google.genai import types
from google.oauth2 import service_account
from pydantic import ValidationError

from .scene_analysis import scene_analysis_to_shot_plan
from .schemas import (
    CameraAnalysis,
    Critique,
    SceneAnalysis,
    SceneEntity,
    SceneRelationship,
    ShotPlan,
)
from .router import route_scene_action


GOOGLE_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
DIRECTOR_OUTPUT_TOKEN_BUDGET = 8192
logger = logging.getLogger(__name__)


DIRECTOR_PROMPT = """You are FramePilot's multimodal scene analyst. Analyze the screenplay, creative
intent, and optional storyboard image. Return exactly one compact JSON object with:
scene_summary:string; characters:SceneEntity[]; objects:SceneEntity[];
environment:SceneEntity[]; camera:CameraAnalysis; preserve:string[]; prohibit:string[];
conflicts:string[]; main_character_id:string|null; relationships:SceneRelationship[].

SceneEntity records contain only:
{"entity_id":string|null,"label":string,"semantic_category":string|null,
"action":string|null,"visual_evidence":string|null,
"screenplay_evidence":string|null,"grounding_source":
"visual"|"screenplay"|"visual_and_screenplay"|"unsupported"|null,
"confidence":number,"visual_confidence":number,
"support":"supported"|"needs_confirmation"|"unsupported"}

CameraAnalysis:
{"movement":string,"visual_evidence":string|null,
"screenplay_evidence":string|null,"confidence":number}

SceneRelationship records contain only:
{"source_id":string,"relation":string,"target_id":string,"action":string|null}
The relationship action must be one concise JSON string or null; never an object, array, or nested
record.

Compactness rules: use short source-local evidence excerpts, not explanations. Do not repeat the
creative intent or preservation/prohibition rules inside entities. Put shared preservation and
prohibition rules once in preserve and prohibit. Use concise actions and relationship actions.
Keep scene_summary under 400 characters; use at most 12 characters, 16 objects, 12 environment
elements, 24 relationships, 12 preserve rules, 16 prohibit rules, and 8 conflicts. Keep labels
under 80 characters, IDs under 64, categories under 64, actions and evidence under 240. When
limits compete, prioritize the most narratively and visually important entities and relationships.

A character is a narratively agentive entity, not necessarily human or living. The main character is
the visible entity functioning as the scene's principal subject, actor, or point of attention. It may
be any person, animal, creature, robot, vehicle, machine, animated object, natural entity, or
abstract/stylized figure. Determine it from composition, screenplay focus, actions, relationships,
and creative intent—not size, centrality, or human appearance. Return at most one
main_character_id, and only when it references an entity_id in characters. Supporting characters
may also be returned. An ordinary non-agentive item is an object unless the scene presents it as
acting, choosing, reacting, or functioning as the narrative subject. Do not use fixed character
types or production-scene labels. Do not return one entity in both characters and objects.

Use stable noun-phrase labels. Preserve distinct controllable entities and use relationships for
causal actions, affected entities, attached/held/worn objects, components, and environmental
effects. A tangible inanimate entity remains in objects even when it is held, worn, attached,
carried, opened, raised, rotated, or otherwise manipulated. Put natural phenomena, atmosphere,
fluid or water motion, weather, light effects, particles, forces, and other ambient effects in
environment. Do not move a tangible object into environment merely because it is being manipulated.
If evidence or role is ambiguous, preserve the entity and set support to needs_confirmation rather
than forcing a classification. Use visual_evidence only for storyboard observations and
screenplay_evidence only for concise source-local screenplay excerpts. Do not claim screenplay-only
content is visually confirmed. Set action to null when no bounded motion is supported. Do not
invent entities, actions, effects, coordinates, or rules. Return one JSON object only: no markdown
fences, prose, or extra fields."""

CRITIC_PROMPT = """You are an exacting film previsualisation critic. Evaluate the supplied
screenplay and shot plan for narrative focus, pacing, cinematic motion versus slide-like
motion, and restraint. Produce a materially improved parameter set, not generic advice.
Keep every numeric value within the requested schema and output only that schema."""

def _service_account_credentials():
    credential_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    if not credential_json:
        return None
    try:
        service_account_info = json.loads(credential_json)
    except json.JSONDecodeError as exc:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON must contain valid JSON") from exc
    if not isinstance(service_account_info, dict):
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON must contain a JSON object")
    return service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=[GOOGLE_CLOUD_PLATFORM_SCOPE],
    )


def _client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    client_options = {"project": project, "location": location}
    credentials = _service_account_credentials()
    if credentials is not None:
        client_options["credentials"] = credentials
    try:
        return genai.Client(enterprise=True, **client_options)
    except TypeError:
        # Compatibility with older google-genai builds that still use vertexai=True.
        return genai.Client(vertexai=True, **client_options)


def _image_part(image_data_url: str | None):
    if not image_data_url:
        return None
    if "," not in image_data_url:
        raise ValueError("Storyboard image must be a valid data URL")
    header, encoded = image_data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        raise ValueError(f"Unsupported storyboard image MIME type: {mime or 'unknown'}")
    raw = base64.b64decode(encoded, validate=True)
    if not raw:
        raise ValueError("Storyboard image must contain non-empty bytes")
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Storyboard image must be smaller than 8 MB")
    return types.Part.from_bytes(data=raw, mime_type=mime)


def _response_text(response) -> str:
    try:
        text = response.text
    except Exception:
        return ""
    return text if isinstance(text, str) else ""


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?[ \t]*\n?(.*?)\n?```", stripped, flags=re.IGNORECASE | re.DOTALL)
    return fenced.group(1).strip() if fenced else stripped


def _finish_reason(response) -> str:
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, (list, tuple)) or not candidates:
        return "unknown"
    reason = getattr(candidates[0], "finish_reason", None)
    if isinstance(reason, (str, int, float, bool)):
        return str(reason)
    name = getattr(reason, "name", None)
    if isinstance(name, str) and name:
        return name
    value = getattr(reason, "value", None)
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return "unknown"


def _log_response_diagnostics(response, text: str) -> None:
    trimmed = text.strip()
    candidates = getattr(response, "candidates", None)
    candidate_count = len(candidates) if isinstance(candidates, (list, tuple)) else 0
    logger.warning(
        "Vertex scene analysis response diagnostics: candidate_count=%d "
        "finish_reason=%s response_text_length=%d starts_with_brace=%s "
        "ends_with_brace=%s markdown_fences_present=%s empty_response=%s",
        candidate_count,
        _finish_reason(response),
        len(text),
        trimmed.startswith("{"),
        trimmed.endswith("}"),
        "```" in trimmed,
        not bool(trimmed),
    )


def _shorten_at_boundary(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    window = value[:limit]
    sentence_ends = [
        index + 1
        for index, character in enumerate(window)
        if character in ".!?" and (index + 1 == len(window) or window[index + 1].isspace())
    ]
    if sentence_ends and max(sentence_ends) >= limit // 2:
        return window[: max(sentence_ends)].rstrip()
    if " " in window:
        return window.rsplit(" ", 1)[0].rstrip()
    return window.rstrip()


def _safe_validation_conflicts(prefix: str, error: ValidationError) -> list[str]:
    conflicts = []
    for item in error.errors():
        location = item.get("loc", ())
        if not isinstance(location, (tuple, list)):
            location = (location,)
        safe_parts = []
        for part in location:
            if isinstance(part, int):
                safe_parts.append(str(part))
            elif isinstance(part, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part):
                safe_parts.append(part)
            else:
                safe_parts.append("unknown")
        path = ".".join([prefix, *safe_parts]) if safe_parts else prefix
        error_type = item.get("type")
        if not isinstance(error_type, str) or not re.fullmatch(
            r"[A-Za-z0-9_.-]+", error_type
        ):
            error_type = "unknown"
        conflicts.append(f"{path}:{error_type}")
    return conflicts or [f"{prefix}:validation_error"]


def _sanitize_bounded_string_list(
    value,
    *,
    path: str,
    limit: int,
    item_limit: int = 240,
) -> tuple[list[str], list[str]]:
    """Keep optional descriptive lists safe without failing core analysis."""
    issues: list[str] = []
    if not isinstance(value, list):
        return [], [f"{path}:{'null' if value is None else 'list_type'}"]

    retained: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, str):
            issues.append(f"{path}.{index}:string_type")
            continue
        normalized = re.sub(r"\s+", " ", item).strip()
        if not normalized:
            issues.append(f"{path}.{index}:empty")
            continue
        bounded = _shorten_at_boundary(normalized, item_limit)
        if bounded != normalized:
            issues.append(f"{path}.{index}:too_long")
        if bounded in seen:
            issues.append(f"{path}.{index}:duplicate")
            continue
        seen.add(bounded)
        retained.append(bounded)

    if len(retained) > limit:
        issues.append(f"{path}:too_many")
        retained = retained[:limit]
    return retained, issues


def _sanitize_camera_analysis(value) -> tuple[dict, list[str]]:
    """Recover a safe camera candidate without weakening core schema validation."""
    fallback = {
        "movement": "drift",
        "visual_evidence": None,
        "screenplay_evidence": None,
        "confidence": 0.0,
    }
    if not isinstance(value, dict):
        return fallback, [f"camera:{'null' if value is None else 'object_type'}"]

    issues: list[str] = []
    allowed_fields = {
        "movement",
        "visual_evidence",
        "screenplay_evidence",
        "confidence",
    }
    if set(value) - allowed_fields:
        issues.append("camera.extra_fields")

    movement = value.get("movement")
    if isinstance(movement, str):
        normalized_movement = re.sub(r"\s+", " ", movement).strip()
        movement = _shorten_at_boundary(normalized_movement, 80)
        if movement != normalized_movement:
            issues.append("camera.movement:too_long")
    if not movement:
        issues.append("camera.movement:invalid")
        movement = fallback["movement"]
    elif not isinstance(movement, str):
        issues.append("camera.movement:string_type")
        movement = fallback["movement"]

    recovered = {"movement": movement}
    for field in ("visual_evidence", "screenplay_evidence"):
        evidence = value.get(field)
        if evidence is None:
            recovered[field] = None
            continue
        if not isinstance(evidence, str):
            issues.append(f"camera.{field}:string_type")
            recovered[field] = None
            continue
        normalized_evidence = re.sub(r"\s+", " ", evidence).strip()
        evidence = _shorten_at_boundary(normalized_evidence, 240)
        recovered[field] = evidence or None
        if evidence != normalized_evidence:
            issues.append(f"camera.{field}:too_long")
        if len(evidence) == 0:
            issues.append(f"camera.{field}:empty")

    confidence = value.get("confidence")
    if (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0 <= confidence <= 1
    ):
        recovered["confidence"] = float(confidence)
    else:
        issues.append("camera.confidence:invalid")
        recovered["confidence"] = fallback["confidence"]

    try:
        return CameraAnalysis.model_validate(recovered).model_dump(), issues
    except ValidationError:
        return fallback, [*issues, "camera:validation_error"]


def _sanitize_scene_entity(value, *, path: str) -> tuple[dict | None, list[str]]:
    """Recover an entity label without allowing malformed evidence to erase it."""
    if not isinstance(value, dict):
        return None, [f"{path}:{'null' if value is None else 'object_type'}"]

    issues: list[str] = []
    allowed_fields = {
        "entity_id",
        "label",
        "semantic_category",
        "action",
        "suggested_motion",
        "visual_evidence",
        "screenplay_evidence",
        "grounding_source",
        "confidence",
        "visual_confidence",
        "support",
        "visible",
        "role_status",
    }
    if set(value) - allowed_fields:
        issues.append(f"{path}:extra_fields")

    label = next(
        (
            value.get(field)
            for field in ("label", "name", "object_name", "entity_name")
            if isinstance(value.get(field), str) and value.get(field).strip()
        ),
        None,
    )
    if label is None:
        return None, [*issues, f"{path}.label:missing"]
    label = _shorten_at_boundary(re.sub(r"\s+", " ", label).strip(), 80)
    if not label:
        return None, [*issues, f"{path}.label:empty"]

    recovered = {"label": label}
    for field, limit in (
        ("entity_id", 64),
        ("semantic_category", 64),
        ("action", 240),
        ("visual_evidence", 240),
        ("screenplay_evidence", 240),
    ):
        raw = value.get(field)
        if field == "action" and raw is None:
            raw = value.get("suggested_motion")
        if raw is None:
            recovered[field] = None
            continue
        if not isinstance(raw, str):
            issues.append(f"{path}.{field}:string_type")
            recovered[field] = None
            continue
        normalized = re.sub(r"\s+", " ", raw).strip()
        bounded = _shorten_at_boundary(normalized, limit)
        if bounded != normalized:
            issues.append(f"{path}.{field}:too_long")
        recovered[field] = bounded or None

    grounding_source = value.get("grounding_source")
    valid_grounding = {
        "visual",
        "screenplay",
        "visual_and_screenplay",
        "unsupported",
    }
    if grounding_source not in valid_grounding:
        if grounding_source is not None:
            issues.append(f"{path}.grounding_source:invalid")
        grounding_source = None

    support = value.get("support")
    if support not in {"supported", "needs_confirmation", "unsupported"}:
        if support is not None:
            issues.append(f"{path}.support:invalid")
        support = "needs_confirmation"
    role_status = value.get("role_status")
    if role_status is not None and role_status not in {
        "certain",
        "uncertain",
        "confirmation_required",
    }:
        issues.append(f"{path}.role_status:literal_error")
        support = "needs_confirmation"

    confidence = value.get("confidence", 0.0)
    if not (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        and 0 <= confidence <= 1
    ):
        issues.append(f"{path}.confidence:invalid")
        confidence = 0.0

    visual_confidence = value.get("visual_confidence", 0.0)
    if not (
        isinstance(visual_confidence, (int, float))
        and not isinstance(visual_confidence, bool)
        and math.isfinite(float(visual_confidence))
        and 0 <= visual_confidence <= 1
    ):
        issues.append(f"{path}.visual_confidence:invalid")
        visual_confidence = 0.0

    visual_evidence = recovered["visual_evidence"]
    screenplay_evidence = recovered["screenplay_evidence"]
    if grounding_source in {"visual", "visual_and_screenplay"} and (
        not visual_evidence or visual_confidence <= 0
    ):
        issues.append(f"{path}.visual_grounding:downgraded")
        recovered["visual_evidence"] = None
        visual_evidence = None
        visual_confidence = 0.0
        grounding_source = "screenplay" if screenplay_evidence else None
        support = "needs_confirmation"
    if grounding_source in {"screenplay", "visual_and_screenplay"} and not screenplay_evidence:
        issues.append(f"{path}.screenplay_grounding:downgraded")
        grounding_source = "visual" if visual_evidence and visual_confidence > 0 else None
        support = "needs_confirmation"
    if grounding_source == "unsupported":
        support = "unsupported"
        recovered["visual_evidence"] = None
        recovered["screenplay_evidence"] = None
        visual_confidence = 0.0
    if support == "unsupported":
        grounding_source = "unsupported"
        recovered["visual_evidence"] = None
        recovered["screenplay_evidence"] = None
        visual_confidence = 0.0

    recovered.update(
        grounding_source=grounding_source,
        confidence=float(confidence),
        visual_confidence=float(visual_confidence),
        support=support,
    )
    entity_id = recovered.get("entity_id")
    if entity_id is not None and not entity_id:
        recovered["entity_id"] = None

    try:
        return SceneEntity.model_validate(recovered).model_dump(), issues
    except ValidationError as error:
        return None, [*issues, *_safe_validation_conflicts(path, error)]


def _sanitize_scene_analysis_payload(payload: dict) -> dict:
    allowed_fields = {
        "scene_summary",
        "characters",
        "objects",
        "environment",
        "camera",
        "preserve",
        "prohibit",
        "conflicts",
        "main_character_id",
        "relationships",
    }
    sanitized = dict(payload)
    diagnostics: list[str] = []

    scene_summary = sanitized.get("scene_summary")
    if not isinstance(scene_summary, str):
        sanitized["scene_summary"] = "Scene summary unavailable."
        diagnostics.append(
            f"scene_summary:{'null' if scene_summary is None else 'string_type'}"
        )
    else:
        normalized_summary = re.sub(r"\s+", " ", scene_summary).strip()
        sanitized["scene_summary"] = (
            _shorten_at_boundary(normalized_summary, 400)
            or "Scene summary unavailable."
        )
        if not normalized_summary:
            diagnostics.append("scene_summary:empty")

    main_character_id = sanitized.get("main_character_id")
    if main_character_id is not None:
        if not isinstance(main_character_id, str):
            sanitized["main_character_id"] = None
            diagnostics.append("main_character_id:string_type")
        else:
            normalized_id = main_character_id.strip()
            if not normalized_id:
                sanitized["main_character_id"] = None
                diagnostics.append("main_character_id:empty")
            else:
                sanitized["main_character_id"] = _shorten_at_boundary(normalized_id, 64)

    conflicts, conflict_issues = _sanitize_bounded_string_list(
        sanitized.get("conflicts", []),
        path="conflicts",
        limit=8,
    )
    diagnostics.extend(conflict_issues)

    for field, limit in (
        ("preserve", 12),
        ("prohibit", 16),
    ):
        sanitized[field], issues = _sanitize_bounded_string_list(
            sanitized.get(field, []),
            path=field,
            limit=limit,
        )
        diagnostics.extend(issues)

    sanitized["camera"], camera_issues = _sanitize_camera_analysis(sanitized.get("camera"))
    diagnostics.extend(camera_issues)

    retained_ids: set[str] = set()
    for category, limit in (
        ("characters", 12),
        ("objects", 16),
        ("environment", 12),
    ):
        entries = sanitized.get(category)
        if category not in sanitized:
            sanitized[category] = []
            continue
        if not isinstance(entries, list):
            sanitized[category] = []
            diagnostics.append(
                f"{category}:{'null' if entries is None else 'list_type'}"
            )
            continue
        if len(entries) > limit:
            diagnostics.append(f"{category}:too_many")
        retained_entities = []
        for index, entry in enumerate(entries):
            recovered_entity, entity_issues = _sanitize_scene_entity(
                entry,
                path=f"{category}.{index}",
            )
            diagnostics.extend(entity_issues)
            if recovered_entity is None:
                continue
            if len(retained_entities) >= limit:
                break
            retained_entities.append(recovered_entity)
            if recovered_entity.get("entity_id"):
                retained_ids.add(recovered_entity["entity_id"])
        sanitized[category] = retained_entities

    relationship_entries = sanitized.get("relationships", [])
    if isinstance(relationship_entries, list):
        retained_relationships = []
        validated_relationships = []

        for index, entry in enumerate(relationship_entries):
            try:
                relationship = SceneRelationship.model_validate(entry)
            except ValidationError as error:
                conflicts.extend(_safe_validation_conflicts(f"relationships.{index}", error))
                continue
            validated_relationships.append((index, relationship))
        if len(validated_relationships) > 24:
            diagnostics.append("relationships:too_many")
            validated_relationships = validated_relationships[:24]
        for index, relationship in validated_relationships:
            missing_fields = [
                field
                for field, entity_id in (
                    ("source_id", relationship.source_id),
                    ("target_id", relationship.target_id),
                )
                if entity_id not in retained_ids
            ]
            if missing_fields:
                conflicts.extend(
                    f"relationships.{index}.{field}:reference_not_found"
                    for field in missing_fields
                )
                continue
            retained_relationships.append(relationship.model_dump())
        sanitized["relationships"] = retained_relationships
    elif sanitized.get("relationships") is None:
        sanitized["relationships"] = []
        diagnostics.append("relationships:null")
    elif "relationships" in sanitized:
        sanitized["relationships"] = []
        diagnostics.append("relationships:list_type")

    all_diagnostics = list(dict.fromkeys([*diagnostics, *conflicts]))
    sanitized["conflicts"] = _sanitize_bounded_string_list(
        all_diagnostics,
        path="conflicts",
        limit=8,
    )[0][:8]
    return sanitized


def _parse_scene_analysis_response(text: str) -> SceneAnalysis:
    normalized = _strip_json_fence(text)
    payload = json.loads(normalized)
    if not isinstance(payload, dict):
        raise ValueError("SceneAnalysis JSON must be an object")
    sanitized = _sanitize_scene_analysis_payload(payload)
    analysis = SceneAnalysis.model_validate(sanitized)
    if not any(
        analysis.characters or analysis.objects or analysis.environment
    ):
        raise ValueError("SceneAnalysis has no recoverable semantic entities")
    return analysis


def generate_shot_plan(screenplay: str, mood: str, image_data_url: str | None) -> ShotPlan:
    routing = route_scene_action(screenplay, mood)
    user_prompt = (
        f"CREATIVE INTENT: {mood}\n\nSCREENPLAY:\n{screenplay}\n\n"
        "SEMANTIC ACTION ROUTING:\n"
        f"classification={routing.classification}\n"
        f"rationale={routing.rationale}\n"
        "Return only the compact semantic JSON contract described above. Do not return ShotPlan "
        "or renderer fields. Request confirmation for screenplay-grounded motion when visual "
        "confidence is low, preserve uncertain role classifications, surface image/screenplay "
        "conflicts, and do not replace explicit screenplay-grounded motion with camera-only motion. "
        "Use stable entity IDs and minimal relationship records. Return one JSON object only, "
        "with no markdown fences or surrounding prose."
    )
    image = _image_part(image_data_url)
    user_parts = [types.Part.from_text(text=user_prompt)]
    if image:
        user_parts.append(image)
    contents = [types.Content(role="user", parts=user_parts)]
    with _client() as client:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=DIRECTOR_PROMPT,
                response_mime_type="application/json",
                max_output_tokens=DIRECTOR_OUTPUT_TOKEN_BUDGET,
                temperature=0.55,
            ),
        )
    response_text = _response_text(response)
    try:
        analysis = _parse_scene_analysis_response(response_text)
    except (ValidationError, ValueError, TypeError):
        _log_response_diagnostics(response, response_text)
        raise
    return scene_analysis_to_shot_plan(analysis, screenplay, mood)


def generate_critique(screenplay: str, plan: ShotPlan) -> Critique:
    payload = json.dumps({"screenplay": screenplay, "shot_plan": plan.model_dump()})
    with _client() as client:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=CRITIC_PROMPT,
                response_mime_type="application/json",
                response_schema=Critique,
                temperature=0.35,
            ),
        )
    return Critique.model_validate_json(response.text)
