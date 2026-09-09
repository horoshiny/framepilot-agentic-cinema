import base64
import json
import logging
import math
import os
import re
from typing import Annotated, get_args, get_origin

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
DIRECTOR_OUTPUT_TOKEN_BUDGET = 16384
DIRECTOR_OUTPUT_TOKEN_BUDGET_ENV = "GEMINI_DIRECTOR_OUTPUT_TOKEN_BUDGET"
_MIN_DIRECTOR_OUTPUT_TOKEN_BUDGET = 1024
_MAX_DIRECTOR_OUTPUT_TOKEN_BUDGET = 65536
logger = logging.getLogger(__name__)


DIRECTOR_PROMPT = """You are FramePilot's multimodal scene analyst. Analyze the screenplay, creative
intent, and optional storyboard image. Return exactly one compact JSON object with:
scene_summary:string; mood:string; characters:SceneEntity[]; objects:SceneEntity[];
environment:SceneEntity[]; camera:CameraAnalysis; preserve:string[]; prohibit:string[];
conflicts:string[]; main_character_id:string|null; main_character_ids:string[];
relationships:SceneRelationship[].

SceneEntity records contain only:
{"entity_id":string|null,"label":string,"entity_type":"character"|"object"|"environment",
"agentive":boolean,"semantic_category":string|null,
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

Compactness rules: mood is one short sentence; scene_summary is at most two short sentences; labels
are short noun phrases; visual_evidence is one concise sentence or null; screenplay_evidence is one
concise sentence or null; action is one concise sentence or null; and relationship action is one
concise sentence or null. Use short source-local evidence excerpts, not explanations. Do not repeat
the same evidence in multiple fields. Do not repeat the creative intent or preservation/prohibition
rules inside entities. Put shared preservation and prohibition rules once in preserve and prohibit.
Avoid explanatory prose outside the JSON object. Keep scene_summary under 400 characters; use at most
12 characters, 16 objects, 12 environment elements, 24 relationships, 12 preserve rules, 16 prohibit
rules, and 8 conflicts. Keep labels under 80 characters, IDs under 64, categories under 64, actions
and evidence under 240. Preserve every clearly visible, motion-relevant entity and camera opportunity;
when limits compete, prioritize distinct entity coverage and motion-planning relationships over verbose
descriptions.

A character is a narratively agentive entity, not necessarily human or living. Set entity_type to
character and agentive to true for any visible agentive or living subject, including a person, animal,
creature, robot, vehicle, plant, animated object, or abstract/stylized figure. A visible group must
remain a set of individual entities; never place people in environment. The main-character decision
comes from visual prominence, composition, agency, screenplay role, actions, and relationships —
not size, centrality, or human appearance, and not from whether the subject is human. Return
main_character_ids for all co-equal main characters and
keep main_character_id as the first ID for compatibility. Every main ID must reference an entity_id
in characters. An ordinary non-agentive physical item is an object unless the scene presents it as
acting, choosing, reacting, or functioning as the narrative subject.

Set entity_type to object and agentive to false for a visible non-agentive physical item, prop,
garment, accessory, toy, book, tool, furniture item, or structure that can be independently
referenced or moved. Set entity_type to environment and agentive to false only for surrounding
atmosphere, weather, lighting, terrain, vegetation, water, particles, background architecture, or
ambient effects. A physical item must not be classified as environmental motion. Do not use fixed
character types, production-scene labels, or scene-specific noun rules. Do not return one entity in
more than one category.

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


def _effective_director_output_token_budget() -> int:
    configured = os.getenv(DIRECTOR_OUTPUT_TOKEN_BUDGET_ENV)
    if configured is None:
        return DIRECTOR_OUTPUT_TOKEN_BUDGET
    try:
        value = int(configured.strip())
    except (AttributeError, TypeError, ValueError):
        return DIRECTOR_OUTPUT_TOKEN_BUDGET
    if not _MIN_DIRECTOR_OUTPUT_TOKEN_BUDGET <= value <= _MAX_DIRECTOR_OUTPUT_TOKEN_BUDGET:
        return DIRECTOR_OUTPUT_TOKEN_BUDGET
    return value


def _log_response_diagnostics(
    response,
    text: str,
    *,
    output_token_budget: int | None = None,
) -> None:
    trimmed = text.strip()
    candidates = getattr(response, "candidates", None)
    candidate_count = len(candidates) if isinstance(candidates, (list, tuple)) else 0
    effective_budget = (
        output_token_budget
        if isinstance(output_token_budget, int)
        else _effective_director_output_token_budget()
    )
    logger.warning(
        "Vertex scene analysis response diagnostics: candidate_count=%d "
        "finish_reason=%s response_text_length=%d starts_with_brace=%s "
        "ends_with_brace=%s markdown_fences_present=%s empty_response=%s "
        "max_output_tokens=%d",
        candidate_count,
        _finish_reason(response),
        len(text),
        trimmed.startswith("{"),
        trimmed.endswith("}"),
        "```" in trimmed,
        not bool(trimmed),
        effective_budget,
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
    if sentence_ends:
        return window[: max(sentence_ends)].rstrip()
    if " " in window:
        return window.rsplit(" ", 1)[0].rstrip()
    return window.rstrip()


def _max_length_from_metadata(metadata) -> int | None:
    for item in metadata:
        max_length = getattr(item, "max_length", None)
        if isinstance(max_length, int):
            return max_length
        nested_metadata = getattr(item, "metadata", None)
        if nested_metadata:
            nested_limit = _max_length_from_metadata(nested_metadata)
            if nested_limit is not None:
                return nested_limit
    return None


def _declared_max_length(model_type, field_name: str, *, item: bool = False) -> int:
    field = model_type.model_fields.get(field_name)
    if field is None:
        raise KeyError(f"{model_type.__name__}.{field_name} is not declared")

    if not item:
        limit = _max_length_from_metadata(field.metadata)
    else:
        annotation = field.annotation
        item_args = get_args(annotation)
        if get_origin(annotation) in {list, tuple, set, frozenset} and item_args:
            annotation = item_args[0]
        if get_origin(annotation) is Annotated:
            limit = _max_length_from_metadata(get_args(annotation)[1:])
        else:
            limit = None
    if limit is None:
        raise ValueError(f"{model_type.__name__}.{field_name} has no declared string maximum")
    return limit


def _shorten_declared_string(
    value: str,
    model_type,
    field_name: str,
    *,
    item: bool = False,
) -> tuple[str, bool]:
    limit = _declared_max_length(model_type, field_name, item=item)
    shortened = _shorten_at_boundary(value, limit)
    return shortened, shortened != value


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
    item_limit: int,
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
        movement, shortened = _shorten_declared_string(
            normalized_movement,
            CameraAnalysis,
            "movement",
        )
        if shortened:
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
        evidence, shortened = _shorten_declared_string(
            normalized_evidence,
            CameraAnalysis,
            field,
        )
        recovered[field] = evidence or None
        if shortened:
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


def _sanitize_scene_entity(
    value,
    *,
    path: str,
    category: str | None = None,
) -> tuple[dict | None, list[str]]:
    """Recover an entity label without allowing malformed evidence to erase it."""
    if not isinstance(value, dict):
        return None, [f"{path}:{'null' if value is None else 'object_type'}"]

    issues: list[str] = []
    allowed_fields = {
        "entity_id",
        "label",
        "entity_type",
        "type",
        "agentive",
        "is_agentive",
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
    label, label_shortened = _shorten_declared_string(
        re.sub(r"\s+", " ", label).strip(),
        SceneEntity,
        "label",
    )
    if label_shortened:
        issues.append(f"{path}.label:too_long")
    if not label:
        return None, [*issues, f"{path}.label:empty"]

    entity_type = value.get("entity_type", value.get("type"))
    if entity_type not in {"character", "object", "environment", None}:
        issues.append(f"{path}.entity_type:literal_error")
        entity_type = None
    recovered = {"label": label, "entity_type": entity_type or category}

    agentive = value.get("agentive", value.get("is_agentive", False))
    if not isinstance(agentive, bool):
        issues.append(f"{path}.agentive:bool_type")
        agentive = False
    recovered["agentive"] = agentive
    for field in (
        "semantic_category",
        "action",
        "visual_evidence",
        "screenplay_evidence",
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
        bounded, shortened = _shorten_declared_string(
            normalized,
            SceneEntity,
            field,
        )
        if shortened:
            issues.append(f"{path}.{field}:too_long")
        recovered[field] = bounded or None

    entity_id = value.get("entity_id")
    if entity_id is None:
        recovered["entity_id"] = None
    elif not isinstance(entity_id, str):
        issues.append(f"{path}.entity_id:string_type")
        recovered["entity_id"] = None
    else:
        normalized_id = entity_id.strip()
        if not normalized_id:
            issues.append(f"{path}.entity_id:empty")
            recovered["entity_id"] = None
        elif len(normalized_id) > _declared_max_length(SceneEntity, "entity_id"):
            # IDs are references, not descriptive prose: never shorten them.
            issues.append(f"{path}.entity_id:string_too_long")
            return None, issues
        else:
            recovered["entity_id"] = normalized_id

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


def _normalize_entity_categories(
    categories: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], list[str]]:
    """Resolve explicit structured taxonomy flags without scene-specific heuristics."""
    category_names = ("characters", "objects", "environment")
    category_for_type = {
        "character": "characters",
        "object": "objects",
        "environment": "environment",
    }
    limits = {"characters": 12, "objects": 16, "environment": 12}
    normalized = {category: [] for category in category_names}
    issues: list[str] = []
    seen_ids: set[str] = set()

    for source_category in category_names:
        for index, entity in enumerate(categories.get(source_category, [])):
            target_category = category_for_type.get(
                entity.get("entity_type"),
                source_category,
            )
            if entity.get("agentive") is True:
                target_category = "characters"
            entity = {**entity, "entity_type": target_category.rstrip("s")}
            if target_category != source_category:
                issues.append(
                    f"{source_category}.{index}:category_corrected_to_{target_category}"
                )

            entity_id = entity.get("entity_id")
            if entity_id and entity_id in seen_ids:
                issues.append(f"{source_category}.{index}.entity_id:duplicate")
            elif entity_id:
                seen_ids.add(entity_id)

            if len(normalized[target_category]) >= limits[target_category]:
                issues.append(f"{source_category}.{index}:target_category_full")
                continue
            normalized[target_category].append(entity)
    return normalized, issues


def _sanitize_scene_analysis_payload(payload: dict) -> dict:
    allowed_fields = {
        "scene_summary",
        "mood",
        "characters",
        "objects",
        "environment",
        "camera",
        "preserve",
        "prohibit",
        "conflicts",
        "main_character_id",
        "main_character_ids",
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
        sanitized["scene_summary"], shortened = _shorten_declared_string(
            normalized_summary,
            SceneAnalysis,
            "scene_summary",
        )
        if shortened:
            diagnostics.append("scene_summary:string_too_long")
        if not normalized_summary:
            diagnostics.append("scene_summary:empty")
        if not sanitized["scene_summary"]:
            sanitized["scene_summary"] = "Scene summary unavailable."

    mood = sanitized.get("mood")
    if mood is not None:
        if not isinstance(mood, str):
            sanitized.pop("mood", None)
            diagnostics.append("mood:string_type")
        else:
            sanitized["mood"], shortened = _shorten_declared_string(
                mood,
                SceneAnalysis,
                "mood",
            )
            if shortened:
                diagnostics.append("mood:string_too_long")

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
            elif len(normalized_id) > _declared_max_length(SceneAnalysis, "main_character_id"):
                sanitized["main_character_id"] = None
                diagnostics.append("main_character_id:string_too_long")
            else:
                sanitized["main_character_id"] = normalized_id

    raw_main_character_ids = sanitized.get("main_character_ids", [])
    main_character_ids: list[str] = []
    if raw_main_character_ids is None:
        diagnostics.append("main_character_ids:null")
    elif not isinstance(raw_main_character_ids, list):
        diagnostics.append("main_character_ids:list_type")
    else:
        for index, value in enumerate(raw_main_character_ids):
            if not isinstance(value, str):
                diagnostics.append(f"main_character_ids.{index}:string_type")
                continue
            normalized_id = value.strip()
            if not normalized_id:
                diagnostics.append(f"main_character_ids.{index}:empty")
                continue
            if len(normalized_id) > _declared_max_length(
                SceneAnalysis,
                "main_character_ids",
                item=True,
            ):
                diagnostics.append(f"main_character_ids.{index}:string_too_long")
                continue
            if normalized_id in main_character_ids:
                diagnostics.append(f"main_character_ids.{index}:duplicate")
                continue
            main_character_ids.append(normalized_id)
    if len(main_character_ids) > _declared_max_length(SceneAnalysis, "main_character_ids"):
        diagnostics.append("main_character_ids:too_many")
        main_character_ids = main_character_ids[
            : _declared_max_length(SceneAnalysis, "main_character_ids")
        ]

    singular_main_character_id = sanitized.get("main_character_id")
    if main_character_ids:
        if (
            singular_main_character_id
            and singular_main_character_id not in main_character_ids
        ):
            diagnostics.append("main_character_id:not_in_main_character_ids")
        sanitized["main_character_id"] = main_character_ids[0]
    elif singular_main_character_id:
        main_character_ids = [singular_main_character_id]
    sanitized["main_character_ids"] = main_character_ids

    conflicts, conflict_issues = _sanitize_bounded_string_list(
        sanitized.get("conflicts", []),
        path="conflicts",
        limit=8,
        item_limit=_declared_max_length(SceneAnalysis, "conflicts", item=True),
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
            item_limit=_declared_max_length(SceneAnalysis, field, item=True),
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
                category=category.rstrip("s"),
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

    normalized_categories, category_issues = _normalize_entity_categories(
        {
            category: sanitized.get(category, [])
            for category in ("characters", "objects", "environment")
        }
    )
    diagnostics.extend(category_issues)
    sanitized.update(normalized_categories)
    retained_ids = {
        entity["entity_id"]
        for category in normalized_categories.values()
        for entity in category
        if entity.get("entity_id")
    }

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
        item_limit=_declared_max_length(SceneAnalysis, "conflicts", item=True),
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
    output_token_budget = _effective_director_output_token_budget()
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
                max_output_tokens=output_token_budget,
                temperature=0.55,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
    response_text = _response_text(response)
    try:
        analysis = _parse_scene_analysis_response(response_text)
    except (ValidationError, ValueError, TypeError):
        _log_response_diagnostics(
            response,
            response_text,
            output_token_budget=output_token_budget,
        )
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
