from __future__ import annotations

import re
import unicodedata

from pydantic import ValidationError

from .schemas import (
    CameraAnalysis,
    MotionBounds,
    MotionCandidate,
    MotionPlan,
    SceneAnalysis,
    SceneEntity,
    SceneRelationship,
    ShotParameters,
    ShotPlan,
)


_FULL_FRAME_BOUNDS = MotionBounds(x_min=0, y_min=0, x_max=1000, y_max=1000)
_CAMERA_MOTION_VALUES = {"push_in", "pull_out", "pan_left", "pan_right", "drift"}
_ZOOM_END = {
    "push_in": 1.12,
    "pull_out": 1.0,
    "pan_left": 1.05,
    "pan_right": 1.05,
    "drift": 1.04,
}
_MAX_MOTION_LABEL_LENGTH = 120
_ACTION_WORDS = frozenset(
    {
        "bend",
        "bends",
        "bending",
        "close",
        "closes",
        "closing",
        "drift",
        "drifts",
        "drifting",
        "fall",
        "falls",
        "falling",
        "flicker",
        "flickers",
        "flickering",
        "flow",
        "flows",
        "flowing",
        "glow",
        "glows",
        "glowing",
        "lift",
        "lifts",
        "lifting",
        "move",
        "moves",
        "moving",
        "open",
        "opens",
        "opening",
        "pulse",
        "pulses",
        "pulsing",
        "raise",
        "raises",
        "raising",
        "reach",
        "reaches",
        "reaching",
        "rotate",
        "rotates",
        "rotating",
        "run",
        "runs",
        "running",
        "swing",
        "swings",
        "swinging",
        "turn",
        "turns",
        "turning",
        "walk",
        "walks",
        "walking",
        "wave",
        "waves",
        "waving",
        "is",
        "are",
        "was",
        "were",
    }
)
_ACTION_WORD_PATTERN = re.compile(
    r"\b(?:" + "|".join(sorted(_ACTION_WORDS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_LABEL_QUALIFIERS = re.compile(
    r"\b(?:once|twice|slowly|gently|carefully|gradually|slightly|"
    r"subtly|briefly|quietly|rapidly|again)\b",
    re.IGNORECASE,
)
_MALFORMED_CANDIDATE_WARNING = (
    "One malformed motion candidate was omitted during deterministic label normalization."
)
_DUPLICATE_ENTITY_WARNING = (
    "One entity ID appeared in more than one semantic category; the duplicate candidate was omitted."
)
_INVALID_RELATIONSHIP_WARNING = (
    "One relationship referenced an entity ID that was not preserved in the compact analysis."
)
_MAIN_CHARACTER_WARNING = (
    "The declared main character ID did not resolve to one visible character entity."
)


def _clean_label_text(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>\r\n]*>", " ", text)
    text = text.replace("**", "").replace("__", "").replace("~~", "").replace("`", "")
    text = "".join(
        " " if unicodedata.category(character).startswith("C") else character
        for character in text
    )
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip("*`~")
    return text.rstrip(" .,;:!?…。！？")


def _action_entity_phrase(text: str) -> str:
    match = _ACTION_WORD_PATTERN.search(text)
    if not match:
        return text

    before = text[: match.start()].strip(" ,:;-")
    if before:
        before = re.sub(r"^(?:a|an|the)\s+", "", before, flags=re.IGNORECASE)
        before = _LABEL_QUALIFIERS.sub(" ", before)
        before = re.sub(r"\s+", " ", before).strip(" ,:;-")
        if before:
            return before

    after = text[match.end() :].strip(" ,:;-")
    after = re.split(
        r"\b(?:across|along|around|at|beside|by|during|for|from|"
        r"into|on|over|through|toward|towards|under|while|with|"
        r"without)\b",
        after,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    after = _LABEL_QUALIFIERS.sub(" ", after)
    after = re.sub(r"^(?:a|an|the)\s+", "", after, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", after).strip(" ,:;-")


def _normalize_motion_label(value: str | None) -> str | None:
    text = _clean_label_text(value)
    if not text:
        return None
    text = _action_entity_phrase(text)
    text = _clean_label_text(text)
    if not text:
        return None
    if len(text) > _MAX_MOTION_LABEL_LENGTH:
        truncated = text[:_MAX_MOTION_LABEL_LENGTH].rstrip()
        text = truncated.rsplit(" ", 1)[0].rstrip() if " " in truncated else truncated
    return text or None


def _motion_candidate(
    label: str,
    **fields,
) -> tuple[MotionCandidate | None, bool]:
    normalized_label = _normalize_motion_label(label)
    if not normalized_label:
        return None, True
    try:
        return MotionCandidate(label=normalized_label, **fields), False
    except ValidationError:
        return None, True


def _has_visual_evidence(entity: SceneEntity | CameraAnalysis) -> bool:
    if isinstance(entity, SceneEntity):
        return bool(
            entity.visual_evidence
            and entity.grounding_source in {None, "visual", "visual_and_screenplay"}
        )
    return bool(entity.visual_evidence)


def _has_screenplay_evidence(entity: SceneEntity | CameraAnalysis) -> bool:
    return bool(entity.screenplay_evidence)


def _grounding(entity: SceneEntity | CameraAnalysis) -> tuple[str, str, float, float]:
    visual = _has_visual_evidence(entity)
    screenplay = _has_screenplay_evidence(entity)
    if isinstance(entity, SceneEntity) and entity.support == "unsupported":
        return "unsupported", "unsupported", 0.0, 0.0
    if isinstance(entity, SceneEntity) and entity.grounding_source is not None:
        support = entity.support
        if support == "unsupported":
            return "unsupported", "unsupported", 0.0, 0.0
        return (
            entity.grounding_source,
            support,
            entity.confidence,
            entity.visual_confidence,
        )
    if visual and screenplay:
        support = (
            "needs_confirmation"
            if isinstance(entity, SceneEntity) and entity.support != "supported"
            else "supported"
        )
        return "visual_and_screenplay", support, entity.confidence, entity.confidence
    if visual:
        support = (
            "needs_confirmation"
            if isinstance(entity, SceneEntity) and entity.role_status != "certain"
            else "supported"
        )
        return "visual", support, entity.confidence, entity.confidence
    if screenplay:
        return "screenplay", "needs_confirmation", entity.confidence, 0.0
    return "unsupported", "unsupported", 0.0, 0.0


def _evidence_text(value: str | None, fallback: str) -> str:
    return value.strip() if value and value.strip() else fallback


def _entity_bound(category: str) -> str:
    if category == "character":
        return (
            "Keep the named narrative agentive entity within its existing identity, silhouette, "
            "and frame position."
        )
    if category == "object":
        return "Keep the named object within its existing identity, shape, and frame position."
    return "Keep the named environmental element within the existing composition and effect budget."


def _entity_candidate(
    entity: SceneEntity,
    category: str,
    mood: str,
) -> tuple[MotionCandidate | None, bool]:
    if not entity.action:
        return None, False
    grounding_source, support, confidence, visual_confidence = _grounding(entity)
    if entity.support != "supported" and support != "unsupported":
        support = "needs_confirmation"
    return _motion_candidate(
        entity.label,
        action=entity.action,
        bound=_entity_bound(category),
        visual_evidence=_evidence_text(
            entity.visual_evidence,
            "The storyboard does not provide separate visual evidence for this motion.",
        ),
        screenplay_evidence=_evidence_text(
            entity.screenplay_evidence,
            "The screenplay does not provide separate evidence for this motion.",
        ),
        intent_evidence=f"Creative intent: {mood or 'cinematic'}.",
        confidence=confidence,
        visual_confidence=visual_confidence,
        grounding_source=grounding_source,
        support=support,
        bounds=_FULL_FRAME_BOUNDS.model_copy(),
    )


def _camera_motion_value(movement: str) -> str | None:
    normalized = movement.lower().replace("-", " ").replace("_", " ")
    if normalized in _CAMERA_MOTION_VALUES:
        return normalized
    if "pan" in normalized or "track" in normalized or "slide" in normalized:
        if "left" in normalized:
            return "pan_left"
        if "right" in normalized:
            return "pan_right"
    if "push" in normalized or "zoom" in normalized or "move closer" in normalized:
        if "out" in normalized or "away" in normalized:
            return "pull_out"
        return "push_in"
    if "pull" in normalized or "away" in normalized:
        return "pull_out"
    return None


def _camera_candidate(
    analysis: CameraAnalysis,
    mood: str,
    camera_motion: str | None,
) -> tuple[MotionCandidate | None, bool]:
    if camera_motion is None:
        grounding_source = "unsupported"
        support = "unsupported"
        confidence = 0.0
        visual_confidence = 0.0
    elif _has_visual_evidence(analysis):
        grounding_source, support, confidence, visual_confidence = _grounding(analysis)
    else:
        # A still image cannot prove camera movement. The compact analysis's
        # requested movement is retained as screenplay/intent-grounded motion.
        grounding_source = "screenplay"
        support = "needs_confirmation"
        confidence = max(analysis.confidence, 0.01)
        visual_confidence = 0.0
    screenplay_evidence = _evidence_text(
        analysis.screenplay_evidence,
        (
            "Camera movement was requested by the screenplay or creative intent: "
            f"{analysis.movement}."
        ),
    )
    visual_evidence = _evidence_text(
        analysis.visual_evidence,
        "The still storyboard does not visually prove camera movement.",
    )
    return _motion_candidate(
        "camera",
        action=analysis.movement,
        bound="Stay within the compositor's overscan and declared pan/zoom limits.",
        visual_evidence=visual_evidence,
        screenplay_evidence=screenplay_evidence,
        intent_evidence=f"Creative intent: {mood or 'cinematic'}.",
        confidence=confidence,
        visual_confidence=visual_confidence,
        grounding_source=grounding_source,
        support=support,
        bounds=_FULL_FRAME_BOUNDS.model_copy(),
    )


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        clean = value.strip()
        if clean and clean not in result:
            result.append(clean)
    return result


def scene_analysis_to_shot_plan(
    analysis: SceneAnalysis,
    screenplay: str,
    mood: str,
) -> ShotPlan:
    category_inputs = (
        ("character", analysis.characters),
        ("object", analysis.objects),
        ("environment", analysis.environment),
    )
    category_candidates: dict[str, list[MotionCandidate]] = {
        "character": [],
        "object": [],
        "environment": [],
    }
    unsupported_actions: list[MotionCandidate] = []
    accepted_entities: list[SceneEntity] = []
    entity_categories: dict[str, str] = {}
    entity_conflicts: list[str] = []
    malformed_candidate = False
    for category, entities in category_inputs:
        for entity in entities:
            if entity.entity_id:
                previous_category = entity_categories.get(entity.entity_id)
                if previous_category:
                    entity_conflicts.append(_DUPLICATE_ENTITY_WARNING)
                    continue
                entity_categories[entity.entity_id] = category
            accepted_entities.append(entity)
            candidate, malformed = _entity_candidate(entity, category, mood)
            if malformed:
                malformed_candidate = True
            if candidate is None:
                continue
            if candidate.support == "unsupported":
                unsupported_actions.append(candidate)
            else:
                category_candidates[category].append(candidate)

    camera_motion = _camera_motion_value(analysis.camera.movement)
    camera, malformed = _camera_candidate(analysis.camera, mood, camera_motion)
    malformed_candidate = malformed_candidate or malformed
    if camera is None:
        # The fixed camera label and bounded internal fields should always
        # validate, but keep this path fault-isolated if the schema changes.
        camera, fallback_malformed = _motion_candidate(
            "camera",
            action="drift",
            bound="Stay within the compositor's overscan and declared pan/zoom limits.",
            visual_evidence="The still storyboard does not visually prove camera movement.",
            screenplay_evidence="No separate camera evidence was supplied.",
            intent_evidence=f"Creative intent: {mood or 'cinematic'}.",
            confidence=0.01,
            visual_confidence=0.0,
            grounding_source="screenplay",
            support="needs_confirmation",
            bounds=_FULL_FRAME_BOUNDS.model_copy(),
        )
        malformed_candidate = malformed_candidate or fallback_malformed
        if camera is None:
            raise RuntimeError("Unable to construct the required camera motion candidate")
    if camera.support == "unsupported":
        unsupported_actions.append(camera)
    conflicts = _unique([*analysis.conflicts, *entity_conflicts])
    relationships: list[SceneRelationship] = []
    for relationship in analysis.relationships:
        if (
            relationship.source_id not in entity_categories
            or relationship.target_id not in entity_categories
        ):
            conflicts.append(_INVALID_RELATIONSHIP_WARNING)
            continue
        relationships.append(relationship)
    main_character = next(
        (
            entity
            for entity in accepted_entities
            if entity.entity_id == analysis.main_character_id
            and entity_categories.get(entity.entity_id) == "character"
            and _has_visual_evidence(entity)
        ),
        None,
    )
    if analysis.main_character_id and main_character is None:
        conflicts.append(_MAIN_CHARACTER_WARNING)
    grounded_entities = [
        (entity, _normalize_motion_label(entity.label))
        for entity in accepted_entities
        if (_has_visual_evidence(entity) or _has_screenplay_evidence(entity))
    ]
    focal_subject = (
        _normalize_motion_label(main_character.label)
        if main_character
        else "No confidently identified focal subject."
    )
    preserved_elements = _unique(
        [
            *analysis.preserve,
            *[
                f"Preserve {label}'s visible identity, shape, and spatial relationship."
                for _, label in grounded_entities
                if label
            ],
        ]
    )
    if not preserved_elements:
        preserved_elements = [
            "Preserve all visible subjects, objects, silhouettes, text, and spatial relationships."
        ]
    prohibited_changes = _unique(
        [
            *analysis.prohibit,
            "Do not add, replace, deform, or transform entities absent from the approved evidence.",
            "Preserve identity, subject count, silhouette, text, architecture, lighting direction, and composition.",
        ]
    )
    conflicts = _unique(conflicts)
    if malformed_candidate:
        conflicts = _unique([*conflicts, _MALFORMED_CANDIDATE_WARNING])
    confirmation_candidates = [
        *category_candidates["character"],
        *category_candidates["object"],
        *category_candidates["environment"],
        *unsupported_actions,
    ]
    confirmation_required = bool(
        conflicts
        or unsupported_actions
        or any(candidate.support == "needs_confirmation" for candidate in confirmation_candidates)
        or camera.support == "needs_confirmation"
    )
    confirmation_question = None
    if confirmation_required:
        confirmation_question = (
            "Review the evidence-grounded motion and approve only bounded actions supported by "
            "the storyboard and screenplay; unsupported or conflicting motion remains excluded."
        )

    source_counts = {"visual": 0, "screenplay": 0, "visual_and_screenplay": 0, "unsupported": 0}
    candidates_for_counts = [
        *category_candidates["character"],
        *category_candidates["object"],
        *category_candidates["environment"],
    ]
    if camera.support != "unsupported":
        candidates_for_counts.append(camera)
    candidates_for_counts.extend(unsupported_actions)
    for candidate in candidates_for_counts:
        if candidate.grounding_source in source_counts:
            source_counts[candidate.grounding_source] += 1

    camera_confidence = camera.confidence
    renderer_camera_motion = camera_motion or "drift"
    shot = ShotParameters(
        duration_seconds=8,
        camera_motion=renderer_camera_motion,
        zoom_start=1.0,
        zoom_end=_ZOOM_END[renderer_camera_motion],
        pan_x=(
            -5
            if renderer_camera_motion == "pan_left"
            else 5
            if renderer_camera_motion == "pan_right"
            else 0
        ),
        pan_y=0,
        parallax_strength=0.28,
        motion_intensity=min(1.0, max(0.12, 0.18 + camera_confidence * 0.35)),
        atmosphere=[],
        transition="fade",
    )
    motion_plan = MotionPlan(
        grounding_summary=(
            f"GROUNDING SOURCES: visual={source_counts['visual']}, "
            f"screenplay={source_counts['screenplay']}, "
            f"visual_and_screenplay={source_counts['visual_and_screenplay']}, "
            f"unsupported={source_counts['unsupported']}. "
            "Screenplay-only motion is confirmation-gated; unsupported motion is excluded."
        ),
        movable_characters=category_candidates["character"],
        movable_objects=category_candidates["object"],
        environmental_motion=category_candidates["environment"],
        camera_movement=camera,
        preserved_elements=preserved_elements[:12],
        prohibited_changes=prohibited_changes[:16],
        unsupported_actions=unsupported_actions[:12],
        grounding_conflicts=conflicts[:8],
        confirmation_required=confirmation_required,
        confirmation_question=confirmation_question,
    )
    return ShotPlan(
        scene_summary=analysis.scene_summary,
        emotional_intent=mood or "cinematic",
        focal_subject=focal_subject,
        depth_notes={
            "foreground": "Depth geometry is not asserted by compact analysis; preserve the supplied frame.",
            "subject": "Use only entities and evidence returned by the multimodal analysis.",
            "background": "Keep unsupported background detail unchanged.",
        },
        shot=shot,
        directing_rationale=(
            "Use the smallest controllable camera move supported by the compact multimodal analysis "
            "while preserving the supplied frame and evidence boundaries."
        ),
        motion_plan=motion_plan,
        depth_layout=None,
        relationships=relationships,
    )