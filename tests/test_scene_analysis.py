import pytest

from cinema_agent.scene_analysis import (
    _normalize_motion_label,
    scene_analysis_to_shot_plan,
)
from cinema_agent.schemas import SceneAnalysis


def _analysis(**updates) -> SceneAnalysis:
    payload = {
        "scene_summary": "A generic scene contains several grounded elements.",
        "characters": [],
        "objects": [],
        "environment": [],
        "camera": {
            "movement": "drift",
            "visual_evidence": None,
            "screenplay_evidence": "The camera holds a quiet drift.",
            "confidence": 0.6,
        },
        "preserve": [],
        "prohibit": [],
        "conflicts": [],
    }
    payload.update(updates)
    return SceneAnalysis.model_validate(payload)


def test_motion_label_normalization_handles_generic_text_shapes():
    assert _normalize_motion_label("  courier  ") == "courier"
    assert _normalize_motion_label("lantern\n\tnear the door\x00") == "lantern near the door"
    assert _normalize_motion_label("**Lantern!!!**") == "Lantern"
    assert _normalize_motion_label("Élodie — 翼") == "Élodie — 翼"
    assert _normalize_motion_label("!!! ... \t") is None

    long_label = " ".join(["architectural"] * 40)
    normalized = _normalize_motion_label(long_label)
    assert normalized
    assert len(normalized) <= 120


def test_action_sentence_labels_become_conservative_entity_phrases():
    assert _normalize_motion_label("A courier raises the signal flag.") == "courier"
    assert _normalize_motion_label("Rotate the signal flag once") == "signal flag"


def test_arbitrary_characters_and_multiple_objects_are_converted():
    analysis = _analysis(
        characters=[
            {
                "label": "courier",
                "visible": True,
                "visual_evidence": "A figure is visible.",
                "screenplay_evidence": "A courier waits.",
                "suggested_motion": "Turn toward the road.",
                "confidence": 0.9,
            }
        ],
        objects=[
            {
                "label": "signal flag",
                "visible": True,
                "visual_evidence": "A flag is visible.",
                "screenplay_evidence": "The flag rotates.",
                "suggested_motion": "Rotate once.",
                "confidence": 0.8,
            },
            {
                "label": "wooden crate",
                "visible": True,
                "visual_evidence": "A crate is visible.",
                "screenplay_evidence": None,
                "suggested_motion": None,
                "confidence": 0.7,
            },
        ]
    )

    plan = scene_analysis_to_shot_plan(analysis, "A courier waits.", "measured resolve")

    assert [item.label for item in plan.motion_plan.movable_characters] == ["courier"]
    assert [item.label for item in plan.motion_plan.movable_objects] == ["signal flag"]
    assert any("wooden crate" in item for item in plan.motion_plan.preserved_elements)


def test_one_malformed_motion_candidate_does_not_discard_valid_candidates():
    analysis = _analysis(
        objects=[
            {
                "label": "!!!",
                "visible": True,
                "visual_evidence": "A physical element is visible.",
                "screenplay_evidence": "The element changes.",
                "suggested_motion": "Change only within the existing frame.",
                "confidence": 0.6,
            },
            {
                "label": "lantern",
                "visible": True,
                "visual_evidence": "A lantern is visible.",
                "screenplay_evidence": "The lantern sways.",
                "suggested_motion": "Sway gently.",
                "confidence": 0.85,
            },
        ]
    )

    plan = scene_analysis_to_shot_plan(analysis, "The lantern sways.", "restrained")

    assert [item.label for item in plan.motion_plan.movable_objects] == ["lantern"]
    assert plan.motion_plan.unsupported_actions == []
    assert any(
        "malformed motion candidate was omitted" in item
        for item in plan.motion_plan.grounding_conflicts
    )
    assert plan.motion_plan.confirmation_required is True
    assert plan.motion_plan.camera_movement.label == "camera"


def test_no_character_scene_and_environmental_motion_remain_generic():
    analysis = _analysis(
        environment=[
            {
                "label": "snow",
                "visible": True,
                "visual_evidence": "Fine particles cross the frame.",
                "screenplay_evidence": "Snow drifts through the courtyard.",
                "suggested_motion": "Drift across the existing composition.",
                "confidence": 0.84,
            }
        ]
    )

    plan = scene_analysis_to_shot_plan(
        analysis,
        "Snow drifts through the courtyard.",
        "quiet concentration",
    )

    assert not plan.motion_plan.movable_characters
    assert plan.motion_plan.environmental_motion[0].label == "snow"
    assert plan.motion_plan.environmental_motion[0].grounding_source == "visual_and_screenplay"


def test_gemini_environment_category_stays_environmental_and_props_stay_objects():
    analysis = _analysis(
        objects=[
            {
                "label": "lantern",
                "visible": True,
                "visual_evidence": "A physical lantern is visible.",
                "screenplay_evidence": "The lantern remains beside the steps.",
                "suggested_motion": "Swing slightly.",
                "confidence": 0.8,
            }
        ],
        environment=[
            {
                "label": "ripples",
                "visible": True,
                "visual_evidence": "Concentric ripples are visible on the surface.",
                "screenplay_evidence": "Ripples move across the water.",
                "suggested_motion": "Move outward across the existing surface.",
                "confidence": 0.85,
            },
            {
                "label": "mist",
                "visible": True,
                "visual_evidence": "A mist layer is visible in the distance.",
                "screenplay_evidence": "Mist drifts through the scene.",
                "suggested_motion": "Drift within the existing atmosphere.",
                "confidence": 0.82,
            },
        ],
    )

    plan = scene_analysis_to_shot_plan(analysis, "Ripples move across the water.", "restrained")

    assert [item.label for item in plan.motion_plan.movable_objects] == ["lantern"]
    assert [item.label for item in plan.motion_plan.environmental_motion] == [
        "ripples",
        "mist",
    ]
    assert not any(
        item.label in {"ripples", "mist"} for item in plan.motion_plan.movable_objects
    )


def test_static_visual_elements_are_preserved_without_motion_candidates():
    analysis = _analysis(
        objects=[
            {
                "label": "stone arch",
                "visible": True,
                "visual_evidence": "A stone arch is visible and remains static.",
                "screenplay_evidence": None,
                "suggested_motion": None,
                "confidence": 0.9,
            }
        ]
    )

    plan = scene_analysis_to_shot_plan(analysis, "A stone arch frames the scene.", "still")

    assert not plan.motion_plan.movable_objects
    assert any("stone arch" in item for item in plan.motion_plan.preserved_elements)


def test_camera_only_scene_is_converted_without_inventing_entities():
    analysis = _analysis(
        camera={
            "movement": "push in",
            "visual_evidence": "The composition supports a gradual inward move.",
            "screenplay_evidence": "The camera pushes in.",
            "confidence": 0.95,
        }
    )

    plan = scene_analysis_to_shot_plan(analysis, "The camera pushes in.", "focused")

    assert not plan.motion_plan.movable_characters
    assert not plan.motion_plan.movable_objects
    assert not plan.motion_plan.environmental_motion
    assert plan.shot.camera_motion == "push_in"
    assert plan.motion_plan.camera_movement.grounding_source == "visual_and_screenplay"


def test_intent_only_camera_motion_is_confirmation_gated_not_unsupported():
    analysis = _analysis(
        camera={
            "movement": "push in",
            "visual_evidence": None,
            "screenplay_evidence": None,
            "confidence": 0.7,
        }
    )

    plan = scene_analysis_to_shot_plan(
        analysis,
        "Hold on the still composition.",
        "The camera moves inward with quiet intimacy.",
    )
    camera = plan.motion_plan.camera_movement

    assert camera.action == "push in"
    assert camera.grounding_source == "screenplay"
    assert camera.support == "needs_confirmation"
    assert camera.visual_confidence == 0
    assert camera not in plan.motion_plan.unsupported_actions
    assert plan.motion_plan.confirmation_required is True
    assert "Camera movement was requested" in camera.screenplay_evidence


def test_camera_motion_outside_renderer_constraints_is_excluded():
    analysis = _analysis(
        camera={
            "movement": "orbit 360 degrees around the subject",
            "visual_evidence": None,
            "screenplay_evidence": "Orbit 360 degrees around the subject.",
            "confidence": 0.9,
        }
    )

    plan = scene_analysis_to_shot_plan(
        analysis,
        "Orbit 360 degrees around the subject.",
        "kinetic",
    )
    camera = plan.motion_plan.camera_movement

    assert camera.grounding_source == "unsupported"
    assert camera.support == "unsupported"
    assert camera in plan.motion_plan.unsupported_actions
    assert plan.shot.camera_motion == "drift"
    assert plan.motion_plan.confirmation_required is True


def test_screenplay_only_and_conflicting_evidence_require_confirmation():
    analysis = _analysis(
        objects=[
            {
                "label": "unknown device",
                "visible": False,
                "visual_evidence": None,
                "screenplay_evidence": "The device opens.",
                "suggested_motion": "Open the device.",
                "confidence": 0.7,
            }
        ],
        conflicts=["The screenplay requests an action the image does not confirm."]
    )

    plan = scene_analysis_to_shot_plan(analysis, "The device opens.", "tense")
    candidate = plan.motion_plan.movable_objects[0]

    assert candidate.grounding_source == "screenplay"
    assert candidate.visual_confidence == 0
    assert candidate.support == "needs_confirmation"
    assert plan.motion_plan.confirmation_required is True
    assert plan.motion_plan.grounding_conflicts


def test_unsupported_motion_is_excluded_and_confidence_is_converted():
    analysis = _analysis(
        objects=[
            {
                "label": "unidentified shape",
                "visible": False,
                "visual_evidence": None,
                "screenplay_evidence": None,
                "suggested_motion": "Transform into an unrelated form.",
                "confidence": 0.2,
            }
        ]
    )

    plan = scene_analysis_to_shot_plan(analysis, "No such object is named.", "uncertain")
    candidate = plan.motion_plan.unsupported_actions[0]

    assert candidate.grounding_source == "unsupported"
    assert candidate.support == "unsupported"
    assert candidate.confidence == 0
    assert plan.motion_plan.confirmation_required is True


@pytest.mark.parametrize(
    "archetype",
    [
        "human",
        "animal",
        "robot",
        "vehicle",
        "animated household object",
        "natural phenomenon",
    ],
)
def test_declared_main_character_can_be_any_agentive_visible_entity(archetype):
    main_label = f"principal {archetype}"
    analysis = _analysis(
        characters=[
            {
                "entity_id": "supporting-entity",
                "label": "supporting entity",
                "visible": True,
                "visual_evidence": "A secondary visible entity remains in the composition.",
                "screenplay_evidence": "A secondary entity shares the scene.",
                "suggested_motion": "Remain within the existing composition.",
                "confidence": 0.72,
            },
            {
                "entity_id": "principal-entity",
                "label": main_label,
                "visible": True,
                "visual_evidence": "The storyboard visibly presents the principal entity.",
                "screenplay_evidence": "The scene follows the principal entity's action.",
                "suggested_motion": "Perform the scene's principal action.",
                "confidence": 0.94,
            },
        ],
        main_character_id="principal-entity",
        relationships=[
            {
                "source_id": "principal-entity",
                "relation": "functions as the scene subject",
                "target_id": "supporting-entity",
                "action": "Focuses the scene's attention on the supporting entity.",
            }
        ],
    )

    plan = scene_analysis_to_shot_plan(analysis, "The principal entity acts.", "focused")

    assert plan.focal_subject == main_label
    assert [item.label for item in plan.motion_plan.movable_characters] == [
        "supporting entity",
        main_label,
    ]
    assert plan.relationships[0].source_id == "principal-entity"
    assert plan.relationships[0].target_id == "supporting-entity"


def test_uncertain_role_is_preserved_and_cross_category_duplicate_is_not_emitted_twice():
    analysis = _analysis(
        characters=[
            {
                "entity_id": "ambiguous-entity",
                "label": "ambiguous visible entity",
                "visible": True,
                "visual_evidence": "A visible entity is present but its narrative role is unclear.",
                "screenplay_evidence": "The entity reacts to the scene.",
                "suggested_motion": "React within the existing frame.",
                "confidence": 0.58,
                "role_status": "confirmation_required",
            }
        ],
        objects=[
            {
                "entity_id": "ambiguous-entity",
                "label": "ambiguous visible entity",
                "visible": True,
                "visual_evidence": "The same identified entity is visible.",
                "screenplay_evidence": None,
                "suggested_motion": "Move within the existing frame.",
                "confidence": 0.58,
            }
        ],
        main_character_id="ambiguous-entity",
    )

    plan = scene_analysis_to_shot_plan(analysis, "The entity reacts to the scene.", "uncertain")

    assert len(plan.motion_plan.movable_characters) == 1
    assert not plan.motion_plan.movable_objects
    assert plan.motion_plan.movable_characters[0].support == "needs_confirmation"
    assert plan.focal_subject == "ambiguous visible entity"
    assert plan.motion_plan.confirmation_required is True
    assert any("more than one semantic category" in item for item in plan.motion_plan.grounding_conflicts)


def test_unresolved_main_character_id_does_not_force_a_focal_subject():
    analysis = _analysis(
        characters=[
            {
                "entity_id": "identified-entity",
                "label": "identified entity",
                "visible": True,
                "visual_evidence": "A visible entity is present.",
                "screenplay_evidence": None,
                "suggested_motion": None,
                "confidence": 0.7,
            }
        ],
        main_character_id="missing-entity",
    )

    plan = scene_analysis_to_shot_plan(analysis, "The scene holds.", "ambiguous")

    assert plan.focal_subject == "No confidently identified focal subject."
    assert any("did not resolve" in item for item in plan.motion_plan.grounding_conflicts)