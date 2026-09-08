import base64
import io
from types import SimpleNamespace
from pathlib import Path

from PIL import Image

from cinema_agent.demo import demo_plan
from cinema_agent.schemas import MotionCandidate, SceneAnalysis
from cinema_agent.video_provider import build_veo_prompt


def image_data_url(size, color):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_dark_storyboard_preserves_generic_screenplay_entities_and_actions():
    screenplay = (
        "A courier slowly raises a signal flag from waist to shoulder height; feet remain planted. "
        "The flag rotates once in the final third. Fog drifts across the bridge. "
        "Camera pans left with a restrained push-in. Preserve the courier's identity and silhouette."
    )

    plan = demo_plan(screenplay, "measured resolve", image_data_url((960, 540), (5, 7, 12)))
    candidates = [
        *plan.motion_plan.movable_characters,
        *plan.motion_plan.movable_objects,
        *plan.motion_plan.environmental_motion,
    ]

    assert any(candidate.label == "courier" for candidate in candidates)
    assert any(candidate.label == "flag" for candidate in candidates)
    assert any(candidate.label == "fog" for candidate in candidates)
    assert all(candidate.grounding_source == "screenplay" for candidate in candidates)
    assert all(candidate.visual_evidence for candidate in candidates)
    assert all(candidate.screenplay_evidence for candidate in candidates)
    assert all("measured resolve" in candidate.intent_evidence for candidate in candidates)
    assert plan.motion_plan.confirmation_required is True


def test_mocked_bright_storyboard_can_mark_visual_and_screenplay_grounding():
    import cinema_agent.vertex as vertex

    screenplay = "A violinist lifts the bow as the orchestra holds its breath."
    response_analysis = SceneAnalysis(
        scene_summary="A violinist and bow are visible.",
        characters=[
            {
                "label": "violinist",
                "visible": True,
                "visual_evidence": "Bright storyboard visibly shows the violinist and bow in frame.",
                "screenplay_evidence": screenplay,
                "suggested_motion": "Lift the bow.",
                "confidence": 0.94,
            }
        ],
        objects=[],
        environment=[],
        camera={
            "movement": "drift",
            "visual_evidence": "The frame supports a restrained drift.",
            "screenplay_evidence": None,
            "confidence": 0.5,
        },
        preserve=[],
        prohibit=[],
        conflicts=[],
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=response_analysis.model_dump_json())

    class FakeClient:
        models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch.setattr(vertex, "_client", lambda: FakeClient())
        result = vertex.generate_shot_plan(
            screenplay,
            "precise anticipation",
            image_data_url((960, 540), (220, 220, 220)),
        )
    finally:
        monkeypatch.undo()

    grounded = result.motion_plan.movable_characters[0]
    assert grounded.grounding_source == "visual_and_screenplay"
    assert grounded.confidence == 0.94
    assert grounded.visual_evidence.startswith("Bright storyboard")
    assert grounded.screenplay_evidence == screenplay
    assert "precise anticipation" in grounded.intent_evidence


def test_mocked_conflicting_evidence_requires_confirmation():
    import cinema_agent.vertex as vertex

    screenplay = "A dancer runs across the stage while the camera tracks right."
    conflict = "The storyboard shows the dancer seated; the screenplay requests a run."
    response_analysis = SceneAnalysis(
        scene_summary="A seated figure conflicts with the requested run.",
        characters=[
            {
                "label": "dancer",
                "visible": False,
                "visual_evidence": None,
                "screenplay_evidence": screenplay,
                "suggested_motion": "Run across the stage.",
                "confidence": 0.7,
            }
        ],
        objects=[],
        environment=[],
        camera={
            "movement": "pan right",
            "visual_evidence": None,
            "screenplay_evidence": "The camera tracks right.",
            "confidence": 0.8,
        },
        preserve=[],
        prohibit=[],
        conflicts=[conflict],
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            return SimpleNamespace(text=response_analysis.model_dump_json())

    class FakeClient:
        models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch = __import__("pytest").MonkeyPatch()
    try:
        monkeypatch.setattr(vertex, "_client", lambda: FakeClient())
        result = vertex.generate_shot_plan(screenplay, "urgent momentum", None)
    finally:
        monkeypatch.undo()

    assert result.motion_plan.grounding_conflicts == [conflict]
    assert result.motion_plan.confirmation_required is True
    assert "conflict" in result.motion_plan.confirmation_question.lower()


def test_unsupported_action_is_rejected_instead_of_invented():
    plan = demo_plan("A performer teleports across the stage.", "surreal tension")

    assert plan.motion_plan.unsupported_actions
    assert all(
        candidate.grounding_source == "unsupported"
        and candidate.support == "unsupported"
        for candidate in plan.motion_plan.unsupported_actions
    )
    assert plan.motion_plan.confirmation_required is True
    assert not plan.motion_plan.movable_characters
    assert not plan.motion_plan.movable_objects


def test_unrelated_scene_is_parsed_without_scene_specific_rules():
    screenplay = (
        "A violinist lifts a bow. The bow rotates slowly. Snow drifts across the courtyard. "
        "The camera pans left."
    )
    plan = demo_plan(screenplay, "quiet concentration")
    candidates = [
        *plan.motion_plan.movable_characters,
        *plan.motion_plan.movable_objects,
        *plan.motion_plan.environmental_motion,
    ]

    assert any(candidate.label == "violinist" for candidate in candidates)
    assert any(candidate.label == "bow" for candidate in candidates)
    assert any(candidate.label == "snow" for candidate in candidates)
    assert "violinist" in build_veo_prompt(screenplay, "quiet concentration", plan, "first_cut")
    assert "bow" in build_veo_prompt(screenplay, "quiet concentration", plan, "first_cut")
    assert "snow" in build_veo_prompt(screenplay, "quiet concentration", plan, "first_cut")


def test_pronoun_reference_reuses_the_most_recent_introduced_object():
    screenplay = (
        "A courier lifts a signal flag. It rotates slowly while the courier remains still. "
        "The camera holds on the bridge."
    )
    plan = demo_plan(screenplay, "measured resolve")

    object_candidates = plan.motion_plan.movable_objects
    assert any(candidate.label == "signal flag" for candidate in object_candidates)
    assert sum("rotates slowly" in candidate.action.lower() for candidate in object_candidates) == 1
    assert all(candidate.grounding_source == "screenplay" for candidate in object_candidates)


def test_fallback_classifies_forces_as_environment_and_keeps_static_setup_static():
    screenplay = (
        "Wind drifts through the alley. A courier walks toward the doorway. "
        "The door stands open beside a still table."
    )
    plan = demo_plan(screenplay, "measured tension")

    assert any(candidate.label == "wind" for candidate in plan.motion_plan.environmental_motion)
    assert any(candidate.label == "courier" for candidate in plan.motion_plan.movable_characters)
    assert not any(candidate.label == "door" for candidate in plan.motion_plan.movable_objects)
    assert not any(candidate.label == "table" for candidate in plan.motion_plan.movable_objects)
    assert all(
        candidate.screenplay_evidence in {
            "Wind drifts through the alley.",
            "A courier walks toward the doorway.",
        }
        for candidate in [
            *plan.motion_plan.movable_characters,
            *plan.motion_plan.movable_objects,
            *plan.motion_plan.environmental_motion,
        ]
    )


def test_camera_evidence_is_sentence_local_and_action_excludes_intent_text():
    screenplay = (
        "The camera pushes in through the doorway while rain crosses the frame. "
        "Preserve the composition and keep the mood intimate and restrained."
    )
    plan = demo_plan(screenplay, "intimate and restrained")
    camera = plan.motion_plan.camera_movement

    assert camera.screenplay_evidence == (
        "The camera pushes in through the doorway while rain crosses the frame."
    )
    assert "intimate and restrained" not in camera.action
    assert "pushes in through the doorway" in camera.action


def test_production_code_contains_no_fixture_specific_entities():
    terms = ("astronomer", "lantern", "portal", "cloak", "mist", "ruins", "floodwater")
    production = "\n".join(
        path.read_text()
        for path in Path("cinema_agent").glob("*.py")
    ).lower()

    assert not any(term in production for term in terms)