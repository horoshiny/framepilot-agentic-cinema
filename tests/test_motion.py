import base64
import io

from PIL import Image

from cinema_agent.demo import demo_plan
from cinema_agent.motion import demo_direction


def image_data_url(size, color):
    image = Image.new("RGB", size, color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_fallback_motion_plan_is_dynamic_to_uploaded_frame_profile():
    landscape = image_data_url((960, 540), (210, 210, 210))
    portrait = image_data_url((540, 960), (25, 25, 25))

    landscape_shot, landscape_plan = demo_direction(
        "The camera drifts across the still scene.", "quiet observation", landscape
    )
    portrait_shot, portrait_plan = demo_direction(
        "The camera drifts across the still scene.", "quiet observation", portrait
    )

    assert landscape_shot != portrait_shot
    assert "960×540" in landscape_plan.camera_movement.visual_evidence
    assert "540×960" in portrait_plan.camera_movement.visual_evidence
    assert "quiet observation" in landscape_plan.grounding_summary
    assert landscape_plan.movable_characters == []
    assert portrait_plan.movable_objects == []
    assert landscape_plan.preserved_elements
    assert landscape_plan.prohibited_changes


def test_fallback_environmental_motion_requires_confirmation_without_semantic_image_analysis():
    image = image_data_url((640, 360), (120, 120, 120))

    shot, plan = demo_direction(
        "The camera pushes in while rain crosses the frame.", "quiet focus", image
    )

    assert shot.atmosphere == []
    assert len(plan.environmental_motion) == 1
    assert plan.environmental_motion[0].label == "rain"
    assert plan.environmental_motion[0].support == "needs_confirmation"
    assert plan.confirmation_required is True
    assert plan.confirmation_question


def test_semantic_request_is_preserved_as_prohibited_not_invented_in_fallback():
    plan = demo_plan(
        "A person walks toward the bright opening and opens the gate.",
        "measured tension",
    )

    assert len(plan.motion_plan.movable_characters) == 1
    assert plan.motion_plan.movable_characters[0].grounding_source == "screenplay"
    assert plan.motion_plan.movable_characters[0].support == "needs_confirmation"
    assert len(plan.motion_plan.movable_objects) == 1
    assert plan.motion_plan.movable_objects[0].grounding_source == "screenplay"
    assert plan.motion_plan.movable_objects[0].support == "needs_confirmation"
    assert plan.motion_plan.confirmation_required is True
    assert any("character movement" in item for item in plan.motion_plan.prohibited_changes)
    assert any("door or gate movement" in item for item in plan.motion_plan.prohibited_changes)


def test_dark_storyboard_keeps_explicit_screenplay_motion_confirmation_gated():
    dark_storyboard = image_data_url((1280, 720), (5, 7, 12))
    screenplay = (
        "The astronomer slowly raises the lantern from waist to shoulder height; feet remain planted. "
        "The cloak hem moves gently in the breeze. Portal rings rotate slowly. "
        "One controlled portal pulse only in the final third. Mist drifts through the ruins. "
        "Subtle ripples move across the floodwater and portal reflection. "
        "Camera: slow slide right with a restrained push-in. "
        "Preserve the astronomer's identity and silhouette, lantern, architecture, portal structure, "
        "composition and blue-hour lighting. No walking, new subjects, cuts, abrupt motion, "
        "object replacement or major deformation."
    )

    plan = demo_plan(screenplay, "quiet dread", dark_storyboard)
    candidates = [
        *plan.motion_plan.movable_characters,
        *plan.motion_plan.movable_objects,
        *plan.motion_plan.environmental_motion,
    ]

    assert {"astronomer", "cloak hem", "portal rings", "portal"} <= {
        candidate.label for candidate in candidates
    }
    assert any("mist" in candidate.label for candidate in candidates)
    assert any("ripples" in candidate.label for candidate in candidates)
    assert all(candidate.grounding_source == "screenplay" for candidate in candidates)
    assert all(candidate.support == "needs_confirmation" for candidate in candidates)
    assert all(candidate.confidence > 0 for candidate in candidates)
    assert all(candidate.visual_confidence == 0 for candidate in candidates)
    assert plan.motion_plan.confirmation_required is True
    assert "screenplay-grounded" in plan.motion_plan.confirmation_question
    assert any(
        "astronomer's identity and silhouette" in item.lower()
        for item in plan.motion_plan.preserved_elements
    )
    assert any("blue-hour lighting" in item.lower() for item in plan.motion_plan.preserved_elements)
    assert "GROUNDING SOURCES: visual=0" in plan.motion_plan.grounding_summary
    assert "unsupported=0" in plan.motion_plan.grounding_summary


def test_demo_plan_does_not_claim_scene_specific_subjects():
    plan = demo_plan("The camera observes a quiet room.", "neutral")

    assert plan.focal_subject == "the visual center of the supplied frame"
    assert "lantern" not in plan.model_dump_json().lower()
    assert "observatory" not in plan.model_dump_json().lower()