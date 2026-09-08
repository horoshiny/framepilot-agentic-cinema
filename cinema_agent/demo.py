from .motion import demo_direction
from .schemas import Critique, ShotPlan
from .depth import heuristic_depth_layout


def demo_plan(screenplay: str, mood: str, image_data_url: str | None = None) -> ShotPlan:
    shot, motion_plan = demo_direction(screenplay, mood, image_data_url)
    frame_label = "uploaded frame" if image_data_url else "bundled frame"

    return ShotPlan(
        scene_summary=f"A conservative camera study of the {frame_label}, grounded in the supplied text and intent.",
        emotional_intent=mood or "rising dread",
        focal_subject="the visual center of the supplied frame",
        depth_notes={
            "foreground": "Only image-space regions identified as nearest are given foreground travel.",
            "subject": "No semantic subject is asserted without confident image evidence.",
            "background": "The far image plane remains visually consistent while the camera moves.",
        },
        shot=shot,
        motion_plan=motion_plan,
        directing_rationale=(
            "Use the smallest supported camera move that serves the supplied intent while preserving "
            "the frame's visible identity and geometry."
        ),
        depth_layout=heuristic_depth_layout(),
    )


def demo_critique(plan: ShotPlan) -> Critique:
    revised = plan.shot.model_copy(
        update={
            "duration_seconds": min(20, plan.shot.duration_seconds + 1.5),
            "zoom_end": max(1.0, plan.shot.zoom_end - 0.03),
            "parallax_strength": max(0, plan.shot.parallax_strength - 0.08),
            "motion_intensity": max(0, plan.shot.motion_intensity - 0.08),
            "atmosphere": plan.shot.atmosphere[:1],
        }
    )
    return Critique(
        focus_score=8,
        pacing_score=6,
        cinematic_motion_score=7,
        restraint_score=6,
        diagnosis=(
            "The Motion Preview is grounded, but the motion budget can be softened to keep the visible "
            "frame consistent and the requested beat legible."
        ),
        revision=revised,
        revision_rationale=(
            "Lengthen the shot, reduce camera travel and retain at most one explicitly supported "
            "environmental cue."
        ),
    )

