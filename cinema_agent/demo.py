from .schemas import Critique, ShotParameters, ShotPlan


def demo_plan(screenplay: str, mood: str) -> ShotPlan:
    subject = "the figure at the illuminated doorway"
    lower = screenplay.lower()
    if "lantern" in lower:
        subject = "the lantern and the character's face"
    elif "door" in lower:
        subject = "the hand approaching the door handle"

    return ShotPlan(
        scene_summary="A solitary character approaches a threshold as the environment quietly signals danger.",
        emotional_intent=mood or "rising dread",
        focal_subject=subject,
        depth_notes={
            "foreground": "dark branches and drifting particles",
            "subject": "character and practical light source",
            "background": "distant architecture softened by haze",
        },
        shot=ShotParameters(
            duration_seconds=8,
            camera_motion="push_in",
            zoom_start=1.0,
            zoom_end=1.16,
            pan_x=-4,
            pan_y=1,
            parallax_strength=0.46,
            motion_intensity=0.38,
            atmosphere=["fog", "dust", "light_flicker"],
            transition="shadow_wipe",
        ),
        directing_rationale=(
            "A restrained push-in gradually removes visual escape routes while the practical light "
            "keeps attention on the character's decision."
        ),
    )


def demo_critique(plan: ShotPlan) -> Critique:
    revised = plan.shot.model_copy(
        update={
            "duration_seconds": 9.5,
            "zoom_end": 1.12,
            "parallax_strength": 0.34,
            "motion_intensity": 0.27,
            "atmosphere": ["fog", "light_flicker"],
        }
    )
    return Critique(
        focus_score=8,
        pacing_score=6,
        cinematic_motion_score=7,
        restraint_score=6,
        diagnosis=(
            "The direction correctly isolates the subject, but the first pass moves too quickly and "
            "layers too many effects for a suspense beat."
        ),
        revision=revised,
        revision_rationale=(
            "Lengthen the shot, soften the push-in and remove dust so the lantern flicker becomes the "
            "single secondary motion cue."
        ),
    )

