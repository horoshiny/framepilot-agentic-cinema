from cinema_agent.camera_grammar import (
    ShotDiversityGuard,
    compile_shot,
    select_camera_grammar,
    signature_text,
    signatures_nearly_identical,
)
from cinema_agent.demo import demo_plan


BASE_SHOT = demo_plan("The camera observes a quiet room.", "cinematic").shot


def test_emotion_grammars_compile_to_distinct_camera_signatures():
    awe = compile_shot(
        BASE_SHOT,
        "A vast majestic reveal opens over the valley.",
        "awe",
    )
    dread = compile_shot(
        BASE_SHOT,
        "An ominous threat waits in the haunted room.",
        "dread",
    )

    assert select_camera_grammar("A vast majestic reveal opens over the valley.", "awe") == "awe"
    assert awe.grammar == "awe"
    assert awe.shot.camera_motion in {"pan_left", "pan_right"}
    assert awe.shot.duration_seconds >= 10
    assert awe.shot.parallax_strength >= 0.8
    assert dread.grammar == "dread"
    assert dread.shot.camera_motion == "pull_out"
    assert dread.signature.zoom_direction == "out"
    assert dread.shot.atmosphere == ["light_flicker"]
    assert not signatures_nearly_identical(awe.signature, dread.signature)
    assert "pan" in signature_text(awe.signature)


def test_remaining_grammars_preserve_visibly_different_treatments():
    urgency = compile_shot(BASE_SHOT, "Hurry, the chase has begun.", "urgency")
    intimacy = compile_shot(BASE_SHOT, "A tender intimate confession.", "intimacy")
    mystery = compile_shot(BASE_SHOT, "A strange secret clue remains unknown.", "mystery")

    assert urgency.shot.duration_seconds <= 5
    assert urgency.shot.motion_intensity >= 0.9
    assert urgency.signature.zoom_direction == "in"
    assert intimacy.shot.camera_motion == "push_in"
    assert intimacy.shot.parallax_strength <= 0.2
    assert intimacy.shot.atmosphere == []
    assert mystery.shot.camera_motion == "drift"
    assert mystery.shot.pan_y < 0
    assert mystery.shot.parallax_strength == 0.56
    assert len({urgency.signature.motion_type, intimacy.signature.motion_type, mystery.signature.motion_type}) == 3


def test_guard_explains_and_applies_deterministic_adjustment_for_near_duplicate_intent():
    guard = ShotDiversityGuard()
    first = guard.compile(BASE_SHOT, "A strange secret clue is hidden.", "mystery", "client")
    second = guard.compile(
        BASE_SHOT,
        "A strange secret clue is hidden in another room.",
        "mystery",
        "client",
    )

    assert first.signature
    if signatures_nearly_identical(first.signature, second.signature):
        assert second.adjustment
        assert not signatures_nearly_identical(first.signature, second.signature)
    else:
        assert second.adjustment is None


def test_critic_revision_can_be_forced_to_a_different_compiled_signature():
    guard = ShotDiversityGuard()
    first = guard.compile(BASE_SHOT, "A quiet room waits.", "balanced", "revision-client")
    revision = guard.compile(
        BASE_SHOT,
        "A quiet room waits.",
        "balanced",
        "revision-client",
        force_distinct=True,
    )

    assert revision.adjustment
    assert not signatures_nearly_identical(first.signature, revision.signature)