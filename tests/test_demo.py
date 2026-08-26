from cinema_agent.demo import demo_critique, demo_plan


def test_demo_plan_is_valid_and_revisable():
    plan = demo_plan("Mara raises a lantern and approaches the door.", "suspense")
    critique = demo_critique(plan)
    assert plan.shot.duration_seconds >= 3
    assert critique.revision.duration_seconds > plan.shot.duration_seconds
    assert critique.revision.motion_intensity < plan.shot.motion_intensity

