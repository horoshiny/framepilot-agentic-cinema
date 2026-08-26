import json
import os

from .demo import demo_critique, demo_plan
from .schemas import Critique, ShotPlan


def create_shot_plan(screenplay: str, mood: str = "cinematic") -> dict:
    """Create a structured, editable cinematic shot plan from a screenplay excerpt."""
    # Deterministic fallback keeps the product demonstrable while credits are pending.
    return demo_plan(screenplay, mood).model_dump()


def critique_shot(plan_json: str) -> dict:
    """Evaluate a shot plan for focus, pacing, motion, and restraint; return a revision."""
    plan = ShotPlan.model_validate(json.loads(plan_json))
    return demo_critique(plan).model_dump()


def runtime_mode() -> str:
    configured = bool(os.getenv("GOOGLE_CLOUD_PROJECT"))
    demo_forced = os.getenv("DEMO_MODE", "true").lower() == "true"
    return "demo" if demo_forced or not configured else "vertex"

