from typing import Literal

from pydantic import BaseModel, Field


class ShotParameters(BaseModel):
    duration_seconds: float = Field(ge=3, le=20)
    camera_motion: Literal["push_in", "pull_out", "pan_left", "pan_right", "drift"]
    zoom_start: float = Field(ge=1.0, le=1.3)
    zoom_end: float = Field(ge=1.0, le=1.35)
    pan_x: float = Field(ge=-20, le=20)
    pan_y: float = Field(ge=-15, le=15)
    parallax_strength: float = Field(ge=0, le=1)
    motion_intensity: float = Field(ge=0, le=1)
    atmosphere: list[Literal["fog", "dust", "rain", "embers", "light_flicker"]]
    transition: Literal["fade", "shadow_wipe", "light_bloom", "hard_cut"]


class ShotPlan(BaseModel):
    scene_summary: str
    emotional_intent: str
    focal_subject: str
    depth_notes: dict[str, str]
    shot: ShotParameters
    directing_rationale: str


class Critique(BaseModel):
    focus_score: int = Field(ge=1, le=10)
    pacing_score: int = Field(ge=1, le=10)
    cinematic_motion_score: int = Field(ge=1, le=10)
    restraint_score: int = Field(ge=1, le=10)
    diagnosis: str
    revision: ShotParameters
    revision_rationale: str


class DirectRequest(BaseModel):
    screenplay: str = Field(min_length=20, max_length=8000)
    mood: str = Field(default="cinematic", max_length=100)
    image_data_url: str | None = None


class DirectResponse(BaseModel):
    mode: Literal["demo", "vertex"]
    plan: ShotPlan
    critique: Critique
    activity: list[dict[str, str]]

