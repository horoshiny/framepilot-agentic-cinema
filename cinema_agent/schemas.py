import re
from typing import Annotated, Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


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


CameraGrammar = Literal["awe", "dread", "urgency", "intimacy", "mystery", "balanced"]


class ShotSignature(BaseModel):
    motion_type: Literal["push_in", "pull_out", "pan_left", "pan_right", "drift"]
    pan_direction: Literal["left", "right", "hold"]
    zoom_direction: Literal["in", "out", "hold"]
    duration_seconds: float = Field(ge=3, le=20)
    parallax_strength: float = Field(ge=0, le=1)
    atmosphere: list[Literal["fog", "dust", "rain", "embers", "light_flicker"]]


class DepthPoint(BaseModel):
    x: int = Field(ge=0, le=1000)
    y: int = Field(ge=0, le=1000)


class DepthRegion(BaseModel):
    polygon: list[DepthPoint] = Field(min_length=3, max_length=64)
    rationale: str = Field(min_length=1, max_length=500)


class DepthLayout(BaseModel):
    foreground_occluders: DepthRegion
    primary_subject_midground: DepthRegion
    background: DepthRegion


class MotionBounds(BaseModel):
    x_min: int = Field(ge=0, le=1000)
    y_min: int = Field(ge=0, le=1000)
    x_max: int = Field(ge=0, le=1000)
    y_max: int = Field(ge=0, le=1000)


class MotionCandidate(BaseModel):
    label: str = Field(min_length=1, max_length=120)
    action: str = Field(min_length=1, max_length=300)
    bound: str = Field(min_length=1, max_length=300)
    visual_evidence: str = Field(min_length=1, max_length=500)
    screenplay_evidence: str = Field(min_length=1, max_length=500)
    intent_evidence: str = Field(
        default="No separate creative-intent evidence was identified.",
        min_length=1,
        max_length=500,
    )
    confidence: float = Field(ge=0, le=1)
    visual_confidence: float = Field(default=0.0, ge=0, le=1)
    grounding_source: Literal[
        "visual",
        "screenplay",
        "visual_and_screenplay",
        "unsupported",
    ] = "screenplay"
    support: Literal["supported", "needs_confirmation", "unsupported"]
    bounds: MotionBounds | None = None

    @field_validator("label")
    @classmethod
    def validate_entity_label(cls, value: str) -> str:
        label = " ".join(value.strip().split()).strip(" .,;:!?")
        if not label:
            raise ValueError("motion labels must contain a complete entity noun phrase")
        words = re.findall(r"[A-Za-z][A-Za-z0-9'’-]*", label)
        normalized = [word.lower() for word in words]
        action_fragments = {
            "bend", "bending", "close", "closing", "drift", "drifting",
            "fall", "falling", "flicker", "flickering", "flow", "flowing",
            "glow", "glowing", "lift", "lifting", "move", "moving",
            "open", "opening", "pulse", "pulsing", "raise", "raising",
            "reach", "reaching", "rotate", "rotating", "run", "running",
            "swing", "swinging", "turn", "turning", "walk", "walking",
            "walks", "wave", "waving", "waves", "open", "opens",
            "opening", "rotate", "rotates",
        }
        function_words = {
            "a", "an", "and", "as", "at", "by", "for", "from", "in",
            "into", "of", "on", "or", "the", "their", "to", "with",
        }
        if not words or all(word in function_words for word in normalized):
            raise ValueError("motion labels must contain a complete entity noun phrase")
        if len(words) == 1 and (
            normalized[0] in action_fragments
            or normalized[0].endswith("ly")
            or normalized[0] in {"brief", "bright", "dark", "gentle", "quiet", "slow"}
        ):
            raise ValueError("motion labels cannot be only an action, adjective, or adverb")
        if normalized in (["screenplay", "named", "subject"], ["screenplay", "named", "object"]):
            raise ValueError("motion labels must identify an entity")
        if any(word in action_fragments for word in normalized):
            raise ValueError("motion labels cannot contain action fragments")
        return label

    @model_validator(mode="after")
    def validate_grounding_confidence(self):
        if self.grounding_source == "unsupported":
            if self.support != "unsupported":
                raise ValueError("unsupported grounding must use unsupported support")
        elif self.confidence <= 0:
            raise ValueError("grounded motion must have non-zero evidence confidence")
        if self.grounding_source in {"visual", "visual_and_screenplay"} and self.visual_confidence <= 0:
            raise ValueError("visual grounding must include non-zero visual confidence")
        if self.grounding_source == "screenplay" and self.visual_confidence > 0:
            raise ValueError("screenplay-only grounding cannot claim visual confidence")
        return self


class MotionPlan(BaseModel):
    grounding_summary: str = Field(min_length=1, max_length=800)
    movable_characters: list[MotionCandidate] = Field(default_factory=list)
    movable_objects: list[MotionCandidate] = Field(default_factory=list)
    environmental_motion: list[MotionCandidate] = Field(default_factory=list)
    visible_characters: list["SceneEntity"] = Field(default_factory=list, max_length=12)
    visible_objects: list["SceneEntity"] = Field(default_factory=list, max_length=16)
    visible_environment: list["SceneEntity"] = Field(default_factory=list, max_length=12)
    camera_movement: MotionCandidate
    preserved_elements: list[str] = Field(min_length=1, max_length=12)
    prohibited_changes: list[str] = Field(default_factory=list, max_length=16)
    unsupported_actions: list[MotionCandidate] = Field(default_factory=list, max_length=12)
    grounding_conflicts: list[str] = Field(default_factory=list, max_length=8)
    confirmation_required: bool = False
    confirmation_question: str | None = Field(default=None, max_length=500)


class SceneEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str | None = Field(default=None, min_length=1, max_length=64)
    label: str = Field(min_length=1, max_length=80)
    entity_type: Literal["character", "object", "environment"] | None = None
    agentive: bool = False
    semantic_category: str | None = Field(default=None, max_length=64)
    action: str | None = Field(
        default=None,
        max_length=240,
        validation_alias=AliasChoices("action", "suggested_motion"),
    )
    visual_evidence: str | None = Field(default=None, max_length=240)
    screenplay_evidence: str | None = Field(default=None, max_length=240)
    grounding_source: Literal[
        "visual",
        "screenplay",
        "visual_and_screenplay",
        "unsupported",
    ] | None = None
    confidence: float = Field(ge=0, le=1)
    visual_confidence: float = Field(default=0.0, ge=0, le=1)
    support: Literal["supported", "needs_confirmation", "unsupported"] = "supported"
    legacy_visible: bool | None = Field(
        default=None,
        exclude=True,
        validation_alias="visible",
    )
    legacy_role_status: Literal[
        "certain",
        "uncertain",
        "confirmation_required",
    ] | None = Field(
        default=None,
        exclude=True,
        validation_alias="role_status",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_support(cls, value):
        if not isinstance(value, dict) or "support" in value or "role_status" not in value:
            return value
        normalized = dict(value)
        normalized["support"] = {
            "certain": "supported",
            "uncertain": "needs_confirmation",
            "confirmation_required": "needs_confirmation",
        }.get(value["role_status"], value["role_status"])
        return normalized

    @model_validator(mode="after")
    def validate_compact_grounding(self):
        if self.grounding_source == "unsupported" or self.support == "unsupported":
            if self.support != "unsupported":
                raise ValueError("unsupported grounding must use unsupported support")
            return self
        if self.grounding_source in {"visual", "visual_and_screenplay"}:
            if not self.visual_evidence or self.visual_confidence <= 0:
                raise ValueError("visual grounding requires visual evidence and confidence")
        if self.grounding_source in {"screenplay", "visual_and_screenplay"}:
            if not self.screenplay_evidence:
                raise ValueError("screenplay grounding requires screenplay evidence")
        if self.grounding_source == "screenplay" and self.visual_confidence > 0:
            raise ValueError("screenplay-only grounding cannot claim visual confidence")
        return self


class SceneRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1, max_length=64)
    relation: str = Field(min_length=1, max_length=96)
    target_id: str = Field(min_length=1, max_length=64)
    action: str | None = Field(default=None, max_length=240)


class ShotPlan(BaseModel):
    scene_summary: str
    emotional_intent: str
    focal_subject: str
    depth_notes: dict[str, str]
    shot: ShotParameters
    directing_rationale: str
    motion_plan: MotionPlan
    depth_layout: DepthLayout | None = None
    relationships: list[SceneRelationship] = Field(default_factory=list, max_length=48)


class CameraAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    movement: str = Field(min_length=1, max_length=80)
    visual_evidence: str | None = Field(default=None, max_length=240)
    screenplay_evidence: str | None = Field(default=None, max_length=240)
    confidence: float = Field(ge=0, le=1)


class SceneAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_summary: str = Field(min_length=1, max_length=400)
    mood: str = Field(default="cinematic", max_length=500)
    characters: list[SceneEntity] = Field(default_factory=list, max_length=12)
    objects: list[SceneEntity] = Field(default_factory=list, max_length=16)
    environment: list[SceneEntity] = Field(default_factory=list, max_length=12)
    camera: CameraAnalysis
    preserve: list[Annotated[str, Field(min_length=1, max_length=240)]] = Field(
        default_factory=list,
        max_length=12,
    )
    prohibit: list[Annotated[str, Field(min_length=1, max_length=240)]] = Field(
        default_factory=list,
        max_length=16,
    )
    conflicts: list[Annotated[str, Field(min_length=1, max_length=240)]] = Field(
        default_factory=list,
        max_length=8,
    )
    main_character_id: str | None = Field(default=None, min_length=1, max_length=64)
    main_character_ids: list[Annotated[str, Field(min_length=1, max_length=64)]] = Field(
        default_factory=list,
        max_length=12,
    )
    relationships: list[SceneRelationship] = Field(default_factory=list, max_length=24)


class RoutingDecision(BaseModel):
    classification: Literal[
        "LOCAL_2_5D",
        "GENERATIVE_VIDEO_REQUIRED",
        "UNSUPPORTED",
    ]
    rationale: str
    matched_actions: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    focus_score: int = Field(ge=1, le=10)
    pacing_score: int = Field(ge=1, le=10)
    cinematic_motion_score: int = Field(ge=1, le=10)
    restraint_score: int = Field(ge=1, le=10)
    diagnosis: str
    revision: ShotParameters
    revision_rationale: str


class VideoEvidence(BaseModel):
    timestamp_seconds: float | None = Field(default=None, ge=0, le=3600)
    description: str = Field(min_length=1, max_length=500)
    certainty: Literal["high", "medium", "low", "uncertain"]


class VideoObservation(BaseModel):
    finding: str = Field(min_length=1, max_length=600)
    support: Literal["supported", "uncertain", "not_observed"]
    evidence: list[VideoEvidence] = Field(default_factory=list, max_length=8)
    uncertainty: str = Field(min_length=1, max_length=500)


class VideoRevisionRecommendation(BaseModel):
    issue: str = Field(min_length=1, max_length=500)
    recommendation: str = Field(min_length=1, max_length=600)
    support: Literal["supported", "uncertain", "not_observed"]
    evidence: list[VideoEvidence] = Field(default_factory=list, max_length=8)
    uncertainty: str = Field(min_length=1, max_length=500)


class VideoCritiquePayload(BaseModel):
    requested_actions_achieved: list[VideoObservation] = Field(default_factory=list, max_length=12)
    requested_actions_missing: list[VideoObservation] = Field(default_factory=list, max_length=12)
    character_object_consistency: list[VideoObservation] = Field(default_factory=list, max_length=12)
    camera_movement_pacing: list[VideoObservation] = Field(default_factory=list, max_length=12)
    visible_artifacts: list[VideoObservation] = Field(default_factory=list, max_length=12)
    emotional_intent_alignment: list[VideoObservation] = Field(default_factory=list, max_length=12)
    revision_recommendations: list[VideoRevisionRecommendation] = Field(
        default_factory=list, max_length=12
    )
    uncertainty: str = Field(min_length=1, max_length=800)


class VideoCritique(VideoCritiquePayload):
    status: Literal["available", "unavailable"]
    source: Literal["gemini_video", "demo_fixture", "unavailable"]
    label: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=300)
    notice: str | None = Field(default=None, max_length=200)
    content_hash: str | None = None
    source_plan_hash: str | None = None
    model: str | None = None


class DirectRequest(BaseModel):
    screenplay: str = Field(min_length=20, max_length=8000)
    mood: str = Field(default="cinematic", max_length=500)
    image_data_url: str | None = None


class DirectResponse(BaseModel):
    mode: Literal["demo", "vertex", "rate_limited_fallback"]
    analysis_source: Literal["vertex_multimodal", "deterministic_fallback"] = (
        "deterministic_fallback"
    )
    scene_key: str | None = None
    source_signature: str | None = None
    image_handle: str | None = None
    depth_source: Literal["image_aware", "heuristic"] = "heuristic"
    plan: ShotPlan
    critique: Critique
    routing: RoutingDecision
    activity: list[dict[str, str]]
    camera_grammar: CameraGrammar = "balanced"
    shot_signature: ShotSignature | None = None
    diversity_adjustment: str | None = None
    revision_camera_grammar: CameraGrammar | None = None
    revision_shot_signature: ShotSignature | None = None
    revision_diversity_adjustment: str | None = None


VideoJobKind = Literal["first_cut", "director_cut"]
VideoJobStatus = Literal[
    "not_generated",
    "approval_required",
    "queued",
    "generating",
    "submission_unknown",
    "completed",
    "failed",
]


class VideoApprovalRequest(BaseModel):
    kind: VideoJobKind
    scene_key: str = Field(min_length=8, max_length=512)
    source_signature: str = Field(min_length=8, max_length=1024)
    screenplay: str = Field(min_length=20, max_length=8000)
    creative_intent: str = Field(min_length=1, max_length=500)
    shot_plan: ShotPlan
    first_cut_job_id: str | None = None
    replacement_for_job_id: str | None = None
    image_handle: str | None = None
    video_critique: VideoCritique | None = None
    analysis_source: Literal[
        "vertex_multimodal",
        "deterministic_fallback",
        "durable_recovery",
    ] | None = None
    direction_response: dict[str, Any] | None = None
    controlled_authorization_id: str | None = Field(default=None, min_length=8, max_length=128)


class VideoApproval(BaseModel):
    kind: VideoJobKind
    scene_key: str
    source_signature: str
    first_cut_job_id: str | None = None
    replacement_for_job_id: str | None = None
    status: Literal["approval_required"] = "approval_required"
    approval_id: str
    expires_at: float
    model: str | None = None
    controlled_authorization_id: str | None = None


class VideoJobRequest(VideoApprovalRequest):
    approved: bool = False
    approval_id: str | None = None


class VideoOutput(BaseModel):
    url: str
    label: Literal["Demo generation", "Vertex AI Veo"]
    source: Literal["bundled_fixture", "vertex_veo"]
    kind: VideoJobKind


class VideoFailure(BaseModel):
    code: Literal[
        "approval_required",
        "generation_disabled",
        "generation_limit_reached",
        "active_job_exists",
        "duplicate_scene_kind",
        "provider_submission_failed",
        "submission_unknown",
        "provider_failed",
        "generation_timeout",
        "video_retrieval_failed",
        "invalid_image",
        "storyboard_image_missing",
    ]
    message: str
    provider_error_code: int | None = None
    provider_error_status: str | None = None
    provider_error_category: str | None = None
    provider_error_message: str | None = None


class VideoAllowanceStatus(BaseModel):
    global_limit: int
    global_used: int
    global_remaining: int
    per_ip_limit: int
    per_ip_used: int
    per_ip_remaining: int
    director_cut_global_limit: int
    director_cut_global_used: int
    director_cut_global_remaining: int
    authorized_replacement_limit: int
    authorized_replacement_used: int
    authorized_replacement_remaining: int
    authorized_replacement_for_job_id: str | None = None


class ControlledAuthorizationRequest(BaseModel):
    scene_key: str = Field(min_length=8, max_length=512)
    source_signature: str = Field(min_length=8, max_length=1024)
    model: str = Field(min_length=1, max_length=200)
    kind: Literal["first_cut"] = "first_cut"
    image_handle: str | None = None
    analysis_source: Literal["vertex_multimodal", "deterministic_fallback"]


class ControlledAuthorizationStatus(BaseModel):
    authorization_id: str | None = None
    available: bool = False
    kind: Literal["first_cut"] = "first_cut"
    scene_key: str | None = None
    source_signature: str | None = None
    model: str | None = None
    source: Literal["authorized_test_attempt"] = "authorized_test_attempt"
    duration_seconds: int = 8
    aspect_ratio: Literal["16:9"] = "16:9"
    audio_enabled: bool = False
    estimate: str | None = None


class VideoSceneSnapshot(BaseModel):
    scene_key: str
    source_signature: str
    screenplay: str | None = None
    creative_intent: str | None = None
    direction_response: dict[str, Any] | None = None
    shot_plan: dict[str, Any] | None = None
    storyboard_url: str | None = None
    analysis_source: Literal[
        "vertex_multimodal",
        "deterministic_fallback",
        "durable_recovery",
    ] | None = None
    storyboard_image_handle: str | None = None


class VideoJob(BaseModel):
    job_id: str
    kind: VideoJobKind
    status: VideoJobStatus
    scene_key: str
    source_signature: str
    provider: Literal["mock", "vertex"]
    created_at: float
    updated_at: float
    model: str | None = None
    replacement_for_job_id: str | None = None
    output: VideoOutput | None = None
    error: VideoFailure | None = None
    video_critique: VideoCritique | None = None
    revision_approved: bool = False
    deduplicated: bool = False
    scene_snapshot: VideoSceneSnapshot | None = None

