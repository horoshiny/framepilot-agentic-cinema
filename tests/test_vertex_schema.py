import base64
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from cinema_agent import vertex
from cinema_agent.schemas import SceneAnalysis, ShotPlan
from cinema_agent.scene_analysis import scene_analysis_to_shot_plan


def _analysis() -> SceneAnalysis:
    return SceneAnalysis(
        scene_summary="A courier stands on a bridge as a flag turns and fog crosses the frame.",
        characters=[
            {
                "entity_id": "courier-entity",
                "label": "courier",
                "visible": True,
                "visual_evidence": "A human figure is visibly centered on the bridge.",
                "screenplay_evidence": "A courier raises a signal flag.",
                "suggested_motion": "Raise the signal flag without changing the courier's silhouette.",
                "confidence": 0.91,
            }
        ],
        objects=[
            {
                "entity_id": "signal-flag-entity",
                "label": "signal flag",
                "visible": False,
                "visual_evidence": None,
                "screenplay_evidence": "The signal flag rotates once.",
                "suggested_motion": "Rotate the signal flag once.",
                "confidence": 0.72,
            },
            {
                "entity_id": "bridge-entity",
                "label": "bridge",
                "visible": True,
                "visual_evidence": "The bridge structure is visible beneath the figure.",
                "screenplay_evidence": None,
                "suggested_motion": None,
                "confidence": 0.86,
            },
        ],
        environment=[
            {
                "entity_id": "fog-entity",
                "label": "fog",
                "visible": True,
                "visual_evidence": "A translucent atmospheric layer crosses the background.",
                "screenplay_evidence": "Fog drifts across the bridge.",
                "suggested_motion": "Drift laterally across the existing frame.",
                "confidence": 0.82,
            }
        ],
        camera={
            "movement": "pan left",
            "visual_evidence": "The wide composition leaves lateral space for a restrained pan.",
            "screenplay_evidence": "Camera pans left.",
            "confidence": 0.88,
        },
        preserve=["Preserve the courier's identity and the bridge silhouette."],
        prohibit=["Do not add a second figure."],
        conflicts=[],
        main_character_id="courier-entity",
        relationships=[
            {
                "source_id": "courier-entity",
                "relation": "acts on",
                "target_id": "signal-flag-entity",
                "action": "raises the signal flag",
            }
        ],
    )


def _semantic_entity(
    entity_id: str,
    label: str,
    entity_type: str,
    *,
    agentive: bool,
    action: str | None = None,
) -> dict:
    return {
        "entity_id": entity_id,
        "label": label,
        "entity_type": entity_type,
        "agentive": agentive,
        "visual_evidence": f"The storyboard visibly contains {label}.",
        "screenplay_evidence": f"The screenplay references {label}.",
        "action": action,
        "grounding_source": "visual_and_screenplay",
        "confidence": 0.9,
        "visual_confidence": 0.9,
        "support": "supported",
    }


class _FakeModels:
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.response_text)


class _FakeClient:
    def __init__(self, models):
        self.models = models

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def test_compact_analysis_is_strict_and_complete():
    analysis = _analysis()
    assert isinstance(analysis, SceneAnalysis)
    entity_payload = analysis.model_dump()["characters"][0]
    assert set(entity_payload) == {
        "entity_id",
        "label",
        "entity_type",
        "agentive",
        "semantic_category",
        "action",
        "visual_evidence",
        "screenplay_evidence",
        "grounding_source",
        "confidence",
        "visual_confidence",
        "support",
    }
    assert set(analysis.model_dump()["relationships"][0]) == {
        "source_id",
        "relation",
        "target_id",
        "action",
    }
    assert "Creative intent" not in json.dumps(entity_payload)
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate({**analysis.model_dump(), "unexpected": True})


def test_scene_analysis_recovers_object_label_when_optional_metadata_is_malformed():
    payload = _analysis().model_dump()
    payload["characters"] = []
    payload["environment"] = []
    payload["objects"] = [
        {
            "name": "lantern",
            "semantic_category": "physical prop",
            "grounding_source": "visual",
            "confidence": 0.88,
            "visual_confidence": 0,
            "unexpected_provider_field": "ignored safely",
        }
    ]

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert [entity["label"] for entity in analysis.model_dump()["objects"]] == ["lantern"]
    recovered = analysis.model_dump()["objects"][0]
    assert recovered["grounding_source"] is None
    assert recovered["support"] == "needs_confirmation"


def test_valid_object_survives_another_malformed_object():
    payload = _analysis().model_dump()
    payload["characters"] = []
    payload["environment"] = []
    payload["objects"] = [
        {
            "entity_id": "invalid-object",
            "label": "",
            "confidence": 0.7,
        },
        {
            "entity_id": "valid-object",
            "label": "worn emblem",
            "visual_evidence": "A tangible emblem is attached to the visible subject.",
            "visual_confidence": 0.86,
            "grounding_source": "visual",
            "confidence": 0.91,
            "support": "supported",
        },
    ]

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert [entity["label"] for entity in analysis.model_dump()["objects"]] == [
        "worn emblem"
    ]
    assert any("objects.0.label:missing" in item for item in analysis.conflicts)


def test_top_level_optional_corruption_does_not_discard_valid_objects():
    payload = _analysis().model_dump()
    payload["characters"] = []
    payload["environment"] = []
    payload["objects"] = [
        {
            "label": "carried instrument",
            "visual_evidence": "A tangible instrument is held by the visible subject.",
            "visual_confidence": 0.84,
            "grounding_source": "visual",
            "confidence": 0.86,
        }
    ]
    payload["scene_summary"] = None
    payload["main_character_id"] = {"not": "an id"}

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert [entity.label for entity in analysis.objects] == ["carried instrument"]
    assert analysis.scene_summary == "Scene summary unavailable."
    assert analysis.main_character_id is None
    assert "scene_summary:null" in analysis.conflicts
    assert "main_character_id:string_type" in analysis.conflicts


def test_overlong_mood_is_shortened_at_sentence_boundary_and_strictly_validated():
    payload = _analysis().model_dump()
    payload["mood"] = (
        "First complete mood sentence. Second complete mood sentence. "
        + ("additional mood detail " * 40)
    )

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert analysis.mood == "First complete mood sentence. Second complete mood sentence."
    assert len(analysis.mood) <= SceneAnalysis.model_fields["mood"].metadata[-1].max_length
    assert "mood:string_too_long" in analysis.conflicts
    assert SceneAnalysis.model_validate(analysis.model_dump()).mood == analysis.mood
    assert [entity.label for entity in analysis.characters] == ["courier"]
    assert [entity.label for entity in analysis.objects] == ["signal flag", "bridge"]
    assert [entity.label for entity in analysis.environment] == ["fog"]
    assert len(analysis.relationships) == 1
    assert analysis.relationships[0].source_id == "courier-entity"
    assert analysis.relationships[0].target_id == "signal-flag-entity"


def test_valid_mood_is_unchanged_and_top_level_limits_are_model_derived():
    payload = _analysis().model_dump()
    valid_mood = "quiet, observant tension"
    payload["mood"] = valid_mood
    payload["scene_summary"] = "A complete summary. " + ("detail " * 100)
    payload["preserve"] = ["A complete preservation rule. " + ("detail " * 80)]

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert analysis.mood == valid_mood
    assert len(analysis.scene_summary) <= 400
    assert len(analysis.preserve[0]) <= 240
    assert "scene_summary:string_too_long" in analysis.conflicts
    assert "preserve.0:too_long" in analysis.conflicts


def test_overlong_mood_without_sentence_boundary_ends_at_word_boundary():
    payload = _analysis().model_dump()
    words = [f"moodword{index}" for index in range(160)]
    payload["mood"] = " ".join(words)

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert len(analysis.mood) <= 500
    assert analysis.mood == " ".join(words[: len(analysis.mood.split())])
    assert not analysis.mood.endswith(" ")
    assert "mood:string_too_long" in analysis.conflicts


def test_unusable_scene_analysis_still_requires_deterministic_fallback():
    with pytest.raises(ValueError, match="no recoverable semantic entities"):
        vertex._parse_scene_analysis_response("{}")


def test_validation_diagnostics_do_not_log_raw_model_output(caplog):
    raw_marker = "RAW_MODEL_MOOD_SHOULD_NOT_APPEAR"
    payload = _analysis().model_dump()
    payload["mood"] = raw_marker + (" detail" * 120)
    raw_response = json.dumps(payload)

    with caplog.at_level("WARNING", logger="cinema_agent.vertex"):
        vertex._log_response_diagnostics(
            SimpleNamespace(text=raw_response, candidates=[]),
            raw_response,
        )

    assert raw_marker not in caplog.text


@pytest.mark.parametrize(
    ("case_name", "characters", "objects", "environment", "main_ids"),
    [
        (
            "four-person ensemble",
            [
                _semantic_entity("person-1", "person one", "character", agentive=True, action="Look toward the group."),
                _semantic_entity("person-2", "person two", "character", agentive=True, action="Hold position."),
                _semantic_entity("person-3", "person three", "character", agentive=True, action="Turn slightly."),
                _semantic_entity("person-4", "person four", "character", agentive=True, action="Raise a hand."),
            ],
            [
                _semantic_entity("coat", "long coat", "object", agentive=False),
                _semantic_entity("lantern", "handheld lantern", "object", agentive=False, action="Sway gently."),
            ],
            [
                _semantic_entity("rain", "rain", "environment", agentive=False, action="Fall through the frame."),
                _semantic_entity("facade", "background facade", "environment", agentive=False),
            ],
            ["person-1", "person-2", "person-3", "person-4"],
        ),
        (
            "single animal protagonist",
            [
                _semantic_entity("animal", "small animal", "character", agentive=True, action="Look toward the light."),
            ],
            [],
            [_semantic_entity("grass", "grass", "environment", agentive=False)],
            ["animal"],
        ),
        (
            "robot with tools",
            [
                _semantic_entity("robot", "maintenance robot", "character", agentive=True, action="Reach toward the tool."),
            ],
            [
                _semantic_entity("wrench", "wrench", "object", agentive=False, action="Lift slightly."),
                _semantic_entity("case", "tool case", "object", agentive=False),
            ],
            [_semantic_entity("worklight", "work light", "environment", agentive=False, action="Flicker softly.")],
            ["robot"],
        ),
        (
            "vehicle protagonist",
            [
                _semantic_entity("vehicle", "autonomous vehicle", "character", agentive=True, action="Move within the lane."),
            ],
            [_semantic_entity("marker", "road marker", "object", agentive=False)],
            [_semantic_entity("mist", "low mist", "environment", agentive=False, action="Drift across the road.")],
            ["vehicle"],
        ),
        (
            "landscape without a character",
            [],
            [],
            [
                _semantic_entity("river", "river", "environment", agentive=False, action="Glint across the surface."),
                _semantic_entity("cloud", "cloud cover", "environment", agentive=False),
            ],
            [],
        ),
        (
            "characters with weather and architecture",
            [
                _semantic_entity("traveler-a", "traveler A", "character", agentive=True),
                _semantic_entity("traveler-b", "traveler B", "character", agentive=True),
            ],
            [_semantic_entity("banner", "hanging banner", "object", agentive=False)],
            [
                _semantic_entity("weather", "wind", "environment", agentive=False, action="Move through the scene."),
                _semantic_entity("architecture", "background architecture", "environment", agentive=False),
            ],
            ["traveler-a", "traveler-b"],
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_unrelated_scene_inventories_keep_generic_taxonomy(
    case_name,
    characters,
    objects,
    environment,
    main_ids,
):
    payload = _analysis().model_dump()
    payload.update(
        characters=characters,
        objects=objects,
        environment=environment,
        main_character_id=main_ids[0] if main_ids else None,
        main_character_ids=main_ids,
        relationships=[],
    )

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert len(analysis.characters) == len(characters), case_name
    assert len(analysis.objects) == len(objects), case_name
    assert len(analysis.environment) == len(environment), case_name
    assert [entity.entity_id for entity in analysis.characters] == [
        entity["entity_id"] for entity in characters
    ]
    assert all(entity.entity_type == "character" for entity in analysis.characters)
    assert all(entity.entity_type == "object" for entity in analysis.objects)
    assert all(entity.entity_type == "environment" for entity in analysis.environment)
    assert analysis.main_character_ids == main_ids


def test_structured_taxonomy_corrects_agentive_and_physical_entities_without_keywords():
    payload = _analysis().model_dump()
    payload.update(
        characters=[],
        objects=[],
        environment=[
            _semantic_entity(
                "agent",
                "unusual agent",
                "environment",
                agentive=True,
                action="React within the frame.",
            ),
            _semantic_entity(
                "garment",
                "distinct garment",
                "object",
                agentive=False,
                action="Shift slightly.",
            ),
        ],
        relationships=[],
    )

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert [entity.entity_id for entity in analysis.characters] == ["agent"]
    assert [entity.entity_id for entity in analysis.objects] == ["garment"]
    assert not analysis.environment
    assert "environment.0:category_corrected_to_characters" in analysis.conflicts
    assert "environment.1:category_corrected_to_objects" in analysis.conflicts


def test_multiple_declared_main_characters_drive_an_ensemble_focal_subject():
    payload = _analysis().model_dump()
    payload.update(
        characters=[
            _semantic_entity("a", "first subject", "character", agentive=True, action="Turn."),
            _semantic_entity("b", "second subject", "character", agentive=True, action="Turn."),
            _semantic_entity("c", "third subject", "character", agentive=True, action="Turn."),
            _semantic_entity("d", "fourth subject", "character", agentive=True, action="Turn."),
        ],
        objects=[],
        environment=[],
        main_character_id="a",
        main_character_ids=["a", "b", "c", "d"],
        relationships=[],
    )

    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))
    plan = scene_analysis_to_shot_plan(analysis, "The group turns together.", "focused")

    assert plan.focal_subject == "first subject, second subject, third subject, fourth subject"
    assert [entity.entity_id for entity in plan.motion_plan.visible_characters] == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_director_prompt_separates_manipulated_objects_from_environment():
    prompt = vertex.DIRECTOR_PROMPT

    assert "tangible inanimate entity remains in objects" in prompt
    assert "held, worn, attached" in prompt
    assert "natural phenomena" in prompt
    assert "Do not move a tangible object into environment" in prompt
    assert not any(
        noun in prompt.lower()
        for noun in ("astronomer", "lantern", "cloak", "portal")
    )


def test_direct_planner_uses_compact_analysis_and_json_mode(monkeypatch):
    models = _FakeModels(_analysis().model_dump_json())
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))

    plan = vertex.generate_shot_plan(
        "A courier raises a signal flag while fog drifts across the bridge.",
        "measured resolve",
        None,
    )

    assert isinstance(plan, ShotPlan)
    assert plan.motion_plan.movable_characters[0].label == "courier"
    assert plan.motion_plan.movable_objects[0].grounding_source == "screenplay"
    assert plan.motion_plan.movable_objects[0].support == "needs_confirmation"
    assert plan.motion_plan.environmental_motion[0].grounding_source == "visual_and_screenplay"
    assert plan.motion_plan.camera_movement.grounding_source == "visual_and_screenplay"
    assert plan.focal_subject == "courier"
    assert plan.relationships[0].action == "raises the signal flag"
    assert plan.shot.camera_motion == "pan_left"
    assert len(models.calls) == 1

    contents = models.calls[0]["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 1
    assert "A courier raises a signal flag" in contents[0].parts[0].text

    config = models.calls[0]["config"]
    assert config.system_instruction == vertex.DIRECTOR_PROMPT
    assert "not necessarily human or living" in vertex.DIRECTOR_PROMPT
    assert "main_character_id" in vertex.DIRECTOR_PROMPT
    assert "not size, centrality, or human appearance" in vertex.DIRECTOR_PROMPT
    assert "Do not repeat the" in vertex.DIRECTOR_PROMPT
    assert "mood is one short sentence" in vertex.DIRECTOR_PROMPT
    assert "scene_summary is at most two short sentences" in vertex.DIRECTOR_PROMPT
    assert "visual_evidence is one concise sentence or null" in vertex.DIRECTOR_PROMPT
    assert "Preserve every clearly visible, motion-relevant entity" in vertex.DIRECTOR_PROMPT
    assert config.response_mime_type == "application/json"
    assert vertex.DIRECTOR_OUTPUT_TOKEN_BUDGET == 16384
    assert config.max_output_tokens == 16384
    assert config.thinking_config.thinking_budget == 0
    assert config.response_schema is None
    assert config.response_json_schema is None
    assert "response_schema" not in config.model_fields_set
    assert "response_json_schema" not in config.model_fields_set


def test_director_output_budget_accepts_safe_environment_override(monkeypatch):
    models = _FakeModels(_analysis().model_dump_json())
    monkeypatch.setenv(vertex.DIRECTOR_OUTPUT_TOKEN_BUDGET_ENV, "12000")
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))

    vertex.generate_shot_plan("A courier waits.", "focused", None)

    config = models.calls[0]["config"]
    assert config.max_output_tokens == 12000


@pytest.mark.parametrize("configured", ["not-a-number", "0", "70000"])
def test_director_output_budget_rejects_unsafe_environment_override(
    monkeypatch, configured
):
    models = _FakeModels(_analysis().model_dump_json())
    monkeypatch.setenv(vertex.DIRECTOR_OUTPUT_TOKEN_BUDGET_ENV, configured)
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))

    vertex.generate_shot_plan("A courier waits.", "focused", None)

    config = models.calls[0]["config"]
    assert config.max_output_tokens == vertex.DIRECTOR_OUTPUT_TOKEN_BUDGET


def test_compact_prompt_includes_creative_intent_once_not_per_entity(monkeypatch):
    models = _FakeModels(_analysis().model_dump_json())
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))
    intent = "UNIQUE_CREATIVE_INTENT_SENTINEL"

    vertex.generate_shot_plan("A courier waits.", intent, None)

    prompt = models.calls[0]["contents"][0].parts[0].text
    assert prompt.count(intent) == 1
    assert intent not in json.dumps(_analysis().model_dump())


def test_compact_analysis_has_bounded_fields_and_item_counts():
    payload = _analysis().model_dump()
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate({**payload, "scene_summary": "x" * 401})
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(
            {**payload, "characters": [payload["characters"][0]] * 13}
        )
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(
            {**payload, "preserve": ["keep"] * 13}
        )
    with pytest.raises(ValidationError):
        SceneAnalysis.model_validate(
            {
                **payload,
                "relationships": [
                    {
                        **payload["relationships"][0],
                        "action": "x" * 241,
                    }
                ],
            }
        )


def _run_with_response(monkeypatch, response_text, response=None):
    models = _FakeModels(response_text)
    monkeypatch.setattr(
        vertex,
        "_client",
        lambda: _FakeClient(
            models
        ),
    )
    if response is not None:
        original = models.generate_content

        def generate_content(**kwargs):
            models.calls.append(kwargs)
            return response

        models.generate_content = generate_content
    return models


def test_direct_planner_accepts_one_standard_json_markdown_fence(monkeypatch):
    payload = _analysis().model_dump_json()
    models = _run_with_response(monkeypatch, f"```json\n{payload}\n```")

    plan = vertex.generate_shot_plan("A courier raises a flag.", "focused", None)

    assert plan.focal_subject == "courier"
    assert len(models.calls) == 1


@pytest.mark.parametrize(
    "response_text",
    [
        "",
        '{"scene_summary": "truncated"',
        "Here is the requested JSON:\n" + _analysis().model_dump_json(),
    ],
)
def test_direct_planner_rejects_empty_truncated_or_surrounding_prose(
    monkeypatch, caplog, response_text
):
    sentinel = "SCREENPLAY_SENTINEL_MUST_NOT_BE_LOGGED"
    response_text = response_text.replace("truncated", sentinel)
    models = _run_with_response(monkeypatch, response_text)

    with pytest.raises((ValidationError, ValueError)):
        vertex.generate_shot_plan(sentinel, "focused", None)

    assert len(models.calls) == 1
    diagnostics = caplog.text
    assert "candidate_count=" in diagnostics
    assert "response_text_length=" in diagnostics
    assert "starts_with_brace=" in diagnostics
    assert "ends_with_brace=" in diagnostics
    assert "markdown_fences_present=" in diagnostics
    assert "empty_response=" in diagnostics
    assert "max_output_tokens=16384" in diagnostics
    assert sentinel not in diagnostics


def test_token_limit_finish_reason_is_logged_without_model_output(monkeypatch, caplog):
    sentinel = "TOKEN_LIMIT_SENTINEL_MUST_NOT_BE_LOGGED"
    response = SimpleNamespace(
        text='{"scene_summary":"incomplete"',
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")],
    )
    _run_with_response(monkeypatch, response.text, response=response)

    with pytest.raises((ValidationError, ValueError)):
        vertex.generate_shot_plan(sentinel, "focused", None)

    assert "finish_reason=MAX_TOKENS" in caplog.text
    assert "empty_response=False" in caplog.text
    assert "starts_with_brace=True" in caplog.text
    assert "ends_with_brace=False" in caplog.text
    assert sentinel not in caplog.text


def test_malformed_semantic_candidate_is_removed_and_recorded_after_json_parsing(
    monkeypatch,
):
    payload = _analysis().model_dump()
    payload["characters"].append(
        {
            "entity_id": "malformed-entity",
            "label": "malformed entity",
            "visible": True,
            "visual_evidence": "Visible.",
            "screenplay_evidence": None,
            "suggested_motion": None,
            "confidence": 0.5,
            "role_status": "not-a-valid-role",
        }
    )
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier raises a signal flag.", "focused", None)

    assert [item.label for item in plan.motion_plan.movable_characters] == ["courier"]
    assert any("characters.1" in item and "literal_error" in item for item in plan.motion_plan.grounding_conflicts)
    assert len(models.calls) == 1


def test_overlong_summary_is_shortened_without_replacement_content(monkeypatch):
    payload = _analysis().model_dump()
    payload["scene_summary"] = (
        "The courier waits on the bridge. "
        + "A long atmospheric description follows without adding a new controllable subject. " * 12
    )
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier waits on a bridge.", "focused", None)

    assert len(plan.scene_summary) <= 400
    assert plan.scene_summary.startswith("The courier waits on the bridge.")
    assert plan.scene_summary.endswith(".")
    assert len(models.calls) == 1


def test_relationships_are_validated_individually_and_valid_entities_survive(
    monkeypatch, caplog
):
    raw_marker = "RAW_RELATIONSHIP_MARKER_MUST_NOT_BE_LOGGED"
    payload = _analysis().model_dump()
    payload["relationships"] = [
        payload["relationships"][0],
        {
            "source_id": "courier-entity",
            "relation": "observes",
            "target_id": "bridge-entity",
            "action": None,
        },
        {
            "source_id": "courier-entity",
            "relation": "touches",
            "target_id": "signal-flag-entity",
            "action": {"raw": raw_marker},
        },
        {
            "source_id": "courier-entity",
            "relation": "carries",
            "target_id": "signal-flag-entity",
            "action": [raw_marker],
        },
        {
            "source_id": "missing-entity",
            "relation": "affects",
            "target_id": "signal-flag-entity",
            "action": "affects the flag",
        },
    ]
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier raises a signal flag.", "focused", None)

    assert len(plan.motion_plan.movable_characters) == 1
    assert len(plan.motion_plan.movable_objects) == 1
    assert len(plan.relationships) == 2
    assert plan.relationships[1].action is None
    assert any("relationships.2.action:string_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert any("relationships.3.action:string_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert any(
        "relationships.4.source_id:reference_not_found" in item
        for item in plan.motion_plan.grounding_conflicts
    )
    assert raw_marker not in caplog.text
    assert len(models.calls) == 1


def test_oversized_descriptive_lists_are_bounded_and_entities_survive(monkeypatch):
    payload = _analysis().model_dump()
    payload["preserve"] = [f"Preserve visual detail {index}." for index in range(20)]
    payload["prohibit"] = [f"Do not alter detail {index}." for index in range(20)]
    payload["conflicts"] = [f"Conflict detail {index}." for index in range(12)]
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier stands on a bridge.", "focused", None)

    assert len(plan.motion_plan.preserved_elements) == 12
    assert len(plan.motion_plan.prohibited_changes) == 16
    assert len(plan.motion_plan.grounding_conflicts) <= 8
    assert plan.motion_plan.movable_characters[0].label == "courier"
    assert any("preserve:too_many" in item for item in plan.motion_plan.grounding_conflicts)
    assert any("prohibit:too_many" in item for item in plan.motion_plan.grounding_conflicts)
    assert len(models.calls) == 1


def test_descriptive_lists_normalize_deduplicate_and_shorten_entries(monkeypatch):
    payload = _analysis().model_dump()
    payload["preserve"] = [
        "  Keep   the bridge silhouette.  ",
        "Keep the bridge silhouette.",
        "",
        42,
        "A long descriptive preservation rule " * 30,
    ]
    analysis = vertex._parse_scene_analysis_response(json.dumps(payload))

    assert analysis.preserve[0] == "Keep the bridge silhouette."
    assert len(analysis.preserve) == 2
    assert all(len(item) <= 240 for item in analysis.preserve)
    assert any("preserve.1:duplicate" in item for item in analysis.conflicts)
    assert any("preserve.2:empty" in item for item in analysis.conflicts)
    assert any("preserve.3:string_type" in item for item in analysis.conflicts)
    assert any("preserve.4:too_long" in item for item in analysis.conflicts)


def test_malformed_camera_is_recovered_without_discarding_valid_entities(monkeypatch):
    payload = _analysis().model_dump()
    payload["camera"] = {
        "movement": {"untrusted": "camera payload"},
        "visual_evidence": ["not a string"],
        "confidence": "not a number",
    }
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier stands on a bridge.", "focused", None)

    assert plan.motion_plan.movable_characters[0].label == "courier"
    assert plan.motion_plan.camera_movement.action == "drift"
    assert plan.shot.camera_motion == "drift"
    assert any("camera.movement:string_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert any("camera.visual_evidence:string_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert any("camera.confidence:invalid" in item for item in plan.motion_plan.grounding_conflicts)
    assert len(models.calls) == 1


def test_valid_camera_survives_malformed_optional_lists(monkeypatch):
    payload = _analysis().model_dump()
    payload["preserve"] = {"not": "a list"}
    payload["prohibit"] = [None, "", "Do not add a second figure."]
    payload["conflicts"] = "not a list"
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier stands on a bridge.", "focused", None)

    assert plan.motion_plan.movable_characters[0].label == "courier"
    assert plan.motion_plan.camera_movement.action == "pan left"
    assert plan.shot.camera_motion == "pan_left"
    assert any("preserve:list_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert any("prohibit.0:string_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert any("conflicts:list_type" in item for item in plan.motion_plan.grounding_conflicts)
    assert len(models.calls) == 1


def test_valid_entities_survive_every_optional_field_failure(monkeypatch):
    payload = _analysis().model_dump()
    payload.pop("camera")
    payload["preserve"] = None
    payload["prohibit"] = 17
    payload["conflicts"] = [{"raw": "must not be logged"}]
    models = _run_with_response(monkeypatch, json.dumps(payload))

    plan = vertex.generate_shot_plan("A courier stands on a bridge.", "focused", None)

    assert plan.motion_plan.movable_characters[0].label == "courier"
    assert plan.motion_plan.movable_objects
    assert plan.motion_plan.environmental_motion
    assert plan.motion_plan.camera_movement.action == "drift"
    assert all("must not be logged" not in item for item in plan.motion_plan.grounding_conflicts)
    assert len(models.calls) == 1


def test_completely_unusable_core_analysis_still_falls_back(monkeypatch):
    payload = {
        "scene_summary": "No recoverable semantic content.",
        "characters": [{"invalid": True}],
        "camera": None,
    }
    models = _run_with_response(monkeypatch, json.dumps(payload))

    with pytest.raises((ValueError, ValidationError), match="recoverable semantic"):
        vertex.generate_shot_plan("An arbitrary scene.", "uncertain", None)

    assert len(models.calls) == 1


def test_direct_planner_keeps_one_user_content_with_image(monkeypatch):
    models = _FakeModels(_analysis().model_dump_json())
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))
    image_data_url = (
        "data:image/png;base64,"
        + base64.b64encode(b"storyboard-bytes").decode("ascii")
    )

    vertex.generate_shot_plan("A quiet bridge scene with fog.", "quiet dread", image_data_url)

    contents = models.calls[0]["contents"]
    assert len(contents) == 1
    assert len(contents[0].parts) == 2
    prompt_part, image_part = contents[0].parts
    assert prompt_part.text.startswith("CREATIVE INTENT: quiet dread")
    assert image_part.inline_data.data == b"storyboard-bytes"
    assert image_part.inline_data.mime_type == "image/png"


@pytest.mark.parametrize(
    ("image_data_url", "message"),
    [
        ("data:image/png;base64,", "non-empty bytes"),
        (
            "data:image/gif;base64," + base64.b64encode(b"gif-bytes").decode("ascii"),
            "Unsupported storyboard image MIME type",
        ),
    ],
)
def test_invalid_storyboard_image_is_rejected_before_provider(monkeypatch, image_data_url, message):
    models = _FakeModels(_analysis().model_dump_json())
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))

    with pytest.raises(ValueError, match=message):
        vertex.generate_shot_plan("A bridge scene with fog.", "quiet dread", image_data_url)
    assert models.calls == []


@pytest.mark.parametrize(
    "response_text",
    [
        "```json\n{}\n```",
        "not json",
        json.dumps({"scene_summary": "incomplete"}),
        json.dumps({**_analysis().model_dump(), "invented": True}),
    ],
)
def test_malformed_analysis_fails_without_retry(monkeypatch, response_text):
    models = _FakeModels(response_text)
    monkeypatch.setattr(vertex, "_client", lambda: _FakeClient(models))

    with pytest.raises((ValueError, ValidationError)):
        vertex.generate_shot_plan("An arbitrary scene with no fixed subject.", "uncertain", None)
    assert len(models.calls) == 1