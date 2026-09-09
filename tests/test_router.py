import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app as app_module
from app import app
from cinema_agent.rate_limit import VertexRateLimiter
from cinema_agent.demo import demo_critique, demo_plan
from cinema_agent.rate_limit import VertexRateLimiter
from cinema_agent.router import route_scene_action
from cinema_agent.tools import runtime_mode


def test_vertex_error_details_preserve_safe_google_fields_without_payloads():
    class FakeClientError(Exception):
        code = 400
        status = "INVALID_ARGUMENT"
        message = (
            "Request contains invalid image data "
            "data:image/png;base64," + ("A" * 220)
        )

    details = app_module._safe_vertex_error_details(FakeClientError())

    assert details == {
        "error_type": "FakeClientError",
        "http_status": "400",
        "google_status": "INVALID_ARGUMENT",
        "category": "invalid_request_or_schema",
        "message": "Request contains invalid image data [redacted image data]",
        "validation_errors": [],
    }


def test_validation_failure_logs_only_bounded_paths_and_types(caplog):
    class FakeValidationError(Exception):
        def errors(self):
            return [
                {
                    "loc": ("shot", "motion_intensity"),
                    "type": "float_parsing",
                    "input": "private-returned-value",
                },
                {
                    "loc": ("depth_notes",),
                    "type": "dict_type",
                    "input": "private-returned-value",
                },
            ]

    details = app_module._safe_vertex_error_details(FakeValidationError())

    assert details == {
        "error_type": "FakeValidationError",
        "http_status": "unknown",
        "google_status": "unknown",
        "category": "response_validation",
        "message": "SceneAnalysis response failed local validation",
        "validation_errors": [
            "shot.motion_intensity:float_parsing",
            "depth_notes:dict_type",
        ],
    }
    with caplog.at_level("WARNING", logger="app"):
        app_module._log_vertex_failure(FakeValidationError())
    assert "validation_errors=shot.motion_intensity:float_parsing,depth_notes:dict_type" in caplog.text
    assert "private-returned-value" not in caplog.text


def test_vertex_failure_log_is_bounded_and_redacts_credentials(caplog):
    class FakeClientError(Exception):
        code = 403
        status = "PERMISSION_DENIED"
        message = "authorization=secret-token message"

    with caplog.at_level("WARNING", logger="app"):
        app_module._log_vertex_failure(FakeClientError())

    assert "http_status=403" in caplog.text
    assert "google_status=PERMISSION_DENIED" in caplog.text
    assert "category=missing_iam_permission" in caplog.text
    assert "authorization=[redacted]" in caplog.text
    assert "secret-token" not in caplog.text


def test_local_renderer_actions_are_routed_to_local_2_5d():
    decision = route_scene_action("The camera pushes in through fog as rain falls.")

    assert decision.classification == "LOCAL_2_5D"
    assert "camera motion" in decision.matched_actions
    assert "2.5D depth and atmosphere" in decision.matched_actions


def test_semantic_subject_actions_require_generative_video():
    decision = route_scene_action("A person walks toward the door and smiles.")

    assert decision.classification == "GENERATIVE_VIDEO_REQUIRED"
    assert "character movement" in decision.matched_actions
    assert "facial movement" in decision.matched_actions
    assert "No paid generation API was called" in decision.rationale


def test_common_character_and_object_motion_requires_generative_video():
    examples = [
        "Mara raises a lantern and looks toward the cabin.",
        "The lantern swings while the flame bends toward the doorway.",
        "A person sits down and reaches for the book.",
        "A car moves across the frame.",
        "A door opens while a curtain flutters.",
        "The statue transforms into a bird.",
    ]

    for screenplay in examples:
        assert route_scene_action(screenplay).classification == "GENERATIVE_VIDEO_REQUIRED"


def test_mood_language_does_not_create_a_semantic_action():
    decision = route_scene_action(
        "The camera pushes in through fog.",
        "running on adrenaline",
    )

    assert decision.classification == "LOCAL_2_5D"


def test_unreliable_actions_are_unsupported():
    decision = route_scene_action("The character teleports across the room.")

    assert decision.classification == "UNSUPPORTED"
    assert "unsupported physical spectacle" in decision.matched_actions


def test_non_graphic_adult_combat_requires_generative_video():
    decision = route_scene_action(
        "Two adults fight on the rain-soaked platform: they dodge, pivot, trade controlled "
        "punches and fall safely. No blood, gore, weapons or severe injury."
    )

    assert decision.classification == "GENERATIVE_VIDEO_REQUIRED"
    assert "non-graphic adult combat choreography" in decision.matched_actions
    assert "local 2.5D compositor" in decision.rationale


def test_combat_classifier_requires_bounded_context_instead_of_allowing_keywords():
    assert route_scene_action("A person fights.").classification == "UNSUPPORTED"
    assert route_scene_action("A detective kicks a door.").classification == "UNSUPPORTED"
    assert (
        route_scene_action("A detective fights a smuggler with a knife.")
        .classification
        == "UNSUPPORTED"
    )


@pytest.mark.parametrize(
    "screenplay",
    [
        "Two adults fight with blood and gore visible.",
        "A detective fights a smuggler and an accomplice with a pistol.",
        "Two adults fight until one is killed.",
        "An adult tortures a captive.",
        "A child is attacked by an adult.",
        "A woman is subjected to sexual violence.",
    ],
)
def test_genuinely_unsafe_combat_categories_remain_blocked(screenplay):
    decision = route_scene_action(screenplay)

    assert decision.classification == "UNSUPPORTED"
    assert decision.matched_actions


def test_direct_endpoint_exposes_routing_decision(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={
            "screenplay": "The camera pans slowly through fog while rain falls over the empty room.",
            "mood": "quiet dread",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "demo"
    assert payload["routing"]["classification"] == "LOCAL_2_5D"
    assert payload["routing"]["rationale"]
    motion_plan = payload["plan"]["motion_plan"]
    assert motion_plan["camera_movement"]["screenplay_evidence"]
    assert motion_plan["preserved_elements"]
    assert motion_plan["prohibited_changes"]


def test_vertex_multimodal_detective_scene_is_eligible_without_provider_calls(monkeypatch):
    import cinema_agent.vertex as vertex
    from cinema_agent.schemas import SceneEntity

    screenplay = (
        "A detective fights a smuggler and an accomplice on the rain-soaked depot platform. "
        "They dodge, pivot and trade controlled punches. No blood, gore, weapons or severe injury."
    )
    image = "data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk"
    base_plan = demo_plan(screenplay, "urgent but controlled", image)
    visible_characters = [
        SceneEntity(
            entity_id=label,
            label=label,
            entity_type="character",
            visual_evidence=f"The storyboard visibly shows the {label}.",
            screenplay_evidence=screenplay,
            grounding_source="visual_and_screenplay",
            confidence=0.95,
            visual_confidence=0.95,
            support="supported",
        )
        for label in ("detective", "smuggler", "accomplice")
    ]
    base_plan = base_plan.model_copy(
        update={
            "motion_plan": base_plan.motion_plan.model_copy(
                update={"visible_characters": visible_characters}
            )
        }
    )
    calls = []

    def fake_plan(*args):
        calls.append("plan")
        return base_plan

    def fake_critique(*args):
        calls.append("critique")
        return demo_critique(base_plan)

    monkeypatch.setattr(app_module, "runtime_mode", lambda: "vertex")
    monkeypatch.setattr(app_module, "vertex_request_limiter", VertexRateLimiter())
    monkeypatch.setattr(vertex, "generate_shot_plan", fake_plan)
    monkeypatch.setattr(vertex, "generate_critique", fake_critique)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={"screenplay": screenplay, "mood": "urgent but controlled", "image_data_url": image},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["analysis_source"] == "vertex_multimodal"
    assert payload["routing"]["classification"] == "GENERATIVE_VIDEO_REQUIRED"
    assert calls == ["plan", "critique"]
    session_id = client.cookies.get(app_module.SESSION_COOKIE_NAME)
    context = app_module.video_job_service.direct_context(session_id)
    assert context["eligibility"]["controlled_test_eligible"] is True
    assert context["eligibility"]["generative_video_required"] is True


def test_direct_endpoint_exposes_camera_grammar_and_signatures(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={
            "screenplay": "A vast majestic reveal opens over the valley.",
            "mood": "awe",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["camera_grammar"] == "awe"
    assert payload["shot_signature"]["motion_type"] in {"pan_left", "pan_right"}
    assert payload["shot_signature"]["parallax_strength"] >= 0.8
    assert payload["revision_camera_grammar"] == "awe"
    assert payload["revision_shot_signature"]
    assert any(item["step"] == "CAMERA" for item in payload["activity"])


def test_vertex_mode_uses_multimodal_planning_for_semantic_scene(monkeypatch):
    import app as app_module
    import cinema_agent.vertex as vertex

    calls = []

    def fake_plan(screenplay, mood, image_data_url):
        calls.append("plan")
        return demo_plan(screenplay, mood)

    def fake_critique(screenplay, plan):
        calls.append("critique")
        return demo_critique(plan)

    monkeypatch.setattr(app_module, "runtime_mode", lambda: "vertex")
    monkeypatch.setattr(app_module, "vertex_request_limiter", VertexRateLimiter())
    monkeypatch.setattr(vertex, "generate_shot_plan", fake_plan)
    monkeypatch.setattr(vertex, "generate_critique", fake_critique)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={
            "screenplay": "A person walks toward the door and opens it slowly.",
            "mood": "quiet dread",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "vertex"
    assert payload["routing"]["classification"] == "GENERATIVE_VIDEO_REQUIRED"
    assert payload["analysis_source"] == "vertex_multimodal"
    assert calls == ["plan", "critique"]
    assert not any(item["step"] == "POLICY" for item in payload["activity"])


def test_unusable_scene_analysis_json_uses_deterministic_fallback(monkeypatch):
    import app as app_module
    import cinema_agent.vertex as vertex

    def unusable_plan(*args):
        vertex._parse_scene_analysis_response("{}")

    monkeypatch.setattr(app_module, "runtime_mode", lambda: "vertex")
    monkeypatch.setattr(app_module, "vertex_request_limiter", VertexRateLimiter())
    monkeypatch.setattr(vertex, "generate_shot_plan", unusable_plan)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={
            "screenplay": "A quiet room remains still.",
            "mood": "restrained",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "demo"
    assert payload["analysis_source"] == "deterministic_fallback"
    assert payload["activity"][0]["detail"] == (
        "Vertex response failed local validation; deterministic fallback used."
    )


def test_vertex_mode_still_plans_local_actions(monkeypatch):
    import app as app_module
    import cinema_agent.vertex as vertex

    calls = []

    def fake_plan(screenplay, mood, image_data_url):
        calls.append("plan")
        return demo_plan(screenplay, mood)

    def fake_critique(screenplay, plan):
        calls.append("critique")
        return demo_critique(plan)

    monkeypatch.setattr(app_module, "runtime_mode", lambda: "vertex")
    monkeypatch.setattr(app_module, "vertex_request_limiter", VertexRateLimiter())
    monkeypatch.setattr(vertex, "generate_shot_plan", fake_plan)
    monkeypatch.setattr(vertex, "generate_critique", fake_critique)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={
            "screenplay": "The camera pans through fog as rain crosses the frame.",
            "mood": "quiet dread",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["routing"]["classification"] == "LOCAL_2_5D"
    assert payload["analysis_source"] == "vertex_multimodal"
    assert calls == ["plan", "critique"]


def test_veo_control_is_disabled_and_cost_labelled():
    client = TestClient(app)

    response = client.get("/")

    assert response.status_code == 200
    assert 'id="generateVeo" class="secondary" disabled' in response.text
    assert "Cost approval required." in response.text


def test_vertex_is_locked_without_explicit_inference_opt_in(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "configured-project")
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ALLOW_VERTEX_INFERENCE", "false")

    assert runtime_mode() == "demo"


def test_vertex_requires_demo_opt_out_and_explicit_inference_opt_in(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "configured-project")
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ALLOW_VERTEX_INFERENCE", "true")

    assert runtime_mode() == "vertex"


def test_locked_demo_endpoint_never_reaches_vertex(monkeypatch):
    import app as app_module
    import cinema_agent.vertex as vertex

    def prohibited(*args, **kwargs):
        raise AssertionError("Locked demo mode must not call Vertex")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "configured-project")
    monkeypatch.setenv("DEMO_MODE", "false")
    monkeypatch.setenv("ALLOW_VERTEX_INFERENCE", "false")
    monkeypatch.setattr(vertex, "generate_shot_plan", prohibited)
    monkeypatch.setattr(vertex, "generate_critique", prohibited)
    monkeypatch.setattr(app_module, "runtime_mode", runtime_mode)
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={
            "screenplay": "The camera pushes in through fog while rain falls over the empty room.",
            "mood": "quiet dread",
        },
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "demo"


def test_rate_limiter_enforces_three_successes_per_client_ip():
    limiter = VertexRateLimiter(per_client_limit=3, global_limit=25)

    for _ in range(3):
        allowed, result = limiter.run("198.51.100.10", lambda: "vertex-result")
        assert allowed is True
        assert result == "vertex-result"

    allowed, result = limiter.run("198.51.100.10", lambda: "should-not-run")

    assert allowed is False
    assert result is None


def test_rate_limiter_enforces_global_success_limit():
    limiter = VertexRateLimiter(per_client_limit=3, global_limit=2)

    assert limiter.run("198.51.100.10", lambda: "first")[0] is True
    assert limiter.run("198.51.100.11", lambda: "second")[0] is True
    allowed, result = limiter.run("198.51.100.12", lambda: "should-not-run")

    assert allowed is False
    assert result is None


def test_rate_limiter_does_not_count_failed_vertex_operations():
    limiter = VertexRateLimiter(per_client_limit=1, global_limit=1)

    with pytest.raises(RuntimeError):
        limiter.run("198.51.100.10", lambda: (_ for _ in ()).throw(RuntimeError("Vertex failed")))

    allowed, result = limiter.run("198.51.100.10", lambda: "retry-success")

    assert allowed is True
    assert result == "retry-success"


def test_rate_limited_fallback_does_not_call_vertex(monkeypatch):
    import app as app_module
    import cinema_agent.vertex as vertex

    limiter = VertexRateLimiter(per_client_limit=1, global_limit=25)
    calls = []

    def fake_plan(screenplay, mood, image_data_url):
        calls.append("plan")
        return demo_plan(screenplay, mood)

    def fake_critique(screenplay, plan):
        calls.append("critique")
        return demo_critique(plan)

    monkeypatch.setattr(app_module, "runtime_mode", lambda: "vertex")
    monkeypatch.setattr(app_module, "vertex_request_limiter", limiter)
    monkeypatch.setattr(vertex, "generate_shot_plan", fake_plan)
    monkeypatch.setattr(vertex, "generate_critique", fake_critique)
    client = TestClient(app)
    payload = {
        "screenplay": "The camera pans through fog as rain crosses the frame.",
        "mood": "quiet dread",
    }

    first_response = client.post("/api/direct", json=payload)
    second_response = client.post("/api/direct", json=payload)

    assert first_response.status_code == 200
    assert first_response.json()["mode"] == "vertex"
    assert second_response.status_code == 200
    assert second_response.json()["mode"] == "rate_limited_fallback"
    assert calls == ["plan", "critique"]
    assert any(item["step"] == "RATE LIMIT" for item in second_response.json()["activity"])


def test_vertex_client_uses_service_account_secret(monkeypatch):
    import cinema_agent.vertex as vertex

    parsed_info = []
    client_calls = []
    credentials = object()
    service_account_info = {
        "type": "service_account",
        "project_id": "credential-project",
        "client_email": "framepilot@example.invalid",
    }

    def fake_from_service_account_info(info, scopes):
        parsed_info.append(info)
        assert scopes == ["https://www.googleapis.com/auth/cloud-platform"]
        return credentials

    class FakeClient:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "configured-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "europe-west1")
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_JSON", json.dumps(service_account_info))
    monkeypatch.setattr(
        vertex.service_account.Credentials,
        "from_service_account_info",
        staticmethod(fake_from_service_account_info),
    )
    monkeypatch.setattr(vertex.genai, "Client", FakeClient)

    client = vertex._client()

    assert isinstance(client, FakeClient)
    assert parsed_info == [service_account_info]
    assert client_calls == [
        {
            "enterprise": True,
            "project": "configured-project",
            "location": "europe-west1",
            "credentials": credentials,
        }
    ]


def test_vertex_client_uses_adc_when_service_account_secret_is_absent(monkeypatch):
    import cinema_agent.vertex as vertex

    client_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)

    def credentials_factory_should_not_run(info):
        raise AssertionError("ADC fallback must not parse service-account credentials")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "configured-project")
    monkeypatch.delenv("GOOGLE_SERVICE_ACCOUNT_JSON", raising=False)
    monkeypatch.setattr(
        vertex.service_account.Credentials,
        "from_service_account_info",
        staticmethod(credentials_factory_should_not_run),
    )
    monkeypatch.setattr(vertex.genai, "Client", FakeClient)

    client = vertex._client()

    assert isinstance(client, FakeClient)
    assert client_calls == [
        {
            "enterprise": True,
            "project": "configured-project",
            "location": "us-central1",
        }
    ]