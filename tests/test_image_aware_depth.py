import base64
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import app
from cinema_agent.cache import AnalysisCache
from cinema_agent.demo import demo_critique, demo_plan
from cinema_agent.schemas import SceneAnalysis


IMAGE_DATA_URL = "data:image/jpeg;base64," + base64.b64encode(b"storyboard-bytes").decode()
SCREENPLAY = "The camera pushes in through fog while rain falls over the empty room."


def test_vertex_director_sends_image_and_text_in_one_structured_request(monkeypatch):
    import cinema_agent.vertex as vertex

    calls = []
    analysis = SceneAnalysis(
        scene_summary="A quiet room with fog and rain.",
        characters=[],
        objects=[],
        environment=[
            {
                "label": "fog",
                "visible": True,
                "visual_evidence": "A fog layer is visible in the frame.",
                "screenplay_evidence": "Fog fills the room.",
                "suggested_motion": "Drift gently across the frame.",
                "confidence": 0.86,
            }
        ],
        camera={
            "movement": "push in",
            "visual_evidence": "The centered composition supports a careful push in.",
            "screenplay_evidence": SCREENPLAY,
            "confidence": 0.88,
        },
        preserve=[],
        prohibit=[],
        conflicts=[],
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(text=analysis.model_dump_json())

    class FakeClient:
        models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(vertex, "_client", lambda: FakeClient())

    result = vertex.generate_shot_plan(SCREENPLAY, "quiet dread", IMAGE_DATA_URL)

    assert result.depth_layout is None
    assert len(calls) == 1
    contents = calls[0]["contents"]
    assert len(contents) == 1
    assert contents[0].role == "user"
    assert len(contents[0].parts) == 2
    prompt_part, image_part = contents[0].parts
    assert prompt_part.text.startswith("CREATIVE INTENT: quiet dread")
    assert SCREENPLAY in prompt_part.text
    assert image_part.inline_data.data == b"storyboard-bytes"
    assert image_part.inline_data.mime_type == "image/jpeg"
    assert calls[0]["config"].response_mime_type == "application/json"
    assert calls[0]["config"].response_schema is None
    assert calls[0]["config"].response_json_schema is None
    assert "SceneEntity" in calls[0]["config"].system_instruction
    assert "visual_evidence" in calls[0]["config"].system_instruction
    assert "screenplay_evidence" in calls[0]["config"].system_instruction


def test_repeated_image_scene_uses_cache_without_second_vertex_operation(monkeypatch):
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
    monkeypatch.setattr(app_module, "analysis_cache", AnalysisCache())
    monkeypatch.setattr(vertex, "generate_shot_plan", fake_plan)
    monkeypatch.setattr(vertex, "generate_critique", fake_critique)
    client = TestClient(app)
    payload = {"screenplay": SCREENPLAY, "mood": "quiet dread", "image_data_url": IMAGE_DATA_URL}

    first = client.post("/api/direct", json=payload)
    second = client.post("/api/direct", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["depth_source"] == "image_aware"
    assert second.json()["depth_source"] == "image_aware"
    assert calls == ["plan", "critique"]
    assert any(item["step"] == "CACHE" for item in second.json()["activity"])


def test_missing_image_layout_falls_back_to_deterministic_depth(monkeypatch):
    import app as app_module
    import cinema_agent.vertex as vertex

    plan_without_layout = demo_plan(SCREENPLAY, "quiet dread").model_copy(
        update={"depth_layout": None}
    )
    monkeypatch.setattr(app_module, "runtime_mode", lambda: "vertex")
    monkeypatch.setattr(app_module, "analysis_cache", AnalysisCache())
    monkeypatch.setattr(vertex, "generate_shot_plan", lambda *args: plan_without_layout)
    monkeypatch.setattr(vertex, "generate_critique", lambda screenplay, plan: demo_critique(plan))
    client = TestClient(app)

    response = client.post(
        "/api/direct",
        json={"screenplay": SCREENPLAY, "mood": "quiet dread", "image_data_url": IMAGE_DATA_URL},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["depth_source"] == "heuristic"
    assert payload["plan"]["depth_layout"]["background"]["polygon"]


def test_client_upload_contract_resizes_before_direct_request():
    app_js = open("static/app.js").read()

    assert "1024 / longestEdge" in app_js
    assert "image/jpeg" in app_js
    assert "toDataURL('image/jpeg', 0.75)" in app_js
    assert "image_data_url: imageData" in app_js