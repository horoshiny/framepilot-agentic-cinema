import base64
import json
import pytest
from fastapi.testclient import TestClient

import app as app_module
from cinema_agent.video_jobs import (
    DuplicateSceneKind,
    GenerationLimitReached,
    MockVideoJobService,
)
from cinema_agent.video_ledger import VideoLedger
from cinema_agent.video_critique import MockVideoCritiqueProvider, VideoCritiqueService
from cinema_agent.video_provider import MockVideoProvider
from tests.test_video_jobs import FakeVertexProvider, approved_request, request


SCREENPLAY = "A figure crosses the empty platform while the signal changes and rain gathers on the glass."
IMAGE_DATA_URL = "data:image/png;base64," + base64.b64encode(b"mock-image").decode()


def js_hash_text(value: str) -> str:
    result = 2166136261
    for byte in value.encode():
        result ^= byte
        result = (result * 16777619) & 0xFFFFFFFF
    return f"scene-{result:08x}"


def approval_payload(direct_result):
    plan = direct_result["plan"]
    return {
        "kind": "first_cut",
        "scene_key": js_hash_text(
            json.dumps(
                {
                    "summary": plan["scene_summary"],
                    "intent": plan["emotional_intent"],
                    "image": True,
                },
                separators=(",", ":"),
            )
        ),
        "source_signature": js_hash_text(
            json.dumps(
                {
                    "shot": plan["shot"],
                    "motionPlan": plan["motion_plan"],
                    "revision": direct_result["critique"]["revision"],
                },
                separators=(",", ":"),
            )
        ),
        "first_cut_job_id": None,
        "screenplay": SCREENPLAY,
        "creative_intent": "quiet uncertainty",
        "shot_plan": plan,
        "image_handle": direct_result["image_handle"],
        "video_critique": None,
    }


@pytest.fixture
def isolated_mock_app(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.setenv("ALLOW_VERTEX_INFERENCE", "false")
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "mock")
    service = MockVideoJobService(provider=MockVideoProvider(), generation_limit=5)
    service.provider_name = "mock"
    monkeypatch.setattr(app_module, "video_job_service", service)
    return service


def test_direct_and_approval_survive_proxy_ip_change_with_same_session(isolated_mock_app):
    direct_client = TestClient(app_module.app, client=("10.60.40.83", 1000))
    direct_response = direct_client.post(
        "/api/direct",
        json={
            "screenplay": SCREENPLAY,
            "mood": "quiet uncertainty",
            "image_data_url": IMAGE_DATA_URL,
        },
    )
    assert direct_response.status_code == 200
    cookie = direct_response.cookies.get(app_module.SESSION_COOKIE_NAME)
    assert cookie and len(cookie) >= 32
    set_cookie = direct_response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert "samesite=lax" in set_cookie.lower()
    assert "Secure" not in set_cookie

    approval_client = TestClient(app_module.app, client=("10.60.5.61", 1000))
    approval_client.cookies.set(app_module.SESSION_COOKIE_NAME, cookie)
    approval_response = approval_client.post(
        "/api/video-jobs/approval",
        json=approval_payload(direct_response.json()),
    )

    assert approval_response.status_code == 200
    assert approval_response.json()["status"] == "approval_required"


def test_another_session_cannot_use_storyboard_handle(isolated_mock_app):
    owner = TestClient(app_module.app, client=("192.0.2.10", 1000))
    direct_response = owner.post(
        "/api/direct",
        json={"screenplay": SCREENPLAY, "mood": "quiet uncertainty", "image_data_url": IMAGE_DATA_URL},
    )
    assert direct_response.status_code == 200

    other_session = TestClient(app_module.app, client=("192.0.2.10", 1000))
    response = other_session.post(
        "/api/video-jobs/approval",
        json=approval_payload(direct_response.json()),
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "invalid_image",
        "message": "Storyboard image is not available for this client.",
    }


def test_storyboard_image_availability_is_session_owned(isolated_mock_app):
    owner = TestClient(app_module.app, client=("192.0.2.11", 1000))
    direct_response = owner.post(
        "/api/direct",
        json={"screenplay": SCREENPLAY, "mood": "quiet uncertainty", "image_data_url": IMAGE_DATA_URL},
    )
    image_handle = direct_response.json()["image_handle"]
    assert owner.get(f"/api/storyboard-images/{image_handle}").json() == {"available": True}

    other_session = TestClient(app_module.app, client=("192.0.2.12", 1000))
    response = other_session.get(f"/api/storyboard-images/{image_handle}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Storyboard image is no longer available for this session."


def test_owned_job_endpoint_returns_existing_video_critique(monkeypatch, isolated_mock_app):
    direct_client = TestClient(app_module.app, client=("192.0.2.13", 1000))
    direct_response = direct_client.post(
        "/api/direct",
        json={"screenplay": SCREENPLAY, "mood": "quiet uncertainty", "image_data_url": None},
    )
    payload = approval_payload(direct_response.json())
    payload["image_handle"] = None
    approval = direct_client.post("/api/video-jobs/approval", json=payload).json()
    created = direct_client.post(
        "/api/video-jobs",
        json={**payload, "approved": True, "approval_id": approval["approval_id"]},
    ).json()
    job_id = created["job_id"]
    isolated_mock_app._next_poll_at[job_id] = 0
    direct_client.get(f"/api/video-jobs/{job_id}")
    isolated_mock_app._next_poll_at[job_id] = 0
    completed = direct_client.get(f"/api/video-jobs/{job_id}")
    assert completed.json()["status"] == "completed"

    critique_service = VideoCritiqueService(provider=MockVideoCritiqueProvider())
    monkeypatch.setattr(app_module, "video_critique_service", critique_service)
    critique = direct_client.post(f"/api/video-jobs/{job_id}/critique")
    assert critique.status_code == 200
    restored = direct_client.get(f"/api/video-jobs/{job_id}")
    assert restored.status_code == 200
    assert restored.json()["video_critique"]["status"] == "available"


def test_protected_video_retrieval_is_session_owned_across_ip_changes(isolated_mock_app):
    owner = TestClient(app_module.app, client=("198.51.100.10", 1000))
    direct_response = owner.post(
        "/api/direct",
        json={"screenplay": SCREENPLAY, "mood": "quiet uncertainty", "image_data_url": IMAGE_DATA_URL},
    )
    approval_client = TestClient(app_module.app, client=("198.51.100.11", 1000))
    session_cookie = direct_response.cookies.get(app_module.SESSION_COOKIE_NAME)
    approval_client.cookies.set(app_module.SESSION_COOKIE_NAME, session_cookie)
    payload = approval_payload(direct_response.json())
    approval = approval_client.post("/api/video-jobs/approval", json=payload).json()

    create_response = approval_client.post(
        "/api/video-jobs",
        json={**payload, "approved": True, "approval_id": approval["approval_id"]},
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["job_id"]
    isolated_mock_app._next_poll_at[job_id] = 0
    generating = approval_client.get(f"/api/video-jobs/{job_id}")
    assert generating.status_code == 200
    isolated_mock_app._next_poll_at[job_id] = 0
    completed = approval_client.get(f"/api/video-jobs/{job_id}")
    assert completed.status_code == 200
    assert completed.json()["status"] == "completed"

    same_session_new_ip = TestClient(app_module.app, client=("203.0.113.20", 1000))
    same_session_new_ip.cookies.set(app_module.SESSION_COOKIE_NAME, session_cookie)
    retrieved = same_session_new_ip.get(f"/api/video-jobs/{job_id}/video")
    assert retrieved.status_code == 200
    assert retrieved.headers["content-type"].startswith("video/")

    other_session = TestClient(app_module.app, client=("203.0.113.20", 1000))
    denied = other_session.get(f"/api/video-jobs/{job_id}/video")
    assert denied.status_code == 404


def test_session_cookie_is_secure_for_https_requests(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    client = TestClient(app_module.app, base_url="https://testserver", client=("203.0.113.30", 1000))
    response = client.get("/api/health")
    assert response.status_code == 200
    set_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert "samesite=lax" in set_cookie.lower()
    assert "Secure" in set_cookie


def test_per_ip_generation_limit_remains_separate_from_session_ownership(tmp_path):
    service = MockVideoJobService(
        provider=MockVideoProvider(),
        generation_limit=1,
        global_limit=10,
        ledger=VideoLedger(str(tmp_path / "limits.sqlite3")),
    )
    service.provider_name = "mock"

    first = service.create(
        approved_request(service, client_id="session-a", scene_key="scene-aaa111"),
        "session-a",
        "198.51.100.40",
    )
    assert first.status == "queued"

    with pytest.raises(GenerationLimitReached, match="limit"):
        service.create(
            approved_request(service, client_id="session-b", scene_key="scene-bbb222"),
            "session-b",
            "198.51.100.40",
        )

    second = service.create(
        approved_request(service, client_id="session-c", scene_key="scene-ccc333"),
        "session-c",
        "198.51.100.41",
    )
    assert second.status == "queued"


class DurableMockVideoJobService(MockVideoJobService):
    def _ensure_provider_available(self):
        return self.provider


def test_restart_recovery_and_duplicate_scene_protection_use_session_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "false")
    ledger_path = str(tmp_path / "restart.sqlite3")
    first_service = DurableMockVideoJobService(
        provider=FakeVertexProvider(),
        ledger=VideoLedger(ledger_path, global_limit=10, per_ip_limit=10),
    )
    job = first_service.create(
        approved_request(first_service, client_id="session-restart"),
        "session-restart",
        "198.51.100.50",
    )

    restarted = DurableMockVideoJobService(
        provider=FakeVertexProvider(),
        ledger=VideoLedger(ledger_path, global_limit=10, per_ip_limit=10),
    )
    recovered = restarted.get(job.job_id, "session-restart")
    assert recovered.job_id == job.job_id
    assert recovered.status in {"generating", "completed"}

    duplicate_request = approved_request(
        restarted,
        client_id="session-restart",
        source_signature="source-999999",
        creative_intent="changed but same scene",
    )
    with pytest.raises(DuplicateSceneKind):
        restarted.create(duplicate_request, "session-restart", "203.0.113.50")