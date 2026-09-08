import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from app import app
from cinema_agent.cache import analysis_cache_key
from cinema_agent.demo import demo_plan
from cinema_agent.schemas import VideoApprovalRequest, VideoJobRequest
from cinema_agent.video_jobs import (
    ApprovalRequired,
    ApprovalInvalid,
    ActiveJobExists,
    GenerationDisabled,
    GenerationLimitReached,
    InvalidImage,
    JobNotFound,
    MockVideoJobService,
    ProviderConfigurationError,
    ReplacementCreditUnavailable,
    RevisionPrerequisiteFailed,
    StoryboardImageMissing,
    DuplicateSceneKind,
    video_cache_key,
)
from cinema_agent.video_ledger import VideoLedger
from cinema_agent.video_provider import (
    ProviderError,
    ProviderOperation,
    ProviderPoll,
    ProviderVideo,
    VertexVeoProvider,
)


@pytest.fixture(autouse=True)
def default_mock_video_provider(monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "mock")


def request(kind="first_cut", **overrides):
    screenplay = overrides.pop(
        "screenplay",
        "A figure crosses the empty platform while the signal changes and rain gathers on the glass.",
    )
    creative_intent = overrides.pop("creative_intent", "quiet uncertainty")
    shot_plan = overrides.pop("shot_plan", demo_plan(screenplay, creative_intent))
    return VideoJobRequest(
        kind=kind,
        scene_key=overrides.pop("scene_key", "scene-123456"),
        source_signature=overrides.pop("source_signature", "source-123456"),
        screenplay=screenplay,
        creative_intent=creative_intent,
        shot_plan=shot_plan,
        **overrides,
    )


def approved_request(service, kind="first_cut", client_id="client-a", **overrides):
    if service.provider_name == "vertex" and "image_handle" not in overrides:
        overrides["image_handle"] = service.register_image(
            "data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk",
            client_id,
        )
    pending = request(kind, **overrides)
    approval = service.request_approval(
        VideoApprovalRequest(**pending.model_dump(exclude={"approved", "approval_id"})),
        client_id,
    )
    return pending.model_copy(update={"approved": True, "approval_id": approval.approval_id})


def test_mock_job_transitions_queued_generating_completed_without_provider_calls():
    now = [100.0]
    service = MockVideoJobService(now=lambda: now[0])

    job = service.create(approved_request(service), "client-a")
    assert job.status == "queued"
    assert job.output is None

    generating = service.get(job.job_id, "client-a")
    assert generating.status == "generating"
    now[0] += 1
    completed = service.get(job.job_id, "client-a")
    assert completed.status == "completed"
    assert completed.output.label == "Demo generation"
    assert completed.output.source == "bundled_fixture"
    assert completed.output.kind == "first_cut"
    assert "demo-generation.mp4" in completed.output.url


def test_approval_is_required_before_job_creation():
    service = MockVideoJobService()

    with pytest.raises(ApprovalRequired):
        service.create(request(), "client-a")


def test_approval_is_bound_to_the_exact_scene_context():
    service = MockVideoJobService()
    pending = request()
    approval = service.request_approval(
        VideoApprovalRequest(**pending.model_dump(exclude={"approved", "approval_id"})),
        "client-a",
    )
    altered = pending.model_copy(
        update={
            "approved": True,
            "approval_id": approval.approval_id,
            "creative_intent": "unapproved change",
        }
    )

    with pytest.raises(ApprovalInvalid):
        service.create(altered, "client-a")


class FakeVertexProvider:
    name = "vertex"

    def __init__(self):
        self.submissions = []
        self.polls = 0

    def submit(self, request):
        self.submissions.append(request)
        return ProviderOperation("fake-vertex-operation")

    def poll(self, operation):
        self.polls += 1
        return ProviderPoll("generating" if self.polls == 1 else "completed")

    def retrieve(self, operation):
        return ProviderVideo(video_bytes=b"vertex-video", mime_type="video/mp4")


class RejectingVertexProvider(FakeVertexProvider):
    def submit(self, request):
        self.submissions.append(request)
        raise ProviderError("provider_submission_failed", "rejected before acceptance")


def seed_failed_accepted_first_cut(ledger, *, job_id="original-failed-job", client_id="client-a"):
    ledger.begin_submission(
        job_id=job_id,
        client_id=client_id,
        client_ip="127.0.0.1",
        kind="first_cut",
        scene_key="scene-123456",
        source_signature="source-123456",
        provider="vertex",
        created_at=10.0,
        request_fingerprint="original-fingerprint",
        cache_key="original-cache",
        model="veo-3.1-generate-001",
    )
    ledger.accept_submission(job_id, "original-operation", 11.0)
    ledger.update_job(
        job_id,
        status="failed",
        updated_at=12.0,
        revision_approved=False,
        output=None,
        video_path=None,
        output_reference=None,
        error_code="provider_failed",
        error_message="Video generation failed.",
        provider_error_code=None,
        provider_error_status=None,
        provider_error_category=None,
        provider_error_message=None,
    )
    return job_id


def authorize_seeded_replacement(ledger, original_job_id="original-failed-job"):
    return ledger.authorize_replacement(
        original_job_id,
        authorized_by="offline-test-authorizer",
        authorization_reason="Test-only explicit replacement authorization.",
        authorized_at=13.0,
    )


def replacement_request(service, original_job_id, *, source_signature="replacement-123456"):
    return approved_request(
        service,
        scene_key="scene-123456",
        source_signature=source_signature,
        replacement_for_job_id=original_job_id,
    )


class TrackingVertexProvider(VertexVeoProvider):
    def __init__(self):
        self.submit_calls = 0
        self.google_generation_calls = 0

        class Models:
            def __init__(self, owner):
                self.owner = owner

            def generate_videos(self, **kwargs):
                self.owner.google_generation_calls += 1
                return ProviderOperation("tracking-operation")

        super().__init__(
            client_factory=lambda: type(
                "FakeClient",
                (),
                {"models": Models(self)},
            )()
        )

    def submit(self, request):
        self.submit_calls += 1
        return super().submit(request)


@pytest.mark.parametrize(
    "configuration",
    [
        {},
        {"VEO_OUTPUT_GCS_BUCKET": "approved-bucket"},
        {"VEO_OUTPUT_GCS_PREFIX": "framepilot"},
        {"VEO_OUTPUT_GCS_BUCKET": "INVALID BUCKET", "VEO_OUTPUT_GCS_PREFIX": "framepilot"},
        {"VEO_OUTPUT_GCS_BUCKET": "approved-bucket", "VEO_OUTPUT_GCS_PREFIX": "../unsafe"},
    ],
)
def test_vertex_storage_preflight_rejects_before_approval_allowance_or_submit(
    monkeypatch, tmp_path, configuration
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.delenv("VEO_OUTPUT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("VEO_OUTPUT_GCS_PREFIX", raising=False)
    for key, value in configuration.items():
        monkeypatch.setenv(key, value)

    provider = TrackingVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    image_handle = service.register_image(
        "data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk",
        "client-a",
    )
    pending = request(image_handle=image_handle)
    approval = service.request_approval(
        VideoApprovalRequest(**pending.model_dump(exclude={"approved", "approval_id"})),
        "client-a",
    )
    approved = pending.model_copy(update={"approved": True, "approval_id": approval.approval_id})

    with pytest.raises(ProviderConfigurationError, match="VEO_OUTPUT_GCS"):
        service.create(approved, "client-a")

    assert provider.submit_calls == 0
    assert provider.google_generation_calls == 0
    assert service.allowance_status("client-a").global_used == 0
    assert approval.approval_id in service._approvals


def test_vertex_storage_preflight_passes_valid_destination_to_generation(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")

    provider = TrackingVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    job = service.create(approved_request(service), "client-a")

    assert job.status == "queued"
    assert provider.submit_calls == 1
    assert provider.google_generation_calls == 1
    assert service.allowance_status("client-a").global_used == 1


def test_effective_model_is_propagated_to_approval_job_provider_and_ledger(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.setenv("VEO_MODEL", "veo-3.1-generate-001")
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    provider = TrackingVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    pending = request(
        image_handle=service.register_image(
            "data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk",
            "client-a",
        )
    )

    approval = service.request_approval(
        VideoApprovalRequest(**pending.model_dump(exclude={"approved", "approval_id"})),
        "client-a",
    )
    approved = pending.model_copy(
        update={"approved": True, "approval_id": approval.approval_id}
    )
    job = service.create(approved, "client-a")

    assert approval.model == "veo-3.1-generate-001"
    assert job.model == "veo-3.1-generate-001"
    assert provider.model == "veo-3.1-generate-001"
    assert service.ledger.load_jobs()[0]["model"] == "veo-3.1-generate-001"


def test_retired_model_fails_before_provider_invocation_or_allowance_reservation(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.setenv("VEO_MODEL", "veo-3.0-generate-001")
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    provider = TrackingVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    pending = request(
        image_handle=service.register_image(
            "data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk",
            "client-a",
        )
    )

    with pytest.raises(ProviderConfigurationError, match="retired"):
        service.request_approval(
            VideoApprovalRequest(**pending.model_dump(exclude={"approved", "approval_id"})),
            "client-a",
        )

    assert provider.submit_calls == 0
    assert provider.google_generation_calls == 0
    assert service.allowance_status("client-a").global_used == 0
    assert service.ledger.load_jobs() == []


def test_vertex_missing_storyboard_fails_before_reservation_or_provider_submit(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    provider = TrackingVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )

    with pytest.raises(StoryboardImageMissing, match="storyboard image"):
        service.create(request(approved=True, approval_id="unused"), "client-a")

    assert provider.submit_calls == 0
    assert provider.google_generation_calls == 0
    assert service.allowance_status("client-a").global_used == 0
    assert service.ledger.load_jobs() == []


@pytest.mark.parametrize(
    "image_data_url",
    [
        "data:image/png;base64,",
        "data:image/gif;base64,ZmFrZQ==",
        "not-a-data-url",
    ],
)
def test_invalid_storyboard_data_fails_before_provider_or_allowance(
    monkeypatch, tmp_path, image_data_url
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    provider = TrackingVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )

    with pytest.raises(ProviderError, match="Storyboard image"):
        service.register_image(image_data_url, "client-a")

    assert provider.submit_calls == 0
    assert service.allowance_status("client-a").global_used == 0


def test_vertex_service_uses_fake_provider_and_retrieves_bytes(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    fake = FakeVertexProvider()
    now = [100.0]
    service = MockVideoJobService(
        provider=fake,
        poll_interval_seconds=1,
        now=lambda: now[0],
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )

    image_handle = service.register_image("data:image/png;base64,ZmFrZQ==", "client-a")
    job = service.create(approved_request(service, image_handle=image_handle), "client-a")
    assert job.provider == "vertex"
    assert len(fake.submissions) == 1
    assert fake.submissions[0].image_bytes == b"fake"
    assert fake.submissions[0].image_mime_type == "image/png"
    generating = service.get(job.job_id, "client-a")
    now[0] += 1
    completed = service.get(job.job_id, "client-a")
    assert generating.status == "generating"
    assert completed.status == "completed"
    assert completed.output.source == "vertex_veo"
    assert service.video_bytes(job.job_id, "client-a") == (b"vertex-video", "video/mp4")


def test_real_allowance_and_active_job_survive_service_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger_path = str(tmp_path / "video-ledger.sqlite3")
    now = [100.0]
    first_provider = FakeVertexProvider()
    first_service = MockVideoJobService(
        provider=first_provider,
        now=lambda: now[0],
        ledger_path=ledger_path,
    )

    job = first_service.create(approved_request(first_service), "client-a")
    allowance = first_service.allowance_status("client-a")
    assert job.status == "queued"
    assert allowance.global_used == 1
    assert allowance.global_remaining == 0
    assert allowance.per_ip_used == 1

    restarted_provider = FakeVertexProvider()
    restarted_service = MockVideoJobService(
        provider=restarted_provider,
        now=lambda: now[0],
        ledger_path=ledger_path,
    )
    recovered = restarted_service.get(job.job_id, "client-a")
    assert recovered.status == "generating"
    assert restarted_provider.submissions == []
    assert restarted_service.allowance_status("client-a").global_remaining == 0


def test_real_completed_cache_hit_after_restart_does_not_consume_or_submit(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger_path = str(tmp_path / "video-ledger.sqlite3")
    now = [100.0]
    first_service = MockVideoJobService(
        provider=FakeVertexProvider(),
        now=lambda: now[0],
        ledger_path=ledger_path,
    )
    original_request = approved_request(first_service)
    job = first_service.create(original_request, "client-a")
    first_service.get(job.job_id, "client-a")
    now[0] += 1
    completed = first_service.get(job.job_id, "client-a")
    assert completed.status == "completed"
    assert first_service.video_bytes(job.job_id, "client-a") == (b"vertex-video", "video/mp4")

    restarted_provider = FakeVertexProvider()
    restarted_service = MockVideoJobService(
        provider=restarted_provider,
        now=lambda: now[0],
        ledger_path=ledger_path,
    )
    cache_hit = restarted_service.create(
        original_request.model_copy(update={"approved": True, "approval_id": "not-needed-for-cache"}),
        "client-a",
    )
    assert cache_hit.job_id == job.job_id
    assert cache_hit.deduplicated is True
    assert restarted_provider.submissions == []
    assert restarted_service.allowance_status("client-a").global_used == 1
    assert restarted_service.video_bytes(job.job_id, "client-a") == (b"vertex-video", "video/mp4")


def test_latest_completed_job_is_scoped_to_the_current_client(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    ledger = VideoLedger(str(tmp_path / "video-ledger.sqlite3"))
    ledger.begin_submission(
        job_id="completed-client-a",
        client_id="client-a",
        client_ip="127.0.0.1",
        kind="first_cut",
        scene_key="scene-client-a",
        source_signature="source-client-a",
        provider="vertex",
        created_at=10.0,
        request_fingerprint="fingerprint-client-a",
        cache_key="cache-client-a",
        model="veo-3.1-generate-001",
    )
    ledger.accept_submission("completed-client-a", "accepted-operation", 11.0)
    ledger.update_job(
        "completed-client-a",
        status="completed",
        updated_at=12.0,
        revision_approved=False,
        output={
            "url": "/api/video-jobs/completed-client-a/video",
            "label": "Vertex AI Veo",
            "source": "vertex_veo",
            "kind": "first_cut",
        },
        video_path=None,
        output_reference="gs://approved/framepilot/client-a.mp4",
        error_code=None,
        error_message=None,
        provider_error_code=None,
        provider_error_status=None,
        provider_error_category=None,
        provider_error_message=None,
    )

    assert ledger.latest_completed_job("client-a")["job_id"] == "completed-client-a"
    assert ledger.latest_completed_job("client-b") is None


def test_real_allowance_is_atomic_across_concurrent_services(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger_path = str(tmp_path / "video-ledger.sqlite3")
    services = [
        MockVideoJobService(provider=FakeVertexProvider(), ledger_path=ledger_path),
        MockVideoJobService(provider=FakeVertexProvider(), ledger_path=ledger_path),
    ]
    requests = [
        approved_request(services[0], client_id="client-0", scene_key="scene-concurrent-a"),
        approved_request(services[1], client_id="client-1", scene_key="scene-concurrent-b"),
    ]

    def submit(index):
        try:
            return services[index].create(requests[index], "client-" + str(index))
        except GenerationLimitReached:
            return "limited"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(submit, range(2)))
    assert sum(result != "limited" for result in results) == 1
    assert services[0].ledger.allowance("client-0").global_used == 1
    assert services[0].ledger.allowance("client-0").global_remaining == 0


def test_failed_provider_submission_releases_pending_slot_without_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")

    class FailingProvider(FakeVertexProvider):
        def submit(self, request):
            self.submissions.append(request)
            raise ProviderError("provider_failed", "fake provider rejected the request")

    service = MockVideoJobService(
        provider=FailingProvider(),
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    pending = approved_request(service)
    failed = service.create(pending, "client-a")
    assert failed.status == "failed"
    assert service.allowance_status("client-a").global_used == 0
    repeated = service.create(
        pending.model_copy(update={"approved": True, "approval_id": "ignored"}),
        "client-a",
    )
    assert repeated.job_id == failed.job_id
    assert repeated.deduplicated is True


def test_timeout_after_acceptance_keeps_allowance_and_reconciles_without_resubmit(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")

    class TimeoutAfterAcceptanceProvider(FakeVertexProvider):
        def submit(self, request):
            self.submissions.append(request)
            raise ProviderError(
                "submission_unknown",
                "provider response timed out after acceptance",
                uncertain=True,
                operation_id="accepted-operation",
            )

    provider = TimeoutAfterAcceptanceProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    job = service.create(approved_request(service), "client-a")
    assert job.status == "submission_unknown"
    assert job.error.message == "Submission status uncertain. No retry sent; allowance remains reserved."
    assert service.allowance_status("client-a").global_remaining == 0
    assert len(provider.submissions) == 1

    restarted_provider = FakeVertexProvider()
    restarted_service = MockVideoJobService(
        provider=restarted_provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    reconciled = restarted_service.get(job.job_id, "client-a")
    assert reconciled.status == "generating"
    assert restarted_provider.submissions == []
    assert restarted_service.allowance_status("client-a").global_used == 1


def test_accepted_operation_still_running_after_thirty_seconds_is_not_failed(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    now = [100.0]
    provider = FakeVertexProvider()
    service = MockVideoJobService(
        provider=provider,
        now=lambda: now[0],
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )

    job = service.create(approved_request(service), "client-a")
    now[0] = 131.0
    still_running = service.get(job.job_id, "client-a")

    assert still_running.status == "generating"
    assert still_running.error is None
    assert provider.polls == 1
    assert len(provider.submissions) == 1
    assert service.allowance_status("client-a").global_used == 1


def test_restart_resumes_accepted_operation_and_completes_without_resubmit(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger_path = str(tmp_path / "video-ledger.sqlite3")
    now = [100.0]
    first_provider = FakeVertexProvider()
    first_service = MockVideoJobService(
        provider=first_provider,
        timeout_seconds=2,
        now=lambda: now[0],
        ledger_path=ledger_path,
    )

    job = first_service.create(approved_request(first_service), "client-a")
    first_service.get(job.job_id, "client-a")
    now[0] = 103.0
    timed_out = first_service.get(job.job_id, "client-a")
    assert timed_out.status == "failed"
    assert timed_out.error.code == "generation_timeout"

    resumed_provider = FakeVertexProvider()
    resumed_service = MockVideoJobService(
        provider=resumed_provider,
        now=lambda: now[0],
        ledger_path=ledger_path,
    )
    resumed = resumed_service.get(job.job_id, "client-a")
    assert resumed.status == "generating"
    assert resumed_provider.submissions == []
    assert resumed_service.allowance_status("client-a").global_used == 1

    now[0] = 104.0
    completed = resumed_service.get(job.job_id, "client-a")
    assert completed.status == "completed"
    assert completed.output.source == "vertex_veo"
    assert resumed_provider.submissions == []
    assert resumed_provider.polls == 2
    assert resumed_service.allowance_status("client-a").global_used == 1
    with resumed_service.ledger._connect() as connection:
        row = connection.execute(
            "SELECT status, operation_id FROM durable_video_jobs WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
    assert row["status"] == "completed"
    assert row["operation_id"] == "fake-vertex-operation"


def test_unknown_submission_without_operation_id_survives_restart_and_blocks_duplicates(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")

    class TimedOutProvider(FakeVertexProvider):
        def submit(self, request):
            self.submissions.append(request)
            raise TimeoutError("provider submission timed out")

    ledger_path = str(tmp_path / "video-ledger.sqlite3")
    first_service = MockVideoJobService(
        provider=TimedOutProvider(),
        ledger_path=ledger_path,
    )
    original = approved_request(first_service)
    unknown = first_service.create(original, "client-a")
    assert unknown.status == "submission_unknown"
    assert first_service.allowance_status("client-a").global_used == 1

    restarted_provider = TimedOutProvider()
    restarted_service = MockVideoJobService(
        provider=restarted_provider,
        ledger_path=ledger_path,
    )
    recovered = restarted_service.get(unknown.job_id, "client-a")
    assert recovered.status == "submission_unknown"
    assert recovered.error.message == "Submission status uncertain. No retry sent; allowance remains reserved."
    assert restarted_provider.submissions == []

    duplicate = restarted_service.create(
        original.model_copy(update={"approved": True, "approval_id": "not-needed-for-duplicate"}),
        "client-a",
    )
    assert duplicate.job_id == unknown.job_id
    assert duplicate.deduplicated is True
    assert restarted_provider.submissions == []

    altered = request(
        screenplay="A different approved screenplay keeps the same scene identity.",
        image_handle=restarted_service.register_image(
            "data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk",
            "client-a",
        ),
    )
    altered_approval = restarted_service.request_approval(
        VideoApprovalRequest(**altered.model_dump(exclude={"approved", "approval_id"})),
        "client-a",
    )
    with pytest.raises(DuplicateSceneKind):
        restarted_service.create(
            altered.model_copy(
                update={"approved": True, "approval_id": altered_approval.approval_id}
            ),
            "client-a",
        )
    assert restarted_service.allowance_status("client-a").global_remaining == 0


def test_real_director_cut_global_limit_is_zero(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    monkeypatch.setenv("MAX_REAL_VEO_GENERATIONS_GLOBAL", "2")
    monkeypatch.setenv("MAX_REAL_VEO_GENERATIONS_PER_IP", "2")
    monkeypatch.setenv("MAX_REAL_DIRECTORS_CUT_GENERATIONS_GLOBAL", "0")
    now = [100.0]
    service = MockVideoJobService(
        provider=FakeVertexProvider(),
        now=lambda: now[0],
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    first = service.create(approved_request(service), "client-a")
    service.get(first.job_id, "client-a")
    now[0] += 1
    service.get(first.job_id, "client-a")
    service.approve_revision(first.job_id, "client-a")

    with pytest.raises(GenerationLimitReached):
        service.create(
            approved_request(service, "director_cut", first_cut_job_id=first.job_id),
            "client-a",
        )
    assert service.allowance_status("client-a").director_cut_global_remaining == 0
    assert len(service.provider.submissions) == 1


def test_accepted_failed_operation_is_terminal_and_keeps_its_allowance(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")

    class FailedOperationProvider(FakeVertexProvider):
        def poll(self, operation):
            self.polls += 1
            return ProviderPoll(
                "failed",
                ProviderError(
                    "provider_failed",
                    "fake operation failed after acceptance",
                    operation_id="fake-vertex-operation",
                    provider_error_code=13,
                    provider_error_status="INTERNAL",
                    provider_error_category="BACKEND_ERROR",
                    provider_error_message="backend failure detail",
                ),
            )

    provider = FailedOperationProvider()
    service = MockVideoJobService(
        provider=provider,
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    job = service.create(approved_request(service), "client-a")
    failed = service.get(job.job_id, "client-a")
    assert failed.status == "failed"
    assert failed.error.provider_error_code == 13
    assert failed.error.provider_error_status == "INTERNAL"
    assert failed.error.provider_error_category == "BACKEND_ERROR"
    assert failed.error.provider_error_message == "backend failure detail"
    assert service.allowance_status("client-a").global_used == 1
    assert provider.polls == 1
    with service.ledger._connect() as connection:
        row = connection.execute(
            """
            SELECT operation_id, provider_error_code, provider_error_status,
                   provider_error_category, provider_error_message
            FROM durable_video_jobs
            WHERE job_id = ?
            """,
            (job.job_id,),
        ).fetchone()
    assert row["operation_id"] == "fake-vertex-operation"
    assert row["provider_error_code"] == 13
    assert row["provider_error_status"] == "INTERNAL"
    assert row["provider_error_category"] == "BACKEND_ERROR"
    assert row["provider_error_message"] == "backend failure detail"


def test_retrieval_failure_persists_operation_and_output_reference_across_restart(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")

    class RetrievalFailureProvider(FakeVertexProvider):
        def poll(self, operation):
            self.polls += 1
            return ProviderPoll(
                "completed",
                output_reference="gs://approved-bucket/framepilot/scene.mp4",
            )

        def retrieve(self, operation):
            raise ProviderError("video_retrieval_failed", "Vertex video retrieval failed.")

    ledger_path = str(tmp_path / "video-ledger.sqlite3")
    provider = RetrievalFailureProvider()
    service = MockVideoJobService(provider=provider, ledger_path=ledger_path)
    job = service.create(approved_request(service), "client-a")
    failed = service.get(job.job_id, "client-a")

    assert failed.status == "failed"
    assert service.allowance_status("client-a").global_used == 1
    with service.ledger._connect() as connection:
        row = connection.execute(
            "SELECT operation_id, output_reference, status FROM durable_video_jobs WHERE job_id = ?",
            (job.job_id,),
        ).fetchone()
    assert row["operation_id"] == "fake-vertex-operation"
    assert row["output_reference"] == "gs://approved-bucket/framepilot/scene.mp4"
    assert row["status"] == "failed"

    restarted_provider = RetrievalFailureProvider()
    restarted = MockVideoJobService(
        provider=restarted_provider,
        ledger_path=ledger_path,
    )
    restored = restarted.get(job.job_id, "client-a")
    assert restored.status == "failed"
    assert restarted_provider.submissions == []
    assert restarted_provider.polls == 0
    assert restarted.allowance_status("client-a").global_used == 1
    assert service.get(job.job_id, "client-a").status == "failed"
    assert provider.polls == 1


def test_vertex_kill_switch_and_exact_allow_value_block_before_provider_submit(monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "True")
    fake = FakeVertexProvider()
    service = MockVideoJobService(provider=fake)

    with pytest.raises(GenerationDisabled):
        service.request_approval(
            VideoApprovalRequest(**request().model_dump(exclude={"approved", "approval_id"})),
            "client-a",
        )
    assert fake.submissions == []


def test_invalid_or_cross_client_image_handles_are_rejected():
    service = MockVideoJobService()
    pending = request(image_handle="image-does-not-exist")

    with pytest.raises(InvalidImage) as missing:
        service.request_approval(
            VideoApprovalRequest(**pending.model_dump(exclude={"approved", "approval_id"})),
            "client-a",
        )
    assert "unavailable" in str(missing.value)


def test_duplicate_requests_return_one_job_and_do_not_consume_another_limit_slot():
    service = MockVideoJobService(generation_limit=1)
    first = service.create(approved_request(service), "client-a")
    duplicate = service.create(approved_request(service), "client-a")

    assert duplicate.job_id == first.job_id
    assert duplicate.deduplicated is True
    with pytest.raises(DuplicateSceneKind):
        service.create(
            approved_request(service, source_signature="another-source"),
            "client-a",
        )


def test_active_and_global_limits_reject_new_jobs_before_submission():
    service = MockVideoJobService(generation_limit=3, global_limit=1)
    first = service.create(approved_request(service), "client-a")
    second = approved_request(service, scene_key="scene-654321")

    with pytest.raises(ActiveJobExists):
        service.create(second, "client-a")

    service.get(first.job_id, "client-a")
    service.get(first.job_id, "client-a")
    with pytest.raises(GenerationLimitReached):
        service.create(
            approved_request(service, client_id="client-b", scene_key="scene-654321"),
            "client-b",
        )


def test_job_polling_and_revision_approval_are_client_scoped():
    now = [100.0]
    service = MockVideoJobService(now=lambda: now[0])
    job = service.create(approved_request(service), "client-a")

    with pytest.raises(JobNotFound):
        service.get(job.job_id, "client-b")
    with pytest.raises(RevisionPrerequisiteFailed):
        service.approve_revision(job.job_id, "client-a")

    service.get(job.job_id, "client-a")
    now[0] += 1
    completed = service.get(job.job_id, "client-a")
    approved = service.approve_revision(completed.job_id, "client-a")
    assert approved.revision_approved is True


def test_director_cut_requires_completed_approved_first_cut():
    now = [100.0]
    service = MockVideoJobService(now=lambda: now[0])
    with pytest.raises(ApprovalRequired):
        service.create(request("director_cut", approved=True, first_cut_job_id="missing"), "client-a")

    first = service.create(approved_request(service), "client-a")
    with pytest.raises(RevisionPrerequisiteFailed):
        service.create(
            approved_request(service, "director_cut", first_cut_job_id=first.job_id),
            "client-a",
        )
    service.get(first.job_id, "client-a")
    now[0] += 1
    completed = service.get(first.job_id, "client-a")
    service.approve_revision(completed.job_id, "client-a")
    director = service.create(
        approved_request(service, "director_cut", first_cut_job_id=first.job_id),
        "client-a",
    )
    assert director.kind == "director_cut"
    assert director.output is None


def test_timeout_and_kill_switch_are_explicit_failures(monkeypatch):
    now = [10.0]
    service = MockVideoJobService(timeout_seconds=2, now=lambda: now[0])
    job = service.create(approved_request(service), "client-a")
    service.get(job.job_id, "client-a")
    now[0] += 2
    failed = service.get(job.job_id, "client-a")
    assert failed.status == "failed"
    assert "deadline" in failed.error.message

    monkeypatch.setenv("VIDEO_GENERATION_KILL_SWITCH", "true")
    with pytest.raises(GenerationDisabled):
        service.create(
            approved_request(service, source_signature="kill-switch-source"),
            "client-a",
        )


def test_replacement_authorization_is_audited_and_duplicate_grants_are_idempotent(tmp_path):
    ledger = VideoLedger(str(tmp_path / "video-ledger.sqlite3"), global_limit=1, per_ip_limit=1)
    original_job_id = seed_failed_accepted_first_cut(ledger)

    first = authorize_seeded_replacement(ledger, original_job_id)
    duplicate = ledger.authorize_replacement(
        original_job_id,
        authorized_by="second-authorizer",
        authorization_reason="Must not create a second grant.",
        authorized_at=14.0,
    )

    assert first == duplicate
    assert first["state"] == "available"
    assert first["authorized_by"] == "offline-test-authorizer"
    assert ledger.allowance("client-a", "127.0.0.1").authorized_replacement_remaining == 1


def test_replacement_reservation_is_consumed_only_after_provider_acceptance(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger = VideoLedger(str(tmp_path / "video-ledger.sqlite3"), global_limit=1, per_ip_limit=1)
    original_job_id = seed_failed_accepted_first_cut(ledger)
    authorize_seeded_replacement(ledger, original_job_id)
    provider = FakeVertexProvider()
    service = MockVideoJobService(provider=provider, ledger=ledger)

    replacement = service.create(replacement_request(service, original_job_id), "client-a", "127.0.0.1")
    snapshot = ledger.allowance("client-a", "127.0.0.1")
    assert replacement.replacement_for_job_id == original_job_id
    assert snapshot.global_used == 1
    assert snapshot.global_remaining == 0
    assert snapshot.authorized_replacement_remaining == 0
    assert snapshot.authorized_replacement_used == 1

    with ledger._connect() as connection:
        submission = connection.execute(
            "SELECT state, allowance_source, replacement_for_job_id "
            "FROM video_generation_submissions WHERE job_id = ?",
            (replacement.job_id,),
        ).fetchone()
        authorization = connection.execute(
            "SELECT state, reserved_job_id "
            "FROM video_replacement_authorizations WHERE original_job_id = ?",
            (original_job_id,),
        ).fetchone()
    assert tuple(submission) == ("accepted", "authorized_replacement", original_job_id)
    assert tuple(authorization) == ("consumed", replacement.job_id)


def test_pre_acceptance_failure_releases_replacement_without_changing_normal_allowance(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger = VideoLedger(str(tmp_path / "video-ledger.sqlite3"), global_limit=1, per_ip_limit=1)
    original_job_id = seed_failed_accepted_first_cut(ledger)
    authorize_seeded_replacement(ledger, original_job_id)
    provider = RejectingVertexProvider()
    service = MockVideoJobService(provider=provider, ledger=ledger)

    failed = service.create(replacement_request(service, original_job_id), "client-a", "127.0.0.1")
    assert failed.status == "failed"
    snapshot = ledger.allowance("client-a", "127.0.0.1")
    assert snapshot.global_used == 1
    assert snapshot.global_remaining == 0
    assert snapshot.authorized_replacement_remaining == 1

    with ledger._connect() as connection:
        state = connection.execute(
            "SELECT state FROM video_generation_submissions WHERE job_id = ?",
            (failed.job_id,),
        ).fetchone()[0]
    assert state == "released"


def test_replacement_double_click_is_deduplicated_and_credit_cannot_be_granted_again(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger = VideoLedger(str(tmp_path / "video-ledger.sqlite3"), global_limit=1, per_ip_limit=1)
    original_job_id = seed_failed_accepted_first_cut(ledger)
    authorize_seeded_replacement(ledger, original_job_id)
    provider = FakeVertexProvider()
    service = MockVideoJobService(provider=provider, ledger=ledger)
    replacement = replacement_request(service, original_job_id)

    first = service.create(replacement, "client-a", "127.0.0.1")
    second = service.create(replacement, "client-a", "127.0.0.1")

    assert second.job_id == first.job_id
    assert second.deduplicated is True
    assert len(provider.submissions) == 1
    with pytest.raises(ReplacementCreditUnavailable):
        service.create(
            replacement_request(service, original_job_id, source_signature="new-replacement-123"),
            "client-a",
            "127.0.0.1",
        )


def test_concurrent_replacement_submissions_can_consume_only_one_credit(monkeypatch, tmp_path):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    ledger = VideoLedger(str(tmp_path / "video-ledger.sqlite3"), global_limit=1, per_ip_limit=1)
    original_job_id = seed_failed_accepted_first_cut(ledger)
    authorize_seeded_replacement(ledger, original_job_id)
    provider = FakeVertexProvider()
    service = MockVideoJobService(provider=provider, ledger=ledger)
    first_request = replacement_request(service, original_job_id, source_signature="replacement-a123")
    second_request = replacement_request(service, original_job_id, source_signature="replacement-b123")

    def submit(request_to_use):
        try:
            return service.create(request_to_use, "client-a", "127.0.0.1")
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(submit, (first_request, second_request)))

    assert sum(getattr(result, "status", None) == "queued" for result in results) == 1
    assert sum(
        isinstance(result, (ReplacementCreditUnavailable, ActiveJobExists))
        for result in results
    ) == 1
    assert len(provider.submissions) == 1
    assert ledger.allowance("client-a", "127.0.0.1").authorized_replacement_remaining == 0


def test_video_cache_keys_are_separate_from_analysis_and_each_video_kind():
    first = video_cache_key("first_cut", "scene-123456", "source-123456")
    director = video_cache_key("director_cut", "scene-123456", "source-123456")
    analysis = analysis_cache_key("image-data", "screenplay text", "intent")

    assert first != director
    assert first.startswith("video-job:")
    assert director.startswith("video-job:")
    assert analysis not in {first, director}


def test_video_endpoints_enforce_approval_and_revision_order(monkeypatch):
    import app as app_module

    monkeypatch.setattr(
        app_module,
        "video_job_service",
        MockVideoJobService(poll_interval_seconds=0.1),
    )
    client = TestClient(app)
    body = request(approved=False).model_dump()

    approval = client.post(
        "/api/video-jobs/approval",
        json={key: value for key, value in body.items() if key not in {"approved", "approval_id"}},
    )
    assert approval.status_code == 200
    assert approval.json()["status"] == "approval_required"

    blocked = client.post("/api/video-jobs", json=body)
    assert blocked.status_code == 400
    assert blocked.json()["detail"]["code"] == "approval_required"

    body["approved"] = True
    body["approval_id"] = approval.json()["approval_id"]
    first = client.post("/api/video-jobs", json=body)
    assert first.status_code == 200
    assert first.json()["status"] == "queued"
    job_id = first.json()["job_id"]
    statuses = []
    for _ in range(20):
        status = client.get(f"/api/video-jobs/{job_id}").json()["status"]
        statuses.append(status)
        if status == "completed":
            break
        time.sleep(0.02)
    assert statuses[0] in {"generating", "completed"}
    assert statuses[-1] == "completed"

    approved = client.post(f"/api/video-jobs/{job_id}/approve-revision")
    assert approved.status_code == 200
    assert approved.json()["revision_approved"] is True


def test_video_allowance_endpoint_exposes_only_safe_aggregate_counts(monkeypatch, tmp_path):
    import app as app_module

    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "vertex")
    monkeypatch.setenv("ALLOW_VEO_GENERATION", "true")
    service = MockVideoJobService(
        provider=FakeVertexProvider(),
        ledger_path=str(tmp_path / "video-ledger.sqlite3"),
    )
    monkeypatch.setattr(app_module, "video_job_service", service)
    client = TestClient(app)
    response = client.get("/api/video-allowance")
    assert response.status_code == 200
    assert set(response.json()) == {
        "global_limit",
        "global_used",
        "global_remaining",
        "per_ip_limit",
        "per_ip_used",
        "per_ip_remaining",
        "director_cut_global_limit",
        "director_cut_global_used",
        "director_cut_global_remaining",
        "authorized_replacement_limit",
        "authorized_replacement_used",
        "authorized_replacement_remaining",
        "authorized_replacement_for_job_id",
    }
    assert response.json()["global_remaining"] == 1