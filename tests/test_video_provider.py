from types import SimpleNamespace

import pytest

from cinema_agent.demo import demo_plan
from cinema_agent.schemas import VideoJobRequest
from cinema_agent.video_provider import (
    DEFAULT_VEO_MODEL,
    ProviderError,
    ProviderOperation,
    ProviderPoll,
    ProviderVideo,
    VertexVeoProvider,
    VideoProviderRequest,
    build_veo_prompt,
    effective_veo_model,
    validate_prompt_enhancement,
    veo_estimate_label,
)


@pytest.fixture(autouse=True)
def approved_output_configuration(monkeypatch):
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")


class FakeModels:
    def __init__(self):
        self.calls = []
        self.operation = SimpleNamespace(name="operations/scene-1", done=False, error=None)

    def generate_videos(self, **kwargs):
        self.calls.append(kwargs)
        return self.operation


class FakeOperations:
    def __init__(self, models):
        self.models = models
        self.poll_count = 0

    def get(self, operation):
        self.poll_count += 1
        if self.poll_count == 1:
            return SimpleNamespace(name=operation.name, done=False, error=None)
        return SimpleNamespace(
            name=operation.name,
            done=True,
            error=None,
            response=SimpleNamespace(
                generated_videos=[
                    SimpleNamespace(video=SimpleNamespace(uri="gs://approved-bucket/framepilot/scene.mp4"))
                ]
            ),
        )


class FakeClient:
    def __init__(self):
        self.models = FakeModels()
        self.operations = FakeOperations(self.models)


def test_vertex_default_model_is_current_and_override_is_supported(monkeypatch):
    monkeypatch.delenv("VEO_MODEL", raising=False)
    assert effective_veo_model() == DEFAULT_VEO_MODEL
    assert VertexVeoProvider(client_factory=lambda: FakeClient()).model == DEFAULT_VEO_MODEL

    monkeypatch.setenv("VEO_MODEL", "veo-custom-model")
    assert effective_veo_model() == "veo-custom-model"
    assert VertexVeoProvider(client_factory=lambda: FakeClient()).model == "veo-custom-model"


def test_veo_estimate_label_only_uses_configured_numeric_cost(monkeypatch):
    monkeypatch.delenv("VEO_ESTIMATED_COST_INR", raising=False)
    assert veo_estimate_label() is None

    monkeypatch.setenv("VEO_ESTIMATED_COST_INR", "1250")
    assert veo_estimate_label() == "Approximate veo-3.1-generate-001 estimate: ₹1,250.00"
def provider_request(plan=None):
    return VideoProviderRequest(
        kind="first_cut",
        prompt=build_veo_prompt(
            "A cyclist waits beneath a blue awning as the ferry horn sounds.",
            "patient suspense",
            plan or demo_plan(
                "A cyclist waits beneath a blue awning as the ferry horn sounds.",
                "patient suspense",
            ),
            "first_cut",
        ),
        image_bytes=b"storyboard-image",
        image_mime_type="image/png",
        duration_seconds=8,
    )


def test_vertex_submission_attaches_image_and_uses_safe_video_settings():
    fake = FakeClient()
    provider = VertexVeoProvider(client_factory=lambda: fake, model="test-veo-model")

    operation = provider.submit(provider_request())

    call = fake.models.calls[0]
    assert operation.operation_id == "operations/scene-1"
    assert call["model"] == "test-veo-model"
    assert call["prompt"].startswith("Create a restrained first cut")
    assert call["image"].image_bytes == b"storyboard-image"
    assert call["image"].mime_type == "image/png"
    assert call["config"].number_of_videos == 1
    assert call["config"].duration_seconds == 8
    assert call["config"].aspect_ratio == "16:9"
    assert call["config"].generate_audio is False
    assert call["config"].output_gcs_uri == "gs://approved-bucket/framepilot/"
    serialized_config = call["config"].model_dump(exclude_none=True)
    assert "enhance_prompt" not in serialized_config
    assert serialized_config == {
        "number_of_videos": 1,
        "duration_seconds": 8,
        "aspect_ratio": "16:9",
        "generate_audio": False,
        "output_gcs_uri": "gs://approved-bucket/framepilot/",
    }


def test_veo3_preflight_rejects_explicit_prompt_enhancement_disable():
    with pytest.raises(ProviderError, match="must use the provider default"):
        validate_prompt_enhancement("veo-3.1-generate-001", False)

    validate_prompt_enhancement("veo-3.1-generate-001", None)
    validate_prompt_enhancement("veo-2.0-generate-001", False)


def test_vertex_submission_logs_bounded_diagnostics_without_payloads(monkeypatch, caplog):
    class FakeGoogleError(Exception):
        code = 400
        status = "INVALID_ARGUMENT"
        reason = "INVALID_ARGUMENT"
        message = (
            "bad storyboard data:image/png;base64,"
            + ("A" * 220)
            + " authorization=secret-token"
        )

    class FailingModels:
        def generate_videos(self, **kwargs):
            raise FakeGoogleError()

    class FailingClient:
        models = FailingModels()

    provider = VertexVeoProvider(
        client_factory=lambda: FailingClient(),
        model="test-veo-model",
    )
    with caplog.at_level("WARNING", logger="cinema_agent.video_provider"):
        with pytest.raises(ProviderError, match="Vertex video submission failed"):
            provider.submit(provider_request())

    assert "exception_type=FakeGoogleError" in caplog.text
    assert "http_status=400" in caplog.text
    assert "google_status=INVALID_ARGUMENT" in caplog.text
    assert "error_code=INVALID_ARGUMENT" in caplog.text
    assert "model=test-veo-model" in caplog.text
    assert "image_supplied=True" in caplog.text
    assert "image_mime_type=image/png" in caplog.text
    assert "duration_seconds=8" in caplog.text
    assert "aspect_ratio=16:9" in caplog.text
    assert "output_count=1" in caplog.text
    assert "generate_audio=False" in caplog.text
    assert "secret-token" not in caplog.text
    assert "data:image/png;base64" not in caplog.text
    assert "storyboard-image" not in caplog.text


class FakeStorageResponse:
    status_code = 200
    content = b"fake-mp4"

    def raise_for_status(self):
        return None


class FakeStorageSession:
    def __init__(self, response=None, error=None):
        self.response = response or FakeStorageResponse()
        self.error = error
        self.calls = []

    def get(self, url, timeout):
        self.calls.append((url, timeout))
        if self.error:
            raise self.error
        return self.response


def test_vertex_operation_polling_and_gcs_retrieval_are_async_and_terminal(monkeypatch):
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    fake = FakeClient()
    storage = FakeStorageSession()
    provider = VertexVeoProvider(
        client_factory=lambda: fake,
        storage_session_factory=lambda: storage,
    )
    operation = provider.submit(provider_request())

    assert provider.poll(operation) == ProviderPoll("generating")
    assert provider.poll(operation) == ProviderPoll(
        "completed",
        output_reference="gs://approved-bucket/framepilot/scene.mp4",
    )
    output = provider.retrieve(operation)

    assert output == ProviderVideo(video_bytes=b"fake-mp4", mime_type="video/mp4")
    assert storage.calls == [
        (
            "https://storage.googleapis.com/storage/v1/b/approved-bucket/o/framepilot%2Fscene.mp4?alt=media",
            30,
        )
    ]


def _terminal_error_provider(error):
    class Operations:
        def get(self, operation):
            return SimpleNamespace(name=operation.name, done=True, error=error)

    class Client:
        operations = Operations()

    return VertexVeoProvider(client_factory=lambda: Client())


def test_vertex_terminal_google_style_error_preserves_safe_metadata(caplog):
    provider = _terminal_error_provider(
        SimpleNamespace(
            code=13,
            status="INTERNAL",
            reason="BACKEND_ERROR",
            message="backend rejected the completed operation",
            details=[
                {
                    "@type": "type.googleapis.com/google.rpc.ErrorInfo",
                    "reason": "BACKEND_ERROR",
                    "domain": "generativelanguage.googleapis.com",
                    "metadata": {"request_id": "do-not-log"},
                }
            ],
        )
    )

    with caplog.at_level("WARNING", logger="cinema_agent.video_provider"):
        poll = provider.poll(ProviderOperation("operations/google-style"))

    assert poll.status == "failed"
    assert poll.error.code == "provider_failed"
    assert poll.error.operation_id == "operations/google-style"
    assert poll.error.provider_error_type == "SimpleNamespace"
    assert poll.error.provider_error_code == 13
    assert poll.error.provider_error_status == "INTERNAL"
    assert poll.error.provider_error_category == "BACKEND_ERROR"
    assert poll.error.provider_error_message == "backend rejected the completed operation"
    assert "error_type=SimpleNamespace" in caplog.text
    assert "provider_error_code=13" in caplog.text
    assert "provider_error_status=INTERNAL" in caplog.text
    assert "provider_error_category=BACKEND_ERROR" in caplog.text
    assert "detail_types=type.googleapis.com/google.rpc.ErrorInfo" in caplog.text
    assert "detail_fields=@type,reason,domain,metadata" in caplog.text
    assert "do-not-log" not in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        {
            "code": 7,
            "status": "PERMISSION_DENIED",
            "message": "output bucket permission denied",
            "details": [
                {
                    "type": "google.rpc.ErrorInfo",
                    "reason": "OUTPUT_BUCKET_PERMISSION",
                    "field": "output_gcs_uri",
                }
            ],
        },
        type(
            "ModelError",
            (),
            {
                "model_dump": lambda self, exclude_none=True: {
                    "code": 400,
                    "status": "INVALID_ARGUMENT",
                    "message": "invalid duration",
                    "details": [
                        {
                            "@type": "google.rpc.BadRequest",
                            "field": "duration_seconds",
                        }
                    ],
                }
            },
        )(),
    ],
)
def test_vertex_terminal_dictionary_and_model_error_shapes_are_supported(error):
    provider = _terminal_error_provider(error)
    poll = provider.poll(ProviderOperation("operations/shaped-error"))

    assert poll.status == "failed"
    assert poll.error.provider_error_code in {7, 400}
    assert poll.error.provider_error_status in {"PERMISSION_DENIED", "INVALID_ARGUMENT"}
    assert poll.error.provider_error_message in {
        "output bucket permission denied",
        "invalid duration",
    }
    assert poll.error.provider_error_category in {
        "OUTPUT_BUCKET_PERMISSION",
        "unsupported_configuration",
    }


def test_vertex_terminal_error_without_details_is_bounded_and_classified():
    provider = _terminal_error_provider(
        SimpleNamespace(
            code=8,
            status="RESOURCE_EXHAUSTED",
            message="capacity is temporarily unavailable",
        )
    )

    poll = provider.poll(ProviderOperation("operations/no-details"))

    assert poll.error.provider_error_code == 8
    assert poll.error.provider_error_status == "RESOURCE_EXHAUSTED"
    assert poll.error.provider_error_category == "quota_capacity"
    assert poll.error.provider_error_message == "capacity is temporarily unavailable"


def test_vertex_terminal_error_redacts_sensitive_message_content(caplog):
    secret = "secret-token"
    long_payload = "A" * 220
    provider = _terminal_error_provider(
        SimpleNamespace(
            code=400,
            status="INVALID_ARGUMENT",
            message=(
                "bad image data:image/png;base64,"
                + long_payload
                + f" authorization={secret}"
            ),
        )
    )

    with caplog.at_level("WARNING", logger="cinema_agent.video_provider"):
        poll = provider.poll(ProviderOperation("operations/redacted"))

    assert secret not in caplog.text
    assert "data:image/png;base64" not in caplog.text
    assert long_payload not in caplog.text
    assert "[redacted image data]" in poll.error.provider_error_message
    assert "authorization=[redacted]" in poll.error.provider_error_message
    assert "[redacted long payload]" not in poll.error.provider_error_message


def test_vertex_retrieval_prefers_inline_video_bytes(monkeypatch):
    fake = FakeClient()
    fake.operations = SimpleNamespace(
        get=lambda operation: SimpleNamespace(
            name=operation.name,
            done=True,
            error=None,
            response=SimpleNamespace(
                generated_videos=[
                    SimpleNamespace(
                        video=SimpleNamespace(
                            uri="gs://unapproved-bucket/anywhere.mp4",
                            video_bytes=b"inline-mp4",
                        )
                    )
                ]
            ),
        )
    )
    storage = FakeStorageSession(error=AssertionError("GCS must not be called for inline bytes"))
    provider = VertexVeoProvider(
        client_factory=lambda: fake,
        storage_session_factory=lambda: storage,
    )
    operation = provider.submit(provider_request())
    monkeypatch.delenv("VEO_OUTPUT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("VEO_OUTPUT_GCS_PREFIX", raising=False)
    assert provider.poll(operation) == ProviderPoll(
        "completed",
        output_reference="gs://unapproved-bucket/anywhere.mp4",
    )
    assert provider.retrieve(operation) == ProviderVideo(
        video_bytes=b"inline-mp4",
        mime_type="video/mp4",
    )


@pytest.mark.parametrize(
    "uri",
    [
        "https://storage.googleapis.com/approved-bucket/framepilot/scene.mp4",
        "gs://other-bucket/framepilot/scene.mp4",
        "gs://approved-bucket/other/scene.mp4",
        "gs://approved-bucket/framepilot/../scene.mp4",
        "gs://approved-bucket/framepilot/scene.mp4?alt=media",
    ],
)
def test_vertex_retrieval_rejects_unapproved_gcs_uris(monkeypatch, uri):
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    fake = FakeClient()
    fake.operations = SimpleNamespace(
        get=lambda operation: SimpleNamespace(
            name=operation.name,
            done=True,
            error=None,
            response=SimpleNamespace(
                generated_videos=[SimpleNamespace(video=SimpleNamespace(uri=uri))]
            ),
        )
    )
    storage = FakeStorageSession()
    provider = VertexVeoProvider(
        client_factory=lambda: fake,
        storage_session_factory=lambda: storage,
    )
    operation = provider.submit(provider_request())
    provider.poll(operation)
    with pytest.raises(ProviderError) as error:
        provider.retrieve(operation)
    assert error.value.code == "video_retrieval_failed"
    assert "unapproved" in str(error.value)
    assert storage.calls == []


def test_vertex_retrieval_reports_missing_approved_storage_configuration(monkeypatch):
    fake = FakeClient()
    provider = VertexVeoProvider(
        client_factory=lambda: fake,
        storage_session_factory=lambda: FakeStorageSession(),
    )
    operation = provider.submit(provider_request())
    monkeypatch.delenv("VEO_OUTPUT_GCS_BUCKET", raising=False)
    monkeypatch.delenv("VEO_OUTPUT_GCS_PREFIX", raising=False)
    provider.poll(operation)
    provider.poll(operation)
    with pytest.raises(ProviderError, match="Approved Vertex output GCS bucket and prefix"):
        provider.retrieve(operation)


def test_vertex_retrieval_sanitizes_storage_permission_failures(monkeypatch):
    monkeypatch.setenv("VEO_OUTPUT_GCS_BUCKET", "approved-bucket")
    monkeypatch.setenv("VEO_OUTPUT_GCS_PREFIX", "framepilot")
    fake = FakeClient()
    storage = FakeStorageSession(error=RuntimeError("403 permission denied credential=secret"))
    provider = VertexVeoProvider(
        client_factory=lambda: fake,
        storage_session_factory=lambda: storage,
    )
    operation = provider.submit(provider_request())
    provider.poll(operation)
    provider.poll(operation)
    with pytest.raises(ProviderError) as error:
        provider.retrieve(operation)
    assert str(error.value) == "Vertex video retrieval failed."
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    ("screenplay", "mood", "subject"),
    [
        (
            "A botanist labels a fragile seedling in a glasshouse before dawn.",
            "tender precision",
            "the visual center of the supplied frame",
        ),
        (
            "A mechanic closes the shutter while a red tram passes through the rain.",
            "compressed urgency",
            "the visual center of the supplied frame",
        ),
        (
            "A diver pauses beneath an ice shelf as a distant whale turns away.",
            "vast solitude",
            "the visual center of the supplied frame",
        ),
    ],
)
def test_prompt_is_dynamic_across_unrelated_scene_fixtures(screenplay, mood, subject):
    plan = demo_plan(screenplay, mood)
    prompt = build_veo_prompt(screenplay, mood, plan, "director_cut")

    assert screenplay in prompt
    assert mood in prompt
    assert subject in prompt
    assert "PRESERVE EXACTLY" in prompt
    assert "PROHIBITED ADDITIONS OR CHANGES" in prompt
    assert "observatory" not in prompt.lower()


def test_prompt_includes_all_screenplay_grounded_dark_scene_actions():
    screenplay = (
        "The astronomer slowly raises the lantern from waist to shoulder height; feet remain planted. "
        "The cloak hem moves gently in the breeze. Portal rings rotate slowly. "
        "One controlled portal pulse only in the final third. Mist drifts through the ruins. "
        "Subtle ripples move across the floodwater and portal reflection. "
        "Camera: slow slide right with a restrained push-in. "
        "Preserve the astronomer's identity and silhouette, lantern, architecture, portal structure, "
        "composition and blue-hour lighting. No walking, new subjects, cuts, abrupt motion, "
        "object replacement or major deformation."
    )
    plan = demo_plan(screenplay, "quiet dread")
    prompt = build_veo_prompt(screenplay, "quiet dread", plan, "first_cut")

    for action in (
        "raises the lantern",
        "cloak hem moves",
        "portal rings rotate",
        "portal pulse",
        "mist drifts",
        "ripples move across",
        "slide right",
    ):
        assert action.lower() in prompt.lower()
    assert "grounding source: screenplay" in prompt
    assert "Execute the bounded screenplay candidates exactly after approval" in prompt


def test_provider_sanitizes_poll_and_retrieval_failures():
    class BrokenOperations:
        def get(self, operation):
            raise RuntimeError("credential=secret image=bytes")

    fake = FakeClient()
    fake.operations = BrokenOperations()
    provider = VertexVeoProvider(client_factory=lambda: fake)
    operation = provider.submit(provider_request())

    with pytest.raises(ProviderError) as error:
        provider.poll(operation)
    assert str(error.value) == "Vertex video polling failed."
    assert "secret" not in str(error.value)
    assert "bytes" not in str(error.value)