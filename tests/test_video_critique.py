import json
from types import SimpleNamespace

import pytest

from cinema_agent.demo import demo_plan
from cinema_agent.schemas import (
    VideoCritiquePayload,
    VideoEvidence,
    VideoObservation,
    VideoRevisionRecommendation,
)
from cinema_agent.video_critique import (
    DEMO_CRITIQUE_NOTICE,
    GeminiVideoCritiqueProvider,
    MockVideoCritiqueProvider,
    VideoCritiqueService,
)
from cinema_agent.video_jobs import MockVideoJobService
from cinema_agent.video_provider import ProviderVideo
from tests.test_video_jobs import approved_request


@pytest.fixture(autouse=True)
def default_mock_video_provider(monkeypatch):
    monkeypatch.setenv("VIDEO_GENERATION_PROVIDER", "mock")


def complete_first_cut(service, client_id="client-a", **overrides):
    job = service.create(approved_request(service, client_id=client_id, **overrides), client_id)
    service.get(job.job_id, client_id)
    service._next_poll_at[job.job_id] = 0
    return service.get(job.job_id, client_id)


def supported_payload(timestamp=1.5):
    evidence = [VideoEvidence(timestamp_seconds=timestamp, description="Visible test evidence.", certainty="high")]
    observation = VideoObservation(
        finding="The requested camera move is visible.",
        support="supported",
        evidence=evidence,
        uncertainty="The finding is limited to the supplied frame sequence.",
    )
    recommendation = VideoRevisionRecommendation(
        issue="The supported pacing issue remains concentrated at the cut.",
        recommendation="Hold the approved opening beat slightly longer.",
        support="supported",
        evidence=evidence,
        uncertainty="A revision is suggested, not guaranteed to improve the result.",
    )
    return VideoCritiquePayload(
        requested_actions_achieved=[observation],
        requested_actions_missing=[],
        character_object_consistency=[observation],
        camera_movement_pacing=[observation],
        visible_artifacts=[],
        emotional_intent_alignment=[observation],
        revision_recommendations=[recommendation],
        uncertainty="Timestamped evidence was available for this mocked response.",
    )


class CountingCritic:
    name = "mock"

    def __init__(self):
        self.calls = []

    def analyse(self, **kwargs):
        self.calls.append(kwargs)
        return supported_payload()


def test_mock_video_critique_is_explicitly_not_a_video_analysis():
    jobs = MockVideoJobService()
    completed = complete_first_cut(jobs)
    result = VideoCritiqueService(provider=MockVideoCritiqueProvider()).analyse_first_cut(
        jobs, completed.job_id, "client-a"
    )

    assert result.status == "available"
    assert result.source == "demo_fixture"
    assert result.notice == DEMO_CRITIQUE_NOTICE
    assert all(
        finding.support == "not_observed"
        for group in (
            result.requested_actions_achieved,
            result.requested_actions_missing,
            result.character_object_consistency,
            result.camera_movement_pacing,
            result.visible_artifacts,
            result.emotional_intent_alignment,
        )
        for finding in group
    )


def test_completed_video_bytes_and_storyboard_are_passed_to_live_provider_without_calling_google(
    monkeypatch,
):
    monkeypatch.setenv("ALLOW_VIDEO_CRITIQUE", "true")
    jobs = MockVideoJobService()
    image_handle = jobs.register_image("data:image/png;base64,ZmFrZS1zdG9yeWJvYXJk", "client-a")
    completed = complete_first_cut(jobs, image_handle=image_handle)
    critic = CountingCritic()
    result = VideoCritiqueService(provider=critic).analyse_first_cut(
        jobs, completed.job_id, "client-a"
    )

    assert result.status == "available"
    assert len(critic.calls) == 1
    assert critic.calls[0]["video_bytes"]
    assert critic.calls[0]["storyboard_bytes"] == b"fake-storyboard"
    assert critic.calls[0]["screenplay"] == completed_context(jobs, completed.job_id)["screenplay"]


def completed_context(jobs, job_id):
    return jobs.get_critique_context(job_id, "client-a")


def test_video_critique_cache_avoids_duplicate_provider_calls():
    jobs = MockVideoJobService()
    completed = complete_first_cut(jobs)
    critic = CountingCritic()
    service = VideoCritiqueService(provider=critic)

    first = service.analyse_first_cut(jobs, completed.job_id, "client-a")
    second = service.analyse_first_cut(jobs, completed.job_id, "client-a")

    assert first.content_hash == second.content_hash
    assert first.source_plan_hash == second.source_plan_hash
    assert len(critic.calls) == 1


@pytest.mark.parametrize(
    "provider",
    [
        type(
            "MalformedCritic",
            (),
            {"name": "mock", "analyse": lambda self, **kwargs: {"not": "the schema"}},
        )(),
        type(
            "BadTimestampCritic",
            (),
            {"name": "mock", "analyse": lambda self, **kwargs: supported_payload(timestamp=99)},
        )(),
    ],
)
def test_malformed_or_out_of_range_critique_is_unavailable(provider):
    jobs = MockVideoJobService()
    completed = complete_first_cut(jobs)
    result = VideoCritiqueService(provider=provider).analyse_first_cut(
        jobs, completed.job_id, "client-a"
    )
    assert result.status == "unavailable"
    assert result.message == "Video critique unavailable."


def test_missing_video_is_unavailable_without_claiming_observations():
    class MissingVideoJobs(MockVideoJobService):
        def video_bytes(self, job_id, client_id):
            raise FileNotFoundError("fixture missing")

    jobs = MissingVideoJobs()
    completed = complete_first_cut(jobs)
    result = VideoCritiqueService().analyse_first_cut(jobs, completed.job_id, "client-a")
    assert result.status == "unavailable"
    assert result.message == "Video critique unavailable."


def test_live_critique_is_disabled_and_cap_is_enforced(monkeypatch):
    monkeypatch.setenv("ALLOW_VIDEO_CRITIQUE", "false")
    jobs = MockVideoJobService()
    completed = complete_first_cut(jobs)
    critic = CountingCritic()
    service = VideoCritiqueService(provider=type("LiveCritic", (CountingCritic,), {"name": "vertex"})(), critique_cap=1)
    disabled = service.analyse_first_cut(jobs, completed.job_id, "client-a")
    assert disabled.status == "unavailable"
    assert len(service.provider.calls) == 0

    monkeypatch.setenv("ALLOW_VIDEO_CRITIQUE", "true")
    service.provider = critic
    service.provider_name = "vertex"
    available = service.analyse_first_cut(jobs, completed.job_id, "client-a")
    assert available.status == "available"
    assert service._successful_live_critiques == 1

    second = complete_first_cut(
        jobs,
        client_id="client-b",
        scene_key="scene-654321",
        source_signature="source-654321",
        creative_intent="different intent",
    )
    capped = service.analyse_first_cut(jobs, second.job_id, "client-b")
    assert capped.status == "unavailable"
    assert len(critic.calls) == 1


class FakeCritiqueModels:
    def __init__(self):
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=json.dumps(supported_payload().model_dump()))


class FakeCritiqueClient:
    def __init__(self):
        self.models = FakeCritiqueModels()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_gemini_provider_sends_video_storyboard_context_and_output_bound():
    fake = FakeCritiqueClient()
    plan = demo_plan("A cyclist waits beneath a blue awning before the ferry arrives.", "patient suspense")
    provider = GeminiVideoCritiqueProvider(client_factory=lambda: fake, model="test-gemini")

    result = provider.analyse(
        video_bytes=b"actual-mp4",
        video_mime_type="video/mp4",
        storyboard_bytes=b"actual-storyboard",
        storyboard_mime_type="image/png",
        screenplay="A cyclist waits beneath a blue awning before the ferry arrives.",
        creative_intent="patient suspense",
        shot_plan=plan,
    )

    call = fake.models.calls[0]
    assert result.revision_recommendations
    assert call["model"] == "test-gemini"
    assert call["config"].max_output_tokens == 2200
    assert len(call["contents"]) == 3
    assert call["contents"][1].inline_data.data == b"actual-mp4"
    assert call["contents"][2].inline_data.data == b"actual-storyboard"


def test_directors_cut_prompt_uses_only_supported_video_findings():
    from cinema_agent.video_provider import build_veo_prompt

    plan = demo_plan("A cyclist waits beneath a blue awning before the ferry arrives.", "patient suspense")
    critique = supported_payload()
    prompt = build_veo_prompt(
        "A cyclist waits beneath a blue awning before the ferry arrives.",
        "patient suspense",
        plan,
        "director_cut",
        critique.model_copy(
            update={
                "status": "available",
                "source": "gemini_video",
                "label": "Gemini video critique",
                "message": "Video critique complete.",
            }
        ),
    )

    assert "Hold the approved opening beat slightly longer." in prompt
    assert "No supported video-critique finding" not in prompt
    assert "do not guarantee" in prompt.lower()