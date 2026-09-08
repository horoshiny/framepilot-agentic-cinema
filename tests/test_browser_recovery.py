import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

import pytest


playwright_sync = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright


CHROMIUM = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
SCREENPLAY = "A figure crosses the empty platform while the signal changes and rain gathers on the glass."
RECOVERY_KEY = "framepilot.video-recovery"


@pytest.fixture(scope="session")
def mock_recovery_url():
    if not CHROMIUM:
        pytest.skip("Chromium is required for browser recovery tests.")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = os.environ.copy()
    env.update(
        {
            "DEMO_MODE": "true",
            "ALLOW_VERTEX_INFERENCE": "false",
            "VIDEO_GENERATION_PROVIDER": "mock",
            "VIDEO_GENERATION_DB_PATH": ":memory:",
            "VIDEO_GENERATION_LIMIT": "20",
            "VIDEO_GENERATION_GLOBAL_LIMIT": "50",
            "VIDEO_CRITIQUE_PROVIDER": "mock",
            "ALLOW_VIDEO_CRITIQUE": "false",
            "ALLOW_VEO_GENERATION": "false",
        }
    )
    with tempfile.TemporaryDirectory() as database_dir:
        env["VIDEO_GENERATION_DB_PATH"] = os.path.join(database_dir, "video_jobs.sqlite3")
        process = subprocess.Popen(
            [
                "python3",
                "-m",
                "uvicorn",
                "app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                try:
                    with urllib.request.urlopen(f"{url}/api/health", timeout=0.2) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(0.05)
            else:
                pytest.fail("Mock recovery server did not start.")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.fixture
def recovery_page(mock_recovery_url):
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=CHROMIUM)
        context = browser.new_context(viewport={"width": 1366, "height": 768})
        page = context.new_page()
        page.goto(mock_recovery_url, wait_until="networkidle")
        try:
            yield page
        finally:
            context.close()
            browser.close()


def direct_and_snapshot(page):
    page.fill("#screenplay", SCREENPLAY)
    page.click("#direct")
    page.wait_for_function("document.querySelector('#stage')?.dataset.state === 'completed'")
    return page.evaluate(
        f"""() => JSON.parse(sessionStorage.getItem({json.dumps(RECOVERY_KEY)}))"""
    )


def store_snapshot(page, snapshot):
    page.evaluate(
        """([key, value]) => sessionStorage.setItem(key, JSON.stringify(value))""",
        [RECOVERY_KEY, snapshot],
    )


def job_payload(job_id, status, *, critique=None):
    return {
        "job_id": job_id,
        "kind": "first_cut",
        "status": status,
        "scene_key": "scene-recovery",
        "source_signature": "source-recovery",
        "provider": "mock",
        "created_at": 1,
        "updated_at": 1,
        "output": (
            {
                "url": "/static/demo-generation.mp4",
                "label": "Demo generation",
                "source": "bundled_fixture",
                "kind": "first_cut",
            }
            if status == "completed"
            else None
        ),
        "error": (
            {
                "code": "submission_unknown",
                "message": "Submission status uncertain. No retry sent; allowance remains reserved.",
            }
            if status == "submission_unknown"
            else None
        ),
        "video_critique": critique,
        "revision_approved": False,
        "deduplicated": False,
    }


def test_refresh_restores_completed_playback_and_critique_without_resubmission(recovery_page):
    page = recovery_page
    direct_and_snapshot(page)
    page.click("#confirmMotion")
    page.click("#generateFirstCut")
    page.wait_for_function("document.querySelector('#videoApprovalDialog')?.open === true")
    page.click("#confirmVideoApproval")
    page.wait_for_function(
        "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'",
        timeout=10000,
    )
    job_id = page.evaluate(
        """key => JSON.parse(sessionStorage.getItem(key)).video_job_ids.first_cut""",
        RECOVERY_KEY,
    )
    assert job_id
    stored = page.evaluate(
        """key => JSON.parse(sessionStorage.getItem(key))""",
        RECOVERY_KEY,
    )
    assert stored["version"] == 1
    assert set(stored) == {
        "version",
        "scene",
        "screenplay",
        "creative_intent",
        "direction_response",
        "video_job_ids",
    }
    serialized = json.dumps(stored)
    assert "approval_id" not in serialized
    assert "image_data_url" not in serialized
    assert "video_bytes" not in serialized

    requests = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.reload(wait_until="networkidle")
    page.wait_for_function("document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'")
    assert page.locator("#firstCutVideo").is_visible()
    assert page.locator("#generateFirstCut").is_disabled()
    assert page.locator("#analyseFirstCut").is_enabled()
    assert not page.locator("#videoApprovalDialog").evaluate("dialog => dialog.open")
    assert not page.locator("#videoCritiqueDialog").evaluate("dialog => dialog.open")
    assert not any(method == "POST" and "/api/video-jobs" in url for method, url in requests)

    page.click("#analyseFirstCut")
    page.wait_for_function("document.querySelector('#videoCritiqueDialog')?.open === true")
    page.click("#confirmVideoCritique")
    page.wait_for_function(
        "document.querySelector('#videoCritiqueStatus')?.innerText.includes('available')"
    )
    page.reload(wait_until="networkidle")
    page.wait_for_function("document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'")
    page.wait_for_function("document.querySelector('#videoCritiquePanel')?.hidden === false")
    assert page.locator("#videoCritiqueLabel").inner_text() == "AVAILABLE"
    assert page.locator("#videoCritiqueNotice").inner_text() == "Demo critique—not an analysis of your video."


def test_confirming_conservative_plan_rerenders_first_cut_with_replacement(
    recovery_page,
):
    page = recovery_page
    snapshot = direct_and_snapshot(page)
    job_id = "failed-replacement-job-123456"
    snapshot["direction_response"]["image_handle"] = "storyboard-handle-123456"
    snapshot["direction_response"]["plan"]["motion_plan"]["confirmation_required"] = True
    snapshot["video_job_ids"]["first_cut"] = job_id
    store_snapshot(page, snapshot)

    page.route(
        "**/api/health",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "ok",
                    "mode": "vertex",
                    "video_provider": "vertex",
                    "video_model": "veo-3.1-generate-001",
                    "video_estimate": "deployment-configured",
                }
            ),
        ),
    )
    page.route(
        "**/api/video-allowance",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "global_limit": 1,
                    "global_used": 1,
                    "global_remaining": 0,
                    "per_ip_limit": 1,
                    "per_ip_used": 1,
                    "per_ip_remaining": 0,
                    "director_cut_global_limit": 0,
                    "director_cut_global_used": 0,
                    "director_cut_global_remaining": 0,
                    "authorized_replacement_limit": 1,
                    "authorized_replacement_used": 0,
                    "authorized_replacement_remaining": 1,
                    "authorized_replacement_for_job_id": job_id,
                }
            ),
        ),
    )
    page.route(
        f"**/api/video-jobs/{job_id}",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    **job_payload(job_id, "failed"),
                    "provider": "vertex",
                }
            ),
        ),
    )
    page.route(
        "**/api/storyboard-images/storyboard-handle-123456",
        lambda route: route.fulfill(status=200, body=b"storyboard"),
    )

    requests = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.reload(wait_until="networkidle")
    page.wait_for_function(
        "document.querySelector('#firstCutStatus')?.innerText === 'FAILED'"
    )

    assert page.locator("#generateFirstCut").is_disabled()
    assert page.evaluate(
        """key => JSON.parse(sessionStorage.getItem(key))
        .direction_response.plan.motion_plan.confirmation_required""",
        RECOVERY_KEY,
    )
    assert page.locator("#motionPlanStatus").inner_text() == "CONFIRMATION NEEDED"
    assert page.locator("#motionConfirmation").is_visible()

    page.click("#confirmMotion")

    assert page.locator("#generateFirstCut").is_enabled()
    assert page.evaluate(
        "document.querySelector('#generateFirstCut').disabled === false"
    )
    assert page.locator("#motionPlanStatus").inner_text() == "GROUNDED"
    assert page.locator("#motionConfirmation").is_hidden()
    assert page.evaluate(
        """key => JSON.parse(sessionStorage.getItem(key))
        .direction_response.plan.motion_plan.confirmation_required""",
        RECOVERY_KEY,
    ) is False
    assert not any(
        method == "POST" and "/api/video-jobs" in url
        for method, url in requests
    )


def test_refresh_resumes_pending_polling_without_submitting_again(recovery_page):
    page = recovery_page
    snapshot = direct_and_snapshot(page)
    snapshot["video_job_ids"]["first_cut"] = "pending-job-123456"
    store_snapshot(page, snapshot)
    poll_count = {"value": 0}

    def pending_job(route):
        poll_count["value"] += 1
        status = "queued" if poll_count["value"] == 1 else "generating"
        route.fulfill(status=200, content_type="application/json", body=json.dumps(job_payload("pending-job-123456", status)))

    page.route("**/api/video-jobs/pending-job-123456", pending_job)
    requests = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.reload(wait_until="domcontentloaded", timeout=10000)
    page.wait_for_function("document.querySelector('#firstCutStatus')?.innerText === 'GENERATING'")
    time.sleep(0.8)
    assert poll_count["value"] >= 2
    assert not any(method == "POST" and "/api/video-jobs" in url for method, url in requests)
    assert page.locator("#generateFirstCut").is_disabled()
    assert page.locator("#analyseFirstCut").is_disabled()


def test_refresh_keeps_submission_unknown_blocked(recovery_page):
    page = recovery_page
    snapshot = direct_and_snapshot(page)
    snapshot["video_job_ids"]["first_cut"] = "uncertain-job-123456"
    store_snapshot(page, snapshot)

    page.route(
        "**/api/video-jobs/uncertain-job-123456",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(job_payload("uncertain-job-123456", "submission_unknown")),
        ),
    )
    requests = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.reload(wait_until="networkidle")
    page.wait_for_function("document.querySelector('#firstCutStatus')?.innerText === 'STATUS UNCERTAIN'")
    assert page.locator("#generateFirstCut").is_disabled()
    assert page.locator("#analyseFirstCut").is_disabled()
    assert not any(method == "POST" and "/api/video-jobs" in url for method, url in requests)
    assert "allowance remains reserved" in page.locator("#firstCutMessage").inner_text()


@pytest.mark.parametrize("job_id", ["missing-job-123456", "another-session-job-123456"])
def test_refresh_reports_unavailable_jobs_without_regeneration(recovery_page, job_id):
    page = recovery_page
    snapshot = direct_and_snapshot(page)
    snapshot["video_job_ids"]["first_cut"] = job_id
    store_snapshot(page, snapshot)
    page.route(
        f"**/api/video-jobs/{job_id}",
        lambda route: route.fulfill(
            status=404,
            content_type="application/json",
            body=json.dumps({"detail": {"message": "Video job was not found for this client."}}),
        ),
    )
    requests = []
    page.on("request", lambda request: requests.append((request.method, request.url)))
    page.reload(wait_until="networkidle")
    page.wait_for_function("document.querySelector('#videoRecoveryMessage')?.hidden === false")
    assert "not regenerated" in page.locator("#videoRecoveryMessage").inner_text()
    assert page.locator("#generateFirstCut").is_disabled()
    assert not any(method == "POST" and "/api/video-jobs" in url for method, url in requests)


def test_malformed_recovery_snapshot_is_discarded_safely(recovery_page):
    page = recovery_page
    page.evaluate(
        """([key]) => sessionStorage.setItem(key, '{not valid json')""",
        [RECOVERY_KEY],
    )
    page.reload(wait_until="networkidle")
    page.wait_for_function("document.querySelector('#videoRecoveryMessage')?.hidden === false")
    assert "outdated or invalid" in page.locator("#videoRecoveryMessage").inner_text()
    assert page.evaluate(
        """key => sessionStorage.getItem(key)""",
        RECOVERY_KEY,
    ) is None
    assert page.locator("#generateFirstCut").is_disabled()
    assert page.locator("#analyseFirstCut").is_disabled()