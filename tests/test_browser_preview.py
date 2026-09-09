import os
import shutil
import json
import time
from pathlib import Path

import pytest


playwright_sync = pytest.importorskip("playwright.sync_api")
from playwright.sync_api import sync_playwright

from cinema_agent.demo import demo_critique, demo_plan


FIXTURE = Path(__file__).parent / "fixtures" / "storyboard.svg"
PORTRAIT_FIXTURE = Path(__file__).parent / "fixtures" / "castors-day-portrait.svg"
SQUARE_FIXTURE = Path(__file__).parent / "fixtures" / "square-storyboard.svg"
BASE_URL = os.getenv("FRAMEPILOT_BASE_URL", "http://127.0.0.1:5000")
SCREENPLAY = "A figure crosses the empty platform while the signal changes and rain gathers on the glass."
OBSERVATORY_SCREENPLAY = """EXT. FLOODED CELESTIAL OBSERVATORY — BLUE HOUR

A cloaked astronomer stands on a circular stone platform above still floodwater. A massive brass astrolabe frames the foreground. Beyond the ruined towers, a luminous celestial portal hangs in the sky.

The astronomer slowly raises an amber lantern from waist to shoulder height, keeping both feet planted. A breeze lifts the cloak’s hem. Mist drifts between the ruins. The portal’s concentric rings turn slowly, releasing one gentle pulse of light. Small ripples carry its reflection across the water. The astronomer holds the lantern steady as the light fades."""
CASTORS_SCREENPLAY = """EXT. FAMILY GARDEN — GOLDEN-HOUR EVENING

Four brothers gather around a paper crown beneath a flowering tree. Castor lifts the crown as the others lean closer, laughing while warm light moves through the leaves."""


def direction_payload(*, provider="vertex"):
    plan = demo_plan(SCREENPLAY, "rising dread")
    return {
        "mode": provider,
        "analysis_source": (
            "vertex_multimodal" if provider == "vertex" else "deterministic_fallback"
        ),
        "image_handle": "image-test" if provider == "vertex" else None,
        "depth_source": "heuristic",
        "plan": plan.model_dump(mode="json"),
        "critique": demo_critique(plan).model_dump(mode="json"),
        "routing": {
            "classification": "GENERATIVE_VIDEO_REQUIRED",
            "rationale": "The requested subject motion requires generative video.",
            "matched_actions": ["character movement"],
        },
        "activity": [
            {"step": "ANALYSE", "detail": "Screenplay analysed."},
            {"step": "ROUTE", "detail": "Generative video required."},
            {"step": "DIRECT", "detail": "Direction ready."},
        ],
        "camera_grammar": "balanced",
        "shot_signature": None,
        "diversity_adjustment": None,
        "revision_camera_grammar": None,
        "revision_shot_signature": None,
        "revision_diversity_adjustment": None,
    }


def floating_market_direction_payload():
    direction = direction_payload(provider="vertex")
    direction["scene_key"] = "scene-floating-ocean-market"
    direction["source_signature"] = "source-floating-ocean-market"
    direction["image_handle"] = "image-floating-market"
    direction["plan"]["scene_summary"] = (
        "A floating ocean market drifts between bright boats and quiet waves."
    )
    direction["plan"]["motion_plan"]["confirmation_required"] = False
    direction["plan"]["motion_plan"]["visible_characters"] = [
        {"label": "market vendor"},
        {"label": "blue robot"},
        {"label": "courier"},
    ]
    direction["plan"]["motion_plan"]["visible_objects"] = [
        {"label": "blue robot"},
        {"label": "fruit"},
        {"label": "crates"},
        {"label": "tools"},
        {"label": "courier bag"},
    ]
    return direction


def allowance_payload(
    *,
    remaining=1,
    authorized_replacement_remaining=0,
    authorized_replacement_for_job_id=None,
):
    return {
        "global_limit": 1,
        "global_used": 1 - remaining,
        "global_remaining": remaining,
        "per_ip_limit": 1,
        "per_ip_used": 1 - remaining,
        "per_ip_remaining": remaining,
        "director_cut_global_limit": 0,
        "director_cut_global_used": 0,
        "director_cut_global_remaining": 0,
        "authorized_replacement_limit": 1 if authorized_replacement_for_job_id else 0,
        "authorized_replacement_used": (
            1 - authorized_replacement_remaining
            if authorized_replacement_for_job_id
            else 0
        ),
        "authorized_replacement_remaining": authorized_replacement_remaining,
        "authorized_replacement_for_job_id": authorized_replacement_for_job_id,
    }


def video_job_payload(
    status="completed",
    *,
    provider="vertex",
    job_id="job-test",
    kind="first_cut",
    revision_approved=False,
    output_url=None,
    scene_snapshot=None,
):
    completed = status == "completed"
    snapshot_scene_key = (scene_snapshot or {}).get("scene_key", "scene-test")
    snapshot_source_signature = (scene_snapshot or {}).get(
        "source_signature",
        "source-test",
    )
    return {
        "job_id": job_id,
        "kind": kind,
        "status": status,
        "scene_key": snapshot_scene_key,
        "source_signature": snapshot_source_signature,
        "provider": provider,
        "created_at": 1,
        "updated_at": 1,
        "output": (
            {
                "url": output_url or "/static/demo-generation.mp4",
                "label": "Vertex AI Veo" if provider == "vertex" else "Demo generation",
                "source": "vertex_veo" if provider == "vertex" else "bundled_fixture",
                "kind": kind,
            }
            if completed
            else None
        ),
        "error": (
            {
                "code": "submission_unknown",
                "message": "Submission status uncertain. No retry sent; allowance remains reserved.",
            }
            if status == "submission_unknown"
            else {"code": "provider_failed", "message": "Video generation failed."}
            if status == "failed"
            else None
        ),
        "video_critique": None,
        "revision_approved": revision_approved,
        "deduplicated": False,
        "replacement_for_job_id": None,
        **({"scene_snapshot": scene_snapshot} if scene_snapshot else {}),
    }


def mock_video_api(
    page,
    *,
    provider="vertex",
    remaining=1,
    authorized_replacement_remaining=0,
    authorized_replacement_for_job_id=None,
    job_status="completed",
    video_estimate="Approximate veo-3.1-generate-001 estimate: deployment-configured",
    latest_job=None,
    job_payloads=None,
):
    counts = {"approval": 0, "submission": 0}
    direction = direction_payload(provider=provider)

    page.route(
        "**/api/health",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "status": "ok",
                    "mode": provider,
                    "video_provider": provider,
                    "video_model": "veo-3.1-generate-001",
                    "video_estimate": video_estimate,
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
                allowance_payload(
                    remaining=remaining,
                    authorized_replacement_remaining=authorized_replacement_remaining,
                    authorized_replacement_for_job_id=authorized_replacement_for_job_id,
                )
            ),
        ),
    )
    page.route(
        "**/api/direct",
        lambda route: route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(direction),
        ),
    )

    def video_route(route):
        request = route.request
        path = request.url.split("?", 1)[0]
        if request.method == "POST" and path.endswith("/api/video-jobs/approval"):
            counts["approval"] += 1
            body = json.loads(request.post_data or "{}")
            kind = body.get("kind", "first_cut")
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "kind": kind,
                        "scene_key": "scene-test",
                        "source_signature": "source-test",
                        "first_cut_job_id": body.get("first_cut_job_id"),
                        "replacement_for_job_id": body.get("replacement_for_job_id"),
                        "status": "approval_required",
                        "approval_id": f"approval-{kind}",
                        "expires_at": 9999999999,
                    }
                ),
            )
            return
        if request.method == "GET" and path.endswith("/api/video-jobs/latest"):
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(latest_job),
            )
            return
        if (
            request.method == "GET"
            and path.endswith("/storyboard")
            and latest_job
            and latest_job.get("scene_snapshot", {}).get("storyboard_url")
        ):
            route.fulfill(
                status=200,
                content_type="image/svg+xml",
                body=FIXTURE.read_bytes(),
            )
            return
        if request.method == "POST" and path.endswith("/api/video-jobs"):
            counts["submission"] += 1
            body = json.loads(request.post_data or "{}")
            kind = body.get("kind", "first_cut")
            job_id = f"job-{kind}"
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    video_job_payload(
                        job_status,
                        provider=provider,
                        job_id=job_id,
                        kind=kind,
                    )
                ),
            )
            return
        if request.method == "POST" and "/approve-revision" in path:
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    video_job_payload(
                        "completed",
                        provider=provider,
                        job_id="job-first_cut",
                        revision_approved=True,
                    )
                ),
            )
            return
        job_path = path.split("/api/video-jobs/", 1)[-1]
        if (
            request.method == "GET"
            and "/api/video-jobs/" in path
            and "/" not in job_path
        ):
            job_id = path.rsplit("/", 1)[-1]
            if job_payloads and job_id in job_payloads:
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(job_payloads[job_id]),
                )
                return
            kind = "director_cut" if "director_cut" in path else "first_cut"
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    video_job_payload(
                        job_status,
                        provider=provider,
                        job_id=f"job-{kind}",
                        kind=kind,
                    )
                ),
            )
            return
        route.continue_()

    page.route("**/api/video-jobs**", video_route)
    return counts


def test_completed_durable_first_cut_restores_without_generation_and_compacts_directors_cut():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    real_url = "/api/video-jobs/video-durable-first-cut/video"
    latest = video_job_payload(
        provider="vertex",
        job_id="video-durable-first-cut",
        output_url=real_url,
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex", remaining=0, latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#firstCutVideo").is_visible()
            assert page.locator("#firstCutVideo").get_attribute("src").startswith(real_url)
            assert page.locator("#firstCutProviderLabel").inner_text() == (
                "Vertex AI Veo · veo-3.1-generate-001 · 8 seconds"
            )
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
            assert page.locator("#firstCutOutput").inner_text() == "Generated Video"
            assert page.locator("#generationDetailsPanel").is_visible()
            assert page.locator("#generationDetailDuration").inner_text() == "8 seconds"
            assert page.locator(".first-cut-player .video-kicker").count() == 0
            assert page.locator("#analyseFirstCut").inner_text() == "Review Generated Video"
            assert page.locator("#generateFirstCut").is_hidden()
            assert page.locator("#analyseFirstCut").is_enabled()
            assert page.locator("#videoAllowanceMessage").is_hidden()
            assert page.locator("#directorCutStatus").inner_text() == "Director’s Cut — Optional"
            assert page.locator("#directorCutMessage").inner_text() == "Unavailable in this demo"
            assert page.locator("#directorCutCard").is_visible()
            assert page.locator("#directorCutVideo").count() == 0
            assert page.locator("#generateDirectorCut").count() == 0
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
        finally:
            browser.close()


def test_completed_durable_first_cut_wins_over_deterministic_fallback_snapshot():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    latest = video_job_payload(
        provider="vertex",
        job_id="video-durable-first-cut",
        output_url="/api/video-jobs/video-durable-first-cut/video",
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex", latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            page.evaluate(
                """value => sessionStorage.setItem(
                    'framepilot.video-recovery',
                    JSON.stringify(value)
                )""",
                {
                    "version": 1,
                    "scene": {
                        "scene_key": "scene-fallback",
                        "source_signature": "source-fallback",
                    },
                    "screenplay": SCREENPLAY,
                    "creative_intent": "rising dread",
                    "direction_response": direction_payload(provider="mock"),
                    "video_job_ids": {"first_cut": None, "director_cut": None},
                },
            )
            page.reload(wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#analysisSource").is_hidden()
            assert page.locator("#generationDetailsPanel").is_visible()
            assert page.locator("#generationDetailsPanel").inner_text().find("Generation Details") >= 0
            assert page.locator("#scenePlanContent").is_hidden()
            assert page.locator("#firstCutVideo").is_visible()
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
        finally:
            browser.close()


def test_completed_floating_market_first_cut_restores_and_plays_after_refresh():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    direction = floating_market_direction_payload()
    latest = video_job_payload(
        provider="vertex",
        job_id="video-floating-market",
        scene_snapshot={
            "scene_key": direction["scene_key"],
            "source_signature": direction["source_signature"],
            "screenplay": (
                "A floating ocean market carries a blue robot, fruit, crates, tools, "
                "and a courier bag."
            ),
            "creative_intent": "buoyant anticipation",
            "analysis_source": "vertex_multimodal",
            "direction_response": direction,
            "storyboard_url": "/api/video-jobs/video-floating-market/storyboard",
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex", remaining=0, latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            for _ in range(2):
                page.wait_for_function(
                    "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
                )
                page.click("#firstCutOutput")
                page.wait_for_function(
                    "document.querySelector('#firstCutVideo')?.readyState >= 1"
                )
                assert page.locator("#firstCutVideo").is_visible()
                assert page.locator("#firstCutVideo").get_attribute("src").startswith(
                    "/static/demo-generation.mp4"
                )
                assert page.locator("#screenplay").input_value().startswith(
                    "A floating ocean market carries"
                )
                if not page.locator("#firstCutOutput").get_attribute("aria-selected") == "true":
                    page.reload(wait_until="networkidle")
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
        finally:
            browser.close()


def test_simplified_three_stage_workflow_switches_between_motion_and_first_cut():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    latest = video_job_payload(
        provider="vertex",
        job_id="video-three-stage",
        output_url="/api/video-jobs/video-three-stage/video",
        scene_snapshot={
            "scene_key": "scene-test",
            "source_signature": "source-test",
            "screenplay": SCREENPLAY,
            "creative_intent": "rising dread",
            "direction_response": direction_payload(provider="vertex"),
            "storyboard_url": "/api/video-jobs/video-three-stage/storyboard",
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex", latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            assert page.locator(".workflow-stage").count() == 3
            assert [
                " ".join(text.split())
                for text in page.locator(".workflow-stage").all_inner_texts()
            ] == ["1 Direct Scene", "2 Motion Preview", "3 First Cut"]
            assert page.locator(".product-story").count() == 0
            assert page.locator(".tabs").count() == 0
            assert page.locator("[data-tab]").count() == 0
            assert page.locator(".agent-progress").count() == 0
            assert page.locator("#howWorks").count() == 0
            assert page.locator(".about-line").count() == 0
            assert page.locator("#mode").inner_text() == "Vertex AI connected"
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#motionPreviewOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#workflowMotion").get_attribute("data-state") == "active"
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == ""
            assert page.locator("#firstCutOutput").inner_text() == "Generated Video"
            assert page.locator("#motionPreviewOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#motionPreviewOutputPane").is_visible()
            assert page.locator("#firstCutOutputPane").is_hidden()
            assert page.locator(".scene-plan").inner_text().find("Creative Direction") >= 0
            assert page.locator("#charactersSection").count() == 1
            assert page.locator("#objectsSection").count() == 1
            assert page.locator("#environmentSection").count() == 1
            assert page.locator("#motionPlan h3").inner_text() == "Camera & Motion"
            assert "Evidence & Limits" in page.locator(".scene-plan").inner_text()

            page.click("#firstCutOutput")
            assert page.locator("#firstCutOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
            assert page.locator("#firstCutVideo").is_visible()
            assert page.locator("#generationDetailsPanel").is_visible()
            assert page.locator("#analyseFirstCut").inner_text() == "Review Generated Video"
        finally:
            browser.close()


def test_completed_recovery_restores_storyboard_into_motion_preview_without_provider_calls():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    latest = video_job_payload(
        provider="vertex",
        job_id="video-restored-storyboard",
        output_url="/api/video-jobs/video-restored-storyboard/video",
        scene_snapshot={
            "scene_key": "scene-observatory",
            "source_signature": "source-observatory",
            "screenplay": OBSERVATORY_SCREENPLAY,
            "creative_intent": "quiet, majestic awe",
            "direction_response": direction_payload(provider="vertex"),
            "storyboard_url": "/api/video-jobs/video-restored-storyboard/storyboard",
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex", latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#motionPreviewOutput").is_visible()
            assert page.locator("#motionPreviewOutput").is_enabled()
            assert page.locator("#motionPreviewOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#motionPreviewOutputPane").is_visible()
            assert page.locator("#uploadedScene").is_visible()
            assert page.locator("#stage").get_attribute("data-state") == "completed"
            assert page.locator("#uploadState").inner_text() == "Storyboard image restored"
            assert page.locator("#workflowMotion").get_attribute("data-state") == "active"
            assert page.locator("#workflowDirect").get_attribute("data-state") == "complete"
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == ""

            page.click("#firstCutOutput")
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
            page.click("#motionPreviewOutput")
            assert page.locator("#workflowMotion").get_attribute("data-state") == "active"
            assert page.locator("#uploadedScene").is_visible()
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
            assert not any(
                "googleapis.com" in url or "vertex" in url.lower() or "gemini" in url.lower()
                for _, url in requests
            )
        finally:
            browser.close()


def test_castors_day_recovery_keeps_vertex_scene_and_does_not_cross_contaminate_observatory():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    castors_scene = {
        "scene_key": "scene-castors-day",
        "source_signature": "source-castors-day",
        "screenplay": CASTORS_SCREENPLAY,
        "creative_intent": "warm family celebration in golden-hour light",
        "analysis_source": "vertex_multimodal",
        "shot_plan": direction_payload(provider="vertex")["plan"],
        "storyboard_url": "/api/video-jobs/video-castors-day/storyboard",
    }
    castors_job = video_job_payload(
        provider="vertex",
        job_id="video-castors-day",
        output_url="/api/video-jobs/video-castors-day/video",
        scene_snapshot=castors_scene,
    )
    observatory_job = video_job_payload(
        provider="vertex",
        job_id="video-observatory",
        output_url="/api/video-jobs/video-observatory/video",
        scene_snapshot={
            "scene_key": "scene-observatory",
            "source_signature": "source-observatory",
            "screenplay": OBSERVATORY_SCREENPLAY,
            "creative_intent": "quiet, majestic awe",
            "direction_response": direction_payload(provider="vertex"),
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chromium,
        )
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(
                page,
                provider="vertex",
                remaining=0,
                latest_job=castors_job,
                job_payloads={
                    "video-castors-day": castors_job,
                    "video-observatory": observatory_job,
                },
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#screenplay").input_value() == CASTORS_SCREENPLAY
            assert page.locator("#analysisSource").inner_text() == "Vertex multimodal"
            assert page.locator("#uploadedScene").is_visible()
            assert page.locator("#uploadState").inner_text() == "Storyboard image restored"
            assert page.locator("#firstCutVideo").get_attribute("src").startswith(
                "/api/video-jobs/video-castors-day/video"
            )
            assert page.locator("#generationDetailModel").inner_text() == (
                "veo-3.1-generate-001"
            )
            assert page.locator("#workflowMotion").get_attribute("data-state") == "active"

            page.evaluate(
                """value => sessionStorage.setItem(
                    'framepilot.video-recovery',
                    JSON.stringify(value)
                )""",
                {
                    "version": 1,
                    "scene": {
                        "scene_key": "scene-observatory",
                        "source_signature": "source-observatory",
                    },
                    "screenplay": OBSERVATORY_SCREENPLAY,
                    "creative_intent": "quiet, majestic awe",
                    "direction_response": direction_payload(provider="vertex"),
                    "video_job_ids": {
                        "first_cut": "video-observatory",
                        "director_cut": None,
                    },
                },
            )
            page.reload(wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#screenplay").input_value() == OBSERVATORY_SCREENPLAY
            assert page.locator("#firstCutVideo").get_attribute("src").startswith(
                "/api/video-jobs/video-observatory/video"
            )
            assert page.locator("#firstCutVideo").get_attribute("src").find(
                "video-castors-day"
            ) == -1
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
            assert not any(
                "googleapis.com" in url
                or "vertex" in url.lower()
                or "gemini" in url.lower()
                for _, url in requests
            )
        finally:
            browser.close()


def test_completed_recovery_without_storyboard_defaults_to_generated_video_without_blank_preview():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    latest = video_job_payload(
        provider="vertex",
        job_id="video-missing-storyboard",
        output_url="/api/video-jobs/video-missing-storyboard/video",
        scene_snapshot={
            "scene_key": "scene-legacy",
            "source_signature": "source-legacy",
            "screenplay": OBSERVATORY_SCREENPLAY,
            "creative_intent": "quiet, majestic awe",
            "direction_response": direction_payload(provider="vertex"),
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex", latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#firstCutOutput").is_visible()
            assert page.locator("#firstCutOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#motionPreviewOutput").is_hidden()
            assert page.locator("#motionPreviewOutput").is_disabled()
            assert page.locator("#firstCutOutputPane").is_visible()
            assert page.locator("#motionPreviewOutputPane").is_hidden()
            assert page.locator("#uploadedScene").is_hidden()
            assert page.locator("#videoRecoveryMessage").inner_text() == (
                "Motion Preview is unavailable for this recovered scene. "
                "The completed generated video remains available."
            )
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
        finally:
            browser.close()


def test_new_storyboard_upload_restores_motion_preview_after_missing_recovery():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    latest = video_job_payload(
        provider="vertex",
        job_id="video-missing-storyboard-upload",
        output_url="/api/video-jobs/video-missing-storyboard-upload/video",
        scene_snapshot={
            "scene_key": "scene-legacy",
            "source_signature": "source-legacy",
            "screenplay": OBSERVATORY_SCREENPLAY,
            "creative_intent": "quiet, majestic awe",
            "direction_response": direction_payload(provider="vertex"),
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex", latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            page.set_input_files("#image", str(FIXTURE))
            page.wait_for_function(
                "document.querySelector('#uploadedScene')?.hidden === false"
            )
            page.click("#direct")
            page.wait_for_function(
                "document.querySelector('#stage')?.dataset.state === 'completed'"
            )
            assert page.locator("#motionPreviewOutput").is_visible()
            assert page.locator("#motionPreviewOutput").is_enabled()
            assert page.locator("#motionPreviewOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#uploadedScene").is_visible()
            assert page.locator("#workflowMotion").get_attribute("data-state") == "active"
        finally:
            browser.close()


def test_completed_recovery_restores_scene_consistency_and_survives_new_scene():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the completed recovery test.")

    latest = video_job_payload(
        provider="vertex",
        job_id="video-observatory",
        output_url="/static/demo-generation.mp4",
        scene_snapshot={
            "scene_key": "scene-observatory",
            "source_signature": "source-observatory",
            "screenplay": OBSERVATORY_SCREENPLAY,
            "creative_intent": (
                "Create quiet, majestic awe in one continuous shot. Prioritise the "
                "astronomer slowly raising the lantern, feet planted."
            ),
        },
    )
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex", latest_job=latest)
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            page.wait_for_function(
                "document.querySelector('#time')?.innerText === '00:00 / 00:02'"
            )
            assert page.locator("#screenplay").input_value() == OBSERVATORY_SCREENPLAY
            assert page.locator("#mood").input_value().startswith("Create quiet, majestic awe")
            assert page.locator("#generationDetailsPanel").is_visible()
            assert page.locator("#generationDetailStatus").inner_text() == "Completed"
            assert page.locator("#generationDetailProvider").inner_text() == "Vertex AI Veo"
            assert page.locator("#generationDetailModel").inner_text() == "veo-3.1-generate-001"
            assert page.locator("#generationDetailDuration").inner_text() == "2 seconds"
            assert page.locator("#scenePlanContent").is_hidden()
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
            assert page.locator("#stateLabel").inner_text() == "GENERATED VIDEO READY"
            assert page.locator("#workflowDirect").get_attribute("data-state") == "complete"
            assert page.locator("#workflowMotion").get_attribute("data-state") == "complete"
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
            assert "READY TO DIRECT" not in page.locator("body").inner_text()

            page.click("#direct")
            page.wait_for_function(
                "document.querySelector('#stage')?.dataset.state === 'completed'"
            )
            assert page.locator("#motionPreviewOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#firstCutOutput").is_hidden()
            assert not any(
                method == "POST" and "/api/video-jobs" in url
                for method, url in requests
            )

            page.reload(wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'"
            )
            assert page.locator("#screenplay").input_value() == OBSERVATORY_SCREENPLAY
            assert page.locator("#firstCutOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#workflowFirstCut").get_attribute("data-state") == "active"
        finally:
            browser.close()


def test_uploaded_preview_is_loaded_without_an_empty_stage_overlay():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chromium,
        )
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex")
            page.goto(BASE_URL, wait_until="networkidle")
            page.set_input_files("#image", str(FIXTURE))
            page.wait_for_function(
                """() => {
                    const stage = document.querySelector('#stage');
                    const scene = document.querySelector('#uploadedScene');
                    const plane = document.querySelector('.image-plane');
                    return stage?.dataset.state === 'image-ready'
                        && scene?.hidden === false
                        && plane?.style.backgroundImage;
                }"""
            )

            metrics = page.evaluate(
                """async () => {
                    const stage = document.querySelector('#stage');
                    const overlay = document.querySelector('#stageState');
                    const plane = document.querySelector('.image-plane');
                    const background = getComputedStyle(plane).backgroundImage;
                    const source = background.match(/^url\\(["']?(.*?)["']?\\)$/)?.[1];
                    const image = new Image();
                    image.src = source;
                    await image.decode();
                    return {
                        naturalWidth: image.naturalWidth,
                        stageState: stage.dataset.state,
                        overlayDisplay: getComputedStyle(overlay).display,
                        overlayVisibility: getComputedStyle(overlay).visibility,
                        planeOpacity: getComputedStyle(plane).opacity,
                        coveringOverlayDisplay: getComputedStyle(stage, '::after').display,
                    };
                }"""
            )
            assert metrics["naturalWidth"] > 0
            assert metrics["stageState"] == "image-ready"
            assert metrics["overlayDisplay"] == "none"
            assert metrics["planeOpacity"] == "1"
            assert metrics["coveringOverlayDisplay"] == "none"

            assert page.get_attribute("#stage", "data-state") == "image-ready"
            assert page.locator("#stageState").evaluate(
                "element => getComputedStyle(element).display"
            ) == "none"

            page.click("#direct")
            page.wait_for_function(
                """() => {
                    const panel = document.querySelector('#motionPlan');
                    return panel?.hidden === false
                        && document.querySelector('#charactersPlan')?.textContent;
                }"""
            )
            assert page.locator("#motionPlanStatus").inner_text() == "CONFIRMATION NEEDED"
            assert "Preserve" in page.locator("#motionPlan").inner_text()
            assert "Main Characters" in page.locator(".scene-plan").inner_text()
            assert "Objects" in page.locator(".scene-plan").inner_text()
            assert "Environment" in page.locator(".scene-plan").inner_text()
            assert "Camera & Motion" in page.locator(".scene-plan").inner_text()
            assert page.get_attribute("#stage", "data-state") == "completed"
        finally:
            browser.close()


@pytest.mark.parametrize(
    ("fixture", "expected_width", "expected_height", "screenshot_name"),
    [
        (PORTRAIT_FIXTURE, 600, 1000, "castors-day-portrait-full-composition.png"),
        (SQUARE_FIXTURE, 700, 700, None),
        (FIXTURE, 960, 540, None),
    ],
)
def test_motion_preview_contains_non_16_by_9_storyboards_without_resizing_requests(
    fixture,
    expected_width,
    expected_height,
    screenshot_name,
):
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser preview regression test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=True,
            executable_path=chromium,
        )
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        requests = []
        page.on("request", lambda request: requests.append((request.method, request.url)))
        try:
            mock_video_api(page, provider="vertex")
            page.goto(BASE_URL, wait_until="networkidle")
            page.set_input_files("#image", str(fixture))
            page.wait_for_function(
                """() => {
                    const scene = document.querySelector('#uploadedScene');
                    const plane = document.querySelector('.image-plane');
                    return scene?.hidden === false && plane?.style.backgroundImage;
                }"""
            )

            metrics = page.evaluate(
                """async () => {
                    const stage = document.querySelector('#stage');
                    const plane = document.querySelector('.image-plane');
                    const backing = document.querySelector('.image-backing');
                    const background = getComputedStyle(plane).backgroundImage;
                    const source = background.match(/^url\\(["']?(.*?)["']?\\)$/)?.[1];
                    const image = new Image();
                    image.src = source;
                    await image.decode();
                    return {
                        naturalWidth: image.naturalWidth,
                        naturalHeight: image.naturalHeight,
                        planeFit: getComputedStyle(plane).backgroundSize,
                        planePosition: getComputedStyle(plane).backgroundPosition,
                        backingFit: getComputedStyle(backing).backgroundSize,
                        backingFilter: getComputedStyle(backing).filter,
                        stageState: stage.dataset.state,
                    };
                }"""
            )
            assert metrics["naturalWidth"] == expected_width
            assert metrics["naturalHeight"] == expected_height
            assert metrics["planeFit"] == "contain"
            assert metrics["planePosition"] == "50% 50%"
            assert metrics["backingFit"] == "cover"
            assert "blur(" in metrics["backingFilter"]
            assert "brightness(" in metrics["backingFilter"]
            assert metrics["stageState"] == "image-ready"

            page.click("#play")
            page.wait_for_function(
                """() => {
                    const stage = document.querySelector('#stage');
                    const plane = document.querySelector('.image-front');
                    return stage?.classList.contains('playing')
                        && getComputedStyle(plane).animationName !== 'none';
                }"""
            )
            assert page.locator("#stage").get_attribute("data-playback-run") == "1"
            assert not any(
                method == "POST"
                and any(path in url for path in ("/api/direct", "/api/video-jobs"))
                for method, url in requests
            )
            assert not any(
                "googleapis.com" in url or "vertex" in url.lower() or "gemini" in url.lower()
                for _, url in requests
            )
            if screenshot_name:
                page.screenshot(path=f"screenshots/{screenshot_name}", full_page=False)
        finally:
            browser.close()


def test_mocked_video_flow_keeps_cuts_separate_and_requires_revision_approval():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="mock")
            page.goto(BASE_URL, wait_until="networkidle")
            page.click("#direct")
            page.wait_for_function("document.querySelector('#stage')?.dataset.state === 'completed'")
            page.click("#confirmMotion")
            assert page.locator("#generateFirstCut").is_enabled()
            assert page.locator("#directorCutCard").is_hidden()

            page.click("#generateFirstCut")
            page.wait_for_function("document.querySelector('#videoApprovalDialog')?.open === true")
            assert page.locator("#firstCutStatus").inner_text() == "APPROVAL REQUIRED"
            assert page.locator("#firstCutVideo").is_hidden()
            page.click("#confirmVideoApproval")

            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'",
                timeout=10000,
            )
            assert page.locator("#firstCutVideo").is_visible()
            assert "Demo generation" in page.locator("#firstCutMessage").inner_text()
            assert page.locator("#approveRevision").is_enabled()
            assert page.locator("#directorCutCard").is_visible()
            assert page.locator("#directorCutStatus").inner_text() == "Director’s Cut — Optional"

            page.click("#motionPreviewOutput")
            page.locator("#planReview").evaluate("element => { element.open = true; }")
            page.click("#approveRevision")
            first_src = page.locator("#firstCutVideo").get_attribute("src")
            assert "/static/demo-generation.mp4" in first_src
            assert page.locator("#directorCutVideo").count() == 0
            assert page.locator("#generateDirectorCut").count() == 0
            assert page.locator("#firstCutOutput").is_visible()
            assert page.locator("#firstCutOutput").get_attribute("aria-selected") == "true"
            assert page.locator("#stage video").count() == 0
        finally:
            browser.close()


def test_exhausted_real_allowance_disables_first_cut_with_exact_message():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            page.route(
                "**/api/health",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"status":"ok","mode":"vertex","video_provider":"vertex"}',
                ),
            )
            page.route(
                "**/api/video-allowance",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=(
                        '{"global_limit":1,"global_used":1,"global_remaining":0,'
                        '"per_ip_limit":1,"per_ip_used":1,"per_ip_remaining":0,'
                        '"director_cut_global_limit":0,"director_cut_global_used":0,'
                        '"director_cut_global_remaining":0,'
                        '"authorized_replacement_limit":0,"authorized_replacement_used":0,'
                        '"authorized_replacement_remaining":0,'
                        '"authorized_replacement_for_job_id":null}'
                    ),
                ),
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#videoAllowanceMessage')?.hidden === false"
            )
            assert page.locator("#generateFirstCut").is_hidden()
            assert page.locator("#videoAllowanceMessage").inner_text() == (
                "Normal First Cut allowance: 1 used / 0 remaining. "
                "Authorized replacement: 0 remaining."
            )
        finally:
            browser.close()


def test_available_replacement_enables_failed_first_cut_and_explains_credit_use():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(
                page,
                provider="vertex",
                remaining=0,
                authorized_replacement_remaining=1,
                authorized_replacement_for_job_id="job-first_cut",
                job_status="failed",
            )
            direction = direction_payload(provider="vertex")
            snapshot = {
                "version": 1,
                "scene": {"scene_key": "scene-test", "source_signature": "source-test"},
                "screenplay": SCREENPLAY,
                "creative_intent": "rising dread",
                "direction_response": direction,
                "video_job_ids": {"first_cut": "job-first_cut", "director_cut": None},
            }
            page.add_init_script(
                f"""sessionStorage.setItem(
                    'framepilot.video-recovery', {json.dumps(json.dumps(snapshot))}
                )"""
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'FAILED'"
            )
            page.click("#confirmMotion")
            assert page.locator("#generateFirstCut").is_enabled()
            assert "Authorized replacement attempt available" in page.locator(
                "#videoAllowanceMessage"
            ).inner_text()

            page.click("#generateFirstCut")
            page.wait_for_function("document.querySelector('#videoApprovalDialog')?.open === true")
            assert page.locator("#videoReplacementNotice").is_visible()
            assert "authorized replacement attempt" in page.locator(
                "#videoReplacementNotice"
            ).inner_text()
            requests = []
            page.on(
                "request",
                lambda request: requests.append(request)
                if request.method == "POST" and "/api/video-jobs" in request.url
                else None,
            )
            page.click("#confirmVideoApproval")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'FAILED'"
            )
            approval_payload = json.loads(requests[0].post_data)
            submission_payload = json.loads(requests[1].post_data)
            assert approval_payload["replacement_for_job_id"] == "job-first_cut"
            assert submission_payload["replacement_for_job_id"] == "job-first_cut"
        finally:
            browser.close()


def test_fresh_direct_preserves_historical_replacement_authorization():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    original_job_id = "job-original-failed-123456"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(
                page,
                provider="vertex",
                remaining=0,
                authorized_replacement_remaining=1,
                authorized_replacement_for_job_id=original_job_id,
                job_status="failed",
            )
            direction = direction_payload(provider="vertex")
            direction["plan"]["motion_plan"]["confirmation_required"] = True
            snapshot = {
                "version": 1,
                "scene": {"scene_key": "scene-test", "source_signature": "source-test"},
                "screenplay": SCREENPLAY,
                "creative_intent": "rising dread",
                "direction_response": direction,
                "video_job_ids": {"first_cut": original_job_id, "director_cut": None},
            }
            page.add_init_script(
                f"""sessionStorage.setItem(
                    'framepilot.video-recovery', {json.dumps(json.dumps(snapshot))}
                )"""
            )
            page.route(
                f"**/api/video-jobs/{original_job_id}",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        video_job_payload(
                            "failed",
                            provider="vertex",
                            job_id=original_job_id,
                        )
                    ),
                ),
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'FAILED'"
            )

            requests = []
            page.on("request", lambda request: requests.append(request))
            page.fill("#screenplay", SCREENPLAY)
            page.click("#direct")
            page.wait_for_function(
                "document.querySelector('#stage')?.dataset.state === 'completed'"
            )
            assert page.locator("#firstCutStatus").inner_text() == "NOT GENERATED"

            page.click("#confirmMotion")
            page.wait_for_function(
                """() => document.querySelector('#videoAllowanceMessage')?.innerText
                    .includes('Authorized replacement attempt available')"""
            )
            assert page.locator("#generateFirstCut").is_enabled()
            assert page.evaluate(
                "document.querySelector('#generateFirstCut').disabled === false"
            )

            approval_payloads = []

            def block_approval(route):
                approval_payloads.append(json.loads(route.request.post_data or "{}"))
                route.abort()

            page.route("**/api/video-jobs/approval", block_approval)
            page.click("#generateFirstCut")
            page.wait_for_function(
                "document.querySelector('#videoApprovalDialog')?.open === true"
            )
            page.click("#confirmVideoApproval")
            page.wait_for_timeout(100)
            assert approval_payloads[0]["replacement_for_job_id"] == original_job_id
            assert not any(
                request.method == "POST"
                and request.url.rstrip("/").endswith("/api/video-jobs")
                for request in requests
            )
        finally:
            browser.close()


def test_replacement_approval_dialog_uses_replacement_copy_and_metadata_without_submission():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    original_job_id = "job-original-failed-copy-123456"
    configured_estimate = "Approximate veo-3.1-generate-001 estimate: ₹1,250.00"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(
                page,
                provider="vertex",
                remaining=0,
                authorized_replacement_remaining=1,
                authorized_replacement_for_job_id=original_job_id,
                video_estimate=configured_estimate,
                job_status="failed",
            )
            direction = direction_payload(provider="vertex")
            direction["plan"]["motion_plan"]["confirmation_required"] = True
            snapshot = {
                "version": 1,
                "scene": {"scene_key": "scene-test", "source_signature": "source-test"},
                "screenplay": SCREENPLAY,
                "creative_intent": "rising dread",
                "direction_response": direction,
                "video_job_ids": {"first_cut": original_job_id, "director_cut": None},
            }
            page.add_init_script(
                f"""sessionStorage.setItem(
                    'framepilot.video-recovery', {json.dumps(json.dumps(snapshot))}
                )"""
            )
            page.route(
                f"**/api/video-jobs/{original_job_id}",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        video_job_payload(
                            "failed",
                            provider="vertex",
                            job_id=original_job_id,
                        )
                    ),
                ),
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#firstCutStatus')?.innerText === 'FAILED'"
            )
            page.click("#confirmMotion")
            page.wait_for_function(
                "document.querySelector('#generateFirstCut')?.disabled === false"
            )

            approval_payloads = []

            def block_approval(route):
                approval_payloads.append(json.loads(route.request.post_data or "{}"))
                route.abort()

            page.route("**/api/video-jobs/approval", block_approval)
            page.click("#generateFirstCut")
            page.wait_for_function(
                "document.querySelector('#videoApprovalDialog')?.open === true"
            )
            assert page.locator("#videoGenerationModel").inner_text() == "veo-3.1-generate-001"
            assert page.locator("#videoGenerationEstimate").inner_text() == configured_estimate
            assert page.locator("#videoGenerationEstimate").is_visible()
            assert page.locator("#videoGenerationAllowance").inner_text() == (
                "One authorized replacement Veo generation attempt will be used."
            )
            assert "remaining First Cut allowance" not in page.locator(
                "#videoGenerationSpec"
            ).inner_text()
            assert approval_payloads == []

            requests = []
            page.on("request", lambda request: requests.append(request))
            page.click("#confirmVideoApproval")
            page.wait_for_timeout(100)
            assert approval_payloads[0]["replacement_for_job_id"] == original_job_id
            assert not any(
                request.method == "POST"
                and request.url.rstrip("/").endswith("/api/video-jobs")
                for request in requests
            )
        finally:
            browser.close()


def test_vertex_first_cut_without_storyboard_handle_stays_disabled():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex")
            direction = direction_payload(provider="vertex")
            direction["image_handle"] = None
            page.route(
                "**/api/direct",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(direction),
                ),
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.fill("#screenplay", SCREENPLAY)
            page.click("#direct")
            page.wait_for_function("document.querySelector('#stage')?.dataset.state === 'completed'")
            assert page.locator("#generateFirstCut").is_hidden()
            assert page.locator("#generateVeo").is_hidden()
        finally:
            browser.close()


def test_vertex_generate_with_veo_uses_explicit_approval_and_no_cancel_submission():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            counts = mock_video_api(page, provider="vertex")
            page.goto(BASE_URL, wait_until="networkidle")
            page.fill("#screenplay", SCREENPLAY)
            page.click("#direct")
            page.wait_for_function("document.querySelector('#stage')?.dataset.state === 'completed'")
            page.click("#confirmMotion")

            assert page.locator("#mode").inner_text().lower() == "vertex ai connected"
            assert page.locator("#firstCutRequestProviderLabel").inner_text().lower() == "vertex ai veo"
            assert page.locator("#generateVeo").is_enabled()
            assert page.locator("#generateFirstCut").is_enabled()

            page.click("#generateVeo")
            page.wait_for_function("document.querySelector('#videoApprovalDialog')?.open === true")
            assert counts == {"approval": 0, "submission": 0}
            assert page.locator("#videoApprovalTitle").inner_text() == "Generate First Cut"
            assert "grounding_summary" in page.locator("#videoMotionPlan").inner_text()
            assert "8-second, video-only output" in page.locator("#videoGenerationSpec").inner_text()
            assert "veo-3.1-generate-001" in page.locator("#videoGenerationSpec").inner_text()
            assert "deployment-configured" in page.locator("#videoGenerationSpec").inner_text()

            page.click("#cancelVideoApproval")
            assert counts == {"approval": 0, "submission": 0}
            assert page.locator("#generateVeo").is_enabled()

            page.click("#generateVeo")
            page.wait_for_function("document.querySelector('#videoApprovalDialog')?.open === true")
            page.evaluate(
                """() => {
                    const button = document.querySelector('#confirmVideoApproval');
                    button.click();
                    button.click();
                }"""
            )
            page.wait_for_function("document.querySelector('#firstCutStatus')?.innerText === 'COMPLETED'")
            assert counts == {"approval": 1, "submission": 1}
            assert page.locator("#generateVeo").is_hidden()
        finally:
            browser.close()


def test_floating_market_activation_reenables_first_cut_without_direct_or_provider_calls():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            counts = mock_video_api(page, provider="vertex", remaining=0)
            direction = floating_market_direction_payload()
            authorization_attempts = []

            def authorization_route(route):
                authorization_attempts.append(json.loads(route.request.post_data or "{}"))
                body = authorization_attempts[-1]
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "authorization_id": "test-auth-floating-market-browser",
                            "available": True,
                            "kind": "first_cut",
                            "scene_key": body["scene_key"],
                            "source_signature": body["source_signature"],
                            "model": body["model"],
                            "source": "authorized_test_attempt",
                            "duration_seconds": 8,
                            "aspect_ratio": "16:9",
                            "audio_enabled": False,
                        }
                    ),
                )

            page.route(
                "**/api/video-test-authorization/pending",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(
                        {
                            "authorization_id": "pending-auth-final-fight-demo",
                            "available": False,
                            "state": "pending",
                            "kind": "first_cut",
                            "model": "veo-3.1-generate-001",
                            "source": "authorized_test_attempt",
                        }
                    ),
                ),
            )
            page.route("**/api/video-test-authorization", authorization_route)
            page.goto(BASE_URL, wait_until="networkidle")
            page.fill(
                "#screenplay",
                "A floating ocean market carries a blue robot, fruit, crates, tools, and a courier bag.",
            )
            page.evaluate(
                """payload => {
                    renderDirectedResult(payload);
                    renderAllVideoStates();
                }""",
                direction,
            )
            page.wait_for_function(
                "document.querySelector('#activateFloatingMarketTest')?.hidden === false"
            )
            assert authorization_attempts == []
            assert page.locator("#activateFloatingMarketTest").inner_text() == (
                "Activate authorized test attempt"
            )
            assert page.locator("#generateFirstCut").is_hidden()
            assert page.locator("#generateVeo").is_hidden()
            assert "activate the authorized test attempt" in (
                page.locator("#firstCutMessage").inner_text().lower()
            )

            page.evaluate(
                """() => {
                    const button = document.querySelector('#activateFloatingMarketTest');
                    button.click();
                    button.click();
                }"""
            )
            page.wait_for_function(
                "document.querySelector('#activateFloatingMarketTest')?.hidden === true"
            )

            assert len(authorization_attempts) == 1
            assert authorization_attempts[0]["analysis_source"] == "vertex_multimodal"
            assert authorization_attempts[0]["kind"] == "first_cut"
            assert page.evaluate("canOpenFirstCutApproval()") is True
            assert page.locator("#generateFirstCut").is_enabled()
            assert page.locator("#generateVeo").is_enabled()
            page.wait_for_function(
                """() => document.querySelector('#videoAllowanceMessage')?.innerText
                    .includes('One authorized First Cut test attempt available')"""
            )
            assert "One authorized First Cut test attempt available" in (
                page.locator("#videoAllowanceMessage").inner_text()
            )
            assert counts == {"approval": 0, "submission": 0}
        finally:
            browser.close()


def test_mock_health_retains_demo_only_video_copy():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="mock")
            page.goto(BASE_URL, wait_until="networkidle")
            assert page.locator("#firstCutRequestProviderLabel").inner_text().lower() == "demo generation"
            assert "Optional generative output" in page.locator("#firstCutRequest").inner_text()
            assert page.locator("#firstCutProviderLabel").inner_text() == "Demo generation"
        finally:
            browser.close()


def test_delayed_vertex_health_never_exposes_demo_provider_copy():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for browser health tests.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            def delayed_health(route):
                time.sleep(1.5)
                route.fulfill(
                    status=200,
                    content_type="application/json",
                    body='{"status":"ok","mode":"vertex","video_provider":"vertex"}',
                )

            page.route("**/api/health", delayed_health)
            page.goto(BASE_URL, wait_until="commit")
            page.wait_for_function(
                """() => document.querySelector('#mode')?.innerText === 'Checking provider…'
                    && document.querySelector('#firstCutRequestProviderLabel')?.innerText
                      ?.toLowerCase() === 'checking provider…'""",
                timeout=1000,
            )
            page.wait_for_function(
                "document.querySelector('#mode')?.innerText === 'Vertex AI connected'"
            )
            assert page.locator("#firstCutRequestProviderLabel").inner_text().lower() == "vertex ai veo"
        finally:
            browser.close()


def test_vertex_health_with_deterministic_fallback_is_labeled_without_provider_downgrade():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for browser direction tests.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex")
            fallback = direction_payload(provider="vertex")
            fallback["mode"] = "demo"
            fallback["analysis_source"] = "deterministic_fallback"
            fallback["activity"].insert(
                0,
                {
                    "step": "FALLBACK",
                    "detail": "Vertex response failed local validation; deterministic fallback used.",
                },
            )
            page.route(
                "**/api/direct",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=json.dumps(fallback),
                ),
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.click("#direct")
            page.wait_for_function(
                "document.querySelector('#stage')?.dataset.state === 'completed'"
            )
            assert page.locator("#mode").inner_text().lower() == "vertex ai connected"
            assert page.locator("#firstCutRequestProviderLabel").inner_text().lower() == "vertex ai veo"
            assert page.locator("#analysisSource").inner_text() == "Deterministic fallback"
            assert page.locator("#progressSummary").count() == 0
        finally:
            browser.close()


def test_legacy_restored_direction_is_labeled_and_cannot_be_approved():
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for browser recovery tests.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex")
            legacy = direction_payload(provider="vertex")
            legacy.pop("analysis_source")
            snapshot = {
                "version": 1,
                "scene": {
                    "scene_key": "scene-legacy",
                    "source_signature": "source-legacy",
                },
                "screenplay": SCREENPLAY,
                "creative_intent": "rising dread",
                "direction_response": legacy,
                "video_job_ids": {"first_cut": None, "director_cut": None},
            }
            page.add_init_script(
                f"""sessionStorage.setItem(
                    'framepilot.video-recovery', {json.dumps(json.dumps(snapshot))}
                )"""
            )
            page.goto(BASE_URL, wait_until="networkidle")
            page.wait_for_function(
                "document.querySelector('#stage')?.dataset.state === 'completed'"
            )
            assert page.locator("#analysisSource").inner_text() == "Legacy/fallback result"
            assert page.locator("#generateVeo").is_hidden()
            assert page.locator("#generateFirstCut").is_hidden()
        finally:
            browser.close()


@pytest.mark.parametrize("job_status", ["queued", "generating", "submission_unknown", "completed", "failed"])
def test_vertex_first_cut_terminal_and_active_states_block_resubmission(job_status):
    chromium = os.getenv("CHROMIUM_PATH") or shutil.which("chromium")
    if not chromium:
        pytest.skip("Chromium is required for the browser video flow test.")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, executable_path=chromium)
        page = browser.new_page(viewport={"width": 1366, "height": 768})
        try:
            mock_video_api(page, provider="vertex", job_status=job_status)
            page.goto(BASE_URL, wait_until="networkidle")
            page.fill("#screenplay", SCREENPLAY)
            page.click("#direct")
            page.wait_for_function("document.querySelector('#stage')?.dataset.state === 'completed'")
            page.click("#confirmMotion")
            page.click("#generateVeo")
            page.wait_for_function("document.querySelector('#videoApprovalDialog')?.open === true")
            page.click("#confirmVideoApproval")
            page.wait_for_function(
                f"document.querySelector('#firstCutStatus')?.innerText === '{'STATUS UNCERTAIN' if job_status == 'submission_unknown' else job_status.upper()}'",
            )
            assert page.locator("#generateVeo").is_hidden()
            assert page.locator("#generateFirstCut").is_hidden()
        finally:
            browser.close()