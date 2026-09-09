from pathlib import Path


STATIC = Path("static")


def test_workspace_has_semantic_brief_stage_and_scene_plan_surfaces():
    html = (STATIC / "index.html").read_text()

    assert '<main id="appShell" class="app-shell">' in html
    assert 'id="sceneBrief"' in html
    assert 'id="briefToggle"' in html
    assert 'aria-controls="briefContent"' in html
    assert 'id="stage"' in html
    assert 'id="firstPass"' in html
    assert 'id="version"' in html
    assert 'class="agent-progress"' not in html
    assert 'data-tab="direction"' not in html
    assert 'data-tab="depth"' not in html
    assert 'data-tab="critique"' not in html
    assert 'id="fineTune"' in html
    assert 'Direct Scene' in html
    assert 'Motion Preview' in html
    assert 'First Cut' in html
    assert 'Generated Video' in html
    assert 'Generation Details' in html
    assert 'id="outputHeading"' not in html
    assert 'LOCAL PROJECT' not in html
    assert 'Fresh pass' not in html
    assert 'RATE-LIMITED' not in html
    assert 'How FramePilot works' not in html
    assert 'about-line' not in html
    assert 'Why this is agentic' not in html
    assert '<span class="stage-state-mark" aria-hidden="true">FP</span>' not in html


def test_frontend_contains_accessible_state_and_progress_contracts():
    html = (STATIC / "index.html").read_text()
    app_js = (STATIC / "app.js").read_text()
    css = (STATIC / "style.css").read_text()

    assert 'aria-selected="true"' in html
    assert 'aria-pressed="false"' in html
    assert "setRequestState('loading'" in app_js
    assert "setRequestState('error'" in app_js
    assert "setRequestState('complete'" in app_js
    assert "brief-collapsed" in app_js
    assert "renderProgress(data.activity)" in app_js
    assert "Analysing composition" in app_js
    assert "Director’s Cut ready" in app_js
    assert "Vertex AI connected" in app_js
    assert "Local fallback" in app_js
    assert '.app-shell.brief-collapsed' in css
    assert '.stage[data-state="loading"]' in css
    assert '.stage[data-state="error"]' in css
    assert ':focus-visible' in css


def test_provider_copy_tolerates_partial_dom_during_startup():
    app_js = (STATIC / "app.js").read_text()

    assert "function setTextIfPresent(selector, value)" in app_js
    assert "if (workspace)" in app_js
    assert "setTextIfPresent('#videoProviderLabel'" in app_js
    assert "setTextIfPresent('#videoAllowanceMessage'" in app_js


def test_frontend_keeps_cut_plans_immutable_and_resets_playback():
    app_js = (STATIC / "app.js").read_text()

    assert "let cutPlans = Object.freeze" in app_js
    assert "setCutPlans(data)" in app_js
    assert "selectedCut = revised ? 'revised' : 'first'" in app_js
    assert "setParams(params, selectedCut)" in app_js
    assert "cutEdits[selectedCut]" in app_js
    assert "setControlValues(activeParams)" in app_js
    assert "progress').style.width = '0%'" in app_js
    assert "stage.dataset.playbackRun" in app_js
    assert "renderRevisionChanges()" in app_js


def test_critique_contains_directors_cut_change_summary_surface():
    html = (STATIC / "index.html").read_text()

    assert 'id="revisionChanges"' in html
    assert 'id="revisionChangeList"' in html
    assert "Changes in Director’s Cut" in html


def test_frontend_exposes_grounded_motion_plan_and_confirmation_surface():
    html = (STATIC / "index.html").read_text()
    app_js = (STATIC / "app.js").read_text()
    css = (STATIC / "style.css").read_text()

    assert 'id="motionPlan"' in html
    assert 'id="charactersPlan"' in html
    assert 'id="objectsPlan"' in html
    assert 'id="environmentPlan"' in html
    assert 'id="motionConfirmation"' in html
    assert "currentMotionPlan()" in app_js
    assert "renderMotionPlan()" in app_js
    assert "confirmation_required" in app_js
    assert ".motion-item" in css


def test_frontend_renders_preserved_visible_entities_without_inventing_motion():
    app_js = (STATIC / "app.js").read_text()

    assert "visible_characters" in app_js
    assert "visible_objects" in app_js
    assert "visible_environment" in app_js
    assert "No bounded motion proposed; preserve this visible entity." in app_js


def test_frontend_exposes_isolated_video_states_and_approval_flow():
    html = (STATIC / "index.html").read_text()
    app_js = (STATIC / "app.js").read_text()

    assert 'id="firstCutOutputPane"' in html
    assert 'data-video-kind="first_cut"' in html
    assert 'data-video-kind="director_cut"' in html
    assert 'id="videoApprovalDialog"' in html
    assert 'id="firstCutVideo"' in html
    assert 'id="directorCutCard"' in html
    assert 'id="directorCutVideo"' not in html
    assert 'id="generateDirectorCut"' not in html
    assert "approval_required" in app_js
    assert "Approve and queue" in html
    assert "revisionApproved" in app_js
    assert "source_signature" in app_js


def test_frontend_exposes_explicit_video_critique_action_and_separate_surface():
    html = (STATIC / "index.html").read_text()
    app_js = (STATIC / "app.js").read_text()

    assert 'id="analyseFirstCut"' in html
    assert 'id="videoCritiqueDialog"' in html
    assert 'id="videoCritiquePanel"' in html
    assert "If live critique is enabled, it uses a Gemini request." in html
    assert "/critique" in app_js
    assert "Video critique unavailable." in app_js
    assert "video_critique" in app_js


def test_frontend_persists_safe_versioned_recovery_snapshot_and_restores_jobs():
    html = (STATIC / "index.html").read_text()
    app_js = (STATIC / "app.js").read_text()

    assert 'id="videoRecoveryMessage"' in html
    assert "VIDEO_RECOVERY_STORAGE_KEY" in app_js
    assert "VIDEO_RECOVERY_VERSION = 1" in app_js
    assert "sessionStorage.setItem(VIDEO_RECOVERY_STORAGE_KEY" in app_js
    assert "sessionStorage.getItem(VIDEO_RECOVERY_STORAGE_KEY" in app_js
    assert "direction_response: data" in app_js
    assert "video_job_ids" in app_js
    assert "approval_id" not in app_js.split("function saveRecoverySnapshot", 1)[1].split(
        "function readRecoverySnapshot", 1
    )[0]
    assert "/api/video-jobs/${encodeURIComponent(jobId)}" in app_js
    assert "Re-upload it before requesting another approved job." in app_js
    assert "Saved recovery data was outdated or invalid and was discarded." in app_js
    assert "restoreDurableDirectContext" in app_js
    assert "restoreVideoRecovery().then(restoreLatestCompletedFirstCut)" in app_js


def test_upload_success_shows_preview_and_hides_empty_state():
    app_js = (STATIC / "app.js").read_text()
    css = (STATIC / "style.css").read_text()

    assert "scene.hidden = false" in app_js
    assert "stage.classList.add('has-upload')" in app_js
    assert "setRequestState('ready', 'READY', 'FRAME READY')" in app_js
    assert "image-ready" in app_js
    assert ':not(.has-upload) .stage-state' in css
    assert ".stage[data-state=\"empty\"] > :not(.stage-state)" in css
    assert "[hidden] { display: none !important; }" in css


def test_replacing_upload_ignores_stale_decode_and_releases_object_urls():
    app_js = (STATIC / "app.js").read_text()

    assert "let uploadRequest = 0" in app_js
    assert "const request = ++uploadRequest" in app_js
    assert "if (request !== uploadRequest) return" in app_js
    assert "URL.revokeObjectURL(objectUrl)" in app_js
    assert "image.onload" in app_js


def test_invalid_upload_enters_error_state_instead_of_empty_state():
    app_js = (STATIC / "app.js").read_text()

    assert "Unable to decode that storyboard image." in app_js
    assert "stage.classList.remove('has-upload')" in app_js
    assert "setRequestState('error', 'ERROR', 'STORYBOARD IMAGE ERROR')" in app_js
    assert "$('#uploadedScene').hidden = true" in app_js


def test_directing_and_cut_switches_preserve_uploaded_preview():
    app_js = (STATIC / "app.js").read_text()

    assert "setRequestState('loading'" in app_js
    assert "data = result" in app_js
    assert "selectPass(revised)" in app_js
    assert "stage.classList.remove('has-upload')" in app_js
    assert app_js.count("stage.classList.remove('has-upload')") == 1
    assert "Deterministic fallback" in app_js
    assert "result.analysis_source === 'deterministic_fallback'" in app_js
    assert "Legacy/fallback result" in app_js
    assert "!isLegacyDirection()" in app_js


def test_stage_state_contract_only_renders_overlay_for_empty_unuploaded_stage():
    app_js = (STATIC / "app.js").read_text()
    css = (STATIC / "style.css").read_text()

    assert "state === 'ready'" in app_js
    assert "state === 'complete' || state === 'fallback'" in app_js
    assert 'data-state="image-ready"' in css or "image-ready" in app_js
    assert ':not(.has-upload) .stage-state' in css