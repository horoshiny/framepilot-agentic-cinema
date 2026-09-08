const $ = selector => document.querySelector(selector);
let data = null;
let timer = null;
let start = 0;
let imageData = null;
let particleFrame = null;
let activeEffects = [];
const stage = $('#stage');
const compositor = window.FramePilotCompositor;
let activeParams = null;
let cutPlans = Object.freeze({ first: null, revised: null });
let cutEdits = { first: {}, revised: {} };
let selectedCut = 'first';
let playbackRun = 0;
let uploadRequest = 0;
let depthSource = 'heuristic';
let lastComposition = null;
let activeDepthLayout = compositor.heuristicDepthLayout();
const videoStates = {
  first_cut: { status: 'not_generated' },
  director_cut: { status: 'not_generated' },
};
const videoPollers = {};
let pendingVideoKind = null;
let videoSceneKey = null;
let videoSourceSignature = null;
let revisionApproved = false;
let videoAllowance = null;
let videoProvider = null;
let videoModel = null;
let videoEstimate = null;
let healthLoaded = false;
let conservativePlanConfirmed = false;
let videoCritique = null;
let videoCritiqueBusy = false;
let durableFirstCutRestored = false;
let recoveredScenePlanAvailable = true;
let outputMode = 'motion_preview';
const VIDEO_RECOVERY_STORAGE_KEY = 'framepilot.video-recovery';
const VIDEO_RECOVERY_VERSION = 1;
const modeLabels = {
  demo: 'Local fallback',
  vertex: 'Vertex AI connected',
  rate_limited_fallback: 'Local fallback',
};
const DEFAULT_VEO_MODEL = 'veo-3.1-generate-001';

function isVertexVideoProvider() {
  return videoProvider === 'vertex';
}

function hasDirectionSource(result = data) {
  return result?.analysis_source === 'vertex_multimodal'
    || result?.analysis_source === 'deterministic_fallback';
}

function isLegacyDirection(result = data) {
  return Boolean(result && !hasDirectionSource(result));
}

function currentMotionPlan(result = data) {
  return result?.plan?.motion_plan || null;
}

function motionPlanConfirmed(result = data) {
  return conservativePlanConfirmed
    && currentMotionPlan(result)?.confirmation_required === false;
}

function hasApprovedMotionPlan() {
  const plan = currentMotionPlan();
  return Boolean(
    plan
      && plan.grounding_summary
      && plan.camera_movement
      && motionPlanConfirmed(),
  ) && !isLegacyDirection();
}

function hasStoryboardImage() {
  return Boolean(data?.image_handle);
}

function hasValidVideoModel() {
  return !isVertexVideoProvider()
    || (typeof videoModel === 'string' && videoModel.length > 0);
}

function realAllowanceUnavailable() {
  return isVertexVideoProvider() && (
    !videoAllowance
      || videoAllowance.global_remaining <= 0
      || videoAllowance.per_ip_remaining <= 0
  );
}

function replacementAuthorizationJobId() {
  const state = videoStates.first_cut;
  if (['failed', 'approval_required'].includes(state.status)) {
    return state.replacement_for_job_id
      || state.job_id
      || videoAllowance?.authorized_replacement_for_job_id
      || null;
  }
  if (['queued', 'generating', 'submission_unknown', 'completed'].includes(state.status)) {
    return null;
  }
  return videoAllowance?.authorized_replacement_for_job_id || null;
}

function hasAvailableReplacementAuthorization() {
  const status = videoStates.first_cut.status;
  const replacementJobId = replacementAuthorizationJobId();
  return Boolean(
    isVertexVideoProvider()
      && !['queued', 'generating', 'submission_unknown'].includes(status)
      && replacementJobId
      && Number(videoAllowance?.authorized_replacement_remaining) > 0
      && videoAllowance?.authorized_replacement_for_job_id === replacementJobId,
  );
}

function firstCutAllowanceUnavailable() {
  return realAllowanceUnavailable() && !hasAvailableReplacementAuthorization();
}

function firstCutSubmissionBlocked() {
  const status = videoStates.first_cut.status;
  return ['approval_required', 'queued', 'generating', 'submission_unknown', 'completed']
    .includes(status)
    || (status === 'failed' && !hasAvailableReplacementAuthorization());
}

function firstCutDisabledReason() {
  if (!data) return 'Direct the scene before requesting a First Cut.';
  if (!healthLoaded) return 'Checking the configured video provider and model…';
  if (isLegacyDirection()) return 'This restored direction is legacy and cannot be approved.';
  if (!currentMotionPlan()) return 'A motion plan is required before requesting a First Cut.';
  if (firstCutSubmissionBlocked()) {
    const status = videoStates.first_cut.status;
    if (status === 'submission_unknown') {
      return 'Submission status uncertain. No retry sent; allowance remains reserved.';
    }
    if (['queued', 'generating'].includes(status)) {
      return 'A First Cut job is already active.';
    }
    if (status === 'completed') return videoStatusMessage('first_cut', videoStates.first_cut);
    if (status === 'approval_required') return 'First Cut approval is already open.';
    return 'The previous First Cut failed and has no available replacement.';
  }
  if (!motionPlanConfirmed()) return 'Use the conservative plan before requesting a First Cut.';
  if (isVertexVideoProvider() && !hasValidVideoModel()) {
    return 'The configured Veo model is unavailable.';
  }
  if (isVertexVideoProvider() && !hasStoryboardImage()) {
    return 'Upload a storyboard image before requesting a Vertex First Cut.';
  }
  if (firstCutAllowanceUnavailable()) {
    return 'Normal First Cut allowance is exhausted and no matching replacement is available.';
  }
  return 'First Cut is ready for explicit approval.';
}

function canOpenFirstCutApproval() {
  return Boolean(
    data
      && hasApprovedMotionPlan()
      && (!isVertexVideoProvider() || (
        hasStoryboardImage()
          && hasValidVideoModel()
          && !firstCutAllowanceUnavailable()
      ))
      && healthLoaded
      && !firstCutSubmissionBlocked(),
  );
}

function setTextIfPresent(selector, value) {
  const element = $(selector);
  if (element) element.textContent = value;
}

function renderVideoProviderCopy() {
  const vertex = isVertexVideoProvider();
  const providerKnown = videoProvider === 'vertex' || videoProvider === 'mock';
  const completedFirstCut = videoStates.first_cut.status === 'completed';
  const workspace = $('#videoWorkspace');
  if (workspace) {
    workspace.setAttribute(
      'aria-label',
       providerKnown
         ? (vertex ? 'Vertex AI Veo video generation' : 'Mocked video generation')
         : 'Video provider status loading',
    );
  }
  const providerLabel = !providerKnown
    ? 'Checking provider…'
    : vertex
      ? 'Vertex AI Veo'
      : 'Demo generation only';
  setTextIfPresent('#videoProviderLabel', providerLabel);
  const workspaceProvider = $('#videoProviderLabel');
  if (workspaceProvider) workspaceProvider.hidden = completedFirstCut;
  setTextIfPresent(
    '#videoIntro',
    !providerKnown
      ? 'Checking the configured video provider…'
      : vertex
        ? 'Motion Preview stays local. First Cut uses one explicitly approved, server-selected Veo request.'
        : 'Motion Preview stays local. These isolated video states use a bundled fixture and never call Veo.',
  );
  setTextIfPresent(
    '#firstCutProviderLabel',
    completedFirstCut && (videoStates.first_cut.provider === 'vertex' || vertex)
      ? `Vertex AI Veo · ${videoStates.first_cut.model || videoModel || DEFAULT_VEO_MODEL} · 8 seconds`
      : !providerKnown ? 'Checking provider…' : vertex ? 'First Cut' : 'Demo generation',
  );
  setTextIfPresent(
    '#firstCutRequestProviderLabel',
    !providerKnown ? 'Checking provider…' : vertex ? 'Vertex AI Veo' : 'Demo generation',
  );
  setTextIfPresent(
    '#directorCutProviderLabel',
    'Unavailable in this demo',
  );
  const allowanceMessage = !providerKnown
    ? 'Checking allowance…'
    : vertex
      ? videoAllowance && Number.isFinite(videoAllowance.authorized_replacement_remaining)
        ? hasAvailableReplacementAuthorization()
          ? `Normal First Cut allowance: ${videoAllowance.global_used} used / ${videoAllowance.global_remaining} remaining. Authorized replacement: ${videoAllowance.authorized_replacement_remaining} remaining. Authorized replacement attempt available.`
          : `Normal First Cut allowance: ${videoAllowance.global_used} used / ${videoAllowance.global_remaining} remaining. Authorized replacement: ${videoAllowance.authorized_replacement_remaining} remaining.`
        : 'First Cut allowance exhausted.'
      : 'Demo generation limit reached.';
  setTextIfPresent('#videoAllowanceMessage', allowanceMessage);
  const allowanceElement = $('#videoAllowanceMessage');
  if (allowanceElement && vertex && videoAllowance) {
    allowanceElement.hidden = completedFirstCut;
  }
}

function renderGenerateVeoButton() {
  const button = $('#generateVeo');
  if (!button) return;
  const needsGenerativeVideo = data?.routing?.classification === 'GENERATIVE_VIDEO_REQUIRED';
  button.disabled = !(
    needsGenerativeVideo
      && canOpenFirstCutApproval()
  );
}

function setStatusLabel(selector, text, state) {
  const status = $(selector);
  if (!status) return;
  const label = status.querySelector('.status-label');
  if (label) label.textContent = text;
  else status.textContent = text;
  if (state) status.dataset.state = state;
}

function setMode(mode) {
  setStatusLabel('#mode', modeLabels[mode] || mode, mode);
  $('#mode').dataset.mode = mode;
}

function setRequestState(state, label, detail = '') {
  const status = $('#requestStatus');
  const stateLabel = $('#stateLabel');
  const stageState = $('#stageState');
  const stage = $('#stage');
  const labels = {
    ready: 'READY',
    loading: 'ANALYSING COMPOSITION',
    complete: 'COMPLETE',
    fallback: 'RATE-LIMITED',
    error: 'ERROR',
  };
  setStatusLabel('#requestStatus', label || labels[state] || state, state);
  stateLabel.textContent = detail || (state === 'complete' ? 'MOTION PREVIEW READY' : labels[state]);
  stateLabel.dataset.state = state;
  stage.dataset.state = state === 'ready'
    ? (stage.classList.contains('has-upload') ? 'image-ready' : 'empty')
    : state === 'complete' || state === 'fallback'
      ? 'completed'
      : state;
  if (state === 'loading' || state === 'error' || state === 'empty') {
    stageState.querySelector('strong').textContent = state === 'loading'
      ? 'Directing the scene…'
      : state === 'error'
        ? 'Direction unavailable'
        : 'Upload a frame to begin.';
    stageState.querySelector('small').textContent = state === 'loading'
      ? 'Analysing the brief and compiling a Motion Preview.'
      : state === 'error'
        ? detail || 'Check the brief and try again.'
        : 'Or direct the bundled scene from the brief.';
  }
}

function setPassSelection(revised) {
  const firstPass = $('#firstPass');
  const version = $('#version');
  firstPass.classList.toggle('active', !revised);
  version.classList.toggle('active', revised);
  firstPass.setAttribute('aria-selected', String(!revised));
  version.setAttribute('aria-selected', String(revised));
  version.disabled = !cutPlans.revised;
}

function renderOutputMode() {
  const firstReady = videoStates.first_cut.status === 'completed';
  const motionButton = $('#motionPreviewOutput');
  const firstButton = $('#firstCutOutput');
  const motionPane = $('#motionPreviewOutputPane');
  const firstPane = $('#firstCutOutputPane');
  if (!motionButton || !firstButton || !motionPane || !firstPane) return;
  if (!firstReady && outputMode === 'first_cut') outputMode = 'motion_preview';
  firstButton.hidden = !firstReady;
  firstButton.disabled = !firstReady;
  const firstSelected = firstReady && outputMode === 'first_cut';
  motionButton.classList.toggle('active', !firstSelected);
  firstButton.classList.toggle('active', firstSelected);
  motionButton.setAttribute('aria-selected', String(!firstSelected));
  firstButton.setAttribute('aria-selected', String(firstSelected));
  motionPane.hidden = firstSelected;
  firstPane.hidden = !firstSelected;
  const generationDetails = $('#generationDetailsPanel');
  if (generationDetails) {
    generationDetails.hidden = !firstSelected;
    if (firstSelected) renderGenerationDetails(videoStates.first_cut);
  }
  const scenePlanContent = $('#scenePlanContent');
  if (scenePlanContent) {
    scenePlanContent.hidden = firstSelected || !recoveredScenePlanAvailable;
  }
  renderWorkflowStages();
}

function selectOutput(mode) {
  if (mode === 'first_cut' && videoStates.first_cut.status !== 'completed') return;
  outputMode = mode;
  renderOutputMode();
}

function renderWorkflowStages() {
  const stages = [
    $('#workflowDirect'),
    $('#workflowMotion'),
    $('#workflowFirstCut'),
  ];
  const activeIndex = videoStates.first_cut.status === 'completed' && outputMode === 'first_cut'
    ? 2
    : data
      ? 1
      : 0;
  stages.forEach((stageNode, index) => {
    if (!stageNode) return;
    stageNode.dataset.state = index < activeIndex
      ? 'complete'
      : index === activeIndex
        ? 'active'
        : '';
    stageNode.classList.toggle('active', index === activeIndex);
  });
}

function renderGenerationDetails(state) {
  setTextIfPresent(
    '#generationDetailStatus',
    state?.status === 'completed' ? 'Completed' : VIDEO_STATUS_LABELS[state?.status] || 'Unavailable',
  );
  setTextIfPresent(
    '#generationDetailProvider',
    state?.provider === 'vertex' || state?.output?.source === 'vertex_veo'
      ? 'Vertex AI Veo'
      : 'Demo generation',
  );
  setTextIfPresent('#generationDetailModel', state?.model || videoModel || DEFAULT_VEO_MODEL);
  const video = $('#firstCutVideo');
  const duration = video && Number.isFinite(video.duration) ? Math.round(video.duration) : 8;
  setTextIfPresent('#generationDetailDuration', `${duration} seconds`);
}

function renderRecoveredSceneView(sceneSnapshot) {
  const hasDirectionResponse = Boolean(
    sceneSnapshot?.direction_response?.plan
      && sceneSnapshot.direction_response.critique
      && sceneSnapshot.direction_response.routing,
  );
  recoveredScenePlanAvailable = hasDirectionResponse;
  if (typeof sceneSnapshot?.screenplay === 'string') {
    $('#screenplay').value = sceneSnapshot.screenplay;
  }
  if (typeof sceneSnapshot?.creative_intent === 'string') {
    $('#mood').value = sceneSnapshot.creative_intent;
  }
  if (hasDirectionResponse) {
    renderDirectedResult(
      sceneSnapshot.direction_response,
      {
        scene_key: sceneSnapshot.scene_key,
        source_signature: sceneSnapshot.source_signature,
      },
    );
  } else {
    setRequestState('complete', 'COMPLETE', 'GENERATED VIDEO READY');
    const source = $('#analysisSource');
    if (source) {
      source.hidden = true;
      source.dataset.state = 'complete';
    }
    conservativePlanConfirmed = false;
    data = null;
    renderVideoProviderCopy();
  }
}

function resetProgress() {
  // Progress activity remains available in the response state; the redundant
  // development strip is intentionally not rendered in the product surface.
}

function renderProgress(activity = []) {
  const detailByStep = new Map();
  activity.forEach(item => {
    const step = item.step === 'ANALYSE' ? 'ANALYSE' : item.step;
    if (['ANALYSE', 'ROUTE', 'DIRECT', 'ANIMATE', 'CRITIQUE', 'REVISE'].includes(step)) {
      detailByStep.set(step, item.detail);
    }
    if (item.step === 'CAMERA') {
      detailByStep.set('DIRECT', [detailByStep.get('DIRECT'), item.detail].filter(Boolean).join(' · '));
    }
  });
  const steps = [...document.querySelectorAll('.progress-step')];
  if (!steps.length) return;
  if (!activity.length) {
    resetProgress();
    return;
  }
  steps.forEach(step => {
    const name = step.dataset.step;
    const detail = detailByStep.get(name);
    step.dataset.state = detail ? 'complete' : '';
    if (detail) step.title = detail;
    step.onclick = () => {
      if (!detail) return;
      const detailPanel = $('#progressDetail');
      detailPanel.textContent = detail;
      detailPanel.hidden = false;
    };
  });
  const extras = activity
    .filter(item => !['ANALYSE', 'ROUTE', 'DIRECT', 'ANIMATE', 'CRITIQUE', 'REVISE', 'CAMERA'].includes(item.step))
    .map(item => item.detail);
  const summary = extras[0] || 'Motion Preview compiled';
  $('#progressSummary').textContent = summary;
  const firstDetail = detailByStep.get('ANALYSE') || detailByStep.get('ROUTE');
  if (firstDetail) {
    $('#progressDetail').textContent = firstDetail;
    $('#progressDetail').hidden = false;
  }
}

function renderLoadingProgress() {
  const first = document.querySelector('.progress-step[data-step="ANALYSE"]');
  if (!first) return;
  resetProgress();
  first.dataset.state = 'active';
  $('#progressSummary').textContent = 'Analysing composition';
}

function renderResponseStatus(result) {
  const cached = result.activity.some(item => item.step === 'CACHE');
  const rateLimited = result.mode === 'rate_limited_fallback';
  const deterministicFallback = result.analysis_source === 'deterministic_fallback';
  const legacy = isLegacyDirection(result);
  const source = $('#analysisSource');
  if (source) {
    source.textContent = legacy
      ? 'Legacy/fallback result'
      : result.analysis_source === 'deterministic_fallback'
        ? 'Deterministic fallback'
        : 'Vertex multimodal';
    source.dataset.state = legacy || result.analysis_source === 'deterministic_fallback'
      ? 'fallback'
      : 'complete';
    source.hidden = false;
  }
  setTextIfPresent(
    '#analysisSourcePlan',
    result.analysis_source === 'deterministic_fallback' ? 'FALLBACK' : 'GROUNDED',
  );
  if (rateLimited || deterministicFallback || legacy) {
    setRequestState('fallback', 'LOCAL FALLBACK', 'LOCAL FALLBACK READY');
    const fallbackDetail = result.activity.find(item => item.step === 'FALLBACK')?.detail;
    setTextIfPresent('#progressSummary', fallbackDetail || 'Local fallback ready');
  } else {
    setRequestState('complete', 'COMPLETE', 'MOTION PREVIEW READY');
    setTextIfPresent('#progressSummary', 'Motion Preview ready · Director’s Cut ready after approval');
  }
}

const VIDEO_STATUS_LABELS = {
  not_generated: 'NOT GENERATED',
  approval_required: 'APPROVAL REQUIRED',
  queued: 'QUEUED',
  generating: 'GENERATING',
  submission_unknown: 'STATUS UNCERTAIN',
  completed: 'COMPLETED',
  failed: 'FAILED',
};

function videoCardIds(kind) {
  return kind === 'first_cut'
    ? {
      card: '[data-video-kind="first_cut"]',
      status: '#firstCutStatus',
      video: '#firstCutVideo',
      placeholder: '#firstCutPlaceholder',
      message: '#firstCutMessage',
      button: '#generateFirstCut',
    }
    : {
      card: '[data-video-kind="director_cut"]',
      status: '#directorCutStatus',
      video: '#directorCutVideo',
      placeholder: '#directorCutPlaceholder',
      message: '#directorCutMessage',
      button: '#generateDirectorCut',
    };
}

function videoStatusMessage(kind, state) {
  const providerLabel = state.provider === 'vertex' || state.output?.source === 'vertex_veo'
    ? 'Vertex AI Veo'
    : 'Demo generation';
  if (state.status === 'not_generated') {
    if (kind === 'director_cut') {
      return 'A revised cut can be generated after First Cut critique when additional Veo allowance is available.';
    }
    return kind === 'first_cut'
      ? (isVertexVideoProvider()
        ? 'Approve a First Cut to begin.'
        : 'Approve a mocked First Cut to begin.')
      : 'Complete and approve the First Cut revision first.';
  }
  if (state.status === 'approval_required') return 'Explicit approval is required; no job has been created.';
  if (state.status === 'submission_unknown') {
    return 'Submission status uncertain. No retry sent; allowance remains reserved.';
  }
  if (state.status === 'queued') return `${providerLabel} queued by the server.`;
  if (state.status === 'generating') {
    return 'Generation is still processing.';
  }
  if (state.status === 'completed') {
    return providerLabel === 'Demo generation'
      ? 'Demo generation complete · bundled fixture video.'
      : 'Vertex AI Veo generation complete.';
  }
  return state.error?.message || state.error || 'Video generation failed. This job is terminal.';
}

function renderVideoMessage(kind, state) {
  const ids = videoCardIds(kind);
  const message = $(ids.message);
  if (!message) return;
  message.replaceChildren(document.createTextNode(videoStatusMessage(kind, state)));
  const error = state?.error;
  if (state?.status !== 'failed' || !error || typeof error !== 'object') return;
  const detail = error.provider_error_message
    || error.provider_error_category
    || error.provider_error_status
    || error.provider_error_code !== null
    && error.provider_error_code !== undefined;
  if (!detail) return;

  const disclosure = document.createElement('details');
  disclosure.className = 'video-error-detail';
  const summary = document.createElement('summary');
  summary.textContent = 'Provider detail';
  disclosure.appendChild(summary);
  const body = document.createElement('span');
  const fields = [];
  if (error.provider_error_code !== null && error.provider_error_code !== undefined) {
    fields.push(`Code ${error.provider_error_code}`);
  }
  if (error.provider_error_status) fields.push(`Status ${error.provider_error_status}`);
  if (error.provider_error_category) fields.push(`Category ${error.provider_error_category}`);
  if (error.provider_error_message) fields.push(error.provider_error_message);
  body.textContent = fields.join(' · ');
  disclosure.appendChild(body);
  message.appendChild(disclosure);
}

function renderVideoState(kind) {
  const state = videoStates[kind];
  const ids = videoCardIds(kind);
  const card = $(ids.card);
  const button = $(ids.button);
  const critiqueButton = kind === 'first_cut' ? $('#analyseFirstCut') : null;
  const video = $(ids.video);
  const placeholder = $(ids.placeholder);
  if (!card) return;
  const request = kind === 'first_cut' ? $('#firstCutRequest') : null;
  const label = kind === 'director_cut'
    ? 'Director’s Cut — Optional'
    : VIDEO_STATUS_LABELS[state.status] || 'FAILED';
  if (kind === 'director_cut') {
    card.hidden = videoStates.first_cut.status !== 'completed';
    card.dataset.status = state.status;
    $(ids.status).textContent = label;
    setTextIfPresent(
      ids.message,
      'Unavailable in this demo',
    );
    renderOutputMode();
    renderVideoProviderCopy();
    renderGenerateVeoButton();
    return;
  }
  if (!button || !video || !placeholder) return;
  if (request) request.hidden = state.status === 'completed';
  const directorAllowanceUnavailable = kind === 'director_cut'
    && state.status === 'not_generated'
    && isVertexVideoProvider()
    && videoAllowance?.director_cut_global_remaining === 0;
  card.dataset.status = state.status;
  card.dataset.optional = String(directorAllowanceUnavailable);
  $(ids.status).textContent = label;
  placeholder.textContent = directorAllowanceUnavailable
    ? 'No Director’s Cut allowance available'
    : label === 'NOT GENERATED' ? 'Not generated' : label;
  video.hidden = state.status !== 'completed';
  placeholder.hidden = state.status === 'completed' || directorAllowanceUnavailable;
  if (state.status === 'completed' && state.output?.url && video.src !== new URL(state.output.url, window.location.href).href) {
    video.src = `${state.output.url}?job=${encodeURIComponent(state.job_id)}`;
  }
  if (state.status === 'completed') outputMode = 'first_cut';
  const busy = ['queued', 'generating'].includes(state.status);
  const firstReady = videoStates.first_cut.status === 'completed';
  const directorReady = firstReady && revisionApproved;
  const globalExhausted = kind === 'first_cut'
    ? firstCutAllowanceUnavailable()
    : realAllowanceUnavailable();
  const directorExhausted = videoAllowance?.director_cut_global_remaining === 0
    && videoProvider === 'vertex';
  const firstCutEligible = kind === 'first_cut' && canOpenFirstCutApproval();
  if (kind === 'first_cut') {
    button.disabled = !firstCutEligible;
    button.hidden = state.status === 'completed';
  }
  if (kind === 'first_cut' && !firstCutEligible) {
    $(ids.message).replaceChildren(document.createTextNode(firstCutDisabledReason()));
  } else {
    renderVideoMessage(kind, state);
  }
  if (critiqueButton) {
    critiqueButton.disabled = !state.job_id || state.status !== 'completed' || videoCritiqueBusy;
  }
  if (kind === 'first_cut' && state.status === 'completed') {
    const allowance = $('#videoAllowanceMessage');
    if (allowance) allowance.hidden = true;
    const workspaceProvider = $('#videoProviderLabel');
    if (workspaceProvider) workspaceProvider.hidden = true;
    setTextIfPresent(
      '#firstCutProviderLabel',
      state.provider === 'vertex'
        ? `Vertex AI Veo · ${state.model || videoModel || DEFAULT_VEO_MODEL} · 8 seconds`
        : 'Demo generation · 8 seconds',
    );
  }
  renderOutputMode();
  renderVideoProviderCopy();
  renderGenerateVeoButton();
}

function formatVideoTimestamp(seconds) {
  if (seconds === null || seconds === undefined) return 'Timestamp unavailable';
  const total = Math.max(0, Number(seconds));
  const minutes = Math.floor(total / 60);
  const remainder = (total - minutes * 60).toFixed(1).padStart(4, '0');
  return `${String(minutes).padStart(2, '0')}:${remainder}`;
}

function renderVideoCritique(result) {
  videoCritique = result?.status === 'available' ? result : null;
  const available = result?.status === 'available';
  const panel = $('#videoCritiquePanel');
  const label = $('#videoCritiqueLabel');
  const notice = $('#videoCritiqueNotice');
  const message = $('#videoCritiqueMessage');
  const groups = $('#videoCritiqueGroups');
  if (!panel || !label || !notice || !message || !groups) return;
  panel.hidden = !available;
  if (available) panel.open = true;
  groups.replaceChildren();
  if (!available) {
    label.textContent = 'NOT ANALYSED';
    notice.textContent = '';
    message.textContent = 'Analyse the completed generated video to inspect the actual footage.';
    return;
  }
  label.textContent = 'AVAILABLE';
  notice.textContent = result.notice || '';
  message.textContent = result.message || 'Video critique unavailable.';
  if (!available) return;

  const sections = [
    ['Requested actions achieved', result.requested_actions_achieved],
    ['Requested actions missing', result.requested_actions_missing],
    ['Character / object consistency', result.character_object_consistency],
    ['Camera movement / pacing', result.camera_movement_pacing],
    ['Visible artifacts', result.visible_artifacts],
    ['Emotional intent alignment', result.emotional_intent_alignment],
    ['Revision recommendations', result.revision_recommendations],
  ];
  sections.forEach(([title, findings]) => {
    const section = document.createElement('section');
    section.className = 'video-critique-group';
    const heading = document.createElement('strong');
    heading.textContent = title;
    section.append(heading);
    if (!findings?.length) {
      const empty = document.createElement('p');
      empty.className = 'video-critique-empty';
      empty.textContent = 'No supported finding.';
      section.append(empty);
    } else {
      findings.forEach(finding => {
        const item = document.createElement('article');
        item.className = 'video-critique-finding';
        const support = document.createElement('small');
        support.textContent = finding.support.replaceAll('_', ' ');
        const text = document.createElement('p');
        text.textContent = finding.finding || finding.recommendation;
        item.append(support, text);
        (finding.evidence || []).forEach(evidence => {
          const evidenceText = document.createElement('small');
          evidenceText.textContent = `${formatVideoTimestamp(evidence.timestamp_seconds)} · ${evidence.description}`;
          item.append(evidenceText);
        });
        const uncertainty = document.createElement('small');
        uncertainty.textContent = finding.uncertainty;
        item.append(uncertainty);
        section.append(item);
      });
    }
    groups.append(section);
  });
}

function renderVideoAllowance(snapshot) {
  videoAllowance = snapshot;
  const message = $('#videoAllowanceMessage');
  renderVideoProviderCopy();
  const exhausted = isVertexVideoProvider()
    ? snapshot.global_remaining === 0 || snapshot.per_ip_remaining === 0
    : snapshot.global_remaining === 0;
  message.hidden = !isVertexVideoProvider();
  if (isVertexVideoProvider()) {
    if (Number.isFinite(snapshot.authorized_replacement_remaining)) {
      message.textContent =
        `Normal allowance: ${snapshot.global_used} used / ${snapshot.global_remaining} remaining. `
        + `Authorized replacement: ${snapshot.authorized_replacement_remaining} remaining.`
        + (hasAvailableReplacementAuthorization()
          ? ' Authorized replacement attempt available.'
          : '');
    } else {
      message.textContent = 'First Cut allowance exhausted.';
    }
  } else if (!exhausted) {
    message.textContent = '';
  }
  renderAllVideoStates();
  renderGenerateVeoButton();
}

async function refreshVideoAllowance() {
  try {
    const response = await fetch('/api/video-allowance');
    if (!response.ok) return;
    renderVideoAllowance(await response.json());
  } catch {
    // Allow the existing local/demo workflow to remain usable if status is unavailable.
  }
}

function renderAllVideoStates() {
  renderVideoState('first_cut');
  renderVideoState('director_cut');
}

function resetVideoStates() {
  Object.keys(videoPollers).forEach(kind => clearTimeout(videoPollers[kind]));
  Object.assign(videoStates, {
    first_cut: { status: 'not_generated' },
    director_cut: { status: 'not_generated' },
  });
  videoSceneKey = null;
  videoSourceSignature = null;
  revisionApproved = false;
  videoCritique = null;
  videoCritiqueBusy = false;
  durableFirstCutRestored = false;
  outputMode = 'motion_preview';
  $('#approveRevision').disabled = true;
  $('#videoCritiqueStatus').hidden = true;
  $('#videoCritiqueStatus').textContent = '';
  $('#critiqueAvailability').textContent = 'A completed First Cut will unlock video critique.';
  renderVideoCritique(null);
  renderAllVideoStates();
  renderOutputMode();
}

function setRecoveryMessage(message = '') {
  const node = $('#videoRecoveryMessage');
  if (!node) return;
  node.hidden = !message;
  node.textContent = message;
}

function clearRecoverySnapshot() {
  try {
    sessionStorage.removeItem(VIDEO_RECOVERY_STORAGE_KEY);
  } catch {
    // Storage may be unavailable in a restricted browser context.
  }
}

function saveRecoverySnapshot() {
  if (!data || !videoSceneKey || !videoSourceSignature) return;
  const snapshot = {
    version: VIDEO_RECOVERY_VERSION,
    scene: {
      scene_key: videoSceneKey,
      source_signature: videoSourceSignature,
    },
    screenplay: $('#screenplay').value,
    creative_intent: $('#mood').value,
    direction_response: data,
    video_job_ids: {
      first_cut: videoStates.first_cut.job_id || null,
      director_cut: videoStates.director_cut.job_id || null,
    },
  };
  try {
    sessionStorage.setItem(VIDEO_RECOVERY_STORAGE_KEY, JSON.stringify(snapshot));
  } catch {
    // Keep the live workflow usable if sessionStorage is unavailable or full.
  }
}

function readRecoverySnapshot() {
  let raw;
  try {
    raw = sessionStorage.getItem(VIDEO_RECOVERY_STORAGE_KEY);
  } catch {
    return null;
  }
  if (!raw) return null;
  try {
    const snapshot = JSON.parse(raw);
    const direction = snapshot?.direction_response;
    const scene = snapshot?.scene;
    const jobIds = snapshot?.video_job_ids;
    const validJobId = value => value === null || (
      typeof value === 'string' && value.length >= 8 && value.length <= 256
    );
    if (
      snapshot.version !== VIDEO_RECOVERY_VERSION
      || !scene
      || typeof scene.scene_key !== 'string'
      || typeof scene.source_signature !== 'string'
      || typeof snapshot.screenplay !== 'string'
      || snapshot.screenplay.length < 20
      || snapshot.screenplay.length > 8000
      || typeof snapshot.creative_intent !== 'string'
      || snapshot.creative_intent.length < 1
      || snapshot.creative_intent.length > 500
      || !direction?.plan
      || !direction?.critique
      || !direction?.routing
      || !Array.isArray(direction?.activity)
      || !jobIds
      || typeof jobIds !== 'object'
      || Array.isArray(jobIds)
      || !validJobId(jobIds.first_cut ?? null)
      || !validJobId(jobIds.director_cut ?? null)
    ) {
      throw new Error('invalid recovery snapshot');
    }
    return snapshot;
  } catch {
    clearRecoverySnapshot();
    setRecoveryMessage('Saved recovery data was outdated or invalid and was discarded. Direct the scene again.');
    return null;
  }
}

function appendRecoveryMessage(message) {
  const existing = $('#videoRecoveryMessage')?.textContent;
  setRecoveryMessage([existing, message].filter(Boolean).join(' '));
}

function hashText(value) {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return `scene-${(hash >>> 0).toString(16)}`;
}

function setVideoSceneIdentity(result, identity = null) {
  if (identity?.scene_key && identity?.source_signature) {
    videoSceneKey = identity.scene_key;
    videoSourceSignature = identity.source_signature;
    return;
  }
  videoSceneKey = hashText(JSON.stringify({
    summary: result.plan.scene_summary,
    intent: result.plan.emotional_intent,
    image: Boolean(imageData),
  }));
  videoSourceSignature = hashText(JSON.stringify({
    shot: result.plan.shot,
    motionPlan: result.plan.motion_plan,
    revision: result.critique.revision,
  }));
}

function videoContextPayload() {
  return {
    screenplay: $('#screenplay').value,
    creative_intent: $('#mood').value,
    shot_plan: data.plan,
    image_handle: data.image_handle || null,
    video_critique: videoCritique?.status === 'available' ? videoCritique : null,
  };
}

function renderDirectedResult(result, identity = null) {
  recoveredScenePlanAvailable = true;
  data = result;
  conservativePlanConfirmed = currentMotionPlan(result)?.confirmation_required === false;
  resetVideoStates();
  setVideoSceneIdentity(result, identity);
  setCutPlans(data);
  stage.classList.remove('revised');
  setPassSelection(false);
  $('#intent').textContent = data.plan.emotional_intent;
  $('#rationale').textContent = data.plan.directing_rationale;
  $('#camera').textContent = data.plan.shot.camera_motion.replaceAll('_', ' ');
  renderCameraMetadata(data.camera_grammar, data.shot_signature, data.diversity_adjustment);
  renderRouting(data.routing);
  renderMotionPlan();
  setParams(cutPlans.first, 'first');
  applyDepthLayout(data.plan.depth_layout, data.depth_source);
  applyEffects(data.plan.shot.atmosphere, data.plan.shot.transition);
  $('#diagnosis').textContent = data.critique.diagnosis;
  $('#revisionText').textContent = data.critique.revision_rationale;
  $('#scores').innerHTML = [
    ['Focus', data.critique.focus_score],
    ['Pacing', data.critique.pacing_score],
    ['Motion', data.critique.cinematic_motion_score],
    ['Restraint', data.critique.restraint_score],
  ].map(score => `<div><span>${score[0]}</span><b>${score[1]}</b></div>`).join('');
  renderRevisionChanges();
  renderProgress(data.activity);
  renderResponseStatus(data);
  const canPreview = data.routing.classification === 'LOCAL_2_5D';
  $('#apply').disabled = !canPreview;
  renderAllVideoStates();
  return canPreview;
}

function configureVideoApprovalDialog(kind) {
  const vertexFirstCut = isVertexVideoProvider() && kind === 'first_cut';
  const replacement = vertexFirstCut && Boolean(videoStates[kind]?.replacement_for_job_id);
  const dialog = $('#videoApprovalDialog');
  $('#videoApprovalTitle').textContent = vertexFirstCut
    ? 'Generate First Cut'
    : kind === 'first_cut'
      ? 'Approve demo generation'
      : 'Approve Director’s Cut';
  $('#videoApprovalCopy').textContent = vertexFirstCut
    ? 'Review the approved motion plan. No Veo request is sent until you choose Approve.'
    : kind === 'first_cut'
      ? 'This creates a mocked video job using only the bundled fixture video. No real Veo or Vertex video request will be made.'
      : 'This requests the server-selected Director’s Cut provider using the approved revision and scene context.';
  const replacementNotice = $('#videoReplacementNotice');
  replacementNotice.hidden = !replacement;
  replacementNotice.textContent = replacement
    ? 'This approval will use the authorized replacement attempt for the failed First Cut.'
    : '';
  $('#videoApprovalDetails').hidden = !vertexFirstCut;
  $('#videoApprovalWarning').hidden = vertexFirstCut;
  if (vertexFirstCut) {
    $('#videoMotionPlan').textContent = JSON.stringify(data.plan.motion_plan, null, 2);
    $('#videoGenerationModel').textContent = videoModel || DEFAULT_VEO_MODEL;
    const estimate = $('#videoGenerationEstimate');
    estimate.textContent = videoEstimate || '';
    estimate.hidden = !videoEstimate;
    $('#videoGenerationAllowance').textContent = replacement
      ? 'One authorized replacement Veo generation attempt will be used.'
      : 'One Veo generation request and the remaining First Cut allowance will be used.';
  }
  $('#confirmVideoApproval').textContent = vertexFirstCut ? 'Approve' : 'Approve and queue';
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
}

function openVideoApproval(kind) {
  if (kind === 'first_cut' && !canOpenFirstCutApproval()) return;
  if (kind === 'director_cut' && (
    !data
      || videoStates.first_cut.status !== 'completed'
      || !revisionApproved
      || realAllowanceUnavailable()
  )) return;
  pendingVideoKind = kind;
  const replacementForJobId = kind === 'first_cut'
    ? replacementAuthorizationJobId()
    : null;
  videoStates[kind] = {
    status: 'approval_required',
    provider: videoProvider,
    job_id: kind === 'first_cut' ? videoStates.first_cut.job_id : undefined,
    replacement_for_job_id: replacementForJobId,
  };
  renderVideoState(kind);
  configureVideoApprovalDialog(kind);
}

async function requestVideoApproval(kind) {
  const response = await fetch('/api/video-jobs/approval', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      kind,
      scene_key: videoSceneKey,
      source_signature: videoSourceSignature,
      first_cut_job_id: kind === 'director_cut' ? videoStates.first_cut.job_id : null,
      replacement_for_job_id: kind === 'first_cut'
        ? replacementAuthorizationJobId()
        : null,
      ...videoContextPayload(),
    }),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(videoErrorMessage(result));
  videoStates[kind] = result;
  renderVideoState(kind);
}

function videoErrorMessage(payload) {
  return payload?.detail?.message || payload?.detail || 'Unable to create the video job.';
}

async function pollVideoJob(kind, jobId) {
  try {
    const response = await fetch(`/api/video-jobs/${encodeURIComponent(jobId)}`);
    const result = await response.json();
    if (!response.ok) throw new Error(videoErrorMessage(result));
    videoStates[kind] = result;
    if (result.video_critique) {
      videoCritique = result.video_critique.status === 'available' ? result.video_critique : null;
      renderVideoCritique(result.video_critique);
    }
    renderVideoState(kind);
    saveRecoverySnapshot();
    if (result.status === 'completed' && kind === 'first_cut') {
      $('#critiqueAvailability').textContent = 'Generated video complete. Analyse the actual video or review the plan review.';
      $('#approveRevision').disabled = false;
      outputMode = 'first_cut';
      renderOutputMode();
    }
    if (['queued', 'generating'].includes(result.status)) {
      videoPollers[kind] = setTimeout(() => pollVideoJob(kind, jobId), 350);
    }
  } catch (error) {
    videoStates[kind] = { ...videoStates[kind], status: 'failed', error: error.message };
    renderVideoState(kind);
    saveRecoverySnapshot();
  }
}

function openVideoCritiqueApproval() {
  if (!videoStates.first_cut.job_id || videoStates.first_cut.status !== 'completed' || videoCritiqueBusy) return;
  const dialog = $('#videoCritiqueDialog');
  $('#videoCritiqueCopy').textContent =
    'This is an explicit request. In demo mode it returns deterministic fixture observations and does not inspect the footage. If live critique is enabled, it sends the completed generated video to Gemini with the original storyboard, screenplay and approved motion plan for comparison.';
  if (typeof dialog.showModal === 'function') dialog.showModal();
  else dialog.setAttribute('open', '');
}

async function analyseFirstCut() {
  if (!videoStates.first_cut.job_id || videoStates.first_cut.status !== 'completed' || videoCritiqueBusy) return;
  videoCritiqueBusy = true;
  renderVideoState('first_cut');
  $('#videoCritiqueStatus').hidden = false;
  $('#videoCritiqueStatus').textContent = 'Analysing generated video…';
  try {
    const response = await fetch(
      `/api/video-jobs/${encodeURIComponent(videoStates.first_cut.job_id)}/critique`,
      { method: 'POST' },
    );
    const result = await response.json();
    if (!response.ok) throw new Error(videoErrorMessage(result));
    renderVideoCritique(result);
    $('#videoCritiqueStatus').textContent = result.status === 'available'
      ? 'Video critique available in the Critique tab.'
      : 'Video critique unavailable.';
    $('#critiqueAvailability').textContent = result.status === 'available'
      ? 'Video critique complete. Supported findings can ground a Director’s Cut.'
      : 'Video critique unavailable. Plan review remains separate and available.';
    saveRecoverySnapshot();
  } catch {
    renderVideoCritique({
      status: 'unavailable',
      message: 'Video critique unavailable.',
      notice: '',
    });
    $('#videoCritiqueStatus').hidden = false;
    $('#videoCritiqueStatus').textContent = 'Video critique unavailable.';
    $('#critiqueAvailability').textContent = 'Video critique unavailable. Plan review remains separate and available.';
  } finally {
    videoCritiqueBusy = false;
    renderVideoState('first_cut');
  }
}

async function createVideoJob(kind) {
  const request = {
    kind,
    scene_key: videoSceneKey,
    source_signature: videoSourceSignature,
    ...videoContextPayload(),
    approval_id: videoStates[kind].approval_id,
    approved: true,
  };
  if (kind === 'director_cut') request.first_cut_job_id = videoStates.first_cut.job_id;
  if (kind === 'first_cut') {
    request.replacement_for_job_id = videoStates[kind].replacement_for_job_id || null;
  }
  videoStates[kind] = { status: 'queued' };
  renderVideoState(kind);
  try {
    const response = await fetch('/api/video-jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(videoErrorMessage(result));
    videoStates[kind] = result;
    renderVideoState(kind);
    saveRecoverySnapshot();
    refreshVideoAllowance();
    pollVideoJob(kind, result.job_id);
  } catch (error) {
    videoStates[kind] = { ...videoStates[kind], status: 'failed', error: error.message };
    refreshVideoAllowance();
    renderVideoState(kind);
    saveRecoverySnapshot();
  }
}

async function confirmVideoApproval() {
  const kind = pendingVideoKind;
  if (!kind) return;
  pendingVideoKind = null;
  const confirmButton = $('#confirmVideoApproval');
  confirmButton.disabled = true;
  const dialog = $('#videoApprovalDialog');
  if (typeof dialog.close === 'function') dialog.close();
  else dialog.removeAttribute('open');
  try {
    await requestVideoApproval(kind);
    await createVideoJob(kind);
  } catch (error) {
    videoStates[kind] = { status: 'failed', error: error.message };
    renderVideoState(kind);
  } finally {
    confirmButton.disabled = false;
  }
}

async function approveRevisionForVideo() {
  const jobId = videoStates.first_cut.job_id;
  if (!jobId) return;
  const button = $('#approveRevision');
  button.disabled = true;
  try {
    const response = await fetch(`/api/video-jobs/${encodeURIComponent(jobId)}/approve-revision`, {
      method: 'POST',
    });
    const result = await response.json();
    if (!response.ok) throw new Error(videoErrorMessage(result));
    videoStates.first_cut = result;
    revisionApproved = true;
    $('#critiqueAvailability').textContent = 'Revision approved. Director’s Cut generation is now available.';
    button.textContent = 'Revision approved';
    renderAllVideoStates();
    saveRecoverySnapshot();
  } catch (error) {
    button.disabled = false;
    $('#critiqueAvailability').textContent = error.message;
  }
}

function formatShotSignature(signature) {
  if (!signature) return 'STANDBY';
  const atmosphere = signature.atmosphere?.length
    ? signature.atmosphere.join('+')
    : 'none';
  const duration = Number(signature.duration_seconds).toFixed(1).replace(/\.0$/, '');
  return `${signature.motion_type.replaceAll('_', ' ')} · pan ${signature.pan_direction} · `
    + `zoom ${signature.zoom_direction} · ${duration}s · `
    + `parallax ${Math.round(signature.parallax_strength * 100)}% · fx ${atmosphere}`;
}

function renderCameraMetadata(grammar, signature, adjustment) {
  $('#grammar').textContent = (grammar || 'balanced').toUpperCase();
  $('#shotSignature').textContent = formatShotSignature(signature);
  const note = $('#diversityNote');
  note.hidden = !adjustment;
  note.textContent = adjustment || '';
}

function renderDepthIndicator(composition, source = depthSource) {
  $('#depthSource').textContent = source === 'image_aware'
    ? 'IMAGE-AWARE DEPTH'
    : 'HEURISTIC DEPTH';
  $('#depthNote').textContent = source === 'image_aware'
    ? 'Gemini polygons · exclusive canvas masks · 16% overscan'
    : 'Deterministic image-space heuristic · 16% overscan';
  composition.planes.forEach(plane => {
    const row = document.querySelector(`[data-depth-plane="${plane.key}"]`);
    if (!row) return;
    row.style.setProperty('--plane-strength', `${plane.strength}%`);
    row.querySelector('[data-strength-value]').textContent = `${plane.strength}%`;
  });
  const path = composition.path;
  const pathParts = [];
  if (path.lateral_slide) pathParts.push('SLIDE');
  if (path.push_in) pathParts.push('PUSH-IN');
  if (path.pull_out) pathParts.push('PULL-OUT');
  if (path.vertical_slide) pathParts.push('VERTICAL');
  $('#depthPath').textContent = pathParts.length ? pathParts.join(' + ') : 'HOLD';
}

function applyDepthLayout(layout, source = 'heuristic') {
  const uploadedScene = $('#uploadedScene');
  const candidateLayout = compositor.normalizeDepthLayout(layout)
    || compositor.heuristicDepthLayout();
  const imageAware = source === 'image_aware'
    && Boolean(imageData)
    && compositor.applyDepthMasks(uploadedScene, candidateLayout);
  if (imageAware) {
    depthSource = 'image_aware';
    activeDepthLayout = candidateLayout;
  } else {
    activeDepthLayout = compositor.heuristicDepthLayout();
    compositor.applyDepthMasks(uploadedScene, activeDepthLayout);
    depthSource = 'heuristic';
  }
  compositor.applyDepthDebugOverlay(uploadedScene, activeDepthLayout);
  uploadedScene.dataset.depthSource = depthSource;
  if (lastComposition) renderDepthIndicator(lastComposition, depthSource);
}

function setCutPlans(result) {
  const first = compositor.freezeShotParams(result.plan.shot);
  const revised = compositor.ensureDistinctShotParams(first, result.critique.revision);
  cutPlans = Object.freeze({
    first,
    revised: compositor.freezeShotParams(revised),
  });
  cutEdits = { first: {}, revised: {} };
  selectedCut = 'first';
}

function setControlValues(params) {
  $('#duration').value = params.duration_seconds;
  $('#zoom').value = params.zoom_end;
  $('#parallax').value = params.parallax_strength;
  $('#motion').value = params.motion_intensity;
}

function applyActiveParams() {
  if (!activeParams) return;
  $('#durationOut').value = `${$('#duration').value}s`;
  $('#zoomOut').value = `${Number($('#zoom').value).toFixed(2)}×`;
  $('#parallaxOut').value = `${Math.round($('#parallax').value * 100)}%`;
  $('#motionOut').value = `${Math.round($('#motion').value * 100)}%`;
  stage.style.setProperty('--zoom', activeParams.zoom_end);
  stage.style.setProperty('--pan-x', activeParams.pan_x);
  stage.style.setProperty('--pan-y', activeParams.pan_y);
  stage.style.setProperty('--parallax', activeParams.parallax_strength);
  stage.style.setProperty('--motion', activeParams.motion_intensity);
  lastComposition = compositor.applyToStage(stage, activeParams);
  renderDepthIndicator(lastComposition, depthSource);
}

function setParams(params, cut = selectedCut) {
  activeParams = {
    ...params,
    ...(cutEdits[cut] || {}),
  };
  setControlValues(activeParams);
  applyActiveParams();
}

function sync() {
  if (!activeParams) return;
  const edits = {
    duration_seconds: Number($('#duration').value),
    zoom_end: Number($('#zoom').value),
    parallax_strength: Number($('#parallax').value),
    motion_intensity: Number($('#motion').value),
  };
  cutEdits[selectedCut] = {
    ...cutEdits[selectedCut],
    ...edits,
  };
  const sourcePlan = cutPlans[selectedCut] || activeParams;
  activeParams = {
    ...sourcePlan,
    ...cutEdits[selectedCut],
  };
  applyActiveParams();
}

['duration', 'zoom', 'parallax', 'motion'].forEach(id => {
  $(`#${id}`).addEventListener('input', sync);
});

function applyEffects(names = [], transition = 'fade') {
  activeEffects = [...new Set(names)];
  ['rain', 'fog', 'dust', 'embers', 'light_flicker'].forEach(name => {
    stage.classList.toggle(`fx-${name}`, activeEffects.includes(name));
  });
  stage.dataset.transition = transition;
}

function formatSigned(value, decimals = 0) {
  const rounded = Number(value).toFixed(decimals);
  return Number(value) > 0 ? `+${rounded}` : rounded;
}

function formatChangeValue(key, value) {
  if (key === 'duration_seconds') return `${Number(value).toFixed(1).replace(/\.0$/, '')}s`;
  if (key === 'zoom_end') return `${Number(value).toFixed(2)}×`;
  if (key === 'pan_x') return `${formatSigned(value)} horizontal`;
  if (key === 'pan_y') return `${formatSigned(value)} vertical`;
  if (key === 'parallax_strength' || key === 'motion_intensity') {
    return `${Math.round(Number(value) * 100)}%`;
  }
  if (key === 'camera_motion') return String(value).replaceAll('_', ' ');
  if (key === 'easing') return String(value).replace('cubic-bezier', 'bezier');
  return String(value);
}

function formatDisplacement(plane) {
  return `${formatSigned(plane.end.x, 1)}px x · ${formatSigned(plane.end.y, 1)}px y`;
}

function renderRevisionChanges() {
  const panel = $('#revisionChanges');
  const list = $('#revisionChangeList');
  if (!panel || !list || !cutPlans.first || !cutPlans.revised) return;

  const changes = compositor.compareShotParams(cutPlans.first, cutPlans.revised).map(change => ({
    label: {
      duration_seconds: 'Duration',
      zoom_end: 'Zoom',
      pan_x: 'Pan X',
      pan_y: 'Pan Y',
      parallax_strength: 'Parallax',
      motion_intensity: 'Motion',
      camera_motion: 'Camera',
      easing: 'Easing',
    }[change.key] || change.key,
    first: formatChangeValue(change.key, change.first),
    revised: formatChangeValue(change.key, change.revised),
  }));
  const firstComposition = compositor.calculateCompositorParams(cutPlans.first);
  const revisedComposition = compositor.calculateCompositorParams(cutPlans.revised);
  const planeLabels = { front: 'Foreground travel', mid: 'Subject travel', back: 'Background travel' };
  firstComposition.planes.forEach((firstPlane, index) => {
    const revisedPlane = revisedComposition.planes[index];
    if (
      Math.abs(firstPlane.end.x - revisedPlane.end.x) < 0.01
      && Math.abs(firstPlane.end.y - revisedPlane.end.y) < 0.01
    ) return;
    changes.push({
      label: planeLabels[firstPlane.key],
      first: formatDisplacement(firstPlane),
      revised: formatDisplacement(revisedPlane),
    });
  });

  list.replaceChildren();
  changes.forEach(change => {
    const item = document.createElement('li');
    const label = document.createElement('span');
    const values = document.createElement('b');
    label.textContent = change.label;
    values.textContent = `${change.first} → ${change.revised}`;
    item.append(label, values);
    list.append(item);
  });
  panel.hidden = changes.length === 0;
}

function renderRouting(routing) {
  const card = $('#routingCard');
  const approval = $('#approvalPanel');
  const useLocal = $('#useLocal');
  const generateVeo = $('#generateVeo');
  card.hidden = !routing;
  if (!routing) return;

  const label = routing.classification.replaceAll('_', ' ');
  card.dataset.classification = routing.classification;
  $('#routingDecision').textContent = label;
  $('#routingRationale').textContent = routing.rationale;
  const needsApproval = routing.classification === 'GENERATIVE_VIDEO_REQUIRED';
  const localSelected = Boolean(routing.local_approximation_selected);
  approval.hidden = !needsApproval || localSelected;
  useLocal.hidden = !needsApproval || localSelected;
  generateVeo.hidden = !needsApproval || localSelected;
  $('#routingStatus').hidden = !localSelected;
  renderGenerateVeoButton();
}

function renderMotionCandidates(plan) {
  const entityPanel = $('#sceneEntities');
  if (entityPanel) entityPanel.hidden = false;
  const groups = [
    ['cameraMotionPlan', [plan.camera_movement]],
    ['charactersPlan', plan.movable_characters],
    ['objectsPlan', plan.movable_objects],
    ['environmentPlan', plan.environmental_motion],
  ];
  groups.forEach(([targetId, candidates]) => {
    const container = $(`#${targetId}`);
    if (!container) return;
    container.replaceChildren();
    if (!candidates?.length) {
      const empty = document.createElement('span');
      empty.className = 'plan-empty';
      empty.textContent = 'None confidently identified';
      container.append(empty);
    } else {
      candidates.forEach(candidate => {
        const item = document.createElement('article');
        item.className = 'plan-item';
        const title = document.createElement('div');
        title.className = 'plan-item-title';
        const name = document.createElement('b');
        name.textContent = candidate.label;
        title.append(name);
        const action = document.createElement('p');
        action.textContent = candidate.action;
        const details = document.createElement('details');
        const summary = document.createElement('summary');
        summary.textContent = 'Confidence & evidence';
        const confidence = document.createElement('small');
        confidence.textContent = `${Math.round(candidate.confidence * 100)}% confidence · ${candidate.support.replaceAll('_', ' ')}`;
        const evidence = document.createElement('small');
        evidence.textContent = `Visual: ${candidate.visual_evidence} Screenplay: ${candidate.screenplay_evidence}`;
        const bound = document.createElement('small');
        bound.textContent = `Limit: ${candidate.bound}`;
        details.append(summary, confidence, evidence, bound);
        item.append(title, action, details);
        container.append(item);
      });
    }
  });
}

function renderMotionPlan() {
  const plan = currentMotionPlan();
  const panel = $('#motionPlan');
  if (!panel || !plan) {
    if (panel) panel.hidden = true;
    return;
  }
  panel.hidden = false;
  $('#motionGrounding').textContent = plan.grounding_summary;
  $('#motionPlanStatus').textContent = plan.confirmation_required ? 'CONFIRMATION NEEDED' : 'GROUNDED';
  renderMotionCandidates(plan);
  const renderList = (selector, values) => {
    const list = $(selector);
    list.replaceChildren();
    (values || []).forEach(value => {
      const item = document.createElement('li');
      item.textContent = value;
      list.append(item);
    });
  };
  renderList('#motionPreserved', plan.preserved_elements);
  renderList('#motionProhibited', plan.prohibited_changes);
  $('#motionConfirmation').hidden = !plan.confirmation_required;
  $('#motionQuestion').textContent = plan.confirmation_question || '';
}

function startParticles(duration) {
  cancelAnimationFrame(particleFrame);
  const mode = ['rain', 'embers', 'dust'].find(effect => activeEffects.includes(effect));
  const canvas = $('#particleCanvas');
  if (!canvas) return;
  const box = canvas.getBoundingClientRect();
  const dpr = Math.min(devicePixelRatio || 1, 2);
  canvas.width = Math.round(box.width * dpr);
  canvas.height = Math.round(box.height * dpr);
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const width = box.width;
  const height = box.height;
  if (!mode) {
    ctx.clearRect(0, 0, width, height);
    return;
  }
  const intensity = Number($('#motion').value);
  const count = Math.round((mode === 'rain' ? 55 : 28) + intensity * (mode === 'rain' ? 65 : 45));
  const particles = Array.from({ length: count }, () => ({
    x: Math.random() * width,
    y: Math.random() * height,
    length: mode === 'rain' ? 4 + Math.random() * 12 : 1 + Math.random() * 3,
    velocity: mode === 'rain' ? 170 + Math.random() * 330 : 12 + Math.random() * 35,
    alpha: 0.08 + Math.random() * 0.28,
    wind: mode === 'rain' ? 18 + Math.random() * 28 : -12 + Math.random() * 24,
    phase: Math.random() * 6.28,
  }));
  const begun = performance.now();

  function frame(now) {
    ctx.clearRect(0, 0, width, height);
    const delta = Math.min(0.035, (now - (frame.last || now)) / 1000);
    frame.last = now;
    particles.forEach(particle => {
      if (mode === 'rain') {
        particle.y += particle.velocity * delta;
        particle.x += particle.wind * delta;
      } else if (mode === 'embers') {
        particle.y -= particle.velocity * delta;
        particle.x += Math.sin(now / 700 + particle.phase) * 10 * delta;
      } else {
        particle.y += particle.velocity * 0.12 * delta;
        particle.x += particle.wind * delta;
      }
      if (
        particle.y > height + 20 ||
        particle.x > width + 20 ||
        particle.y < -20 ||
        particle.x < -20
      ) {
        particle.y = mode === 'embers' ? height + 10 : -10 - Math.random() * height * 0.15;
        particle.x = Math.random() * width;
      }
      ctx.beginPath();
      if (mode === 'rain') {
        ctx.moveTo(particle.x, particle.y);
        ctx.lineTo(particle.x - particle.length * 0.16, particle.y + particle.length);
        ctx.strokeStyle = `rgba(205,226,238,${particle.alpha})`;
        ctx.lineWidth = 0.45 + particle.alpha * 1.8;
        ctx.stroke();
      } else {
        ctx.arc(particle.x, particle.y, particle.length, 0, Math.PI * 2);
        ctx.fillStyle = mode === 'embers'
          ? `rgba(255,143,48,${particle.alpha + 0.1})`
          : `rgba(220,205,170,${particle.alpha})`;
        ctx.fill();
      }
    });
    if (now - begun < duration) {
      particleFrame = requestAnimationFrame(frame);
    } else {
      ctx.clearRect(0, 0, width, height);
    }
  }

  particleFrame = requestAnimationFrame(frame);
}

function formatTime(seconds) {
  return `00:${String(Math.floor(seconds)).padStart(2, '0')}`;
}

function syncFirstCutTime(video) {
  if (!video || !Number.isFinite(video.duration)) return;
  $('#time').textContent = `${formatTime(video.currentTime)} / ${formatTime(video.duration)}`;
  if (videoStates.first_cut.status === 'completed' && outputMode === 'first_cut') {
    renderGenerationDetails(videoStates.first_cut);
  }
}

const firstCutVideoElement = $('#firstCutVideo');
if (firstCutVideoElement) {
  firstCutVideoElement.addEventListener('loadedmetadata', () => syncFirstCutTime(firstCutVideoElement));
  firstCutVideoElement.addEventListener('timeupdate', () => syncFirstCutTime(firstCutVideoElement));
}

function play() {
  clearInterval(timer);
  stage.classList.remove('playing');
  void stage.offsetWidth;
  stage.classList.add('playing');
  playbackRun += 1;
  stage.dataset.playbackRun = String(playbackRun);
  start = Date.now();
  const duration = Number(activeParams?.duration_seconds ?? $('#duration').value) * 1000;
  $('#progress').style.width = '0%';
  $('#time').textContent = `${formatTime(0)} / ${formatTime(Math.round(duration / 1000))}`;
  startParticles(duration);
  timer = setInterval(() => {
    const progress = Math.min(1, (Date.now() - start) / duration);
    $('#progress').style.width = `${progress * 100}%`;
    $('#time').textContent = `${formatTime(progress * duration / 1000)} / ${formatTime(Math.round(duration / 1000))}`;
    if (progress === 1) clearInterval(timer);
  }, 50);
}

$('#play').onclick = play;

$('#briefToggle').onclick = event => {
  const button = event.currentTarget;
  const expanded = button.getAttribute('aria-expanded') === 'true';
  const next = !expanded;
  button.setAttribute('aria-expanded', String(next));
  button.title = next ? 'Collapse scene brief' : 'Expand scene brief';
  button.querySelector('.sr-only').textContent = next ? 'Collapse scene brief' : 'Expand scene brief';
  $('#appShell').classList.toggle('brief-collapsed', !next);
};

$('#toggleDepthDebug').onclick = event => {
  const button = event.currentTarget;
  const enabled = button.getAttribute('aria-pressed') !== 'true';
  button.setAttribute('aria-pressed', String(enabled));
  button.textContent = enabled ? 'HIDE DEPTH REGIONS' : 'SHOW DEPTH REGIONS';
  stage.classList.toggle('show-depth-debug', enabled);
  compositor.applyDepthDebugOverlay($('#uploadedScene'), activeDepthLayout);
};

function resizeImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const objectUrl = typeof URL !== 'undefined' && URL.createObjectURL
      ? URL.createObjectURL(file)
      : null;
    const cleanup = () => {
      if (objectUrl && URL.revokeObjectURL) URL.revokeObjectURL(objectUrl);
    };
    image.onerror = () => {
      cleanup();
      reject(new Error('Unable to decode that storyboard image.'));
    };
    image.onload = () => {
      try {
        const longestEdge = Math.max(image.naturalWidth, image.naturalHeight);
        const scale = Math.min(1, 1024 / longestEdge);
        const width = Math.max(1, Math.round(image.naturalWidth * scale));
        const height = Math.max(1, Math.round(image.naturalHeight * scale));
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        canvas.getContext('2d').drawImage(image, 0, 0, width, height);
        resolve(canvas.toDataURL('image/jpeg', 0.75));
      } catch (error) {
        reject(new Error('Unable to read that storyboard image.'));
      } finally {
        cleanup();
      }
    };
    if (objectUrl) {
      image.src = objectUrl;
      return;
    }
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Unable to read that storyboard image.'));
    reader.onload = () => { image.src = reader.result; };
    reader.readAsDataURL(file);
  });
}

$('#image').onchange = event => {
  const file = event.target.files[0];
  if (!file) return;
  const request = ++uploadRequest;
  if (file.size > 8 * 1024 * 1024) {
    alert('Please choose an image smaller than 8 MB.');
    event.target.value = '';
    return;
  }
  imageData = null;
  resizeImage(file).then(resizedImage => {
    if (request !== uploadRequest) return;
    imageData = resizedImage;
    $('#uploadState').textContent = `${file.name} · ready`;
    const scene = $('#uploadedScene');
    scene.hidden = false;
    scene.querySelectorAll('.image-plane,.image-backing').forEach(plane => {
      plane.style.backgroundImage = `url(${resizedImage})`;
    });
    stage.classList.add('has-upload');
    applyEffects([]);
    stage.querySelectorAll(':scope > .layer,:scope > .moon,:scope > .fog,:scope > .shadow-wipe')
      .forEach(element => { element.style.display = 'none'; });
    applyDepthLayout(compositor.heuristicDepthLayout(), 'heuristic');
    setRequestState('ready', 'READY', 'FRAME READY');
  }).catch(error => {
    if (request !== uploadRequest) return;
    imageData = null;
    event.target.value = '';
    $('#uploadState').textContent = 'Unable to decode image';
    $('#uploadedScene').hidden = true;
    stage.classList.toggle('has-upload', false);
    setRequestState('error', 'ERROR', 'STORYBOARD IMAGE ERROR');
    alert(error.message);
  });
};

$('#direct').onclick = async () => {
  const button = $('#direct');
  clearRecoverySnapshot();
  setRecoveryMessage('');
  button.disabled = true;
  button.querySelector('span').textContent = 'Directing…';
  setRequestState('loading', 'ANALYSING COMPOSITION', 'ANALYSING COMPOSITION');
  renderLoadingProgress();
  try {
    const response = await fetch('/api/direct', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        screenplay: $('#screenplay').value,
        mood: $('#mood').value,
        image_data_url: imageData,
      }),
    });
    const result = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(result.detail)
        ? result.detail.map(error => `${error.loc?.at(-1) || 'field'}: ${error.msg}`).join('\n')
        : result.detail || 'Unable to direct this scene.';
      throw new Error(detail);
    }

    const canPreview = renderDirectedResult(result);
    renderAllVideoStates();
    saveRecoverySnapshot();
    if (canPreview) play();
  } catch (error) {
    setRequestState('error', 'ERROR', error.message);
    setTextIfPresent('#progressSummary', 'Direction failed');
    setTextIfPresent('#progressDetail', error.message);
    alert(error.message);
  } finally {
    button.disabled = false;
    button.querySelector('span').textContent = 'Direct scene';
  }
};

$('#useLocal').onclick = () => {
  if (!data?.routing) return;
  data.routing = {
    ...data.routing,
    local_approximation_selected: true,
  };
  renderRouting(data.routing);
  $('#apply').disabled = false;
  play();
};

$('#generateVeo').onclick = () => openVideoApproval('first_cut');
$('#generateFirstCut').onclick = () => openVideoApproval('first_cut');
$('#analyseFirstCut').onclick = openVideoCritiqueApproval;
$('#confirmVideoApproval').onclick = confirmVideoApproval;
$('#confirmVideoCritique').onclick = () => {
  const dialog = $('#videoCritiqueDialog');
  if (typeof dialog.close === 'function') dialog.close();
  else dialog.removeAttribute('open');
  analyseFirstCut();
};
['cancelVideoApproval', 'closeVideoApproval'].forEach(id => {
  $(`#${id}`).onclick = () => {
    const kind = pendingVideoKind;
    pendingVideoKind = null;
    const dialog = $('#videoApprovalDialog');
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
    if (kind) {
      videoStates[kind] = { status: 'not_generated' };
      renderVideoState(kind);
    }
  };
});
['cancelVideoCritique', 'closeVideoCritique'].forEach(id => {
  $(`#${id}`).onclick = () => {
    const dialog = $('#videoCritiqueDialog');
    if (typeof dialog.close === 'function') dialog.close();
    else dialog.removeAttribute('open');
  };
});

$('#approveRevision').onclick = approveRevisionForVideo;

$('#confirmMotion').onclick = () => {
  const canonicalPlan = currentMotionPlan();
  if (!canonicalPlan) return;
  conservativePlanConfirmed = true;
  canonicalPlan.confirmation_required = false;
  data.plan.motion_plan = canonicalPlan;
  renderMotionPlan();
  renderAllVideoStates();
  saveRecoverySnapshot();
  refreshVideoAllowance().then(() => {
    renderAllVideoStates();
    saveRecoverySnapshot();
  });
};

function selectPass(revised) {
  if (!data || !cutPlans.first || !cutPlans.revised) return;
  selectedCut = revised ? 'revised' : 'first';
  const params = cutPlans[selectedCut];
  stage.classList.toggle('revised', revised);
  setParams(params, selectedCut);
  applyEffects(activeParams.atmosphere, activeParams.transition);
  renderCameraMetadata(
    revised ? (data.revision_camera_grammar || data.camera_grammar) : data.camera_grammar,
    revised ? (data.revision_shot_signature || data.shot_signature) : data.shot_signature,
    revised ? data.revision_diversity_adjustment : data.diversity_adjustment,
  );
  setPassSelection(revised);
  setRequestState('complete', revised ? 'Preview revision ready' : 'Complete', revised ? 'MOTION PREVIEW REVISION READY' : 'MOTION PREVIEW READY');
  setTextIfPresent('#progressSummary', revised
    ? 'Motion Preview revision ready'
    : 'Motion Preview ready · Director’s Cut ready after approval');
  play();
}

$('#firstPass').onclick = () => selectPass(false);
$('#version').onclick = () => selectPass(true);
$('#apply').onclick = () => selectPass(true);
$('#motionPreviewOutput').onclick = () => selectOutput('motion_preview');
$('#firstCutOutput').onclick = () => selectOutput('first_cut');

function closeVideoDialogs() {
  pendingVideoKind = null;
  ['videoApprovalDialog', 'videoCritiqueDialog'].forEach(id => {
    const dialog = $(`#${id}`);
    if (!dialog) return;
    if (typeof dialog.close === 'function' && dialog.open) dialog.close();
    else dialog.removeAttribute('open');
  });
}

async function restoreVideoRecovery() {
  const snapshot = readRecoverySnapshot();
  if (!snapshot) return;

  closeVideoDialogs();
  videoCritiqueBusy = false;
  try {
    $('#screenplay').value = snapshot.screenplay;
    $('#mood').value = snapshot.creative_intent;
    imageData = null;
    $('#uploadState').textContent = snapshot.direction_response.image_handle
      ? 'Storyboard image preview requires re-upload after refresh'
      : 'No storyboard image saved';
    $('#uploadedScene').hidden = true;
    stage.classList.remove('has-upload');
    renderDirectedResult(snapshot.direction_response, snapshot.scene);
    setRecoveryMessage('');

    await refreshVideoAllowance();
    const jobEntries = Object.entries(snapshot.video_job_ids)
      .filter(([, jobId]) => jobId);
    const restoredJobs = await Promise.all(jobEntries.map(async ([kind, jobId]) => {
      try {
        const response = await fetch(`/api/video-jobs/${encodeURIComponent(jobId)}`);
        const result = await response.json();
        if (!response.ok) {
          return {
            kind,
            jobId,
            unavailable: true,
            message: result?.detail?.message || 'Saved video job is no longer available for this session.',
          };
        }
        return { kind, jobId, result };
      } catch {
        return {
          kind,
          jobId,
          unavailable: true,
          message: 'Saved video job could not be recovered. It was not regenerated.',
        };
      }
    }));

    const unavailableJobs = [];
    restoredJobs.forEach(({ kind, jobId, result, unavailable, message }) => {
      if (kind === 'first_cut' && durableFirstCutRestored) return;
      if (unavailable) {
        videoStates[kind] = {
          status: 'failed',
          error: message || 'Saved video job is no longer available for this session.',
        };
        unavailableJobs.push(`${kind === 'first_cut' ? 'First Cut' : 'Director’s Cut'} was not regenerated.`);
        return;
      }
      videoStates[kind] = result;
      if (result.video_critique) {
        videoCritique = result.video_critique.status === 'available'
          ? result.video_critique
          : null;
        renderVideoCritique(result.video_critique);
      }
    });

    revisionApproved = Boolean(
      videoStates.first_cut.revision_approved
      && videoStates.first_cut.status === 'completed',
    );
    renderAllVideoStates();
    saveRecoverySnapshot();

    const pendingJobs = restoredJobs.filter(({ result }) => (
      result && ['queued', 'generating'].includes(result.status)
    ));
    pendingJobs.forEach(({ kind, jobId }) => pollVideoJob(kind, jobId));

    if (data.image_handle) {
      try {
        const imageResponse = await fetch(
          `/api/storyboard-images/${encodeURIComponent(data.image_handle)}`,
        );
        if (!imageResponse.ok) {
          appendRecoveryMessage(
            'The uploaded storyboard image is no longer available for this session. '
            + 'Re-upload it before requesting another approved job.',
          );
        } else {
          appendRecoveryMessage(
            'The scene and video jobs were restored, but the storyboard image preview '
            + 'is not stored in the browser. Re-upload it to view the image again.',
          );
        }
      } catch {
        appendRecoveryMessage(
          'The uploaded storyboard image could not be verified after refresh. '
          + 'Re-upload it before requesting another approved job.',
        );
      }
    }
    if (unavailableJobs.length) appendRecoveryMessage(unavailableJobs.join(' '));
  } catch {
    clearRecoverySnapshot();
    resetVideoStates();
    closeVideoDialogs();
    setRecoveryMessage('Saved recovery data could not be restored. Direct the scene again.');
  }
}

async function restoreLatestCompletedFirstCut() {
  try {
    const response = await fetch('/api/video-jobs/latest?kind=first_cut');
    if (!response.ok) return;
    const result = await response.json();
    if (!result || result.status !== 'completed') return;

    durableFirstCutRestored = true;
    const sceneSnapshot = result.scene_snapshot;
    if (sceneSnapshot?.direction_response) {
      renderRecoveredSceneView(sceneSnapshot);
    } else {
      resetVideoStates();
      renderRecoveredSceneView(sceneSnapshot);
    }
    videoStates.first_cut = result;
    outputMode = 'first_cut';
    videoSceneKey = result.scene_key || videoSceneKey;
    videoSourceSignature = result.source_signature || videoSourceSignature;
    revisionApproved = Boolean(result.revision_approved);
    if (result.video_critique) {
      videoCritique = result.video_critique.status === 'available'
        ? result.video_critique
        : null;
      renderVideoCritique(result.video_critique);
    }
    setRequestState('complete', 'COMPLETE', 'GENERATED VIDEO READY');
    renderVideoState('first_cut');
    renderVideoState('director_cut');
    renderOutputMode();
    saveRecoverySnapshot();
  } catch {
    // Durable recovery is additive; the existing local recovery path remains usable.
  }
}

setRequestState('empty', 'READY', 'READY TO DIRECT');
resetProgress();
renderVideoProviderCopy();
refreshVideoAllowance();

fetch('/api/health')
  .then(response => response.json())
  .then(health => {
    setMode(health.mode);
    videoProvider = health.video_provider || 'mock';
    videoModel = health.video_model || DEFAULT_VEO_MODEL;
    videoEstimate = health.video_estimate || null;
    healthLoaded = true;
    renderVideoProviderCopy();
    renderAllVideoStates();
    renderGenerateVeoButton();
  })
  .catch(() => {});

restoreVideoRecovery().then(restoreLatestCompletedFirstCut);