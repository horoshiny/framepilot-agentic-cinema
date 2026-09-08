(function attachCompositor(root, factory) {
  const compositor = factory();
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = compositor;
  } else {
    root.FramePilotCompositor = compositor;
  }
}(typeof window !== 'undefined' ? window : globalThis, function createCompositor() {
  const DEPTH_PLANES = [
    { key: 'front', label: 'Foreground', travel: 1 },
    { key: 'mid', label: 'Subject / Midground', travel: 0.68 },
    { key: 'back', label: 'Background', travel: 0.34 },
  ];
  const CAMERA_DIRECTIONS = {
    pan_left: -1,
    pan_right: 1,
    drift: 0.35,
    push_in: 0,
    pull_out: 0,
  };
  const DEPTH_KEYS = ['foreground_occluders', 'primary_subject_midground', 'background'];
  const DEPTH_DEBUG_COLORS = {
    foreground_occluders: 'rgba(255, 134, 81, 0.22)',
    primary_subject_midground: 'rgba(255, 202, 92, 0.18)',
    background: 'rgba(96, 164, 255, 0.12)',
  };
  const MOTION_FIELDS = [
    'duration_seconds',
    'zoom_end',
    'pan_x',
    'pan_y',
    'parallax_strength',
    'motion_intensity',
    'camera_motion',
    'easing',
  ];
  const EASINGS = {
    push_in: 'cubic-bezier(.22,.61,.36,1)',
    pull_out: 'cubic-bezier(.33,1,.68,1)',
    pan_left: 'cubic-bezier(.25,.46,.45,.94)',
    pan_right: 'cubic-bezier(.25,.46,.45,.94)',
    drift: 'cubic-bezier(.37,0,.63,1)',
  };
  const HEURISTIC_DEPTH_LAYOUT = {
    foreground_occluders: {
      polygon: [
        { x: 0, y: 700 },
        { x: 1000, y: 700 },
        { x: 1000, y: 1000 },
        { x: 0, y: 1000 },
      ],
      rationale: 'Lower frame and edge detail is treated as the nearest occluding plane.',
    },
    primary_subject_midground: {
      polygon: [
        { x: 220, y: 220 },
        { x: 780, y: 220 },
        { x: 820, y: 700 },
        { x: 180, y: 700 },
      ],
      rationale: 'The central image band is reserved for the primary subject or midground action.',
    },
    background: {
      polygon: [
        { x: 0, y: 0 },
        { x: 1000, y: 0 },
        { x: 1000, y: 1000 },
        { x: 0, y: 1000 },
      ],
      rationale: 'The full frame is the backing region, with nearer polygons composited above it.',
    },
  };

  function number(value, fallback) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function clamp(value, minimum, maximum) {
    return Math.min(maximum, Math.max(minimum, value));
  }

  function round(value, places = 2) {
    const factor = 10 ** places;
    return Math.round(value * factor) / factor;
  }

  function cloneShotParams(params = {}) {
    return JSON.parse(JSON.stringify(params));
  }

  function deepFreeze(value) {
    if (!value || typeof value !== 'object' || Object.isFrozen(value)) return value;
    Object.values(value).forEach(deepFreeze);
    return Object.freeze(value);
  }

  function easingFor(params = {}) {
    return params.easing || EASINGS[params.camera_motion] || EASINGS.drift;
  }

  function prepareShotParams(params = {}) {
    const prepared = cloneShotParams(params);
    prepared.easing = easingFor(prepared);
    return prepared;
  }

  function freezeShotParams(params = {}) {
    return deepFreeze(prepareShotParams(params));
  }

  function valuesEqual(first, second) {
    if (typeof first === 'number' && typeof second === 'number') {
      return Math.abs(first - second) < 0.005;
    }
    return first === second;
  }

  function compareShotParams(first = {}, second = {}) {
    return MOTION_FIELDS
      .filter(key => !valuesEqual(
        key === 'easing' ? easingFor(first) : first[key],
        key === 'easing' ? easingFor(second) : second[key],
      ))
      .map(key => ({
        key,
        first: key === 'easing' ? easingFor(first) : first[key],
        revised: key === 'easing' ? easingFor(second) : second[key],
      }));
  }

  function nudge(value, amount, minimum, maximum) {
    const raised = round(clamp(value + amount, minimum, maximum));
    return raised !== value ? raised : round(clamp(value - amount, minimum, maximum));
  }

  function ensureDistinctShotParams(first = {}, candidate = {}) {
    const base = prepareShotParams(first);
    const revised = prepareShotParams(candidate);
    const differences = compareShotParams(base, revised);
    if (differences.length >= 2) return revised;

    const changed = new Set(differences.map(difference => difference.key));
    if (!changed.has('duration_seconds')) {
      revised.duration_seconds = nudge(
        number(base.duration_seconds, 8),
        number(base.duration_seconds, 8) <= 10 ? 1 : -1,
        3,
        20,
      );
      changed.add('duration_seconds');
    }
    if (changed.size < 2 && !changed.has('parallax_strength')) {
      revised.parallax_strength = nudge(
        number(base.parallax_strength, 0.46),
        number(base.parallax_strength, 0.46) <= 0.5 ? 0.18 : -0.18,
        0,
        1,
      );
      changed.add('parallax_strength');
    }
    if (changed.size < 2 && !changed.has('pan_x')) {
      revised.pan_x = nudge(number(base.pan_x, 0), 6, -20, 20);
      changed.add('pan_x');
    }
    if (changed.size < 2 && !changed.has('zoom_end')) {
      revised.zoom_end = nudge(number(base.zoom_end, 1.16), 0.04, 1, 1.35);
    }
    revised.easing = easingFor(revised);
    return revised;
  }

  function normalizeDepthLayout(layout) {
    if (!layout || typeof layout !== 'object') return null;
    const normalized = {};
    for (const key of DEPTH_KEYS) {
      const region = layout[key];
      if (!region || !Array.isArray(region.polygon) || region.polygon.length < 3) return null;
      const polygon = region.polygon.map(point => {
        const x = Number(point?.x);
        const y = Number(point?.y);
        if (!Number.isFinite(x) || !Number.isFinite(y)) return null;
        return { x: clamp(x, 0, 1000), y: clamp(y, 0, 1000) };
      });
      if (polygon.some(point => point === null)) return null;
      normalized[key] = { polygon, rationale: String(region.rationale || '') };
    }
    return normalized;
  }

  function heuristicDepthLayout() {
    return JSON.parse(JSON.stringify(HEURISTIC_DEPTH_LAYOUT));
  }

  function drawPolygon(ctx, polygon, width, height) {
    ctx.beginPath();
    polygon.forEach((point, index) => {
      const x = point.x / 1000 * width;
      const y = point.y / 1000 * height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fill();
  }

  function strokePolygon(ctx, polygon, width, height) {
    ctx.beginPath();
    polygon.forEach((point, index) => {
      const x = point.x / 1000 * width;
      const y = point.y / 1000 * height;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.stroke();
  }

  function applyDepthMasks(scene, layout) {
    const normalized = normalizeDepthLayout(layout);
    if (!normalized || typeof document === 'undefined') return false;
    const bounds = scene.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    const canvases = {};
    const contexts = {};
    const planeElements = {
      foreground_occluders: scene.querySelector('.image-front'),
      primary_subject_midground: scene.querySelector('.image-mid'),
      background: scene.querySelector('.image-back'),
    };
    if (Object.values(planeElements).some(plane => !plane)) return false;

    DEPTH_KEYS.forEach(key => {
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      canvases[key] = canvas;
      contexts[key] = canvas.getContext('2d');
      contexts[key].fillStyle = '#fff';
    });

    drawPolygon(contexts.background, normalized.background.polygon, width, height);
    drawPolygon(contexts.primary_subject_midground, normalized.primary_subject_midground.polygon, width, height);
    contexts.background.globalCompositeOperation = 'destination-out';
    drawPolygon(
      contexts.background,
      normalized.primary_subject_midground.polygon,
      width,
      height,
    );
    drawPolygon(contexts.foreground_occluders, normalized.foreground_occluders.polygon, width, height);
    contexts.background.globalCompositeOperation = 'destination-out';
    drawPolygon(contexts.background, normalized.foreground_occluders.polygon, width, height);
    contexts.primary_subject_midground.globalCompositeOperation = 'destination-out';
    drawPolygon(
      contexts.primary_subject_midground,
      normalized.foreground_occluders.polygon,
      width,
      height,
    );

    DEPTH_KEYS.forEach(key => {
      const maskUrl = canvases[key].toDataURL('image/png');
      const plane = planeElements[key];
      plane.style.maskImage = `url("${maskUrl}")`;
      plane.style.webkitMaskImage = `url("${maskUrl}")`;
      plane.style.webkitMaskSize = '100% 100%';
      plane.style.maskSize = '100% 100%';
      plane.style.maskPosition = 'center';
      plane.style.webkitMaskPosition = 'center';
      plane.style.maskRepeat = 'no-repeat';
      plane.style.webkitMaskRepeat = 'no-repeat';
      plane.style.maskMode = 'alpha';
      plane.style.webkitMaskMode = 'alpha';
    });
    scene.dataset.depthSource = 'image-aware';
    return true;
  }

  function applyDepthDebugOverlay(scene, layout) {
    const normalized = normalizeDepthLayout(layout);
    const canvas = scene && scene.querySelector('.depth-debug-overlay');
    if (!normalized || !canvas || typeof canvas.getContext !== 'function') return false;
    const bounds = scene.getBoundingClientRect();
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);
    DEPTH_KEYS.forEach(key => {
      const polygon = normalized[key].polygon;
      ctx.fillStyle = DEPTH_DEBUG_COLORS[key];
      drawPolygon(ctx, polygon, width, height);
      ctx.strokeStyle = DEPTH_DEBUG_COLORS[key].replace(/[\d.]+\)$/, '0.9)');
      ctx.lineWidth = 2;
      strokePolygon(ctx, polygon, width, height);
    });
    return true;
  }

  function calculateCompositorParams(params = {}) {
    const zoomStart = clamp(number(params.zoom_start, 1), 1, 1.3);
    const zoomEnd = clamp(number(params.zoom_end, 1.16), 1, 1.35);
    const panX = clamp(number(params.pan_x, 0), -20, 20);
    const panY = clamp(number(params.pan_y, 0), -15, 15);
    const parallax = clamp(number(params.parallax_strength, 0.46), 0, 1);
    const intensity = clamp(number(params.motion_intensity, 0.38), 0, 1);
    const direction = CAMERA_DIRECTIONS[params.camera_motion] || 0;
    const lateralSlide = panX + direction * intensity * 4;
    const verticalSlide = panY;
    const motionTravel = 0.72 + intensity * 0.28;
    const baseX = lateralSlide * motionTravel * 2.4;
    const baseY = verticalSlide * motionTravel * 2.4;
    const pathMagnitude = clamp(
      (Math.abs(baseX) / 20 + Math.abs(baseY) / 15 + Math.abs(zoomEnd - zoomStart) / 0.35) / 3,
      0,
      1,
    );
    const overscanPercent = 16;
    const overscanScale = 1 + (overscanPercent * 2) / 100;

    const planes = DEPTH_PLANES.map(plane => {
      const depthTravel = 0.28 + parallax * (0.22 + plane.travel * 0.78);
      const startScale = overscanScale * zoomStart * (1 + parallax * (0.012 + plane.travel * 0.025));
      const endScale = overscanScale * zoomEnd * (1 + parallax * (0.016 + plane.travel * 0.034));
      const strength = Math.round(clamp(
        (0.35 + intensity * 0.65)
          * (0.35 + parallax * 0.65)
          * plane.travel
          * (0.55 + pathMagnitude * 0.45),
        0,
        1,
      ) * 100);

      return {
        key: plane.key,
        label: plane.label,
        strength,
        easing: easingFor(params),
        start: { x: 0, y: 0, scale: Number(startScale.toFixed(4)) },
        end: {
          x: Number((baseX * depthTravel).toFixed(4)),
          y: Number((baseY * depthTravel).toFixed(4)),
          scale: Number(endScale.toFixed(4)),
        },
      };
    });

    return {
      duration_seconds: clamp(number(params.duration_seconds, 8), 3, 20),
      easing: easingFor(params),
      overscan_percent: overscanPercent,
      overscan_scale: overscanScale,
      path: {
        lateral_slide: Math.abs(baseX) > 0.001,
        vertical_slide: Math.abs(baseY) > 0.001,
        push_in: zoomEnd > zoomStart,
        pull_out: zoomEnd < zoomStart,
      },
      planes,
    };
  }

  function applyToStage(stage, params) {
    const composition = calculateCompositorParams(params);
    stage.style.setProperty('--compositor-overscan', `${composition.overscan_percent}%`);
    stage.style.setProperty('--duration', `${composition.duration_seconds}s`);
    stage.style.setProperty('--compositor-easing', composition.easing);

    composition.planes.forEach(plane => {
      stage.style.setProperty(`--${plane.key}-x-start`, `${plane.start.x}px`);
      stage.style.setProperty(`--${plane.key}-y-start`, `${plane.start.y}px`);
      stage.style.setProperty(`--${plane.key}-scale-start`, plane.start.scale);
      stage.style.setProperty(`--${plane.key}-x-end`, `${plane.end.x}px`);
      stage.style.setProperty(`--${plane.key}-y-end`, `${plane.end.y}px`);
      stage.style.setProperty(`--${plane.key}-scale-end`, plane.end.scale);
      stage.style.setProperty(`--${plane.key}-easing`, plane.easing);
    });

    return composition;
  }

  return {
    calculateCompositorParams,
    applyDepthDebugOverlay,
    applyDepthMasks,
    applyToStage,
    compareShotParams,
    ensureDistinctShotParams,
    freezeShotParams,
    heuristicDepthLayout,
    normalizeDepthLayout,
    prepareShotParams,
  };
}));