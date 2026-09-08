import json
import subprocess
from pathlib import Path

from cinema_agent.demo import demo_critique, demo_plan


COMPOSITOR = Path(__file__).parents[1] / "static" / "compositor.js"


def calculate_compositor(params):
    script = (
        "const compositor = require(process.argv[1]);"
        "console.log(JSON.stringify(compositor.calculateCompositorParams(JSON.parse(process.argv[2]))));"
    )
    result = subprocess.run(
        ["node", "-e", script, str(COMPOSITOR), json.dumps(params)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_uploaded_image_planes_have_independent_depth_transforms():
    composition = calculate_compositor(
        {
            "camera_motion": "pan_right",
            "zoom_start": 1.0,
            "zoom_end": 1.2,
            "pan_x": 12,
            "pan_y": -4,
            "parallax_strength": 0.7,
            "motion_intensity": 0.8,
        }
    )
    planes = {plane["key"]: plane for plane in composition["planes"]}

    assert set(planes) == {"front", "mid", "back"}
    assert abs(planes["front"]["end"]["x"]) > abs(planes["mid"]["end"]["x"])
    assert abs(planes["mid"]["end"]["x"]) > abs(planes["back"]["end"]["x"])
    assert planes["front"]["end"]["scale"] > planes["mid"]["end"]["scale"]
    assert planes["mid"]["end"]["scale"] > planes["back"]["end"]["scale"]


def test_stationary_image_composition_preserves_full_frame_overscan():
    composition = calculate_compositor(
        {
            "camera_motion": "drift",
            "zoom_start": 1.0,
            "zoom_end": 1.0,
            "pan_x": 0,
            "pan_y": 0,
            "parallax_strength": 0,
            "motion_intensity": 0,
        }
    )

    assert composition["overscan_percent"] == 16
    assert composition["overscan_scale"] == 1.32
    assert all(plane["start"]["scale"] == 1.32 for plane in composition["planes"])
    assert all(plane["end"]["scale"] == 1.32 for plane in composition["planes"])
    assert composition["path"] == {
        "lateral_slide": False,
        "vertical_slide": False,
        "push_in": False,
        "pull_out": False,
    }


def test_compositor_preserves_compound_slide_and_push_in_motion():
    composition = calculate_compositor(
        {
            "camera_motion": "pan_left",
            "zoom_start": 1.0,
            "zoom_end": 1.24,
            "pan_x": -14,
            "pan_y": 2,
            "parallax_strength": 0.6,
            "motion_intensity": 0.65,
        }
    )

    assert composition["path"]["lateral_slide"] is True
    assert composition["path"]["push_in"] is True
    assert all(plane["end"]["x"] < 0 for plane in composition["planes"])
    assert all(plane["end"]["scale"] > plane["start"]["scale"] for plane in composition["planes"])


def test_pull_back_and_pan_directions_remain_distinct_in_browser_renderer():
    pull_back = calculate_compositor(
        {
            "camera_motion": "pull_out",
            "zoom_start": 1.18,
            "zoom_end": 1.02,
            "pan_x": -8,
            "parallax_strength": 0.42,
            "motion_intensity": 0.3,
        }
    )
    pan_left = calculate_compositor(
        {
            "camera_motion": "pan_left",
            "zoom_start": 1.0,
            "zoom_end": 1.0,
            "pan_x": -14,
            "parallax_strength": 0.5,
            "motion_intensity": 0.5,
        }
    )
    pan_right = calculate_compositor(
        {
            "camera_motion": "pan_right",
            "zoom_start": 1.0,
            "zoom_end": 1.0,
            "pan_x": 14,
            "parallax_strength": 0.5,
            "motion_intensity": 0.5,
        }
    )

    assert pull_back["path"]["pull_out"] is True
    assert pull_back["path"]["lateral_slide"] is True
    assert pull_back["planes"][0]["end"]["scale"] < pull_back["planes"][0]["start"]["scale"]
    assert all(plane["end"]["x"] < 0 for plane in pan_left["planes"])
    assert all(plane["end"]["x"] > 0 for plane in pan_right["planes"])
    assert pan_left["planes"] != pan_right["planes"]


def test_compositor_uses_sufficient_overscan_for_uploaded_images():
    composition = calculate_compositor({"pan_x": 20, "pan_y": 15, "zoom_end": 1.35})
    css = (COMPOSITOR.parent / "cinematic.css").read_text()

    assert composition["overscan_percent"] >= 12
    assert ".image-plane{inset:0;" in css
    assert ".depth-debug-overlay" in css
    assert "background-size:cover" in css
    assert "filter:none" in css
    assert "opacity:1" in css
    assert "border:0" in css
    assert "transform:scale(1)" in css
    assert ".image-backing{inset:0;z-index:0" in css


def test_strong_parallax_keeps_foreground_readable_and_depth_ordered():
    composition = calculate_compositor(
        {
            "camera_motion": "pan_right",
            "zoom_start": 1,
            "zoom_end": 1.08,
            "pan_x": 20,
            "pan_y": 0,
            "parallax_strength": 1,
            "motion_intensity": 1,
        }
    )
    planes = {plane["key"]: plane for plane in composition["planes"]}

    assert 40 <= planes["front"]["end"]["x"] <= 80
    assert planes["front"]["end"]["x"] > planes["mid"]["end"]["x"] > planes["back"]["end"]["x"]
    assert planes["front"]["end"]["x"] - planes["back"]["end"]["x"] >= 20


def test_depth_debug_overlay_is_hidden_until_explicitly_enabled():
    css = (COMPOSITOR.parent / "cinematic.css").read_text()
    html = (COMPOSITOR.parent / "index.html").read_text()
    app_js = (COMPOSITOR.parent / "app.js").read_text()

    assert ".depth-debug-overlay{z-index:7;display:none" in css
    assert ".show-depth-debug .depth-debug-overlay{display:block}" in css
    assert 'id="toggleDepthDebug"' in html
    assert 'aria-pressed="false"' in html
    assert "stage.classList.toggle('show-depth-debug', enabled)" in app_js


def test_critic_revision_changes_composited_motion():
    plan = demo_plan("The camera pushes in through fog as rain falls.", "suspense")
    revision = demo_critique(plan).revision

    first_pass = calculate_compositor(plan.shot.model_dump())
    revised_pass = calculate_compositor(revision.model_dump())

    assert first_pass["duration_seconds"] != revised_pass["duration_seconds"]
    assert first_pass["planes"] != revised_pass["planes"]


def test_cut_helpers_freeze_plans_and_make_identical_revisions_observable():
    script = r"""
const compositor = require(process.argv[1]);
const first = {
  duration_seconds: 8,
  camera_motion: 'push_in',
  zoom_start: 1,
  zoom_end: 1.16,
  pan_x: -4,
  pan_y: 1,
  parallax_strength: .46,
  motion_intensity: .38,
  atmosphere: ['fog'],
  transition: 'fade',
};
const frozen = compositor.freezeShotParams(first);
const revised = compositor.ensureDistinctShotParams(frozen, first);
console.log(JSON.stringify({
  frozen: Object.isFrozen(frozen) && Object.isFrozen(frozen.atmosphere),
  differences: compositor.compareShotParams(frozen, revised),
  first: compositor.calculateCompositorParams(frozen),
  revised: compositor.calculateCompositorParams(revised),
}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(COMPOSITOR)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["frozen"] is True
    assert len(payload["differences"]) >= 2
    assert payload["first"]["planes"] != payload["revised"]["planes"]
    assert payload["first"]["duration_seconds"] != payload["revised"]["duration_seconds"]


def test_identical_cut_plans_report_no_false_changes():
    script = (
        "const compositor = require(process.argv[1]);"
        "const plan = {duration_seconds: 8, camera_motion: 'push_in', zoom_start: 1, "
        "zoom_end: 1.16, pan_x: -4, pan_y: 1, parallax_strength: .46, "
        "motion_intensity: .38, atmosphere: ['fog'], transition: 'fade'};"
        "console.log(JSON.stringify(compositor.compareShotParams(plan, plan)));"
    )
    result = subprocess.run(
        ["node", "-e", script, str(COMPOSITOR)],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_polygon_depth_layout_produces_exclusive_canvas_masks():
    script = r"""
const compositor = require(process.argv[1]);
const operations = [];
const planes = new Map([
  ['.image-front', { style: {} }],
  ['.image-mid', { style: {} }],
  ['.image-back', { style: {} }],
]);
const canvases = [];
function context() {
  return {
    globalCompositeOperation: 'source-over',
    fillStyle: '',
    beginPath() {},
    moveTo() {},
    lineTo() {},
    closePath() {},
    fill() { operations.push(this.globalCompositeOperation); },
  };
}
global.document = {
  createElement(tag) {
    if (tag !== 'canvas') throw new Error('Unexpected element');
    const ctx = context();
    const canvas = {
      width: 0,
      height: 0,
      getContext() { return ctx; },
      toDataURL() { canvases.push({ width: this.width, height: this.height }); return 'data:image/png;base64,mask'; },
    };
    return canvas;
  },
};
const scene = {
  dataset: {},
  getBoundingClientRect() { return { width: 960, height: 540 }; },
  querySelector(selector) { return planes.get(selector); },
};
const result = compositor.applyDepthMasks(scene, compositor.heuristicDepthLayout());
console.log(JSON.stringify({
  result,
  canvases,
  operations,
  masks: [...planes.values()].map(plane => plane.style.maskImage),
  source: scene.dataset.depthSource,
}));
"""
    result = subprocess.run(
        ["node", "-e", script, str(COMPOSITOR)],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["result"] is True
    assert len(payload["canvases"]) == 3
    assert all(canvas == {"width": 960, "height": 540} for canvas in payload["canvases"])
    assert "destination-out" in payload["operations"]
    assert all(mask.startswith('url("data:image/png') for mask in payload["masks"])
    assert payload["source"] == "image-aware"