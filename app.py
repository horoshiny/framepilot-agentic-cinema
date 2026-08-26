from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from cinema_agent.demo import demo_critique, demo_plan
from cinema_agent.schemas import DirectRequest, DirectResponse
from cinema_agent.tools import runtime_mode
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
STATIC = ROOT / "static"

app = FastAPI(title="FramePilot", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "mode": runtime_mode()}


@app.post("/api/direct", response_model=DirectResponse)
def direct_scene(request: DirectRequest):
    mode = runtime_mode()
    fallback = None
    if mode == "vertex":
        try:
            from cinema_agent.vertex import generate_critique, generate_shot_plan

            plan = generate_shot_plan(request.screenplay, request.mood, request.image_data_url)
            critique = generate_critique(request.screenplay, plan)
        except Exception as exc:
            # The demo remains available if credits, billing, IAM, quota, or model access
            # are not ready. The UI makes this fallback visible instead of hiding it.
            mode = "demo"
            fallback = f"Vertex call unavailable: {type(exc).__name__}"
            plan = demo_plan(request.screenplay, request.mood)
            critique = demo_critique(plan)
    else:
        plan = demo_plan(request.screenplay, request.mood)
        critique = demo_critique(plan)

    activity = [
        {"step": "ANALYSE", "detail": "Mapped subject, depth and emotional intent"},
        {"step": "DIRECT", "detail": "Selected camera, focus and atmosphere"},
        {"step": "ANIMATE", "detail": "Compiled deterministic browser motion"},
        {"step": "CRITIQUE", "detail": critique.diagnosis},
        {"step": "REVISE", "detail": critique.revision_rationale},
    ]
    if fallback:
        activity.insert(0, {"step": "FALLBACK", "detail": fallback})
    return DirectResponse(
        mode=mode,
        plan=plan,
        critique=critique,
        activity=activity,
    )
