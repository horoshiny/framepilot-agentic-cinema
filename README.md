# FramePilot

FramePilot is an agentic previsualisation studio for the Agentic Cinema hackathon. It converts screenplay excerpts and storyboard art into editable 2.5D animatics, then critiques and revises its own directing choices.

## Current checkpoint

This repository contains a demonstrable local MVP with both Vertex AI Gemini calls and a deterministic fallback. The renderer contract and ADK tool schemas remain stable if billing, quota, IAM or model access is still pending.

## Run locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the app with `uvicorn app:app --reload`.
4. Open `http://127.0.0.1:8000`.

The app is locked to deterministic demo mode by default, even when a Google Cloud project is configured. For intentional local live Gemini direction, copy `.env.example` to `.env`, provide the Google Cloud project ID, set both `DEMO_MODE=false` and `ALLOW_VERTEX_INFERENCE=true`, and authenticate locally with Application Default Credentials. Keep `ALLOW_VERTEX_INFERENCE=false` when publishing the demo.

## Agent workflow

1. Analyse screenplay and storyboard.
2. Produce a validated shot plan.
3. Compile the plan into deterministic browser animation parameters.
4. Critique focus, pacing, cinematic motion and restraint.
5. Apply a targeted revision and compare passes.

## Cloud integration plan

- Gemini multimodal analysis through Vertex AI
- Google ADK tool orchestration and session state
- Vertex AI Agent Engine deployment
- Replit Agent-built web application and `replit.app` deployment

Copy `.env.example` to `.env` only for local development. Never commit credentials.
