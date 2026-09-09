# FramePilot

FramePilot is an agentic previsualisation studio for the Agentic Cinema hackathon. It converts screenplay excerpts and storyboard art into editable 2.5D animatics, then critiques and revises its own directing choices.

## Current checkpoint

This repository contains a production-configured FramePilot demo with Vertex AI Gemini multimodal direction, an optional Vertex AI Veo adapter, and a deterministic fallback. Paid generation remains disabled by default and is guarded by server-side allowances, scene/session binding, and explicit approval.

## Run locally

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the app with `python3 run.py`.
4. Open `http://127.0.0.1:5000`.

The launcher binds to `0.0.0.0` and uses Replit's assigned `PORT` when published, defaulting to `5000` locally. The app is locked to deterministic demo mode by default, even when a Google Cloud project is configured. For intentional local live Gemini direction, copy `.env.example` to `.env`, provide the Google Cloud project ID, set both `DEMO_MODE=false` and `ALLOW_VERTEX_INFERENCE=true`, and authenticate locally with Application Default Credentials. Keep `ALLOW_VERTEX_INFERENCE=false` when publishing the demo unless the deployment has been explicitly configured for Vertex access.

## Environment configuration

`.env.example` documents the supported non-secret configuration without credential values. The published-safe defaults are:

- `DEMO_MODE=true`
- `ALLOW_VERTEX_INFERENCE=false`
- `VIDEO_GENERATION_PROVIDER=mock`
- `ALLOW_VEO_GENERATION=false`
- bounded global and per-IP generation allowances
- `ALLOW_VIDEO_CRITIQUE=false`

Real Vertex AI use additionally requires the Google Cloud project, location, service-account configuration, approved Veo output bucket, and the corresponding Replit Secrets. Never commit those values or paste them into source control.

## Production deployment

Use the Replit Publish tool with the `python3 run.py` command. The app is configured for a long-running web deployment because its SQLite ledger and session-bound completed-job recovery must remain durable. Direct Scene and local Motion Preview remain available to public visitors; real Veo generation is not unlimited and is disabled unless the server-side controls and explicit approval path are satisfied.

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
