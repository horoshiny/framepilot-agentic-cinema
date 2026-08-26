import base64
import json
import os

from google import genai
from google.genai import types

from .schemas import Critique, ShotPlan


DIRECTOR_PROMPT = """You are FramePilot's previsualisation director. Analyse the supplied
screenplay excerpt and optional storyboard image, then return one restrained, controllable
2.5D shot plan. Direct the emotional beat rather than decorating the frame. Camera motion,
depth, atmosphere and transition must be justified by narrative focus. Never claim full
character rigging or invent action that is absent from the screenplay. Output only the
requested schema."""

CRITIC_PROMPT = """You are an exacting film previsualisation critic. Evaluate the supplied
screenplay and shot plan for narrative focus, pacing, cinematic motion versus slide-like
motion, and restraint. Produce a materially improved parameter set, not generic advice.
Keep every numeric value within the requested schema and output only that schema."""


def _client():
    project = os.environ["GOOGLE_CLOUD_PROJECT"]
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    try:
        return genai.Client(enterprise=True, project=project, location=location)
    except TypeError:
        # Compatibility with older google-genai builds that still use vertexai=True.
        return genai.Client(vertexai=True, project=project, location=location)


def _image_part(image_data_url: str | None):
    if not image_data_url or "," not in image_data_url:
        return None
    header, encoded = image_data_url.split(",", 1)
    mime = header.split(";")[0].replace("data:", "")
    if mime not in {"image/png", "image/jpeg", "image/webp"}:
        return None
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("Storyboard image must be smaller than 8 MB")
    return types.Part.from_bytes(data=raw, mime_type=mime)


def generate_shot_plan(screenplay: str, mood: str, image_data_url: str | None) -> ShotPlan:
    contents = [f"CREATIVE INTENT: {mood}\n\nSCREENPLAY:\n{screenplay}"]
    image = _image_part(image_data_url)
    if image:
        contents.append(image)
    with _client() as client:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=DIRECTOR_PROMPT,
                response_mime_type="application/json",
                response_schema=ShotPlan,
                temperature=0.55,
            ),
        )
    return ShotPlan.model_validate_json(response.text)


def generate_critique(screenplay: str, plan: ShotPlan) -> Critique:
    payload = json.dumps({"screenplay": screenplay, "shot_plan": plan.model_dump()})
    with _client() as client:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=payload,
            config=types.GenerateContentConfig(
                system_instruction=CRITIC_PROMPT,
                response_mime_type="application/json",
                response_schema=Critique,
                temperature=0.35,
            ),
        )
    return Critique.model_validate_json(response.text)

