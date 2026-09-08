---
name: Vertex structured-output boundary
description: Vertex accepts JSON MIME mode for Direct but rejects the sanitized structured-output schema
---

For FramePilot's Direct request, Vertex accepted the same multimodal content with the production system instruction and with `response_mime_type="application/json"` alone. The same request failed with `400 INVALID_ARGUMENT` when the sanitized `response_json_schema` was present. Keep Direct on JSON MIME mode with local strict validation rather than provider structured-output configuration.

**Why:** Two isolated live diagnostics showed that text, image parts, system instructions, and JSON MIME mode work independently; only the structured-output configuration failed. This is a provider boundary, not evidence that arbitrary storyboard understanding should be replaced with scene-specific fallback rules.

**How to apply:** Generate a bounded compact model-derived `SceneAnalysis` JSON contract in the system instruction, use JSON MIME mode with the 8192-token Direct budget, parse the complete object before local validation, shorten only overlong summaries at boundaries, validate entities, relationships, camera, and optional descriptive lists independently, retain sanitized path/type conflicts, reject surrounding prose/unknown fields, strip only one complete JSON markdown fence, then validate the retained payload strictly. Preserve the single user Content and `Part.from_bytes` image path. Optional list failures must not discard valid semantic entities; require at least one recoverable entity before accepting an otherwise empty analysis.