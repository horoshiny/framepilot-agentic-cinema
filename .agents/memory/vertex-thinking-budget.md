---
name: Vertex thinking budget
description: The reasoning-token setting used for compact Gemini semantic JSON extraction.
---

Compact SceneAnalysis extraction explicitly sets `ThinkingConfig(thinking_budget=0)` for
Gemini 2.5 Flash requests. The installed google-genai SDK documents zero as the disabled
setting; leaving `thinking_config` unset delegates the default to the model.

**Why:** A truncated JSON response can fail before any field-level recovery runs. Reserving
the response budget for the compact semantic contract is safer than relying on a
model-dependent thinking default.

**How to apply:** Keep thinking disabled for this extraction request unless a future provider
change demonstrates that a bounded nonzero budget improves entity completeness without
consuming the JSON output budget. Do not add provider schemas to this path.