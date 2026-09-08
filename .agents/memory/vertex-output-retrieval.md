---
name: Vertex output retrieval
description: The safe retrieval boundary for real Vertex Veo video outputs.
---

Vertex Veo output retrieval must prefer inline video bytes and otherwise use authenticated Cloud Storage only after validating an approved bucket and object prefix. The Gemini file-download endpoint is not a Vertex retrieval path.

**Why:** The installed Google Gen AI SDK rejects `Files.download` for Vertex clients, while Vertex samples expose generated video output as a `gs://` URI. Treating the URI as an arbitrary URL could expose unrelated private objects.

**How to apply:** Preflight the approved bucket/prefix locally before approval consumption or allowance reservation, pass the validated destination to generation, reject non-`gs://` or out-of-scope references before any storage request, and preserve accepted job accounting when retrieval fails.