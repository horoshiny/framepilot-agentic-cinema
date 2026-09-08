---
name: Completed scene recovery
description: How to present durable completed First Cuts when their original analysis snapshot is partial or missing
---

Completed video output must never be presented beside inputs or Scene Plan values from another source signature. Prefer the persisted scene snapshot; for legacy jobs with only screenplay and intent, show those stored inputs and label the view as recovered while hiding unrelated plan values.

**Why:** A durable video can outlive the browser’s current bundled demo state, and mixing the two makes the recovered output look associated with the wrong scene.

**How to apply:** Return the session-owned scene linkage with the latest completed job, use it to restore the UI after stale browser recovery, and verify recovery with GET-only browser checks against the existing job.

The completed generated-video view should expose concise Generation Details rather than recovery terminology; keep recovery diagnostics internal or limited to failure disclosures.

**Why:** Judges need production-facing output context, not implementation history about how the scene was recovered.

**How to apply:** Keep the generated-video panel focused on status, provider, model, and duration while preserving the underlying scene-consistency safeguards.

The completed generated-video surface uses a compact header row for shot, state, workflow stages, and timer; do not reintroduce a large page title above the video.

**Why:** The video is the primary artifact, and the existing switch plus three-stage indicator already communicate the selected view without consuming vertical space.

**How to apply:** Keep the header directly above the output switch and preserve the generated video as the strongest visual element.