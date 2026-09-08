---
name: Mocked video boundary
description: The safety boundary for the storyboard-to-video Phase 1 workflow.
---

Phase 1 storyboard-to-video generation is intentionally a mocked, approval-gated workflow: it may create only process-local jobs backed by the bundled fixture, never a real video-provider request or Motion Preview output.

**Why:** The product needs a demonstrable First Cut/Director’s Cut lifecycle without paid generation, external side effects, persistent job storage, or accidentally presenting the local compositor as generated video.

**How to apply:** Keep video job kinds, lifecycle state, output references, and cache identity separate from analysis and compositor state. Any future provider integration requires an explicit scope change and a new safety review.