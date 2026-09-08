---
name: First Cut pre-generation controls
description: Where the First Cut approval and allowance controls belong in the simplified output architecture
---

First Cut approval and allowance controls must remain visible from the Motion Preview state and move out of the completed First Cut output pane once a video exists.

**Why:** The completed-output pane is intentionally hidden until durable First Cut recovery or generation succeeds; placing the only generation action there makes new First Cuts impossible to request.

**How to apply:** Keep pre-generation status, allowance copy, recovery messaging, and the approval action in the Motion Preview workspace. Use the First Cut pane only for completed playback, review, and post-generation state.

Session-scoped durable recovery must be verified through the browser’s existing session identity and `/api/video-jobs/latest`; never bypass ownership by hardcoding a job into frontend behavior.