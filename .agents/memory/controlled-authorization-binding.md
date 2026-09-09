---
name: Controlled authorization binding
description: Safety rules for tightly controlled, session-bound test allowances.
---

Controlled test allowances must never be bound from an authorization name, an old completed video job, or the preview’s default scene. A stale unused row must be revoked with its original binding and audit fields preserved. The replacement allowance must be activated only from the current browser session’s server-recorded Direct result after validating the exact scene key, source signature, analysis source, client-owned storyboard handle, First Cut kind, and required model.

**Why:** Durable video jobs can outlive the browser scene that created them. Reusing their identifiers can silently authorize the wrong storyboard, while an unverified browser claim can authorize another client or image.

**How to apply:** Keep the activation provider-free and idempotent, create at most one active bound row, require final approval before reservation, and leave other scenes, clients, images, analysis sources, and Director’s Cut ineligible.