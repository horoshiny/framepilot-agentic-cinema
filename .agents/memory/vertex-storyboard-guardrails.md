---
name: Vertex storyboard submission guardrails
description: Rules for preserving storyboard inputs and diagnosing Vertex video submission failures safely.
---

Vertex storyboard-to-video requests must carry a validated, client-bound storyboard image into the provider call. Missing, empty, unsupported, or unavailable image data must fail locally before allowance reservation or provider submission. Provider diagnostics may record bounded exception metadata and request shape, but never prompts, image bytes, data URLs, credentials, or tokens.

**Why:** A missing storyboard can otherwise silently become a text-to-video request, while generic provider wrapping can hide the information needed to distinguish invalid input, access, quota, and provider failures.

**How to apply:** Keep the image behind a server-side/session-bound handle, validate it at approval and again before ledger admission, construct the SDK image from validated bytes and MIME type, and release reservations for failures before an operation is accepted.