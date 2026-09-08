---
name: Vertex error diagnostics
description: Safe handling of Google GenAI APIError metadata in Vertex multimodal failures
---

When diagnosing Google GenAI failures, use the SDK error's structured `code`, `status`, and `message` fields with bounded redaction; do not log `str(error)`, `details`, request bodies, image data, or credentials.

**Why:** The SDK's `ClientError` class name alone only proves a client-side HTTP failure category, while its full string/details can contain provider response or request-sensitive data.

**How to apply:** Record HTTP status, Google status, a conservative category, and a short sanitized message at the provider exception boundary. Treat missing fields as unknown rather than inferring a specific API, model, IAM, quota, or request cause.