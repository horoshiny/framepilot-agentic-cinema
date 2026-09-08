---
name: Veo model configuration
description: Durable rules for selecting and exposing the production Veo model
---

The production Veo model must have a current default, remain overrideable through deployment configuration, and reject known retired model IDs before approval or allowance reservation. Approval responses, job metadata, provider submission, diagnostics, and UI copy should all use the same effective model. Cost copy must show the configured numeric deployment estimate when present and remain absent when it is not configured; never invent or preserve a stale model-specific price. Veo 3/3.1 generation configs must omit `enhance_prompt` and use the provider default.

**Why:** Provider model retirement caused a real 404, hardcoded model/price copy can make approval UI claim a configuration that the server will not use, and Veo 3 rejects an explicit prompt-enhancement disable.

**How to apply:** Keep the model selection in one provider helper, pass it through approval/job metadata, validate it before ledger admission, return no estimate label when `VEO_ESTIMATED_COST_INR` is absent or invalid, omit `enhance_prompt` for Veo 3/3.1, and reject any future explicit `False` before submission.