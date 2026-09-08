---
name: Motion confirmation source
description: The canonical browser state used to confirm a conservative motion plan before First Cut approval.
---

The only canonical confirmation value is `data.plan.motion_plan.confirmation_required`. Motion-plan rendering, First Cut approval gating, and recovery persistence must derive their state from that nested plan rather than parallel or cached fields.

**Why:** A stale or duplicated confirmation value can leave the UI showing `CONFIRMATION NEEDED` and keep First Cut disabled even when replacement authorization and all other eligibility checks are valid.

**How to apply:** Use one accessor for the current nested motion plan and one boolean confirmation helper. The confirmation handler must set the nested field to the boolean `false`, rerender the motion plan and video controls, and save the existing recovery snapshot.