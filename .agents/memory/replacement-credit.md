---
name: Replacement First Cut credit
description: Durable rules for an explicitly authorized replacement attempt tied to one failed accepted First Cut.
---

An authorized replacement is a single job-scoped credit with its own audit record and reservation state. It must not modify, release, or hide the original accepted submission, and replacement submissions must retain a direct relationship to that original job.

**Why:** A failed accepted provider operation may require one explicitly approved recovery attempt without silently increasing the normal generation budget or allowing double-click/concurrent over-consumption.

**How to apply:** Reserve the credit atomically with the replacement submission, consume it only when the provider returns an accepted operation, restore it for pre-acceptance failures, keep it reserved for uncertain submissions, and exclude replacement rows from normal global/per-IP counters.

The client may expose a replacement retry when the session-scoped allowance response names an available failed First Cut authorization, reports a positive credit, and the current scene has a storyboard handle. A fresh Direct must not erase that historical authorization just because its current First Cut card is `not_generated`. The server remains authoritative for every approval and submission.

**Why:** A replacement authorization is job-scoped but survives a fresh Direct scene reset; deriving it only from the current card loses valid recovery credits, while ignoring session and state boundaries could broaden access.

**How to apply:** Treat `authorized_replacement_for_job_id` from the session allowance as the source when the current card is `not_generated`; require a positive available count, no active job, and storyboard handle, then include that original job ID in both approval and creation payloads.

Accepted Veo operations must remain resumable after a local polling timeout; reopening a timed-out accepted job must clear only local failure fields and must never create a submission or alter allowance accounting.

**Why:** Veo can finish after the application’s original short deadline. Treating that deadline as terminal loses accepted provider work and makes a valid completed operation unrecoverable.

**How to apply:** Use a configurable long terminal deadline, restore legacy locally-timed-out jobs with an operation ID to `generating`, poll the same operation, and preserve the accepted submission and replacement ledger rows.