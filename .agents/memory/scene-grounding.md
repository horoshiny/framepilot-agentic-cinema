---
name: Scene-grounded fallback
description: The safety boundary between deterministic image metadata and semantic storyboard interpretation.
---

The deterministic fallback must not treat image dimensions, tone, contrast, or texture as recognition of a character, object, or effect. It may use those observations to choose restrained camera/depth parameters. When the screenplay explicitly names a bounded action, carry it as screenplay-grounded and confirmation-gated rather than visually confirmed; unsupported or invented actions stay excluded. The fallback parser must remain generic rather than encoding nouns from one fixture or scene.

**Why:** A fallback that fills in plausible cinematic details can silently contradict the uploaded storyboard and screenplay, but dropping explicit screenplay actions produces a Veo prompt that cannot honor the approved scene. Separating visual confirmation from screenplay grounding preserves both safety and narrative fidelity, while scene-specific rules make unrelated scenes regress and conceal parser weaknesses.

**How to apply:** When extending fallback direction, expose separate visual, screenplay, and intent evidence plus a grounding source and confidence. Resolve bounded pronoun/repeated references against the most recent grounded entity when possible. Mark screenplay-only candidates with zero visual confidence and needs_confirmation, keep unsupported requests out of approved candidates, surface image/screenplay conflicts for confirmation, and preserve explicit prohibited-change constraints. Provider-backed analysis and the final Veo prompt should carry the same contract.

The compact SceneAnalysis converter must preserve Gemini's supplied entity category: environmental entries remain environmental motion, physical props remain movable objects, and static entries remain preserved without motion candidates. Camera movement is a separate bounded capability: non-visual requested movement is screenplay/intent-grounded with zero visual confidence and needs_confirmation; only movement outside renderer constraints is unsupported and excluded.

**Why:** Reclassifying environmental effects or treating all camera motion as unsupported loses valid scene direction, while accepting unsupported camera vocabulary can leak impossible actions into the approved plan.

**How to apply:** Keep category routing data-driven from SceneAnalysis arrays. Map only the renderer's allowed camera vocabulary to shot parameters, retain invalid requests as rejected unsupported candidates, and never add noun-specific heuristics.

Malformed optional metadata must not erase a usable provider-supplied entity label. Recover the label, strip unknown fields, downgrade invalid visual grounding to `needs_confirmation`, and retain a bounded diagnostic; drop only entities with no usable label.

**Why:** A valid Gemini scene response can contain one malformed evidence or legacy field that causes strict whole-entity validation to discard an object, making the app report no recoverable entities even though the object was identified.

**How to apply:** Normalize compact entity fields before `SceneEntity` validation, preserve `objects`/`characters`/`environment` independently, and keep the recovered label out of visually confirmed motion until evidence is valid.

Deterministic conversion must normalize provider entity labels before `MotionCandidate` validation and isolate malformed candidates; a successful compact analysis must not be discarded because one label contains markup, controls, punctuation, an action sentence, or excessive length.

**Why:** Pydantic's strict motion-label validator can reject one provider-returned label after JSON and `SceneAnalysis` validation have already succeeded, causing an avoidable whole-result fallback.

**How to apply:** Unicode-normalize and sanitize labels, conservatively extract an explicit entity phrase from action sentences, cap length, omit only labels that remain invalid, and add a value-free grounding warning while preserving the other candidates.

Production Direct-scene analysis should use the existing structured multimodal provider for every eligible scene action, including semantic/generative requests. If that call is unavailable or rate-limited, identify the deterministic result as fallback and never present it as visual understanding.

**Why:** A semantic scene can be safely planned by the provider without silently bypassing the screenplay; hiding the fallback boundary makes unsupported visual claims look authoritative.

**How to apply:** Keep provider and fallback source labels in the Direct response, preserve the same candidate schema on both paths, and keep fallback visual confidence at zero.

The browser’s service health state must stay separate from a direction result’s analysis mode; restored or fallback direction data must never overwrite the live Vertex/mock provider label. Recovery records without `analysis_source` are legacy and must remain visibly marked and ineligible for Veo approval.

**Why:** Direction mode describes how one result was produced, while health/provider state describes what the current server can do. Conflating them made a valid Vertex health response appear as Demo mode after restoring older fallback data.

**How to apply:** Resolve provider copy and approval availability from `/api/health` plus the current result source, not from `result.mode` alone. Treat missing source metadata as legacy/fallback until a new Direct response replaces it.

The compact semantic contract treats a character as any narratively agentive entity, not a fixed biological or production type. Gemini owns character/object/environment classification and may return one `main_character_id`; Python must select only that declared visible ID, preserve uncertain roles for confirmation, and never infer a focal subject from size, position, or appearance.

**Why:** Vehicles, animals, robots, animated objects, natural phenomena, and stylized figures can all function as a scene subject, while ordinary props remain objects unless the scene gives them agency.

**How to apply:** Require stable entity IDs for main-character references, preserve causal relationships by ID, omit duplicate IDs across categories with a conflict, and leave unresolved or ambiguous main-character declarations without a forced focal subject.

The compact provider envelope stays strict at the top level, while its semantic lists are recovered item-by-item: malformed entity evidence, camera metadata, descriptions, or relationships become bounded conflicts without discarding independently valid entities.

**Why:** Relaxing the whole envelope can hide provider contract drift, but strict whole-entity validation caused one malformed optional field to erase otherwise valid object recognition and trigger deterministic fallback.

**How to apply:** Keep unknown top-level fields invalid; sanitize known optional lists and entity fields independently, preserve valid labels/categories, and classify only the explicit no-recoverable-entities condition as local validation fallback.

The provider contract now includes explicit `entity_type` and `agentive` flags. The parser may correct a
conflicting array placement from those structured flags, recording a value-free conflict, before validating
relationships; it must not use scene-specific noun lists to make that correction.

**Why:** Generic storyboards can contain agentive vehicles, animals, robots, ensembles, or animated objects,
and physical props can otherwise be mislabeled as environmental motion. Structured correction preserves the
visible inventory without encoding one scene's vocabulary.

**How to apply:** Let `agentive=true` resolve to characters and explicit object/environment types resolve to
their matching categories. Preserve all entity IDs, validate relationships after correction, and keep plural
main-character IDs for co-equal subjects.