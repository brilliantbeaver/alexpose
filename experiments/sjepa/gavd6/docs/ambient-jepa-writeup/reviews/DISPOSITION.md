# Independent review and revision record

**28 September 2026.** An independent agent first evaluated the source study and proposed research direction without seeing the author's plan, then reviewed the draft plan against the evidence. This is independent critical review, not an independent replication of the experiments.

## Initial review

The [initial review](initial-adversarial-review.md) identified five major risks. All changed the delivered plan:

| Objection | Revision |
|---|---|
| An unsupported JEPA or world-model success narrative | Sections 1, 3 and 5 center measurement preservation, the zero-response baseline, and all three uncertain primary effects. |
| Clinical inferences from gait labels or asymmetry | Sections 2 and 8 separate measurement, gait classification, and prospective clinical prediction. |
| 2D and scalar ambiguities obscured by attractive diagrams | Sections 4.1-4.2 explain projection, temporal information loss, side ambiguity, and illustrative examples. Figures distinguish schematic from empirical content. |
| Unequal training influence and adaptive reuse | Sections 4.3-4.5 and 5 explain gradient imbalance, update/exposure differences, reused development people, and uncertainty scope. |
| Tool stacking mistaken for novelty or truth | Section 7 adds falsifiable baselines, independent reference measurements, and the 2026 OpenCap Monocular preprint as a direct prior-work constraint. |

## Draft review

The [draft review](draft-adversarial-review.md) found no remaining central overclaim or discrepancy in the checked headline numbers. It requested the following substantive changes:

| Finding | Disposition and location |
|---|---|
| P1.1: the next paired experiment is underspecified | **Accepted.** Section 8 now separates same-motion observation corruption from independently referenced within-person movement contrasts. It distinguishes locked-person 2D confirmation from a new 3D study, fixes a proposed per-leg preservation endpoint, and requires a predeclared testing hierarchy. |
| P1.2: preprocessing can leak future frames | **Accepted.** Section 7.1 restricts every input-producing operation to the observed prefix, requires a truncation check, and separates full-clip restoration from causal forecasting. |
| P1.3: stage-by-stage choices can consume a new test cohort | **Accepted.** Section 8 separates fitting, policy/reward/threshold selection, and final evaluation; later adaptive choices require fresh confirmation or an untouched final cohort. |
| P1.4: a stationary chair does not test object-motion generation | **Accepted.** Section 7.7 distinguishes contact-conditioned body generation from a subsequent human/rigid-box generation task, with persistence and uncoupled baselines. |
| P2.1: independent restoration is easy to misread | **Accepted.** Section 4.4 and Figure 1 explicitly restore each development window separately. |
| P2.1: intervals must remain unchanged for a pure rebuild | **Accepted.** Plot code reads existing means/effect intervals; it does not bootstrap or calculate new intervals. The participant seed-mean calculation is only an equality check against the exported value. |

The review record retains the objections so later writers can see why the scope and controls matter. Completed asset checks and any final reviewer findings are recorded separately in the final review and visual QA record.

## Final verification changes

The final reviewer independently matched all 248 empirical marks, including 80 interval/range rows, to the source exports and confirmed the recorded input hashes. Two further corrections were accepted: the new preservation-endpoint explanation now states that equal per-leg errors can cancel in a signed difference, and the rebuild instructions name the actual Python 3.12.9 interpreter used rather than assuming the repository's bare `python` resolves correctly.
