# Final disposition

28 September 2026. Lead-editor record for the compact brief. The lead agent wrote the manuscript; bounded evidence, methodology, and future/figure agents supplied audits. The evidence auditor and future-direction auditor reviewed the draft without authoring it.

## Review findings and revisions

| Finding | Resolution |
|---|---|
| Two floated figures created separate pages and violated the brief limit. | Rebalanced the three pages, removed the redundant abstract, and integrated both figures at readable proportions. Final PDF: three body pages plus one reference page. Original typography and page geometry are unchanged. |
| Absence of a completed confirmation evaluation was confused with absence of an assigned cohort. | The manuscript now states that no untouched confirmation evaluation was completed. |
| The proposed endpoint could be mistaken for v08's projected 2D quantity. | The proposal explicitly names anatomical 3D knee excursion in each leg, using synchronized independent references. Intended movement changes exceed reference uncertainty in at least one leg; observation-only nulls and unchanged-leg controls remain meaningful. |
| Ambiguity search was underspecified. | The hypothesis now seeks trajectories that fit visible joints equally well but imply different per-leg changes. Failure to find an alternative does not certify uniqueness. |
| Closest uncertainty precedent and baseline were insufficiently visible. | Added Pace et al. alongside the reconstruction and JEPA gait precedents, and a calibrated change-uncertainty comparator in addition to visibility and ensemble disagreement. |
| Repeated comparisons and selective reporting could bias apparent improvement. | Participants are separated for fitting, calibration, and final testing; uncertainty is estimated across people. Success compares the same proportion of whole trial pairs and audits which participants are declined. The working notes specify retaining repeated pairs/masks within each person and auditing occluded/atypical strata. |
| Failure costs could be mistaken for observed motion magnitudes; the schedule could include untrained controls. | Named 720°/180° as fixed worst-case costs and qualified the 2,000+2,000-update schedule with “pretrained encoders.” |
| Source provenance and novelty could be overstated during compression. | Named the training-subset addition as diagnostic evidence, retained clinical/generalization limits, cited the anonymous v08 source explicitly, and made no “first-ever” claim. |

The independent reviews verified the reported quantitative claims, privileged-supervision and restoration terminology, clinical limits, external figure license and identity, and byte-identical reuse of v08's result figure. The future reviewer closed all material findings after revision. The final independent evidence/layout review is recorded in [adversarial-final-review.md](adversarial-final-review.md).

## Final artifact inspection

The lead agent inspected all four final PNG pages after the final compilation, including the added v08 reference and final future-work wording. Both figures and captions are on body pages; no orphaned figure, clipped line, missing glyph, unresolved citation, or page overflow remains. All PDF fonts are embedded. Long bibliography URLs produce underfull-box warnings without clipping. The figures' original proportions are retained.

The final source/PDF hashes and source-study hash are in [qa/verification.json](../qa/verification.json). The inspected manuscript hash is `6331d57e78fbfd8ee2a03d29ee0f2bf144d757cd9e7940d2996c81242238fc6f`; the inspected PDF hash is `f3f9c6f28d19721ca353447f58ed48926804edd21200dfda9e97f86901ff23a1`.

## Remaining scientific uncertainty

There is no unresolved numerical contradiction in the retained claims. The completed development motion count is not established, training subsets required artifact inspection, and full predictions/checkpoints are absent from the retained packet. The study's repeated development evaluation does not establish clinical or older-adult generalization. The recommended diagnostic still requires an experiment and an advantage over simpler baselines; the literature search establishes close precedents, not proof of unprecedented novelty.

## User-requested terminology clarification

Replaced “privileged synthetic supervision” in the methods and corresponding caption terminology with “clean synthetic reference poses available only during training.” Updated the editable manuscript and rebuilt the PDF. The affected pages 1–2 were visually inspected; the three-body-page plus one-reference-page layout and build checks still pass. This wording-only change does not modify the scientific claims or reopen the independent review.

Current manuscript SHA-256: `86a176ee816b0eb5db2cbaaa2b2bd73229cc7ad909ec4c1014a8f4784f268e54`. Current PDF SHA-256: `a2805e52128b6ea58cb534497102dcee080bdea7294f2916a2c478576aa4c783`. Earlier hashes above identify the independently reviewed version.

## Whole-document terminology revision

The user requested clearer loss names, fuller explanations, and a systematic pass over technical language. The lead editor revised the manuscript and captions, retaining the unchanged original graphics and mapping their legacy labels to the new names. The [terminology audit](TERMINOLOGY-AUDIT.md) records the definitions, source-fidelity checks, and visual inspection. This pass did not rerun the earlier independent scientific review or perform new experiments.

The rebuilt PDF retains three body pages plus references. All four pages were visually inspected after the substantive revision, and the final caption adjustment was rechecked on page 1. The latest manuscript SHA-256 is `060da1457627834401980c3b73cd8102be3a807da4fac9bcb651e0f8050ea44e`; the latest PDF SHA-256 is `98b96374c9de8303be4716a4c2c95fd0b1e0ddba31a5bf25a02a59ef5b8f69ab`. Earlier hashes and reviewer statements above refer to the corresponding historical revisions.

## User-requested em-dash removal

Replaced both paired em-dash constructions with a relative clause and direct sentence phrasing. Confirmed that neither the editable manuscript nor extracted PDF text contains em-dashes. Rebuilt the PDF and inspected the affected first page; the three body pages plus references and artifact checks still pass. Current manuscript SHA-256: `76770c1bee8f151f5cfd9099376572efe8b0d3c22e9bcabd356e8abd4a117d51`. Current PDF SHA-256: `1b77227705e370628a53ba026bb1026e014de7970fd9f98c10e6f37cf70eac7d`.

## Whole-document punctuation and transition review

Reviewed every paragraph, heading, and caption for sentence flow and punctuation. Replaced list-like clauses with connected explanations, clarified the relationships between the training objectives and evaluation measures, and recast the proposed study's procedural instructions as proposal prose. Retained the quantitative results, technical definitions, citations, and scientific qualifications. Shortened the future-study heading without changing the proposed research question.

Rebuilt the PDF and visually inspected all four newly rendered pages. The document retains three body pages and one reference page, with unchanged typography, margins, and figure assets. Build checks pass with no overfull boxes or unresolved references; neither manuscript nor extracted PDF text contains em-dashes. This editorial pass did not rerun the independent scientific review or conduct new experiments.

Current manuscript SHA-256: `44b98200b503530f8ed87b1980b5d25652724b40f123b6218bee37662268edf3`. Current PDF SHA-256: `8daf8cf46d21687f3190370cf48da27a495727389572497cdfb588fed2a94150`. Earlier hashes identify historical revisions.
