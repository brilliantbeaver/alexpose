# Proposal narrative and mathematical review — 24 September 2026

This review covers the revised [proposal](../proposal.html), its [Markdown source](../README.md), offline mathematical rendering, and the new [two-readout diagram](../images/proposal-readout-design.svg). It assesses scientific communication and document behavior, not the completion or success of the follow-up experiment.

## Independent review process

Three review agents independently assessed the original proposal before the rewrite:

- Scientific motivation, causal reasoning, story structure, and experimental purpose.
- JEPA terminology, exact objectives, calibration, evaluation, and agreement with the implementation.
- Primary-source support for the Delp/OpenCap and Landay/Stanford Ambient Intelligence connections, plus readability.

The author rewrote the scientific source and regenerated the HTML. All three reviewers then reassessed the new text; the scientific and technical reviews explicitly challenged alternative explanations and overstated interpretations. The technical reviewer also supplied the new diagram in a separate, bounded task. The author inspected its rendering and performed the final browser checks.

## Findings and revisions

| Review finding | Revision |
| --- | --- |
| The proposal predates the completed core and the two-readout amendment. | Distinguish observed core results from the proposed `jepa-response-02` design: nine encoders, eighteen readouts, 27 phases, and eighteen final models. Preserve the original primary comparison and disclose the amendment's post-core timing. |
| Protocol detail obscures the research motivation. | Organize the argument around measurement validity, a controlled movement change, the core findings, the pretraining hypothesis, and the evidence needed to interpret the result. Move exact scaling formulas into an expandable appendix and operational detail into linked documents. |
| Weak JEPA results are treated as evidence that movement information was lost. | Present encoder information, readout behavior, objective tradeoffs, and prediction reliability as competing explanations. |
| The reason for two readouts is unclear. | Show the core's direct/base and direct/change results, then explain why each new encoder is tested under both downstream objectives. |
| A result under one readout could be mistaken for an established interaction. | Require the direct difference between effect estimates and its uncertainty; a clear result under one readout and an uncertain result under another is insufficient. |
| Lower response error could reflect shared bias, small true changes, or fewer failures. | Explain the role of individual excursions, waveforms, a zero-change benchmark, response curves, and the matched-denominator failure decomposition. Avoid equating a reliability gain with better movement information. |
| Shuffled-reference control could be confused with a movement-pairing control. | State that no new movement re-pairing arm isolates a unique causal contribution of anatomically correct pairing. |
| Stanford application links could imply adoption or clinical validation. | Cite OpenCap's actual measurement pipeline and the AmI lab's stated planned studies; limit transfer claims and explicitly avoid claiming collaboration or adoption. |
| Sobolev training is not the same as this paired difference objective. | Distinguish derivative supervision from unnormalized feature differences. |
| Equations are code blocks and the HTML has no math renderer. | Parse LaTeX at build time and render vector SVG equations with source alternative text. Use standard notation without the prohibited operator-name macro. |

## Validation

The independent technical review checked all 31 mathematical instances and visually inspected all eight display equations. It numerically compared the split centering definitions, shared-center cancellation, cross-term identity, latent objectives, and coordinate rescaling/reduction with the production implementation. No mathematical discrepancy remained. The final text also distinguishes gradient-scale calibration from loss-value scaling and explicitly states the primary no-change exclusion.

The author checked the generated page in headless Chromium at 1440 × 1000 and 390 × 844. All 33 content images loaded. The document had no page-level horizontal overflow or browser console errors. Long mobile equations scroll within their own containers. Both interactive calculators produced the expected values for shared, opposed, zero, and attenuated examples; controls enabled successfully, and the figure dialog opened and closed.

The [document validation record](../records/proposal-revision-validation-20260924.json) records structural checks, final artifact hashes, and browser findings. Source and generated HTML are rebuilt together so a future regeneration preserves the revision.

The core numerical table was checked against the supplied development analysis. Those exported results do not verify absent raw predictions or source-cohort receipts. No follow-up scientific finding, real-world transfer, clinical validity, privacy guarantee, or camera-placement optimum is established by this document review.
