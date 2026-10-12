# Version 04 — independent scope, literature, writing, and rendered-PDF review

Reviewed `paper-v03.tex` → `paper-v04.tex` changes and rendered v04 PDF pages 2–9 with Poppler at 100 dpi; visually inspected pages 2 and 4–9, then page 8 again at 180 dpi. `pdfinfo` reports 11 total pages. The main discussion ends on page 9; disclosure/references occupy subsequent pages. The inspected figure pages have readable labels, intact equations, no clipped boxes, and no text/arrow overlaps. Page 5 is dense but remains within the intended text area.

The current version reads more like a complete paper. It defines normalization and the assignment diagnostic where needed, explains important regularizer and optimizer choices, and states the condition-reaggregation weights. These are real reproducibility improvements. The closest-work account remains accurate, and earlier misleading scalar-only, mirror, and “ranking” language has been corrected.

| ID | Severity / status | Evidence | Correction or disposition | Residual limitation |
|---|---|---|---|---|
| E04-1 | Resolved | Readout is explained at first use; repeated “through” is removed; discussion now characterizes implemented procedures and states the 93% descriptive recovery directly. | Accept the copy edits. | Primary benefit remains unresolved and is stated plainly. |
| E04-2 | Minor literature-placement issue | Section 4 attaches both DINO and VICReg citations immediately to the 25/25/1 invariance/variance/covariance weights. DINO does not define that regularizer. | Attach DINO to centered/sharpened teacher self-distillation and VICReg alone to the invariance/variance/covariance term. | No change to scientific findings. |
| E04-3 | Moderate narrative opportunity | Laterality appears in the motivation and schematic but its empirical results are one dense paragraph after the repair analysis. The user asked for a laterality-focused paper. | Add a compact naming-condition plot from the completed exports, retaining its exploratory status, and move the training-population probe to an appendix. Use the freed space to show the failure pattern directly. | Does not become a predeclared primary endpoint or clinical side-identification test. |
| E04-4 | Reproducibility improved, still incomplete | Normalization, initial gradient calibration, regularizer coefficients, optimizer schedule, and reaggregation weights are now in the main text. Artifact references remain descriptive rather than an exact executable appendix. | Add concise appendix commands and source-to-claim/figure paths in the final package; distinguish summary reconstruction from original fitting. | Raw coordinates/checkpoints are unavailable locally, so a complete rerun remains unsupported. |
| E04-5 | Rendered figure objections resolved | Figures 1–5 are legible at their PDF placement. Figure 2 shows a one-state path and explains that paired readout fitting compares two outputs. | Accept current figure rendering; rerender any new empirical plot at the same scale. | No empirical raw-frame trajectory demonstration is supplied. |
| E04-6 | Persistent empirical limits, correctly bounded | Fourteen reused development people; three seeds; incomplete independent test; uncertain feature and repair primaries; zero-response predictor wins pooled response error. | Retain these limitations visibly. Do not replace the uncertainty with a more exciting interpretation. | Requires new experiments/data. |

## Concrete laterality revision for v05

Use a standalone grouped-dot plot with **assignment failure (%)** on a 0–100 axis and the three naming conditions: correct, temporary swap, global swap. Show Direct/base, Endpoint/change, and Delta/change with both color and distinct markers. All three should use the same pooled-estimator, person/seed aggregation already used for the laterality numbers. The caption must define wrong/ambiguous/missing predictions as failures, identify the reference-separated joint-pair denominator, say 14 development people and three seeds, and mark this as a post hoc descriptive analysis. Do not add a 50% chance line.

This plot answers a question that the pooled response scores cannot: **Do the restorers recover anatomical naming when the input channels are corrupted?** The main text should connect the answer to the scalar result without claiming a statistical effect: delta/change's lower response point estimate coexists with slightly higher pooled assignment failure than endpoint/change. Both retain large failures under global renaming. Preserve the separate excursion-result caveat so the reader does not infer direct/base leads every measurement.

Move the training-person probe paragraph to an appendix because its result cannot establish deployment generalization or unique information loss. Replace the long naming-number paragraph with the plot, caption, and two sentences of interpretation. Keep the ViTPose-only readout repair visually separate from this pooled-estimator diagnostic. A figure mixing those populations would need prominent panel-specific context and is less readable here.

## Fixed-rubric scores

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Relevance and contribution | 20% | 6.0 | Clear and useful diagnostic evaluation, still narrow and lacking new confirmatory evidence. |
| Claim accuracy and evidence support | 20% | 9.0 | Material scope corrections retained; one minor citation placement issue. |
| Evaluation and statistical rigor | 15% | 6.0 | Better described aggregation; same underlying sample, reuse, optimization, and external-baseline limits. |
| Scientific insight and related-work positioning | 15% | 7.0 | Close precedents correctly positioned; empirical laterality argument could be clearer; no unique mechanism. |
| Reproducibility | 10% | 7.5 | Exact normalization, calibration, optimizer, and scoring/reaggregation information materially improve reproducibility. Full artifacts remain unavailable locally. |
| Clarity and narrative | 10% | 8.0 | First-use definitions and copy edits are improved. Laterality findings remain buried in a dense paragraph. |
| Figures | 5% | 8.5 | Printed-size figures are readable and accurate; a laterality result plot would better serve the claimed focus. |
| Submission fit | 5% | 9.0 | Nine-page main text and post-main disclosure/reference placement verified; anonymous metadata and correct title present. Human submission verification remains outstanding. |
| **Weighted total** | **100%** | **73.75 /100** | Gain reflects reproducibility and rendered compliance, not stronger empirical evidence. |

No unresolved material literature or scope misstatement was found in this pass. The remaining major weaknesses require new experiments and should stay explicit through v07.
