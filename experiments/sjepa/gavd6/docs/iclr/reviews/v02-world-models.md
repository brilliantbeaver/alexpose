# Version 02: independent world-models and methods review

Reviewed frozen `paper-v02.tex`, its complete diff from v01 and `figures/v02` diagram PNGs. This reviewer did not edit the manuscript. Prior implementation audit remains the evidence base; the official S-JEPA paper was independently opened for the closest-work check below.

## Objection status

| ID | Severity | Evidence and status | Correction/disposition | Residual limitation |
|---|---|---|---|---|
| W01-1 | Material, **resolved** | Caption now explicitly separates idealized exchange of already-computed excursions from the executed 3D mirror, which is reprojected and need not reverse `A`. The revised figure uses “Idealized exchange of side measurements.” | No further factual correction needed. | Actual learned equivariance remains untested. |
| W01-2 | Moderate, **resolved** | Abstract and Results headline now say original change-supervised objective, while methods retain its scalar+geometry composition. | Keep this attribution in all summaries. | Dense-versus-scalar still does not isolate one mechanism. |
| W01-3 | Moderate, **resolved** | Dense equation now includes expectation over `(a,b)` and states equal supported-pair weighting. | Matches `repair_objectives.py:20–37`. | None beyond support scope. |
| W01-4 | Moderate, **partly resolved** | Hidden-value exclusion is now explicit, and availability replaces ambiguous “mask” in the diagram. The median/quantile-span normalization and degenerate fallback remain absent. | Give exact normalization briefly, or provide a precise reproducibility supplement pointer. | Privileged raw assets/checkpoints are outside compact evidence. |
| W01-5 | Moderate, **unresolved layout** | Model diagram now has inset boxes, but “+ confidence, availability” extends through the input edge onto the arrow; “Projected reference” and “Trained residual readout” exceed their boxes; the bottom residual arrow still crosses the explanatory text. Laterality right title also slightly exceeds its box. | Wrap text, shorten labels without shrinking final-size typography, increase box widths/gaps, and move the residual explanation below the arc. | The diagram must be inspected after layout changes. |
| W01-6 | Minor, **open** | The 0.05 VICReg coefficient, invariance term, exact scalar/dense calibration, EMA schedule, and coordinate-MSE units are still implicit. | Compact methods additions can resolve this; prioritize exact calibration and normalization. | They cannot prove convergence or match optimization budgets. |
| W02-1 | Moderate positioning | Closest skeletal feature-prediction precedent S-JEPA is present in bibliography but uncited. Its official paper uses masked 3D skeleton features, EMA targets, centered/sharpened CE with temperatures .1/.06, and downstream action recognition. The local recipe inherits central choices but differs in target privilege, 2D restoration and deployed encoder. | Cite S-JEPA directly in the predictive-method paragraph and explain the evaluated adaptation. Do not imply I-JEPA is the nearest source of this channel-CE skeleton recipe. | No exact reproduction or state-of-the-art comparison is claimed; preserve that boundary. |
| W02-2 | Bounded research limitation | The test does not expose intervention labels/actions to the model or forecast a missing future interval. | Current full-window restoration wording is correct and should remain. | A world-model claim requires different experiments. |

The newly specified assignment rule accurately matches `evaluation.py:70–85`: reference bilateral separation at least 4 px, 2 px named-versus-swapped margin, with wrong/ambiguous/nonfinite predictions retained as failures. Its denominator is separate from angular support. The introduction now correctly notes that signed bilateral response does not uniquely identify the changed leg.

The strongest narrative remains the conditional representation hypothesis and the diagnostic sequence from scalar response to waveform damage to controlled readout weighting. The negative primary results and counterevidence remain visible. No new material numerical or method misstatement was found in this version. The open items are reproducibility/positioning and figure quality, together with scientific limits already disclosed.

Primary reference checked: [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf), particularly method and implementation sections. It studies action recognition from skeleton sequences; it does not establish the utility of the present synthetic-reference restorer.

## Fixed rubric

| Dimension | Weight | Score /10 | Concrete reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution |20%|6.5|Bounded empirical representation evaluation; the population and incomplete optimization controls still limit reach.|
| Claim accuracy and evidence support |20%|9.0|Physical-mirror and compound-loss statements corrected; fixed-scope claims now match source.|
| Evaluation and statistical rigor |15%|7.0|No new evidence; small adaptive cohort and three seeds unchanged.|
| Scientific insight and positioning |15%|7.5|Useful control logic; closest S-JEPA attribution remains underdeveloped.|
| Reproducibility |10%|7.5|Hidden-value boundary improved; exact normalization/calibration still incompletely stated.|
| Clarity and narrative |10%|8.0|Laterality meaning improved, but core scientific narrative has not materially changed.|
| Figures |5%|6.5|Factual diagram correction earns credit; important overflow/arrow collision persists.|
| Submission fit |5%|7.5|Rendered nine-page and final-reference checks remain outside this methods review.|
| **Weighted total** |**100%**|**75.25 /100**|Increase is due to corrected claims and diagram contract, not version number.|

No deduction for unresolved data limitations should disappear through wording alone. A later manuscript can be more precise and reviewable while retaining the same ceiling on empirical scope.
