# Version 03: independent world-models and methods review

Reviewed frozen `paper-v03.tex`, its diff from v02, both diagram PNGs, and the changed figure-generation logic. This reviewer did not edit the paper. New numerical per-leg and laterality contrasts remain subject to the independent evidence reviewer; the implemented definitions were independently checked.

## Findings and status

| ID | Severity/status | Evidence | Correction or disposition | Residual limitation |
|---|---|---|---|---|
| W01-1/W01-2/W01-3 | Resolved, retained | Idealized side-exchange contract, original compound-loss attribution, and outer equal-pair dense reduction remain correct. | No change requested. | No equivariance or unique sparsity mechanism claimed. |
| W01-5 | Resolved in substance | v03 diagram labels are wrapped/shortened, boxes are no longer clipped, and a separate residual lane avoids the former text collision. The bottom-right pretraining loss label is close to its box edge but readable. | A small optional shortening to “CE + regularizer / delta or endpoint auxiliary” would improve padding. | Final PDF-scale inspection remains necessary. |
| W02-1 | Resolved | S-JEPA is now directly credited in Methods. Discussion distinguishes its action-recognition setting and target-encoder deployment from the current 2D restoration path. PoseBERT, MotionBERT and masked-skeleton feature prediction establish close precedents without claiming external baselines were run. | This is the correct positioning. Preserve finite-procedure evaluation scope. | No state-of-the-art comparison or generic JEPA conclusion. |
| W01-4/W01-6 | Moderate reproducibility, open | Exact context median/quantile normalization, its fixed fallback, base CE/VICReg formula/weight, optimizer/EMA, auxiliary calibration and support details remain absent from main text. | Add an appendix with formulas and code/artifact pointers. Detailed source-grounded compact formulas have been sent to the writer. | Correct formulas cannot compensate for absent raw/checkpoint packet or convergence sweeps. |
| W03-1 | Minor mathematical interpretation | Feature residuals use each endpoint's own masked observation-derived normalization. A latent difference can therefore include normalization/context effects as well as the intended movement change (`training.py:565–575`; `response_calibration.py:68–91`). | Note this limitation in appendix or diagnostics discussion; do not equate feature distance with a physical response. | No-change controls and probes are diagnostic, not proof of physical latent semantics. |
| W03-2 | Minor clarity | Discussion now says the low-weight control “recovers most of the primary repair arm's mean recovery,” an awkward repetition. | “recovers most of the mean improvement obtained by dense repair” states the intended descriptive comparison. | No empirical issue. |
| W03-3 | Bounded scope | Laterality contrast is now correctly labeled post hoc/unadjusted, and per-leg counterevidence prevents a universal direct-model ranking. | Retain both the geometric-diagnostic caveat and counterevidence after statistical verification. | Assignment is still a projected geometric diagnostic rather than anatomical ground truth. |

No unresolved material method misstatement was found in v03. The task remains full-window, privileged-reference restoration; the framing does not claim action-conditioned dynamics, long-horizon world modeling, anatomical pathology, or successful encoder transfer from a teacher-only probe. The improved related-work paragraph makes the empirical contribution more credible without manufacturing algorithmic novelty.

## Fixed rubric

| Dimension | Weight | Score /10 | Concrete reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution |20%|6.5|Empirical contribution unchanged; independent confirmation and stronger optimization controls remain absent.|
| Claim accuracy and evidence support |20%|9.0|Source-correct definitions and bounded claims; new post hoc status is explicit.|
| Evaluation and statistical rigor |15%|7.0|No new independent data, seeds or control experiments.|
| Scientific insight and positioning |15%|8.0|Closest skeletal latent-prediction and temporal restoration work is now integrated accurately.|
| Reproducibility |10%|7.5|Appendix still needed for exact normalization, calibration, support and update rules.|
| Clarity and narrative |10%|8.5|Feature/readout meanings and laterality caveats are clearer, with added counterevidence.|
| Figures |5%|8.0|Major diagram collisions repaired; immutable exact-interval sourcing improves accountability.|
| Submission fit |5%|7.5|Final rendered page count and bibliography compliance pending elsewhere.|
| **Weighted total** |**100%**|**77.25 /100**|Improvement reflects closer positioning, clearer scope and repaired diagrams.|

The data and optimization limitations retain their previous deductions. The proposed appendix can improve reproducibility but should not change the evidence-rigor score unless it reveals additional verified evidence.
