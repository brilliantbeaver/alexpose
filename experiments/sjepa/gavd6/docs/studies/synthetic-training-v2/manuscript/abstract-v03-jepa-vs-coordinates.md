# Revision 03 — make the representation comparison explicit

19 September 2026. Follows [revision 02](abstract-v02-preservation.md), incorporating the independent laterality reviewer’s critique.

## Proposed title

**Testing Latent Prediction for Paired Synthetic Pose Restoration**

## Abstract

Can paired synthetic supervision improve estimated 2D pose trajectories while preserving movement amplitude, timing and left–right relationships? We examine what predicting learned target features adds when coordinate-based methods receive the same synthetic targets. Two earlier studies motivate the controls: summary-reported laterality experiments found that artificial joint-name swaps affected side-sensitive measurements without demonstrating a benefit from informative uncertainty, and retained synthetic-adaptation outcomes showed no benefit from response personalization over a matched ablation. Our pipeline pairs pose estimates from controlled renders with projected anatomical proxies. A joint-embedding predictive architecture (JEPA) predicts the reference encoder’s features; a separate readout converts its frozen representation into coordinates. Comparisons include coordinate pretraining, direct restoration, an initialized encoder with a fitted readout, and shuffled pretraining pairs. In a summary-reported synthetic pilot with two development people and one seed, paired JEPA reduces normalized coordinate error by 35.6–45.3% across three extractors, one held from fitting. Relative to coordinate pretraining, gains are below 1% on two extractors and error increases by 14.0% on the third. Initialized and shuffled controls yield similar coordinate-error point estimates. The movement evaluation includes signed image-plane ankle separation, but amplitude is supported on only 12 of 48 correlated records and paired JEPA has no eligible timing comparisons under the current peak rules. Calibration controls are planned to assess coordinate gains, while reference self-comparison will distinguish reference ineligibility from prediction failures. Motion preservation and transfer to real video remain unestablished.

## Changes and critique

The contribution now concerns the incremental value of predicting target representations when coordinate controls have the same privileged synthetic labels. “Similar point estimates” avoids implying statistical equivalence. The draft identifies the implemented bilateral proxy as signed image-plane ankle separation, and separates the two pending diagnostic questions: calibration concerns coordinate gains; reference self-comparison concerns timing eligibility.

The scientific positioning follows primary sources reviewed in the [literature record](abstract-literature-positioning-20260919-v01.md): S-JEPA already predicts skeletal representations, SmoothNet refines pose sequences, and PoseBERT/MotionBERT already learn from corrupted motion observations. None supports assuming that JEPA intrinsically discards anatomical side, or that a lower latent loss establishes preservation. No “first,” state-of-the-art, world-model forecasting or clinical-utility claim is made.

Remaining weaknesses: the opening suggests a broad preservation study, but the available bilateral endpoint is narrow and there are no reported amplitude-error values. “Three extractors” may appear to enlarge the sample if the two-person limit is overlooked. Coordinate gains over unchanged tracks occupy more space than the decisive matched comparison. Revision 04 should lead the results with that comparison, state uncertainty where it changes interpretation, and explain why absent timing support cannot distinguish reference weakness from motion distortion.

## Fixed-rubric assessment

| Dimension | Weight | Score / 10 | Reason and remaining weakness |
| --- | ---: | ---: | --- |
| Conference relevance and contribution | 20% | 7.0 | Explicitly asks what learned target features add under common supervision, a clearer representation-learning contribution. |
| Claim accuracy and evidence support | 20% | 8.5 | Uses proxy targets and descriptive point estimates; summary-only availability is clear, but timing support and broad bilateral scope still need explanation. |
| Evaluation and statistical rigor | 15% | 3.5 | No new evidence: two evaluation people, one seed, incomplete movement support and no verified transfer. |
| Scientific insight and related-work positioning | 15% | 7.0 | Distinguishes a same-supervision objective comparison from existing denoising and masked motion pretraining; mechanisms remain unresolved. |
| Reproducibility | 10% | 5.5 | Available code and aggregate audits aid inspection; missing original predictions limit independent reconstruction. |
| Clarity and narrative | 10% | 7.0 | The main comparison leads, but history, architecture and limitations make the prose dense. |
| Figures | 5% | 7.5 | Shared schematic is clean and reviewed; readout supervision and signed ankle separation are explained in its caption. |
| Submission fit | 5% | 7.5 | Fits representation-learning analysis and avoids an unsupported new-method claim; complete-paper evidence remains weak. |

**Weighted total: 66.75 / 100.** The higher positioning score reflects an explicit distinction from verified prior work, not new performance evidence. Evaluation and reproducibility remain fixed.

## Evidence and figure

See the [laterality audit](abstract-laterality-audit-20260919-v01.md), [original-study audit](abstract-original-study-audit-20260919-v01.md), and [source-pilot summary](../proposal/README.md). The [common pipeline figure](images/abstract-training-pipeline-20260919-v01.svg) is conceptual. Projected training references supervise the coordinate readout; inference uses only observations. No proposed calibration or timing self-check is represented as completed.


**Precision note from final validation:** direct rounding of the reported errors gives a paired-versus-unchanged reduction range of **35.6–45.2%** at one decimal, or **35.63–45.25%** at two decimals. The abstract above used 35.6–45.3% after intermediate rounding; this note corrects that precision issue while preserving the draft. Revisions 05–07 omit this range and emphasize matched controls.
