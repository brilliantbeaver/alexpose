# Revision 02 — connect the studies to the preservation question

19 September 2026. Follows [revision 01](abstract-v01-pilot.md), incorporating the independent original-study reviewer’s critique. Historical files and the preceding draft are preserved.

## Proposed title

**Paired Synthetic Supervision for 2D Pose Restoration: Accuracy and Movement Preservation**

## Abstract

We ask whether paired synthetic supervision improves 2D pose trajectories while preserving movement amplitude, timing and left–right joint relationships. Earlier controlled laterality experiments showed why exchanges of joint names matter for side-sensitive measurements, although a uniform-uncertainty control reproduced the proposed model’s gain. In a separate synthetic-adaptation study, response-based lesson selection did not improve on its matched response-free control. These findings motivate testing the contribution of learned representations against simpler explanations. We pair pose estimates from rendered observations with same-view projected body-model joints, used as anatomical proxies, and compare predicting target features with reconstructing target coordinates. Each pretrained encoder receives a separately fitted coordinate readout; additional controls include direct restoration, an initialized encoder and shuffled pretraining pairs. A reported one-seed pilot evaluates two development people and three pose extractors, including one excluded from fitting. Paired JEPA lowers normalized coordinate error by 35.6–45.3% relative to unchanged tracks, but improves only two extractors by less than 1% over coordinate pretraining and worsens the third by 14.0%. The initialized readout and shuffled pairing are competitive. Amplitude evaluation is supported on 12 of 48 correlated records; none satisfies the reference/prediction peak requirements for paired-JEPA timing evaluation. Thus coordinate gains leave motion preservation unresolved. Planned calibration and reference self-comparison checks will test whether systematic correction or weak observability explains these findings. The laterality and restoration results are summary-reported here, and real-video preservation remains untested.

## Changes and critique

The earlier study now enters through its matched response-free control, rather than treating an unmatched scene-policy comparison as a causal ablation. The laterality finding motivates bilateral measurement, and the target is described as an anatomical proxy. The draft also clarifies that coordinate and JEPA arms share synthetic supervision.

The remaining weakness is emphasis: readers still encounter a history of projects before understanding the paper’s representation-learning contribution. “Predicting target features” is accurate but needs an explanation of how that differs from the coordinate objective. The title names a research scope; it does not claim that preservation succeeded.

For revision 03, foreground the empirical question of what latent prediction adds when the same paired targets and readout architecture are available. Use the predecessor findings briefly to justify strong controls, then place this comparison beside S-JEPA, SmoothNet, PoseBERT and MotionBERT in the accompanying rationale. No new experimental evidence is added by that positioning.

## Fixed-rubric assessment

| Dimension | Weight | Score / 10 | Reason and remaining weakness |
| --- | ---: | ---: | --- |
| Conference relevance and contribution | 20% | 6.5 | Question now concerns representation value with common synthetic supervision; broader scientific contribution is still preliminary. |
| Claim accuracy and evidence support | 20% | 8.5 | Anatomical proxies, matched response-free control and timing eligibility are explicit; unavailable underlying runs still limit verification. |
| Evaluation and statistical rigor | 15% | 3.5 | Unchanged evidence: two development people, one seed, correlated records and no demonstrated preservation or real transfer. |
| Scientific insight and related-work positioning | 15% | 6.0 | Separates latent targets from coordinate targets, but the abstract still spends considerable space on predecessor histories. |
| Reproducibility | 10% | 5.5 | Original aggregate artifacts are inspectable; laterality and restoration predictions remain unavailable locally. |
| Clarity and narrative | 10% | 7.0 | Motivation is more coherent and support is explained, although the abstract still contains several competing threads. |
| Figures | 5% | 7.5 | Shared pipeline has been rendered and independently inspected; no empirical trajectories are supplied by the available evidence. |
| Submission fit | 5% | 7.5 | Substantive, anonymous and relevant to representation learning; present evidence is insufficient for a strong complete-paper claim. |

**Weighted total: 64.25 / 100.** Rigor and reproducibility remain unchanged. Better specification supports the accuracy and clarity increases; the score is not an acceptance estimate.

## Evidence and illustration

The [laterality summary](../../latent-laterality/results/validation.md), [retained synthetic-study audit](abstract-original-study-audit-20260919-v01.md), and [reported restoration pilot](../proposal/README.md) concern different experiments. Their sample sizes, errors and uncertainties must not be pooled.

![Conceptual paired-supervision training pipeline, with unfinished diagnostic checks marked as pending.](images/abstract-training-pipeline-20260919-v01.svg)

Projected references also supervise readout fitting on training people; the encoder and readout are fixed at inference. “L/R separation” means signed horizontal left-minus-right ankle separation, not clinical anatomical-side recovery. This common diagram is conceptual; its figure score stays constant throughout the series.


**Precision note from final validation:** direct rounding of the reported errors gives a paired-versus-unchanged reduction range of **35.6–45.2%** at one decimal, or **35.63–45.25%** at two decimals. The abstract above used 35.6–45.3% after intermediate rounding; this note corrects that precision issue while preserving the draft. Revisions 05–07 omit this range and emphasize matched controls.
