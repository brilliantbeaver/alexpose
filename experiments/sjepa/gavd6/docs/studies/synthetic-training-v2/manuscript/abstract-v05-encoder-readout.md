# Revision 05 — interpret the readout control

19 September 2026. Follows [revision 04](abstract-v04-baselines.md), incorporating the independent original-study reviewer’s second critique.

## Proposed title

**Evaluating Paired Synthetic Supervision for Movement-Preserving Pose Restoration**

## Abstract

Pose restoration for movement analysis needs to retain amplitude, event timing and left–right joint relationships as it corrects coordinates. Reported laterality experiments show how temporary joint-name swaps distort side-sensitive summaries; their uniform control and an earlier synthetic study’s response-free selector reproduced gains attributed to more elaborate mechanisms. We therefore evaluate paired synthetic supervision with controls for the contribution of encoder pretraining. Our pipeline aligns image-estimator tracks from rendered clips with projected body-model joints used as anatomical proxies. A joint-embedding predictive architecture (JEPA) predicts reference features, then uses a separately fitted coordinate readout on its frozen encoder. Coordinate pretraining and direct restoration receive the same targets; further controls use an untrained frozen encoder or shuffled pretraining pairs. Fitting excludes development people and ViTPose. In a summary-reported pilot with two development people and one seed, paired JEPA improves coordinate error over coordinate pretraining by less than 1% for HRNet and ViTPose, with reported intervals including zero, but worsens RTMPose by 14.0%. Fitted readouts on untrained encoders also substantially improve agreement with synthetic coordinates, weakening attribution to encoder pretraining. For signed image-plane ankle separation, amplitude is supported on 12 of 48 correlated records; no paired-JEPA record satisfies the joint reference-and-prediction timing criteria. Calibration and reference self-comparison remain necessary to distinguish coordinate correction from representation gains and to locate the timing limitation. These checks, repeated-seed evaluation and independent real-video references are future work; movement preservation is unresolved.

## Changes and critique

The initialized control is now described as an untrained frozen encoder followed by a fitted coordinate readout. This localizes the question to the value of encoder pretraining, while retaining the fact that the readout learns. “Shuffled pretraining pairs” is deliberate: its readout still receives aligned training targets. HRNet, ViTPose and RTMPose are named so the worsening result cannot be mistaken for the held-family outcome. Removing the raw-track percentage range creates space for this interpretation.

The predecessor sentence is still compressed. A reader unfamiliar with the studies may not understand what the uniform control or response-free selector means. The end also reads like a work plan because the completed evidence cannot resolve preservation. That scientific limitation must remain, but the prose can state the current result and separate the next analyses more economically. The title's “movement-preserving” describes the goal; it should not be used without “evaluating” while preservation is unproven.

For revision 06, explain the predecessor findings as two short connected observations, expand JEPA without overwhelming the method sentence, and identify the output decoder/readout in ordinary language. Preserve the matched-comparison numbers and uncertainty. Do not replace missing preservation evidence with a confident claim that JEPA erases motion or that calibration explains the gains.

## Fixed-rubric assessment

| Dimension | Weight | Score / 10 | Reason and remaining weakness |
| --- | ---: | ---: | --- |
| Conference relevance and contribution | 20% | 7.0 | Keeps the paper centered on representations that support movement measurement; an effective preserving method has not been established. |
| Claim accuracy and evidence support | 20% | 9.0 | Names the affected extractors, untrained encoder, proxy targets and shuffled-pretraining boundary without asserting equivalence. |
| Evaluation and statistical rigor | 15% | 3.5 | Same single-seed, two-person synthetic pilot, low support and missing independent real references. |
| Scientific insight and related-work positioning | 15% | 7.5 | Readout-only improvement is connected to the attribution question; calibration remains a hypothesis grounded in primary literature. |
| Reproducibility | 10% | 5.5 | Available aggregate evidence is inspectable; the summary-backed newer results cannot yet be independently reconstructed. |
| Clarity and narrative | 10% | 8.0 | Less numerical distraction and clearer control interpretation, although the historical sentence remains compressed. |
| Figures | 5% | 7.5 | The shared figure explains the pipeline and pending work, with print-size and empirical-plot limitations unchanged. |
| Submission fit | 5% | 8.0 | Within the editorial abstract-length target and focused on meaningful learning questions; the paper still needs substantive additional evidence. |

**Weighted total: 69.75 / 100.** Insight and clarity improve because the trained readout is distinguished from encoder pretraining. Accuracy stays at 9.0: these are carefully qualified summary-backed claims, not independently verified runs.

## Selected and omitted evidence

The [original audit](abstract-original-study-audit-20260919-v01.md) supports the matched response-free comparison; the [laterality audit](abstract-laterality-audit-20260919-v01.md) supports controlled naming errors and the uniform-control finding. Their full metrics stay outside the abstract because they do not measure v2 restoration.

The [source summary](../proposal/README.md) supplies the current numbers. The [new exploratory ratio analysis](abstract-20260919-exploratory-summary.json) supports arithmetic checks only. No calibration fit, global anatomical-side recovery or real-video transfer is claimed.

The [shared conceptual figure](images/abstract-training-pipeline-20260919-v01.svg) depicts pretraining and readout fitting separately. Projected targets also supervise the readout during training. Its bilateral endpoint means signed horizontal ankle separation, and the final strip names pending analyses.

