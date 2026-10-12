# Revision 04 — lead with the decisive controls

19 September 2026. Follows [revision 03](abstract-v03-jepa-vs-coordinates.md), incorporating the independent literature reviewer’s critique.

## Proposed title

**Coordinate Gains and Motion Fidelity in Paired Synthetic Pose Restoration**

## Abstract

We investigate whether paired synthetic supervision improves 2D pose trajectories while preserving amplitude, timing and left–right relationships, and whether latent prediction adds value beyond learning coordinates from the same targets. Earlier controlled laterality reports found sensitivity to joint-name swaps without an advantage for informative uncertainty; an audited synthetic-adaptation study likewise found no benefit from response personalization over a matched control. Our pipeline pairs image-estimator tracks with same-view projected anatomical proxies. A joint-embedding predictive architecture learns to predict reference features, after which a separately trained coordinate readout reconstructs trajectories from a frozen encoder. Comparisons include coordinate pretraining, direct restoration, initialized encoders and shuffled pairs. In a summary-reported synthetic pilot with two development people and one seed, paired JEPA improves coordinate error by less than 1% over coordinate pretraining on two extractors, with both reported paired intervals including zero, and worsens the third by 14.0%. One extractor is excluded from fitting. Error reductions of 35.6–45.3% relative to unchanged tracks therefore do not establish a latent-prediction benefit. Movement evaluation uses image-plane ankle separation: amplitude is supported on 12 of 48 correlated records, while no paired-JEPA record satisfies the joint reference-and-prediction timing criteria. Reference self-comparison is needed to locate this support limitation; calibration controls will test a simpler explanation for coordinate gains. These pilot findings leave movement preservation and real-video transfer unresolved, while identifying the comparisons required to attribute restoration gains to learned representations.

## Changes and critique

The reviewer identified a material ambiguity in revision 03: its 35.6–45.3% reduction lacked an explicit baseline. This version names unchanged tracks, leads instead with coordinate pretraining, and includes the two intervals that cover zero. It also replaces “no eligible comparisons” with the joint reference-and-prediction requirements; reference eligibility has not yet been established by a self-check.

The last sentence describes an evaluation contribution, not a demonstrated general principle. Its remaining weakness is abstraction: “informative uncertainty” and “response personalization” make readers decode predecessor terminology. The initialized-readout result is mentioned as a control but not interpreted. Broad bilateral relationships also extend beyond the implemented ankle-separation proxy.

For revision 05, explain predecessor outcomes using ordinary language and give the trained-readout result its proper role. Keep the goal broad but name the executed bilateral measurement. Choose a small number of useful comparisons; omit the historical oracle percentage and distant real-video classification results because they cannot support this restoration claim. Calibration remains future work.

## Fixed-rubric assessment

| Dimension | Weight | Score / 10 | Reason and remaining weakness |
| --- | ---: | ---: | --- |
| Conference relevance and contribution | 20% | 7.0 | Frames a controlled question about the incremental value of latent prediction; generality of the insight is still untested. |
| Claim accuracy and evidence support | 20% | 9.0 | Explicit baseline, uncertainty, proxy targets and joint timing criteria prevent the main misinterpretations. |
| Evaluation and statistical rigor | 15% | 3.5 | Unchanged experiment: two development people, one seed, low support and no real-video preservation evidence. |
| Scientific insight and related-work positioning | 15% | 7.0 | Positions the contribution as attribution under matched supervision, but the proposed calibration mechanism remains untested. |
| Reproducibility | 10% | 5.5 | Artifact availability is unchanged; the reported v2 intervals are transcribed, not reconstructed from predictions. |
| Clarity and narrative | 10% | 7.5 | Matched comparisons now lead the result; predecessor mechanisms still require effort from readers. |
| Figures | 5% | 7.5 | Common conceptual figure remains clean, with limits at manuscript print scale and no empirical trajectories. |
| Submission fit | 5% | 8.0 | Honest, substantial abstract suited to an existing research submission; completeness and contribution still need stronger evidence. |

**Weighted total: 68.50 / 100.** Accuracy improves because a specific ambiguity was repaired. This is not evidence that the models or evaluation became stronger.

## Evidence decisions

Use the [reported v2 table and intervals](../proposal/README.md); do not recompute confidence intervals from three extractor means. The [exploratory arithmetic](abstract-20260919-exploratory-summary.json) recomputes ratios from rounded reported errors only. It adds no people, seeds or empirical observations.

The [laterality audit](abstract-laterality-audit-20260919-v01.md) separates recorded motion under imposed errors from the adjacent real-video classification probe. That real-video probe has source-video overlap and tests a different task, so it is excluded from the abstract. The [original study](abstract-original-study-audit-20260919-v01.md) supports the matched-control motivation, not movement preservation.

The [shared pipeline](images/abstract-training-pipeline-20260919-v01.svg) depicts the training design. Readout fitting uses projected training targets; all references are absent from inference. “L/R” denotes signed image-plane ankle separation.


**Precision note from final validation:** direct rounding of the reported errors gives a paired-versus-unchanged reduction range of **35.6–45.2%** at one decimal, or **35.63–45.25%** at two decimals. The abstract above used 35.6–45.3% after intermediate rounding; this note corrects that precision issue while preserving the draft. Revisions 05–07 omit this range and emphasize matched controls.
