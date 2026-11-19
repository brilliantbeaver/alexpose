# Statistical and contribution review of draft03

## Scope and judgment

This is an independent statistical and scientific assessment of the frozen study and the proposed scale-focused revision. I did not author the revised manuscript or the coordinator's `REVIEWER.md`. I have not fitted a new model. The earlier exact-artifact acceptance remains a record of its earlier publication identity; this advisory does not silently extend that acceptance to a changed manuscript.

The paper can lead with its 456 annotated gait sequences and 2,815 forecast windows. Those counts describe substantial observed walking material across heterogeneous public videos. The same opening must retain the 91 source recording groups and the 29 normal recording groups that support the main inference. Changing the order in which these counts are introduced does not require changing the frozen endpoint or its statistical unit.

The strongest positive result is useful posture-conditioned forecasting. The broader contribution is a reproducible benchmark showing which additional comparisons are needed before forecasting error can be interpreted as evidence of abnormal movement organization. The completed experiments do not establish a better anomaly detector or a benefit from the tested structured masks. This is a credible benchmark and validation story with limited algorithmic novelty. A claim of a new state-of-the-art gait anomaly detector would conflict with the evidence.

## Scale, sequence meaning, and the largest-dataset wording

The complete local GAVD inventory contains 1,874 annotation-defined sequences and 458,116 annotation rows from 348 source identifiers. The five requested annotations contain 666 sequences from 103 sources. The modeled cohort retains 456 sequences from 91 source recording groups after the declared availability, reference, observation, and common-target gates. The 1,874-sequence inventory describes the source dataset, rather than the number evaluated by this model.

| Retained GAVD annotation | Annotated sequences | Forecast windows | Source recording groups |
| --- | ---: | ---: | ---: |
| Normal | 141 | 465 | 29 |
| Parkinson's | 37 | 292 | 11 |
| Stroke | 58 | 695 | 16 |
| Cerebral palsy | 52 | 518 | 8 |
| Myopathic | 168 | 845 | 27 |
| Total | 456 | 2,815 | 91 |

The primary GAVD paper describes sequences as uninterrupted walking sections in one camera-relative direction, with at least two gait cycles. That is the annotation contract. It does not establish that every retained section has been visually reverified in this study, that every model prefix contains two cycles, or that different sections are independent participants. The model's prefix is only 1.5 seconds between endpoints and should not inherit the full sequence's cycle guarantee. [GAVD paper, Sections 3.2 and 3.2.1](https://arxiv.org/html/2407.04190).

The separate literature review supports an attributed and scoped sentence: "GAVD was introduced as the largest publicly accessible collection of abnormal-gait videos annotated by expert clinicians." Its inventory includes both normal and abnormal sequences. This does not establish that GAVD is currently the largest gait dataset across videos, force plates, wearables, and motion capture. It also does not establish that this study's retained subset is the largest published gait-anomaly benchmark. Those broader rankings would require explicit scope and comparable evidence. The targeted literature verification is in [literature_scale.md](literature_scale.md).

"456 GAVD-annotated gait sequences" is clearer than "456 independent gait samples." If the word "unique" is used, it should refer to unique sequence identifiers. It must not imply 456 non-overlapping physical walking episodes, because the audited annotations include shared frames.

## Straight sections do not establish statistical independence

Several observations from one walking person can be separate sequences while still sharing body proportions, movement habits, diagnosis, assistive devices, camera placement, background, pose-estimation failures, and recording conditions. Removing turns can improve the consistency of the task. It does not remove those shared causes of correlated errors. Even perfectly straight, temporally separated sections from the same person would not automatically become independent evidence about generalization to new people.

The data audit provides direct evidence beyond this general argument. Seventy-three of the 91 sources contribute multiple retained sequences. A normal source can contribute as many as 30 sequences. Seventeen pairs of retained sequence IDs share raw annotation frames within their source, covering 34 sequence IDs across three sources. The retained forecast windows have 4,427 overlapping pairs, including 155 pairs across different sequence IDs. These counts describe exact source/frame overlap, rather than proving participant identity. All related sequences and windows nevertheless remain in their original source folds, so these overlaps do not cross the declared outer-fold boundary. See [sequence_audit.json](sequence_audit.json).

The audit also finds literal front/right-side annotation transitions in three retained sequences and mixed view labels in three model windows. A view-label change is not a verified physical turn. However, the available evidence cannot support the stronger assertion that every retained model observation has been visually confirmed to be straight and turn-free. The correct wording describes the intended annotation boundaries and the extent of actual validation separately.

A sequence-level train/test split that lets other sequences from the same source appear in training would answer a different, easier generalization question. It could measure prediction within a familiar recording context. It would not replace the recording-held-out question used in the current paper. A separately declared within-recording analysis can be informative, but its uncertainty, information sharing, and claim boundary must remain explicit.

The current source recording group is itself a grouping proxy. Unknown repeated people across uploads remain possible. Therefore neither "91 independent patients" nor "456 independent participants" is supported. There is no justified numerical conversion from these counts into an effective patient sample size.

## The units that belong beside each result

| Quantity | Correct observational or statistical unit |
| --- | --- |
| Dataset coverage | The 456 annotated sequences and 2,815 windows describe the amount and variety of observed walking. |
| Coordinate prediction error | Observed coordinate discrepancies are averaged within leg, then legs, windows within sequence, and sequences within source recording group. Units are squared prefix-torso units. |
| Main normal forecasting result | The final mean gives equal weight to 29 normal recording groups. These groups contain 141 annotated sequences and 465 windows. |
| Main internal anomaly ranking | Scores rank 62 abnormal and 29 normal recording groups. The primary AUROC compares only within-fold pairs and weights folds by their pair counts. |
| Main forecasting uncertainty | The paired bootstrap resamples the 29 normal recording components, conditional on saved fits. |
| Main anomaly uncertainty | The paired bootstrap resamples recording components within the original fold and binary-label strata. |
| Normal empirical calibration | Each component uses seven or eight disjoint normal source recording groups. Sequences and windows do not increase that calibration count. |
| Frozen external transfer | The evaluation unit is one of 90 dataset-coded participants, after windows are averaged within recording and recordings within participant. |
| Clinical generalization | Participant-disjoint, site-disjoint, or prospective inference would require additional verified identities and study design. |

The five outer folds contain only five or six held-out normal recording groups each. Their training sets overlap. The bootstrap intervals therefore describe uncertainty conditional on these saved fits and the recording-group sampling assumptions. They omit full model-development and retraining variability and unknown participant links. Optimization seeds and alternative partitions add sensitivity information but do not add people.

A sequence-weighted sensitivity can address the user's preferred observational unit without changing the primary result. Average each sequence's window errors, weight all 141 normal sequences equally, and preserve all 29 recording components in paired cluster resampling. Larger recordings then receive more total weight, which changes the target population. It does not establish independence. This analysis must have its own post hoc identity, preserve all original primary results, and carry every selected component's sequences and bootstrap multiplicity together. The bootstrap's total sequence count can vary between draws because components contain different numbers of sequences.

That sensitivity has now been completed as `reference-v2-sequence-scale-posthoc-v2` and independently reconstructed. Equal weighting of the 141 normal sequences gives posture MSE 0.08084 versus persistence MSE 0.13850. The absolute improvement is 0.05766 [0.04178, 0.07967], or 41.6% [35.9%, 46.9%]. Uncertainty still resamples the same 29 recording components within the original five folds. The reviewer explicitly expanded every sampled component into all of its actual sequence rows for each of 5,000 draws; selected sequence counts ranged from 61 to 240. Every point and both paired intervals reproduce within `1e-12`.

The forecasting direction therefore agrees under equal-recording and equal-sequence weighting. This is a sensitivity on the same predictions, rather than independent cohort replication or a replacement for the primary 40.8% result. The first attempt stopped before estimation because pandas retained a column-axis label on the pivoted metadata. Its exact source and failure are preserved. The reviewed successor clears that metadata label without changing the estimator. See [the pre-computation release](sequence_sensitivity_review.json) and [the independent result review](sequence_sensitivity_result_review.json).

## What the intervals support

The compact posture transition has normal-source MSE 0.07174, compared with 0.12126 for persistence. The relative reduction is 40.8%, with a saved-fit conditional 95% interval of 33.3% to 47.4%. The paired absolute improvement is 0.04952 [0.03447, 0.06628]. The positive interval is meaningful evidence of forecasting capability on this population. It should be reported with its paired comparison and grouping, rather than presented as an unqualified confirmatory or clinical result.

The two untuned post hoc priors sharpen that interpretation. Mean future pose has MSE 0.10898, and anchor plus mean displacement has MSE 0.12235. Current posture improves on the mean-pose prior by 0.03724 [0.02539, 0.05150]. Thus the tested constant normal pose does not explain the full observed forecasting gain. A posture-dependent mapping toward typical later articulation remains compatible with the result. This comparison is explicitly post hoc and does not establish that detailed temporal organization or a physiological state has been recovered.

Adding the recent-velocity state gives MSE 0.07693, compared with 0.07174 for posture alone. The paired reference-minus-candidate difference is -0.00519 [-0.01022, -0.00046]. Bilateral history improves on own history by only 0.00111 [-0.00267, 0.00529]. These controls prevent the positive forecasting result from becoming a claim that history or bilateral coordination supplies a reliable incremental advantage.

There is a narrower favorable timing comparison: bilateral history improves on the separately fitted window-specific shifted-history control by 0.00591 [0.00147, 0.01110]. This supports the usefulness of the observed alignment relative to that particular perturbation. Because own history remains competitive and current posture forecasts more accurately, it does not establish that opposite-leg history adds value beyond all required controls or identify a neurological mechanism.

The primary conditional two-sided detector has within-fold AUROC 0.4624, compared with 0.8106 for low motion. Its paired difference is -0.3482 [-0.5056, -0.1908]. This is evidence against superiority of this locked candidate under the declared internal endpoint. It is not evidence that all movement-organization models are ineffective. The internal candidate's interval also spans chance, so the point AUROC below 0.5 should not be described as a demonstrated below-chance population performance.

Frozen external transfer evaluates 90 coded participants, including 27 normal participants. Its candidate AUROC is 0.5908, compared with 0.7625 for low motion. The paired difference is -0.1717 [-0.3566, 0.0091]. This comparison does not demonstrate improvement. Because the interval includes zero, it does not precisely establish an external population disadvantage either. The external dataset adds a distinct acquisition setting; it does not remove its age, view, native-frame-rate, quality-retention, identity, and source-to-participant calibration differences. The combined abnormal endpoint includes osteoarthritis as well as Parkinson's disease and is not a test of neurological specificity.

Anatomical masking has MSE 0.07962 versus 0.07761 for uniform masking, with paired uniform-minus-anatomical difference -0.00201 [-0.00426, -0.00002]. Its geometry-matched random-group comparison is 0.00042 [-0.00086, 0.00173]. Motion balancing also has a worse point forecast than uniform masking. The experiment does not support the proposed masking improvement. These are exploratory contrasts in one model and objective, without a confirmatory multiplicity-adjusted claim. There is no need to invent a p-value or a significance label to explain them.

Historical GAVD outcome exposure remains material to every internal result, including a newly named analysis. Normal-only hyperparameter selection is a strength, but it cannot reverse earlier design exposure. The paper should retain both the positive capability and these limits in the same argument.

## Likely workshop reading and paper length

My qualitative assessment is that the work has a defensible workshop contribution as a movement-observation benchmark and validation study. The clearest value is the controlled separation of useful forecasting from incremental abnormality ranking, supported by simple alternatives, a full candidate registry, independent reconstruction, and frozen external transfer. These are relevant to the workshop's stated interests in dynamics, uncertainty, validation, and benchmarks. Short observation forecasts do not establish disease trajectories or intervention simulation. [Official Med-WM call](https://med-wm.github.io/aaai27-mwm/).

The main reviewer concern is likely to be the scientific advance, rather than whether there are enough correlated windows. Normal-only skeletal forecasting and structured masking are established ideas. A compact predictor beating persistence is useful but is not, by itself, a strong new algorithmic result. Full-paper reviewers may ask what a broader research community can now test or understand that it could not test before, whether the benchmark transfers reliably, and whether its strongest simple references are sufficient. Increasing model size or emphasizing the parent dataset's ranking does not answer those questions.

I would favor a focused short benchmark/validation paper for the completed evidence, with the full protocol, registry, and additional figures in supplementary material. The official call currently permits two to four body pages for short papers and six to eight for full papers, excluding references and appendices. A full paper remains defensible if the reusable benchmark, the failure mechanism, and the external comparison form a clear substantive contribution that needs the space. A seven-page body fits that format, but format eligibility alone does not settle novelty. This is a qualitative judgment about framing, not an acceptance probability or a recommendation to conceal the unsuccessful hypotheses.

## Improvements with the clearest scientific payoff

1. Lead with the actual sequence coverage, publish the sequence-to-source and fold membership tables, and retain the recording-group labels beside every main endpoint. Explain annotation intent, actual frame overlap, and verified versus unverified straightness together. This corrects the scale story without changing a result.
2. Make the benchmark reusable through its permitted manifests, exact masks, target support, baseline definitions, source scores, and executable minimal example. State one concrete ambiguity that this resource resolves and compare that contribution with the closest gait-anomaly benchmark. A reusable evaluation advance is stronger than an unsupported "first" or "largest" claim.
3. For a new experimental iteration, prioritize additional independent normal participants and reliable calibration coverage. All 32 annotated normal GAVD sources are already in scope, and 29 survive the gates. Processing other GAVD conditions or more overlapping windows cannot solve this bottleneck. Freeze the new cohort, eligibility, score, and stopping rules before its outcomes are inspected.
4. If strengthening the forecasting claim, add one particularly strong simple competitor under identical support and grouped selection, such as normal-only nearest-posture retrieval or robust recent-velocity extrapolation. The current latest-two-point velocity reference is sensitive to pose noise. Any additional comparison must retain its new identity and post hoc status on reused data; it should test a competing explanation rather than search for a favorable number.
5. For stronger medical claims, obtain verified participant grouping, quality-controlled observations, acquisition-matched normal references, and independent clinical validation. Full-pipeline retraining uncertainty can be added when feasible. Neither more seeds nor a new random split creates an untouched cohort.

These priorities address sample support, alternative explanations, and community value. None requires renaming a source-grouped interval as sequence-level significance or treating an unexecuted extension as a positive result.

## Suggested manuscript language

"Our benchmark evaluates 456 GAVD-annotated gait sequences, represented by 2,815 forecast windows from 91 source recordings. All related sequences from a recording stay in one evaluation group. The normal forecasting and calibration population contains 141 sequences from 29 recording groups."

"GAVD was introduced as the largest publicly accessible collection of abnormal-gait videos annotated by expert clinicians. Its complete inventory contains 1,874 normal and abnormal sequences. Our retained cohort reflects the stated media, observation-quality, and shared-target requirements."

"A gait sequence describes a section of observed walking. Several sequences can share a person, recording context, or source frames, so sequence count describes observational coverage while recording groups define our main evaluation and uncertainty calculations."

"The current-posture model reduced normal-recording forecast error by 40.8% relative to persistence, with a paired conditional 95% interval of 33.3% to 47.4%. The strongest supported contribution is useful short-horizon prediction and a reproducible evaluation showing why that capability does not, on its own, establish improved gait anomaly detection."

The accompanying [statistical_review.json](statistical_review.json) records the exact frozen inputs, verified counts, numerical checks, and scope of this advisory. A revised manuscript needs its own final artifact review.
