# Reproducible results audit

Notebook-anchored execution: `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z`.

Corrected verification execution: `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20261009T060955236580Z`.

The top-level notebook 06 prose reports the September 21 retained full-v1 run, including the duplicated-ankle RF; its seven code cells have no execution outputs. Use notebook_primary_metrics for the notebook-anchored manuscript and corrected_verification_metrics only as explicitly labeled reanalysis. The initial audit's primary_metrics key is retained for compatibility and refers to the corrected latest execution.

All retained scalar metrics and confusion matrices were independently recomputed from OOF rows. Source/clip attribution and complete OOF coverage were checked against the frozen registry; all referenced checkpoint and embedding manifest hashes verified.

## Dataset and estimand

| Class | Raw clips | Usable clips | Sources | Frames | Windows | Padded clips |
|---|---:|---:|---:|---:|---:|---:|
| normal | 26 | 24 | 16 | 1427 | 59 | 5 |
| ms | 30 | 29 | 13 | 3450 | 171 | 0 |
| pd | 35 | 35 | 12 | 2766 | 128 | 10 |

Source IDs come from filename prefixes. Registry asserts source separation, not participant separation. Raw video bytes, unique people and clinical diagnoses are not independently authenticated by this audit.

Source-weighted metrics pool clip confusion contributions with clip weight 1 / usable clips in its source. They retain clip decisions and are not one classification per source. Fold standard deviations use ddof=0 and are descriptive, not standard errors.

## Verified notebook-anchored results

| Model | Source-weighted macro-F1 | Clip-weighted macro-F1 | Source-weighted accuracy | Fold macro-F1 mean ± population SD |
|---|---:|---:|---:|---:|
| sjepa | 0.456624 | 0.397034 | 0.457045 | 0.438068 ± 0.195980 |
| mean_pose | 0.451893 | 0.409840 | 0.454061 | 0.427332 ± 0.132210 |
| rf | 0.411263 | 0.397316 | 0.414170 | 0.395384 ± 0.097960 |
| visibility | 0.317565 | 0.318057 | 0.317707 | 0.284918 ± 0.104069 |
| majority | 0.259398 | 0.202125 | 0.317073 | 0.159394 ± 0.021986 |

## Paired source-resampling sensitivity

Paired class-stratified source bootstrap; each sampled occurrence carries total source weight one; percentile interval.

10,000 resamples; seed 20261004 (the draft's stated settings); class source counts fixed at 16 normal, 13 MS, 12 PD.

| Difference in source-weighted macro-F1 | Observed | 95% percentile interval |
|---|---:|---:|
| sjepa_minus_mean_pose | 0.004731 | [-0.160496, 0.173999] |
| sjepa_minus_rf | 0.045361 | [-0.133447, 0.220114] |
| sjepa_minus_visibility | 0.139060 | [-0.028780, 0.301128] |
| sjepa_minus_majority | 0.197226 | [0.044549, 0.353041] |

Retrospective descriptive sensitivity intervals conditional on fixed OOF predictions, models, split and class source counts. Assumes source clusters independent, although participant independence is unknown and training sets overlap. No retraining, model selection, split or seed uncertainty; not a confirmatory population confidence interval or an equivalence test.

The near-zero S-JEPA versus mean-pose difference does not establish equivalence. These results do not establish that temporal order contributed to classification; a matched time-order ablation remains unrun.

## Execution lineage and corrected RF

Both executions report computed folds and have different checkpoint/embedding bytes and numerically different embedding arrays, with identical S-JEPA hard predictions. Their registry, data, seed and split are the same. This is repeat execution, not independent seed or cohort replication; direct cache reuse is not demonstrated.

All 88 S-JEPA predictions coincide between executions; 12 RF predictions change. Mean-pose, visibility and majority predictions also coincide.

Current feature extraction reproduces all 88 latest RF predictions. Replacing only left_ankle_range with right_ankle_range reproduces all 88 historical RF predictions. Thus the historical versus current RF difference is reproducible from this one feature correction with the saved split and RF settings. This audit does not modify production features or train neural models.

The corrected RF has source-weighted macro-F1 0.325184; the historical duplicated-column baseline has 0.411263. The latter must be identified as historical and should not be substituted for the current corrected result.

The current extractor supplies 82 feature fields, of which 15 vary in each training fold. The remaining fields are constant for this extraction route. Neither the current nor historical RF should be described as a fully populated 82-feature gait model.

## Corrected-execution source aggregation sensitivity

Exploratory hard-label plurality aggregation. Saved OOF rows contain no probabilities, so calibrated or averaged probability aggregation cannot be recovered from those rows alone. The range is sensitivity to all possible tie decisions, not a confidence interval. Fixed label-order tie resolution is arbitrary and should not become a headline result.

| Model | Tied sources | Macro-F1 with fixed normal/MS/PD tie priority | Range over every tie assignment |
|---|---:|---:|---:|
| sjepa | 11 | 0.459222 | [0.363985, 0.562393] |
| mean_pose | 11 | 0.406611 | [0.363645, 0.554335] |
| rf | 7 | 0.343616 | [0.242063, 0.389491] |
| visibility | 12 | 0.235904 | [0.193651, 0.413919] |
| majority | 0 | 0.259398 | [0.259398, 0.259398] |

## Dates and scope

The latest timestamp, 2026-10-09 06:09:55 UTC, is 2026-10-08 23:09:55 in Los Angeles. The apparent next-day folder date is a timezone difference, not a future-dated experiment.

The g1/R1 cache has 47 clips and a different registry, training budget, and readout; its 0.4384 S-JEPA and 0.6666 RF scores cannot be merged with the 88-clip full-v1 evaluation.

This audit reads existing data and model artifacts and refits only the small deterministic RF baselines. It does not establish participant-level generalization, clinically verified diagnoses, disease severity estimation, or external validity.

Reproduction: `.venv/bin/python docs/wacv2007/evidence/results_audit.py`. Exact input hashes, per-class metrics, confusion matrices, fold scores, source-vote tie ranges and manifest checks are in `results-audit.json`.
