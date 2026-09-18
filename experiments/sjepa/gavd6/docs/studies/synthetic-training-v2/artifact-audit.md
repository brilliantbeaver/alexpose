# Retained evidence audit — 18 September 2026

The completed pilot means reproduce from CSVs. The strongest achieved selector in the frozen panel is scene/domain; the response selector does not establish useful personalization. These are development results from two validation estimators, not confirmation of gait restoration or JEPA training.

The new [audit module](../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/audit.py) implements reconstruct_pilot(repo_root, output_dir=None). A fresh output folder receives pilot-audit.json and pilot-table.csv; historical notebook bundles are protected. The JSON retains input hashes, counts, selected policies, numerical discrepancies, the oracle action universe and limitations. This audit performs no fitting or model inference.

## Exact source and arithmetic

The four outcomes.csv files at notebook_runs/synthetic-training/run-02-v1/source/ contain 576 rows each: **2,304 unique rows**. The key is (student_id, domain_id, budget, action); family and split are also checked for consistency. Each student has 24 scene conditions, original/probe outcomes at budget zero, and eleven actions at each of budgets 25 and 75. No blank or nonfinite source errors were observed.

The selectors/validation_predictions.csv file in run-03-v1 contains **6,912 decisions**, uniquely identified by (view, kind, parameter, budget, student_id, domain_id). Every one of the **144 configurations** has all 48 validation student/condition pairs. The validation_summary.csv file contains one row per configuration. The two validation students are HRNet-W48 and RTMPose-S; fitting students are HRNet-W32 and RTMPose-M. All four trial records name seed 17.

Each decision joins its source error on the source key above. The largest discrepancy is **9.71445146547012e-17**, consistent with cross-file floating-point serialization; the largest summary-mean discrepancy is **3.469446951953614e-18**. “Exactly matched” should mean matched within declared numerical tolerance, not byte-identical decimals. The audit uses absolute tolerance 1e-12 for cross-file equality, then source-backed errors for gains/ties. Direct comparisons across rounded files spuriously count one exact tie as a tiny win.

The selection.json file freezes **75 subsequent updates**; full and matched source-progress both use nearest-neighbor k=1, domain uses k=3. The matched control is reconstructed from source_progress rows because separate source_progress_matched prediction rows are not stored. Budget and selector settings were chosen using this validation panel, so these are selected development errors. Fixed front was chosen on training outcomes ([workflow](../../../src/gavd6_sjepa/research_directions/synthetic_training/workflow.py)).

| Policy | Mean normalized visible-landmark error |
|---|---:|
| Replay after common probe | 0.027320536651784428 |
| Full-budget replay from original | 0.027314849615335837 |
| Pooled synthetic | 0.027036717405310930 |
| Full response selector | 0.027026479998541170 |
| Matched source-progress | 0.027025762386308140 |
| Fixed front | 0.026899156973405572 |
| Scene/domain | 0.026780739842100870 |
| Shared-scene retrospective oracle | 0.026675882342393962 |
| Student-specific retrospective oracle | 0.026665059463932696 |

The [metric implementation](../../../src/gavd6_sjepa/research_directions/synthetic_training/measurements.py) computes Euclidean error for 12 visible COCO body landmarks, divided by the supplied reference person-box diagonal. A missing prediction receives a penalty of one diagonal. It averages visible joints within frame, frames within person, then people; the CSV table above equally averages 48 student/scene outcomes. Units are dimensionless box-diagonal fractions, not pixels, meters, 3D error or clinical severity. Missing-reference frames are unsupported rather than zero error. Per-frame predictions are absent here, so this session reconstructs aggregate arithmetic and traces metric code; it cannot regenerate each original metric.

All four trial.json files retain **10 probe-loss entries**, with batch 20 comprising two synthetic and 18 real examples. Current configuration and branch code corroborate ten common probe updates. Thus the 75-update branch includes ten common plus 75 subsequent optimizer steps; full replay uses 85 from the original checkpoint. The plan's “20 common-probe” figure corresponds to 20 synthetic examples drawn across ten steps, not 20 optimizer updates. This audit records the distinction without rewriting historical artifacts.

## Oracle and decision diagnostics

The action set is the eight saved lessons plus replay. Pooled synthetic and full-budget replay are comparators, not selectable oracle actions in the historical selector ([workflow](../../../src/gavd6_sjepa/research_directions/synthetic_training/workflow.py)).

For each condition, the shared oracle minimizes mean error across the two validation students. The student-specific oracle independently minimizes each student/condition row. The absolute gap is **0.000010822878461266144**, or **0.0405717731%** relative to the shared oracle. Shared choices account for **95.3767648%** of oracle opportunity beyond fixed front. These use evaluation outcomes unavailable at deployment. They are retrospective diagnostics of this small library/panel, not achieved policies, population upper bounds or proof that personalization is always unhelpful.

Full changes 3/48 actions relative to matched source-progress: one helps, one harms and one changes the lesson with unchanged error. Source-backed full decisions improve replay in **37**, harm it in **10**, and tie in **one**; replay is never selected. Mean error falls **1.07632093%** versus matched replay. RTMPose-S side_low_occluded_blur and side_low_clear_blur contribute **74.3055103%** of full's net disadvantage to domain.

## Independence, exposure and unavailable artifacts

The 48 validation rows repeat two estimator checkpoints and 24 scene settings; they are not 48 independent training experiments. Preparation code uses two motion clips per context/reference role across all 24 conditions, and the executed notebook reports 128 frames per role/condition with 64-frame clips. The full source manifest is absent, so this session cannot certify realized person IDs or compute person uncertainty. Reference variants and scores from the same adapted checkpoint are correlated.

The retained run-02-v1 bundle contains **zero NPZ files and zero PT files**. The original outputs/synthetic-training/pilot-01 directory is absent locally. Missing objects include per-frame predictions, context/response feature caches, rendered manifests and adapted-head checkpoints. This audit cannot measure feature scales/distances, gradient norms, parameter changes, prediction displacement, teacher collapse or alternative hyperparameters. Current code freezes the video encoder and deletes its predictor; no S-JEPA is trained in that workflow. Current response summaries average across frames. Those code facts do not establish why a specific architecture adapted weakly.

The AMASS split registry at manifests/amass/amass_subject_splits.csv has **201 identities: 156 training, 26 validation, 19 test**, SHA-256 ec1fbec0b478e509e10a458c7e9532fabaa5b9043575e0fc4bafd81c298fcef7. The loader requires approved, nonexcluded subjects and preserves the registry split. Inventory eligibility means a readable supported motion, not verified walking. The preparation preview includes a pose-03-open leg motion.

The scripts/research_directions/source_scaling/known-exposure.csv file has **128 recording IDs**, SHA-256 de1fb80d9d0bb7053879f047e7b0ac233b50d24130c1e424b56433358115475e, matching its JSON ledger. It is explicitly an incomplete historical union. Source-scaling documents record 304 development and 44 reserved confirmation recordings out of 348; the original reservation bundle is absent here. These counts cannot identify fresh available media. Do not promote inspected validation conditions, unknown-exposure recordings or a held-model label such as ViTPose to untouched confirmation. Known aliases/people and protected reservations must enter the new supplied manifest.

## Earlier negative results stay separate

The aggregate at ../gavd5-drift/neurips-laterality/docs/fmts_revisions/review/numerical_evidence.json is available. All eight means were recalculated from seed arrays with zero discrepancy. Initialization expanded-readout R² is **0.22254357355538518**; five trained-teacher means range **0.10077719741431432–0.11420943905203321**. All 25 paired teacher-minus-initial seed differences are negative. The cohort is 625 clips/93 videos and five seeds, not 625 verified people. Raw predictions/checkpoints remain unavailable; intervals were not regenerated. These results alone do not diagnose constant-feature collapse.

The [direct-v3 report](../future-feature-prediction/gate/results.md) retains a development **STOP**: skeleton increment **−0.00024242 R²**, interval **[−0.00147402, +0.00091980]**, 36.35% positive draws, on 50 clips/43 recordings. Its original bundle is absent; this audit inspected the report instead of refitting it.

The [accessibility report copy](../future-feature-prediction/accessibility/evidence/cached-panel-report.json) hashes to the recorded e182537f7864750e3458b37d2f0518f918a33384327d442db4c881aba55914f0. The primary real-minus-no-skeleton estimate is **+0.001993743867998478 R²**, interval **[−0.006672679157896601, +0.01260843146998897]**, 68.35% positive draws: **no_supported_temporal_lead**. Its secondary support-only panel cannot replace that primary. Both experiments are deterministic cached comparisons: seed zero is an artifact identity, not a training replicate; source bootstrap intervals condition on fitted models.

The newer [temporal-gait status](../temporal-gait/results/README.md) records software verification and fixtures, with real GAVD experiments pending. Fixtures are neither real success nor another failed real study. The latent-laterality uniform-control STOP also remains intact; this study does not reopen stopped seeds or sealed tests.

## New scoring and uncertainty contract

[Evaluation](../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/evaluation.py) receives original pixel coordinates, physical timestamps and independent evaluation scales separately from inputs. It reports visible, lower-limb and large-error coordinates; missing predictions; 0.2-second displacement at exactly supported timestamps; signed ankle separation; RMS ankle-separation amplitude; and operational 2D ankle-maximum timing with sufficient full-window support and matching event counts. These maxima are not clinical heel strikes or cadence. Unsupported quantities remain NaN with counts/status. Synthetic all-joint/occluded metrics require explicit synthetic-proxy provenance and remain separate from real-visible scores.

Aggregation balances variants within windows, windows within motions, motions within people, then people. Seeds and extractors remain separate. Paired bootstrap requires identical method keys and resamples people plus motions nested within people, keeping their correlated windows/variants together. It exports draws and matched keys. Unsupported windows cannot disappear silently; mismatched keys raise errors. Intervals condition on fitted models and supplied audited independent-group identities.

Scientific gates require supported coordinate improvement, preservation and clean-retention evidence under an explicit margin. Fixtures, planned stages and missing endpoints return insufficient_evidence. Passing software tests never authorizes empirical spending or confirmation access.
