# Experiment 0 repair: a calibrated, complete development STOP

The repaired comparison is complete. **Past skeleton coordinates and confidence
did not improve this predictor's held-source prediction of contextual teacher
features beyond RGB, nuisance inputs and the validity-only comparison.** The
measured increment was **−0.00024242 R²**, with a 95% paired source-bootstrap
interval of **[−0.00147402, +0.00091980]**. Only **36.35%** of paired draws had a
positive increment, below the required 90%. This is a valid STOP for the specified
predictor, target and data regime, rather than an incomplete execution.

The run is
[`future-innovation-direct-v3-dev-20260911`](../../../outputs/future-innovation-direct-v3-dev-20260911/reports/gate-report.md).
Its [decision JSON](../../../outputs/future-innovation-direct-v3-dev-20260911/reports/gate-decision.json),
[numerical verification](../../../outputs/future-innovation-direct-v3-dev-20260911/reports/numerical-verification.json)
and [frozen protocol](../../../outputs/future-innovation-direct-v3-dev-20260911/config/frozen-protocol.md)
are the authoritative measured record. The [protocol document](direct-v3-repair-protocol.md)
was written before the real comparison. Calibration and initialization have the
same production fingerprint as every fitted fold and the final report:
`cd7567c95c7a4337ada8ec47441a19f299f415ace1ee0380a8e8d5f1f9f00023`.
No scientific setting was changed after viewing the repaired held-out scores.

The original `gate-v2` / `direct-v2` STOP remains intact. Before editing, the
forensic evaluator reproduced all 60 historical fitted predictions and 8,000
bootstrap rows. It did so again after implementation. Its maximum prediction
difference remains approximately 5.13e-6 under the local Torch runtime. The
historical ridge R² of 0.271699, real-head R² of 0.028456, and matched increment
of −0.00006651 agree with the saved investigation. Weight removal and
zero-initialized refits remain post hoc diagnostics, not evidence from a new
nested-validation comparison.

## What was repaired and how it was checked

The trace is now CLI/notebook or Slurm wrapper → frozen child configuration →
verified parent cache → unchanged source splits → raw controls → training-only
input/target preprocessing → inner penalty selection → typed saved model →
reloaded held-out predictions → source-balanced scores and paired uncertainty →
numerical verification → sealed decision. All new outputs and execution
provenance belong to the child. The original cache's embedded binding still
refers to the original parent contract.

| Confirmed cause and visible effect | Repair | Verification evidence |
|---|---|---|
| `TrainingScaler.fit` divided constant missingness columns by 1e-8; ordinary validation changes became 1e8 in outer/inner 1/1 and 3/0 | Separate `SupportedInput` learns observation support, masks, weighted imputation/variance and unit-aware scale floors inside training sources; unsupported columns transform to zero | The same 19 and 20 affected columns reproduce the old explosion and become exact zero. Across all revised held-out partitions, maximum absolute X transform is 31.37, not 1e8 |
| The second RGB branch had 609,792 weights and thousands of unsupported directions; its effect dominated outer prediction loss | A float64, jointly regularized linear solution replaces the residual head; no arbitrary second RGB branch exists | Disabled skeleton reproduces weighted ridge; selected coefficients are recomputed from the declared training data and penalties during verification |
| Fitting ridge's own in-sample residuals from RGB reversed useful regularization | Fit Y directly with separate positive RGB and skeleton penalties and an unpenalized intercept | No residual targets or sequential correction training are used; selected RGB penalties and baseline identities are checked |
| Every historical correction candidate lost to ridge but one had to win | Every arm receives the exact shared RGB reference as a typed `baseline_only` candidate | Inferior/tied candidates select baseline; saving/reloading returns exact zero prediction difference. Real data selected baseline in 14/20 fold-arm fits |
| Input and target scaling shared a historical class | Keep historical `TrainingScaler` unchanged; use separate new input and target types | Missing/near-constant input fixtures, schema/weight checks, serialization and held-out-statistic immutability pass. Target variation survives input masking |
| Neural training loss was recorded before an update, validation afterward | Label those historical API timing states explicitly; joint-model inner losses use the same fitted state | Historical checkpoint predictions still reproduce. New inner train/validation records are closed-form fit losses; outer-train and inner-validation summaries are labeled as different fits |

The [joint-model module](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_joint_models.py)
contains scaling, temporal summaries and the solve. The
[nested fitting module](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_joint_training.py)
contains source isolation, candidate selection, typed reload and training-statistic
verification. [Cache inheritance](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_cache_reuse.py)
and [numerical reporting](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_joint_reporting.py)
keep lineage and arithmetic checks separate from file presence.

The old convolutions had a five-frame receptive field followed by global
averaging. Their ability to use the intended temporal order was an additional
measurement concern, not the confirmed dominant numerical failure. The repaired
representation uses four ordered eight-frame bins with valid coordinates,
adjacent-frame velocities, confidence and support fractions. It has 924 nominal
columns. Velocities require valid adjacent endpoints; imputation cannot invent
movement. Every raw control passes through the same constructor. Shuffle and
mismatch alter observation support as well as coordinates, which limits a purely
motion-specific interpretation of these contrasts.

## Calibration preceded the real fits

The [fixed calibration implementation](../../../src/gavd6_sjepa/research_directions/future_innovation/fi_joint_calibration.py)
uses the full penalty grid, five outer and three inner source folds, and all four
arms. The statistical fixtures contain 80 generated sources with two clips each,
16 competing RGB inputs, and explicitly planted or absent skeleton information.
These are software positive/negative controls, not physiological simulations or
real-data evidence. They demonstrate sensitivity to the specified strong ordered
signal; they do not establish power for every subtle real motion pattern.

| Prespecified fixture | Required result | Observed result |
|---|---|---|
| RGB-only signal; irrelevant skeletons; seeds 1101–1108 | Mean real gain and mean matched increment each ≤0.03 R² | Both means −0.00015755; one finite draw had a small positive gain |
| Planted skeleton signal, seed 2201 | Real gain and matched increment each >0.10 | Both +0.794774 |
| Planted temporal signal, seed 3301 | Real gain and matched increment each >0.10; real minus designated shuffle >0.10 | Gains +0.728572; real minus shuffle +0.741838 |
| Constant, entirely missing and near-constant inputs | Safe transform, stable schema and no held-out influence | Passed, including exact reproduction of old 1e8 behavior |
| Perfect reference / zero-error candidate, inferior additions and failed candidate | Exact fallback; failures visible and completeness enforced | Passed through selection and reload; an injected inner numerical failure produced incomplete measurement |

All fixture scores, selections and checks are retained in
[`calibration-final.json`](../../../work/artifacts/future-innovation-repair-2026-09-11/calibration-final.json)
and copied into the child's immutable configuration. The preliminary calibration
and final run used identical choices and reproduced the same synthetic outcomes.
Synthetic results cannot authorize scientific ADVANCE.

## Measured development comparison

The real comparison reused all 50 clips from 43 source videos and the parent fold
labels. Outer-training partitions contain 39–41 clips. Teacher encoding,
projection, nuisance features, pose normalization, target and control definitions
were reused unchanged. The new RGB reference is selected under the repaired
preprocessing and finite penalty grid, so its improvement over the *historical*
ridge is not an estimate of skeleton benefit.

| Predictor | R² | Gain over the new shared RGB reference |
|---|---:|---:|
| RGB + nuisance reference | 0.334215 | Reference |
| Real skeleton | 0.333973 | −0.00024242 |
| Four-frame block shuffle | 0.334247 | +0.00003150 |
| Different-source clip mismatch | 0.335453 | +0.00123743 |
| No skeleton, validity retained | 0.334215 | 0.00000000 |

Every outer fold selected RGB penalty 100. The six selected joint models also
used RGB penalty 100, with skeleton penalties 1,000 or 10,000. Thus the observed
joint winners did not choose a different RGB penalty from their shared reference,
although the family allowed that possibility prospectively. Real selected joint
models in folds 0 and 3; shuffle in fold 0; mismatch in folds 2, 3 and 4.
No-skeleton selected baseline in all five folds. Its full predictions therefore
equal RGB exactly, making the real-minus-no-skeleton increment equal to the
real-minus-RGB gain in this run.

All 740 pooled arm candidates were valid: 720 joint candidates and 20 explicit
baselines. Their 2,220 inner loss records remain visible, with a separate shared
RGB search. There were no numerical-fit failures. The nominal X block has 2,382
features; 2,380 are supported in each outer fit. Supported skeleton features vary
by arm and fold: real 822–862, shuffle 830–890, and no-skeleton 162–202. Equal
nominal dimensions do not establish equal statistical flexibility. The full
[selected-model table](../../../work/artifacts/future-innovation-repair-2026-09-11/selected-models.csv)
and [scaling diagnostics](../../../work/artifacts/future-innovation-repair-2026-09-11/scaling-diagnostics.json)
retain this detail. Revised X maxima in the two previously unstable inner
partitions are 24.42 and 15.03; their unsupported missingness columns are zero.
The largest held-out skeleton transform is 72.63, retained and reported rather
than used to exclude a clip.

![Repaired gains, selected model types and missingness scaling](../../../work/artifacts/future-innovation-repair-2026-09-11/repaired-development-comparison.png)

The real gain fails the +0.05 threshold, the real-versus-shuffle rule fails, and
the matched mean increment is negative. Mismatch remains below the +0.01 ceiling.
Both required positive-draw fractions are 36.35%. All measurement-validity checks
pass, but the point criteria already fail, so STOP is the correct classification.
The interval crossing zero expresses uncertainty about a small increment; it
does not turn failed point criteria into INCONCLUSIVE.

The model is deterministic. Numeric seed 0 identifies its prediction rows; it is
not an optimization replicate. The prospective protocol made stochastic
three-seed stability inapplicable and retained every effect/control and
source-bootstrap threshold. The 95% interval is conditional on the saved models;
it omits repeated fitting, model selection and adaptive redesign. The 2,000 draws
add no independent sources. Source separation also does not prove participant
separation across recordings.

## Verification, execution and preservation

Independent reconstruction checked **51,200** held-out rows, all **20** typed
selected artifacts and **8,000** paired bootstrap rows. All **256** target
coordinates survive the intersection of training-variance masks. Verification
checks source/window identities, target units and inverse transforms, shared RGB
predictions, selected penalties/types, fitted statistics, control donors,
featurewise scores, aggregate scores, intervals, positive fractions and the final
decision. It rejects altered arithmetic even when the corresponding checksum is
updated. Float64 prediction tolerance is 1e-10 absolute/relative; score tolerance
is 1e-12 absolute and 1e-10 relative. The historical Torch tolerance is not reused
as a blanket tolerance for the repaired float64 models.

| Actual local command/check | Result |
|---|---|
| `.venv/bin/python -m unittest discover -s tests -p 'test_future_innovation_*.py'` | 125 tests run: 123 passed, 2 optional official-V-JEPA-source checks skipped; [log](../../../work/artifacts/future-innovation-repair-2026-09-11/all-future-innovation-tests-hardened.log) |
| `gavd6 ... calibrate-repair --output .../calibration-final.json`, through `.venv/bin/python -m gavd6_sjepa.command_line_interface` | All frozen exact/statistical calibration checks passed; [log](../../../work/artifacts/future-innovation-repair-2026-09-11/calibration-final.log) |
| `init-cached-run`, `run-gate --device cpu`, `score-gate`, `build-report` with the child root | Real cache imported read-only, five folds fitted and complete STOP sealed; [fit log](../../../work/artifacts/future-innovation-repair-2026-09-11/real-fit.log) |
| `verify-repair --run-root outputs/future-innovation-direct-v3-dev-20260911 --output work/artifacts/future-innovation-repair-2026-09-11/independent-real-verification.json` | Independent numerical reconstruction passed; [record](../../../work/artifacts/future-innovation-repair-2026-09-11/independent-real-verification.json) |
| Canonical generator `--check` and Python `compileall` on experiment modules/scripts | Passed |
| `verify_future_innovation_tutorials.py --pipeline-smoke --output-parent work/artifacts/future-innovation-repair-2026-09-11/notebook-integration` | All teaching notebooks and both historical protocols passed execution, incomplete-report failure, verified audit rejection and sealed resume; [record](../../../work/artifacts/future-innovation-repair-2026-09-11/notebook-integration/verification-ec0vt8id/verification.json) |
| `execute_future_innovation_notebook.py --notebook 00` through `04`, `--mode execute --device cpu`, child root and child `notebook_runs/local-cached` output directory | All five passed, verifying/reusing the completed real run; [log](../../../work/artifacts/future-innovation-repair-2026-09-11/real-notebook-execution.log) |
| Bash syntax and mocked Slurm command/dependency checks | Original paths and both cached CPU paths passed; no GPU requested by cached submissions |
| Historical forensic evaluator before and after repair | All 60 old fits and 8,000 bootstrap rows reproduced; [after-repair record](../../../work/artifacts/future-innovation-repair-2026-09-11/parent-recomputation-after-repair.json) |

The [execution guide](../../../slurm/future-innovation/README.md#calibrated-cached-development-repair-direct-v3)
gives complete copyable local and HAIC commands, exact output locations, and
resumption behavior. [Notebook execution](../../../slurm/future-innovation/NOTEBOOKS.md#cached-direct-v3-notebooks)
uses the same CLI functions and stage locks. Local tests exercised interrupted
initialization, interrupted folds, completed reuse, duplicate writers, changed
parent artifacts and invalid selected checkpoint types. Inspect mode stays
read-only, with file presence distinct from report integrity and saved numerical
evidence. All five [executed real notebooks](../../../outputs/future-innovation-direct-v3-dev-20260911/notebook_runs/local-cached/00_question_and_worked_example.ipynb)
and both [generator snapshots](../../../outputs/future-innovation-direct-v3-dev-20260911/notebook_runs/source-snapshots/generator-direct-v3.py)
are retained inside the child.

Local computation used Python 3.12.10, NumPy 2.5.2, SciPy 1.18.0, scikit-learn
1.9.0 and Torch 2.13.0 on macOS. The full software record is in
[`runtime-contract.json`](../../../outputs/future-innovation-direct-v3-dev-20260911/config/runtime-contract.json).
Actual HAIC jobs were not submitted. The optional official-source tests require
`FI_TEST_VJEPA_ROOT`, which was not configured. No pretrained teacher inference
or fresh raw-video audit is claimed; the original teacher evidence is explicitly
reused. These limitations do not block this cached CPU measurement. Jupyter
initially failed because the sandbox prohibited local sockets; approved execution
outside that sandbox passed. An ad hoc notebook import wrapper also failed before
execution; the documented notebook CLI then ran all five successfully.

The final [preservation check](../../../work/artifacts/future-innovation-repair-2026-09-11/completion-verification.json)
compares SHA-256, size, modification time and inventory against the
[before snapshot](../../../work/artifacts/future-innovation-repair-2026-09-11/parent-snapshot-before.json)
and [after snapshot](../../../work/artifacts/future-innovation-repair-2026-09-11/parent-snapshot-after.json).
All 337 parent-run files and all 10 files in the historical notebook directory
(including its nine executed notebooks) are unchanged. A real-run snapshot also
proved that final numerical reconstruction made no changes inside the child.
Existing user investigation files and unrelated documentation edits were retained.

## Post-fit recovery and verification hardening

Final review found that a missing child cache or readiness record could fall
through to the old teacher-setup path. The CLI now dispatches direct-v3 directly
to inherited-evidence verification, so missing evidence raises an error without
constructing a teacher. A regression test removes each required record and proves
that the teacher factory is never called. The model verifier also checks the
stored solver's penalty metadata against its selected specification.

These changes do not alter prediction, preprocessing, search or decision rules.
They are recovery/verification hardening, not a scientific amendment. The sealed
models, predictions and report are retained exactly as measured. The subsequent
[verification record](../../../work/artifacts/future-innovation-repair-2026-09-11/post-fit-hardening-verification.json)
records the current verification fingerprint separately from the fit-time
fingerprint and reproduces the same STOP and increment without writing inside
the run. The full 125-test suite passed after this change. Exact fingerprinted
[fit-time sources](../../../outputs/future-innovation-direct-v3-dev-20260911/source_snapshots/fitting-code-contract.json)
were archived by reversing only these two edits; their fingerprint matches the
calibration, initialization, all fitted folds and sealed report.

## Critical interpretation and next work

The repaired result is not generated by an unrestricted RGB residual refit:
there is one penalized joint solve, and the reference is shared. Unsupported
columns cannot affect predictions. Saved input/target statistics are reconstructed
from their training partitions, and the target mask never comes from input
support. All arms receive identical candidate opportunities; donors stay inside
their partitions and use context metadata only. One deterministic fit is scored
once. Model selection is recorded before outer prediction, and no held-out result
changed the chosen feature schema, penalty grid or rule. A selected addition was
allowed to lose on outer sources, which happened here. The pipeline therefore
supports a correct negative outcome without forcing arbitrary corrections.

The initial root-cause note's emphasis on zero initialization was superseded by
the saved-model evaluation. Legacy teacher-selectivity prerequisites remain
historical and were not combined with direct-v2/direct-v3 rules. Neither
synthetic calibration nor post hoc weight removal supplies scientific evidence
for distillation.

**Stop this specified comparison.** The data do not support proceeding to full
S-JEPA training or distillation on the strength of this target/predictor result.
If further research is justified, define a separate target, representation or
larger-source study with a specific hypothesis and fresh calibration. Keep this
inspected cohort as development data. A promising redesign would need a frozen
independent-source confirmation before scaling. A later JEPA comparison should
separate raw skeletons, pretrained features and matched initial/random encoders;
a skeleton-only student's intended target must itself be shown predictable from
skeletons. Those experiments require their own evidence and implementation.
