# Source Learning-Curve Experiment on HAIC

The separate `source-learning-curve-v1` experiment tests whether training on more
recordings increases the contribution of skeleton history beyond RGB and nuisance
inputs. It keeps the repaired predictor and contextual teacher target. The
historical 50-clip experiment retains its completed STOP.

Use this guide for the dedicated source-learning-curve CLI and jobs `20`–`23`.
The [main HAIC guide](README.md) covers the historical gate commands. Read the
[frozen scientific design](../../docs/studies/future-innovation/source-learning-curve-protocol.md)
for the comparison rules and the
[validation record](../../docs/studies/future-innovation/source-learning-curve-validation.md)
for measured evidence and execution limits.

**The saved reservation and synthetic calibration are complete; the expanded
real-data learning curve has not run.** The main path below resumes the retained
`future-innovation-source-curve-dev-20260911-v2` study. To create a separate study,
use [Calibrate and freeze a new study](#calibrate-and-freeze-a-new-study) first.
The directory suffix `-v2` identifies the final implementation attempt; its
scientific protocol is still `source-learning-curve-v1`.

## Prepare the environment and inputs

On HAIC, set explicit persistent paths. Reuse the same study directory to resume.
The wrapper requires absolute `GAVD6_ROOT`, `FI_SCALING_ROOT`, and
`FI_PARENT_ROOT` paths and an already frozen `config/study.json`.

```bash
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export GAVD_FULL_ROOT="$GAVD6_ROOT/data/gavd_full"
export VJEPA2_ROOT="/hai/scratch/$USER/vendor/vjepa2"
export FI_TEACHER_CHECKPOINT="/hai/scratch/$USER/models/vjepa2_1_vitb_dist_vitG_384.pt"
export FI_POSE_MODEL="/hai/scratch/$USER/models/pose_landmarker_lite.task"
export FI_ANNOTATION_ROOT="$GAVD_FULL_ROOT/annotations/GAVD/data"
export FI_VIDEO_ROOT="$GAVD_FULL_ROOT/youtube/all"
export FI_SCALING_ROOT="$GAVD6_ROOT/outputs/future-innovation-source-curve-dev-20260911-v2"
export FI_PARENT_ROOT="$GAVD6_ROOT/outputs/source-curve-parent-snapshot"
export FI_PYTHON="$GAVD6_ROOT/.venv/bin/python"
cd "$GAVD6_ROOT"
```

Use the existing compatible environment. The [environment setup section](README.md#prepare-the-environment-and-inputs)
describes the pinned teacher checkout, checkpoint and HAIC dependencies. The
scaling wrapper does not install them. `FI_PYTHON` selects the interpreter for
both the commands below and the submitted jobs; jobs default to `.venv/bin/python`
when it is unset. The Linux CUDA lock must not replace an existing macOS environment.

Required expanded-data layout:

```text
$GAVD_FULL_ROOT/
  annotations/GAVD/data/GAVD_Clinical_Annotations_1.csv ... _5.csv
  youtube/all/<video_id>.mp4
```

All five official annotation partitions are needed for per-frame alignment.
Use the existing `gavd6 gavd download` workflow to obtain missing source videos.
The frozen study already contains copies of its sequence and video manifests;
their presence does not establish that the media are available or eligible.

Before resuming on HAIC, transfer the frozen study, its matching implementation,
and the exact parent snapshot. `FI_PARENT_ROOT` must name the direct-v2 parent
directory containing `config/`, `manifests/`, and `teacher-cache/`. Locally this
parent is `outputs/future-innovation`. A larger original HAIC run with additional
files is not the same snapshot.

The parent check covers the complete file inventory, contents, sizes and
nanosecond modification times. Use a transfer method that preserves those times,
then check the copy with `status` below before submitting. The supplied
`--parent-root` resolves relocated cache paths while preserving their original
embedded bindings. The study's `implementation/manifest.json` and archived
source files identify the matching implementation. Its frozen code hashes are
enforced; documentation and new test files do not change the fitting fingerprint.

## Inspect the frozen reservation and submit preparation

Check the study and parent before allocating jobs:

```bash
"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py status \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT"
```

`status` verifies frozen inputs, the parent snapshot and the cohort-audit receipt,
then lists stage presence and the next missing stage. It does not verify every
completed stage or reconstruct numerical results. A complete-report file being
present is separate from a successfully verified measurement.

The retained reservation has 348 recordings: 44 reserved for confirmation and
304 assigned to development, before expanded eligibility checks. Previously
inspected recordings remain development data. No participant registry was
available, so these are recording counts, not participant counts. Confirmation
recordings are excluded from development pose processing, teacher encoding,
donor pools, fitting and evaluation.

Preview the full submission without submitting jobs or writing logs:

```bash
bash slurm/future-innovation-scaling/submit-source-learning-curve.sh all --dry-run
```

For an unattended run, submit every stage with its dependencies:

```bash
bash slurm/future-innovation-scaling/submit-source-learning-curve.sh all
```

Alternatively, submit preparation separately:

```bash
bash slurm/future-innovation-scaling/submit-source-learning-curve.sh prepare
```

This submits job `20`, which rebuilds alignment candidates, processes development
poses and records exclusions. It writes `data/manifests/development-windows.csv`
and `data/cohort-complete.json`. The latter is a completion receipt: it records
the artifacts whose integrity later stages must check. Inspect the preparation
logs and eligibility records before continuing with `compute`.

## Cache, audit, fit, and score

Once preparation has completed, submit the remaining stages:

```bash
bash slurm/future-innovation-scaling/submit-source-learning-curve.sh compute
```

The complete dependency chain is:

```text
20 prepare (CPU, four CPUs, 64 GB RAM, 24 h)
  -> 21 cache / audit-teacher / plan (one H100, four CPUs, 96 GB RAM, 24 h)
  -> 22 fit (five CPU array tasks, four CPUs / 64 GB / 24 h per fold)
  -> 23 report / verify (CPU, four CPUs, 64 GB RAM, 12 h)
```

These are requested allocations, not measured runtime estimates. Jobs use the
`mind` account and `hai` partition. All dependencies use `afterok`: the next job
starts only when its predecessor succeeds. Failed dependencies are canceled with
`--kill-on-invalid-dep=yes`. An upstream failure prevents automatic reporting;
it does not produce a scientific STOP. Recover the failed stage before resubmitting
its downstream stages.

Job `21` reuses verified parent windows and encodes additional development
windows with the original teacher projection. It then checks repeatability and
prefix isolation on three preselected new windows and freezes the eligible
source-subset plan. Jobs `22` and `23` use cached CPU arrays without loading the
teacher. Larger fits and retained decoded frames may require substantial time
and storage; full-cohort requirements have not yet been benchmarked.

Five fixed outer folds hold out the same development recordings at every training
size. Within each fold, three source-subset samples provide nested training sets
near 40, 80, 160 and all available sources, subject to eligibility. Every sequence
from a selected recording stays together. The deterministic full-training fit
is shared across the three samples; repeated solver seed labels would add no
evidence about stability.

Preprocessing and penalty selection use only the applicable training sources.
The weighted loss is a sum, with weights totaling the number of training clips
`n`; each penalty is `lambda = n * rho`. Inner validation selects each block's
`rho` from `[0.0025, 0.025, 0.25, 2.5, 25, 250]`. This keeps regularization
comparable as the training set grows. Every arm can select the exact RGB-only
baseline instead of a joint predictor.

Reports compare RGB-only prediction with real history, time shuffle, clip
mismatch and no skeleton, including real-minus-no-skeleton at every size. The
no-skeleton arm retains joint validity, so this increment tests what coordinates
and confidence add beyond that observation information.
Inputs use frames 0–31. The projected target at frames 38–39 was encoded using
the full 64-frame clip, so it remains a contextual teacher target. A growing
matched increment would motivate a separate confirmation experiment; better
absolute scores with a flat increment would favor a target or representation
study. This pipeline does not launch student training.

## Outputs, inspection, and resumption

All new outputs belong to `FI_SCALING_ROOT`; the parent remains read-only.

| Location beneath the study root | Contents |
|---|---|
| `config/` and `implementation/` | Frozen design, source reservation, parent snapshot, calibration and software identity |
| `reports/cohort-audit.json` | Counts, exposure history and initial eligibility evidence |
| `data/` | Prepared inputs, expanded cache, teacher audits and completion receipts |
| `manifests/learning-plan.json` | Fixed evaluation folds, training subsets and unique fit identities |
| `models/` and `predictions/` | Selected models, candidate outcomes, preprocessing/donor audits and predictions |
| `reports/learning-curve.json` and `reports/learning-curve.svg` | Aggregate comparisons, interpretation and curve plot |
| `reports/per-subset.csv`, `reports/raw-errors.csv`, `reports/source-bootstrap.parquet` | Subset scores, raw teacher-unit errors and paired uncertainty |
| `reports/complete.json` | Final report integrity receipt |
| `logs/` | Job stdout/stderr and `submissions.tsv` with job IDs and dependencies |

For quick inspection, rerun `status` or open
[notebook 23](../../23_source_learning_curves.ipynb). For a completed report, check
its numerical results independently:

```bash
"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py verify \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT"
```

`verify` reconstructs selected coefficients and preprocessing, checks predictions,
and recomputes scores and paired uncertainty. It performs CPU refits for checking
but does not write scientific artifacts. It does not refit every rejected
candidate. Bootstrap intervals are conditional on the saved fits and resample
whole recordings; they do not add independent recordings or participants.

Resume an interrupted stage with the same command. Completed artifacts are
checked before reuse; partial windows or fits without receipts are recomputed.
An artifact that disagrees with its receipt is rejected. Stage and fit locks
prevent duplicate writers and release when a process exits.

| Wrapper mode | Stages submitted | Prerequisite |
|---|---|---|
| `all` | `20` → `21` → `22` → `23` | Frozen study and required input resources |
| `prepare` | `20` | Frozen study and required input resources |
| `compute` | `21` → `22` → `23` | Completed preparation |
| `fit` | `22` → `23` | Completed cache, teacher audits and subset plan |
| `report` | `23` | All required fits complete |

For example, resume CPU fitting after cache, audits and planning have completed:

```bash
bash slurm/future-innovation-scaling/submit-source-learning-curve.sh fit
```

Do not run `freeze` again to resume a frozen study or delete receipts to bypass
a mismatch. Complete scientific reports are sealed. A deliberate change to
media, pose processing, teacher, features, target or fitting implementation
requires a new study with matching calibration and a new compatibility assessment.

## Run the cached CPU stages locally

After transferring a completed expanded cache and its audits, set `GAVD6_ROOT`,
`FI_SCALING_ROOT`, `FI_PARENT_ROOT` and `FI_PYTHON` to their local absolute paths,
then change to `GAVD6_ROOT`. The original 50-window cache alone cannot supply an
expanded learning curve. Use the existing local environment:

```bash
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py plan \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT"

"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py run \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT"

"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py report \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT"

"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py verify \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT"
```

`plan` requires valid cache and audit receipts. `run` fits all five folds; add
`--fold 0` through `--fold 4` to run one fold. The dedicated script also supports
the module entry point
`python -m gavd6_sjepa.research_directions.future_innovation_scaling.fi_scaling_cli`.
Use either entry point's subcommand `--help` for individual stage options.

## Calibrate and freeze a new study

Skip this section when resuming the retained study. For a fresh reproduction,
start from the local repository root and choose unused study and calibration
directories. The example below uses new names rather than overwriting the saved
September 11 records. On HAIC, use the HAIC paths from the setup section instead.

```bash
export GAVD6_ROOT="$PWD"
export SJEPA_ROOT="$(dirname "$GAVD6_ROOT")"
export FI_PYTHON="$GAVD6_ROOT/.venv/bin/python"
export FI_PARENT_ROOT="$GAVD6_ROOT/outputs/future-innovation"
export FI_SCALING_ROOT="$GAVD6_ROOT/outputs/future-innovation-source-curve-dev-reproduction-01"
export FI_CALIBRATION_ROOT="$GAVD6_ROOT/work/artifacts/source-learning-curve-reproduction-01/calibration"
export FI_EXPOSURE_ROOT="$GAVD6_ROOT/work/archive/gait-parity-pre-orbit-mask-2026-08-19/real"

"$FI_PYTHON" scripts/research_directions/future_innovation/calibrate_source_learning_curve.py \
  --output-root "$FI_CALIBRATION_ROOT"
```

After calibration passes, freeze the source reservation before expanded processing:

```bash
"$FI_PYTHON" scripts/research_directions/future_innovation/run_source_learning_curve.py freeze \
  --run-root "$FI_SCALING_ROOT" \
  --parent-root "$FI_PARENT_ROOT" \
  --sequence-manifest "$GAVD6_ROOT/manifests/gavd/gavd_full_sequences.csv" \
  --video-manifest "$GAVD6_ROOT/manifests/gavd/gavd_full_videos.csv" \
  --inspected-manifest "$FI_PARENT_ROOT/manifests/gate-windows.csv" \
  --inspected-manifest "$SJEPA_ROOT/gavd5-drift/work/artifacts/gavd_valid_sequences.csv" \
  --inspected-manifest "$FI_EXPOSURE_ROOT/gavd96-cpu-strong-v1/compute/window_manifest.csv" \
  --inspected-manifest "$FI_EXPOSURE_ROOT/gavd96-cpu-strong-v1/exposure/window_manifest.csv" \
  --inspected-manifest "$FI_EXPOSURE_ROOT/validation-only/exposure/window_manifest.csv" \
  --calibration "$FI_CALIBRATION_ROOT/calibration.json"
```

These historical manifests form the documented exposure registry. Add further
`--inspected-manifest` files before freezing if other recordings have been
inspected. An optional `--participant-registry` CSV supplies explicit
`video_id,participant_id` links; blank IDs remain unknown. Changing either
registry after reservation requires a new study and reservation.

Calibration and freezing must use the same implementation fingerprint. The
retained final calibration is under
`work/artifacts/source-learning-curve-20260911/calibration-final-v3/`; the earlier
study directory without `-v2` is preserved as an implementation attempt. Run
`status` on the new study, then use the HAIC submission path with its new root.

## Validation and execution status

Run focused checks from the repository root with the selected interpreter:

```bash
"$FI_PYTHON" -m unittest discover \
  -s tests -p 'test_future_innovation_scaling.py' -v

"$FI_PYTHON" -m unittest discover \
  -s tests -p 'test_future_innovation_*.py' -v
```

The September 11 validation exercised local calibration, source reservation,
regression tests, notebook inspection, shell syntax and submission dry runs.
Synthetic calibration tests the implementation; it does not measure the benefit
of more real training data.

Expanded preparation failed locally because the full annotation files were
absent. The HAIC connection returned `Permission denied (keyboard-interactive)`.
No real expanded teacher encoding, Slurm submission or learning-curve score is
claimed. A read-only status check on September 12 still identifies
`data/cohort-complete.json` as the next missing stage. The
[validation record](../../docs/studies/future-innovation/source-learning-curve-validation.md#concrete-blocker-and-the-next-executable-step)
links the exact failed commands and retained logs. Once the inputs and access
are available, resume from preparation using the frozen study above.
