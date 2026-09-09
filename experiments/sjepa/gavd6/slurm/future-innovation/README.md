# Future Innovation Experiment 0 on HAIC

This implements the [experiment guide](../../notes/future-innovation-distillation/experiment-0-guide.md): 50 aligned full-GAVD windows, a frozen V-JEPA 2.1 ViT-B/16 teacher, source-disjoint nested ridge/residual fits, all five arms, three initialization seeds, and a reproducible `ADVANCE` / `STOP` / `INCONCLUSIVE` report. It does not train S-JEPA or a distillation adapter. No scientific experiment has been run as part of implementing these scripts.

## Prepare the environment and inputs

On HAIC, set explicit persistent paths. Each repaired protocol needs a new run ID and change reason. The full-GAVD source cache and annotation checkout live under the gavd6 checkout. `FI_RUN_ROOT` may be inside or outside the checkout; choose a versioned directory that is ignored by Git when it is inside.

```bash
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export GAVD_FULL_ROOT="$GAVD6_ROOT/data/gavd_full"
export VJEPA2_ROOT="/hai/scratch/$USER/vendor/vjepa2"
export FI_TEACHER_CHECKPOINT="/hai/scratch/$USER/models/vjepa2_1_vitb_dist_vitG_384.pt"
export FI_POSE_MODEL="/hai/scratch/$USER/models/pose_landmarker_lite.task"
export FI_ANNOTATION_ROOT="$GAVD_FULL_ROOT/annotations/GAVD/data"
export FI_RUN_ROOT="/hai/scratch/$USER/experiments/future-innovation/gate-v1"
export FI_CHANGE_REASON="Initial Experiment 0 preregistration"
cd "$GAVD6_ROOT"
bash slurm/future-innovation/00-setup-environment.sh
```

The setup script pins the official source to `204698b45b3712590f06245fbfba32d3be539812`, downloads public checkpoints only when absent, and runs `uv sync --frozen --extra future-innovation`. The optional dependencies pin torchvision, timm, and einops without changing the project's `torch==2.6.0+cu124` pin. OpenCV already provides the decoder; no additional decoder is needed. `FI_ENVIRONMENT` can select a separate locked project environment. The CUDA lock targets HAIC Linux; do not synchronize it over an existing macOS development environment.

The reviewed [official Hub builder](https://github.com/facebookresearch/vjepa2/blob/204698b45b3712590f06245fbfba32d3be539812/src/hub/backbones.py) points its automatic checkpoint downloader at localhost. The adapter therefore invokes the same builder with `pretrained=False`, discards its unused predictor, and strictly loads the documented `ema_encoder` state from the explicit local checkpoint. No uninitialized teacher is accepted. The input checkpoint, teacher code, lockfile, package versions, configurations, and implementation files are fingerprinted.

Required data layout:

```text
$GAVD_FULL_ROOT/
  manifests/gavd_full_sequences.csv
  manifests/gavd_full_videos.csv
  annotations/GAVD/data/GAVD_Clinical_Annotations_1.csv ... _5.csv
  youtube/all/<video_id>.mp4   # existing supported extensions also work
```

Use the existing `gavd6 gavd download` workflow to fill missing videos. The sequence manifest alone has no per-frame boxes; all five non-overlapping annotation partitions are needed. `FI_ANNOTATION_ROOT` is set above to the `GAVD/data` directory inside the checked-out annotation source. Do not mix incremental `1.1`, `1.2`, or `1.3` files with the five official partitions, or substitute the historical GAVD96 pose cache.

## Build and inspect the frozen cohort

```bash
bash slurm/future-innovation/submit-fi-pipeline.sh prepare
```

This submits `01` followed by `02` with an `afterok` dependency. Scripts use HAIC's existing `mind` account and `hai` partition. All submitted logs use absolute paths beneath `$FI_RUN_ROOT/logs`; submission IDs and dependencies are recorded in `submissions.tsv`.

Candidate order and window starts are fixed by seed `260905`. The single conversion from one-based GAVD annotation frames to zero-based decoder frames occurs in candidate construction. Eligibility requires 64 consecutive annotated boxes, successful exact decoding, at least 90% crop retention in the context and target frames, nonempty person/background token regions, and at least 0.45 valid whole-body context coverage. Requiring boxes across all 64 frames also makes every selected window eligible to supply the pixel-edit audit. Sources contribute at most two windows. Selection stops at exactly 50, excludes no rows based on teacher features or scores, and records all exclusions.

The frozen cohort is `manifests/gate-windows.csv`. Inspect at least one image from each of its five `outer_fold` values in `qc/alignment-overlays/`. Each sheet labels source/local frame numbers, observed/hidden frames, boxes, context landmarks, and target frames 38–39. Record the IDs you actually inspected:

```bash
uv run --no-sync gavd6 future-innovation review-alignment \
  --run-root "$FI_RUN_ROOT" --reviewer "$USER" \
  --window-ids WINDOW_FROM_FOLD_0 WINDOW_FROM_FOLD_1 WINDOW_FROM_FOLD_2 WINDOW_FROM_FOLD_3 WINDOW_FROM_FOLD_4 \
  --note "Describe observed RGB/box/landmark alignment and any reviewed concerns."
```

This is the guide's actual pre-feature alignment inspection. The pipeline deliberately records it as evidence rather than claiming that rendering an image proves someone inspected it. Incorrect alignment requires a repaired, newly versioned cohort.

## Cache, audit, fit, and score

```bash
# Leave the final qualitative capacity interpretation open until scores exist.
export FI_SCORE_ONLY=1
bash slurm/future-innovation/submit-fi-pipeline.sh compute
```

The dependency chain is:

```text
03 cache (one H100, 96 GB host RAM, 8 h)
  -> 04 pixel/causal audits (one H100, 96 GB, 8 h)
  -> 05 fit (five CPU array tasks, four CPUs/32 GB/12 h per fold)
  -> 06 score/report (CPU, 16 GB, 2 h; afterany)
```

CPU-only cohort/pose jobs request 32/64 GB respectively. Small full-batch heads run on CPUs; no teacher weights are loaded into those tasks. `05` fits all three seeds and five arms for one outer fold, reusing the person-target baseline across the four person-target arms. Its default inner grid uses ridge alphas `(0.1, 1, 10, 100, 1000)`, AdamW weight decays `(0.01, 0.1)`, and update budgets `(25, 50, 100, 200)`. All heads have width 64 and the same parameter count. Inner source validation selects each head's decay and duration; final outer fits use those fixed choices. Outer test sources never select hyperparameters.

Every fitted mean, scale, imputation statistic, residual, and inner baseline uses only its training partition. Clip-mismatch donors are matched separately within each inner/outer training, validation, and test partition using context metadata and different-source constraints. Hungarian assignment avoids replacement when possible; a documented replacement fallback handles an infeasible partition. Shuffle blocks contain four frames, with coordinates/confidence/validity moved together. No-skeleton inputs zero coordinates and confidence while preserving validity.

The pre-attention teacher mask retains only past tubelets. Crop statistics use image dimensions and fixed constants. Skeleton normalization and camera/background/pose-quality features use frames 0–31 only. The full-clip target at frames 38–39 is contextual: it may contain information from frames 40–63. Background targets also retain teacher attention to the person; the report states this limitation.

Ten audit windows and different-source pixel donors are selected before feature extraction. Audits repeat float32 inference, randomize all future pixels, and compare future-person replacement with static-background replacement. Both interventions use an inward eight-pixel feather. Donor person regions use reversed future timing; static backgrounds use a donor frame with its person box inpainted. Contact sheets and all measurements are saved. Person appearance and editing artifacts can influence these tests. Any failed validity audit prevents fitting.

Failed dependencies are canceled with `--kill-on-invalid-dep=yes`; the report job uses `afterany` to produce a `STOP` report when fitting is unavailable. If the scheduler cannot resolve an upstream dependency, run `build-report` manually once the job states are settled. A STOP caused by incomplete/invalid measurement is distinguished from a completed valid null result.

## Interpret capacity and seal the decision

When `06` completes, inspect `reports/aggregate-metrics.csv`, `paired-controls.csv`, `uncertainty.json`, and the pixel-edit contact sheets. The real-minus-no-skeleton comparison includes a paired source-bootstrap interval and the fraction of draws in which real wins. The guide intentionally gives no numerical capacity cutoff; an identified reviewer must state why motion attribution is clear or unresolved.

```bash
uv run --no-sync gavd6 future-innovation assess-capacity \
  --run-root "$FI_RUN_ROOT" --reviewer "$USER" --finding unresolved \
  --evidence "Describe the actual paired capacity difference, interval, and per-seed evidence."
uv run --no-sync gavd6 future-innovation build-report --run-root "$FI_RUN_ROOT"
```

Choose `--finding clear` only when that interpretation is supported by the saved comparison. Omitting an assessment leaves capacity unresolved. For fully unattended finalization, submit `compute` with `FI_SCORE_ONLY=0` (the default): valid point passes with no capacity interpretation are `INCONCLUSIVE`. Do not choose a new cutoff after looking at scores.

`reports/gate-decision.json` and `gate-report.md` are sealed once written. `ADVANCE` requires all frozen point/validity checks, positive real gain in at least 90% of source bootstrap draws, positive gain in all three seeds, at least two seeds reaching 0.05, and clear capacity attribution. Adapter training is always disallowed by this raw-skeleton gate.

## Outputs, resumption, and validation

The guide's output checklist is implemented: configuration/lock/environment records, frozen cohort and audit manifests, poses and decoded RGB, projection and checksummed caches, ridge/preprocessing objects, 75 residual checkpoints, split/donor/inner-selection audits, raw and standardized long-form predictions for every arm, featurewise scores, seed metrics, 2,000 paired source-bootstrap draws, and the final decision/report. Person and background targets each use their own training-variance feature mask. Undefined F8 and invalid feature scores remain explicitly missing; NaN/infinite input scores cannot silently become zero.

Completed cache entries and complete outer folds are verified and reused. Interrupted folds refit deterministically from their frozen configuration; training has at most 200 updates. An incomplete pose stage can restart from its fixed candidates. CLI stage locks reject duplicate writers to the same stage/fold and are released by the OS after a crash. Code, runtime, manifest, and configuration changes fail closed and require a new run ID; final reports cannot be overwritten after inspecting scores.

Focused tests and a small complete model/report smoke run:

```bash
uv run --no-sync python -m unittest discover -s tests -p 'test_future_innovation_*.py'
FI_TEST_VJEPA_ROOT="$VJEPA2_ROOT" uv run --no-sync python -m unittest discover \
  -s tests -p 'test_future_innovation_official_adapter.py'
uv run --no-sync gavd6 future-innovation smoke \
  --run-root "/hai/scratch/$USER/experiments/future-innovation/synthetic-smoke-v1"
```

The optional official-source test uses a small randomly initialized instance of the actual V-JEPA 2.1 encoder to check mask order, preprocessing, token order, and causal invariance. The smoke command exercises all 75 tiny residual fits and reports from synthetic cached features; it always forbids scientific advancement. Neither substitutes for real checkpoint inference or cohort audits on HAIC.

All commands have `--help`, and each numbered `.sbatch` can be submitted independently with the required environment. If submitting manually, override `--output` and `--error` with absolute paths to an existing logs directory and arrange the same dependencies.
