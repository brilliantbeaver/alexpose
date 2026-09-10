# Future Innovation Experiment 0 on HAIC

This implements the [experiment guide](../../notes/future-innovation-distillation/experiment-0-guide.md): 50 aligned full-GAVD windows, a frozen V-JEPA 2.1 ViT-B/16 teacher, source-disjoint nested ridge/residual fits, all five arms, three initialization seeds, and a reproducible `ADVANCE` / `STOP` / `INCONCLUSIVE` report. It does not train S-JEPA or a distillation adapter.

Start with the [study overview and five notebooks](../../docs/studies/future-innovation/)
for the question, controls and current evidence. Copied 10 September 2026 HAIC
notebooks show a completed cohort/cache, a failed validity audit, and an incomplete
STOP. See the [investigation](../../docs/studies/future-innovation/notebook-run-investigation.md)
for the observed failures and repairs. This does not infer live HAIC job status.

To execute and retain the five notebooks themselves, use the separate
[notebook HAIC path](NOTEBOOKS.md): `submit-fi-notebooks.sh all` schedules jobs
`10`–`14`, including the five-fold notebook array. It shares the setup below and
the production CLI. The original `01`–`06` scripts remain the direct CLI path.

New runs discover exact source IDs throughout declared storage and accept
explicit full-source paths in the video manifest. Set `FI_VIDEO_ROOTS` to
additional colon-separated HAIC directories before initialization. The
[notebook guide](NOTEBOOKS.md) describes supported paths and the single shared
notebook output folder. Existing frozen cohorts retain their original selection;
use a new run ID to apply changed discovery rules.

## Prepare the environment and inputs

On HAIC, set explicit persistent paths. Reuse the same run directory to resume; choose a new directory when intentionally changing the dataset, teacher, or scientific configuration. `FI_CHANGE_REASON` is optional descriptive metadata. The full-GAVD source cache and annotation checkout live under the gavd6 checkout. `FI_RUN_ROOT` may be inside or outside the checkout; relative run paths are resolved against `GAVD6_ROOT` and exported as absolute paths to every job.

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

The setup script pins the official source to `204698b45b3712590f06245fbfba32d3be539812`, downloads public checkpoints only when absent, and runs `uv sync --frozen --extra future-innovation`. The optional dependencies pin torchvision, timm, and einops without changing the project's `torch==2.6.0+cu124` pin. OpenCV already provides the decoder; no additional decoder is needed. `FI_ENVIRONMENT` can select a separate project environment. Stage execution records the actual runtime and does not require exact version-string equality with the initialization environment; imports, checkpoint loading, tensor geometry and validity audits still have to work. The CUDA lock targets HAIC Linux; do not synchronize it over an existing macOS development environment.

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

For an unattended run, submit every stage with its dependencies:

```bash
bash slurm/future-innovation/submit-fi-pipeline.sh all
```

This also resumes an existing run: completed stages verify and reuse their artifacts. The submission wrapper can infer `GAVD6_ROOT` from its own location. Keep `FI_RUN_ROOT` explicit so outputs do not accidentally go to a different run.

Alternatively, submit preparation separately:

```bash
bash slurm/future-innovation/submit-fi-pipeline.sh prepare
```

This submits `01` followed by `02` with an `afterok` dependency. Scripts use HAIC's existing `mind` account and `hai` partition. All submitted logs use absolute paths beneath `$FI_RUN_ROOT/logs`; submission IDs and dependencies are recorded in `submissions.tsv`. Non-array logs use `<stage>-<job>.out/.err`; fit array logs use `fit-<array-job>_<fold>.out/.err`. Older `_4294967294` suffixes are Slurm's non-array placeholder, not an error.

Candidate order and window starts are fixed by seed `260905`. The single conversion from one-based GAVD annotation frames to zero-based decoder frames occurs in candidate construction. Eligibility requires 64 consecutive annotated boxes, successful exact decoding, at least 90% crop retention in the context and target frames, nonempty person/background token regions, and at least 0.45 valid whole-body context coverage. Requiring boxes across all 64 frames also makes every selected window eligible to supply the pixel-edit audit. Sources contribute at most two windows. Selection stops at exactly 50, excludes no rows based on teacher features or scores, and records all exclusions.

The frozen cohort is `manifests/gate-windows.csv`. The pipeline verifies exact frame decoding, crop retention, landmark validity, and all frozen-artifact hashes before it proceeds. It also saves labeled overlays for every window in `qc/alignment-overlays/` for optional post-run diagnosis; they do not block the next stage.

## Cache, audit, fit, and score

```bash
bash slurm/future-innovation/submit-fi-pipeline.sh compute
```

Use `compute` once preparation has completed. It checks for the saved cohort contract before allocating jobs. Use `all` if preparation is still needed; its teacher job waits for pose extraction.

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

Failed dependencies are canceled with `--kill-on-invalid-dep=yes`; the report job waits with `afterany` for the teacher, audits, and fitting to terminate, including failed/canceled jobs. If the scheduler cannot resolve an upstream dependency, run `build-report` manually once the job states are settled. Missing evidence produces a resumable STOP report with `measurement_complete: false`, not a sealed scientific result. If damaged configuration/artifacts prevent even that report, the CLI writes `reports/pipeline-error.json` and exits nonzero with the concrete cause.

## Automatic decision and report

Stage `06` automatically scores the complete out-of-fold predictions and writes `reports/gate-decision.json` and `reports/gate-report.md`. Complete, valid measurements have `measurement_complete: true` and are sealed with `reports/final-report-contract.json`, including valid null results. The report includes the paired real-minus-no-skeleton comparison, source-bootstrap interval, and fraction of draws in which real wins. `ADVANCE` requires all frozen point/validity checks, positive real gain in at least 90% of source bootstrap draws, positive gain in all three seeds, and at least two seeds reaching 0.05. Adapter training is always disallowed by this raw-skeleton gate.

## Outputs, resumption, and validation

The guide's output checklist is implemented: configuration/lock/environment records, frozen cohort and audit manifests, poses and decoded RGB, projection and checksummed caches, ridge/preprocessing objects, 75 residual checkpoints, split/donor/inner-selection audits, raw and standardized long-form predictions for every arm, featurewise scores, seed metrics, 2,000 paired source-bootstrap draws, and the final decision/report. Person and background targets each use their own training-variance feature mask. Undefined F8 and invalid feature scores remain explicitly missing; NaN/infinite input scores cannot silently become zero.

Completed candidates, poses, caches, passing audits, and outer folds are verified and reused. Complete caches/audits do not reload the teacher. Interrupted folds refit from their frozen configuration; training has at most 200 updates. An incomplete pose stage can restart from its fixed candidates. Frozen projection matrices are reused byte-for-byte, without recomputing a potentially different QR decomposition on a new numerical runtime. CLI stage locks reject duplicate writers to the same stage/fold and are released by the OS after a crash.

Code and runtime differences are provenance, not failures: each process records its code fingerprint, runtime versions, and Slurm identity in `logs/provenance/`. The original run/configuration contracts are not rewritten, so existing cohort and cache bindings remain valid. The report's `code_sha256` is the initialization fingerprint; `report_code_sha256` identifies the reporting implementation. A changed hash cannot tell whether the experiment is broken. The pipeline continues to enforce actual configuration/artifact integrity, checkpoint identity, array shapes, finite values, source isolation, causality and target-validity checks. This does not establish numerical equivalence between arbitrary code/runtime versions; intentionally changing scientific behavior warrants a new run.

Incomplete STOP reports do not block later fitting. On regeneration, the prior report is preserved under `reports/attempts/<decision-hash>/`. This also recovers incomplete STOP reports sealed by older scripts. Complete scientific reports remain immutable.

## Recover the existing gate-v1 run

After updating the checkout with these scripts, keep the existing artifacts and resume:

```bash
export GAVD6_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
export FI_RUN_ROOT="$GAVD6_ROOT/outputs/future-innovation/gate-v1"
cd "$GAVD6_ROOT"
bash slurm/future-innovation/submit-fi-pipeline.sh compute
```

If preparation did not finish, use `all` instead. An initialized run reads its saved input paths, so resumption does not require re-exporting the initialization-only model/input variables. Do not edit saved hashes or delete the run contract to resume. Code/version drift no longer requires `gate-v2`.

The MediaPipe `Feedback manager requires a model with a single signature inference` message is a nonfatal library warning. Check the pose job's Slurm state/exit code and stdout for actual failures; changing the model or suppressing all stderr is unnecessary.

## Validation

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
