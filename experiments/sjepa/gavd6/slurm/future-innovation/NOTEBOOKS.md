# Run Experiment 0 through notebooks on HAIC

The five notebooks in `notebooks/experiments/future_innovation/` now have three
modes: `teach` (small generated examples, the default), `inspect` (read saved
artifacts), and `execute` (run the existing production CLI stages). Execution
reproduces the full **Experiment 0 gate**, including all controls and nested
selection. New runs use direct-v2: **50 clips**, four arms and an explicit
real-minus-no-skeleton comparison, without background-quality/selectivity gates.
The full dataset is reserved for the real experiment. ADVANCE recommends planning
that full-GAVD JEPA comparison; it never launches training or enables distillation.
See the [protocol](../../docs/studies/future-innovation/direct-gate-protocol.md).

The separate jobs `10`–`14` execute these notebooks in fresh kernels. The original
jobs `01`–`06` and `submit-fi-pipeline.sh` remain the direct CLI path. Both paths
use the same run contracts, code, deterministic thread settings and resumable
artifacts. Choose one submission path for a run at a time; concurrent duplicate
writers fail the existing stage locks.

| Notebook | Production commands | HAIC allocation |
| --- | --- | --- |
| 00: question and setup | `init-run`, or validate the existing run | 4 CPUs, 32 GB, 2 hours |
| 01: cohort and alignment | `build-cohort`, `extract-poses` | 8 CPUs, 64 GB, 10 hours |
| 02: teacher and validity | `cache-teacher`, `audit-teacher` | 1 H100, 8 CPUs, 96 GB, 16 hours |
| 03: predictors and controls | `run-gate --outer-fold N --device cpu` | Five tasks, each 4 CPUs, 32 GB, 12 hours |
| 04: results | `score-gate`, `build-report` | 4 CPUs, 16 GB, 2 hours |

## Prepare and submit

First follow the [existing environment and input setup](README.md): stage the
full GAVD manifests, annotation partitions and videos, the MediaPipe model, the
pinned V-JEPA checkout and checkpoint, and run `00-setup-environment.sh`. This
reuses the locked HAIC environment, including Jupyter, nbclient and ipykernel.
Neither submission nor notebook execution installs dependencies or downloads
research inputs. Do not change code or the environment while jobs are active.

For a new run, set the same inputs used by the original jobs:

```bash
export GAVD6_ROOT=/path/to/checkout/gavd6
export FI_RUN_ROOT=/path/to/runs/fi-direct-v2
export FI_EXPERIMENT_PROTOCOL=direct-v2
export GAVD_FULL_ROOT=/path/to/gavd-full
export VJEPA2_ROOT=/path/to/pinned/vjepa2
export FI_TEACHER_CHECKPOINT=/path/to/vjepa2_1_vitb384.pt
export FI_POSE_MODEL=/path/to/pose_landmarker_heavy.task
# Optional: FI_ENVIRONMENT, FI_ANNOTATION_ROOT, FI_CHANGE_REASON (see README).
# Additional full-source video directories, if needed:
# export FI_VIDEO_ROOTS=/path/to/other/videos:/path/to/another/cache
cd "$GAVD6_ROOT"
bash slurm/future-innovation/submit-fi-notebooks.sh all
```

`all` submits 00 → 01 → 02 using `afterok`. The five-task 03 array uses `afterany`
on 02 so it retains a diagnostic notebook even if the audit fails. **The fitting
CLI still prohibits training unless all validity checks pass.** Notebook 04
uses `afterany` for **every submitted predecessor**, so upstream failures can
still produce a diagnostic report. Invalid dependent jobs are cancelled. Job
IDs and dependencies are saved in `$FI_RUN_ROOT/logs/notebook-submissions.tsv`.
Slurm stdout/stderr use `logs/notebook-*.out` / `*.err`, with array task IDs for 03.

Alternatively, submit `prepare` (00–01), wait for the frozen cohort, then submit
`compute` (02–04). `compute` requires `config/cohort-contract.json`; production
commands subsequently verify its contents. For an initialized run, only
`GAVD6_ROOT`, `FI_RUN_ROOT`, and the environment are needed; recorded input paths
must remain accessible. Relative run paths resolve from the checkout.

## Outputs and recovery

Each submission creates **one notebook-only folder**, printed before submission:

```text
$FI_RUN_ROOT/notebook_runs/haic-<batch-id>/
    00_question_and_worked_example.ipynb
    01_cohort_and_alignment.ipynb
    02_teacher_features_and_validity.ipynb
    03_matched_predictors_and_controls_fold-0.ipynb
    ... fold-1 through fold-4 ...
    04_results_and_next_decision.ipynb
```

All jobs inherit `FI_NOTEBOOK_OUTPUT_DIR`. Fold suffixes prevent concurrent
writers from overwriting each other. Interpreter, source hashes, job IDs and
execution status are embedded in each notebook's `metadata.fi_execution`.
A completed diagnostic for a verified audit rejection has `status: blocked`,
`execution_completed: true`, and `scientific_outcome.measurement_complete: false`.
Its process exits zero so downstream diagnostics can finish; this does not
authorize fitting or mark the scientific measurement complete.
There are no duplicate source snapshots, execution JSON files or font/kernel
caches in the notebook folder. Durable command/output/exit logs are stored in
`$FI_RUN_ROOT/logs/notebooks/haic-<batch-id>/`; normal Slurm logs remain in `logs/`.
Notebook saves are atomic before each cell, after completed/error cells, and on
normal or exceptional exit. A hard kill can leave status `running` and the last
completed cell; consult Slurm accounting and the stage logs for that case.
The active long-running cell streams its CLI output into a durable stage log.
Canonical notebooks stay output-free and are never overwritten by execution.

Re-submit the same phase and run directory after repairing an execution issue.
Completed data, caches, audits and folds are verified and reused. Contracts are
not deleted or replaced. A protocol change needs a new versioned run. Code and
runtime drift are recorded by the existing provenance checks; review that record
before claiming a reproduction across versions.

For newly initialized runs, source discovery checks full-source paths declared
in video-manifest columns `video_path`, `local_path`, `cached_path` or
`source_path`, then exact video IDs recursively under the configured YouTube
directory and optional `FI_VIDEO_ROOTS`. Relative manifest paths resolve from
the manifest's directory or the declared storage roots. Supported containers
include MP4, MKV, WebM, MOV, M4V and AVI. Ambiguous exports require an explicit
path; symlinks and hard links to the same file are deduplicated.

Notebook 01 reports source availability separately from candidate eligibility
and cohort selection. The 50-window limit and two-window source cap are selection
outcomes, not missing-video failures. A sequence still needs 64 contiguous
annotated source frames; source-video duration alone does not establish this.
An annotation gap no longer discards another intact 64-frame window. No frames
are padded, repeated or interpolated. Missing-source discovery can be retried
before candidates freeze. Applying new discovery rules to an already frozen
cohort requires a new run (for example `gate-v2`); existing run artifacts remain
unchanged on resume.

Notebook 02 displays failed check names, frozen thresholds and per-window
stability and prefix-isolation CSVs. Target-sensitivity tables belong only to
legacy runs; direct-v2 does not create or require them. A complete, internally
consistent audit that fails a frozen criterion ends with **TRAINING BLOCKED**.
Notebook 03 verifies the same rejection, displays its fold inventory and skips
fitting. Missing/corrupt artifacts or unexpected CLI failures still produce
failed notebooks. The audit CLI returns exit 2 for a verified rejection on
both first execution and reuse; notebook orchestration validates the retained
evidence before recognizing that outcome.

Notebook 04 builds an unsealed diagnostic STOP for a verified audit rejection,
bound to that audit's checksum. It keeps `measurement_complete: false`,
`metrics: null`, and all advancement permissions false (including `allow_jepa_training_comparison` for direct-v2). A fresh blocked
report must match the current audit; arbitrary or stale STOP records cannot
hide execution errors. Other scoring failures still build diagnostics and
return failure. Complete `STOP` and `INCONCLUSIVE` measurements remain sealed
successful executions. Synthetic smoke artifacts remain marked synthetic and
cannot authorize advancement.

A legacy audit is never reclassified by changing its saved thresholds. The
separately initialized direct-v2 gate adopts the author's revised scientific
question and skips selectivity tests. The GOjuXSEB legacy sensitivity ratio was **1.360**,
below the frozen **2.0** requirement. See the
[investigation](../../docs/studies/future-innovation/notebook-run-investigation.md#run-haic-gojuxseb-verified-audit-rejection)
for evidence, repairs and remaining real-data checks.

## Interactive execution

On a suitably allocated compute node, execute all cells of 00 → 04 with
`FI_TUTORIAL_MODE=execute` and the same explicit `FI_RUN_ROOT`. The equivalent
single-notebook command, using the active project environment, is:

```bash
uv run --no-sync python scripts/research_directions/future_innovation/execute_future_innovation_notebook.py \
  --notebook 03 --mode execute --run-root "$FI_RUN_ROOT"
```

Without `--outer-fold`, 03 runs all five folds sequentially with all three seeds,
all four direct-v2 arms and the frozen selection grid (five arms for legacy runs). `--outer-fold 0` through `4` runs one
fold. The executor clears inherited fold settings unless the option is supplied.
Teacher execution defaults to CUDA (`--device cpu` is available for a deliberate
CPU run). Per-cell timeouts are unlimited by default; Slurm enforces the job
wall time. `--timeout SECONDS` can impose a positive per-cell limit locally.

Without an explicit output directory or batch environment, standalone runs
write into `$FI_RUN_ROOT/notebook_runs/manual/`. Use `--output-dir /path/to/folder`
to group a manual sequence; rerunning the same notebook there replaces its
previous executed copy atomically. Use a new folder when retaining attempts.

## Local verification

See the [tutorial maintenance guide](../../scripts/research_directions/future_innovation/future_innovation_tutorial_guide.md)
for generation and checks. `verify_future_innovation_tutorials.py --pipeline-smoke`
executes both frozen protocols on separate synthetic fixtures: missing-fit failure,
cache/readiness reuse, all five folds (60 direct-v2 heads; 75 legacy heads), complete
report and sealed resume. It also checks rejected readiness/audit outcomes in 02,
all five 03 kernels and 04, with zero fits and incomplete predictive measurement.
`--pipeline-protocol direct-v2` selects only the current protocol.

The fixture reduces width, update/search and bootstrap budgets explicitly and
cannot measure real accuracy or teacher behavior. Separate data-flow tests use
actual generated video decoding with an injected detector/teacher, including
absent background pixels and absent shared flow support. Real GAVD/MediaPipe and
pretrained H100 inference still require HAIC verification.
