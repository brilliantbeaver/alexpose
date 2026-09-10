# Run Experiment 0 through notebooks on HAIC

The five notebooks in `notebooks/experiments/future_innovation/` now have three
modes: `teach` (small generated examples, the default), `inspect` (read saved
artifacts), and `execute` (run the existing production CLI stages). Execution
reproduces the full **Experiment 0 gate**, including all controls and nested
selection. It does not train a distillation adapter or authorize a later study.

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
export FI_RUN_ROOT=/path/to/runs/fi-gate-v1
export GAVD_FULL_ROOT=/path/to/gavd-full
export VJEPA2_ROOT=/path/to/pinned/vjepa2
export FI_TEACHER_CHECKPOINT=/path/to/vjepa2_1_vitb384.pt
export FI_POSE_MODEL=/path/to/pose_landmarker_heavy.task
# Optional: FI_ENVIRONMENT, FI_ANNOTATION_ROOT, FI_CHANGE_REASON (see README).
cd "$GAVD6_ROOT"
bash slurm/future-innovation/submit-fi-notebooks.sh all
```

`all` submits 00 → 01 → 02 → the five-task 03 array using `afterok`. Notebook 04
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

Each launch creates a unique `$FI_RUN_ROOT/notebook_runs/execute-NN-…/` directory;
03 also includes its fold number. It contains the executed `.ipynb`,
`execution.json` with interpreter, source hash, job IDs and status, copies of the
notebook orchestration sources, and per-command logs and exit-status receipts.
Notebook saves are atomic before each cell, after completed/error cells, and on
normal or exceptional exit. A hard kill can leave status `running` and the last
completed cell; consult Slurm accounting and the stage logs for that case.
The active long-running cell streams its CLI output into a durable `stage.log`.
Canonical notebooks stay output-free and are never overwritten by execution.

Re-submit the same phase and run directory after repairing an execution issue.
Completed data, caches, audits and folds are verified and reused. Contracts are
not deleted or replaced. A protocol change needs a new versioned run. Code and
runtime drift are recorded by the existing provenance checks; review that record
before claiming a reproduction across versions.

Notebook 04 attempts a report even when scoring fails, then returns failure for
incomplete, invalid or unsealed measurement. A complete `STOP` or `INCONCLUSIVE`
returns success: the scientific outcome is in `reports/gate-decision.json` and
`reports/gate-report.md`, independently of notebook process status. Synthetic
smoke artifacts remain marked synthetic and cannot authorize advancement.

## Interactive execution

On a suitably allocated compute node, execute all cells of 00 → 04 with
`FI_TUTORIAL_MODE=execute` and the same explicit `FI_RUN_ROOT`. The equivalent
single-notebook command, using the active project environment, is:

```bash
uv run --no-sync python scripts/research_directions/future_innovation/execute_future_innovation_notebook.py \
  --notebook 03 --mode execute --run-root "$FI_RUN_ROOT"
```

Without `--outer-fold`, 03 runs all five folds sequentially with all three seeds,
all five arms and the frozen selection grid. `--outer-fold 0` through `4` runs one
fold. The executor clears inherited fold settings unless the option is supplied.
Teacher execution defaults to CUDA (`--device cpu` is available for a deliberate
CPU run). Per-cell timeouts are unlimited by default; Slurm enforces the job
wall time. `--timeout SECONDS` can impose a positive per-cell limit locally.

## Local verification

See the [tutorial maintenance guide](../../scripts/research_directions/future_innovation/future_innovation_tutorial_guide.md)
for generation and checks. `verify_future_innovation_tutorials.py --pipeline-smoke`
executes the real notebook/CLI path on a separate synthetic fixture: missing-fit
failure, cache/audit reuse, all five folds and 75 final heads, complete report,
then sealed resume. The fixture reduces model width, updates, search and bootstrap
budget explicitly; it does not measure real accuracy or teacher causality.
Real cohort/pose extraction and real H100 teacher inference still require HAIC
inputs and are not certified by that local smoke test.
