# Source Learning-Curve Experiment on HAIC

This experiment tests whether more training recordings make skeleton history more
useful for predicting the existing teacher target. It reuses your completed
**gate-v2** run and writes the expanded study to a **new run directory**.

Use the same environment variables as [README.md](README.md). The launcher below
runs calibration and freezes the source reservation automatically. You do not
need separate calibration commands, exposure-manifest arguments, or
`FI_SCALING_ROOT`.

## 1. Set the paths

On HAIC, use the model and data paths from your working gate-v2 setup. Adjust
`FI_PARENT_ROOT` if gate-v2 is stored elsewhere. `FI_RUN_ROOT` must be a separate,
new directory; keep that same path when resuming.

```bash
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export GAVD_FULL_ROOT="$GAVD6_ROOT/data/gavd_full"
export VJEPA2_ROOT="/hai/scratch/$USER/vendor/vjepa2"
export FI_TEACHER_CHECKPOINT="/hai/scratch/$USER/models/vjepa2_1_vitb_dist_vitG_384.pt"
export FI_POSE_MODEL="/hai/scratch/$USER/models/pose_landmarker_lite.task"
export FI_ANNOTATION_ROOT="$GAVD_FULL_ROOT/annotations/GAVD/data"
export FI_PARENT_ROOT="$GAVD6_ROOT/outputs/future-innovation/gate-v2"
export FI_RUN_ROOT="$GAVD6_ROOT/outputs/future-innovation/source-learning-curve-v1"
cd "$GAVD6_ROOT"
```

`FI_PARENT_ROOT` supplies the existing cache and stays read-only. `FI_RUN_ROOT`
receives all new outputs. Use your existing environment; `FI_ENVIRONMENT` works
as in the main README and otherwise defaults to `.venv`.

The launcher finds full manifests under `$GAVD_FULL_ROOT/manifests/`, or uses
`manifests/gavd/` in the checkout. It expects the five official annotation CSVs in
`FI_ANNOTATION_ROOT` and source videos under `$GAVD_FULL_ROOT/youtube/all/`.
Missing required paths are reported before submission.

## 2. Submit the experiment

```bash
bash slurm/future-innovation-scaling/launch/submit.sh all
```

This is the command for both a new study and resuming an existing one. It submits:

```text
CPU: calibration and source reservation
  → CPU: alignment and pose preparation
  → GPU: teacher cache, validity audits and training-subset plan
  → CPU: five-fold learning-curve fitting
  → CPU: report and numerical verification
  → CPU: execute the source learning-curve notebook
```

Data and model stages wait for the previous stage to succeed. The final notebook
waits for all submitted stages to finish, including failures, so it can also show
incomplete evidence. New initialization includes
this repository's known exposure inventory and available sibling
`gate-*/manifests/gate-windows.csv` files, including gate-v1. Those recordings stay
in development. Confirmation recordings are reserved before expanded processing.
If you have inspected additional recordings outside these inventories, add them
as described in the [advanced instructions](SOURCE_LEARNING_CURVE_DETAILS.md#additional-exposure-or-participant-information)
before the first submission.

For a preview without submitting jobs, append `--dry-run` to the command above.
No transfer of the separately frozen local study is required when starting this
new HAIC study against your HAIC gate-v2 directory.

## 3. Check progress and results

```bash
bash slurm/future-innovation-scaling/launch/submit.sh status
```

Status reports the next missing stage; it does not reconstruct numerical results.
The report job performs numerical verification after fitting.

| Location under `$FI_RUN_ROOT` | What to read |
|---|---|
| `logs/` | Job output, errors and submitted job IDs |
| `reports/cohort-audit.json` | Recording counts and development/confirmation reservation |
| `reports/learning-curve.svg` | Learning-curve figure, available after reporting |
| `reports/learning-curve.json` | All arm comparisons, uncertainty and the development decision |
| `notebook_runs/<batch>/23_source_learning_curves.ipynb` | Executed notebook with saved tables and figures |

`all` executes [notebook 23](../../notebooks/future_innovation/23_source_learning_curves.ipynb)
automatically. To execute it again for an existing
study, without rerunning data processing or fitting:

```bash
bash slurm/future-innovation-scaling/launch/submit.sh notebooks
```

Each execution creates a new notebook batch and uses `FI_RUN_ROOT` automatically.
The notebook inspects saved artifacts; successful execution does not establish
that the experiment is complete or numerically verified.

The main scientific comparison is **real history minus no skeleton** at each
training size, alongside gain over RGB-only and the shuffle/mismatch controls.
The [protocol](../../docs/studies/future-innovation/source-learning-curve-protocol.md)
defines the model, source subsets, metric and decision rules.

## Resume after a failure

Read the failed job's log, fix the reported input or environment problem, then
rerun `all` with the same paths. Completed stages verify and reuse their outputs.
A frozen reservation is not regenerated. If all data stages have completed, you
can resume directly from CPU fitting:

```bash
bash slurm/future-innovation-scaling/launch/submit.sh fit
```

Keep gate-v2 unchanged after initialization. For manual commands, other resume
modes, or relocation of the older frozen study, see the
[advanced instructions](SOURCE_LEARNING_CURVE_DETAILS.md). The
[validation record](../../docs/studies/future-innovation/source-learning-curve-validation.md)
distinguishes local checks from actual HAIC execution; no expanded real-data
learning curve has been measured locally.
