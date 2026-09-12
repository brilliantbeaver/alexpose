# Source Learning-Curve Experiment on HAIC

This experiment tests whether more training recordings make skeleton history more
useful for predicting the existing teacher target. It uses **available videos
matched to the GAVD manifests**, reuses your completed **gate-v2** cache where
applicable, and writes the expanded study to a **new run directory**.

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
export FI_RUN_ROOT="$GAVD6_ROOT/outputs/future-innovation/source-learning-curve-available-dev-20260912"
cd "$GAVD6_ROOT"
```

`FI_PARENT_ROOT` supplies the existing cache and stays read-only. `FI_RUN_ROOT`
receives all new outputs. Use your existing environment; `FI_ENVIRONMENT` works
as in the main README and otherwise defaults to `.venv`.

The launcher finds full manifests under `$GAVD_FULL_ROOT/manifests/`, or uses
`manifests/gavd/` in the checkout. It expects the five official annotation CSVs in
`FI_ANNOTATION_ROOT` and source videos under `$GAVD_FULL_ROOT/youtube/all/`.
If your videos are in other subfolders under `youtube/`, set
`export FI_VIDEO_ROOT="$GAVD_FULL_ROOT/youtube"`; discovery searches recursively
for exact video IDs. These must be full recording files, not sequence clips.

Submission matches the manifests to local files and verifies the original
annotation/model checksums. **Missing, ambiguous, empty or unreadable recordings
are listed and excluded before processing.** Confirmation recordings stay
reserved. Frame decoding and pose quality are checked later. To preview the
available development count without submitting or creating a run:

```bash
bash slurm/future-innovation-scaling/launch/submit.sh check
```

For example, 305 planned development recordings with 12 unavailable produces
293 recordings for preparation; actual pose eligibility may reduce that count.
Every excluded ID and its reason is saved. Extra files absent from the manifests
are ignored.

This is the `available-development-v1` cohort amendment. If your earlier command
stopped at preflight before initialization, you can keep that intended run root.
If it already froze a study under the old policy/code, choose a fresh directory.
Keep the same directory and code for subsequent retries; do not edit saved hashes.

## 2. Submit the experiment

```bash
bash slurm/future-innovation-scaling/launch/submit.sh all
```

This is the command for both a new study and resuming an existing one. It submits:

```text
CPU: calibration, source reservation and available-video selection
  → CPU: alignment and pose preparation
  → GPU: teacher cache, validity audits and training-subset plan
  → CPU: five-fold learning-curve fitting
  → CPU: report and numerical verification
  → CPU: execute the source learning-curve notebook
```

The GPU job loads the original V-JEPA checkpoint and encodes the newly eligible
development clips. It reuses the 50 original cached windows where applicable;
those 50 arrays cannot supply features for the remaining videos. The subsequent
fitting, reporting and notebook jobs use the completed cache on CPUs. Notebook
23 displays the measured curve after these jobs succeed. Running the notebook
alone cannot create additional teacher features or fit the curve.

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
| `logs/` | Slurm output, errors and submitted job IDs |
| `logs/stages/` | Separate command, exit status and output for every stage attempt |
| `reports/cohort-audit.json` | Recording counts and development/confirmation reservation |
| `config/media-availability.csv` | Every manifest recording, availability, inclusion and exclusion reason |
| `config/processing-sequences.csv`, `config/processing-videos.csv` | Frozen processing inputs containing only available development recordings |
| `data/manifests/development-windows.csv` | Actual eligible expanded clips and folds after preparation |
| `data/teacher-cache/` | Newly encoded arrays; the parent's original cache stays in `FI_PARENT_ROOT/teacher-cache/` |
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
It shows initial inventory separately from expanded processing counts, all stage
attempts, and whether a successful verification log matches the current report.
That log records a previous check; the notebook does not reconstruct models.

The main scientific comparison is **real history minus no skeleton** at each
training size, alongside gain over RGB-only and the shuffle/mismatch controls.
The [protocol](../../docs/studies/future-innovation/source-learning-curve-protocol.md)
defines the model, source subsets, metric and decision rules. The
[cohort amendment](../../docs/studies/future-innovation/source-learning-curve-available-cohort-protocol.md)
defines availability selection. Conclusions concern the eligible available
recordings; their representativeness of the full GAVD corpus remains unestablished.

## Resume after a failure

Read the failed job's log, fix the reported input or environment problem, then
rerun `all` with the same paths. Completed stages verify and reuse their outputs.
A frozen reservation is not regenerated. If all data stages have completed, you
can resume directly from CPU fitting:

```bash
bash slurm/future-innovation-scaling/launch/submit.sh fit
```

Missing files excluded during initialization do not need to be downloaded to
run this study. Later downloads do not change its frozen membership. If an
**included** file disappears or changes before preparation completes, restore
that file or use a new run; the existing study is not silently changed on retry.
If a stage log still says `running` after a job stops, check Slurm's final state:
an abrupt termination can prevent the process from saving its final log record.

Keep gate-v2 unchanged after initialization. For manual commands, other resume
modes, or relocation of the older frozen study, see the
[advanced instructions](SOURCE_LEARNING_CURVE_DETAILS.md). The
[available-cohort validation](../../docs/studies/future-innovation/source-learning-curve-available-cohort-validation.md)
distinguishes local checks from actual HAIC execution. HAIC is not accessible
from the current assistant session, and no expanded real-data learning curve
has been measured locally.
