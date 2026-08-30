# Latent Laterality: ordered HAIC run guide

The local synthetic gate is a code/mechanism check only. It does not authorize
real SG-JEPA training. Run the real sequence gate first and stop unless its
`gate_decision.json` sets `ready_for_sg_jepa=true`.

## Inputs to supply

- `GAVD6_ROOT`, `AMASS_EXTRACTED_ROOT`, `AMASS_RUN_ROOT`,
  `AMASS_BODY_MODEL_ROOT`, `AMASS_SUBJECT_SPLITS`, and `GAVD_FULL_ROOT` from
  the HAIC environment contract below.
- The eligible AMASS inventory under `AMASS_RUN_ROOT/manifests`. Do not use the
  unrestricted inventory: the eligible inventory contains the 201 identities
  covered by the audited subject split.
- A full-GAVD pose manifest with `pose_path`, `sequence_id`, `video_id`,
  `split`, `fps`, `aspect_ratio`, and optional `condition`. Every source video
  must occur in one split only.
- Tensor/output roots with enough space.

The following variables are already part of the HAIC environment:

```bash
export GAVD_FULL_ROOT="/hai/scratch/$USER/datasets/gavd_full"
export CODY_JEPA_ROOT="/hai/scratch/$USER/cody-jepa"
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export GAVD5TM_ROOT="$SJEPA_ROOT/gavd5-tm"
export AMASS_ROOT="$GAVD6_ROOT/data/amass"
export AMASS_EXTRACTED_ROOT="$AMASS_ROOT/extracted"
export AMASS_RUN_ROOT="$AMASS_ROOT/outputs"
export AMASS_BODY_MODEL_ROOT="/hai/scratch/$USER/body_models"
export AMASS_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"
export GAIT_PARITY_POSE_DIR="$GAVD5TM_ROOT/work/artifacts/real/poses"
export BABEL_ROOT="$GAVD6_ROOT/data/babel_v1.0_release"
export SWAP_PROBE_CHECKPOINT="$AMASS_RUN_ROOT/repaired-jepa-seed7-v2/seed-7_standard_sjepa_best.pt"
export SWAP_PROBE_OUTPUT_DIR="$AMASS_RUN_ROOT/swap-probe-seed7"
```

Define only the study-specific derived paths:

```bash
export LATENT_LATERALITY_RUN_ROOT="$AMASS_RUN_ROOT/latent-laterality"
export AMASS_INVENTORY="$AMASS_RUN_ROOT/manifests/amass_raw_inventory_eligible.csv"
export GAVD_SEQUENCE_MANIFEST="$GAVD_FULL_ROOT/manifests/gavd_full_sequences.csv"
export GAVD_VIDEO_MANIFEST="$GAVD_FULL_ROOT/manifests/gavd_full_videos.csv"
export GAVD_ANNOTATION_ROOT="$GAVD_FULL_ROOT/annotations"
export GAVD_POSE_ROOT="$GAVD_FULL_ROOT/poses/latent-laterality"
export GAVD_POSE_MANIFEST="$GAVD_POSE_ROOT/full_gavd_pose_split_manifest.csv"

cd "$GAVD6_ROOT"
mkdir -p "$LATENT_LATERALITY_RUN_ROOT"
```

`CODY_JEPA_ROOT`, `BABEL_ROOT`, `GAVD5TM_ROOT`, and the two swap-probe
variables are not inputs to this study. In particular, `GAIT_PARITY_POSE_DIR`
is the older fixed 96-sequence cache and is not a substitute for a full-GAVD
pose manifest.

### Status of `manifests/gavd`

The checked-in GAVD index is internally consistent: it contains 1,874 unique
annotated sequences over 348 unique YouTube videos and 458,116 annotated
frames. `gavd_full_videos.csv` reconciles exactly with the aggregation of
`gavd_full_sequences.csv`.

These three files are enough to audit the corpus and downloaded-video
coverage, but they are **not enough by themselves to extract poses**. They do
not contain the per-frame `bbox` rows from the five original
`GAVD_Clinical_Annotations_[1-5].csv` files, source widths, a frozen
source-video split, actual decoded FPS/aspect ratio, pose paths, or extraction
status. Nine source videos also have inconsistent annotated source heights, so
video geometry must be measured from the downloaded file rather than inferred
from this index.

Before full-GAVD extraction, stage the following under `GAVD_FULL_ROOT`:

```text
$GAVD_FULL_ROOT/
├── annotations/GAVD_Clinical_Annotations_1.csv ... _5.csv
├── manifests/gavd_full_sequences.csv
├── manifests/gavd_full_videos.csv
├── youtube/all/<video_id>.<video-extension>
└── poses/latent-laterality/             # generated, not supplied
```

A standalone full-corpus pose extractor and source-video split builder is
still required to generate `GAVD_POSE_MANIFEST`. Do not point
`GAVD_POSE_MANIFEST` at `gavd_full_sequences.csv`: the schemas are different.
The current foundation notebooks are locked to 96 sequences, and
`scripts/data_preparation/extract_augmented_poses.py` handles only the separate
augmented-normal cohort.

Preflight the currently available inputs:

```bash
test -d "$GAVD6_ROOT"
test -d "$AMASS_EXTRACTED_ROOT"
test -f "$AMASS_INVENTORY"
test -f "$AMASS_SUBJECT_SPLITS"
test -f "$AMASS_BODY_MODEL_ROOT/smplh/male/model.npz"
test -f "$AMASS_BODY_MODEL_ROOT/smplh/female/model.npz"
test -f "$AMASS_BODY_MODEL_ROOT/dmpls/male/model.npz"
test -f "$AMASS_BODY_MODEL_ROOT/dmpls/female/model.npz"
test -f "$GAVD_SEQUENCE_MANIFEST"
test -f "$GAVD_VIDEO_MANIFEST"
```

## Run the study through Slurm

Submit the study from the HAIC login node with the batch scripts in
`slurm/latent-laterality`. Do not run the Python workloads directly on the
login node. Each batch script imports the submitting shell environment, checks
its inputs, changes to `GAVD6_ROOT`, and invokes the appropriate project entry
point inside its allocation.

GPU jobs request one H100. Manifest construction and sequence benchmarks are
CPU-only. Training matrices use Slurm arrays so every model gets an independent
allocation, exit status, and log.

| Job | Runs | HAIC request |
| --- | --- | --- |
| `01-convert-amass-neutral.sbatch` | AMASS neutral Core11 conversion | 1 H100, 8 CPU, 64 GB, 24 h |
| `02-build-amass-gauge-manifest.sbatch` | Seed-7 full-sequence gauge draws | 4 CPU, 16 GB, 4 h |
| `03-run-amass-sequence-benchmark.sbatch` | Real AMASS benchmark gate | 8 CPU, 64 GB, 8 h |
| `04-build-gavd-core11.sbatch` | Full-GAVD pose-to-Core11 adapter | 8 CPU, 64 GB, 8 h |
| `05-source-screen-seed7.sbatch` | Six route/variant runs, array `0-5` | 1 H100 per task, 8 CPU, 64 GB, 24 h |
| `06-evaluate-source-transfer.sbatch` | Frozen validation readout over six checkpoints | 1 H100, 8 CPU, 64 GB, 8 h |
| `07-build-selected-gauge-manifest.sbatch` | Gauge draws for the selected single-source route | 4 CPU, 16 GB, 4 h |
| `08-run-selected-sequence-benchmark.sbatch` | Selected-route benchmark gate | 8 CPU, 64 GB, 8 h |
| `09-train-gauge-seed7.sbatch` | Correction-first, SG-JEPA, uniform; array `0-2` | 1 H100 per task, 8 CPU, 64 GB, 24 h |
| `10-train-gauge-confirmation.sbatch` | Correction-first and SG-JEPA at seeds 19/31; array `0-3` | 1 H100 per task, 8 CPU, 64 GB, 24 h |

The array task mappings are frozen as follows:

| Array | Task IDs |
| --- | --- |
| Source screen | `0` AMASS standard; `1` GAVD standard; `2` staged standard; `3` AMASS equivariant; `4` GAVD equivariant; `5` staged equivariant |
| Decisive seed 7 | `0` correction-first; `1` SG-JEPA; `2` uniform posterior |
| Confirmation | `0` correction-first seed 19; `1` SG-JEPA seed 19; `2` correction-first seed 31; `3` SG-JEPA seed 31 |

Array logs use `%A_%a`, so each task has separate standard-output and error
files keyed by the parent job ID and task ID.

## 1. Build neutral AMASS tensors and run the real gate

The conversion job uses the gauge-neutral traveling-only policy. It rejects a
sequence when forward direction cannot be estimated from pelvis trajectory and
never uses named left/right joints to select or validate orientation.

Submit conversion, gauge-manifest construction, and the benchmark as one
dependency chain:

```bash
cd "$GAVD6_ROOT"
j1=$(sbatch --parsable --export=ALL slurm/latent-laterality/01-convert-amass-neutral.sbatch)
j2=$(sbatch --parsable --export=ALL --dependency="afterok:$j1" slurm/latent-laterality/02-build-amass-gauge-manifest.sbatch)
j3=$(sbatch --parsable --export=ALL --dependency="afterok:$j2" slurm/latent-laterality/03-run-amass-sequence-benchmark.sbatch)
printf 'AMASS jobs: %s -> %s -> %s\n' "$j1" "$j2" "$j3"
```

The jobs produce:

```text
$LATENT_LATERALITY_RUN_ROOT/
├── amass-neutral/
│   ├── core11/
│   ├── manifest.csv
│   ├── rejects.csv
│   └── gauge-seed7.csv
└── amass-benchmark-seed7/
    ├── benchmark_gates.csv
    ├── gate_decision.json
    ├── sequence_metrics.csv
    └── uncertainty_summary.csv
```

The benchmark exits with status 2 when the scientific gate fails, so Slurm
will correctly mark `j3` failed even though it wrote the diagnostic outputs.
Inspect the decision before continuing:

```bash
cat "$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7/gate_decision.json"
```

Continue only when it contains `"ready_for_sg_jepa": true`.

Stop if mask-only path-NLL improvement exceeds 1%, the absolute-chart AUROC
upper 95% bound is at least 0.55, or oracle odd error improves on continuity by
less than 5%. If oracle is within 5% of continuity, continuity is the result;
do not train SG-JEPA.

## 2. Build GAVD Core11 and screen the three data routes

This stage cannot start until the separate full-corpus pose extraction has
produced `GAVD_POSE_MANIFEST`. Verify that prerequisite, then submit the Core11
adapter, the six-task source-screen array, and the frozen readout as a dependency
chain:

```bash
test -f "$GAVD_POSE_MANIFEST"
j4=$(sbatch --parsable --export=ALL slurm/latent-laterality/04-build-gavd-core11.sbatch)
j5=$(sbatch --parsable --export=ALL --dependency="afterok:$j4" slurm/latent-laterality/05-source-screen-seed7.sbatch)
j6=$(sbatch --parsable --export=ALL --dependency="afterok:$j5" slurm/latent-laterality/06-evaluate-source-transfer.sbatch)
printf 'GAVD/source jobs: %s -> %s -> %s\n' "$j4" "$j5" "$j6"
```

Job `05` trains seed 7 for `standard_sjepa` and
`reflection_equivariant` under the AMASS-only, GAVD-only, and AMASS-to-GAVD
routes. It never evaluates test, never reads condition labels, and samples GAVD
training windows uniformly by source video. Job `06` waits for all six array
tasks and passes their best checkpoints to the common frozen validation readout.

The principal outputs are:

```text
$LATENT_LATERALITY_RUN_ROOT/
├── gavd-core11/
│   ├── tensors/
│   ├── manifest.csv
│   └── rejects.csv
├── source-screen/<route>-<variant>-seed7/
└── source-transfer-readout-seed7/
    ├── source_transfer_summary.csv
    ├── source_transfer_predictions.csv
    └── evaluation_contract.json
```

Select one route from source-video-macro odd-orbit/even error and feature
variance in `source_transfer_summary.csv`. Do not use self-KL or GAVD condition
accuracy for selection. If raw coordinates or the random encoder win, report
that result and stop representation expansion.

## 3. Gate the selected route and run the decisive comparison

The current decisive-training entry point is safe for the `amass-only` and
`gavd-only` selections. Do **not** submit a staged `amass-to-gavd` decisive run
yet: correction-first retains a loaded initial model, while the current SG and
uniform branches rebuild that model. The batch arrays reject
`GAUGE_INITIAL_CHECKPOINT` to prevent an asymmetric comparison.

Bind the Core11 inputs for exactly one selected single-source route:

```bash
# AMASS-only selection
export SELECTED_ROUTE_NAME="amass-only"
export SELECTED_SOURCE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
export SELECTED_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"

# Or GAVD-only selection; uncomment all three lines instead of the AMASS block.
# export SELECTED_ROUTE_NAME="gavd-only"
# export SELECTED_SOURCE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv"
# export SELECTED_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors"
```

Submit gauge-manifest construction and the selected-route benchmark gate:

```bash
j7=$(sbatch --parsable --export=ALL slurm/latent-laterality/07-build-selected-gauge-manifest.sbatch)
j8=$(sbatch --parsable --export=ALL --dependency="afterok:$j7" slurm/latent-laterality/08-run-selected-sequence-benchmark.sbatch)
printf 'Selected-route gate jobs: %s -> %s\n' "$j7" "$j8"
```

Inspect the generated decision:

```bash
cat "$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME/gate-seed7/gate_decision.json"
```

Only when it contains `"ready_for_sg_jepa": true`, submit the three seed-7
arms—correction-first, SG-JEPA, and the uniform-posterior control:

```bash
j9=$(sbatch --parsable --export=ALL slurm/latent-laterality/09-train-gauge-seed7.sbatch)
printf 'Decisive seed-7 array: %s\n' "$j9"
```

Review the common frozen odd/even readout. Run seeds 19 and 31 only if seed 7
shows that SG-JEPA improves over correction-first without a clean/even penalty:

```bash
j10=$(sbatch --parsable --export=ALL slurm/latent-laterality/10-train-gauge-confirmation.sbatch)
printf 'Confirmation array: %s\n' "$j10"
```

The confirmation array repeats only `correction_first_sjepa` and `sg_jepa`; it
does not repeat the uniform control or the full source-screen matrix.

The decisive outputs are stored under:

```text
$LATENT_LATERALITY_RUN_ROOT/
├── selected-route/$SELECTED_ROUTE_NAME/
│   ├── gauge-seed7.csv
│   └── gate-seed7/
└── decisive/$SELECTED_ROUTE_NAME/
    ├── correction-first-seed7/
    ├── sg-jepa-seed7/
    ├── uniform-seed7/
    ├── correction-first-seed19/
    ├── sg-jepa-seed19/
    ├── correction-first-seed31/
    └── sg-jepa-seed31/
```

Each training directory returns `run_result.json`, `history.csv`, and its best
checkpoint. Stop in favor of correction-first if SG-JEPA does not improve the
common odd error, if its advantage exists only in self-loss, or if
posterior-minus-MAP error is nonnegative in the high-uncertainty stratum.

## Monitoring and reruns

Use Slurm to inspect pending/running work and accounting results:

```bash
squeue --me
sacct -j "$j1,$j2,$j3" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
```

Standard output and error files are written in the submission directory as
`slurm-ll-*.out` and `slurm-ll-*.err`. Follow an individual log with `tail -f`.

All output guards are conservative: a job refuses to replace a completed
manifest or write into a non-empty experiment directory. The AMASS converter is
the exception because it has its own compatibility checks and safely resumes
valid existing tensors. To rerun another experiment, choose a new run root or
deliberately archive/remove only the exact prior output after reviewing it.
