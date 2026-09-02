# Latent Laterality: ordered HAIC run guide

The local synthetic gate is a code/mechanism check only. The real paired-AMASS
v2 gate has already passed and authorizes the current seed-7 training run. Do
not rerun that gate in place. A fresh run root must run the real sequence gate
and stop unless its `gate_decision.json` sets `ready_for_sg_jepa=true`.

## Inputs and prerequisites

### Required for the current paired-AMASS v2 run

- `GAVD6_ROOT`, `AMASS_EXTRACTED_ROOT`, `AMASS_RUN_ROOT`,
  `AMASS_BODY_MODEL_ROOT`, and `AMASS_SUBJECT_SPLITS` from the HAIC environment
  contract below. The latter three are needed only if script 01 must rebuild
  the neutral AMASS tensors.
- The eligible AMASS inventory under `AMASS_RUN_ROOT/manifests`. Do not use the
  unrestricted inventory: the eligible inventory contains the 201 identities
  covered by the audited subject split.
- Tensor/output roots with enough space.

### Required only for the deferred GAVD/source-route screen

- `GAVD_FULL_ROOT` and a full-GAVD pose manifest with `pose_path`,
  `sequence_id`, `video_id`, `split`, `fps`, `aspect_ratio`, and optional
  `condition`. Every source video must occur in one split only.

The following variables are already part of the HAIC environment:

```bash
export GAVD_FULL_ROOT="/hai/scratch/$USER/datasets/gavd_full"
export CODY_JEPA_ROOT="/hai/scratch/$USER/cody-jepa"
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export GAVD5TM_ROOT="$SJEPA_ROOT/gavd5-tm"
export AMASS_ROOT="$GAVD6_ROOT/data/amass"
export AMASS_EXTRACTED_ROOT="$AMASS_ROOT/extracted"
export AMASS_RUN_ROOT="$GAVD6_ROOT/outputs"
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
variables are not inputs to the current paired-AMASS v2 run. In particular,
`GAIT_PARITY_POSE_DIR` is the older fixed 96-sequence cache and is not a
substitute for a full-GAVD pose manifest.

### AMASS source-rebuild preflight (only if script 01 is needed)

Do not run these checks for the current seed-7 training path unless the shared
neutral AMASS tensors are genuinely absent. They are the prerequisites for a
fresh execution of script 01:

```bash
test -d "$GAVD6_ROOT"
test -d "$AMASS_EXTRACTED_ROOT"
test -f "$AMASS_INVENTORY"
test -f "$AMASS_SUBJECT_SPLITS"
test -f "$AMASS_BODY_MODEL_ROOT/smplh/male/model.npz"
test -f "$AMASS_BODY_MODEL_ROOT/smplh/female/model.npz"
test -f "$AMASS_BODY_MODEL_ROOT/dmpls/male/model.npz"
test -f "$AMASS_BODY_MODEL_ROOT/dmpls/female/model.npz"
```

## Deferred GAVD prerequisites (not required for the current AMASS run)

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

For the deferred GAVD/source-route screen, also verify:

```bash
test -f "$GAVD_SEQUENCE_MANIFEST"
test -f "$GAVD_VIDEO_MANIFEST"
```

## Current execution order: paired AMASS v2

Submit jobs from the HAIC login node with the batch scripts in
`slurm/latent-laterality`. Do not run the Python workloads directly on the
login node. GPU jobs request one H100; manifest construction and the benchmark
gate are CPU-only. Array jobs give each model its own allocation, exit status,
and log.

This is the only current route for the AMASS SG-JEPA experiment:

| Order | Script | Current action | Purpose |
| --- | --- | --- | --- |
| Prerequisite | `01-convert-amass-neutral.sbatch` | **Already complete.** Run only if the neutral AMASS tensors or their manifest are genuinely absent. | Create the shared neutral Core11 tensors. |
| Completed | `11-build-amass-gauge-v2-chart-paired.sbatch` | **Do not resubmit.** | Create the paired v2 corruption manifest. |
| Completed | `12-run-amass-sequence-benchmark-v2.sbatch` | **Passed. Do not resubmit.** | Verify that the paired manifest is fair and leaves meaningful learning headroom. |
| **Run now** | `13-train-amass-gauge-v2-seed7.sbatch` | Submit once as its declared three-task array. | Train correction-first, SG-JEPA, and the uniform-uncertainty control at seed 7. |
| Then | `14-evaluate-amass-gauge-v2-seed7-validation.sbatch` | Submit only after every task in 13 succeeds. | Compare all three seed-7 runs on validation people only. |
| Conditional | `15-train-amass-gauge-v2-confirmation.sbatch` | Submit only after manually reviewing 14. | Repeat correction-first and SG-JEPA at seeds 19 and 31. |
| Last | `16-evaluate-amass-gauge-v2-confirmation-test.sbatch` | Submit only after every task in 15 succeeds. | Read the sealed test split once for all three seeds. |

Scripts `02` and `03` are the failed historical v1 AMASS diagnostic. Scripts
`04` through `10` are a separate, future GAVD/source-route screen. Neither
group is part of the approved AMASS v2 run below.

### 0. Verify the completed v2 artifacts; do not rebuild them

The current v2 manifest and gate already exist and passed. Scripts 11 and 12
intentionally refuse to replace their existing manifest or non-empty output
directory. Rebuilding either one in place would fail, and creating a different
manifest would require a new gate before training.

On HAIC, verify that the stored files are present and still match the passing
gate before requesting GPUs:

```bash
GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
GATE_DECISION="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired/gate_decision.json"

test -f "$GAUGE_MANIFEST"
test -f "$GATE_DECISION"
jq -e '.ready_for_sg_jepa == true' "$GATE_DECISION"
test "$(shasum -a 256 "$GAUGE_MANIFEST" | awk '{print $1}')" = \
  "ffd0b358fadaacaa875aec886caa48c941b00de60e947be89d9aeb898493c125"
```

The passing record also reports `chart_pairing_verified: true`, 5,466 paired
chart views, and 2,733 independent fitting/development source draws. It did
not read the sealed test rows. A gate pass makes the training comparison fair;
it does not predict that SG-JEPA will win.

If these v2 artifacts are genuinely missing in a fresh run root, the rebuild
order is `01` (only if needed) → `11` → `12`. Stop after 12 and inspect its
decision before submitting 13. Do not use that rebuild recipe to overwrite the
current approved artifacts.

### 1. Run the seed-7 development comparison now

Script 13 is a three-task array. Its task mapping is fixed:

| Task | Model arm |
| --- | --- |
| `0` | Correction-first baseline |
| `1` | SG-JEPA |
| `2` | Uniform-uncertainty control |

It rechecks both the gate decision and the manifest fingerprint itself. Submit
it once:

```bash
cd "$GAVD6_ROOT"
j13=$(sbatch --parsable --export=ALL slurm/latent-laterality/13-train-amass-gauge-v2-seed7.sbatch)
printf 'Seed-7 training array: %s\n' "$j13"
```

### 2. Evaluate seed 7 on validation people only

Wait until **all three** tasks in `j13` have succeeded. Then submit 14 with an
`afterok` dependency; it refuses to run if any expected `run_result.json` is
missing. This evaluation uses the validation split, not the test split.

```bash
j14=$(sbatch --parsable --export=ALL --dependency="afterok:$j13" \
  slurm/latent-laterality/14-evaluate-amass-gauge-v2-seed7-validation.sbatch)
printf 'Seed-7 validation readout: %s\n' "$j14"
```

Review `amass-gauge-v2-seed7-validation/gauge_readout_summary.csv` before any
further submission. Advance only if SG-JEPA improves the identity-macro
side-sensitive score over correction-first without a material even-channel
loss, and if the uniform-uncertainty control does not reproduce the gain.

### 3. Confirm only a development-supported result

Script 15 does not automatically interpret the validation result. Do **not**
submit it merely because script 14 completed. Submit it only after the manual
decision in Step 2 supports the pre-set SG-JEPA contrast.

Its four tasks are fixed:

| Task | Model arm and seed |
| --- | --- |
| `0` | Correction-first, seed 19 |
| `1` | SG-JEPA, seed 19 |
| `2` | Correction-first, seed 31 |
| `3` | SG-JEPA, seed 31 |

```bash
j15=$(sbatch --parsable --export=ALL slurm/latent-laterality/15-train-amass-gauge-v2-confirmation.sbatch)
printf 'Confirmation training array: %s\n' "$j15"
```

### 4. Read the sealed test split once

Script 16 is the only step in this path that evaluates AMASS test rows. Wait
for all four tasks in `j15` to succeed, then submit it once. It evaluates the
six required completed runs: correction-first and SG-JEPA at seeds 7, 19, and
31.

```bash
j16=$(sbatch --parsable --export=ALL --dependency="afterok:$j15" \
  slurm/latent-laterality/16-evaluate-amass-gauge-v2-confirmation-test.sbatch)
printf 'Sealed confirmation test: %s\n' "$j16"
```

The final outputs are `gauge_readout_summary.csv`,
`gauge_readout_predictions.csv`, `gauge_path_metrics.csv`, and
`evaluation_contract.json` under `amass-gauge-v2-confirmation-test`. The final
claim requires the same SG-JEPA direction at seeds 7, 19, and 31. It remains an
unanchored controlled-robustness claim: the score does not name an anatomical
left or right side.

Array logs use `%A_%a`, so each task has separate standard-output and error
files keyed by the parent job ID and task ID.

## Historical v1 AMASS diagnostic (do not run for v2)

The conversion job uses the gauge-neutral traveling-only policy. It rejects a
sequence when forward direction cannot be estimated from pelvis trajectory and
never uses named left/right joints to select or validate orientation.

This chain reproduces the v1 diagnostic only. Its recorded gate failed, and it
must not authorize v2 SG-JEPA training. Do not submit scripts 02 or 03 as part
of the current run; keep them only for historical reproduction:

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

The historical decision is `ready_for_sg_jepa: false`. Its output documents
why the v2 paired benchmark was needed; it is not a route to current training.

## Separate GAVD/source-route screen (not part of the current AMASS run)

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

### If the GAVD/source-route screen is intentionally pursued

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
sacct -j "$j13" --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
```

Standard output and error files are written in the submission directory as
`slurm-ll-*.out` and `slurm-ll-*.err`. Follow an individual log with `tail -f`.

All output guards are conservative: a job refuses to replace a completed
manifest or write into a non-empty experiment directory. The AMASS converter is
the exception because it has its own compatibility checks and safely resumes
valid existing tensors. To repeat an experiment, use a new run root and retain
the approved manifest and gate as an immutable record.
