# Stage 2 tutorial: BABEL-guided walking continuation

Stage 2 starts from each successful Stage 1 checkpoint and continues the same self-supervised JEPA training on clean walking windows.

BABEL determines where these windows come from. It is not a model input, training target, or clinical label.

> **Current boundary.** The repository does not yet contain the production BABEL manifest builder or AMASS continuation trainer used in the example commands below. They are explicit target interfaces for the implementation.

## 1. Start only after Stage 1 is frozen

For every Stage 2 run, identify exactly one Stage 1 parent with the same:

- variant;
- seed;
- matching regime;
- skeleton schema;
- architecture configuration.

Example parent table:

```text
variant,seed,matching_regime,parent_checkpoint,parent_state_sha256
standard,7,exposure,.../stage1/standard/seed-7/checkpoint.pt,...
paired_unconstrained,7,exposure,.../stage1/paired_unconstrained/seed-7/checkpoint.pt,...
reflection_equivariant,7,exposure,.../stage1/reflection_equivariant/seed-7/checkpoint.pt,...
```

Do not select “the best Stage 1 seed” separately for each variant. Seeds are paired experimental units.

## 2. Locate BABEL on HAIC

Set the BABEL release path:

```bash
export BABEL_ROOT=/hai/scratch/YOUR_SUNETID/babel/babel_v1.0_release
```

Confirm the dense files:

```bash
for name in train val test; do
  test -f "$BABEL_ROOT/${name}.json"
done
```

Use `train.json`, `val.json`, and `test.json` as annotation sources only if GaitParity will not later claim an untouched BABEL action-recognition test. Record that choice. Keep `extra_train.json` and `extra_val.json` outside the primary build until a multiple-annotator consensus policy is frozen.

## 3. Match BABEL records to the frozen AMASS inventory

Join BABEL's `feat_p` to the inventory's `relative_path` using:

1. exact normalized path;
2. unique suffix match when exact matching fails;
3. manual review for everything else.

Never join automatically by filename alone. The current inventory contains thousands of repeated filenames.

Expected command:

```bash
uv run python scripts/build_babel_walking_manifests.py match \
  --babel-root "$BABEL_ROOT" \
  --amass-inventory "$AMASS_RUN_ROOT/manifests/amass_raw_inventory.csv" \
  --output "$AMASS_RUN_ROOT/manifests/babel_amass_matches.csv" \
  --rejects "$AMASS_RUN_ROOT/manifests/babel_amass_match_rejects.csv"
```

Require exact provenance columns including BABEL sequence ID, release split, `feat_p`, matched AMASS path, match method, source dataset, subject, and motion ID.

## 4. Extract walking without losing overlapping actions

Use BABEL's normalized `act_cat` values. A walking interval is one whose category list contains `walk`.

Do not search only the raw text for the word “walk.” A description such as “pacing” may map to the normalized walk category.

For a sequence with frame-level annotations, use those intervals. Use a sequence-level label only when no frame-level interval set is available, and mark it `sequence_level_assumed`.

Because BABEL permits overlapping actions, split the timeline into atomic intervals whose active category set is constant. For example:

```text
walk:   1.0–8.0 seconds
carry:  3.0–6.0 seconds

becomes
1.0–3.0  walk
3.0–6.0  walk + carry
6.0–8.0  walk
```

Assign tiers:

- `clean_walk`: walk is active with no incompatible action;
- `walk_with_concurrent_action`: walk plus a compatible concurrent action;
- `transition_or_conflict`: walk overlaps standing, running, jumping, sitting, or a transition;
- `not_walk`: walk is absent.

Expected command:

```bash
uv run python scripts/build_babel_walking_manifests.py segments \
  --matches "$AMASS_RUN_ROOT/manifests/babel_amass_matches.csv" \
  --babel-root "$BABEL_ROOT" \
  --primary-tier clean_walk \
  --output "$AMASS_RUN_ROOT/manifests/babel_walking_segments.csv"
```

Keep all tiers in the manifest even though the primary Stage 2 corpus uses only `clean_walk`.

## 5. Inherit the Stage 1 subject split

Stage 2 must not create a new random split. Join every walking segment to:

```text
$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv
```

The BABEL release split is annotation provenance, not the model split.

Required assertions:

```python
assert walking_segments.groupby("identity")["split"].nunique().max() == 1
assert set(walking_segments["identity"]) <= set(subject_splits["identity"])
```

Any segment whose identity cannot be recovered from the frozen Stage 1 registry must be excluded and reported, not assigned a new split.

## 6. Cut walking windows from the same converted coordinates

Reuse the exact converted core-11 sequences from Stage 1. Do not rerun forward kinematics with different body models, mappings, or frame conventions.

Freeze a Stage 2 window policy:

```yaml
source_segments: babel_walking_segments.csv
primary_tier: clean_walk
annotation_level: frame
boundary_margin_seconds: 0.25
canonical_fps: 30
window_frames: 64
stride_frames: 32
required_walk_coverage: 1.0
```

The 0.25-second margin keeps strict windows away from uncertain action boundaries. At 30 Hz, a 64-frame window lasts about 2.13 seconds, so a segment must be at least 2.63 seconds long to yield one strict window.

Expected command:

```bash
uv run python scripts/prepare_amass_windows.py make-walking-windows \
  --converted-root "$AMASS_RUN_ROOT/core11" \
  --segments "$AMASS_RUN_ROOT/manifests/babel_walking_segments.csv" \
  --splits "$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv" \
  --tier clean_walk \
  --annotation-level frame \
  --boundary-margin-seconds 0.25 \
  --frames 64 \
  --stride 32 \
  --required-walk-coverage 1.0 \
  --output "$AMASS_RUN_ROOT/manifests/stage2_walking_windows.csv"
```

The Stage 2 manifest must keep `segment_id`, BABEL provenance, identity, model split, active categories, tensor path, and source checksum.

## 7. Apply kinematic quality control

BABEL says that an action is walking; it does not prove that the resulting skeleton window is usable.

Hard-fail windows with:

- missing or changed source files;
- nonfinite coordinates;
- the wrong tensor shape;
- timestamps outside the source motion;
- failed core-11 conversion;
- severe joint discontinuity;
- a split different from the parent sequence.

Soft-flag rather than automatically reject:

- walking in place;
- substantial turning;
- weak detectable leg alternation;
- unusually low or high pelvis translation;
- extreme left/right motion-energy ratios;
- insufficient evidence for one complete cycle.

Also compare each BABEL segment's pelvis-travel direction with the full AMASS
sequence's saved forward axis. Flag weak or negative alignment. A mixed-action
file can contain a clean walking interval whose direction differs from the
file-level direction used during Stage 1 conversion. Use the saved
`coordinates_m`, `pelvis_world_m`, and inverse transform for this audit; do not
rerun SMPL+H forward kinematics or silently give Stage 2 a different skeleton
mapping.

Expected command:

```bash
uv run python scripts/prepare_amass_windows.py gait-qc \
  --window-manifest "$AMASS_RUN_ROOT/manifests/stage2_walking_windows.csv" \
  --output "$AMASS_RUN_ROOT/manifests/stage2_walking_window_qc.csv" \
  --report "$AMASS_RUN_ROOT/reports/stage2_walking_qc.json"
```

The primary training manifest should select hard-QC passes and the frozen soft-QC policy. Preserve every failed row with a reason.

Visually inspect original/mirror pairs from clean, concurrent, transition, turning, walking-in-place, suffix-match, and QC-failure groups.

## 8. Prove that BABEL is absent from the model batch

The dataset loader may use BABEL columns to locate a tensor, but its returned training item should contain only model inputs and provenance needed for auditing:

```python
def __getitem__(self, index):
    row = self.windows.iloc[index]
    coordinates = load_tensor(row.tensor_path)
    return {
        "coordinates": coordinates,
        "valid_tokens": build_valid_tokens(coordinates),
        "window_id": row.window_id,
    }
```

Add a test that rejects action fields:

```python
FORBIDDEN_MODEL_FIELDS = {
    "raw_label",
    "proc_label",
    "act_cat",
    "walk_tier",
    "babel_sid",
}

assert not (FORBIDDEN_MODEL_FIELDS & set(training_batch))
```

Provenance can remain in the CSV and training manifest without becoming an input tensor.

## 9. Freeze the Stage 2 configuration

Stage 2 should preserve the architecture and objective while giving the continuation phase its own learning schedule.

Example:

```yaml
study: gait-parity
stage: stage2_babel_walking
schema: core11-v1
window_manifest: /hai/scratch/USER/gait-parity/amass-v1/manifests/stage2_walking_windows.csv
parent_stage: stage1_broad_amass
variants:
  - standard
  - paired_unconstrained
  - reflection_equivariant
seeds: [7, 17, 29]
matching_regime: exposure
load_from_parent:
  student_encoder: true
  target_encoder: true
  predictor: true
  projector: true
  model_buffers: true
reset_for_stage2:
  optimizer: true
  learning_rate_scheduler: true
optimizer: adamw
learning_rate: 0.00005
weight_decay: 0.05
mask_fraction: 0.60
vicreg_weight: 0.05
odd_vicreg_weight: 1.0
device: cuda
precision: amp-fp16
```

The lower example learning rate reflects continuation rather than training from scratch. Treat it as a starting value to freeze through nonclinical development, not as an already validated optimum.

## 10. Verify parent compatibility before training

The trainer must stop unless parent and child agree on:

```text
variant
seed
matching regime
joint schema and ordering
coordinate-frame version
window length
time-patch length
embedding width
encoder and predictor depth
mirror operator version
```

It must also verify the parent checkpoint hash listed in the Stage 2 run manifest.

Expected smoke command:

```bash
uv run python scripts/train_gait_parity_amass.py \
  --config configs/gait_parity/stage2_babel_walking.yaml \
  --variant reflection_equivariant \
  --seed 7 \
  --matching-regime exposure \
  --parent-checkpoint /absolute/path/to/stage1/checkpoint.pt \
  --max-updates 20 \
  --run-root "$GAITPARITY_RUN_ROOT/stage2-smoke"
```

Confirm in the saved manifest that the optimizer was newly created while the student, teacher, predictor, projector, and model buffers came from the declared parent.

## 11. Submit Stage 2 on HAIC

Create `stage2_one_run.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gp-s2
#SBATCH --partition=hai
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=/hai/scratch/YOUR_SUNETID/gait-parity/logs/gp-s2-%j.out
#SBATCH --error=/hai/scratch/YOUR_SUNETID/gait-parity/logs/gp-s2-%j.err

set -euo pipefail

: "${HAIC_ACCOUNT:?Submit with an approved HAIC account}"
: "${GAITPARITY_REPO:?Set GAITPARITY_REPO}"
: "${GAITPARITY_RUN_ROOT:?Set GAITPARITY_RUN_ROOT}"
: "${GP_VARIANT:?Set GP_VARIANT}"
: "${GP_SEED:?Set GP_SEED}"
: "${GP_PARENT_CHECKPOINT:?Set the exact Stage 1 parent}"

cd "$GAITPARITY_REPO/gavd6"
export GAIT_PARITY_DEVICE=cuda
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

nvidia-smi

uv run python scripts/train_gait_parity_amass.py \
  --config configs/gait_parity/stage2_babel_walking.yaml \
  --variant "$GP_VARIANT" \
  --seed "$GP_SEED" \
  --matching-regime exposure \
  --parent-checkpoint "$GP_PARENT_CHECKPOINT" \
  --run-root "$GAITPARITY_RUN_ROOT/stage2"
```

Submit only after the corresponding Stage 1 checkpoint passes its audit:

```bash
export GP_VARIANT=reflection_equivariant
export GP_SEED=7
export GP_PARENT_CHECKPOINT=/absolute/path/to/stage1/reflection_equivariant/seed-7/checkpoint.pt

sbatch \
  --account="$HAIC_ACCOUNT" \
  --export=ALL \
  stage2_one_run.sbatch
```

If Stage 1 and Stage 2 are submitted together, use a Slurm `afterok` dependency so Stage 2 cannot run after a failed parent job:

```bash
STAGE1_JOB_ID=$(sbatch --parsable --account="$HAIC_ACCOUNT" --export=ALL stage1_one_run.sbatch)
sbatch \
  --account="$HAIC_ACCOUNT" \
  --dependency="afterok:${STAGE1_JOB_ID}" \
  --export=ALL \
  stage2_one_run.sbatch
```

Resolving the correct parent path after Stage 1 completes should be done through its saved manifest, not by guessing the newest file.

## 12. Audit Stage 2 and compare it with Stage 1

Expected audit command:

```bash
uv run python scripts/audit_gait_parity_pretraining.py \
  --stage stage2 \
  --run-root "$GAITPARITY_RUN_ROOT/stage2" \
  --window-manifest "$AMASS_RUN_ROOT/manifests/stage2_walking_windows.csv" \
  --compare-parent-stage "$GAITPARITY_RUN_ROOT/stage1" \
  --output "$GAITPARITY_RUN_ROOT/stage2/audit"
```

Report for Stage 1 and Stage 2:

- held-out masked-prediction loss on the same frozen AMASS validation sets;
- representation variance and effective rank;
- mean pairwise cosine similarity;
- even- and odd-channel energy;
- commutation residuals at every claimed equivariant layer;
- response to controlled unilateral attenuation;
- windows, updates, masked tokens, wall time, GPU type, and memory;
- parent-child parameter distance and checkpoint hashes.

Do not choose the Stage 2 stopping point by looking at clinical force performance.

## 13. Preserve two validation views

Use both:

1. a frozen broad-motion AMASS validation set, to detect catastrophic forgetting;
2. a frozen walking-only AMASS validation set, to measure adaptation to gait.

The expected pattern is not “walking improves and broad must stay identical.” Some tradeoff is possible. The purpose is to measure it rather than hide it.

## 14. Freeze checkpoints for downstream work

For each variant and seed, save:

```text
stage1_checkpoint_path
stage1_state_sha256
stage2_checkpoint_path
stage2_state_sha256
stage2_parent_state_sha256
data_manifest_sha256
subject_split_sha256
configuration_sha256
mirror_operator_version
schema_version
training_status
health_gate_status
```

Downstream clinical analyses must request an exact checkpoint or manifest. They must not search for a file named `latest.pt`.

## 15. Stage 2 completion checklist

```text
[ ] BABEL JSON checksums recorded
[ ] Path join audited; no basename-only automatic matches
[ ] Overlapping action intervals preserved
[ ] Walking tiers frozen before clinical results
[ ] Stage 1 subject split inherited exactly
[ ] Core-11 converted coordinates reused unchanged
[ ] Boundary and gait QC reports saved
[ ] BABEL fields absent from model batches and loss
[ ] Each child uses the matching variant/seed parent
[ ] New optimizer and scheduler recorded
[ ] Broad and walking validation views reported
[ ] Health, odd-channel, and commutation gates passed
[ ] Exact Stage 2 checkpoints frozen for readout studies
```

Return to the [two-stage overview](./05-two-stage-amass-training-overview.md) for interpretation and comparison rules.
