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
from this index. The summary names the five annotation parts but does not bind
their hashes, so record the SHA-256 of each source annotation file before
extraction.

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

The missing extraction step must write one `sequence` array of shape
`[frames, 33, 4]` per annotated sequence, using full-frame MediaPipe
`[x, y, z, visibility]` coordinates. It must also probe the decoded video's
FPS and width/height, freeze `train`/`validation`/`test` at the `video_id`
level, and write paths relative to the pose manifest's own directory. A valid
output layout is:

```text
$GAVD_POSE_ROOT/
├── full_gavd_pose_split_manifest.csv
└── sequences/<sequence_id>.npz
```

with manifest rows such as:

```csv
pose_path,sequence_id,video_id,split,fps,aspect_ratio,condition
sequences/cljan9b4p00043n6ligceanyp.npz,cljan9b4p00043n6ligceanyp,B5hrxKe2nP8,train,30.0,1.7777778,parkinsons
```

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

## 1. Build neutral tensors and run the benchmark gate

The converter's traveling-only policy rejects sequences whose forward direction
cannot be estimated from pelvis trajectory. It never uses named left/right
joints to choose or validate orientation.

```bash
uv run --no-sync python scripts/convert_amass_core11.py \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$AMASS_INVENTORY" \
  --subject-splits "$AMASS_SUBJECT_SPLITS" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --output-root "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11" \
  --output-manifest "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv" \
  --rejects "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/rejects.csv" \
  --forward-policy gauge-neutral-travel \
  --allow-rejects \
  --device cuda

uv run --no-sync python scripts/build_sequence_gauge_manifest.py \
  --source-manifest "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv" \
  --tensor-root "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11" \
  --output-manifest "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7.csv" \
  --draws 1 --seed 7

uv run --no-sync python scripts/run_sequence_benchmark.py \
  --gauge-manifest "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7.csv" \
  --tensor-root "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11" \
  --seed 7 \
  --output-dir "$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7"
```

Return `benchmark_gates.csv`, `gate_decision.json`, `sequence_metrics.csv`, and
`uncertainty_summary.csv`.

Stop if mask-only path-NLL improvement exceeds 1%, the absolute-chart AUROC
upper 95% bound is at least 0.55, or oracle odd error improves on continuity by
less than 5%. If oracle is within 5% of continuity, continuity is the result;
do not train SG-JEPA.

## 2. Build full-GAVD Core11 and screen the three data routes

This stage is blocked until the full-corpus extraction step described above
has produced the pose manifest:

```bash
test -f "$GAVD_POSE_MANIFEST"
```

```bash
uv run --no-sync python scripts/build_gavd_core11.py \
  --pose-manifest "$GAVD_POSE_MANIFEST" \
  --output-root "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors" \
  --output-manifest "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv" \
  --rejects "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/rejects.csv"
```

Run seed 7 for `standard_sjepa` and `reflection_equivariant` under each route.
The trainer never evaluates test, never reads condition labels, and samples
GAVD training windows uniformly by source video.

```bash
for variant in standard_sjepa reflection_equivariant; do
  uv run --no-sync python scripts/train_latent_laterality.py \
    --route amass-only --variant "$variant" --seed 7 \
    --amass-manifest "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv" \
    --amass-tensor-root "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11" \
    --output-dir "$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-only-$variant-seed7"

  uv run --no-sync python scripts/train_latent_laterality.py \
    --route gavd-only --variant "$variant" --seed 7 \
    --gavd-manifest "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv" \
    --gavd-tensor-root "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors" \
    --output-dir "$LATENT_LATERALITY_RUN_ROOT/source-screen/gavd-only-$variant-seed7"

  uv run --no-sync python scripts/train_latent_laterality.py \
    --route amass-to-gavd --variant "$variant" --seed 7 \
    --amass-manifest "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv" \
    --amass-tensor-root "$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11" \
    --gavd-manifest "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv" \
    --gavd-tensor-root "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors" \
    --output-dir "$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-to-gavd-$variant-seed7"
done
```

Pass the six final `*_best.pt` paths to the common frozen readout:

```bash
uv run --no-sync python scripts/evaluate_source_transfer.py \
  --gavd-manifest "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv" \
  --gavd-tensor-root "$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors" \
  --checkpoint "amass-standard=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-only-standard_sjepa-seed7/stage-amass/seed-7_standard_sjepa_best.pt" \
  --checkpoint "amass-equivariant=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-only-reflection_equivariant-seed7/stage-amass/seed-7_reflection_equivariant_best.pt" \
  --checkpoint "gavd-standard=$LATENT_LATERALITY_RUN_ROOT/source-screen/gavd-only-standard_sjepa-seed7/stage-gavd/seed-7_standard_sjepa_best.pt" \
  --checkpoint "gavd-equivariant=$LATENT_LATERALITY_RUN_ROOT/source-screen/gavd-only-reflection_equivariant-seed7/stage-gavd/seed-7_reflection_equivariant_best.pt" \
  --checkpoint "staged-standard=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-to-gavd-standard_sjepa-seed7/stage-gavd/seed-7_standard_sjepa_best.pt" \
  --checkpoint "staged-equivariant=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-to-gavd-reflection_equivariant-seed7/stage-gavd/seed-7_reflection_equivariant_best.pt" \
  --output-dir "$LATENT_LATERALITY_RUN_ROOT/source-transfer-readout-seed7"
```

Return `source_transfer_summary.csv` and `source_transfer_predictions.csv`.
Select one data route from source-video-macro odd-orbit/even error and feature
variance. Do not use self-KL or GAVD condition accuracy to select it. If raw or
random features win, report that and stop representation expansion.

## 3. Run only the decisive structured comparison

Build a gauge manifest for the selected route with
`scripts/build_sequence_gauge_manifest.py`, run its benchmark gate, and require
that gate to pass. Then compare correction-first with SG-JEPA. For a staged
route, run AMASS first and pass its checkpoint as `--initial-checkpoint` during
the GAVD adaptation stage.

After selecting the route, bind its generated artifacts rather than using
unresolved `/path/to` placeholders:

```bash
export SELECTED_GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/selected-route/gauge.csv"
export SELECTED_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/tensors"
export SELECTED_GATE_DECISION="$LATENT_LATERALITY_RUN_ROOT/selected-route/gate/gate_decision.json"
```

```bash
uv run --no-sync python scripts/train_gauge_jepa.py \
  --arm correction_first_sjepa \
  --gate-decision "$SELECTED_GATE_DECISION" \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed 7 --output-dir "$LATENT_LATERALITY_RUN_ROOT/decisive/correction-first-seed7"

uv run --no-sync python scripts/train_gauge_jepa.py \
  --arm sg_jepa \
  --gate-decision "$SELECTED_GATE_DECISION" \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed 7 --output-dir "$LATENT_LATERALITY_RUN_ROOT/decisive/sg-jepa-seed7"

uv run --no-sync python scripts/train_gauge_jepa.py \
  --arm uniform_posterior \
  --gate-decision "$SELECTED_GATE_DECISION" \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed 7 --output-dir "$LATENT_LATERALITY_RUN_ROOT/decisive/uniform-seed7"
```

Run seeds 19 and 31 only if seed 7 shows SG-JEPA improves the common frozen
odd/even readout over correction-first without a clean/even penalty. Repeat
only `correction_first_sjepa` and `sg_jepa`; do not repeat the full matrix.

Return `run_result.json`, `history.csv`, the selected checkpoints, and the
common source-transfer/readout tables for the two finalists. Stop in favor of
correction-first if SG-JEPA does not improve common odd error, if its advantage
exists only in self-loss, or if posterior-minus-MAP error is nonnegative in the
high-uncertainty stratum.
