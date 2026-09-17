# Running the temporal gait study

This guide describes the implemented local software interfaces and the HAIC execution procedure. **No real GAVD result or HAIC runtime measurement is supplied by these instructions.** CPU fixtures check mechanics. The current real-data milestone consumes full source videos plus explicitly supplied complete-bout pose exports; it does not discover videos, download data, or run a hidden pose extractor. E3 dense/deep supervision and frozen RGB-video transfer remain gated extensions.

## 1. Prepare the environment and exact inputs

Run from the `gavd6` checkout with its existing Python environment. The local verified interpreter is Python 3.12.10 with Torch 2.13.0 on CPU; this is not a verification of the HAIC CUDA stack. HAIC GPU jobs need the project's compatible CUDA/PyTorch environment, notebook dependencies, NumPy/Pandas/SciPy/scikit-learn and `ffprobe` on `PATH`. Record versions with each run. No package installation or checkpoint download occurs inside the launchers.

The original full-video presentation timestamps are checked with `ffprobe`, including variable frame rate. A provided pose archive must preserve **every original frame in its complete walking bout**, representing failed detections with observation masks. A sparsely sampled or bidirectionally smoothed archive cannot be silently relabelled as a complete causal observation stream. See the exact [input schemas](#7-input-manifest-and-pose-schema).

Start with [pilot.example.json](../../../../slurm/temporal-gait/pilot.example.json). It keeps width 96, encoder depth 4, predictor depth 2, four heads, patch size 4 and no added clock channels for the first timing/support contrast. The matching recipe uses 1,200 updates, batch size 20 and fixed EMA 0.999. `masked_index` and `masked` apply different preparation to the same raw windows; this is **not** an exact replay of the historical whole-clip paper result.

The initial template sets `cohort_scope: "historical_overlap"`. Training and evaluation use only allowed recordings explicitly marked `historical_laterality_member: true` in the supplied exposure audit. Preparation still indexes all allowed complete bouts. This isolates the timing/support mechanism from expanded training-source exposure; it does not claim full-cohort training. No historical list of 93 IDs is imported automatically, and the user must supply audited membership.

After inspecting that contrast, [full-cohort.example.json](../../../../slurm/temporal-gait/full-cohort.example.json) changes only the cohort scope to `full_allowed`: all allowed complete-bout sources in each role can contribute. Use a **new run root**, retain the same role assignments for common sources, and start at `pilot` again. The [confirmatory template](../../../../slurm/temporal-gait/confirmatory.example.json) also has full-allowed scope and the complete five-seed grid; its name is not permission to skip inventory, development or selection gates. Revised clocks, patch size 2 or schedules are later named comparisons, each in a separate run root. A patch-2 or clock-channel configuration must remove `masked_index`, since that arm is restricted to the initial coordinate-only, patch-4 control.

Set exact paths; the examples below are placeholders, not discovered HAIC locations:

```bash
cd /ABSOLUTE/PATH/TO/gavd6
export GAVD6_ROOT="$PWD"
export TG_PYTHON="$GAVD6_ROOT/.venv/bin/python"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TG_CONFIG="$GAVD6_ROOT/slurm/temporal-gait/pilot.example.json"
export TG_RUN_ROOT="/ABSOLUTE/HAIC/OUTPUT/temporal-gait/pilot-01"
export TG_VIDEO_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/full-videos.json"
export TG_SEQUENCE_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/complete-walking-bouts.json"
export TG_POSE_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/complete-bout-poses.json"
export TG_IDENTITY_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/verified-links.json"
export TG_EXPOSURE_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/exposure.json"
export TG_RESERVATION_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/reservations.json"
export TG_SPLIT_MANIFEST="/ABSOLUTE/HAIC/MANIFESTS/source-roles.json"
```

All seven manifest files are explicit versioned JSON. An empty verified-links list is permitted with the required provenance attestation; missing identity, exposure or reservation files are not permission to assume independent fresh sources. Each real source needs an explicit role and exposure/reservation record.

Empty path values in the shipped examples are filled by these exports. A nonempty JSON path and its `TG_*` override must resolve to the same canonical path; other settings must agree exactly. Contradictory settings fail. Mode is `real` or `synthetic`; precision is `bfloat16` for explicitly selected CUDA or `float32`. The new parser uses `TG_RESERVATION_MANIFEST`, not the motion-preservation study's `MP_GAVD_RESERVATION` or the early planning shorthand `TG_RESERVATION`.

## 2. Inspect before starting a development run

These checks construct plans without reading input manifests or media:

```bash
"$TG_PYTHON" scripts/research_directions/temporal_gait/run_stage.py \
  --config "$TG_CONFIG" --phase pilot --stage inventory --dry-run
bash slurm/temporal-gait/submit.sh pilot --dry-run
```

Check the resolved config, global task IDs and scheduler dependencies. Dry-run `DRY_*` identifiers illustrate dependencies and are not valid Slurm IDs. Dry runs do not check media availability, scientific eligibility or the saved expansion decision.

Run a metadata inventory as a retained notebook, explicitly enabling the stage:

```bash
"$TG_PYTHON" scripts/research_directions/temporal_gait/execute_notebook.py \
  --notebook 00 --stage inventory --phase pilot --run-root "$TG_RUN_ROOT" \
  --config "$TG_CONFIG" --execute \
  --output-dir "$TG_RUN_ROOT/notebook_runs/inventory-preview"
```

Without `--execute` or `TG_EXECUTE=1`, the notebook prints its plan only. Inventory validates explicit schemas, exposure and split identities; source videos/poses remain unopened at this stage. Check every excluded source and the declared seconds before launching preparation.

```bash
bash slurm/temporal-gait/submit.sh pilot
```

Pilot submission freezes the **complete** arm×seed mapping in `manifests/task-grid.json`, then selects `pilot_seeds` only. Global task IDs remain stable across phases; do not assume an array index means the same arm after editing `arms` or `seeds`.

```text
inventory → prepare → audit → masked array ─┐
                          └→ future array ──┴→ evaluate development
```

Every edge is `afterok`. Evaluation depends on the full IDs of both arrays, not one task. `--kill-on-invalid-dep=yes` prevents stranded invalid dependencies. A failed notebook exits nonzero and stops dependent work. A successful scheduler exit does not establish a useful scientific result; package gates inspect the saved measurement and development decisions.

## 3. Stages, phases and resource limits

| Command | Work and boundary |
| --- | --- |
| `inventory` | Freeze explicit metadata, role/exposure and configuration contracts |
| `prepare` | Check original PTS and complete pose caches; prepare train/development windows |
| `audit` | Observation support and simple forecasting/measurement controls |
| `masked` | `masked_index` and `masked` tasks for the selected phase |
| `future` | Future, wrong-source and wrong-time tasks for the selected phase |
| `evaluate` | Matched train-fit readouts and complete development comparison |
| `develop` | Expand promising settings to configured development seeds after a positive decision |
| `confirm` | Run the configured finalist seed replication after a positive decision; no automatic test |
| `calibrate` | Freeze selected analysis; uncertainty fitting is not implemented in this milestone |
| `test` | Explicit locked test evaluation; no training or selection |
| `aggregate` | Read retained predictions and produce the evidence summary |
| `extensions`, `cache-video` | Return the explicit unsupported/gated state for this milestone |

For example:

```bash
bash slurm/temporal-gait/submit.sh develop --dry-run
bash slurm/temporal-gait/submit.sh develop
bash slurm/temporal-gait/submit.sh evaluate --phase develop --dry-run

# Final stages below apply ONLY to a full_allowed run with a positive confirm decision.
# historical_overlap is development-only and rejects final locking/test access.
# Inspect the selected comparison; finish confirm before calibrate, then calibrate before test.
bash slurm/temporal-gait/submit.sh confirm --dry-run
bash slurm/temporal-gait/submit.sh calibrate --phase confirm --dry-run
bash slurm/temporal-gait/submit.sh test --phase confirm --dry-run
```

Dropping `--dry-run` submits the corresponding work. `develop` and `confirm` require `decisions/development.json` with `ready_for_expansion: true`; a stop decision prevents submission before a task grid is written. The default final seed list is 42–46; the development list is 42–44. A new seed is another optimization realization, not another person. These phases expand seeds within the same frozen scientific configuration; they do not change cohort scope. Historical-overlap runs are always development-only: no historical member can be certified as untouched test. Use the separate full-allowed run for final locking/test, and retain development-only status if that run has no untouched eligible test sources. The `calibrate` stage currently writes the immutable selection/analysis lock and records `not_required_no_uncertainty_model`; it neither fits uncertainty nor opens calibration media.

The distinct cohort-expansion setup is explicit:

```bash
# Only after reviewing the matched historical-overlap contrast; keep manifest roles fixed.
export TG_CONFIG="$GAVD6_ROOT/slurm/temporal-gait/full-cohort.example.json"
export TG_RUN_ROOT="/ABSOLUTE/HAIC/OUTPUT/temporal-gait/full-cohort-01"
bash slurm/temporal-gait/submit.sh pilot --dry-run
```

Inspect this new plan before omitting `--dry-run`. Do not point a full-cohort configuration at the historical-overlap output directory.

Independent stage submissions can wait for existing jobs:

```bash
TG_DEPENDENCY=123456 bash slurm/temporal-gait/submit.sh prepare
TG_DEPENDENCY=afterok:123456:123457 bash slurm/temporal-gait/submit.sh evaluate
```

| Stage | Scheduler limit |
| --- | --- |
| Inventory | 4 CPU, 16 GB, 30 minutes |
| Complete-bout preparation | 8 CPU, 64 GB, 12 hours; no GPU for the supplied-pose path |
| Information audit | 1 H100, 8 CPU, 32 GB, 2 hours; direct-MLP baseline fits on the configured CUDA device |
| Each masked/future training task | 1 H100, 8 CPU, 64 GB, 12 hours |
| Development/test evaluation | 1 H100, 8 CPU, 48 GB, 8 hours |
| Lock/calibration or aggregation | 8 CPU, 32 GB, 2 hours |

These are allocation limits, **not measured runtime estimates**. Full-source hashing and `ffprobe` are real work even with precomputed poses. Measure throughput, wall time, peak memory, physical walking minutes, processed windows and GPU hours before expanding. `TG_ARRAY_CONCURRENCY=2` limits each training array; two simultaneous arrays can therefore occupy four GPUs. Set it to 1 for up to two concurrent training GPUs, or submit individual families sequentially with dependencies. Training is one GPU per process, not DDP.

Optional overrides: `TG_ACCOUNT`, `TG_PARTITION`, `TG_ARRAY_CONCURRENCY`, `TG_DEPENDENCY`, `TG_TORCH_THREADS`, `TG_NOTEBOOK_OUTPUT_DIR`. The thread limit is capped at `SLURM_CPUS_PER_TASK`. `common.sh` defaults the CUDA determinism workspace to `:4096:8` and rejects unsupported workspace values; retain this export for direct CLI execution too. Neither scheduler settings nor `TG_PHASE` changes the frozen scientific configuration.

## 4. Logs, outputs and resume

Every actual submission appends phase, stage, job ID, dependencies, task IDs, the exact scheduler command and the explicit resume checkpoint (empty for a fresh invocation) to `logs/submissions.tsv`. Stdout/stderr filenames include `%A_%a`, preserving array task identity. Executed notebooks normally land in `notebook_runs/run-NN-PHASE-STAGE[-task-ID]/`; they save after cells and on failure.

Core artifacts include `config/resolved.json`, `config/identity.json`, the immutable task grid, prepared window archives and records, per-condition checkpoints, per-window/horizon predictions, source summaries, `decisions/development.json`, locks and `receipts/*.json`. Follow the paths returned by the package stage result; those records are the authoritative location map.

Resume requires matching scientific configuration, explicit manifest hashes, relevant source code and runtime identity. `resume_from` is an invocation setting: the immutable scientific configuration and complete task grid canonicalize it to `null`. Changing a checkpoint path therefore does not change the experiment, but changing the learning rate, schedule, cohort, seed grid or other scientific setting still fails. A notebook retry is a new executed copy, not permission to change the science inputs.

For an interrupted task, supply the **exact existing checkpoint belonging to that task**, not a directory or a guessed latest-file search. The examples below assume the frozen global task ID `10`; select the actual ID and checkpoint from your retained task artifacts:

```bash
export TG_NOTEBOOK_OUTPUT_DIR="$TG_RUN_ROOT/notebook_runs/future-task-retry-02"
export TG_RESUME_FROM="$TG_RUN_ROOT/training/task-0010/checkpoint-000300.pt"
bash slurm/temporal-gait/submit.sh future --phase pilot --task-id 10 --dry-run
# Inspect the one-task plan, then submit this single-task array only.
bash slurm/temporal-gait/submit.sh future --phase pilot --task-id 10
unset TG_NOTEBOOK_OUTPUT_DIR TG_RESUME_FROM
```

`TG_RESUME_FROM` fills an empty/null JSON `resume_from`; a contradictory nonempty value fails. It never requests a fresh model, changes the total update budget, or permits fitting on another source role. The scheduler rejects a resume attached to a phase DAG or a family-wide array without `--task-id`; one checkpoint must never be shared across several tasks. The trainer verifies the task/config/data/runtime signature and restores model, EMA teacher, optimizer and random-generator states. If using an already allocated single GPU directly, `run_stage.py --stage future --phase pilot --task-id 10` accepts the same explicit environment setting without submitting a job.

Existing output notebooks are never overwritten. A common `TG_NOTEBOOK_OUTPUT_DIR` is allowed for the single-task retry above but must not be used for a multi-task array because its tasks would collide; normally leave it unset for array submission. A completed task is validated rather than retrained. If a checkpoint is incompatible, preserve the existing artifacts and investigate the mismatch; a materially revised experiment needs a new run root. Runtime failures after test access retain the original test-open record and lock.

Inspect configuration and receipts:

```bash
"$TG_PYTHON" scripts/research_directions/temporal_gait/validate_run.py \
  --config "$TG_CONFIG" --run-root "$TG_RUN_ROOT" --phase pilot
"$TG_PYTHON" scripts/research_directions/temporal_gait/validate_run.py \
  --config "$TG_CONFIG" --run-root "$TG_RUN_ROOT" --phase pilot --verify-artifacts
```

The second command verifies the frozen identity and every **present** receipt/output digest. It does not establish that all required tasks exist; the evaluation comparison owns complete-grid checks. Neither command is a scientific pass.

## 5. Explicit local synthetic verification

Use [synthetic.example.json](../../../../slurm/temporal-gait/synthetic.example.json) with a new output directory and clear all real manifest exports first. It selects CPU, one seed, small model dimensions, two JEPA updates and two direct-model updates. Synthetic configuration rejects private manifest inputs.

```bash
unset TG_VIDEO_MANIFEST TG_SEQUENCE_MANIFEST TG_POSE_MANIFEST TG_IDENTITY_MANIFEST
unset TG_EXPOSURE_MANIFEST TG_RESERVATION_MANIFEST TG_SPLIT_MANIFEST
export TG_CONFIG="$GAVD6_ROOT/slurm/temporal-gait/synthetic.example.json"
export TG_RUN_ROOT="/ABSOLUTE/LOCAL/OUTPUT/temporal-gait-software-check"

"$TG_PYTHON" scripts/research_directions/temporal_gait/run_stage.py \
  --config "$TG_CONFIG" --stage inventory --phase pilot
"$TG_PYTHON" scripts/research_directions/temporal_gait/run_stage.py \
  --config "$TG_CONFIG" --stage prepare --phase pilot
"$TG_PYTHON" scripts/research_directions/temporal_gait/run_stage.py \
  --config "$TG_CONFIG" --stage audit --phase pilot
"$TG_PYTHON" scripts/research_directions/temporal_gait/run_stage.py \
  --config "$TG_CONFIG" --phase pilot --dry-run
```

Select each task ID printed by the last command and call `--stage masked` for `masked_index`/`masked`, or `--stage future` for the three future arms, always supplying `--task-id`. Then call `--stage evaluate --phase pilot`. This makes task ownership visible and prevents a notebook from launching the entire grid internally. Do not use Slurm just to run a CPU fixture.

For fresh-kernel verification of a selected stage:

```bash
"$TG_PYTHON" scripts/research_directions/temporal_gait/execute_notebook.py \
  --notebook 00 --run-root "$TG_RUN_ROOT" --config "$TG_CONFIG" --execute \
  --output-dir "$TG_RUN_ROOT/notebook_runs/synthetic-inventory-check"
```

Software tests and deterministic notebook regeneration:

```bash
PYTHONPATH=src .venv/bin/python -m unittest tests.temporal_gait.test_hpc_notebooks -v
.venv/bin/python scripts/research_directions/temporal_gait/build_notebooks.py
```

The bounded end-to-end software verifier executes each stage in a **fresh kernel**, including every small arm/seed task, both explicit extension gates, synthetic lock/test, aggregation, and one real-mode plan-only notebook with deliberately nonexistent manifest paths:

```bash
.venv/bin/python scripts/research_directions/temporal_gait/verify_software.py \
  --run-root /ABSOLUTE/LOCAL/OUTPUT/temporal-gait-fresh-kernel-verification
```

Run only after finishing code changes, since the fixture freezes the same provenance identity as real runs. The output directory must be new or empty. There is no real-mode option: the command loads the shipped bounded CPU fixture, clears inherited `TG_*` overrides, and writes the resolved fixture before inventory freezes it. Executed copies stay under that output root, with per-notebook hashes, wall times and completion/failure status in `verification.json`. Source notebooks remain output-free. Kernel startup needs local loopback sockets, which some sandboxes block. A completed report says `software_verified_only`; it must retain `real_scientific_readiness: false`. This command is a reproducible check, not a statement that a particular run has passed.

## 6. Common failures and interpretation

| Failure | Meaning and action |
| --- | --- |
| Contradictory `TG_*` / JSON value | Use one consistent value or leave that path empty in the template; do not change an existing frozen run. |
| Missing versioned manifest or wrong kind | Supply the exact schema below; legacy CSV is not automatically interpreted. |
| Missing full-bout pose cache | Prepare a compatible upstream cache explicitly. The study does not locate or extract another pose source. |
| PTS/frame/hash mismatch | Restore the declared file or correct the upstream export in a new cohort/run. Do not derive timestamps from nominal FPS to silence the failure. |
| Incompatible frozen artifact/identity | A configuration, code, manifest or environment changed. Preserve the old result and start a new run unless an exact documented resume is supported. |
| Changed inventory or analysis-lock output | Preparation/test access is deliberately blocked. Restore the authenticated artifact or establish a new authorized run; do not rewrite receipts to bless a changed source roster or selection. |
| Secondary decoder `not_estimable` | Training/inner-validation support cannot estimate that horizon. Its predictions stay unavailable; inspect `decoder_horizon_support`. A supported0.50s analysis can continue. Missing primary readout-validation support still fails; do not borrow another horizon or tune on test. |
| Existing executed notebook | Select a fresh output directory; source and prior executed outputs remain intact. |
| Development stop or missing task | Inspect the complete comparison, support and baseline results. Do not expand compute or open test to search for a better outcome. |
| `gated_not_implemented` | E3/RGB extension is not available in this milestone. It is neither a model-loading fallback nor empirical evidence. |

The primary observable forecast is at 0.50 seconds on paired knee/ankle/heel/foot-tip landmarks, with 20 ms timestamp tolerance and prefix body-length normalization. At least three bilateral pairs must be observed; interpolated target coordinates are not observations. Average windows within bouts, bouts within recordings, then recordings equally. Bootstrap complete duplicate/known-person groups and report seed variation separately. Historical laterality scores retain their original definition.

The stored skeleton has all 33 MediaPipe joints; the primary score uses joint IDs `[25,26,27,28,29,30,31,32]`. It scores only complete observed bilateral pairs, with at least three of the four pairs available for a target. Coordinates are Euclidean full-frame pixels with aspect ratio preserved. For each prefix, the scale is the median valid side/time chain length `||shoulder−hip|| + ||hip−knee|| + ||knee−ankle||`, using left IDs `[11,23,25,27]` and right IDs `[12,24,26,28]`. Require at least eight distinct valid original chain times on each side and scale at least `max(20 pixels, 0.02 × frame diagonal)`. Freeze this past-only scale for all future targets. These are the new primary measurement's units, not a redefinition of historical laterality.

Issue time is `b`. The 64 prefix queries are exactly `b−2.56+i/25`, for `i=0,...,63`; the last observed-grid query is `b−0.04`, so report that forecast-issue offset. For horizon `h`, score the nearest actual original PTS to `b+h`, within ±0.02 seconds, with an earlier-time tie break. A target at or before `b`, outside the declared bout, or without sufficient observation support is invalid; do not interpolate it into a valid label. The latent target interval uses the two queries `b+h−0.04` and `b+h`, separately from this single coordinate endpoint. Wrong-target controls and causality tests may alter only strictly future observations; target masks, original target timestamps and target values never enter the prediction interface. Report candidate, SSL-eligible and score-eligible window/bout/video counts separately, including common-support denominators used for each paired comparison.

## 7. Input-manifest and pose schema

Each input is a JSON object with `schema_version: "temporal-gait-manifest-v1"`, its `kind`, and a `rows` list. `video_id` / `sequence_id` are nonempty strings; IDs must be unique in their manifest. SHA256 values are the actual 64-hex-character content digests, not generated IDs. All media/cache paths are absolute. Do not place raw video, linkable source IDs or private paths into a public manuscript by default.

| Kind | Row fields |
| --- | --- |
| `video` | `video_id`, `video_path`, `sha256`, `is_full_video: true` |
| `sequence` | `sequence_id`, `video_id`, `start_pts`, `end_pts_exclusive`, `boundary_convention: "pts_half_open"`, `walking_status: "complete_walking_bout"`, `annotation_provenance` |
| `pose` | `sequence_id`, `pose_path`, `sha256`, `source_video_sha256`, `extractor`, `extractor_version`, `checkpoint_sha256`, `tracking_provenance`, `causal_preprocessing: true`, `coordinate_frame: "full_frame_pixels"`, `joint_order: "mediapipe33"`, `time_kind: "video_pts"`, `complete_bout: true` |
| `identity` | `video_id`, verified `participant_id` and/or `duplicate_id`, `provenance`; an empty `rows` list is allowed |
| `exposure` | `video_id`, `status`, `historical_laterality_member` boolean, `provenance`; status is `certified_unexposed`, `development_exposed`, `test_evaluated` or `unknown` |
| `reservation` | `video_id`, `protected` boolean, `reason` |
| `split` | `video_id`, `role`, `reason`; role is `train`, `development`, `calibration`, `test` or `excluded` |

The `identity` envelope additionally requires `verified_links_complete: true`: attest only that the known verified links have been accounted for, not that every recording has a known participant. The `exposure` envelope requires `historical_laterality_93_checked: true` and `audit_provenance`; supply evidence of that audit rather than asserting it by default. Every exposure row must say whether it belongs to the audited historical laterality set. Membership is distinct from the exposure category and cannot certify a previously evaluated source as untouched.

Example envelope structure, with illustrative IDs and explicit placeholders to replace:

```json
{
  "schema_version": "temporal-gait-manifest-v1",
  "kind": "video",
  "rows": [{
    "video_id": "recording-A",
    "video_path": "/EXACT/HAIC/FULL/VIDEO.mp4",
    "sha256": "REPLACE_WITH_THE_ACTUAL_64_HEX_CONTENT_DIGEST",
    "is_full_video": true
  }]
}
```

Exposure, reservation and split manifests cover **exactly** the full video roster. Protected sources must be `excluded`; test/calibration sources must be `certified_unexposed`. Every linked component shares one role. Pose entries refer only to declared bouts, and each eligible complete bout requires its matching pose export. Missing targets do not exclude a usable SSL window; unsupported primary metric rows remain explicit.

Pose NPZ files contain no pickle/object arrays:

| Key | Shape / content |
| --- | --- |
| `times` | `[F]`, finite strictly increasing original video PTS in seconds |
| `coords` | `[F,33,2]`, full-frame image coordinates in pixels, aspect ratio preserved |
| `observed` | boolean `[F,33]`, including failed detections as false |
| `confidence` | `[F,33]`, finite values in [0,1] |
| `frame_indices` | integer `[F]`, strictly increasing zero-based full-video frame IDs |
| `width`, `height` | scalar positive original full-frame pixel dimensions |

The frame-index list must equal every original frame whose PTS is in `[start_pts,end_pts_exclusive)`. Timestamp comparisons use the original `ffprobe` frame PTS; dimensions must agree with the full video. Upstream failures remain represented as rows with false observation masks rather than removed frames. All extraction, tracking and smoothing dependencies must satisfy the declared past-only provenance. No manually written manifest assertion can make a future-aware tracker causal.
