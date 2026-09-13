# Proposal 01 on HAIC

The launchers execute the [six motion-preservation notebooks](../../notebooks/motion_preservation/README.md) and save their outputs. They use the existing HAIC defaults, account `mind` and partition `hai`. GPU stages request one H100. Use independent conditions or seeds to occupy additional GPUs after a passing pilot.

## 1. Set paths in the existing experiment environment

Run from the gavd6 checkout. The examples below are paths to replace with the corresponding files on HAIC; they do not download data or models.

```bash
export GAVD6_ROOT="$PWD"
export MP_PYTHON="$GAVD6_ROOT/.venv/bin/python"
export MP_RUN_ROOT="/hai/scratch/$USER/motion-preservation/pilot-01"
export MP_CONFIG="$GAVD6_ROOT/slurm/motion-preservation/pilot.example.json"
export MP_AMASS_ROOT="/hai/scratch/$USER/amass"
export MP_BODY_MODEL_ROOT="/hai/scratch/$USER/body_models"
export MP_GAVD_VIDEO_ROOT="/hai/scratch/$USER/gavd_full/youtube/all"
export MP_MODE=real
export MP_DEVICE=cuda
```

`MP_AMASS_ROOT` contains paths relative to `manifests/amass/amass_raw_inventory_eligible.csv`. The existing `AMASS_EXTRACTED_ROOT` and `AMASS_BODY_MODEL_ROOT` environment variables are accepted when explicit `MP_*` settings are absent. GAVD selection uses `manifests/gavd/gavd_full_sequences.csv` and recording IDs. Manifests record the available inventory; they are not copied or replaced by generated fixture data.

The licensed body-model directory contains `smplh/{male,female}/model.npz` and `dmpls/{male,female}/model.npz`. An explicit `MP_DMPL_ROOT` can point to a separate DMPL tree. Raw AMASS parameters are converted to the first 22 SMPL-H joints and the corresponding body mesh, including the saved shape and DMPL parameters. Model-space coordinates use meters and a positive-up vertical axis; exported predictions must match this convention.

Use the existing project environment with notebook, PyTorch, body-model and image-processing dependencies installed. `MP_PYTHON` can point to another compatible environment. The notebook runner always starts its kernel with that interpreter. It does not use a possibly stale user Jupyter kernel.

## 2. Configure released models

For the intended first pass, point to the official [MoMask](https://github.com/EricGuo5513/momask-codes) repository and RVQ checkpoint, and a released [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT) checkout/checkpoint. Model-specific dependencies belong in the chosen environment. Required licensed body-model assets must already be available.

```bash
export MP_MOMASK_REPO="/hai/scratch/$USER/models/momask-codes"
export MP_MOMASK_CHECKPOINT="$MP_MOMASK_REPO/checkpoints/t2m/rvq_nq6_dc512_nc512_noshare_qdp0.2"
export MP_FLOW_REPO="/hai/scratch/$USER/models/SEA-RAFT"
export MP_FLOW_CHECKPOINT="/hai/scratch/$USER/models/sea-raft/model.safetensors"
export MP_FLOW_CONFIG="$MP_FLOW_REPO/config/eval/spring-M.json"
```

If these files are not already on HAIC, the following preparation commands download the author code and released weights. Run the dependency step from the gavd6 checkout with its usual `.venv` selected as `MP_PYTHON`:

```bash
cd "$GAVD6_ROOT"
uv sync --extra motion-preservation
uv pip install --python "$MP_PYTHON" gdown

# Clone only repositories that are not already present.
mkdir -p "$(dirname "$MP_MOMASK_REPO")" "$(dirname "$MP_FLOW_REPO")"
git clone --depth 1 https://github.com/EricGuo5513/momask-codes.git "$MP_MOMASK_REPO"
git clone --depth 1 https://github.com/princeton-vl/SEA-RAFT.git "$MP_FLOW_REPO"
```

Download the HumanML3D bundle using the exact file published in MoMask's [author download script](https://github.com/EricGuo5513/momask-codes/blob/main/prepare/download_models.sh). This bundle contains other HumanML3D models alongside the RVQ model. A separate author-hosted RVQ-only download was not verified. The KIT bundle is unnecessary for this experiment.

```bash
mkdir -p "$MP_MOMASK_REPO/checkpoints/t2m"
"$MP_PYTHON" -m gdown --fuzzy \
  'https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing' \
  --output "$MP_MOMASK_REPO/checkpoints/t2m/humanml3d_models.zip"
unzip -n "$MP_MOMASK_REPO/checkpoints/t2m/humanml3d_models.zip" \
  -d "$MP_MOMASK_REPO/checkpoints/t2m"
```

The author's alternative command is `bash prepare/download_models.sh` from the MoMask repository root. That script first deletes its existing `checkpoints` directory and then downloads both dataset bundles. The commands above fetch only HumanML3D and preserve existing extracted files.

Download the [author's SEA-RAFT medium checkpoint](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/tree/main), approximately 78.8 MB. Its architecture matches the [official `spring-M.json` configuration](https://github.com/princeton-vl/SEA-RAFT/blob/main/config/eval/spring-M.json).

```bash
mkdir -p "$(dirname "$MP_FLOW_CHECKPOINT")"
curl --fail --location \
  'https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/resolve/main/model.safetensors' \
  --output "$MP_FLOW_CHECKPOINT"
```

`MP_MOMASK_CHECKPOINT` is a directory, not a single weight file. It contains `opt.txt`, `meta/mean.npy`, `meta/std.npy` and `model/net_best_fid.tar`. The bridge defaults to the author's bundled `example_data/000612.npy` as its HumanML3D reference skeleton. Optional `MP_TARGET_SKELETON` accepts a canonical HumanML3D `[T, 263]` representation or `[T, 22, 3]` joint sequence. It does not accept unconverted AMASS pose parameters. MoMask's representation may reconstruct one fewer frame; saved frame indices align outputs rather than inventing a last frame from reference truth.

MoMask needs its author code and `einops`; SEA-RAFT needs its author code, `huggingface_hub`, `safetensors` and compatible PyTorch/torchvision. The `motion-preservation` extra provides these additions. With an existing alternative environment, install the small additions with `uv pip install --python "$MP_PYTHON" einops safetensors huggingface_hub gdown` and retain that environment's matching PyTorch/torchvision pair. The wrappers use the RVQ and flow modules directly, so MoMask's text encoders, text-generation dependencies and separate legacy environment are unnecessary. This launcher never downloads weights. Missing assets or failed model loads stop the real run rather than choosing a synthetic prior.

If SEA-RAFT is unavailable, the [official torchvision RAFT-small weight](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html) is an explicit fallback:

```bash
export MP_FLOW_BACKEND=torchvision_raft
export MP_FLOW_CHECKPOINT="/hai/scratch/$USER/models/raft_small_C_T_V2-01064c6d.pth"
curl --fail --location \
  'https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth' \
  --output "$MP_FLOW_CHECKPOINT"
```

That backend needs no external repository or config. Imported prior or comparator predictions must correspond to the exact saved cases. Set `MP_PRIOR_PREDICTIONS` and a distinct `MP_PRIOR_ID` for an imported prior, or the `external_methods` mapping in JSON for external comparisons. A name alone does not verify that the original published method ran.

External prior files are `{case_id}.npz` with `joints[K,22,3]`, `frame_indices[K]` and `metadata_json`. The JSON records `model`, `checkpoint`, `coordinate_system: "world_y_up"`, and `context: "whole_supplied_clip"`. This interchange supports separately generated MDM or other outputs; it is not a complete MDM reconstruction implementation. Generate those outputs using the same observed case and retain their actual supported frame indices.

## 3. Inspect the small pilot, then submit it

Notebook 00 is inexpensive and inventories the selected paths. It does not load every checkpoint or run the final test.

```bash
"$MP_PYTHON" scripts/research_directions/motion_preservation/execute_notebook.py \
  --notebook 00 --run-root "$MP_RUN_ROOT" --mode real --device cpu \
  --config "$MP_CONFIG"

bash slurm/motion-preservation/submit.sh pilot --dry-run
bash slurm/motion-preservation/submit.sh pilot
```

The submission sequence is:

```text
00 inventory → 01 controlled pairs → 02 prior/flow cache → 03 train/calibrate → 04 development evaluation
```

Every arrow is an `afterok` dependency. A failed notebook prevents dependent jobs from starting. `pilot` never opens the reserved final event family and never launches GAVD automatically. The dry run prints scheduler commands and creates the output/log folders, but submits no jobs.

| Stage | Default resource | Time limit | Main output |
| --- | --- | --- | --- |
| 00 inventory | 4 CPU, 16 GB | 30 minutes | Manifest and asset inventory |
| 01 pairs | 1 H100, 8 CPU, 64 GB | 8 hours | Controlled cases and index |
| 02 evidence | 1 H100, 8 CPU, 64 GB | 12 hours | Frozen predictions and evidence |
| 03 fit | 1 H100, 8 CPU, 48 GB | 8 hours | Small gates and calibration strengths |
| 04 evaluation | 1 H100, 8 CPU, 64 GB | 12 hours | Scores, plots and decision |
| 05 GAVD | 1 H100, 8 CPU, 64 GB | 8 hours | Stress table and available overlays |

These are scheduler limits, not measured runtime estimates. The notebooks print elapsed stage times. Start with the example's eight motions per role and one optimization seed, then expand only when the first mechanism check passes. The larger proposal allocation is a 280 H100-hour cap, not a request for eight-way training.

## 4. Resume, evaluate the final family, or inspect GAVD

Individual stage names are `inventory`, `pairs`, `cache`, `fit`, `evaluate`, `final`, and `gavd`.

```bash
# Example: submit caching after an already queued pair-building job.
MP_DEPENDENCY=123456 bash slurm/motion-preservation/submit.sh cache

# After a passing development decision, open the reserved family once.
bash slurm/motion-preservation/submit.sh final

# Separate stress path; protect the existing FI confirmation sources.
export MP_GAVD_RESERVATION="/hai/scratch/$USER/previous-fi-run/config/source-reservation.csv"
# Optional 22-joint trajectories already projected into original video pixels.
export MP_GAVD_POSE_ROOT="/hai/scratch/$USER/gavd_projected_pose_exports"
bash slurm/motion-preservation/submit.sh gavd
```

`final` runs notebook 04 with `MP_EVALUATION_SPLIT=final`. It builds/caches final cases and evaluates the saved gate and operating points. It never retrains or recalibrates. `MP_DEPENDENCY` accepts a job ID or `afterok:jobid[:jobid]`. Set `MP_ACCOUNT` or `MP_PARTITION` to override scheduler defaults.

The current final setting changes the event family, camera angle and corruption mechanism together. Interpret it as a combined stress test, not an isolated estimate of event-family transfer.

The GAVD stage requires the existing source-reservation CSV before decoding real video. The prior study's 43 held source IDs are not stored in the local full-video manifest. Without the reservation file, it returns `not_run_missing_source_reservation`. This preserves that study's unopened sources without guessing their identities. Demo mode returns `not_run_demo` for GAVD.

Optional pose exports are `{sequence_id}.npz` containing `source_frames[N]` (zero-based original-video frame IDs), `joints2d[N,22,2]` (full-frame pixels), and `coordinate_system="full_frame_pixels"`. `repaired_joints2d` is optional. A WHAM 3D estimate needs an upstream camera projection first. The implemented GAVD stage produces flow galleries, contact sheets and diagnostics; it does not automatically claim learned 3D-gate transfer or ground-truth event preservation.

For several optimization seeds, edit `seeds` in a new configuration or set `MP_SEEDS=17,23,42`. The initial example uses one seed. To compare different model or data conditions, use separate run directories. Preserve the same case identities, feature conventions and saved calibration when claiming transfer to a second prior.

## 5. Read outputs and failures

Each submission prints its executed-notebook folder. The run contains:

```text
config.json                 selected experiment configuration
notebook_runs/haic-*/        executed notebook copies
logs/                       Slurm stdout/stderr and submissions.tsv
...                         cases, cached predictions, model fits and reports
```

Notebook outputs are saved after cells and on failure. A model-loading error is therefore visible both in the Slurm log and the partial executed notebook. Existing executed notebook files are not overwritten; a new submission creates another output folder.

The notebooks display the experiment's scores and backend metadata. Successful execution is not a passing scientific result. The relevant result is the locked preservation-versus-repair comparison against the strongest implemented baseline.

## CPU walkthrough, explicitly separate from research

This walkthrough generates small fixtures and uses stand-in prior/flow estimators. It needs neither HAIC nor raw AMASS files. It is useful for understanding the sequence and checking notebook execution.

```bash
export GAVD6_ROOT="$PWD"
export MP_RUN_ROOT="/tmp/motion-preservation-demo"
export MP_MODE=demo
export MP_DEVICE=cpu
unset MP_CONFIG
for notebook in 00 01 02 03 04; do
  "$GAVD6_ROOT/.venv/bin/python" \
    scripts/research_directions/motion_preservation/execute_notebook.py \
    --notebook "$notebook" --run-root "$MP_RUN_ROOT" --mode demo --device cpu
done
```

Keep the demo directory separate from a real run. Synthetic demonstration scores cannot establish event erasure by a public prior, realistic optical-flow performance, clinical validity, or an ICLR contribution.
