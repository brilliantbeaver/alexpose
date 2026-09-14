# Motion-preservation experiments on HAIC

The launchers execute the [six motion-preservation notebooks](../../notebooks/motion_preservation/README.md) and save their outputs. They use the existing HAIC defaults, account `mind` and partition `hai`. GPU stages request one H100. Use independent experimental conditions to occupy additional GPUs after a passing pilot. The configured optimization seeds run sequentially within each fit job.

## 1. Set paths in the existing experiment environment

The six experiments are notebook stages **00–05** sharing one run directory. `pilot` submits 00–04 in order; `gavd` submits 05 separately. Export variables in the shell where you call `submit.sh`; it passes them to every submitted job with `--export=ALL`.

### Shared variables

| Variable | Purpose and requirement |
| --- | --- |
| `MP_RUN_ROOT` | **Required by `submit.sh`.** Output directory shared by all stages; use a new directory for a new experiment condition. |
| `GAVD6_ROOT` | gavd6 checkout. `submit.sh` detects it when unset; exporting it is required when submitting an `.sbatch` file directly. |
| `MP_PYTHON` | Python interpreter containing the study dependencies. Defaults to `$GAVD6_ROOT/.venv/bin/python`. |
| `MP_CONFIG` | Optional JSON configuration. Use `pilot.example.json` for the small pilot; when unset, an existing `$MP_RUN_ROOT/config.json` is loaded, otherwise built-in defaults apply. |
| `MP_MODE`, `MP_DEVICE` | Optional overrides: `real` or `demo`, and `cuda` or `cpu`. The pilot JSON selects `real` and `cuda`; explicit demo mode defaults to CPU. |

### Variables needed by each stage

The table describes **real runs using the default MoMask and SEA-RAFT backends**. Asset locations can also be set in JSON; the corresponding environment variables are overrides, not additional mandatory exports.

| Notebook / submission | Inputs to configure |
| --- | --- |
| **00 — Inventory** / `inventory` | Shared variables above. Set asset paths before this stage to inventory them; it reports their availability without loading the models. |
| **01 — Controlled pairs** / `pairs` | `MP_AMASS_ROOT`: extracted motion files. `MP_BODY_MODEL_ROOT`: licensed SMPL-H and DMPL assets. Optional `MP_DMPL_ROOT` if DMPL files live separately. |
| **02 — Prior and flow** / `cache` | `MP_MOMASK_REPO`: author checkout; `MP_MOMASK_CHECKPOINT`: RVQ model **directory**. `MP_FLOW_REPO`: SEA-RAFT checkout; `MP_FLOW_CHECKPOINT`: weight **file**; `MP_FLOW_CONFIG`: matching architecture JSON. Reads the cases from 01. |
| **03 — Train and calibrate** / `fit` | Same run and configuration; reads the cache from 02. Optional `MP_EPOCHS` and `MP_SEEDS` control training. |
| **04 — Evaluate** / `evaluate` or `final` | Same run with fitted models and calibration from 03. `final` also needs the AMASS/body-model and pretrained-model assets from 01–02 to build and cache final cases. The launcher selects `MP_EVALUATION_SPLIT` automatically. |
| **05 — GAVD stress** / `gavd` | `MP_GAVD_VIDEO_ROOT`: real videos; `MP_GAVD_RESERVATION`: existing source-reservation CSV. Reuses the SEA-RAFT path settings from 02. Optional `MP_GAVD_POSE_ROOT` adds pose overlays. This stage does not load MoMask or the trained repair gate. |

**Precedence:** explicit `MP_*` experiment settings override the selected JSON. Legacy AMASS variables fill only fields absent from both; built-in defaults fill the rest. Keep the same configuration when resuming. Unsetting `MP_CONFIG` selects the saved run configuration, but other exported overrides still apply.

Run from the gavd6 checkout. Replace the example HAIC asset paths below with your installed locations; these exports do not download anything. The pilot JSON already sets the mode, device, backends and one training seed.

```bash
export GAVD6_ROOT="$PWD"
export MP_PYTHON="$GAVD6_ROOT/.venv/bin/python"
export MP_RUN_ROOT="/hai/scratch/$USER/motion-preservation/pilot-01"
export MP_CONFIG="$GAVD6_ROOT/slurm/motion-preservation/pilot.example.json"
export MP_AMASS_ROOT="/hai/scratch/$USER/amass"
export MP_BODY_MODEL_ROOT="/hai/scratch/$USER/body_models"
```

`MP_AMASS_ROOT` is the base directory for relative file paths in `manifests/amass/amass_raw_inventory_eligible.csv`; its built-in default is `data/amass`. `AMASS_EXTRACTED_ROOT` and `AMASS_BODY_MODEL_ROOT` are fallbacks only when the corresponding `amass_root` or `body_model_root` field is absent from both JSON and explicit `MP_*` settings. There is no built-in body-model location. GAVD selection uses `manifests/gavd/gavd_full_sequences.csv` and recording IDs. Its video-root default is `data/gavd_full/youtube/all`; configure it in section 4, or earlier to include the installed videos in notebook 00's inventory.

The licensed body-model directory contains `smplh/{male,female}/model.npz` and `dmpls/{male,female}/model.npz`. An explicit `MP_DMPL_ROOT` can point to a separate DMPL tree. Raw AMASS parameters are converted to the first 22 SMPL-H joints and the corresponding body mesh, including the saved shape and DMPL parameters. Model-space coordinates use meters and a positive-up vertical axis; exported predictions must match this convention.

Use the existing project environment with notebook, PyTorch, body-model and image-processing dependencies installed. `MP_PYTHON` can point to another compatible environment. The notebook runner always starts its kernel with that interpreter. It does not use a possibly stale user Jupyter kernel.

### Optional overrides

| Variables | When to use them |
| --- | --- |
| `MP_MAX_MOTIONS`, `MP_EPOCHS`, `MP_IMAGE_SIZE`, `MP_SEEDS` | Change motions per role, training epochs, render size or comma-separated training seeds. The pilot JSON uses `8`, `10`, `128` and `17`, respectively. Choose these before creating run artifacts. |
| `MP_ACCOUNT`, `MP_PARTITION`, `MP_DEPENDENCY` | Override `mind`, `hai`, or add a prerequisite job ID / `afterok:jobid[:jobid]`. |
| `MP_TORCH_THREADS` | Set the OpenMP/MKL/OpenBLAS thread limit; defaults to `4`. It does not request more Slurm CPUs or GPUs. |
| `MP_NOTEBOOK_OUTPUT_DIR` | Choose where executed notebooks are saved. Normally leave unset so each submission creates a fresh folder under the run directory. |

Alternative model settings (`MP_FLOW_BACKEND`, `MP_PRIOR_BACKEND`, `MP_PRIOR_ID`, `MP_PRIOR_PREDICTIONS`) and optional `MP_TARGET_SKELETON` are explained in section 2.

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
uv sync --extra motion-preservation --inexact
uv pip install --python "$MP_PYTHON" gdown

# Clone only repositories that are not already present.
mkdir -p "$(dirname "$MP_MOMASK_REPO")" "$(dirname "$MP_FLOW_REPO")"
[[ -d "$MP_MOMASK_REPO/.git" ]] || git clone --depth 1 https://github.com/EricGuo5513/momask-codes.git "$MP_MOMASK_REPO"
[[ -d "$MP_FLOW_REPO/.git" ]] || git clone --depth 1 https://github.com/princeton-vl/SEA-RAFT.git "$MP_FLOW_REPO"
```

Download the HumanML3D bundle using the exact file published in MoMask's [author download script](https://github.com/EricGuo5513/momask-codes/blob/main/prepare/download_models.sh). This bundle contains other HumanML3D models alongside the RVQ model. A separate author-hosted RVQ-only download was not verified. The KIT bundle is unnecessary for this experiment.

```bash
mkdir -p "$MP_MOMASK_REPO/checkpoints/t2m" &&
"$MP_PYTHON" -m gdown \
  'https://drive.google.com/file/d/1vXS7SHJBgWPt59wupQ5UUzhFObrnGkQ0/view?usp=sharing' \
  --output "$MP_MOMASK_REPO/checkpoints/t2m/humanml3d_models.zip" &&
unzip -n "$MP_MOMASK_REPO/checkpoints/t2m/humanml3d_models.zip" \
  -d "$MP_MOMASK_REPO/checkpoints/t2m"
```

Pass the share URL directly: [gdown 6 removed `--fuzzy`](https://github.com/wkentaro/gdown/releases/tag/v6.0.0) because Google Drive URL parsing is automatic. An `unrecognized arguments: --fuzzy` error occurs before downloading; a subsequent missing-ZIP error is a consequence. The `&&` operators above stop the sequence if directory creation or downloading fails. Check the installed version with `"$MP_PYTHON" -m gdown --version`.

The author's alternative command is `bash prepare/download_models.sh` from the MoMask repository root. That script first deletes its existing `checkpoints` directory and then downloads both dataset bundles. The commands above fetch only HumanML3D and preserve existing extracted files.

Download the [author's SEA-RAFT medium checkpoint](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/tree/main), approximately 78.8 MB. Its architecture matches the [official `spring-M.json` configuration](https://github.com/princeton-vl/SEA-RAFT/blob/main/config/eval/spring-M.json).

```bash
mkdir -p "$(dirname "$MP_FLOW_CHECKPOINT")"
curl --fail --location \
  'https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/resolve/main/model.safetensors' \
  --output "$MP_FLOW_CHECKPOINT"
```

If loading this checkpoint reports missing `cnet`/`fnet` keys ending in `downsample.1.weight`, `.bias`, `.running_mean`, or `.running_var`, sync the updated [gavd6 adapter](../../src/gavd6_sjepa/research_directions/motion_preservation/pretrained_models.py) to HAIC and restart the notebook kernel before retrying. This exact alias mismatch does not require downloading the checkpoint again or changing `MP_FLOW_CONFIG`.

SEA-RAFT registers these BatchNorm tensors under both `bn3` and `downsample.1`; the released safetensors file stores them once under `bn3`. The adapter restores names that refer to the same tensor in the model, then loads with `strict=True`. Do not suppress the error with `strict=False`, which could also hide genuinely missing weights. See [safetensors shared-tensor documentation](https://huggingface.co/docs/safetensors/torch_shared_tensors) for the storage convention.

`MP_MOMASK_CHECKPOINT` is a directory, not a single weight file. It contains `opt.txt`, `meta/mean.npy`, `meta/std.npy` and `model/net_best_fid.tar`. The bridge defaults to the author's bundled `example_data/000612.npy` as its HumanML3D reference skeleton. Optional `MP_TARGET_SKELETON` accepts a canonical HumanML3D `[T, 263]` representation or `[T, 22, 3]` joint sequence. It does not accept unconverted AMASS pose parameters. MoMask's representation may reconstruct one fewer frame; saved frame indices align outputs rather than inventing a last frame from reference truth.

MoMask needs its author code and `einops`; SEA-RAFT needs its author code, `huggingface_hub`, `safetensors` and compatible PyTorch/torchvision. The `motion-preservation` extra provides these additions. `--inexact` retains other packages in the shared environment. With an existing alternative environment, install the small additions with `uv pip install --python "$MP_PYTHON" einops safetensors huggingface_hub gdown` and retain that environment's matching PyTorch/torchvision pair. The wrappers use the RVQ and flow modules directly, so MoMask's text encoders, text-generation dependencies and separate legacy environment are unnecessary.

The body-model dependency must be the official GitHub version selected in this project's `pyproject.toml`, not the older PyPI release. The old release has a different constructor and does not apply the study's dynamic shape parameters. `uv sync` above installs the selected version. For a separate environment, install it explicitly:

```bash
uv pip install --python "$MP_PYTHON" --reinstall-package human-body-prior \
  'human-body-prior @ git+https://github.com/nghorbani/human_body_prior.git@78c86eae5ed518ae22bf197fd74211bbfa45551a'
```

This launcher never downloads weights. Missing assets or failed model loads stop the real run rather than choosing a synthetic prior.

If SEA-RAFT is unavailable, the [official torchvision RAFT-small weight](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.optical_flow.raft_small.html) is an explicit fallback:

```bash
export MP_FLOW_BACKEND=torchvision_raft
export MP_FLOW_CHECKPOINT="/hai/scratch/$USER/models/raft_small_C_T_V2-01064c6d.pth"
curl --fail --location \
  'https://download.pytorch.org/models/raft_small_C_T_V2-01064c6d.pth' \
  --output "$MP_FLOW_CHECKPOINT"
```

That backend needs no external repository or config. Imported prior or comparator predictions must correspond to the exact saved cases. For an imported prior, set `MP_PRIOR_BACKEND=external`, `MP_PRIOR_PREDICTIONS` to its export directory, and a distinct `MP_PRIOR_ID`. Use the `external_methods` mapping in JSON for external comparisons. A name alone does not verify that the original published method ran.

External prior files are `{case_id}.npz` with `joints[K,22,3]`, `frame_indices[K]` and `metadata_json`. The JSON records `model`, `checkpoint`, `coordinate_system: "world_y_up"`, and `context: "whole_supplied_clip"`. This interchange supports separately generated MDM or other outputs; it is not a complete MDM reconstruction implementation. Generate those outputs using the same observed case and retain their actual supported frame indices.

## 3. Inspect the small pilot, then submit it

Notebook 00 is inexpensive and inventories the selected paths. It does not load every checkpoint or run the final test.

```bash
"$MP_PYTHON" scripts/research_directions/motion_preservation/execute_notebook.py \
  --notebook 00 --run-root "$MP_RUN_ROOT" \
  --config "$MP_CONFIG"

bash slurm/motion-preservation/submit.sh pilot --dry-run
bash slurm/motion-preservation/submit.sh pilot
```

The submission sequence is:

```text
00 inventory → 01 controlled pairs → 02 prior/flow cache → 03 train/calibrate → 04 development evaluation
```

Every arrow is an `afterok` dependency. A failed notebook prevents dependent jobs from starting. `pilot` never opens the reserved final event family and never launches GAVD automatically. The dry run prints scheduler commands and creates the output/log folders, but submits no jobs.

Explicit `MP_*` settings override JSON values. Otherwise, the launcher respects `MP_CONFIG`, or the existing run's `config.json` when `MP_CONFIG` is unset. With neither configuration, the initial defaults are real data and CUDA. Notebook 00 performs only an inventory, so a CPU allocation is sufficient even when the configured model device is CUDA.

| Stage | Default resource | Time limit | Main output |
| --- | --- | --- | --- |
| 00 inventory | 4 CPU, 16 GB | 30 minutes | Manifest and asset inventory |
| 01 pairs | 1 H100, 8 CPU, 64 GB | 8 hours | Controlled cases and index |
| 02 evidence | 1 H100, 8 CPU, 64 GB | 12 hours | Frozen predictions and evidence |
| 03 fit | 1 H100, 8 CPU, 48 GB | 8 hours | Small gates and calibration strengths |
| 04 evaluation | 1 H100, 8 CPU, 64 GB | 12 hours | Scores, plots and decision |
| 05 GAVD | 1 H100, 8 CPU, 64 GB | 8 hours | Stress table and available overlays |

These are scheduler limits, not measured runtime estimates. The notebooks print elapsed stage times. Stage 01 uses the GPU for body reconstruction, but its triangle rendering runs on the CPU; more GPUs will not directly accelerate that rendering. Stage 04 also builds and caches cases when opening the final split, which is why it reserves a GPU. Start with the example's eight motions per role and one optimization seed, then expand only when the first mechanism check passes. The larger proposal allocation is a 280 H100-hour cap, not a request for eight-way training.

## 4. Resume, evaluate the final family, or inspect GAVD

Individual stage names are `inventory`, `pairs`, `cache`, `fit`, `evaluate`, `final`, and `gavd`.

```bash
# Example: submit caching after an already queued pair-building job.
MP_DEPENDENCY=123456 bash slurm/motion-preservation/submit.sh cache

# After a passing development decision, open the reserved family once.
bash slurm/motion-preservation/submit.sh final

# Separate stress path; protect the existing FI confirmation sources.
export MP_GAVD_VIDEO_ROOT="/hai/scratch/$USER/gavd_full/youtube/all"
export MP_GAVD_RESERVATION="/hai/scratch/$USER/previous-fi-run/config/source-reservation.csv"
# Optional 22-joint trajectories already projected into original video pixels.
export MP_GAVD_POSE_ROOT="/hai/scratch/$USER/gavd_projected_pose_exports"
bash slurm/motion-preservation/submit.sh gavd
```

`final` runs notebook 04 with `MP_EVALUATION_SPLIT=final`; every other `submit.sh` command sets it to `development`, overriding any shell value. Set this variable yourself only when executing notebook 04 directly. Final evaluation builds/caches final cases and uses the saved gate and operating points; it never retrains or recalibrates.

The current final setting changes the event family, camera angle and corruption mechanism together. Interpret it as a combined stress test, not an isolated estimate of event-family transfer.

The GAVD stage requires an existing source-reservation CSV before decoding real video. Set `MP_GAVD_RESERVATION` explicitly for reproducibility. If no reservation is configured, it searches `future-innovation*/config/source-reservation.csv` under `outputs/` and the run directory's parent. The prior study's 43 held source IDs are not stored in the local full-video manifest. If none is found, the stage returns `not_run_missing_source_reservation`; an explicitly configured nonexistent file raises an error. Demo mode returns `not_run_demo` for GAVD.

Optional pose exports are `{sequence_id}.npz` containing `source_frames[N]` (zero-based original-video frame IDs), `joints2d[N,22,2]` (full-frame pixels), and `coordinate_system="full_frame_pixels"`. `repaired_joints2d` is optional. A WHAM 3D estimate needs an upstream camera projection first. The implemented GAVD stage produces flow galleries, contact sheets and diagnostics; it does not automatically claim learned 3D-gate transfer or ground-truth event preservation.

For several optimization seeds, edit `seeds` in a new configuration or set `MP_SEEDS=17,23,42`. The initial example uses one seed. To compare different model or data conditions, use separate run directories. Preserve the same case identities, feature conventions and saved calibration when claiming transfer to a second prior.

## 5. Read outputs and failures

Each submission prints its executed-notebook folder. The run contains:

```text
config.json                 selected experiment configuration
notebook_runs/run-XX/        executed notebook copies (XX is the notebook number)
logs/                       Slurm stdout/stderr and submissions.tsv
...                         cases, cached predictions, model fits and reports
```

Notebook outputs are saved after cells and on failure. A model-loading error is therefore visible both in the Slurm log and the partial executed notebook. Existing executed notebook files are not overwritten; a new submission creates another output folder.

The notebooks display the experiment's scores and backend metadata. Successful execution is not a passing scientific result. The relevant result is the locked preservation-versus-repair comparison against the strongest implemented baseline.

The [implementation review](../../docs/studies/motion-preservation/results/implementation-review.md) corrected the repair metric and renderer visibility. Use a fresh run directory for the corrected pilot. Old operating points must be recalibrated because noise removal now measures observed joints, with missing-joint completion reported separately. Previously inspected final results remain development evidence.

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
