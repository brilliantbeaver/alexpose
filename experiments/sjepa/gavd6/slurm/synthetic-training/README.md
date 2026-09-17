# Synthetic-training experiments on HAIC

Follow these steps from the **gavd6 checkout on HAIC**. The [notebook guide](../../notebooks/synthetic_training/README.md) explains the experiments; this page covers installation, assets, submission, and results.

```text
Source experiment: 00 inventory → 01 data → 02 student array → 03 fit selectors → 07 report
Real evaluation:   04 prepare GAVD → 05 deploy → 08 exchange lessons → 06 score
```

Run the source experiment first. GAVD deployment and scoring are separate submissions.

## 1. Set the project, Python, and run paths

Use absolute paths available on both login and compute nodes. Change the examples to match your installation, and choose a fresh run directory for a new experiment.

```bash
cd "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
export GAVD6_ROOT="$PWD"
export ST_PYTHON="/hai/scratch/$USER/envs/synthetic-training/bin/python"
export ST_MODEL_ROOT="/hai/scratch/$USER/models/synthetic-training"
export ST_RUN_ROOT="$GAVD6_ROOT/outputs/synthetic-training/pilot-01"
export ST_CONFIG="$ST_RUN_ROOT/pilot.json"
unset ST_STUDENT_ID ST_DEPENDENCY ST_NOTEBOOK_OUTPUT_DIR
```

| Variable | Meaning |
| --- | --- |
| `GAVD6_ROOT` | This checkout, containing `src`, `manifests`, and `slurm`. |
| `ST_PYTHON` | The study environment's **Python executable**, not its directory. Every executed notebook uses it. |
| `ST_MODEL_ROOT` | Download location for MMPose and student weights. Step 4 writes these paths into the configuration. |
| `ST_RUN_ROOT` | Shared output directory for every stage of this experiment. |
| `ST_CONFIG` | Input JSON created in step 4. Create it before submitting jobs. |

Keep exports in the shell that calls `submit.sh`. Save them in a shell file and `source` it after reconnecting. Slurm captures exported values when you submit; later shell changes do not update queued jobs.

## 2. Install the study environment and pose models

Use a **separate Python 3.11 environment**. The motion-preservation environment's PyTorch 2.6 stack differs from this study's pinned MMPose stack. These commands target Linux x86-64 with a CUDA 12.1-compatible NVIDIA driver.

You need `git`, `curl`, `unzip`, and [uv](https://docs.astral.sh/uv/getting-started/installation/). If `uv` is missing:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Create the environment once, then install packages in this order:

```bash
export UV_CONSTRAINT="$GAVD6_ROOT/slurm/synthetic-training/constraints-haic.txt"
export UV_NO_CONFIG=1
uv venv --python 3.11 "$(dirname "$(dirname "$ST_PYTHON")")"

uv pip install --python "$ST_PYTHON" \
  'torch==2.1.0' 'torchvision==0.16.0' \
  --index-url https://download.pytorch.org/whl/cu121
uv pip install --python "$ST_PYTHON" 'numpy==1.26.4'
uv pip install --python "$ST_PYTHON" 'mmcv==2.1.0' --only-binary mmcv \
  --find-links https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
uv pip install --python "$ST_PYTHON" \
  'mmengine==0.10.7' 'mmdet==3.3.0' 'mmpretrain==1.2.0' 'opencv-python==4.9.0.80'

bash slurm/synthetic-training/download-students.sh
uv pip install --python "$ST_PYTHON" --editable "$ST_MODEL_ROOT/mmpose"
uv pip install --python "$ST_PYTHON" \
  nbclient nbformat ipykernel jupyterlab pandas scipy scikit-learn matplotlib \
  Pillow tqdm pyrender trimesh einops 'timm==1.0.15' iopath PyYAML
uv pip install --python "$ST_PYTHON" \
  'human-body-prior @ git+https://github.com/nghorbani/human_body_prior.git@78c86eae5ed518ae22bf197fd74211bbfa45551a'
```

The [constraints file](constraints-haic.txt) keeps these versions fixed; `UV_NO_CONFIG` prevents the checkout's usual `uv` settings from selecting another stack. Keep both variables set during installation. The notebooks import this checkout directly, so **do not run the project's default `uv sync` in this study environment**.

If MMPose installation fails while building `chumpy`, run the following, then repeat the failed installation and remaining package commands:

```bash
uv pip install --python "$ST_PYTHON" pip setuptools
uv pip install --python "$ST_PYTHON" --no-build-isolation chumpy
```

After installation, check the environment:

```bash
uv pip check --python "$ST_PYTHON"
unset UV_CONSTRAINT UV_NO_CONFIG
"$ST_PYTHON" -c 'import torch, mmcv, mmpose; from mmcv.ops import nms; print(torch.__version__, mmcv.__version__, mmpose.__version__, torch.cuda.is_available())'
```

Expect PyTorch `2.1.0`, MMCV `2.1.0`, and MMPose `1.3.2`. CUDA availability should be `True` on a GPU node; it may be `False` on the login node. The [PyTorch release instructions](https://pytorch.org/get-started/previous-versions/#v210) and [MMPose installation guide](https://mmpose.readthedocs.io/en/latest/installation.html) provide upstream details. This installation has not been executed on HAIC during this documentation revision.

[download-students.sh](download-students.sh) downloads MMPose v1.3.2 and five pose checkpoints, preserving existing files. An existing MMPose checkout must already be compatible. The pilot uses RTMPose-m/HRNet-w32 for fitting, RTMPose-s/HRNet-w48 for validation, and ViTPose-base only for held-architecture deployment. The script does not download datasets, body models, rendering assets, or V-JEPA.

## 3. Prepare the source data and encoder

### Existing motion and rendering assets

Set these before running `data` or `source`. Replace every `/path/to/...` value with an existing asset.

```bash
export ST_AMASS_ROOT="${AMASS_EXTRACTED_ROOT:-$GAVD6_ROOT/data/amass}"
export ST_BODY_MODEL_ROOT="${AMASS_BODY_MODEL_ROOT:-/hai/scratch/$USER/body_models}"
export ST_TEXTURE_DIR="/path/to/body_texture_images"
export ST_BACKGROUND_DIR="/path/to/photographic_backgrounds"
export ST_UV_PATH="/path/to/smplh_uv_template.obj"
export ST_COCO_IMAGE_ROOT="/hai/scratch/$USER/coco/train2017"
export ST_COCO_ANNOTATIONS="/hai/scratch/$USER/coco/annotations/person_keypoints_train2017.json"
```

| Asset | Required contents |
| --- | --- |
| `ST_AMASS_ROOT` | Extracted raw AMASS parameters. Each inventory entry must exist at `$ST_AMASS_ROOT/<relative_path>` from `manifests/amass/amass_raw_inventory_eligible.csv`. |
| `ST_BODY_MODEL_ROOT` | `smplh/{male,female}/model.npz` and `dmpls/{male,female}/model.npz`. If DMPL lives elsewhere, set `ST_DMPL_ROOT` to the directory containing its `male/` and `female/` folders. |
| `ST_TEXTURE_DIR`, `ST_BACKGROUND_DIR` | Body textures and photographic backgrounds as PNG/JPG/JPEG images, found recursively. |
| `ST_UV_PATH` | Triangular OBJ with UV indices, or NPZ containing `uv`, `face_uv`, and `faces`. Its topology must match the loaded SMPL-H mesh. |
| COCO paths | Training images and **person-keypoint** annotations used for real-data replay. |

AMASS, licensed body models, and compatible texture/UV assets require separate preparation; these launchers do not supply them. Reuse existing installations. Slurm defaults to `PYOPENGL_PLATFORM=egl` for headless rendering; unset an old override or set it to `egl`. The GPU node must have working EGL support.

If COCO is missing, download and extract the [official archives](https://cocodataset.org/#download). Reuse a shared copy when available; the image archive is large.

```bash
mkdir -p "/hai/scratch/$USER/coco"
curl --fail --location https://images.cocodataset.org/zips/train2017.zip \
  --output "/hai/scratch/$USER/coco/train2017.zip" && \
unzip -n "/hai/scratch/$USER/coco/train2017.zip" -d "/hai/scratch/$USER/coco"
curl --fail --location https://images.cocodataset.org/annotations/annotations_trainval2017.zip \
  --output "/hai/scratch/$USER/coco/annotations_trainval2017.zip" && \
unzip -n "/hai/scratch/$USER/coco/annotations_trainval2017.zip" -d "/hai/scratch/$USER/coco"
```

### V-JEPA and the image comparator

The example configuration uses the [official V-JEPA 2.1 ViT-B checkpoint](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints). Set existing locations, or use these locations and download commands:

```bash
export ST_CONTEXT_REPO="${VJEPA2_ROOT:-$ST_MODEL_ROOT/vjepa2}"
export ST_CONTEXT_CHECKPOINT="$ST_MODEL_ROOT/vjepa2_1_vitb_dist_vitG_384.pt"
export ST_IMAGE_CHECKPOINT="$ST_MODEL_ROOT/pose/resnet50-11ad3fa6.pth"

# Clone only if the author checkout is absent. Skip downloads for existing weights.
[[ -d "$ST_CONTEXT_REPO" ]] || git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "$ST_CONTEXT_REPO"
curl --fail --location https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
  --output "$ST_CONTEXT_CHECKPOINT.part" && mv "$ST_CONTEXT_CHECKPOINT.part" "$ST_CONTEXT_CHECKPOINT"
curl --fail --location https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
  --output "$ST_IMAGE_CHECKPOINT.part" && mv "$ST_IMAGE_CHECKPOINT.part" "$ST_IMAGE_CHECKPOINT"
```

The author checkout must contain `vjepa2_1_vit_base_384` in `hubconf.py`. The pilot matches this checkpoint with `context_builder="vjepa2_1_vit_base_384"`, `context_checkpoint_key="ema_encoder"`, `context_image_size=384`, and `context_frames=64`. If using different weights, edit the matching settings in `pilot.json` before preparing data. The key, image size, and frame count have no environment-variable overrides.

`ST_IMAGE_CHECKPOINT` adds the [ResNet-50 V2 comparator](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) alongside V-JEPA. Set it before `data`; leave it unset only when deliberately omitting that comparison.

## 4. Create the pilot configuration

Create this file once after downloading the pose models. The snippet replaces the example's model paths with your `ST_MODEL_ROOT`:

```bash
"$ST_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["GAVD6_ROOT"])
models = Path(os.environ["ST_MODEL_ROOT"]).expanduser().resolve()
config = json.loads((root / "slurm/synthetic-training/pilot.example.json").read_text())
for student in config["students"]:
    for key in ("config", "checkpoint"):
        student[key] = str(models / Path(student[key]).relative_to("models"))
destination = Path(os.environ["ST_CONFIG"])
destination.parent.mkdir(parents=True, exist_ok=True)
with destination.open("x") as handle:
    handle.write(json.dumps(config, indent=2) + "\n")
print(destination)
PY
```

Keep the [pilot defaults](pilot.example.json) for the first pass: 4 clips per lesson, 64 frames per clip, 10 probe updates, 25/75 adaptation updates, and batches of 20 images. These are small trial settings, not a validated training budget.

**Configuration rule:** explicit `ST_*` overrides take precedence over JSON. The workflow saves resolved settings in `$ST_RUN_ROOT/config.json`; `unset ST_CONFIG` reuses that file. Use `unset`, not an empty export, to remove an override. JSON paths are relative to `GAVD6_ROOT` and do not expand shell variables. Changing `ST_MODEL_ROOT` later does not change student paths already written into JSON.

## 5. Inspect the inventory, then run the source experiment

```bash
bash slurm/synthetic-training/submit.sh inventory --dry-run
bash slurm/synthetic-training/submit.sh inventory
```

Wait for notebook 00 and inspect its asset table under `$ST_RUN_ROOT/notebook_runs/run-00/`. Missing paths need fixing before rendering. Inventory checks availability; it does not load every model or validate rendering.

```bash
unset ST_STUDENT_ID ST_DEPENDENCY ST_NOTEBOOK_OUTPUT_DIR
bash slurm/synthetic-training/submit.sh source --dry-run
bash slurm/synthetic-training/submit.sh source
```

`source` chains `00 → 01 → 02 array → 03 → 07` with `afterok` dependencies. Fitting waits for all four source students. Each array task requests one H100; the example therefore uses at most four H100s concurrently. Read notebook 07's source report before proceeding to GAVD. No real reference outcomes are opened by this pipeline.

Dry runs submit nothing, but create log directories and array rosters. Optional controls:

| Variable | Default / use |
| --- | --- |
| `ST_ACCOUNT`, `ST_PARTITION` | `mind`, `hai`; override if your allocation differs. GPU stages request `gpu:h100:1`. |
| `ST_MAX_PARALLEL` | `8`; caps simultaneous student tasks, not GPUs per student. |
| `ST_DEPENDENCY` | Prerequisite job ID, or `afterok:jobid[:jobid]`. Prefer a command-scoped assignment. |
| `ST_STUDENT_ID` | Restricts `trials`, `deploy`, or `crossover` to one configured student. Leave unset for a complete run. |

## 6. Prepare GAVD and deploy the frozen selectors

Set these when the source experiment is ready:

```bash
export ST_GAVD_VIDEO_ROOT="$GAVD6_ROOT/data/gavd_full/youtube/all"
export ST_GAVD_RESERVATION="/path/to/previous-fi-run/config/source-reservation.csv"
unset ST_GAVD_VIEWS
bash slurm/synthetic-training/submit.sh gavd
```

`ST_GAVD_VIDEO_ROOT` contains full recordings named by `video_id`. The reservation CSV must already exist and contain `video_id` and `role` (`development` or `confirmation`). Only development IDs are eligible; confirmation and unlisted IDs are excluded. There is no automatic reservation-file search.

The first pass writes `$ST_RUN_ROOT/data/gavd-view-template.csv` and stops. Complete the rows you have checked: `coarse_view` (`side`, `oblique`, `frontal_rear`), `view_checked=true`, `person_height_px`, and `bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2` in full-frame pixels. Then rerun:

```bash
export ST_GAVD_VIEWS="$ST_RUN_ROOT/data/gavd-view-template.csv"
bash slurm/synthetic-training/submit.sh gavd
```

If `gavd_view_csv` is already set in your JSON, remove that setting for the initial template pass. The second pass prepares disjoint context/reference recordings, caches context features, and creates annotation pages. Wait for it to finish, then:

```bash
unset ST_STUDENT_ID
bash slurm/synthetic-training/submit.sh deploy --dry-run
bash slurm/synthetic-training/submit.sh deploy
```

Deployment runs all five students, including held ViTPose, and saves choices and predictions without human reference coordinates. After the entire deployment array finishes, exchange the selected lessons:

```bash
bash slurm/synthetic-training/submit.sh crossover
```

Alternatively, submit crossover with `ST_DEPENDENCY=<deployment-job-id>` so it waits automatically. Crossovers need the other students' saved choices.

## 7. Annotate references and score the results

Annotation can proceed while deployment runs. Copy the run's **entire `data/` directory**, including `gavd/`, to your local machine.

1. Open `data/annotate-early.html` in a browser.
2. Enter the annotator name, draw the independent reference box, and mark each of the twelve landmarks or label it hidden. Use full-frame images, not model predictions.
3. Save JSON progress. Have a reviewer load it, check the frames, and mark them reviewed.
4. Export `gavd-early-annotations.csv` and copy it to an annotation directory on HAIC. Repeat with `annotate-confirmation.html` for confirmation; that split requires a reviewer different from the annotator.

After predictions, crossovers, and early annotations are complete:

```bash
export ST_GAVD_ANNOTATIONS="/path/to/independent_annotations"
bash slurm/synthetic-training/submit.sh evaluate
```

The directory must contain `gavd-early-annotations.csv`. `evaluate` opens only early references. Keep the selectors frozen before viewing real results; if those results change the method, report them as labeled development.

When ready for the separate untouched confirmation, add `gavd-confirmation-annotations.csv` to the same directory and run:

```bash
bash slurm/synthetic-training/submit.sh confirmation
```

The launcher selects the split; no `ST_EVALUATION_SPLIT` export is needed. Scores compare real landmark error against source-progress controls and full-budget replay. Successful execution alone does not establish useful teaching; see the [study guide](../../docs/studies/synthetic-training/README.md).

## 8. Find outputs or rerun one stage

| Under `ST_RUN_ROOT` | Contents |
| --- | --- |
| `logs/` | Slurm output/errors, job IDs in `submissions.tsv`, and array rosters. |
| `notebook_runs/run-NN/` | Timestamped executed notebooks; student folders for 02/05/08 and split folders for 06. Failed attempts retain their outputs. |
| `data/` | Prepared source/GAVD manifests, rendered images, and annotation tools. |
| `source/`, `selectors/`, `reports/` | Training outcomes, frozen selectors, and source comparisons. |
| `deployment/`, `evaluation/<split>/` | Chosen lessons, saved predictions, real scores, intervals, and crossover results. |

Individual commands are `inventory`, `data`, `trials`, `fit`, `report`, `gavd`, `deploy`, `crossover`, `evaluate`, and `confirmation`. For example:

```bash
ST_STUDENT_ID=rtmpose_m bash slurm/synthetic-training/submit.sh trials
# Fit only after all configured source students have completed:
ST_DEPENDENCY=123456 bash slurm/synthetic-training/submit.sh fit
```

Keep the same configuration for a retry. Each notebook attempt gets a new timestamped folder; leave `ST_NOTEBOOK_OUTPUT_DIR` unset for pipelines and arrays. Rebuilding data after source outcomes exist, or repeating trials/fitting after selectors are frozen, requires a new run. For new fitting or another condition, choose a new run root and reset `ST_CONFIG` and any old overrides.

To run a notebook directly in an appropriate allocation:

```bash
export PYOPENGL_PLATFORM=egl
"$ST_PYTHON" scripts/research_directions/synthetic_training/execute_notebook.py \
  --notebook 00 --run-root "$ST_RUN_ROOT"
```

The executor inherits `ST_CONFIG`, or reads the saved configuration if it is unset. Use `--student-id` for 02/05/08 and `--evaluation-split early` or `confirmation` for 06. For interactive work, launch `"$ST_PYTHON" -m jupyter lab notebooks/synthetic_training` and select the study environment's kernel. If missing, register it once with `"$ST_PYTHON" -m ipykernel install --user --name synthetic-training --display-name "Synthetic training"`.
