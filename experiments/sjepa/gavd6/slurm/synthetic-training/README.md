# Synthetic-training experiments on HAIC

Follow these steps from the **gavd6 checkout on HAIC**. The [notebook guide](../../notebooks/synthetic_training/README.md) explains the experiments; this page covers installation, assets, submission, and results.

```text
Source experiment: 00 inventory → 01 data → 02 student array → 03 fit selectors → 07 report
Real evaluation:   04 prepare GAVD → 05 deploy → 08 exchange lessons → 06 score
```

Run the source experiment first. GAVD deployment and scoring are separate submissions.

## Updating from the earlier instructions

Keep your existing dedicated Python 3.11 study environment, downloaded models, datasets, and experiment outputs. If inventory already ran, reuse `$ST_RUN_ROOT/config.json`; you do not need a file named `pilot.json`. [Step 4](#4-create-or-correct-the-run-configuration) preserves your settings and corrects the student paths. Changing `ST_CONFIG` alone only selects a file; it does not change the paths inside it.

After updating the checkout, set step 1 to your existing paths and rerun the setup commands in step 2 if the environment needs updating. You do not need to delete the environment or repeat `uv venv` to correct asset paths.

The setup script handles the old `UV_CONSTRAINT` and `UV_NO_CONFIG` exports internally. It synchronizes package versions with the study lockfile and may remove packages outside that set. If your environment is shared with another project, choose a new dedicated `ST_PYTHON` path. If setup reports an incompatible Python version, also choose a fresh environment path; preserve the old environment.

If you previously ran `uv add chumpy` or another `uv add` command, inspect possible changes to the repository's root dependency files:

```bash
git diff -- pyproject.toml uv.lock
```

Those additions are unnecessary for the standalone study setup. Review the diff and remove only accidental changes, preserving other work; do not use a broad `git reset` to undo the earlier instructions.

Use the commands under `slurm/synthetic-training/` for this study. The launcher under `slurm/future-innovation-scaling/` runs a separate experiment and is not part of this setup.

## 1. Set the project, Python, and run paths

Use absolute paths available on both login and compute nodes. Change the examples to match your installation, and choose a fresh run directory for a new experiment.

```bash
cd "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
export GAVD6_ROOT="$PWD"
export ST_PYTHON="/hai/scratch/$USER/envs/synthetic-training/bin/python"
export ST_MODEL_ROOT="/hai/scratch/$USER/models"
export ST_RUN_ROOT="$GAVD6_ROOT/outputs/synthetic-training/pilot-01"
export ST_CONFIG="$ST_RUN_ROOT/config.json"
unset ST_STUDENT_ID ST_DEPENDENCY ST_NOTEBOOK_OUTPUT_DIR PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
```

| Variable | Meaning |
| --- | --- |
| `GAVD6_ROOT` | This checkout, containing `src`, `manifests`, and `slurm`. |
| `ST_PYTHON` | The study environment's **Python executable**, not its directory. Every executed notebook uses it. |
| `ST_MODEL_ROOT` | Parent of `pose/` and `mmpose/`. Keep your existing location; step 4 writes these paths into the configuration. |
| `ST_RUN_ROOT` | Shared output directory for every stage of this experiment. |
| `ST_CONFIG` | Path to the run's JSON **file**, not a directory. Step 4 creates it or corrects an existing file before submission. |

If you already use a separate `pilot.json`, set `ST_CONFIG` to that existing file instead. Keep the same selection for all stages. The repository's `slurm/synthetic-training/pilot.example.json` is a template; its relative `models/...` paths point inside `GAVD6_ROOT`, which may differ from your shared model directory.

Keep exports in the shell that calls `submit.sh`. Save them in a shell file and `source` it after reconnecting. Slurm captures exported values when you submit; later shell changes do not update queued jobs.

## 2. Install the locked study environment and pose models

The study has its own [pyproject.toml](pyproject.toml) and [uv.lock](uv.lock). They select **Python 3.11 on Linux x86-64**, with the CUDA 12.1 build of PyTorch 2.1 and the matching MMCV wheel. Keep the `ST_PYTHON` path from step 1; the setup script creates that environment if it is missing or synchronizes an existing one.

You need `git`, `curl`, `unzip`, the standard Linux `realpath` and `flock` commands, and **uv >=0.12.15,<0.13**. Check an existing installation with `uv --version`. If uv is missing or outside that range, install the tested version:

```bash
curl -LsSf https://astral.sh/uv/0.12.15/install.sh | sh
source "$HOME/.local/bin/env"
uv --version
```

Then run these two commands from the gavd6 checkout:

```bash
bash slurm/synthetic-training/setup-environment.sh
bash slurm/synthetic-training/download-students.sh
```

The first command runs `uv sync --locked` against the **study's** manifest and lockfile, targets `ST_PYTHON`, and checks the installed runtime. It handles old `UV_CONSTRAINT`, `UV_NO_CONFIG`, and other installation overrides internally. You do not need to activate the environment, create it manually, or install packages one at a time. The repository's root manifest and `.venv` belong to a different Torch stack; use the wrapper above for this study.

Use this environment only for synthetic training. Synchronization removes packages outside the study's locked dependency set, so do not point `ST_PYTHON` at an environment shared with another project. Run setup before submitting jobs, or after jobs using that environment have finished.

The second command downloads the matching MMPose configuration checkout and five released pose checkpoints under `ST_MODEL_ROOT`. It checks the checkout against the package's locked revision and preserves existing files. A different revision or edited configuration requires a clean matching checkout or a fresh `ST_MODEL_ROOT`; update the run configuration afterward. MMPose's Python package is already installed by setup; no editable installation is needed.

```text
ST_MODEL_ROOT/
├── mmpose/configs/          # Model configuration files
└── pose/
    ├── rtmpose-m.pth        # Fitting student
    ├── hrnet-w32.pth        # Fitting student
    ├── rtmpose-s.pth        # Validation student
    ├── hrnet-w48.pth        # Validation student
    └── vitpose-base.pth     # Held architecture
```

`Keeping existing ...` means that checkpoint file exists and was not downloaded again. It does not check its integrity or load the model. The downloader does not edit `ST_CONFIG`; complete step 4 even when all five weights already exist. Datasets, body models, textures, and V-JEPA remain separate assets in step 3.

The installed versions include:

| Component | Study version |
| --- | --- |
| Python | 3.11 |
| PyTorch / torchvision | 2.1.0+cu121 / 0.16.0+cu121 |
| MMCV / MMPose | 2.1.0 / 1.3.2 |
| NumPy / SciPy | 1.26.4 / 1.14.1 |
| Human Body Prior | 2.2.2.0, from the locked Git revision |
| Chumpy | 0.71, from the locked Git revision |

The manifest supplies Chumpy's missing build dependency and retains the compatible Human Body Prior revision. It also pins Setuptools to retain `pkg_resources`, which this MMEngine release still uses. The earlier [body-model](../../docs/studies/synthetic-training/body-model-dependency-repair.md) and [Chumpy](../../docs/studies/synthetic-training/chumpy-installation-repair.md) investigations explain why these choices are necessary; their manual repair commands have been replaced by the setup script.

To check an existing environment without installing or removing packages:

```bash
bash slurm/synthetic-training/setup-environment.sh --check
```

CUDA may be unavailable on a login node. Inside an existing GPU allocation, additionally require a working CUDA runtime:

```bash
bash slurm/synthetic-training/setup-environment.sh --check --require-cuda
```

From a login node, request a short allocation for that check with:

```bash
srun --account="${ST_ACCOUNT:-mind}" --partition="${ST_PARTITION:-hai}" \
  --gres=gpu:h100:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
  bash slurm/synthetic-training/setup-environment.sh --check --require-cuda
```

These checks do not establish that the licensed assets, pose checkpoints, V-JEPA forward pass, or EGL rendering work. Continue with asset preparation and notebook 00 before starting the source experiment. The [validation record](../../docs/studies/synthetic-training/locked-environment-validation.md) distinguishes local checks from the remaining HAIC/GPU validation.

If installation is interrupted, or a previous manual installation failed, fix the reported cause and rerun `setup-environment.sh` with the same `ST_PYTHON`. Keep the existing environment. If only the asset download fails, rerun `download-students.sh`. Do not run `uv add`, root-level `uv sync`, or an unpinned package upgrade to repair this study environment. Changes to dependencies belong in the study manifest and regenerated lockfile, followed by validation; routine setup uses the checked-in lock unchanged.

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

AMASS, licensed body models, and compatible texture/UV assets require separate preparation; these launchers do not supply them. Reuse existing installations. The texture images must suit the UV layout, and the UV template's face indices must exactly match the loaded SMPL-H mesh. The body-model directory alone does not supply these three rendering paths; an empty directory will not work. Keep GAVD evaluation recordings out of the source background collection.

Slurm defaults to `PYOPENGL_PLATFORM=egl` for headless rendering; unset an old override or set it to `egl`. The GPU node must have working EGL support.

If COCO is missing, download and extract the [official archives](https://cocodataset.org/#download). Reuse a shared copy when available; the image archive is large.

```bash
mkdir -p "/hai/scratch/$USER/coco"
curl --fail --location https://s3.amazonaws.com/images.cocodataset.org/zips/train2017.zip \
  --output "/hai/scratch/$USER/coco/train2017.zip" && \
unzip -n "/hai/scratch/$USER/coco/train2017.zip" -d "/hai/scratch/$USER/coco"
curl --fail --location https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip \
  --output "/hai/scratch/$USER/coco/annotations_trainval2017.zip" && \
unzip -n "/hai/scratch/$USER/coco/annotations_trainval2017.zip" -d "/hai/scratch/$USER/coco"
```

### V-JEPA and the image comparator

The example configuration uses the [official V-JEPA 2.1 ViT-B checkpoint](https://github.com/facebookresearch/vjepa2#v-jepa-21-pretrained-checkpoints). Set existing locations, or use these locations and download commands:

```bash
export ST_CONTEXT_REPO="${ST_CONTEXT_REPO:-${VJEPA2_ROOT:-$ST_MODEL_ROOT/vjepa2}}"
export ST_CONTEXT_CHECKPOINT="${ST_CONTEXT_CHECKPOINT:-$ST_MODEL_ROOT/vjepa2_1_vitb_dist_vitG_384.pt}"
export ST_IMAGE_CHECKPOINT="${ST_IMAGE_CHECKPOINT:-$ST_MODEL_ROOT/pose/resnet50-11ad3fa6.pth}"

# Clone only if the author checkout is absent. Skip downloads for existing weights.
[[ -d "$ST_CONTEXT_REPO" ]] || git clone --depth 1 https://github.com/facebookresearch/vjepa2.git "$ST_CONTEXT_REPO"
mkdir -p "$(dirname "$ST_CONTEXT_CHECKPOINT")" "$(dirname "$ST_IMAGE_CHECKPOINT")"
if [[ ! -f "$ST_CONTEXT_CHECKPOINT" ]]; then
  curl --fail --location https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
    --output "$ST_CONTEXT_CHECKPOINT.part" && mv "$ST_CONTEXT_CHECKPOINT.part" "$ST_CONTEXT_CHECKPOINT"
fi
if [[ ! -f "$ST_IMAGE_CHECKPOINT" ]]; then
  curl --fail --location https://download.pytorch.org/models/resnet50-11ad3fa6.pth \
    --output "$ST_IMAGE_CHECKPOINT.part" && mv "$ST_IMAGE_CHECKPOINT.part" "$ST_IMAGE_CHECKPOINT"
fi
```

The author checkout must contain `vjepa2_1_vit_base_384` in `hubconf.py`. The pilot matches this checkpoint with `context_builder="vjepa2_1_vit_base_384"`, `context_checkpoint_key="ema_encoder"`, `context_image_size=384`, and `context_frames=64`. If using different weights, edit the matching settings in the JSON selected by `ST_CONFIG` before preparing data. The key, image size, and frame count have no environment-variable overrides.

`ST_IMAGE_CHECKPOINT` adds the [ResNet-50 V2 comparator](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) alongside V-JEPA. Set it before `data`; leave it unset only when deliberately omitting that comparison.

## 4. Create or correct the run configuration

Choose the file in step 1, then run this after downloading the five pose models. The filenames have distinct uses:

| File | Use |
| --- | --- |
| [`pilot.example.json`](pilot.example.json) | Repository template. Leave it unchanged; the command below copies its defaults only for a new run. |
| `$ST_RUN_ROOT/config.json` | Default input in this guide. Inventory also creates this file if it is absent. Reuse it when it is your only saved configuration. |
| `$ST_RUN_ROOT/pilot.json` | Optional separate input used by earlier instructions. The filename is not required; select it explicitly if it contains your settings. |

The command preserves the selected file's settings, validates all ten student paths, and backs up an existing file before updating it. If the selected file is absent, it starts from saved `config.json`, or from the template for a new run. Use this for initial setup or path corrections before training; see [rerun rules](#8-find-outputs-or-rerun-one-stage) for changes after collecting outcomes.

```bash
"$ST_PYTHON" - <<'PY'
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

root = Path(os.environ["GAVD6_ROOT"]).resolve()
models = Path(os.environ["ST_MODEL_ROOT"]).expanduser().resolve()
destination = Path(os.environ["ST_CONFIG"]).expanduser().resolve()
template_path = root / "slurm/synthetic-training/pilot.example.json"
if destination == template_path.resolve():
    raise SystemExit('Select "$ST_RUN_ROOT/config.json", not the repository template.')
saved = Path(os.environ["ST_RUN_ROOT"]) / "config.json"
source = destination if destination.is_file() else saved if saved.is_file() else template_path
config = json.loads(source.read_text())
template = json.loads(template_path.read_text())
expected = {student["student_id"]: student for student in template["students"]}
if sorted(student["student_id"] for student in config["students"]) != sorted(expected):
    raise SystemExit("Expected the five pilot students; no changes made. Map custom students manually.")

for student in config["students"]:
    for key in ("config", "checkpoint"):
        relative = Path(expected[student["student_id"]][key]).relative_to("models")
        path = models / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise SystemExit(f"Missing or empty file: {path}\nNo changes made.")
        student[key] = str(path)
        print(f"OK: {student['student_id']} {key}: {path}")

if destination.exists():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = destination.with_name(f"{destination.name}.{stamp}.bak")
    shutil.copy2(destination, backup)
    print(f"Backup: {backup}")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(config, indent=2) + "\n")
print(f"Settings read from: {source}")
print(f"Updated: {destination}")
PY
```

Continue after it prints `Updated:`. For example, with `ST_MODEL_ROOT=/hai/scratch/tedmui/models`, RTMPose-m's checkpoint becomes `/hai/scratch/tedmui/models/pose/rtmpose-m.pth`. A relative `models/pose/rtmpose-m.pth` would instead point inside the gavd6 checkout. Exporting `ST_MODEL_ROOT` alone does not change existing JSON paths.

Keep the [pilot defaults](pilot.example.json) for the first pass: 4 clips per lesson, 64 frames per clip, 10 probe updates, 25/75 adaptation updates, and batches of 20 images. These are small trial settings, not a validated training budget.

**Configuration rule:** supported `ST_*` overrides take precedence over JSON; JSON paths are relative to `GAVD6_ROOT` and do not expand `$USER` or `$ST_MODEL_ROOT`. Keep the selected `ST_CONFIG` and your asset exports in the submitting shell. If `ST_CONFIG` is unset, the workflow reads `$ST_RUN_ROOT/config.json`. Inventory writes that file only when absent, so a separate corrected `pilot.json` does not refresh an older `config.json`. Use `unset`, not an empty export, to remove an override.

## 5. Inspect the inventory, then run the source experiment

First check the effective student paths and rendering inputs in the submitting shell. This reads the selected configuration without starting a job or loading a model:

```bash
"$ST_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ["GAVD6_ROOT"]) / "src"))
from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig

print("Configuration:", os.environ["ST_CONFIG"])
cfg = RunConfig.from_env()
for student in cfg.students:
    for field in ("config", "checkpoint"):
        path = Path(student[field])
        print(f"{student['student_id']} {field}: exists={path.is_file()}  {path}")
for field in ("render_texture_dir", "render_background_dir", "render_uv_path"):
    value = getattr(cfg, field)
    path = Path(value) if value else None
    if field == "render_uv_path":
        ready = path is not None and path.is_file() and path.suffix.lower() in {".obj", ".npz"}
    else:
        ready = path is not None and path.is_dir() and any(
            p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            for p in path.rglob("*")
        )
    print(f"{field}: available={ready}  {value or '(unset)'}")
PY
```

If student paths still point inside the checkout instead of your shared model directory, repeat step 4 with the same `ST_CONFIG`. Configure missing rendering inputs in step 3. Then submit inventory:

```bash
bash slurm/synthetic-training/submit.sh inventory --dry-run
bash slurm/synthetic-training/submit.sh inventory
```

Wait for completion and open the newest timestamped notebook under `$ST_RUN_ROOT/notebook_runs/run-00/`. Names sort by UTC start time: `20260917T171723416837Z` is later than `20260917T084858134020Z`. Check its completion metadata and asset table; an older copied notebook will retain its original results.

To print the latest inventory's complete paths without notebook table truncation:

```bash
"$ST_PYTHON" - <<'PY'
import os
from pathlib import Path
import pandas as pd

path = Path(os.environ["ST_RUN_ROOT"]) / "asset_inventory.csv"
print(pd.read_csv(path, keep_default_na=False).to_string(index=False))
PY
```

| Result | Action |
| --- | --- |
| Ten student entries are `True` | The five configurations and five checkpoints exist at their selected paths. Actual loading is still untested. |
| Student entries are `False` despite `Keeping existing` from the downloader | Check the full paths and selected `ST_CONFIG`; step 4 corrects the JSON. A completed job may still have used the template or older settings. |
| Three rendering entries are blank/`False` | Set `ST_TEXTURE_DIR`, `ST_BACKGROUND_DIR`, and `ST_UV_PATH` to real assets. Model downloads do not configure them. |
| Notebook says `completed` with missing assets | Inventory reports missing files without raising an error. Fix them and rerun `inventory` before continuing. |

Every required asset must be available before its dependent stage. Inventory checks existence; it does not validate image contents, UV compatibility, body-model contents, checkpoint loading, or CUDA. Manifest row counts also do not prove every sequence is available.

**For your first run, execute notebook 01 separately after fixing inventory:**

```bash
unset ST_STUDENT_ID ST_DEPENDENCY ST_NOTEBOOK_OUTPUT_DIR
bash slurm/synthetic-training/submit.sh data
```

Wait for success, then inspect rendered people, projected landmarks, and context extraction outputs. Before collecting student trials, confirm the RGB lessons and labels are usable. Human GAVD landmark annotations are prepared later in steps 6–7.

Submit notebook 02 for all four source students:

```bash
bash slurm/synthetic-training/submit.sh trials
```

Wait for **every** array task to succeed, then fit the selectors:

```bash
bash slurm/synthetic-training/submit.sh fit
```

After fitting succeeds, generate and read the source report:

```bash
bash slurm/synthetic-training/submit.sh report
```

**For a fresh run with an already validated setup**, the full source chain can submit these dependencies automatically:

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
| `config.json` | Run configuration; created by inventory if absent. A separate input may be selected by `ST_CONFIG`. |
| `asset_inventory.csv` | Full paths and existence flags from the latest inventory execution; overwritten on each inventory run. |
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

Executed notebook links are relative to their original output directory. When copying an attempt to a different directory depth, adjust only its Markdown link targets to the local checkout; preserve code, outputs, execution counts, and metadata. Use the [source notebook guide](../../notebooks/synthetic_training/README.md) for navigation until those links are corrected. The repository's [v1 inventory copy](../../notebook_runs/synthetic-training/run-00-v1/00_question_and_assets.ipynb) and [v2 inventory copy](../../notebook_runs/synthetic-training/run-00-v2/00_question_and_assets.ipynb) retain historical results; they are not live readiness checks.

To run a notebook directly in an appropriate allocation:

```bash
export PYOPENGL_PLATFORM=egl
"$ST_PYTHON" scripts/research_directions/synthetic_training/execute_notebook.py \
  --notebook 00 --run-root "$ST_RUN_ROOT"
```

The executor inherits `ST_CONFIG`, or reads the saved configuration if it is unset. Use `--student-id` for 02/05/08 and `--evaluation-split early` or `confirmation` for 06. For interactive work, launch `"$ST_PYTHON" -m jupyter lab notebooks/synthetic_training` and select the study environment's kernel. If missing, register it once with `"$ST_PYTHON" -m ipykernel install --user --name synthetic-training --display-name "Synthetic training"`.
