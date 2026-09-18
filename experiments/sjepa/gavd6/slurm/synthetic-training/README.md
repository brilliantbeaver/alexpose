# Synthetic training on HAIC

Keep the existing `pilot-01` configuration, datasets, and checkpoints. Follow these steps in order; continue only after each command succeeds. The source experiment can run after preflight passes. Real GAVD evaluation also requires checked crops and independent human annotations.

**Already blocked at GPU preflight? Resume at Step 4.** The current HAIC setup requires **`torch==2.6.0+cu124`** and **`torchvision==0.21.0+cu124`**.

**1. Load the environment.** Use an updated gavd6 checkout on HAIC. If the new helpers are still only on your Mac, copy them first from the local gavd6 directory. The second command copies the profile only if HAIC does not already have one:

```bash
scp scripts/research_directions/synthetic_training/{prepare_render_assets,save_effective_config,preflight_experiment}.py \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/scripts/research_directions/synthetic_training/
rsync -av --ignore-existing slurm/synthetic-training/pilot-01.env \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/slurm/synthetic-training/
```

On HAIC:

```bash
cd "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
source slurm/synthetic-training/pilot-01.env
```

The [profile](pilot-01.env) contains the Python, AMASS, COCO, GAVD, and encoder paths. For an existing profile, add or update the following exports while preserving your other paths:

```bash
export ST_DMPL_ROOT="$ST_BODY_MODEL_ROOT/dmpls"
export ST_TEXTURE_DIR="$ST_MODEL_ROOT/synthetic-rendering/smplitex/textures"
export ST_UV_PATH="$ST_MODEL_ROOT/synthetic-rendering/smplitex/smpl_uv.obj"
export ST_BACKGROUND_DIR="$ST_MODEL_ROOT/synthetic-rendering/coco-backgrounds"
export ST_GAVD_RESERVATION="$GAVD6_ROOT/outputs/future-innovation/learning-curve/config/source-reservation.csv"
```

Edit the profile if an asset lives elsewhere, then source it again. `ST_GAVD_RESERVATION` must be the actual previous FI reservation **CSV**, not its parent directory. Source the profile after every login; Slurm captures exported settings when you submit.

**2. Check the Python environment.** All commands below run on HAIC unless marked Mac.

```bash
bash slurm/synthetic-training/setup-environment.sh --check
```

If the environment is missing or package checks fail, complete **Steps 4a–4c** to build it on a compute node, then return to Step 3. Use the profile's dedicated `ST_PYTHON`.

Setup needs `git`, `curl`, `realpath`, `flock`, and **uv >=0.12.15,<0.13**. If uv is missing or incompatible:

```bash
curl -LsSf https://astral.sh/uv/0.12.15/install.sh | sh
source "$HOME/.local/bin/env"
```

**3. Prepare assets and save the configuration.** First check the existing reservation and body models:

```bash
ls -lh "$ST_GAVD_RESERVATION" \
  "$ST_BODY_MODEL_ROOT/smplh/male/model.npz" \
  "$ST_BODY_MODEL_ROOT/smplh/female/model.npz" \
  "$ST_DMPL_ROOT/male/model.npz" \
  "$ST_DMPL_ROOT/female/model.npz"
```

Correct missing paths in the profile. If the reservation is elsewhere, locate the original file under the earlier FI run; do not create a replacement split. Missing licensed body files must be obtained using the [AMASS body-model instructions](https://github.com/nghorbani/amass#body-models). The study requires SMPL-H with 16 shape components and 8 DMPL components.

Then populate the rendering paths and persist the effective settings:

```bash
"$ST_PYTHON" scripts/research_directions/synthetic_training/prepare_render_assets.py && \
"$ST_PYTHON" scripts/research_directions/synthetic_training/save_effective_config.py
```

The first helper prepares 250 [SMPLitex body textures](https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset.html), their UV template, and 256 photographic backgrounds from existing COCO images without person-keypoint annotations. The second backs up `config.json`, saves the exports and matching V-JEPA settings, and preserves pilot budgets and student roles. It refuses configuration changes after scientific results exist. Existing pose checkpoints are reused; run `bash slurm/synthetic-training/download-students.sh` only if student files are missing.

**4. Repair the environment and pass GPU preflight.** Follow 4a–4e in order. This replaces the earlier CUDA 12.1 and manual MMCV rebuild instructions. Keep your existing `pilot-01` configuration, datasets, rendering assets, and model checkpoints.

The repair creates a new Python environment with **Torch 2.6.0+cu124**, its matching **Torchvision 0.21.0+cu124**, and **MMCV 2.1.0 compiled for H100**. The setup job also prepares CUDA toolkit 12.4.1 under your scratch directory. You do not need an existing Conda installation, administrator access, or a guessed `/usr/local/cuda-*` path.

**4a. Copy the updated setup files from your Mac.** Run this from the local `gavd6` checkout. The first command preserves the environment profile already on HAIC.

```bash
haic_root="tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6"
rsync -av --exclude=pilot-01.env --exclude=wheels/ --exclude=.venv/ --exclude=__pycache__/ \
  slurm/synthetic-training/ "$haic_root/slurm/synthetic-training/"
rsync -av scripts/research_directions/synthetic_training/check_environment.py \
  "$haic_root/scripts/research_directions/synthetic_training/"
rsync -av src/gavd6_sjepa/research_directions/synthetic_training/estimators.py \
  "$haic_root/src/gavd6_sjepa/research_directions/synthetic_training/"
```

These files travel together: the manifest and lock select Torch 2.6, the setup scripts build MMCV against it, and the estimator loader handles the released checkpoints' metadata under Torch 2.6.

**4b. Select the new environment on HAIC.** Open the saved profile:

```bash
cd "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
nano slurm/synthetic-training/pilot-01.env
```

Change its `ST_PYTHON` line to the following, then save the file. Keep your other asset paths.

```bash
export ST_PYTHON="/hai/scratch/$USER/envs/synthetic-training-cu124/bin/python"
```

Load the profile and clear the invalid compiler override from the earlier attempt:

```bash
source slurm/synthetic-training/pilot-01.env
unset CUDA_HOME
mkdir -p "$ST_RUN_ROOT/logs"
uv --version
```

Require **uv >=0.12.15,<0.13**; Step 2 gives the installation command if needed. The old Python environment remains available. The new toolkit and its package cache live under `/hai/scratch/$USER/toolchains/synthetic-training-cu124`.

**4c. Build and check the environment.** Submit this job on HAIC:

```bash
setup_job=$(sbatch --parsable \
  --account="$ST_ACCOUNT" --partition="$ST_PARTITION" \
  --export=ALL --chdir="$GAVD6_ROOT" \
  --output="$ST_RUN_ROOT/logs/setup-%j.log" \
  slurm/synthetic-training/setup-environment.sbatch)
setup_job="${setup_job%%;*}"
printf 'Setup job: %s\n' "$setup_job"
```

Require a numeric job ID. If `sbatch` reports an error or the ID is empty, stop and fix that submission error; no job log exists yet. Record the ID so you can find it after reconnecting.

This requests one H100, 8 CPUs, 64 GB of memory, and two hours. The first run downloads about 3.5 GiB for the toolkit, plus the Python packages, and compiles MMCV. Later runs reuse the installation and build cache. HAIC's `hai` partition accepts **`sbatch`** jobs; use the batch commands here.

Watch its progress:

```bash
squeue -j "$setup_job"
tail -F "$ST_RUN_ROOT/logs/setup-$setup_job.log"
```

The log appears when the job starts. Press Ctrl-C to stop watching; the job continues. After it finishes, check:

```bash
sacct -j "$setup_job" --format=JobID,State,ExitCode,Elapsed
```

**Continue only when the job is `COMPLETED` with exit code `0:0` and the log contains `ST_ENVIRONMENT_READY`.** This confirms the exact package versions, imports, and actual Torch/Torchvision/MMCV GPU operations. It does not yet check your rendering assets or model checkpoints.

If setup fails, read the first error in that log. Keep the log and existing directories; correct the reported problem before resubmitting the same command. The helper retries downloads and resumes recognized toolkit installations. If it reports an unrecognized partial toolkit directory, set `ST_TOOLCHAIN_ROOT` to a fresh scratch directory in `pilot-01.env`, source the profile, and resubmit. Do not return to the old wheel or CUDA 12.1 repair commands.

If you skipped Step 3 because Python was unavailable, complete Step 3 now.

**4d. Run the full GPU preflight.** Once the environment is ready and Step 3's assets/configuration exist, submit:

```bash
preflight_job=$(sbatch --parsable \
  --account="$ST_ACCOUNT" --partition="$ST_PARTITION" \
  --export=ALL --chdir="$GAVD6_ROOT" \
  --output="$ST_RUN_ROOT/logs/preflight-%j.log" \
  slurm/synthetic-training/preflight.sbatch)
preflight_job="${preflight_job%%;*}"
printf 'Preflight job: %s\n' "$preflight_job"
```

Require and record a numeric job ID, as in Step 4c. Then watch the job:

```bash
squeue -j "$preflight_job"
tail -F "$ST_RUN_ROOT/logs/preflight-$preflight_job.log"
```

This job has a 45-minute limit. It checks the environment, selected data, body/UV compatibility, EGL rendering, both context encoders, and prediction plus one disposable update for all five students.

After stopping the log viewer, check its final state:

```bash
sacct -j "$preflight_job" --format=JobID,State,ExitCode,Elapsed
```

**Require `COMPLETED`, exit code `0:0`, and `SOURCE_PREFLIGHT_PASSED`.** `ASSET_PREFLIGHT_PASSED` from an earlier CPU run is insufficient. A failure here names the next asset or runtime problem to fix; after correcting it, resubmit this preflight job. Rebuild the environment only if its package checks fail.

**4e. Inspect the rendered previews.** The successful log prints `Preflight artifacts:` followed by a new timestamped directory. Its `report.json` must have `"mode": "gpu"` and `"status": "passed"`.

On your Mac, copy the preflight outputs:

```bash
mkdir -p "$HOME/Downloads/synthetic-training-preflight"
rsync -av tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01/preflight/ \
  "$HOME/Downloads/synthetic-training-preflight/"
open "$HOME/Downloads/synthetic-training-preflight"
```

Open the timestamped folder named in the successful job's log. Inspect its `render-*-overlay.png` files: body appearance, camera orientation, and landmark alignment should look sensible. Continue to training only after both the automated checks and this visual check pass.

**5. Run the source experiment.** On HAIC, return to the checkout and reload the saved profile after any new login:

```bash
cd "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
source slurm/synthetic-training/pilot-01.env
bash slurm/synthetic-training/submit.sh source --dry-run
```

Check the printed run paths and jobs, then submit once:

```bash
bash slurm/synthetic-training/submit.sh source
squeue -u "$USER"
```

The launcher chains **00 inventory → 01 data → 02 four-student array → 03 fit → 07 report**. Each stage waits for the preceding stage to succeed. Job IDs are printed and saved in `$ST_RUN_ROOT/logs/submissions.tsv`; output and error logs are in `logs/`.

Use this check throughout Steps 5–7 to inspect the recorded jobs, including individual array tasks:

```bash
sacct --array \
  -j "$(cut -f3 "$ST_RUN_ROOT/logs/submissions.tsv" | paste -sd, -)" \
  --format=JobID,State%24,ExitCode,Elapsed
```

This includes earlier submissions and retries. Find the IDs just printed; every job and array task must be `COMPLETED` with `0:0`. A disappearing queue entry alone does not establish success.

Wait for the source chain to finish, including all four notebook 02 tasks. Read the executed notebook 07 under `$ST_RUN_ROOT/notebook_runs/run-07/` and its source results before proceeding to real deployment.

**6. Prepare the real-video panel, then deploy.** The commands in this step are separate submissions. **Wait for each job to complete successfully before running the next command.** Array jobs must succeed for every student.

First, create the view/crop template on HAIC:

```bash
export ST_GAVD_VIEWS=''
bash slurm/synthetic-training/submit.sh gavd
```

The explicit empty value requests a template even if `config.json` contains an older view-file path. After this job succeeds, edit `$ST_RUN_ROOT/data/gavd-view-template.csv`. Each row is a video sequence. Inspect candidates using their `video_path`, `first_frame`, and `last_frame`; leave unchecked rows as `view_checked=false`. For each sequence you check:

- Check `coarse_view`: `side`, `oblique`, or `frontal_rear`.
- Measure `person_height_px` and the crop box (`bbox_x1`, `bbox_y1`, `bbox_x2`, `bbox_y2`) in full-frame pixels.
- Set `view_checked=true` after checking the view and crop.
- Give recordings from the same source the same `related_recording_id`.

The script uses checked, available rows and selects at most one sequence per related recording. It assigns low/high size groups automatically using median person height. The three views × two sizes form six groups; each needs 3 context, 4 early, and 6 confirmation recordings: **78 selected recordings total**. You may need to check more than 78 rows to fill every group.

Run the second preparation pass:

```bash
export ST_GAVD_VIEWS="$ST_RUN_ROOT/data/gavd-view-template.csv"
bash slurm/synthetic-training/submit.sh gavd
```

Save that export in `pilot-01.env`. This second pass validates the panel and creates the annotation pages and images used in Step 7. If it reports a group with too few recordings, check more eligible sequences and resubmit this pass.

After it succeeds, deploy all five students:

```bash
bash slurm/synthetic-training/submit.sh deploy
```

After **all five deployment tasks succeed**, exchange their selected lessons:

```bash
bash slurm/synthetic-training/submit.sh crossover
```

Wait for **all five crossover tasks** to succeed before evaluation.

**7. Annotate and evaluate.** After the second GAVD preparation pass succeeds, copy the annotation pages and images to your Mac:

```bash
annotation_remote="tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01"
annotation_local="$HOME/Downloads/synthetic-training-annotations"
mkdir -p "$annotation_local"
scp "$annotation_remote/data/annotate-early.html" \
    "$annotation_remote/data/annotate-confirmation.html" "$annotation_local/"
mkdir -p "$annotation_local/gavd"
rsync -av "$annotation_remote/data/gavd/" "$annotation_local/gavd/"
open "$annotation_local/annotate-early.html"
open "$annotation_local/annotate-confirmation.html"
```

Enter your annotator name, draw independent reference boxes, and mark all twelve landmarks or their visibility for every frame. Use **Save progress JSON** regularly and before closing a page; the browser does not save your work automatically.

Confirmation requires a different reviewer. Give them the same HTML page, `gavd/` folder, and saved progress JSON. They use **Load progress**, inspect every frame, enter their own Reviewer name, and click **Mark frame reviewed** for every frame. Use **Export reference CSV** to produce `gavd-early-annotations.csv` and the reviewed `gavd-confirmation-annotations.csv`.

On HAIC, create their destination and save the export in `pilot-01.env`:

```bash
export ST_GAVD_ANNOTATIONS="$ST_RUN_ROOT/independent-annotations"
mkdir -p "$ST_GAVD_ANNOTATIONS"
```

Have the reviewer return the reviewed confirmation CSV. Place both annotation CSVs in your Mac's Downloads folder, then upload them from your Mac:

```bash
scp "$HOME/Downloads/gavd-early-annotations.csv" \
    "$HOME/Downloads/gavd-confirmation-annotations.csv" \
    "$annotation_remote/independent-annotations/"
```

Once crossover and annotation review are complete, evaluate the early split on HAIC:

```bash
bash slurm/synthetic-training/submit.sh evaluate
```

Inspect the early results. Keep the source-selected method frozen after viewing real outcomes. When ready to evaluate the separate confirmation split, run:

```bash
bash slurm/synthetic-training/submit.sh confirmation
```

All outputs remain under `ST_RUN_ROOT`: `preflight/` holds readiness reports and previews, `notebook_runs/` holds executed attempts, `logs/` holds Slurm logs, and `reports/` and `evaluation/` hold results. Retry failures with the same scientific configuration. Use a new run root when changing scientific conditions after results exist. See the [notebook guide](../../notebooks/synthetic_training/README.md) for interpretation.

The setup follows the official [Torch 2.6 / Torchvision 0.21 CUDA 12.4 pairing](https://pytorch.org/get-started/previous-versions/#v260) and [NVIDIA's versioned toolkit packages](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/index.html#conda-installation). MMCV stays at 2.1 because [MMDetection 3.3 requires MMCV below 2.2](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/__init__.py). Successful execution of Steps 4c–4e on HAIC is the runtime validation for your installation.
