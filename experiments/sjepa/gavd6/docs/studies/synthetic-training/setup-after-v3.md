These instructions complete the setup shown in `run-00-v3` using your existing HAIC paths. The main corrections are three missing rendering paths, an explicit DMPL path, and `ST_GAVD_RESERVATION` pointing to a CSV instead of a directory. Source readiness is established by a real GPU preflight before the next notebook submission. Full real-data scoring additionally requires manually checked GAVD crops and independent human annotations; no environment export creates those inputs.

Run each step only after the previous command succeeds. These instructions assume `pilot-01` has inventory attempts but no collected source outcomes or frozen selectors. The configuration-saving step rejects a run that already contains scientific results.

1. Copy the new setup helpers and environment profile from your Mac to HAIC. Run this on the Mac, where these files were created:

```bash
cd /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6
scp scripts/research_directions/synthetic_training/prepare_render_assets.py \
    scripts/research_directions/synthetic_training/preflight_experiment.py \
    scripts/research_directions/synthetic_training/save_effective_config.py \
    tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/scripts/research_directions/synthetic_training/
scp slurm/synthetic-training/pilot-01.env \
    tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/slurm/synthetic-training/
ssh tedmui@haic.stanford.edu
```

All remaining setup/launch commands run on HAIC unless explicitly marked Mac. These copies do not update other repository files; they assume HAIC has the same synthetic-training implementation used by the v3 notebook.

2. Load the complete environment profile:

```bash
cd "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6"
source slurm/synthetic-training/pilot-01.env
```

The [profile](../../../slurm/synthetic-training/pilot-01.env) contains your original exports and the following additions/corrections:

```bash
export ST_DMPL_ROOT="$ST_BODY_MODEL_ROOT/dmpls"
export ST_TEXTURE_DIR="$ST_MODEL_ROOT/synthetic-rendering/smplitex/textures"
export ST_UV_PATH="$ST_MODEL_ROOT/synthetic-rendering/smplitex/smpl_uv.obj"
export ST_BACKGROUND_DIR="$ST_MODEL_ROOT/synthetic-rendering/coco-backgrounds"
export ST_GAVD_RESERVATION="$GAVD6_ROOT/outputs/future-innovation/learning-curve/config/source-reservation.csv"
export ST_CONTEXT_KIND="vjepa"
export ST_CONTEXT_BUILDER="vjepa2_1_vit_base_384"
export ST_DEVICE="cuda"
export PYOPENGL_PLATFORM="egl"
export PYTHONNOUSERSITE=1
export ST_ACCOUNT="mind"
export ST_PARTITION="hai"
export ST_MAX_PARALLEL=4
```

In particular, preserve `ST_AMASS_ROOT="$AMASS_ROOT/extracted"` and your existing checkpoint path `ST_CONTEXT_CHECKPOINT="$ST_MODEL_ROOT/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"`. The profile clears old per-student, dependency, output-directory, and training-budget shell overrides so that the JSON controls the pilot budgets. Change the account/partition if your allocation differs. Source this profile after each new SSH login and before submitting jobs; Slurm captures exported values at submission time.

3. Verify the existing environment and the previously supplied assets:

```bash
bash slurm/synthetic-training/setup-environment.sh --check
```

If the environment check fails, synchronize this dedicated environment, then rerun the check:

```bash
bash slurm/synthetic-training/setup-environment.sh
```

Do not resynchronize while jobs are using that environment. The wrapper uses the study lockfile; do not use the repository-root environment or install unpinned packages to repair it. v3 already found all ten pose-model files, so redownloading them is unnecessary. `download-students.sh` remains available if a file is actually absent.

Check the reservation and licensed body-model layout before downloading appearance assets:

```bash
"$ST_PYTHON" - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['ST_BODY_MODEL_ROOT'])
dmpl = Path(os.environ['ST_DMPL_ROOT'])
paths = [Path(os.environ['ST_GAVD_RESERVATION'])]
paths += [root / 'smplh' / gender / 'model.npz' for gender in ('male', 'female')]
paths += [dmpl / gender / 'model.npz' for gender in ('male', 'female')]
missing = []
for path in paths:
    ready = path.is_file() and path.stat().st_size > 0
    print(('OK  ' if ready else 'MISSING  ') + str(path))
    if not ready:
        missing.append(path)
if missing:
    print('Existing reservation candidates:')
    for path in sorted((Path(os.environ['GAVD6_ROOT']) / 'outputs').rglob('source-reservation.csv')):
        print(path)
    raise SystemExit('Correct the missing paths in pilot-01.env, source it again, and repeat this check.')
PY
```

If the reservation is elsewhere, set `ST_GAVD_RESERVATION` in the profile to the **actual existing reservation from the earlier FI study**, using the printed candidate paths and that study's provenance. Do not invent a new CSV or select an arbitrary older reservation. If the file cannot be recovered, GAVD evaluation setup remains incomplete.

If body files live elsewhere, change `ST_BODY_MODEL_ROOT`/`ST_DMPL_ROOT` to their real locations. The body loader also accepts a root directly containing `male/` and `female/`; in that case adapt the preliminary check above to omit `/smplh`. If files are missing rather than misplaced, obtain the AMASS-compatible SMPL+H model and DMPL package from the [official AMASS body-model instructions](https://github.com/nghorbani/amass#body-models). This study needs 16 shape components and 8 active DMPL components; changing environment variables cannot supply those files.

4. Populate the three appearance paths with actual assets:

```bash
"$ST_PYTHON" scripts/research_directions/synthetic_training/prepare_render_assets.py
```

This helper downloads the author's UV template and all 250 published [SMPLitex generated body textures](https://dancasas.github.io/projects/SMPLitex/SMPLitex-dataset.html), using pinned repository revisions. It verifies the UV checksum and decodes downloaded images. It copies 256 deterministic photographic backgrounds from your existing COCO training images, choosing image IDs absent from person-keypoint annotations. Those backgrounds are separate from the person-annotated images eligible for pose replay; absence of an annotation is not a guarantee that no person appears in an image. No GAVD frames are used as backgrounds.

The appearance package is roughly 100 MB of textures plus the local COCO copies. It retains a provenance JSON beside the UV file. Exact compatibility with your licensed body models is checked in the next step.

Expected successful marker: `APPEARANCE_ASSETS_PREPARED: 250 textures, 256 photographic backgrounds`. Rerunning preserves and validates existing images. It rejects unexpected images in the dedicated folders rather than silently mixing libraries.

5. Persist the effective settings into the selected run configuration. This prevents a later process from falling back to an older JSON after environment overrides disappear:

```bash
"$ST_PYTHON" scripts/research_directions/synthetic_training/save_effective_config.py
```

This preserves existing pilot budgets and student roles, fixes the model paths from the known template, records all effective asset exports, and backs up the original JSON. Review the printed settings. The run retains four source students and one held ViTPose architecture.

6. Verify actual inputs and GPU behavior before running another notebook:

```bash
"$ST_PYTHON" scripts/research_directions/synthetic_training/preflight_experiment.py --assets-only
```

Require `ASSET_PREFLIGHT_PASSED`. This checks file types, image decoding, the actual selected AMASS windows/pools, SMPL-H/DMPL files, exact UV face compatibility, every selected COCO replay image, and the GAVD reservation schema/available recording count. It does not inspect human reference coordinates. By default, at least 78 permitted available recording IDs are necessary for the six GAVD groups, although view coverage and related-recording independence still require later checks.

Then request an H100 for real loading, rendering, inference, and an update:

```bash
srun --account="$ST_ACCOUNT" --partition="$ST_PARTITION" \
  --gres=gpu:h100:1 --cpus-per-task=8 --mem=64G --time=00:45:00 \
  bash -c 'set -euo pipefail
    cd "$GAVD6_ROOT"
    bash slurm/synthetic-training/setup-environment.sh --check --require-cuda
    "$ST_PYTHON" scripts/research_directions/synthetic_training/preflight_experiment.py'
```

The 45 minutes is an allocation limit, not a measured runtime estimate. Require `SOURCE_PREFLIGHT_PASSED`. The full check renders real AMASS body motion through EGL, runs ordered and shuffled V-JEPA features plus the image comparator, and loads every student for prediction and one mixed supervised update at the configured batch size. It verifies finite outputs and changed head parameters. Updates are discarded; artifacts stay under `$ST_RUN_ROOT/preflight/<timestamp>/`.

Inspect `render-male-overlay.png` and `render-female-overlay.png` from the latest successful attempt, if both genders are selected. Check recognizable body texture, reasonable camera orientation, landmarks aligned with the body, and sensible visible/hidden markers (green/red). To view the artifacts locally, run on your Mac:

```bash
scp -r tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01/preflight \
    "$HOME/Downloads/synthetic-training-preflight"
open "$HOME/Downloads/synthetic-training-preflight"
```

Resolve any preflight failure and rerun preflight before submitting notebooks. A successful single update does not establish scientific benefit or guarantee the runtime of every longer stage, but it exercises the external dependencies that inventory omitted. These helpers have passed local CPU tests; their HAIC/GPU execution is for this step.

7. Launch the source experiment once preflight and visual inspection pass. Run on HAIC:

```bash
cd "$GAVD6_ROOT"
bash slurm/synthetic-training/submit.sh source --dry-run
```

The dry run should show inventory, data preparation, a four-student trial array, selector fitting, and the report, with dependencies. Then:

```bash
bash slurm/synthetic-training/submit.sh source
squeue -u "$USER"
```

The chain is `00 → 01 → 02 array → 03 → 07`; the next notebook is inventory 00 at the start of this source run. It is no longer being used as the only readiness test. Fitting waits for all source students. All attempts are saved separately, and later jobs run only if their predecessors succeed. The source report is the decision point before real deployment.

If you want to inspect notebook 01 before queuing student training, use `bash slurm/synthetic-training/submit.sh data` instead of `source`, then submit `trials`, `fit`, and `report` individually, waiting for each stage to succeed. Do not submit both alternatives into the same run concurrently.

8. Complete the real-evaluation inputs before claiming the entire experiment is ready. After the source report supports continuation, the first GAVD preparation pass is:

```bash
unset ST_GAVD_VIEWS
bash slurm/synthetic-training/submit.sh gavd
```

The initial JSON must also have `gavd_view_csv` empty. The supplied v3 setup did not name a view CSV; if the printed JSON in step 5 already contains one, retain it only if it is the checked panel intended for this run.

The first pass creates `$ST_RUN_ROOT/data/gavd-view-template.csv`. Complete checked rows with `coarse_view` (`side`, `oblique`, `frontal_rear`), `view_checked=true`, `person_height_px`, and the four full-frame person-box coordinates. Mark recordings derived from the same original source with the same `related_recording_id`. Under pilot defaults, each of the six view/size groups needs 3 context, 4 early-reference, and 6 confirmation recordings. Selection enforces recording separation and rejects insufficient groups.

Once those manual checks are complete:

```bash
export ST_GAVD_VIEWS="$ST_RUN_ROOT/data/gavd-view-template.csv"
bash slurm/synthetic-training/submit.sh gavd
```

After that second job succeeds, submit deployment:

```bash
bash slurm/synthetic-training/submit.sh deploy
```

After **all five deployment array tasks** succeed, submit crossover:

```bash
bash slurm/synthetic-training/submit.sh crossover
```

For independent labels, copy the annotation pages and their sibling `gavd/` image directory to your Mac. Run on the Mac:

```bash
mkdir -p "$HOME/Downloads/synthetic-training-annotations"
scp tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01/data/annotate-early.html \
    tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01/data/annotate-confirmation.html \
    "$HOME/Downloads/synthetic-training-annotations/"
scp -r tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01/data/gavd \
    "$HOME/Downloads/synthetic-training-annotations/"
open "$HOME/Downloads/synthetic-training-annotations/annotate-early.html"
open "$HOME/Downloads/synthetic-training-annotations/annotate-confirmation.html"
```

On each page, draw an independent reference box, mark the twelve landmarks or their visibility, save progress, and complete review. Confirmation requires a reviewer different from the annotator. Export the resulting CSVs. The exported files must be named `gavd-early-annotations.csv` and `gavd-confirmation-annotations.csv`.

On HAIC, choose and create the annotation destination:

```bash
export ST_GAVD_ANNOTATIONS="$ST_RUN_ROOT/independent-annotations"
mkdir -p "$ST_GAVD_ANNOTATIONS"
```

Copy the exported CSVs into that directory. If your browser saves them in Downloads, the Mac command is:

```bash
scp "$HOME/Downloads/gavd-early-annotations.csv" \
    "$HOME/Downloads/gavd-confirmation-annotations.csv" \
    tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training/pilot-01/independent-annotations/
```

After crossover succeeds and the early annotations are available, on HAIC:

```bash
bash slurm/synthetic-training/submit.sh evaluate
```

Only when the separate confirmation evaluation is intended:

```bash
bash slurm/synthetic-training/submit.sh confirmation
```

Retain `ST_GAVD_VIEWS` and `ST_GAVD_ANNOTATIONS` in your saved environment profile once their files exist, or re-export them after reconnecting. They are the two remaining stage-specific path variables. The launcher sets the evaluation split; no manual `ST_EVALUATION_SPLIT` export is needed. Keep the source-selected method frozen after opening real outcomes.
