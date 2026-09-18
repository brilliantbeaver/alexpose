# Synthetic training v2 on HAIC

Use your existing synthetic-training environment, AMASS data, body models, rendering assets, and MMPose checkpoints. The launcher creates the v2 configurations and manages Slurm jobs and cost records. You do not need to copy JSON fragments or export settings for each stage.

The experiment prepares paired synthetic **2D body-12 tracks**, then compares eight temporal restoration methods. The image pose estimators stay frozen. Start with a small source run to check the pipeline.

**Update HAIC once, before starting a run.** If these changes are only on your Mac, run this from the local `gavd6` checkout:

```bash
bash slurm/synthetic-training-v2/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6
```

This previews one transfer containing the required code and verified historical evidence. Review the list, then repeat with `--apply`:

```bash
bash slurm/synthetic-training-v2/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6 --apply
```

The destination checkout must already exist. Existing study profiles are preserved; replaced files receive a backup. Transfer before initialization, since changing scientific code after a run starts changes its identity. If a local historical hash fails, use the [historical recovery guide](historical-audit.md).

All remaining commands run on **HAIC**, except the optional preview download in Step 3.

**1. Initialize one run using your existing environment.**

Your current `ST_*` exports can be used directly. After a fresh login, first source your usual `slurm/synthetic-training/pilot-01.env` profile. Then:

```bash
cd "$GAVD6_ROOT" &&
bash slurm/synthetic-training-v2/submit.sh init \
  --name source-smoke-01 --gpu-hours 4 --prior-gpu-hours 0 &&
source outputs/synthetic-training-v2/source-smoke-01/session.env
```

`source-smoke-01` is the new run directory name. Choose a fresh name; initialization will not overwrite an existing run. The example authorizes a **four-GPU-hour total scope** and declares zero earlier costs charged to that scope. Replace those values with your actual allowance and earlier setup/failed-allocation costs. Initialization submits no jobs. The initial preparation needs room for one one-hour allocation; training needs room for two more. See [cost accounting](costs.md) for carrying a previous run's costs forward.

The initializer checks the selected Python environment, generates the configurations and review worksheets, and saves a `session.env` that remembers the run. It reuses these paths:

| Existing setting | Used for |
| --- | --- |
| `GAVD6_ROOT` | The checkout you run from |
| `ST_PYTHON`, or an explicit `STV2_PYTHON` | Study interpreter |
| `ST_AMASS_ROOT` | Extracted AMASS motions |
| `ST_BODY_MODEL_ROOT`, `ST_DMPL_ROOT` | Licensed body models |
| `ST_UV_PATH`, `ST_TEXTURE_DIR`, `ST_BACKGROUND_DIR` | Rendering assets |
| `ST_MODEL_ROOT` | MMPose configuration files and checkpoints |
| `ST_ACCOUNT`, `ST_PARTITION`, if set | Slurm account and partition; defaults are `mind` and `hai` |

The generated estimator list contains **RTMPose-m, HRNet-W32, and ViTPose-Base**. ViTPose is excluded from temporal-model training and included in development evaluation. Its previous exposure still limits the claims you can make. COCO training, V-JEPA2, and GAVD video assets are not needed for this first source experiment.

Your supplied `ST_PYTHON` points to `envs/synthetic-training`. Its installed packages determine compatibility. If initialization rejects that interpreter and you already built the repaired environment, select it explicitly and repeat initialization:

```bash
export STV2_PYTHON="/hai/scratch/$USER/envs/synthetic-training-cu124/bin/python"
```

Require **Torch 2.6.0+cu124, Torchvision 0.21.0+cu124, and MMCV 2.1.0**. If the repaired environment is unavailable, follow Steps 4a–4c of the [original HAIC guide](../synthetic-training/README.md). Do not rebuild an environment that already passes its checks.

**2. Complete the two source reviews, then check the inputs.**

Initialization creates draft worksheets under:

```text
$STV2_WORK/inputs/review-drafts/
```

Follow [Prepare the reviewed AMASS inputs](source-inputs.md). Save the completed records as:

```text
$STV2_WORK/inputs/locomotion-audit.csv
$STV2_WORK/inputs/person-reservations.csv
```

For the first engineering check, aim for two eligible training people and two development people, with two reviewed windows per person. The source review must preserve original splits, known aliases, reservations, and exposure history. The old `ST_GAVD_RESERVATION` file describes video recordings and cannot replace the AMASS person file.

This review is the manual prerequisite: motion filenames alone cannot establish locomotion, and a generated configuration cannot establish permission to use a reserved person. The launcher leaves these decisions blank.

Now run:

```bash
bash slurm/synthetic-training-v2/submit.sh check
```

**Continue when `STV2_INPUTS_PASSED` appears and the printed people/window counts match your intended panel.** The check covers the environment, historical evidence, CSV joins, required files, estimator configurations, and the held family. It reports missing inputs together. GPU operators, rendering, and checkpoint execution are checked in the next step.

If your assets use nonstandard filenames, the generated file to inspect is `$STV2_WORK/config/preparation.json`. Normal setup does not require editing it. Correct asset paths before preparation; completed attempts keep their own configuration snapshots.

**3. Prepare the paired data and inspect the overlays.**

```bash
bash slurm/synthetic-training-v2/submit.sh prepare
bash slurm/synthetic-training-v2/submit.sh status
```

The launcher repeats the input checks, creates a fresh preparation scope, and submits **one H100, 8 CPUs, 96 GB, for at most one hour**. Within that allocation, preparation checks the CUDA operators, loads the body models and frozen estimators, renders the reviewed windows, and saves their tracks. A separate preflight submission is unnecessary.

`status` prints each job's state, exit code, log path, and paired-data directory. Run it again later to check progress. Require the preparation job to show **`COMPLETED` and `0:0`**, and inspect its `overlay-*.png` files. Check body orientation, framing, anatomical sides, and landmark placement. The source launcher also requires a successful `preparation-status.json` and a valid bundle.

Open the overlays through your remote file browser, or copy just the PNGs to your Mac. For the first attempt with the run name above, run on the **Mac**:

```bash
mkdir -p "$HOME/Downloads/stv2-source-smoke-01"
rsync -av \
  'tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/paired-01/overlay-*.png' \
  "$HOME/Downloads/stv2-source-smoke-01/"
open "$HOME/Downloads/stv2-source-smoke-01"
```

For a retry or another run name, use the paired-data directory printed by `status`. Preparation processes every admitted audit row; it does not automatically cap the panel. The one-hour limit is a bound, not a throughput estimate.

**4. Train, monitor, and read the results.**

After the overlays pass your review, run on **HAIC**:

```bash
bash slurm/synthetic-training-v2/submit.sh source --overlays-reviewed
bash slurm/synthetic-training-v2/submit.sh status
```

This records your overlay review and submits the sequential stage chain. It creates the source configuration and carries preparation costs forward automatically. Only `direct` and `jepa` request a GPU: each has one H100 and a one-hour limit. The remaining stages use CPUs. The first recipe has seed 17, 200 pretraining updates, and 200 readout updates. Failed attempts count toward the same allowance.

The first run includes unchanged/filter baselines and these learned comparisons:

| Methods | Purpose |
| --- | --- |
| `direct`, `smoothnet`, `static` | Practical restoration and coordinate-history controls |
| `initialized` | Readout from an untrained frozen encoder |
| `coordinate` | Clean-coordinate pretraining with a separately fitted readout |
| `ordinary_jepa` | Pretraining against observed-track latent targets |
| `paired_jepa` | Pretraining against aligned clean synthetic latent targets |
| `shuffled_jepa` | Test the value of correct pretraining pairs |

Require a successful completed attempt for every source stage. Earlier failed attempts remain in the history and budget. Then read:

```bash
less "$STV2_WORK/source-01/report.md"
```

The report links metrics, uncertainty, support counts, and scientific gates. A one-seed engineering run establishes feasibility. `insufficient_evidence` is expected when calibration, repeated seeds, or motion-reference support is missing; a successful Slurm job is not a scientific pass. These results concern synthetic 2D joint-center proxies with supplied boxes, not real gait measurement or 3D/world-model capability.

**After reconnecting.** Restore the saved run, then use the same commands:

```bash
source "/hai/scratch/$USER/alexpose/experiments/sjepa/gavd6/outputs/synthetic-training-v2/source-smoke-01/session.env"
cd "$STV2_ROOT"
bash slurm/synthetic-training-v2/submit.sh status
```

For a preview, add `--dry-run` to `prepare` or `source`; the source preview also needs `--overlays-reviewed`. Previews do not submit jobs or create phase configurations. Actual GPU execution remains the allocated job's check.

**If something fails.**

| Symptom | Next step |
| --- | --- |
| Missing review CSVs | Complete the worksheets in [source inputs](source-inputs.md), then run `check`. |
| Historical hash failure | Use the [historical recovery guide](historical-audit.md); retain the original recorded hashes. |
| Environment check fails | Select the verified `STV2_PYTHON`, or use the original guide's environment repair. |
| Missing model/render assets | Reuse the original guide's asset helpers; v2 does not need `save_effective_config.py`. |
| Preparation fails | Read its printed log, correct the cause, then use `prepare --retry`. |
| Training fails or leaves pending dependencies | Use the [retry instructions](costs.md#retry-a-failed-job); preserve the recorded jobs and costs. |
| Budget is insufficient or a scientific recipe must change | Start a fresh named run with the prior cost ledger; see [cost accounting](costs.md). |
| Uncertain scheduler response | Use [submission recovery](costs.md#recover-an-uncertain-submission); do not repeatedly submit the same job. |

For scientific extensions and claim limits, read the [development protocol](../../docs/studies/synthetic-training-v2/protocol.md) and [review findings](../../docs/studies/synthetic-training-v2/haic-workflow-review.md). The [notebook guide](../../notebooks/synthetic_training_v2/README.md) is a CPU fixture walkthrough; source execution uses this Slurm workflow.
