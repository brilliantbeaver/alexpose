# Run the paired restoration study on HAIC

The new [notebooks](../../notebooks/synthetic_training_v2/README.md) and [protocol](../../docs/studies/synthetic-training-v2/protocol.md) distinguish software fixtures from source results and protected confirmation. No GPU jobs have been submitted for this implementation. The default GPU authorization is **zero**. The 48 H100-hour proposal is not permission to spend it.

Use the established checkout `/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6` and interpreter `/hai/scratch/tedmui/envs/synthetic-training-cu124/bin/python`. These are configured paths, not a claim that remote access was tested. `common.sh` verifies both paths and requires Torch **2.6.0+cu124**. The allocated preflight executes CUDA arithmetic and Torchvision/MMCV NMS, requiring Torchvision **0.21.0+cu124** and MMCV **2.1.0**. Preserve the [existing environment manifest](../synthetic-training/pyproject.toml) and working MMCV build. Do not run root `uv sync`, change Torch, or set a guessed `CUDA_HOME`. This study adds no GPU package dependency.

Start with local dry-run inspection, which submits nothing and does not open data:

```bash
PYTHONPATH=src /private/tmp/gavd6-stv2-cpu/bin/python scripts/research_directions/synthetic_training_v2/submit.py \
  --config slurm/synthetic-training-v2/source.example.json --dry-run
bash -n slurm/synthetic-training-v2/common.sh
bash -n slurm/synthetic-training-v2/stage.sbatch
bash -n slurm/synthetic-training-v2/preflight.sbatch
bash -n slurm/synthetic-training-v2/prepare.sbatch
```

Before a real batch submission, copy the templates into a unique run configuration directory, replace every `REPLACE`/`DEFINE` value, and record the **explicitly authorized** compute scope. Set `authorized_gpu_hours`, measured prior hours, projected hours for one stage, and a real `cost_ledger` path. The ledger includes all previous GPU attempts, including preflight, body models, EGL, extraction and failed retries; its measured sum must match configuration. Use separate run IDs/configurations for preparation, the one-seed screen and later expansions. Import previous attempt costs into the next ledger. The immutable run identity rejects changes to configuration, source bundle, protocol, code, prior ledger or decision specification.

The source preparation audit CSV must contain `relative_path,start_s,locomotion_status,audit_reviewer,audit_evidence,audit_date,exposure,canonical_person_id`. Locomotion status must be `audited_locomotion`; an AMASS availability row is not sufficient. The authoritative reservation CSV contains `person_id,canonical_person_id,original_split,reserved,exposure`, covering every audited person. Preserve all existing protected identities and aliases. Audit 24 train/8 development people only if enough eligible assets exist; the program reports achieved counts and rejects unsupported windows. Supply local MMPose `StudentSpec` dictionaries (`student_id,family,config,checkpoint`, optional `head_checkpoint`) in the preparation JSON. Name the excluded extractor family before fitting. Existing ViTPose names do not establish lack of exposure.

After verifying the above on HAIC, these are the concrete batch commands. The first command prepares environment variables only. Submission remains an explicit user action:

```bash
cd /hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6
source slurm/synthetic-training-v2/study.env
export STV2_CONFIG="$STV2_ROOT/outputs/synthetic-training-v2/config-source-smoke.json"
export STV2_PREFLIGHT_OUTPUT="$STV2_ROOT/outputs/synthetic-training-v2/preflight-UNIQUE.json"
PYTHONPATH="$STV2_ROOT/src" "$STV2_PYTHON" -c 'from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig; import os; RunConfig.load(os.environ["STV2_CONFIG"]).require_gpu_scope()'
sbatch slurm/synthetic-training-v2/preflight.sbatch
```

Use account `mind`, partition `hai`, and batch jobs; the existing partition rejected interactive allocation. A preflight job requests one H100 for 10 minutes. Record its actual cost before updating a **new** preparation scope. Then:

```bash
export STV2_PREPARATION_CONFIG="$STV2_ROOT/outputs/synthetic-training-v2/config-paired-data.json"
export STV2_PREPARATION_OUTPUT="$STV2_ROOT/outputs/synthetic-training-v2/paired-UNIQUE"
sbatch slurm/synthetic-training-v2/prepare.sbatch
```

Preparation holds body motion, shape, times, appearance, camera and seed fixed; it creates clean, blur and obstruction tracks, and a development-only blur+obstruction combination. It saves overlays, achieved pixel heights/framing/clipping, target/input arrays, actual scores, pair hashes, and attempt cost. Review the contact sheet and approximate landmark convention before source fitting. Without independent annotations, the target remains a projected joint-center proxy. Camera-distance/view experiments and real detector-box evidence are pending.

Once the prepared bundle and its provenance have passed inspection, set a new source scope pointing to it and include all preparation costs. Preview and then submit the dependency chain:

```bash
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/submit.py --config "$STV2_CONFIG" --dry-run
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/submit.py --config "$STV2_CONFIG"
```

The launcher builds audit → data → adaptation status → information controls → direct baselines → JEPA → evaluation → optional status → snapshot → report. Only direct/JEPA request GPUs. Each reserves one H100 for at most one hour, sequentially: **two GPU jobs, maximum concurrency one, two reserved H100-hours**, in addition to prior preparation. There is no automatic sweep/job-array expansion. The budget guard accounts for actual stage attempts and stops at projected/remaining time; Slurm is the allocation limit. CPU preparation, storage, annotations and queue time are separate. The stage guard includes load/prediction/checkpoint overhead, while per-fit logs isolate training costs.

Resume only with unchanged content/configuration:

```bash
"$STV2_PYTHON" scripts/research_directions/synthetic_training_v2/run.py --config "$STV2_CONFIG" --stage direct --resume
```

Completed receipts are reused after checking their entire dependency chain. A partial model resumes from its latest checkpoint only with `--resume`; the trainer validates data, masks, preprocessing, code, optimizer, RNG and runtime identity. Changing the recipe requires a new run ID. Failed preparation uses a new attempt ID, and its cost still counts.

The initial comparison matches data/steps and reports unequal costs. For the distinct practical compute comparison, use a new configuration with `resource_contrast: "equal_total_compute"` and `total_compute_seconds` set from the complete paired-JEPA pretraining-plus-readout `training.json` for the corresponding seed/hardware. The direct/SmoothNet/static fits use a synchronized elapsed-time limit and a maximum update count. Report actual seconds, steps, ceiling termination and one-step overhead; if the update ceiling prevents spending the budget, no exact compute-matching claim is supported. Preserve both configurations and source identities. An official GPU throughput projection must come from allocated measurements, never CPU fixture timing.

Gate B stays `insufficient_evidence` for fixtures and uncalibrated/single-seed screens. A pre-fit `decision_spec` can enable source adjudication after independent calibration and three-seed evidence; see the exact rules in the protocol. Gate A is a separate pending augmented-COCO/adaptation experiment. No failure in Gate A cancels temporal restoration. A failed JEPA gate does not erase a useful direct-denoising result.

Real transfer requires supplied independent, blinded temporal annotations. `annotations.annotation_template` exports blank development landmarks with timing/effort fields; it never exports candidate output as truth. Notebook 07 makes a **development snapshot only**. Confirmation freezing and explicit access are separate APIs; automatic protected confirmation scoring is intentionally unavailable until independent references, exposure, margins and scorer review exist. No source or notebook command opens protected labels.
