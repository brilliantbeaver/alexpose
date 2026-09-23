# Run Gait Fidelity on HAIC

Use this guide to prepare data, submit training through Slurm and inspect the results. The notebook guide explains how to use the same saved run interactively. GPU work runs in allocated workers; setup and ordinary preflight run on CPU.

## Start here: set or unset these variables on HAIC

Run this section in a **HAIC shell**, not on your Mac. Run one block at a time and resolve any error before continuing. Mac commands are labelled separately.

Your existing variables are suitable. **Keep all five**; there is no need to unset `VJEPA2_ROOT` even though this study does not use it:

```bash
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export VJEPA2_ROOT="/hai/scratch/$USER/vendor/vjepa2"
export AMASS_ROOT="$GAVD6_ROOT/data/amass"
export GAVD_FULL_ROOT="$GAVD6_ROOT/data/gavd_full"
```

| Variable | How Gait Fidelity uses it |
| --- | --- |
| `SJEPA_ROOT` | Convenient parent path for defining `GAVD6_ROOT`; not read directly by the runner. |
| `GAVD6_ROOT` | Asset checkout containing manifests and data. Keep this pointing at `gavd6`, even when code runs from a separate release. |
| `VJEPA2_ROOT` | Unused by this study; it can remain set for your other work. |
| `AMASS_ROOT` | In the environment-based asset fallback, the runner appends `/extracted`. With your value it looks under `$GAVD6_ROOT/data/amass/extracted`. A saved preparation configuration takes precedence. |
| `GAVD_FULL_ROOT` | Used when explicitly creating a GAVD plan. It does not automatically add real videos to an AMASS training run. |

**Choose one of the next two paths.** Starting a new run and resuming an existing run use different sources of settings.

### A. Starting a new HAIC run

Clear settings that may point to a previous study or a local CPU fixture, then select the established CUDA interpreter explicitly:

```bash
unset GF_ROOT GF_WORK GF_PYTHON STV2_PYTHON
unset PYTHONHOME PYTHONPATH
export GF_PYTHON="/hai/scratch/$USER/envs/synthetic-training-cu124/bin/python"
export ST_ACCOUNT="mind"
export ST_PARTITION="hai"
if [ -x "$GF_PYTHON" ]; then
  "$GF_PYTHON" --version
else
  printf 'STOP: Python executable missing: %s\n' "$GF_PYTHON"
fi
```

If you see `STOP` or Python cannot start, locate the working CUDA interpreter before continuing; set `GF_PYTHON` to that executable. `mind` and `hai` are the previous study's account and partition—change them before setup if your allocation requires others. Do not install over the working environment or run `uv sync` here.

Leave `GF_ROOT` and `GF_WORK` **unset until setup writes the new session**. `GF_ROOT` will identify the code release; `GF_WORK` will identify the saved run. Setup may source the older synthetic-training-v2 session to reuse assets. Explicit `GF_PYTHON` avoids its `STV2_PYTHON` fallback; inspect the generated configuration before launch because that older session can also supply asset and resource settings.

Do not unset all `ST_*` variables indiscriminately: some may identify your licensed assets. Optional asset overrides and their precedence are explained below. Do not clear scheduler-provided `SLURM_*` or GPU visibility variables, or add a fake `SLURM_JOB_ID`.

### B. Resuming an initialized HAIC run

Skip the new-run block above. Choose the existing directory, then load its saved settings. Replace `study-01` if your run has a different name:

```bash
gf_resume_work="$GAVD6_ROOT/outputs/gait-fidelity/study-01"
unset GF_ROOT GF_WORK GF_PYTHON
unset PYTHONHOME PYTHONPATH
if [ -f "$gf_resume_work/session.env" ]; then
  source "$gf_resume_work/session.env"
  printf 'Code: %s\nRun: %s\nPython: %s\n' "$GF_ROOT" "$GF_WORK" "$GF_PYTHON"
else
  printf 'No saved session at %s/session.env; check the run name.\n' "$gf_resume_work"
fi
```

Continue only if the session was found and the three printed paths are correct. Do not run `setup` again to resume, redirect `GF_ROOT` to a newer checkout, or edit the saved configuration. The run remains tied to its original code and inputs. Source the session again in each new shell; notebook kernels must inherit it when they start.

## 1. New run only: copy a code release from your Mac

In your **Mac terminal**, from the local GAVD6 checkout:

```bash
bash slurm/gait-fidelity/sync-to-haic.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6 --apply
```

Replace both occurrences of `tedmui` if your HAIC username differs. This uses the explicit hostname and login rather than assuming a `haic` SSH alias exists. Omitting `--apply` performs a local packaging preview only.

A successful transfer prints a setup command containing a unique path under `releases/gait-fidelity/`. Save that **exact release path** for the next step. Data, checkpoints and Python environments stay in their existing HAIC locations. If transfer fails, repeat it to create a fresh release and use the path printed by the successful attempt. Keep each release unchanged while its run is active.

## 2. New run only: choose the experiment and initialize it

Back in the **HAIC shell** prepared in path A, set `gf_release` to the release path printed by the transfer. Replace the placeholder below; do not type `REPLACE_WITH_RELEASE_ID` literally. Choose a new run name; this example uses `study-02` to avoid an existing `study-01`.

```bash
gf_release="$GAVD6_ROOT/releases/gait-fidelity/REPLACE_WITH_RELEASE_ID"
gf_new_work="$GAVD6_ROOT/outputs/gait-fidelity/study-02"
if [ -f "$gf_release/slurm/gait-fidelity/run.sh" ]; then
  printf 'Release found: %s\n' "$gf_release"
else
  printf 'STOP: no launcher under %s. Check the successful transfer output.\n' "$gf_release"
fi
```

Continue only after `Release found` appears; otherwise correct `gf_release`. Decide the experiment set now:

| Setup choice | Final models / optimization phases | What it includes |
| --- | --- | --- |
| `--experiment-set core` | 30 / 39 | Graph-time coordinate and JEPA models, direct models, initialized and shuffled-reference controls, each with base/paired-change supervision and three seeds. This is the source default. |
| `--experiment-set full` | 102 / 135 | All 34 recipes, including the additional masking, topology, practical and label/pairing controls. |

**The walkthrough below selects `core`**, preserving the main direct-training and representation controls while allowing more of the compute budget to cover independent people and recordings. Choose `full` before setup if the extra attribution experiments fit the available time. Omitted tutorial groups show their teaching calculations without implying fitted results. The saved plan determines the run.

```bash
if [ -e "$gf_new_work" ]; then
  printf 'Already exists: %s. Resume it or choose a new run name.\n' "$gf_new_work"
else
  bash "$gf_release/slurm/gait-fidelity/run.sh" setup "$gf_new_work" \
    --root "$GAVD6_ROOT" \
    --source-selection full_manifest \
    --cohort-preset named_walking \
    --experiment-set core \
    --num-shards 8
fi
```

Continue only after successful initialization. The current source default selects a cohort from the full AMASS manifests; it is not restricted to the historical 24/8-person roster. `named_walking` selects eligible named walking candidates, while `treadmill_walking`, `all_eligible` and `reviewed` are separate choices; the reviewed preset needs a suitable `--motion-review-csv` file. Inspect the frozen cohort's counts, exclusions and reservations before consuming GPU time. Eight preparation shards do not change the eight-GPU concurrency cap.

Setup first tries the retained synthetic-training-v2 preparation configuration. If none is found, full-manifest setup can derive asset paths from the environment. To select a particular configuration, append `--source-config /absolute/path/to/preparation.json`. The full manifests and required asset files must already exist. To reproduce a historical roster instead, explicitly choose `--source-selection legacy_roster --source-bundle /absolute/path/to/bundle`; `--source-bundle` is not accepted with `full_manifest`.

Load the newly written session:

```bash
source "$gf_new_work/session.env"
printf 'Code: %s\nRun: %s\nPython: %s\n' "$GF_ROOT" "$GF_WORK" "$GF_PYTHON"
```

## 3. Check the effective settings before any GPU submission

**Both new and resumed runs continue here**, after sourcing the correct session. This displays the values the program will actually use:

```bash
"$GF_PYTHON" - <<'PY'
import json, os
from pathlib import Path
work = Path(os.environ['GF_WORK'])
cfg = json.loads((work / 'config.json').read_text())
assert not cfg['fixture'], 'This is a CPU fixture; select a source run.'
for key in ('code_root', 'work', 'python', 'asset_root', 'inherited_configuration', 'experiment_set'):
    print(f'{key}: {cfg.get(key)}')
print('source selection:', cfg['data'].get('source_selection', 'legacy_roster'))
print('resources:', cfg['resources'])
print('assets:', json.dumps(cfg['preparation'], indent=2))
print('plan counts:', json.loads((work / 'plan.json').read_text())['counts'])
PY
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" cohort "$GF_WORK"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" preflight "$GF_WORK"
```

Expect `fixture` to be false and preflight to finish with `CPU_PREFLIGHT_PASSED`. The source environment must contain Torch `2.6.0+cu124`, Torchvision `0.21.0+cu124`, compatible MMCV/MMPose/MMEngine, rendering/body-model dependencies and FFmpeg. Preflight reports missing assets or import/version failures. Resolve those before launching; passing this CPU check does not yet certify EGL rendering or CUDA operators on a worker.

The saved account/partition are used by both coordinator and workers. `launch --account ... --partition ...` overrides the coordinator only, so it is not a way to change the whole run's allocation settings. If initialization captured unwanted settings, use a new correctly initialized run rather than editing a frozen one.

## 4. Prepare the shared data, then wait and review it

For a new run:

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK" --prepare-only
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" status "$GF_WORK"
```

The first command submits a CPU coordinator and returns a job ID; it does **not** wait for preparation. The coordinator starts GPU workers, which test CUDA/MMCV before rendering and pose extraction. Repeat `status` to inspect progress and check the coordinator log named in `control/coordinator.json`.

Wait for the coordinator log to report `PREPARATION_COMPLETE`, for `status` to contain the preparation result, and for that coordinator job to finish. Open the viewer reported by status and inspect anatomical sides, camera projection, timestamps, visibility and the movement interventions. Notebook 01 can help with this inspection. A successful automated screen does not replace review of the references.

A resumed run with completed preparation can reuse it. Do not launch the full stage while the preparation coordinator is still active: the launcher returns the existing job and does not queue the new stage. Wait for it to finish, then issue the next command.

## 5. Launch the selected comparison

After preparation and reference review:

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK"
```

The coordinator profiles representative work on an allocated H100, selects the common full or half update schedule from measured cost, and stops if neither fits. It then runs the saved experiment set across at most eight one-H100 workers, sharing identical pretraining where allowed. Profiling and failed attempts count toward the budget.

To monitor or resume, load the saved session using path B, inspect status/logs, then repeat `launch` after resolving a failure. A running coordinator is reused. If submission is reported as ambiguous, reconcile the reported job name with Slurm rather than deleting its receipt and submitting a duplicate. If an attempt limit is exhausted, repeated launches will not repair the underlying failure.

Keep `config.json`, the release and prepared inputs unchanged. Changes to the protocol or code require a new release/run. To stop a run, use `scancel` on the coordinator **and its active worker job IDs** from status; stopping the coordinator alone does not stop already submitted workers.

## 6. Read results after evaluation finishes

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" status "$GF_WORK"
```

Wait until `completed_phases` equals `total_phases`, no workers are active, and `report_available` is true. Also check the coordinator log for successful evaluation; completion of training alone does not establish that evaluation succeeded. Then:

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" report "$GF_WORK"
srun --account=mind --partition=hai --cpus-per-task=4 --mem=64G --time=04:00:00 \
  bash "$GF_ROOT/slurm/gait-fidelity/run.sh" verify "$GF_WORK"
```

Verification checks retained artifacts and recomputes metrics in bounded batches on CPU; allow it to finish. Use your saved account and partition if they differ from `mind` and `hai`.

| Artifact under `GF_WORK` | Purpose |
| --- | --- |
| `config.json`, `session.env`, `plan.json` | Saved paths, environment, protocol and phase dependencies. |
| `cohort/` for full-manifest runs | Frozen source selection, exclusions and reserved test inventory. |
| `data/viewer.html` and preparation result | Source/reference review and admission checks. |
| `control/coordinator.json`, `logs/`, `attempts/` | Job identities, logs, failures and allocation accounting. |
| `evaluation/per-person.csv`, `evaluation/coverage.csv` | Movement/coordinate scores and retained failures. |
| `evaluation/responses.csv`, `evaluation/comparisons.json` | Response preservation and declared comparisons. |
| `report.md` | Summary with the supported evidence scope. |

For notebook use, continue with the [environment and walkthrough guide](../../notebooks/gait_fidelity/ENVIRONMENT.md). Notebook 04 submits jobs asynchronously; wait for the source run before running evaluation notebooks 05–06.

## 7. Download results to your Mac

In your **Mac terminal**, use the actual HAIC run name in both paths:

```bash
bash slurm/gait-fidelity/retrieve.sh \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/gait-fidelity/study-02 \
  outputs/gait-fidelity/downloads/study-02
```

This copies reports, predictions, metadata and viewer assets, excluding `.pt`/`.pth` model checkpoints. Repeat to retrieve later completed outputs from the same run. To include checkpoints:

```bash
rsync -a --partial --progress \
  tedmui@haic.stanford.edu:/hai/scratch/tedmui/alexpose/experiments/sjepa/gavd6/outputs/gait-fidelity/study-02/ \
  outputs/gait-fidelity/downloads/study-02/
```

Read downloaded Markdown/HTML and CSVs directly. **Do not source the downloaded HAIC session on your Mac** or run the canonical source notebooks against a moved run: saved paths, assets and interpreter identify its original HAIC location. Copying checkpoints alone does not make replay portable.

## Asset overrides and optional GAVD work

Your existing `AMASS_ROOT` and `GAVD_FULL_ROOT` values can remain set throughout. A discovered or explicitly supplied preparation configuration takes precedence over the environment fallback. Changing a variable afterward does not update a saved run. The [complete variable reference](../../notebooks/gait_fidelity/ENVIRONMENT.md#optional-asset-overrides-and-precedence) explains the `ST_*` fallback paths.

GAVD is a separate, optional stage. Merely exporting `GAVD_FULL_ROOT` does not start it. With the correct source session loaded, `gavd-plan` reads `$GAVD_FULL_ROOT/youtube/all`, `$GAVD_FULL_ROOT/annotations/GAVD/data` and the manifests under `$GAVD6_ROOT/manifests/gavd`. Explicit CLI paths can override those defaults. Planning hashes the referenced files and writes an immutable plan; it can take time on a large collection.

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" gavd-plan "$GF_WORK"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK" --gavd-only
```

Use these only when adding that stage deliberately, after reviewing its data reservations and while no other coordinator stage is active. Reuse an existing GAVD plan rather than recreating it. Extraction shares the study budget and GPU cap. These commands do not complete GAVD evaluation or unlock confirmation; those require separate checkpoint/reference and exposure decisions. GAVD labels are not dense joint-position references.

## Local CPU check and resource limits

For a local software check, follow the [Mac CPU instructions](../../notebooks/gait_fidelity/ENVIRONMENT.md#run-the-cpu-tutorials-on-your-mac). They clear HAIC run selectors and use local Python with generated data.

The historical [compute plan](../../docs/studies/gait-fidelity/methods/execution.md) allocated 360 H100-hours plus 120 hours of reserve. The default saved runner limit is **360 GPU-hours**; the historical reserve is not automatically added. Inspect the current cohort, plan and measured profile rather than treating those allowances as runtime estimates.

The current coordinator defaults to two CPUs, **64 GB RAM**, a 72-hour limit and the saved account/partition; each worker requests one H100. If the partition rejects the coordinator time, use `launch --controller-hours HOURS` with an allowed value. A fresh HAIC run is still needed to validate actual source throughput and memory requirements; local notebook checks do not establish them.

## 8. Extract GAVD and evaluate real-video transfer

AMASS supplies paired projected reference coordinates. GAVD adds real-video evaluation using its gait-pattern annotations, with recordings kept together across splits. These annotations do not provide reference joint coordinates or affected-side ground truth. Run this stage after the current AMASS coordinator has finished; the same queue and budget cover both datasets.

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" gavd-plan "$GF_WORK"
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK" --gavd-only
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" status "$GF_WORK"
```

The plan inventories all 1,874 manifest sequences, checks available videos and saves training/development/confirmation groups. Related clips from a video stay together. Supply `--identity-csv PATH` when a reviewed cross-video person mapping is available; without it, independence is established at recording-group level. `--reservation-csv PATH` protects previously exposed groups. Source file hashes also prevent identical video copies entering different splits.

Extraction needs the original per-frame annotation CSVs to identify the annotated person's bounding box. The sequence summary cannot establish the correct person in a crowded frame. If asset discovery fails, use `gavd-plan --video-root PATH --annotation-root PATH`; its `--help` lists the optional manifest path, shard count and frame-origin convention. The extractor stores decoder-reported presentation timestamps and checks their agreement with the nominal constant-frame-rate grid, rejecting repeated/nonmonotonic clocks or excessive disagreement. This is a decoded-video clock, not an independently verified camera capture clock. Missing boxes, unavailable videos, short sequences, unsupported sampling rates and failed extractions remain recorded exclusions.

`--gavd-only` extracts training and development groups. It shares the AMASS coordinator, eight-H100 concurrency cap and allocation ledger. A later stage request while another coordinator is active does not queue that stage: wait for completion, then repeat the intended command. More shards can divide the work into smaller retryable jobs without changing the selected population.

After extraction and AMASS fitting have completed, run the real-video evaluation in a CPU allocation:

```bash
srun --account=mind --partition=hai --cpus-per-task=8 --mem=64G --time=04:00:00 \
  bash "$GF_ROOT/slurm/gait-fidelity/run.sh" gavd-evaluate "$GF_WORK" --device cpu
```

The default selects the saved primary candidate and direct-training comparator across all declared seeds. It applies those fixed restorers and trains a small linear gait-pattern classifier on GAVD training groups only. Unchanged tracks, temporal filtering and a camera/height-only classifier provide controls. Read `gavd/evaluation/*/scores.json`, `comparisons.json`, `by-view.csv` and `extraction-coverage.csv` together. A classifier gain concerns accessible gait-label information; it does not certify anatomical accuracy or clinical validity. Notebook 05 derives the group-weighted classifier fit and explains the measurement boundary.

## 9. Run a frozen confirmation evaluation

Keep original AMASS test people and GAVD confirmation groups locked while selecting the method and interpreting development results. A declaration must record real previous-exposure evidence; a split name or successful script cannot establish that somebody was never inspected before.

For AMASS, after the development report is complete:

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" lock-confirmation "$GF_WORK" \
  --reviewed-by "$USER" --evidence "Reviewed previous experiment exposure records" \
  --exposure-ledger /absolute/path/to/reviewed-amass-exposure.csv
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK" --confirmation-only
```

Replace the path and evidence description with the actual reviewed records. The exposure ledger follows the cohort person-reservation format, and every predeclared confirmation identity must be explicitly unreserved and `unexposed_verified`. If those facts cannot be established, retain the development scope. Preparation checks the frozen person list before opening their motion arrays.

Wait for `CONFIRMATION_PREPARATION_COMPLETE`, then evaluate the frozen checkpoints and training-only calibration:

```bash
srun --account=mind --partition=hai --cpus-per-task=8 --mem=64G --time=04:00:00 \
  bash "$GF_ROOT/slurm/gait-fidelity/run.sh" evaluate-confirmation "$GF_WORK"
```

The default paths are `confirmation/lock.json`, `confirmation/bundle` and `confirmation/evaluation`. This stage does not fit a model or calibration to confirmation references.

For GAVD, use its separate reviewed video-exposure ledger:

```bash
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" gavd-lock "$GF_WORK" \
  --exposure-csv /absolute/path/to/reviewed-gavd-exposure.csv
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" launch "$GF_WORK" --gavd-confirmation
```

That CSV requires `video_id,exposure,reviewed_by,evidence`; every protected video must have verified unused history marked `unexposed_verified`. After extraction completes, repeat the allocated `gavd-evaluate` command with `--split confirmation`. Checkpoint hashes and the saved data plan are verified, and outputs remain separate from development. Unknown cross-video person overlap still limits the strength of real-video independence claims.

## Implementation checks

The [full-data validation record](validation/README.md) lists the manifest census, automated checks, notebook execution and independent review findings. It distinguishes local software validation from the source preparation and H100 profiling checks that still need to run on HAIC.
