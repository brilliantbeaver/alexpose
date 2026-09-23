# Notebook environment and walkthrough

Use this guide to open the notebooks against a saved HAIC source run or run the generated-data tutorials on your Mac. For first-time data preparation and Slurm submission, follow the [HAIC guide](../../slurm/gait-fidelity/README.md).

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

## Open notebooks for the saved HAIC run

If this is a new experiment, complete steps 1–3 of the [HAIC guide](../../slurm/gait-fidelity/README.md) first. Do not point `GF_WORK` at an empty directory: selecting a source run requires its saved configuration and session. For an existing experiment, use path B above.

The session must be loaded **before the Jupyter server/kernel starts**. In that same HAIC shell, prepare the Python import and headless-rendering environment, then check dependencies:

```bash
export PYTHONPATH="$GF_ROOT/src"
export PYTHONNOUSERSITE=1
export PYOPENGL_PLATFORM=egl
bash "$GF_ROOT/slurm/gait-fidelity/run.sh" preflight "$GF_WORK"
"$GF_PYTHON" -c 'import ipykernel, nbformat, nbclient, numpy, pandas, scipy, matplotlib, torch; print("Notebook dependencies available; Torch", torch.__version__)'
```

Expect `CPU_PREFLIGHT_PASSED` and the dependency message. Source preflight requires **Torch `2.6.0+cu124` and Torchvision `0.21.0+cu124`**, compatible MMCV/MMPose/MMEngine, pyrender, trimesh, OpenCV, human_body_prior and FFmpeg. It also checks saved assets and cohort metadata. This CPU check does not exercise the worker's CUDA kernels or rendering.

Select a kernel using the saved `GF_PYTHON`. If that interpreter is not listed in your notebook interface, register it on HAIC:

```bash
"$GF_PYTHON" -m ipykernel install --user \
  --name gait-fidelity-cu124 \
  --display-name "Gait Fidelity (HAIC CUDA 12.4)"
```

Use your normal HAIC notebook-access method, with the server/kernel launched from the prepared environment, and select **Gait Fidelity (HAIC CUDA 12.4)**. If using JupyterLab from this shell, and it is installed in that environment:

```bash
"$GF_PYTHON" -m jupyterlab --no-browser "$GF_ROOT/notebooks/gait_fidelity"
```

Follow your usual HAIC connection/tunnelling procedure for access. If using an already running managed notebook server or IDE, configure its kernel environment explicitly with the saved `GF_ROOT`, `GF_WORK`, `GF_PYTHON` and the three exports above, then restart the kernel. Sourcing a file in an unrelated terminal does not update an existing server's environment. Kernel registration chooses Python; it does not bake your run's variables into the kernel specification.

Before executing the study cells, check the kernel with this short cell:

```python
import json, os, sys
from pathlib import Path
print("Kernel interpreter:", sys.executable)
for name in ("GF_ROOT", "GF_WORK", "GF_PYTHON"):
    assert os.environ.get(name), f"Missing {name}; load the HAIC session before starting this kernel."
    print(name, os.environ[name])
config = json.loads((Path(os.environ["GF_WORK"]) / "config.json").read_text())
assert not config["fixture"], "This is generated test data, not the HAIC source study."
print("Source selection:", config["data"].get("source_selection", "legacy_roster"))
print("Experiment set:", config.get("experiment_set", "full"))
```

Confirm the kernel interpreter is the intended CUDA environment and the code/run paths are the saved ones. `GF_PYTHON` controls notebook subprocesses; setting it does not switch a kernel that is already using another interpreter. The notebooks do not automatically load a `.env` file. Without `GF_WORK`, their default is a local CPU fixture, so check the mode rather than assuming an opened notebook uses real source data.

## Work through the notebooks in this order

Open the files under **`$GF_ROOT/notebooks/gait_fidelity`**, so notebook code and the saved release agree.

| Stage | What to do | When to continue |
| --- | --- | --- |
| **00 · Start here** | Inspect saved settings, run CPU preflight and read the experiment plan. | The displayed source run, paths and plan match your intent. |
| **01 · Data and references** | Work through the examples. The source preparation cell prints a command for your HAIC shell; it does not submit it. | Submit `launch --prepare-only`, wait for `PREPARATION_COMPLETE` and coordinator exit, then run the remaining validation/inspection cells and review references. If already prepared, reuse the saved bundle. |
| **02–03 · Masking and architecture** | Inspect the prepared inputs, masks, model shapes and selected experiment matrix. | Shared preparation has completed. |
| **04 · Losses and execution** | Run the teaching examples, then launch or resume the central Slurm run. | Submission returns immediately. Monitor status/logs and wait for all required phases and successful evaluation. |
| **05 · Evaluation** | Recompute/read the saved scores and figures. | All final predictions exist. Evaluation is CPU work and may take time; use a permitted CPU allocation where required. |
| **06 · Verification** | Verify artifact hashes and reconstruct metrics. | Verification succeeds; then interpret the results at their supported evidence level. |
| **A–E · Experiment tutorials** | Inspect the comparisons, checkpoints and results from the same run. | They do not submit duplicate experiments. Groups outside a core plan have no trained results. |

Use `bash "$GF_ROOT/slurm/gait-fidelity/run.sh" status "$GF_WORK"` in the HAIC shell to monitor progress. The source workflow cannot proceed from start to finish with unattended “Run All” while Slurm prerequisites are pending. If a preparation coordinator is still active, a full launch returns that existing job and does not queue the next stage; wait for exit and launch again.

A new source run defaults to **full-manifest AMASS selection, `named_walking`, eight preparation shards and the `core` experiment set**. Core has 30 final models and 39 optimization phases; full has 102 and 135. The HAIC walkthrough explicitly selects full to cover all five experiment groups. Read your saved plan rather than assuming historical 24/8-person or 102-model counts apply. Confirmation remains separately reserved.

## Run the CPU tutorials on your Mac

Use a **Mac terminal** in the local GAVD6 checkout, not the HAIC shell. Your HAIC asset variables need not be deleted, but clear run/interpreter and Python import settings before starting local Python:

```bash
cd /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd6
unset GF_ROOT GF_WORK GF_PYTHON STV2_PYTHON
unset PYTHONHOME PYTHONPATH
```

For a fresh, automated execution of all twelve notebooks:

```bash
gf_cpu_check="$(mktemp -d /tmp/gait-fidelity-cpu.XXXXXX)"
.venv/bin/python notebooks/gait_fidelity/execute_tutorials.py \
  --work "$gf_cpu_check/work" \
  --output "$gf_cpu_check/executed"
```

The checker sets its own `GF_*` variables, uses generated data, runs CPU training synchronously and saves executed copies plus `execution.json`. It cannot submit a source study. Jupyter needs permission to open local kernel sockets; a sandbox can block startup independently of notebook correctness.

For interactive local use, start the local Jupyter server after the unsets and select this checkout's `.venv` kernel. Notebook 00 creates `outputs/gait-fidelity/tutorial-fixture` by default. If that directory contains a run frozen against older code, use the fresh checker command above rather than editing its receipts. Do not source a downloaded HAIC `session.env` on the Mac.

## Variable reference: run selection and automatic settings

| Variable | Who supplies it | Meaning |
| --- | --- | --- |
| `GF_ROOT` | HAIC setup writes it to `session.env`; local checker supplies it | Code release/checkout. Local interactive setup otherwise searches upward from the working directory. |
| `GF_WORK` | Saved session or local checker | Initialized run directory. Without it, interactive notebooks default to the generated-data fixture. |
| `GF_PYTHON` | Set explicitly for new HAIC setup; then restored from the session | Python for subprocesses and workers. Local helpers otherwise use the kernel's `sys.executable`. |
| `STV2_PYTHON` | Optional previous-study session | Setup fallback only when `GF_PYTHON` is absent. Not needed to resume a Gait Fidelity session. |
| `ST_ACCOUNT`, `ST_PARTITION` | New-run setup environment, default `mind` / `hai` | Captured in `config.json`; both coordinator and workers default to these saved settings. Later exports do not rewrite them. Explicit launch account/partition flags affect the coordinator only. |
| `PYTHONPATH`, `PYTHONNOUSERSITE` | Notebook subprocess helper and HAIC launcher; exported above for the notebook kernel | `$GF_ROOT/src` and `1`. |
| `PYOPENGL_PLATFORM` | HAIC launcher; exported above for notebook imports | `egl` before Python starts. Not needed for the CPU fixture. |
| `CUBLAS_WORKSPACE_CONFIG` | HAIC launcher | `:4096:8` for deterministic CUDA operations. |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | HAIC launcher / local checker | Each is `4` in HAIC launcher subprocesses/jobs and `1` in the automated local check. An interactive kernel does not inherit these limits merely from `session.env`. |
| `MPLCONFIGDIR`, `XDG_CACHE_HOME` | Notebook helper if unset; local checker explicitly | Writable cache directories under the run or execution output. Existing values must be writable in that environment. |
| `MPLBACKEND` | Local automated checker | `Agg` for headless plots. |
| `SLURM_JOB_ID` | Slurm | Required for source coordinator/worker execution. Never set it manually to simulate an allocation. |
| `TZ` | Scheduler recovery query | `UTC`; no manual setup. |
| `USER`, `PATH` | Normal login shell | Username for default paths; Slurm commands and `ffmpeg` must be discoverable on HAIC. The local checker prepends its Python directory to `PATH`. |
| `TMPDIR` | Optional shell setting | Temporary release-packaging directory; defaults to `/tmp`. Not a study-selection variable. |

Leave cluster GPU visibility settings to Slurm. `CUDA_HOME` is relevant when building the earlier toolchain; it is not an additional notebook variable to invent when reusing the installed CUDA environment. Do not blanket-unset `PATH`, library paths or cluster settings.

## Optional asset overrides and precedence

For a **new full-manifest run**, setup tries an explicit `--source-config` first, then known synthetic-training-v2 preparation configurations. If no configuration is found and none was explicitly required, it derives paths from the environment below. An existing saved run continues using its saved configuration. A malformed explicit configuration is an error, not permission to fall back silently.

| Fallback variable | Default with your HAIC layout |
| --- | --- |
| `ST_AMASS_ROOT` | `$AMASS_ROOT/extracted`; set this override only if the extracted motion root is elsewhere. It is used as-is, with no `/extracted` appended. |
| `ST_MODEL_ROOT` | `/hai/scratch/$USER/models`; supplies MMPose configuration/checkpoint and rendering defaults. |
| `ST_BODY_MODEL_ROOT` | `/hai/scratch/$USER/body_models` |
| `ST_DMPL_ROOT` | `dmpls` under the effective body-model root |
| `ST_UV_PATH` | `synthetic-rendering/smplitex/smpl_uv.obj` under the effective model root |
| `ST_TEXTURE_DIR` | `synthetic-rendering/smplitex/textures` under the effective model root |
| `ST_BACKGROUND_DIR` | `synthetic-rendering/coco-backgrounds` under the effective model root |

The AMASS manifests default to `$GAVD6_ROOT/manifests/amass`. If inherited values are unexpected, inspect `config.json` and the printed `inherited_configuration`; changing `AMASS_ROOT` will not override a saved preparation file. Choose the intended `--source-config` and a fresh run instead. Custom estimator configs/checkpoints should be supplied through that preparation configuration.

For optional GAVD planning, `GAVD_FULL_ROOT` supplies `youtube/all` and `annotations/GAVD/data`; GAVD manifests default to `$GAVD6_ROOT/manifests/gavd`. `gavd-plan --video-root`, `--annotation-root` and `--manifest-dir` override these paths. The plan is then saved and immutable. These settings do not make an arbitrary MP4 folder a synthetic training bundle, and no separate `COCO_ROOT` or `GAVD_ROOT` variable is required by this workflow.

## Troubleshooting before submission

| Symptom | Specific next step |
| --- | --- |
| Missing `config.json` after selecting `GF_WORK` | Check the run name. Initialize a new run in the HAIC guide, or source the session of the existing one. Do not use a raw video directory as `GF_WORK`. |
| Mode prints `software fixture` on HAIC | Stop before interpreting outputs. Source the intended source session and restart the kernel with that environment. |
| Wrong Torch version or missing MMCV/MMPose | Check both `GF_PYTHON` and the kernel's `sys.executable`. Use the established CUDA interpreter; avoid `uv sync` or replacing compiled dependencies in place. |
| Missing AMASS/body-model/checkpoint path | Read the effective path in `config.json`; check inherited-configuration precedence before changing environment variables. |
| `sbatch` or `ffmpeg` not found | Make the cluster's installed command available in `PATH`, then repeat CPU preflight. |
| Configuration, plan or code changed after launch | Resume the original unmodified release, or start a new release/run for the changed protocol. |
| A new stage returns the old coordinator ID | The previous stage is active. Wait for it to exit, then submit the requested stage again. |
| Notebook 05 cannot find completed predictions | Training or evaluation is still incomplete; inspect status and logs rather than rerunning preparation. |

## What the earlier local checks establish

These receipts describe earlier notebook revisions. The full-manifest revision
adds AMASS/GAVD population audits, bounded-memory aggregation, weighted
calibration and real-video probe calculations. It requires its own fresh
execution receipt; earlier exact-output comparisons do not certify the changed
donor-sampling and population-weighting protocol.

The [earlier readiness receipt](../../docs/studies/gait-fidelity/records/notebook-readiness-20260922.json) records the notebook-directory rename and validation of the previous 53-code-cell tutorials on 22 September 2026. It also records a local packaging preview and unavailable live HAIC authentication. That historical receipt does not describe the expanded mathematical lessons. CPU tutorial timings are not GPU-throughput estimates.

The [expanded-tutorial receipt](../../docs/studies/gait-fidelity/records/transparent-tutorials-20260922.json) records successful execution of all **12 notebooks and 94 code cells**, with zero error outputs. The 50 study tests and six layout tests passed. The fresh CPU fixture reproduced all 135 model states, 102 final prediction exports and seven evaluation CSV tables exactly against the earlier fixture. Final review corrections were checked with targeted kernel reruns. These checks did not access HAIC or validate an H100 environment.

The expanded lessons add small CPU scratch calculations before the canonical stages. Model construction and translated-view examples use isolated random generators, and scratch fitting never loads a retained checkpoint. The architecture lesson also checks the default source tensor dimensions on CPU. These checks establish code and shape behavior; AMASS rendering, MMPose extraction and H100 execution still require the allocated HAIC worker checks.
