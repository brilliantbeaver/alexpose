# A locked environment for the synthetic-training study

The study now has a dedicated Python dependency manifest and lockfile. The
[setup guide](../../../slurm/synthetic-training/README.md) replaces the sequence
of manual package installations with one environment command followed by a
separate model download. This change concerns installation and execution
consistency; it does not change the scientific protocol or add experimental
results.

## Why the study has a separate project

The repository's root project selects PyTorch 2.6 with CUDA 12.4. The current
pose estimators use PyTorch 2.1, torchvision 0.16 and MMCV 2.1 with CUDA 12.1.
Combining those requirements in the root dependency list would recreate the
version conflicts encountered during setup. The standalone
[pyproject.toml](../../../slurm/synthetic-training/pyproject.toml) and
[uv.lock](../../../slurm/synthetic-training/uv.lock) keep this study within one
compatible declared dependency graph. They are not members of a shared uv
workspace, and the root manifest and lockfile remain unchanged.

The lock resolves 159 entries, including the environment project itself;
158 packages are selected for installation on Linux x86-64 with CPython 3.11.
The tested package manager is uv 0.12.15. Both the manifest and setup script
require `>=0.12.15,<0.13` so an older uv cannot silently ignore newer settings.

| Previous problem | Recorded choice | Verification |
|---|---|---|
| `uv add` selected the root project's environment | Explicit study project and `UV_PROJECT_ENVIRONMENT` derived from `ST_PYTHON` | Shell tests reject root environments, aliases to them, wrong interpreters and inconsistent checkout paths |
| Chumpy's isolated build omitted `pip` | Exact Chumpy 0.71 Git revision and declared extra build dependencies | Actual isolated build succeeded without manually installing `pip` into its target environment |
| Chumpy 0.70 used removed Python/NumPy APIs | Preserve the tested upstream repair, NumPy 1.26.4 and SciPy 1.14.1 | Three numerical compatibility tests pass |
| Human Body Prior 2.3 required a different Torch range | Preserve the tested official 2.2.2.0 revision | Actual build and three body-model numerical tests pass |
| MMEngine still uses `pkg_resources` | Pin runtime Setuptools 80.9.0; constrain isolated build tools | Runtime checker exercises MMEngine package discovery; full Linux check remains pending |
| Installed MMPose code could differ from its downloaded configurations | Pin one Git revision in the manifest and validate the asset checkout against it | Actual built package contains the required `.mim` configs; tests reject different revisions and edited configs |
| Successful setup could be followed by imports from an inherited Python path | Clear `PYTHONHOME`/`PYTHONPATH` and disable user-site packages in setup, launchers and notebook kernels | Shell regressions and an actual temporary notebook execution pass |

The [earlier body-model diagnosis](body-model-dependency-repair.md) and
[Chumpy diagnosis](chumpy-installation-repair.md) retain the original failures
and numerical evidence. Their manual commands document the investigation;
routine installation now follows the setup guide.

## What setup does

[`setup-environment.sh`](../../../slurm/synthetic-training/setup-environment.sh)
requires an explicit absolute `ST_PYTHON` pointing into a dedicated virtual
environment. It rejects unsupported hosts and protected paths before changing
packages, preserves an existing directory that is not a usable virtual
environment, and refuses to replace an environment using the wrong Python
version. A missing environment is created with Python 3.11 through uv.

The command uses `uv sync --locked`: the manifest must agree with the checked-in
lock, and installation must use that dependency set. Synchronization can remove
packages outside the set, so the environment must belong exclusively to this
study. Old manual-install overrides are cleared within the wrapper; cache,
proxy, certificate and offline settings remain available. Git revisions are
fixed, and MMCV comes from an explicit CPython 3.11 Linux wheel URL.

An advisory lock beside the environment prevents two setup/check processes
from operating on it together. The operating system releases that lock when
the process exits, including an interrupted installation. Its empty lock file
remains in place; do not delete it while another process may be using it.
Rerunning setup with the same `ST_PYTHON` resumes package synchronization.
Existing experiment outputs and model assets are outside that operation.

After synchronization, `uv pip check` checks declared dependency consistency.
The [runtime checker](../../../scripts/research_directions/synthetic_training/check_environment.py)
then verifies critical version pins, exact Git provenance, package locations,
small CPU operations, Torchvision/MMCV non-maximum suppression (removing
overlapping duplicate detections), MMPose/MMPreTrain APIs, bundled COCO metadata,
Chumpy computations, the body-model interface, and notebook support imports.
It saves `synthetic-training-environment.json` inside the environment, including
manifest and lock hashes, package paths, individual results and explicit limits.

`--check` checks the lock against the installed environment offline and runs
the probes without synchronizing packages or replacing the saved report.
`--require-cuda` adds real GPU arithmetic and both CUDA suppression operations.
A login-node check permits unavailable CUDA and records that GPU execution was
not tested. Neither mode loads licensed body assets, full pose checkpoints,
V-JEPA weights or an EGL renderer.

## Measured validation and its limits

Evidence is retained under
[`work/artifacts/synthetic-training-locked-env-20260916/`](../../../work/artifacts/synthetic-training-locked-env-20260916/).
The [machine-readable summary](../../../work/artifacts/synthetic-training-locked-env-20260916/validation.json)
records outcomes and file hashes; the [executed commands](../../../work/artifacts/synthetic-training-locked-env-20260916/commands.json)
identify the environments and corresponding logs.
The local machine is macOS arm64. The separate source-build fixture uses
Python 3.11.12; it does not modify the repository environment or the earlier
body-model and Chumpy test environments.

| Check actually executed | Result |
|---|---|
| Dedicated lock creation and offline consistency check | Passed; 159 resolved entries |
| Linux x86-64, CPython 3.11 installation preview | Passed; 158 packages selected, with no package installation |
| Actual isolated source builds | Chumpy, Human Body Prior, MMPose and iopath built successfully; a separate uncached PyOpenGL build also passed |
| Installed source metadata and packaged configs | Exact three Git commit IDs confirmed; MMPose wheel contains the required pilot and COCO `.mim` files |
| Synthetic-training suite | 72 tests: 66 passed, six optional dependency tests skipped in the ordinary project environment |
| Optional dependency tests in their existing pinned environments | Three Chumpy tests and three body-model tests passed separately |
| Shell setup regression coverage | Includes existing/new environments, failure propagation, interrupted retry, real advisory-lock contention, unsupported hosts and wrong checkout/config identities |
| Actual notebook kernel | Passed with the selected interpreter and clean import environment despite conflicting variables in the parent process |
| Source pipeline Slurm preview | Passed; produced the four-student roster and `00 → 01 → 02 array → 03 → 07` commands without submitting jobs |
| Root project preservation and static checks | Root manifest/lock unchanged; shell syntax, Python syntax, Markdown links and whitespace checked |

The source-build fixture deliberately installs only five source packages with
`uv pip install --no-deps` to isolate wheel-building behavior on this Mac.
That command is a build test, not the study installation recipe. Production
setup resolves and installs all runtime dependencies with `uv sync --locked`.
The Linux preview checks resolution and package selection; it cannot run Linux
binary extensions on macOS.

No HAIC access was available. `docker version` also confirmed that the local
Docker engine was not running, so no substitute Linux container test ran.
The production wrapper correctly rejected this Mac before changing an
environment. The full Linux installation, actual MMCV/CUDA checks, system EGL
rendering, real model loading and scientific notebook stages remain to be
verified on the intended host. A successful package check alone does not
establish those later outcomes.

## Reproduce or update the environment

For normal HAIC use, follow Steps 1–2 of the setup guide, then run:

```bash
bash slurm/synthetic-training/setup-environment.sh
bash slurm/synthetic-training/download-students.sh
bash slurm/synthetic-training/setup-environment.sh --check
```

Inside a GPU allocation, add:

```bash
bash slurm/synthetic-training/setup-environment.sh --check --require-cuda
```

Maintain the lock from the study project only. After intentionally editing
its manifest, a maintainer can regenerate and check it with:

```bash
uv lock --project slurm/synthetic-training --python 3.11
uv lock --project slurm/synthetic-training --check --offline --python 3.11
```

Validate a changed lock in a fresh study environment before replacing an
environment used by queued or running jobs. Changes to the MMPose revision
also require matching configuration assets and a newly reviewed pilot JSON.
The old `constraints-haic.txt` remains as historical evidence; the manifest
and lock are the current installation contract.

Local regression commands used the existing project environment:

```bash
.venv/bin/python -m unittest discover -s tests/synthetic_training -v
bash -n slurm/synthetic-training/setup-environment.sh \
  slurm/synthetic-training/download-students.sh slurm/synthetic-training/common.sh
git diff --check
```

The actual-kernel test needs permission to bind a local loopback socket. It
skips explicitly on hosts that deny that permission; the recorded successful
run allowed local ports and executed a real kernel. The source-build fixture,
Linux preview and separate numerical commands are recorded with their logs in
the linked evidence directory.
