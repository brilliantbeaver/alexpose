# Repairing the synthetic-training body-model dependency

This document retains the original diagnosis and numerical evidence. Routine
installation now uses the dedicated lockfile through
[`setup-environment.sh`](../../../slurm/synthetic-training/setup-environment.sh);
follow the current [setup guide](../../../slurm/synthetic-training/README.md).

The September 16, 2026 installation failure comes from incompatible instructions:
the pose environment pins Torch 2.1.0, while the previously listed
`human_body_prior` Git revision declares `torch>=2.5,<2.6`. No Torch version can
satisfy both. The resolver correctly stops before installing the conflicting
package. This is independent of the earlier missing Python executable.

The [upstream metadata at the old pin](https://github.com/nghorbani/human_body_prior/blob/78c86eae5ed518ae22bf197fd74211bbfa45551a/pyproject.toml)
identifies that package as version 2.3.0. The initial compatibility review
checked the Torch/torchvision/MMCV combination but missed this additional
requirement. The installation instructions have now been corrected.

## Selected repair

Keep Torch 2.1.0, torchvision 0.16.0, the matching CUDA 12.1 MMCV 2.1.0 wheel,
and NumPy 1.26.4. Change the body-model source to the official revision
`4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc`, whose package version is 2.2.2.0.
Its [BodyModel implementation](https://github.com/nghorbani/human_body_prior/blob/4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc/src/human_body_prior/body_model/body_model.py)
provides `bm_fname`, `dmpl_fname`, SMPL-H shape parameters and dynamic deformation
coefficients, as required by the study's existing `SMPLHBody` adapter.

The older revision's `setup.py` has an empty `install_requires` list. That alone
does not establish runtime compatibility. The used body-model forward path
imports NumPy and Torch; the corrected installation explicitly requests both,
and numerical tests exercise the actual installed backend and production
adapter. VPoser training and other upstream tools are outside this verification.

The [study constraints](../../../slurm/synthetic-training/constraints-haic.txt)
now pin `human-body-prior==2.2.2.0`, and the
[installation guide](../../../slurm/synthetic-training/README.md) supplies the
exact Git revision. A bare PyPI installation is not a substitute: the older
PyPI API does not meet this study's dynamic-body interface. The repository-wide
`pyproject.toml`, `uv.lock`, scientific model code, existing environments and
saved experiments were not changed.

Neither `--no-deps` nor a forced Torch override is used. Upgrading only Torch
would invalidate the selected binary combination with MMCV; suppressing the
resolver conflict would leave the newer package outside its declared bounds.

## Local evidence

All evidence is retained under
[`work/artifacts/synthetic-training-body-dependency-20260916/`](../../../work/artifacts/synthetic-training-body-dependency-20260916/).

| Check | Result and scope |
|---|---|
| Original dependency resolution | Reproduced the same unsatisfiable Torch 2.1.0 / body-prior 2.3.0 requirements, targeting Linux x86-64 and Python 3.11 |
| Corrected dependency resolution | Torch 2.1.0, NumPy 1.26.4 and the pinned body backend resolve together, including with the corrected study constraints; 24 packages resolved for Linux |
| Isolated installation | Installed the exact body revision with Torch 2.1.0 and NumPy 1.26.4 in a separate local Python 3.11.12 environment; Git provenance checked in installed metadata |
| Package consistency | `uv pip check` passed for all 17 packages in that isolated CPU test environment |
| Body-model numerical integration | All three tests passed: shape/DMPL changes, root rotation/translation, and production-adapter batching/coordinate conversion |
| Existing synthetic-training suite | 35 passed; the three new body tests skipped in the ordinary project environment and were run successfully in the isolated pinned environment |
| Existing motion-preservation suite | 54 passed; one author-repository check and three checkpoint tests requiring the optional motion-preservation dependencies were skipped |

The numerical fixture uses a constructed 52-vertex, 52-joint mesh with 16 shape
and eight dynamic coefficients. It runs the real upstream body model, without
mocking it, and checks analytical expected vertices/joints with absolute
tolerance `2e-6`. Nonzero DMPL coefficients must change the output. The production
adapter processes five frames in batches of two and must preserve those changes
while converting coordinates. These invented coefficients are software tests,
not licensed body assets or experimental motion evidence.

The [test source](../../../tests/synthetic_training/test_body_model_compatibility.py)
is reusable on HAIC after installation. The
[installed metadata](../../../work/artifacts/synthetic-training-body-dependency-20260916/installed-body-metadata.json),
[original resolver failure](../../../work/artifacts/synthetic-training-body-dependency-20260916/incompatible-resolution.log),
[corrected Linux resolution](../../../work/artifacts/synthetic-training-body-dependency-20260916/constrained-linux.txt),
and [numerical test log](../../../work/artifacts/synthetic-training-body-dependency-20260916/body-model-tests.log)
distinguish package resolution, CPU execution and actual experimental validity.

## Recovery on HAIC

Keep the existing environment; rerun the corrected body-model installation:

```bash
export UV_NO_CONFIG=1
export UV_CONSTRAINT="$GAVD6_ROOT/slurm/synthetic-training/constraints-haic.txt"

uv pip install --python "$ST_PYTHON" \
  'torch==2.1.0' 'numpy==1.26.4' \
  'human-body-prior @ git+https://github.com/nghorbani/human_body_prior.git@4c246d8a83ce16d3cff9c79dcf04d81fa440a6bc'

uv pip check --python "$ST_PYTHON"
"$ST_PYTHON" -c 'import torch, importlib.metadata as m; from human_body_prior.body_model.body_model import BodyModel; print(torch.__version__, m.version("human-body-prior"))'
```

Expected versions are Torch `2.1.0` (normally shown as `2.1.0+cu121` on HAIC) and
human-body-prior `2.2.2.0`. After copying the new test file, run its CPU check:

```bash
cd "$GAVD6_ROOT"
ST_TEST_BODY_MODEL=1 PYTHONPATH="$GAVD6_ROOT/src" \
  "$ST_PYTHON" -m unittest tests.synthetic_training.test_body_model_compatibility -v
```

This command was exercised locally with the isolated interpreter; it was not
run on HAIC. Linux dependency resolution is not a Linux execution test. The
full MMPose/V-JEPA/GPU installation and forward passes with licensed body assets
remain unverified on HAIC, which is not accessible in this session.

## Reproducing the local checks

The original and corrected requirement inputs are retained with the logs.
The following commands were exercised from the checkout; the first deliberately
returns failure and the others succeed:

```bash
uv --no-config pip compile --python-version 3.11 \
  --python-platform x86_64-unknown-linux-gnu \
  work/artifacts/synthetic-training-body-dependency-20260916/incompatible.in

uv --no-config pip compile --python-version 3.11 \
  --python-platform x86_64-unknown-linux-gnu \
  --constraint slurm/synthetic-training/constraints-haic.txt \
  work/artifacts/synthetic-training-body-dependency-20260916/candidate.in \
  -o work/artifacts/synthetic-training-body-dependency-20260916/constrained-linux.txt

ST_TEST_BODY_MODEL=1 PYTHONPATH=src \
  work/artifacts/synthetic-training-body-dependency-20260916/venv/bin/python \
  -m unittest tests.synthetic_training.test_body_model_compatibility -v

.venv/bin/python -m unittest discover -s tests/synthetic_training -v
.venv/bin/python -m unittest discover -s tests/motion_preservation -v
```
