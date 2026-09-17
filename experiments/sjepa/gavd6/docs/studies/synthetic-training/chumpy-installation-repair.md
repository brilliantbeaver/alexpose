# Chumpy installation: environment selection, build dependencies and import checks

This document retains the original diagnosis and isolated repair commands.
Routine installation now uses the dedicated lockfile and declared build
dependencies through [`setup-environment.sh`](../../../slurm/synthetic-training/setup-environment.sh);
follow the current [setup guide](../../../slurm/synthetic-training/README.md).

The reported `uv add chumpy` failure has two independent causes. First, `uv add`
is a project operation: it resolves dependencies for `gavd6-sjepa` and normally
targets the checkout's `.venv`, even when the shell activates another environment.
The warning explicitly says that the active synthetic-training environment was
ignored, and the traceback shows a Python 3.12 build interpreter. Second,
Chumpy 0.70 imports `pip` from its build script without declaring it as a build
dependency. The temporary isolated build therefore lacks a required import.
See [uv's environment selection](https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path)
and [Chumpy's upstream build script](https://github.com/mattloper/chumpy/blob/580566eafc9ac68b2614b64d6f7aaa84eebb70da/setup.py).

The repair uses `uv pip install --python "$ST_PYTHON"` to target the study's
existing Python 3.11 environment. It installs `pip`, `setuptools` and `wheel`
there, then builds Chumpy with `--no-build-isolation`. This lets the build import
the installed `pip`; dependency resolution and study constraints still apply.
`uv add --active` would target the active environment but still resolve the
project's different Torch stack, so it is unsuitable for this setup.
`--frozen` would skip locking/syncing rather than repair the missing build import.

## A second failure found during verification

Building Chumpy 0.70 successfully is insufficient on Python 3.11. After the
build workaround succeeded locally, importing it failed with:

```text
AttributeError: module 'inspect' has no attribute 'getargspec'
```

Its installed source also imports removed NumPy scalar aliases. The official
Chumpy revision `580566eafc9ac68b2614b64d6f7aaa84eebb70da`, version 0.71, fixes
these import problems. The [upstream change](https://github.com/mattloper/chumpy/commit/580566eafc9ac68b2614b64d6f7aaa84eebb70da)
replaces `getargspec` with `getfullargspec`; the selected revision's package
initializer no longer imports the removed NumPy aliases.

The [README](../../../slurm/synthetic-training/README.md) now installs this exact
revision before MMPose, which lists Chumpy as a dependency. The
[study constraints](../../../slurm/synthetic-training/constraints-haic.txt) pin
version 0.71. Use the Git command: version 0.71 is selected from that source,
and the version constraint alone does not tell an installer where to obtain it.
The project's `pyproject.toml` and `uv.lock` are unchanged locally.

## Recovery on HAIC

Run these commands in the existing study environment:

```bash
export ST_PYTHON="/hai/scratch/$USER/envs/synthetic-training/bin/python"
export UV_NO_CONFIG=1
export UV_CONSTRAINT="$GAVD6_ROOT/slurm/synthetic-training/constraints-haic.txt"

"$ST_PYTHON" --version
uv pip install --python "$ST_PYTHON" pip setuptools wheel
uv pip install --python "$ST_PYTHON" --no-build-isolation \
  'chumpy @ git+https://github.com/mattloper/chumpy.git@580566eafc9ac68b2614b64d6f7aaa84eebb70da'

uv pip check --python "$ST_PYTHON"
"$ST_PYTHON" -c 'import chumpy, numpy; print(chumpy.__version__, numpy.__version__)'
```

The import check should print `0.71 1.26.4`. Then repeat the MMPose installation
if it previously failed, and continue the remaining Step 2 checks. Activation
alone is not needed when each command supplies the interpreter explicitly.

On HAIC, `git diff -- pyproject.toml uv.lock` can reveal whether the failed
`uv add` left any project edits. Its error message does not establish that those
edits persisted. Review any diff before changing it; existing user changes must
be preserved.

## Measured local validation

The checks used a separate Python 3.11.12 environment under
[`work/artifacts/synthetic-training-chumpy-20260916/`](../../../work/artifacts/synthetic-training-chumpy-20260916/).
The project environment and the earlier body-model test environment were not
modified.

| Check | Result |
|---|---|
| Fresh isolated Chumpy 0.70 build | Reproduced `ModuleNotFoundError: No module named 'pip'` |
| Build after installing tools and disabling isolation | Chumpy 0.70 built and installed successfully |
| Import of installed Chumpy 0.70 | Failed with the removed `inspect.getargspec` call |
| Pinned official Chumpy 0.71 build/import | Passed with NumPy 1.26.4 and SciPy 1.14.1 |
| Runtime dependency metadata check | `uv pip check` passed for the eight installed packages |
| Bare Chumpy requirement after the pinned installation | Offline constrained installation preview retained version 0.71 and proposed no changes |
| Three focused numerical tests | Passed: polynomial values/derivatives, callback argument inspection, and a basic array-expression pickle roundtrip |
| Existing synthetic-training suite | 41 tests discovered: 35 passed and six optional dependency checks skipped in the project environment |

The [test source](../../../tests/synthetic_training/test_chumpy_compatibility.py)
uses the actual installed package. Expected polynomial values and derivatives
are analytical, with absolute tolerance `1e-12`. The test can be run after
copying the updated source to HAIC:

```bash
cd "$GAVD6_ROOT"
ST_TEST_CHUMPY=1 "$ST_PYTHON" -m unittest tests.synthetic_training.test_chumpy_compatibility -v
```

The regression command executed locally was
`.venv/bin/python -m unittest discover -s tests/synthetic_training -v`.
The six skips are the three Chumpy checks and three body-model checks that need
their dedicated environments; the Chumpy checks were run separately above.
Logs of the failing builds, successful installation and test runs remain in
the linked evidence directory.

These are targeted compatibility checks. An additional exploratory pickle test
of a shared expression graph, `x*x + 2*x`, failed with `KeyError: 'a'` in Chumpy's
`__setstate__`. The passing pickle fixture establishes only the tested array
case; it does not certify arbitrary Chumpy graphs or old licensed model files.
No graph-pickling behavior was changed or hidden behind a compatibility shim.

No HAIC job, GPU operation, complete MMPose import, V-JEPA forward pass or licensed
asset load was performed in this investigation. Package metadata consistency
and these CPU checks cannot establish those remaining runtime properties.
