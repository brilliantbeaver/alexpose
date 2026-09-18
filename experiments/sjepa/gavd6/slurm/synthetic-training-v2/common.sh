#!/usr/bin/env bash
set -euo pipefail
: "${STV2_ROOT:?Source slurm/synthetic-training-v2/study.env}"
: "${STV2_PYTHON:?Explicit study interpreter required}"
[[ -d "$STV2_ROOT/src/gavd6_sjepa" ]] || { echo 'Configured HAIC checkout is missing' >&2; exit 1; }
[[ -x "$STV2_PYTHON" ]] || { echo 'Configured HAIC interpreter is missing' >&2; exit 1; }
cd "$STV2_ROOT"
export PYTHONPATH="$STV2_ROOT/src"
export PYTHONNOUSERSITE=1 PYOPENGL_PLATFORM=egl
"$STV2_PYTHON" -c 'import torch; assert torch.__version__ == "2.6.0+cu124", "Required HAIC Torch stack changed"'
