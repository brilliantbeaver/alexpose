#!/usr/bin/env bash
# Small shared environment for the Proposal 01 notebooks.
set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT to the gavd6 checkout}"
: "${MP_RUN_ROOT:?Export MP_RUN_ROOT to the experiment output directory}"
[[ -d "$GAVD6_ROOT/src/gavd6_sjepa" ]] || { echo "Invalid GAVD6_ROOT: $GAVD6_ROOT" >&2; exit 1; }
export GAVD6_ROOT="$(cd "$GAVD6_ROOT" && pwd -P)"
[[ "$MP_RUN_ROOT" == /* ]] || MP_RUN_ROOT="$GAVD6_ROOT/$MP_RUN_ROOT"
mkdir -p "$MP_RUN_ROOT/logs"
export MP_RUN_ROOT="$(cd "$MP_RUN_ROOT" && pwd -P)"
# Leave omitted mode/device values to MP_CONFIG or the saved run configuration.
# Exporting defaults here would silently replace those selected settings.
export MP_EVALUATION_SPLIT="${MP_EVALUATION_SPLIT:-development}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${MP_TORCH_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
cd "$GAVD6_ROOT"
if [[ -n "${MP_CONFIG:-}" ]]; then
  [[ -f "$MP_CONFIG" ]] || { echo "MP_CONFIG not found: $MP_CONFIG" >&2; exit 1; }
fi

mp_notebook() {
  local number="$1"
  shift
  local python="${MP_PYTHON:-$GAVD6_ROOT/.venv/bin/python}"
  [[ -x "$python" ]] || { echo "Python not found: $python. Set MP_PYTHON to the environment interpreter." >&2; return 1; }
  local options=(--notebook "$number" --run-root "$MP_RUN_ROOT")
  [[ -z "${MP_MODE:-}" ]] || options+=(--mode "$MP_MODE")
  [[ -z "${MP_DEVICE:-}" ]] || options+=(--device "$MP_DEVICE")
  [[ -z "${MP_CONFIG:-}" ]] || options+=(--config "$MP_CONFIG")
  [[ -z "${MP_NOTEBOOK_OUTPUT_DIR:-}" ]] || options+=(--output-dir "$MP_NOTEBOOK_OUTPUT_DIR")
  "$python" "$GAVD6_ROOT/scripts/research_directions/motion_preservation/execute_notebook.py" "${options[@]}" "$@"
}

echo "Motion preservation: mode=${MP_MODE:-config/default} root=$MP_RUN_ROOT job=${SLURM_JOB_ID:-local}"
