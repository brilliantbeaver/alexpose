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
export MP_MODE="${MP_MODE:-real}"
export MP_DEVICE="${MP_DEVICE:-cuda}"
export MP_EVALUATION_SPLIT="${MP_EVALUATION_SPLIT:-development}"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${MP_TORCH_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
cd "$GAVD6_ROOT"

mp_notebook() {
  local number="$1"
  local python="${MP_PYTHON:-$GAVD6_ROOT/.venv/bin/python}"
  [[ -x "$python" ]] || { echo "Python not found: $python. Set MP_PYTHON to the environment interpreter." >&2; return 1; }
  local options=(--notebook "$number" --run-root "$MP_RUN_ROOT" --mode "$MP_MODE" --device "$MP_DEVICE")
  [[ -z "${MP_CONFIG:-}" ]] || options+=(--config "$MP_CONFIG")
  [[ -z "${MP_NOTEBOOK_OUTPUT_DIR:-}" ]] || options+=(--output-dir "$MP_NOTEBOOK_OUTPUT_DIR")
  "$python" "$GAVD6_ROOT/scripts/research_directions/motion_preservation/execute_notebook.py" "${options[@]}"
}

echo "Motion preservation: mode=$MP_MODE root=$MP_RUN_ROOT job=${SLURM_JOB_ID:-local}"
