#!/usr/bin/env bash
# Sourced through GAVD6_ROOT, never relative to Slurm's spool directory.
set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT to the gavd6 checkout}"
: "${FI_RUN_ROOT:?Export FI_RUN_ROOT to a new versioned run outside the checkout}"
[[ -d "$GAVD6_ROOT/src/gavd6_sjepa" ]] || { echo "Invalid checkout: $GAVD6_ROOT" >&2; exit 1; }
case "$FI_RUN_ROOT/" in
  "$GAVD6_ROOT/"*) echo "Store experiment artifacts outside the checkout" >&2; exit 1 ;;
esac
export OMP_NUM_THREADS="${FI_TORCH_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export FI_TORCH_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=260905
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export UV_PROJECT_ENVIRONMENT="${FI_ENVIRONMENT:-$GAVD6_ROOT/.venv}"
cd "$GAVD6_ROOT"
mkdir -p "$FI_RUN_ROOT/logs"
fi_python() {
  uv run --no-sync python -m gavd6_sjepa.command_line_interface future-innovation "$@" --run-root "$FI_RUN_ROOT"
}
echo "FI run=$FI_RUN_ROOT job=${SLURM_JOB_ID:-local} task=${SLURM_ARRAY_TASK_ID:-none} host=$(hostname)"
