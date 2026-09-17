#!/usr/bin/env bash
# Shared launcher environment; no source-data discovery or mode overrides.
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT to the gavd6 checkout}"
[[ -d "$GAVD6_ROOT/src/gavd6_sjepa" ]] || { echo "Invalid GAVD6_ROOT: $GAVD6_ROOT" >&2; exit 1; }
export GAVD6_ROOT="$(cd "$GAVD6_ROOT" && pwd -P)"
export TG_PYTHON="${TG_PYTHON:-$GAVD6_ROOT/.venv/bin/python}"
[[ -x "$TG_PYTHON" ]] || { echo "Set TG_PYTHON to an existing compatible interpreter." >&2; exit 1; }
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
case "$CUBLAS_WORKSPACE_CONFIG" in
  :4096:8|:16:8) ;;
  *) echo "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8." >&2; exit 2 ;;
esac
threads="${TG_TORCH_THREADS:-4}"
[[ "$threads" =~ ^[1-9][0-9]*$ ]] || { echo "TG_TORCH_THREADS must be positive." >&2; exit 2; }
allocated="${SLURM_CPUS_PER_TASK:-$threads}"
if (( threads > allocated )); then threads="$allocated"; fi
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
cd "$GAVD6_ROOT"

tg_notebook() {
  local number="$1" stage="$2"
  : "${TG_RUN_ROOT:?Export TG_RUN_ROOT or use submit.sh with a configured run_root}"
  local options=(--notebook "$number" --stage "$stage" --run-root "$TG_RUN_ROOT"
                 --phase "${TG_PHASE:-pilot}" --execute)
  [[ -z "${TG_CONFIG:-}" ]] || options+=(--config "$TG_CONFIG")
  [[ -z "${TG_NOTEBOOK_OUTPUT_DIR:-}" ]] || options+=(--output-dir "$TG_NOTEBOOK_OUTPUT_DIR")
  if [[ "$stage" == masked || "$stage" == future ]]; then
    local task="${SLURM_ARRAY_TASK_ID:-${TG_TASK_ID:-}}"
    [[ "$task" =~ ^[0-9]+$ ]] || { echo "Training needs a global TG_TASK_ID/SLURM_ARRAY_TASK_ID." >&2; return 2; }
    options+=(--task-id "$task")
  fi
  "$TG_PYTHON" "$GAVD6_ROOT/scripts/research_directions/temporal_gait/execute_notebook.py" "${options[@]}"
}
