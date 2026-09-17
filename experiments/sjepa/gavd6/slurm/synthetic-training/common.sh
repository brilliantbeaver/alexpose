#!/usr/bin/env bash
# Shared HAIC environment for synthetic-training notebooks.
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

: "${GAVD6_ROOT:?Export GAVD6_ROOT to the gavd6 checkout}"
: "${ST_RUN_ROOT:?Export ST_RUN_ROOT to the experiment output directory}"
: "${ST_PYTHON:?Export ST_PYTHON to the dedicated study environment; see setup-environment.sh}"
[[ "$ST_PYTHON" == /* && -x "$ST_PYTHON" ]] || {
  echo "Study Python not found at an absolute executable path: $ST_PYTHON. Run setup-environment.sh first." >&2
  exit 1
}
[[ -d "$GAVD6_ROOT/src/gavd6_sjepa" ]] || { echo "Invalid GAVD6_ROOT: $GAVD6_ROOT" >&2; exit 1; }
export GAVD6_ROOT="$(cd "$GAVD6_ROOT" && pwd -P)"
[[ "$ST_RUN_ROOT" == /* ]] || ST_RUN_ROOT="$GAVD6_ROOT/$ST_RUN_ROOT"
mkdir -p "$ST_RUN_ROOT/logs"
export ST_RUN_ROOT="$(cd "$ST_RUN_ROOT" && pwd -P)"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export OMP_NUM_THREADS="${ST_TORCH_THREADS:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
cd "$GAVD6_ROOT"
if [[ -n "${ST_CONFIG:-}" ]]; then
  [[ -f "$ST_CONFIG" ]] || { echo "ST_CONFIG not found: $ST_CONFIG" >&2; exit 1; }
fi

st_python() {
  local interpreter="$ST_PYTHON"
  [[ -x "$interpreter" ]] || { echo "Python not found: $interpreter. Set ST_PYTHON to the study environment." >&2; return 1; }
  "$interpreter" "$@"
}

st_array_student() {
  # The roster was resolved from train/validation or deployment configuration
  # on submission. One array task runs one independent estimator.
  : "${ST_STUDENT_ROSTER:?Use submit.sh trials/deploy, or set ST_STUDENT_ID and submit without an array}"
  : "${SLURM_ARRAY_TASK_ID:?A student roster requires a Slurm array task}"
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^[0-9]+$ ]] || { echo "Invalid array index." >&2; return 1; }
  [[ -f "$ST_STUDENT_ROSTER" ]] || { echo "Student roster not found: $ST_STUDENT_ROSTER" >&2; return 1; }
  export ST_STUDENT_ID="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$ST_STUDENT_ROSTER")"
  [[ -n "$ST_STUDENT_ID" ]] || { echo "No student for array task $SLURM_ARRAY_TASK_ID." >&2; return 1; }
}

st_notebook() {
  local number="$1"
  shift
  local options=(--notebook "$number" --run-root "$ST_RUN_ROOT")
  [[ -z "${ST_CONFIG:-}" ]] || options+=(--config "$ST_CONFIG")
  [[ -z "${ST_DEVICE:-}" ]] || options+=(--device "$ST_DEVICE")
  [[ -z "${ST_STUDENT_ID:-}" ]] || options+=(--student-id "$ST_STUDENT_ID")
  [[ -z "${ST_EVALUATION_SPLIT:-}" ]] || options+=(--evaluation-split "$ST_EVALUATION_SPLIT")
  [[ -z "${ST_NOTEBOOK_OUTPUT_DIR:-}" ]] || options+=(--output-dir "$ST_NOTEBOOK_OUTPUT_DIR")
  st_python "$GAVD6_ROOT/scripts/research_directions/synthetic_training/execute_notebook.py" "${options[@]}" "$@"
}

echo "Synthetic training: root=$ST_RUN_ROOT job=${SLURM_JOB_ID:-local} student=${ST_STUDENT_ID:-stage-wide}"
