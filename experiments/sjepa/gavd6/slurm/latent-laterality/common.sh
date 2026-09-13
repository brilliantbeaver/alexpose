#!/usr/bin/env bash
# Source through GAVD6_ROOT: sbatch runs a spool copy of each job script.
set -euo pipefail

ll_die() {
  echo "$*" >&2
  exit 1
}

ll_require_file() {
  [[ -f "$1" ]] || ll_die "Missing ${2:-file}: $1"
}

ll_require_dir() {
  [[ -d "$1" ]] || ll_die "Missing ${2:-directory}: $1"
}

ll_require_absent() {
  [[ ! -e "$1" && ! -L "$1" ]] || ll_die "Refusing to replace existing output: $1"
}

ll_require_empty_output() {
  local contents
  [[ ! -L "$1" ]] || ll_die "Refusing symlink output directory: $1"
  [[ -e "$1" ]] || return 0
  [[ -d "$1" && -r "$1" && -x "$1" ]] || ll_die "Output is not an accessible directory: $1"
  # Emit a fixed marker so even entries named only with newlines are detected.
  contents=$(find "$1" -mindepth 1 -maxdepth 1 -exec printf x \; -quit) ||
    ll_die "Cannot inspect output directory: $1"
  [[ -z "$contents" ]] || ll_die "Output already exists and is not empty: $1"
}

ll_require_array_task() {
  local max_task="$1"
  : "${SLURM_ARRAY_TASK_ID:?Submit this file as its declared job array}"
  # Validate before arithmetic evaluation or indexing (including overflow).
  [[ "$SLURM_ARRAY_TASK_ID" =~ ^(0|[1-9][0-9]*)$ &&
     ${#SLURM_ARRAY_TASK_ID} -le ${#max_task} ]] ||
    ll_die "SLURM_ARRAY_TASK_ID must be an integer from 0 to $max_task"
  [[ "$SLURM_ARRAY_TASK_ID" -le "$max_task" ]] ||
    ll_die "SLURM_ARRAY_TASK_ID must be an integer from 0 to $max_task"
}

ll_require_selected_route() {
  : "${SELECTED_ROUTE_NAME:?Set SELECTED_ROUTE_NAME to amass-only or gavd-only}"
  case "$SELECTED_ROUTE_NAME" in
    amass-only|gavd-only) ;;
    *) ll_die "SELECTED_ROUTE_NAME must be amass-only or gavd-only; staged decisive training is unsupported" ;;
  esac
}

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
ll_require_dir "$GAVD6_ROOT" "checkout"
LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"

# Resolve relative data/output paths consistently for both checks and Python.
cd "$GAVD6_ROOT"
