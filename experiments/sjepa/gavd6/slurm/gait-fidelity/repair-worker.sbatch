#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: repair-worker.sbatch WORK PHASE ATTEMPT' >&2; exit 2; }
: "${GF_ROOT:?Saved GF_ROOT is required}"
source "$GF_ROOT/slurm/gait-fidelity/common.sh"
gf_load_session "$1"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'GPU workers require a Slurm allocation.' >&2; exit 1; }
exec "$GF_PYTHON" -u "$GF_ROOT/slurm/gait-fidelity/repair-worker.py" "$GF_WORK" "$2" "$3"
