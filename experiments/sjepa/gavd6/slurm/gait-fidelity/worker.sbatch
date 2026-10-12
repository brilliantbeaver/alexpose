#!/usr/bin/env bash
# One GPU allocation. The coordinator supplies all resource and accounting caps.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: worker.sbatch WORK PHASE_ID ATTEMPT_DIRECTORY' >&2; exit 2; }
: "${GF_ROOT:?Saved GF_ROOT is required by the Slurm job}"
source "$GF_ROOT/slurm/gait-fidelity/common.sh"
gf_load_session "$1"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'GPU workers must run inside a Slurm allocation.' >&2; exit 1; }
exec "$GF_PYTHON" -u "$GF_ROOT/slurm/gait-fidelity/worker.py" "$GF_WORK" "$2" "$3"
