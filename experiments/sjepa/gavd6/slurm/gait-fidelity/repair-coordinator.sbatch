#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 ]] || { echo 'Usage: repair-coordinator.sbatch WORK STAGE' >&2; exit 2; }
: "${GF_ROOT:?Saved GF_ROOT is required}"
source "$GF_ROOT/slurm/gait-fidelity/common.sh"
gf_load_session "$1"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Use repair.sh launch.' >&2; exit 1; }
exec "$GF_PYTHON" -u -m gavd6_sjepa.research_directions.gait_fidelity.repair run --work "$GF_WORK" --stage "$2"
