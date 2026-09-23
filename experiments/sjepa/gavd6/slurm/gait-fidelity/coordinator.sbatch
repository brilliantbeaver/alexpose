#!/usr/bin/env bash
# CPU-only allocation. submit.py supplies account, partition, time and log path.
set -euo pipefail
[[ $# == 3 ]] || { echo 'Usage: coordinator.sbatch WORK MAX_JOBS full|prepare-only|gavd-only|gavd-confirmation' >&2; exit 2; }
: "${GF_ROOT:?Saved GF_ROOT is required by the Slurm job}"
source "$GF_ROOT/slurm/gait-fidelity/common.sh"
gf_load_session "$1"
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'Submit this script through run.sh launch.' >&2; exit 1; }
gf_arguments=(run --work "$GF_WORK" --max-jobs "$2")
case "$3" in
  prepare-only) gf_arguments+=(--prepare-only) ;;
  gavd-only) gf_arguments+=(--gavd-only) ;;
  gavd-confirmation) gf_arguments+=(--gavd-confirmation) ;;
  confirmation-only) gf_arguments+=(--confirmation-only) ;;
  full) ;;
  *) echo 'Expected full, prepare-only, gavd-only or gavd-confirmation.' >&2; exit 2 ;;
esac
exec "$GF_PYTHON" -u -m gavd6_sjepa.research_directions.gait_fidelity "${gf_arguments[@]}"
