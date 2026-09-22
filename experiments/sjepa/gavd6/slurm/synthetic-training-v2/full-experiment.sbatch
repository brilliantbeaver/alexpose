#!/usr/bin/env bash
# All resources and the additional hard walltime cap are explicit in the launcher.
set -euo pipefail
: "${STV2_ROOT:?Saved checkout required}"
source "$STV2_ROOT/slurm/synthetic-training-v2/common.sh"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
[[ $# == 2 ]] || { echo 'Usage: full-experiment.sbatch REQUEST DIGEST' >&2; exit 2; }
exec "$STV2_PYTHON" -u scripts/research_directions/synthetic_training_v2/full_experiment.py run --request "$1" --expected-digest "$2"
