#!/usr/bin/env bash
# Resources are explicit in the submission helper; this job requests no GPU.
set -euo pipefail
source "$STV2_ROOT/slurm/synthetic-training-v2/common.sh"
[[ $# == 1 ]] || { echo 'Usage: postrun-checks.sbatch REQUEST_JSON' >&2; exit 2; }
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MPLBACKEND=Agg MPLCONFIGDIR="$(dirname "$1")/matplotlib-cache"
exec "$STV2_PYTHON" -u scripts/research_directions/synthetic_training_v2/postrun_checks.py run --request "$1"
