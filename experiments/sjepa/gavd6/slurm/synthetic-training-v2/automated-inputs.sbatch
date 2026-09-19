#!/usr/bin/env bash
#SBATCH --job-name=stv2-automated-inputs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --no-requeue
set -euo pipefail
source "$STV2_ROOT/slurm/synthetic-training-v2/common.sh"
[[ $# == 2 ]] || { echo 'Usage: automated-inputs.sbatch CONFIG OUTPUT' >&2; exit 2; }
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
exec "$STV2_PYTHON" -u scripts/research_directions/synthetic_training_v2/automated_inputs.py --config "$1" --output-dir "$2"
