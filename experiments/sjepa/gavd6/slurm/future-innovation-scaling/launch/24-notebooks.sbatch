#!/usr/bin/env bash
#SBATCH --job-name=fi-scale-notebooks
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail
source "$GAVD6_ROOT/slurm/future-innovation-scaling/launch/common.sh"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
"$FI_PYTHON" slurm/future-innovation-scaling/launch/notebooks.py --run-root "$FI_RUN_ROOT"
