#!/usr/bin/env bash
#SBATCH --job-name=fi-scale-init
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
source "$GAVD6_ROOT/slurm/source-scaling/common.sh"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$FI_PYTHON" scripts/research_directions/source_scaling/run_stage.py initialize
