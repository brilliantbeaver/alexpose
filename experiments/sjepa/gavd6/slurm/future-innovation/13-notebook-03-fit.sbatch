#!/usr/bin/env bash
#SBATCH --job-name=fi-nb-fit
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%5
#SBATCH --output=fi-nb-fit-%A_%a.out
#SBATCH --error=fi-nb-fit-%A_%a.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Submit as the declared five-fold array}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-notebook-common.sh"
fi_notebook 03 --outer-fold "$SLURM_ARRAY_TASK_ID"
