#!/usr/bin/env bash
#SBATCH --job-name=tg-prepare
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/temporal-gait/common.sh"
tg_notebook 01 prepare
