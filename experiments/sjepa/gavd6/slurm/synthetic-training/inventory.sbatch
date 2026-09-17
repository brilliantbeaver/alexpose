#!/usr/bin/env bash
#SBATCH --job-name=st-inventory
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/synthetic-training/common.sh"
st_notebook 00

