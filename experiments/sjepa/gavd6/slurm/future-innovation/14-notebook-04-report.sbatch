#!/usr/bin/env bash
#SBATCH --job-name=fi-nb-report
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=fi-nb-report-%j.out
#SBATCH --error=fi-nb-report-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-notebook-common.sh"
fi_notebook 04
