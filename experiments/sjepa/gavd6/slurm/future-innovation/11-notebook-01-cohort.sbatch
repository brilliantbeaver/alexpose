#!/usr/bin/env bash
#SBATCH --job-name=fi-nb-cohort
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=fi-nb-cohort-%j.out
#SBATCH --error=fi-nb-cohort-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-notebook-common.sh"
fi_notebook 01
