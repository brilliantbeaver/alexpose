#!/usr/bin/env bash
#SBATCH --job-name=fi-poses
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=fi-poses-%j.out
#SBATCH --error=fi-poses-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
fi_python extract-poses
