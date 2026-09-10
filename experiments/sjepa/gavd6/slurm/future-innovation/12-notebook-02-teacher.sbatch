#!/usr/bin/env bash
#SBATCH --job-name=fi-nb-teacher
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --output=fi-nb-teacher-%j.out
#SBATCH --error=fi-nb-teacher-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-notebook-common.sh"
fi_notebook 02 --device cuda
