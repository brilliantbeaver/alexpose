#!/usr/bin/env bash
#SBATCH --job-name=mp-diagnose
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/motion-preservation/common.sh"
export MP_DIAGNOSTIC_ROLES="${MP_DIAGNOSTIC_ROLES:-calibration,development}"
mp_notebook 06
