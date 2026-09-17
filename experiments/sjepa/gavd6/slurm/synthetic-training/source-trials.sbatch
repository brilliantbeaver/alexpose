#!/usr/bin/env bash
#SBATCH --job-name=st-trials
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/synthetic-training/common.sh"
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then st_array_student; fi
: "${ST_STUDENT_ID:?Set ST_STUDENT_ID for direct submission}"
st_notebook 02

