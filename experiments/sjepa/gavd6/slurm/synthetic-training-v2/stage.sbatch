#!/usr/bin/env bash
#SBATCH --job-name=stv2-stage
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --no-requeue
set -euo pipefail
: "${STV2_ROOT:?Source study.env}"
source "$STV2_ROOT/slurm/synthetic-training-v2/common.sh"
[[ $# == 2 ]] || { echo 'Usage: stage.sbatch CONFIG STAGE' >&2; exit 2; }
exec "$STV2_PYTHON" -u scripts/research_directions/synthetic_training_v2/run.py --config "$1" --stage "$2" --resume
