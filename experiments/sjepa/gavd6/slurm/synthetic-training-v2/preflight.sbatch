#!/usr/bin/env bash
#SBATCH --job-name=stv2-preflight
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --no-requeue
set -euo pipefail
source "$STV2_ROOT/slurm/synthetic-training-v2/common.sh"
: "${STV2_CONFIG:?Explicit run scope config required}"
: "${STV2_PREFLIGHT_OUTPUT:?Choose a unique preflight JSON path}"
exec "$STV2_PYTHON" -u scripts/research_directions/synthetic_training_v2/preflight.py --config "$STV2_CONFIG" --output "$STV2_PREFLIGHT_OUTPUT"
