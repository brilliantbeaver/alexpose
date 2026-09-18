#!/usr/bin/env bash
#SBATCH --job-name=stv2-prepare
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
set -euo pipefail
source "$STV2_ROOT/slurm/synthetic-training-v2/common.sh"
: "${STV2_PREPARATION_CONFIG:?Audited asset/preparation config required}"
: "${STV2_PREPARATION_OUTPUT:?Unique paired preparation output required}"
exec "$STV2_PYTHON" -u scripts/research_directions/synthetic_training_v2/prepare.py --config "$STV2_PREPARATION_CONFIG" --output "$STV2_PREPARATION_OUTPUT"
