#!/usr/bin/env bash
#SBATCH --job-name=st-preflight
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Source pilot-01.env before submission}"
source "$GAVD6_ROOT/slurm/synthetic-training/common.sh"
export ST_DEVICE=cuda PYOPENGL_PLATFORM=egl
bash slurm/synthetic-training/setup-environment.sh --check --require-cuda
exec "$ST_PYTHON" -u "$GAVD6_ROOT/scripts/research_directions/synthetic_training/preflight_experiment.py"
