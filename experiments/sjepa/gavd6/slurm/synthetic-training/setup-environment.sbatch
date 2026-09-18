#!/usr/bin/env bash
#SBATCH --job-name=st-setup
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Source pilot-01.env before submission}"
: "${ST_RUN_ROOT:?Source pilot-01.env before submission}"
: "${ST_PYTHON:?Set the new synthetic-training-cu124 Python path in pilot-01.env}"
cd "$GAVD6_ROOT"
unset PYTHONHOME PYTHONPATH VIRTUAL_ENV
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export ST_TOOLCHAIN_ROOT="${ST_TOOLCHAIN_ROOT:-/hai/scratch/$USER/toolchains/synthetic-training-cu124}"
printf 'Setup job %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
bash slurm/synthetic-training/setup-cuda-toolkit.sh
source "$ST_TOOLCHAIN_ROOT/cuda.env"
bash slurm/synthetic-training/setup-environment.sh --require-cuda
echo "ST_ENVIRONMENT_READY"
