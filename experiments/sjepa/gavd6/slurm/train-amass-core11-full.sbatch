#!/usr/bin/env bash

#SBATCH --job-name=gp-core11-train
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -eu

cd "$GAVD6_ROOT"

export AMASS_RUN_TRAINING=1
export AMASS_PROFILE=full
export AMASS_DEVICE=cuda
export AMASS_RUN_ID="${AMASS_RUN_ID:-amass-core11-full-v1}"
export AMASS_OUTPUT_DIR="$AMASS_RUN_ROOT/run1"

mkdir -p "$AMASS_OUTPUT_DIR"

exec uv run --no-sync train-amass-core11
