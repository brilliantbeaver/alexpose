#!/usr/bin/env bash

#SBATCH --job-name=gp-core11
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -eu

cd "$GAVD6_ROOT"

exec uv run --no-sync python scripts/convert_amass_core11.py \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$AMASS_RUN_ROOT/manifests/amass_raw_inventory_eligible.csv" \
  --subject-splits "$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --canonical-fps 30 \
  --device cuda \
  --batch-size 256 \
  --output-root "$AMASS_RUN_ROOT/core11" \
  --output-manifest "$AMASS_RUN_ROOT/manifests/amass_core11_conversion.csv" \
  --rejects "$AMASS_RUN_ROOT/manifests/amass_core11_conversion_rejects.csv"
