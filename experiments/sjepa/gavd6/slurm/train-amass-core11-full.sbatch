#!/usr/bin/env bash

#SBATCH --job-name=gp-jepa
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

: "${GAVD6_ROOT:?Set GAVD6_ROOT to the repaired gavd6 checkout before submitting}"
: "${AMASS_RUN_ROOT:?Set AMASS_RUN_ROOT to the converted AMASS run root before submitting}"

cd "$GAVD6_ROOT"

export AMASS_RUN_TRAINING=1
export AMASS_PROFILE=full
export AMASS_SYNTHETIC_SMOKE=0
export AMASS_DEVICE=cuda

# Run the standard single-view S-JEPA control and all three capacity-matched
# paired variants with seed 7.
# Keep the real test split sealed until the architecture and protocol are frozen.
export AMASS_SEEDS=7
export AMASS_VARIANTS=standard_sjepa
export AMASS_EVALUATE_TEST=0

# Freeze the repaired pilot protocol. Do not inherit scientific or execution
# settings from the shell that submits this job.
export AMASS_EPOCHS=100
export AMASS_PATIENCE=12
export AMASS_EVALUATION_DRAWS=1
export AMASS_NUM_WORKERS=4

export AMASS_RUN_ID="${AMASS_RUN_ID:-amass-core11-full-repaired-seed7-v3}"
export AMASS_CONVERSION_MANIFEST="${AMASS_CONVERSION_MANIFEST:-$AMASS_RUN_ROOT/manifests/amass_core11_conversion.csv}"
export AMASS_TENSOR_ROOT="${AMASS_TENSOR_ROOT:-$AMASS_RUN_ROOT/core11}"
export AMASS_OUTPUT_DIR="${AMASS_OUTPUT_DIR:-$AMASS_RUN_ROOT/repaired-jepa-seed7-v3}"

if [[ ! -f "$AMASS_CONVERSION_MANIFEST" ]]; then
  echo "Missing AMASS conversion manifest: $AMASS_CONVERSION_MANIFEST" >&2
  exit 1
fi
if [[ ! -d "$AMASS_TENSOR_ROOT" ]]; then
  echo "Missing AMASS Core11 tensor root: $AMASS_TENSOR_ROOT" >&2
  exit 1
fi

mkdir -p "$AMASS_OUTPUT_DIR"

echo "Starting repaired AMASS JEPA seed-7 pilot"
echo "  checkout: $GAVD6_ROOT"
echo "  manifest: $AMASS_CONVERSION_MANIFEST"
echo "  tensors:  $AMASS_TENSOR_ROOT"
echo "  output:   $AMASS_OUTPUT_DIR"
echo "  test evaluation: disabled"

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface amass train
