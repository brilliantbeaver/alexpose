#!/usr/bin/env bash

# Submit only after reviewing the seed-7 validation readout from script 14:
# SG-JEPA must improve on correction-first without an even-channel penalty,
# and the uniform control must not reproduce the gain. Repeat the prespecified
# contrast at seeds 19 and 31; keep test sealed until script 16.

#SBATCH --job-name=ll-amass-confirm-v2
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-ll-amass-confirm-v2-%A_%a.out
#SBATCH --error=slurm-ll-amass-confirm-v2-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
ll_require_array_task 3

GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
GATE_DECISION="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired/gate_decision.json"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_ROOT="$LATENT_LATERALITY_RUN_ROOT/decisive/amass-v2-chart-paired"
arms=(correction_first_sjepa sg_jepa)
labels=(correction-first sg-jepa)
seeds=(19 31)
arm="${arms[$((SLURM_ARRAY_TASK_ID % 2))]}"
label="${labels[$((SLURM_ARRAY_TASK_ID % 2))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 2))]}"
OUTPUT_DIR="$OUTPUT_ROOT/$label-seed$seed"

ll_require_file "$GAUGE_MANIFEST" "v2 gauge manifest"
ll_require_file "$GATE_DECISION" "v2 gate decision"
ll_require_dir "$TENSOR_ROOT" "AMASS tensors"
ll_require_empty_output "$OUTPUT_DIR"

echo "AMASS v2 confirmation task $SLURM_ARRAY_TASK_ID: arm=$arm seed=$seed"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality train \
  --arm "$arm" \
  --gate-decision "$GATE_DECISION" \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed "$seed" \
  --profile full \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
