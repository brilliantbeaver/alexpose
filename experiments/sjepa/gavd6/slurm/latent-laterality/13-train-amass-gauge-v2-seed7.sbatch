#!/usr/bin/env bash

# Submit only after the v2 gate has passed.  The entry point checks the gate
# again and verifies that it was produced from this exact manifest hash.

#SBATCH --job-name=ll-amass-gauge-v2
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-ll-amass-gauge-v2-%A_%a.out
#SBATCH --error=slurm-ll-amass-gauge-v2-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
ll_require_array_task 2

GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
GATE_DECISION="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired/gate_decision.json"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_ROOT="$LATENT_LATERALITY_RUN_ROOT/decisive/amass-v2-chart-paired"
arms=(correction_first_sjepa sg_jepa uniform_posterior)
labels=(correction-first sg-jepa uniform)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
label="${labels[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$OUTPUT_ROOT/$label-seed7"

ll_require_file "$GAUGE_MANIFEST" "v2 gauge manifest"
ll_require_file "$GATE_DECISION" "v2 gate decision"
ll_require_dir "$TENSOR_ROOT" "AMASS tensors"
ll_require_empty_output "$OUTPUT_DIR"

echo "AMASS v2 seed-7 task $SLURM_ARRAY_TASK_ID: arm=$arm"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality train \
  --arm "$arm" \
  --gate-decision "$GATE_DECISION" \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed 7 \
  --profile full \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
