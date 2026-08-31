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
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as its declared job array}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
GATE_DECISION="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired/gate_decision.json"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_ROOT="$LATENT_LATERALITY_RUN_ROOT/decisive/amass-v2-chart-paired"
arms=(correction_first_sjepa sg_jepa uniform_posterior)
labels=(correction-first sg-jepa uniform)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
label="${labels[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$OUTPUT_ROOT/$label-seed7"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$GAUGE_MANIFEST" ]] || { echo "Missing v2 gauge manifest: $GAUGE_MANIFEST" >&2; exit 1; }
[[ -f "$GATE_DECISION" ]] || { echo "Missing v2 gate decision: $GATE_DECISION" >&2; exit 1; }
[[ -d "$TENSOR_ROOT" ]] || { echo "Missing AMASS tensors: $TENSOR_ROOT" >&2; exit 1; }
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not empty: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$GAVD6_ROOT"
echo "AMASS v2 seed-7 task $SLURM_ARRAY_TASK_ID: arm=$arm"
exec uv run --no-sync python scripts/train_gauge_jepa.py \
  --arm "$arm" \
  --gate-decision "$GATE_DECISION" \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed 7 \
  --profile full \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
