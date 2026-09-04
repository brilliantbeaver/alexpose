#!/usr/bin/env bash

#SBATCH --job-name=ll-gauge-confirm
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=slurm-ll-gauge-confirm-%A_%a.out
#SBATCH --error=slurm-ll-gauge-confirm-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
: "${SELECTED_ROUTE_NAME:?Set SELECTED_ROUTE_NAME to amass-only or gavd-only}"
: "${SELECTED_TENSOR_ROOT:?Point SELECTED_TENSOR_ROOT at the selected tensor root}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as its declared job array}"

[[ "$SELECTED_ROUTE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "Invalid SELECTED_ROUTE_NAME: $SELECTED_ROUTE_NAME" >&2
  exit 1
}
if [[ -n "${GAUGE_INITIAL_CHECKPOINT:-}" ]]; then
  echo "GAUGE_INITIAL_CHECKPOINT is intentionally unsupported; see the HAIC run guide." >&2
  exit 1
fi

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
SELECTED_ROUTE_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME"
SELECTED_GAUGE_MANIFEST="$SELECTED_ROUTE_ROOT/gauge-seed7.csv"
SELECTED_GATE_DECISION="$SELECTED_ROUTE_ROOT/gate-seed7/gate_decision.json"
arms=(correction_first_sjepa sg_jepa)
labels=(correction-first sg-jepa)
seeds=(19 31)
arm="${arms[$((SLURM_ARRAY_TASK_ID % 2))]}"
label="${labels[$((SLURM_ARRAY_TASK_ID % 2))]}"
seed="${seeds[$((SLURM_ARRAY_TASK_ID / 2))]}"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/decisive/$SELECTED_ROUTE_NAME/$label-seed$seed"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$SELECTED_GAUGE_MANIFEST" ]] || { echo "Missing selected gauge manifest: $SELECTED_GAUGE_MANIFEST" >&2; exit 1; }
[[ -f "$SELECTED_GATE_DECISION" ]] || { echo "Missing selected gate decision: $SELECTED_GATE_DECISION" >&2; exit 1; }
[[ -d "$SELECTED_TENSOR_ROOT" ]] || { echo "Missing selected tensor root: $SELECTED_TENSOR_ROOT" >&2; exit 1; }
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not an empty directory: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$GAVD6_ROOT"
echo "Confirmation task $SLURM_ARRAY_TASK_ID: route=$SELECTED_ROUTE_NAME arm=$arm seed=$seed"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality train \
  --arm "$arm" \
  --gate-decision "$SELECTED_GATE_DECISION" \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed "$seed" \
  --profile full \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
