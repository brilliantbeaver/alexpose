#!/usr/bin/env bash

#SBATCH --job-name=ll-gauge-seed7
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=slurm-ll-gauge-seed7-%A_%a.out
#SBATCH --error=slurm-ll-gauge-seed7-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
ll_require_selected_route
: "${SELECTED_TENSOR_ROOT:?Point SELECTED_TENSOR_ROOT at the selected tensor root}"
ll_require_array_task 2

if [[ -n "${GAUGE_INITIAL_CHECKPOINT:-}" ]]; then
  echo "GAUGE_INITIAL_CHECKPOINT is intentionally unsupported by this array." >&2
  echo "The current SG/uniform implementation rebuilds its model after loading an initial checkpoint." >&2
  exit 1
fi

SELECTED_ROUTE_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME"
SELECTED_GAUGE_MANIFEST="$SELECTED_ROUTE_ROOT/gauge-seed7.csv"
SELECTED_GATE_DECISION="$SELECTED_ROUTE_ROOT/gate-seed7/gate_decision.json"
arms=(correction_first_sjepa sg_jepa uniform_posterior)
labels=(correction-first sg-jepa uniform)
arm="${arms[$SLURM_ARRAY_TASK_ID]}"
label="${labels[$SLURM_ARRAY_TASK_ID]}"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/decisive/$SELECTED_ROUTE_NAME/$label-seed7"

ll_require_file "$SELECTED_GAUGE_MANIFEST" "selected gauge manifest"
ll_require_file "$SELECTED_GATE_DECISION" "selected gate decision"
ll_require_dir "$SELECTED_TENSOR_ROOT" "selected tensor root"
ll_require_empty_output "$OUTPUT_DIR"

echo "Decisive seed-7 task $SLURM_ARRAY_TASK_ID: route=$SELECTED_ROUTE_NAME arm=$arm"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality train \
  --arm "$arm" \
  --gate-decision "$SELECTED_GATE_DECISION" \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed 7 \
  --profile full \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
