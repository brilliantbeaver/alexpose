#!/usr/bin/env bash

#SBATCH --job-name=ll-amass-confirm-test-v2
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-amass-confirm-test-v2-%j.out
#SBATCH --error=slurm-ll-amass-confirm-test-v2-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
GATE_DECISION="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired/gate_decision.json"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
RUN_ROOT="$LATENT_LATERALITY_RUN_ROOT/decisive/amass-v2-chart-paired"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/amass-gauge-v2-confirmation-test"
run_dirs=(
  "correction-first-seed7=$RUN_ROOT/correction-first-seed7"
  "sg-jepa-seed7=$RUN_ROOT/sg-jepa-seed7"
  "correction-first-seed19=$RUN_ROOT/correction-first-seed19"
  "sg-jepa-seed19=$RUN_ROOT/sg-jepa-seed19"
  "correction-first-seed31=$RUN_ROOT/correction-first-seed31"
  "sg-jepa-seed31=$RUN_ROOT/sg-jepa-seed31"
)

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$GAUGE_MANIFEST" ]] || { echo "Missing v2 gauge manifest: $GAUGE_MANIFEST" >&2; exit 1; }
[[ -f "$GATE_DECISION" ]] || { echo "Missing v2 gate decision: $GATE_DECISION" >&2; exit 1; }
[[ -d "$TENSOR_ROOT" ]] || { echo "Missing AMASS tensors: $TENSOR_ROOT" >&2; exit 1; }
for item in "${run_dirs[@]}"; do
  [[ -f "${item#*=}/run_result.json" ]] || { echo "Missing completed run: ${item#*=}" >&2; exit 1; }
done
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not empty: $OUTPUT_DIR" >&2
  exit 1
fi

args=(
  --gate-decision "$GATE_DECISION"
  --gauge-manifest "$GAUGE_MANIFEST"
  --tensor-root "$TENSOR_ROOT"
  --output-dir "$OUTPUT_DIR"
  --mode confirmation
  --split test
  --device cuda
  --batch-size 64
)
for item in "${run_dirs[@]}"; do
  args+=(--run-dir "$item")
done
cd "$GAVD6_ROOT"
exec uv run --no-sync python scripts/evaluate_gauge_jepa.py "${args[@]}"
