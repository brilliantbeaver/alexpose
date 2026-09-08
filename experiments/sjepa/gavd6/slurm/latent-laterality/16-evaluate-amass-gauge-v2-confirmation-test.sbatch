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
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"

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

ll_require_file "$GAUGE_MANIFEST" "v2 gauge manifest"
ll_require_file "$GATE_DECISION" "v2 gate decision"
ll_require_dir "$TENSOR_ROOT" "AMASS tensors"
for item in "${run_dirs[@]}"; do
  ll_require_file "${item#*=}/run_result.json" "completed run"
done
ll_require_empty_output "$OUTPUT_DIR"

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
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality evaluate "${args[@]}"
