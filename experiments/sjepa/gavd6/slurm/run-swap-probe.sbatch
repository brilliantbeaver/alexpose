#!/usr/bin/env bash

#SBATCH --job-name=swap-probe
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-swap-probe-%j.out
#SBATCH --error=slurm-swap-probe-%j.err

set -euo pipefail

: "${GAVD6_ROOT:?Set GAVD6_ROOT to the gavd6 checkout before submitting}"
: "${AMASS_RUN_ROOT:?Set AMASS_RUN_ROOT to the converted AMASS run root}"

cd "$GAVD6_ROOT"

CHECKPOINT="${SWAP_PROBE_CHECKPOINT:-$AMASS_RUN_ROOT/outputs/repaired-jepa-seed7-v2/seed-7_standard_sjepa_best.pt}"
OUTPUT_DIR="${SWAP_PROBE_OUTPUT_DIR:-$AMASS_RUN_ROOT/outputs/swap-probe-seed7}"
CHECKPOINT_SHA256="${SWAP_PROBE_CHECKPOINT_SHA256:-d12ddf0a8412bcae58ed167cc1ec560b978bffa590fd631ab23e519216b646bd}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Missing exact frozen checkpoint: $CHECKPOINT" >&2
  exit 1
fi
if [[ ! -f "$AMASS_RUN_ROOT/manifests/amass_core11_conversion.csv" ]]; then
  echo "Missing AMASS manifest under $AMASS_RUN_ROOT/manifests" >&2
  exit 1
fi
if [[ ! -d "$AMASS_RUN_ROOT/core11" ]]; then
  echo "Missing Core11 tensors under $AMASS_RUN_ROOT/core11" >&2
  exit 1
fi
if [[ -d "$OUTPUT_DIR" ]] && [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Output directory is non-empty; choose a new SWAP_PROBE_OUTPUT_DIR: $OUTPUT_DIR" >&2
  exit 1
fi

echo "Swap probe: train_windows=${SWAP_PROBE_TRAIN_WINDOWS:-20000}, validation_windows=${SWAP_PROBE_VALIDATION_WINDOWS:-2800}, output=$OUTPUT_DIR"

exec uv run --no-sync python scripts/run_swap_probe.py \
  --manifest "$AMASS_RUN_ROOT/manifests/amass_core11_conversion.csv" \
  --tensor-root "$AMASS_RUN_ROOT/core11" \
  --checkpoint "$CHECKPOINT" \
  --expected-checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  --seed "${SWAP_PROBE_SEED:-7}" \
  --batch-size "${SWAP_PROBE_BATCH_SIZE:-64}" \
  --encoder-batch-size "${SWAP_PROBE_ENCODER_BATCH_SIZE:-256}" \
  --num-workers "${SWAP_PROBE_NUM_WORKERS:-4}" \
  --max-train-windows "${SWAP_PROBE_TRAIN_WINDOWS:-20000}" \
  --max-validation-windows "${SWAP_PROBE_VALIDATION_WINDOWS:-2800}" \
  --event-probability "${SWAP_PROBE_EVENT_PROBABILITY:-0.80}"
