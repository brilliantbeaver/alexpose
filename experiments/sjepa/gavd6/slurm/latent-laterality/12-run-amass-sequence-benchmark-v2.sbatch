#!/usr/bin/env bash

#SBATCH --job-name=ll-amass-gate-v2
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-amass-gate-v2-%j.out
#SBATCH --error=slurm-ll-amass-gate-v2-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$GAUGE_MANIFEST" ]] || { echo "Missing v2 gauge manifest: $GAUGE_MANIFEST" >&2; exit 1; }
[[ -d "$TENSOR_ROOT" ]] || { echo "Missing AMASS tensors: $TENSOR_ROOT" >&2; exit 1; }
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not empty: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$GAVD6_ROOT"
# Exit status 2 is an intentional scientific stop: inspect the complete
# diagnostics written before the program exits.  Do not submit training unless
# gate_decision.json says ready_for_sg_jepa=true.
exec uv run --no-sync python scripts/run_sequence_benchmark.py \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed 7 \
  --output-dir "$OUTPUT_DIR"
