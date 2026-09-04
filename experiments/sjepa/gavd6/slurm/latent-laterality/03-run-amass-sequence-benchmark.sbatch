#!/usr/bin/env bash

#SBATCH --job-name=ll-amass-gate
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-amass-gate-%j.out
#SBATCH --error=slurm-ll-amass-gate-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$GAUGE_MANIFEST" ]] || { echo "Missing gauge manifest: $GAUGE_MANIFEST" >&2; exit 1; }
[[ -d "$TENSOR_ROOT" ]] || { echo "Missing tensor root: $TENSOR_ROOT" >&2; exit 1; }
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not an empty directory: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$GAVD6_ROOT"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality benchmark \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed 7 \
  --output-dir "$OUTPUT_DIR"
