#!/usr/bin/env bash

#SBATCH --job-name=ll-selected-gate
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-selected-gate-%j.out
#SBATCH --error=slurm-ll-selected-gate-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
: "${SELECTED_ROUTE_NAME:?Set SELECTED_ROUTE_NAME to amass-only or gavd-only}"
: "${SELECTED_TENSOR_ROOT:?Point SELECTED_TENSOR_ROOT at the selected tensor root}"

[[ "$SELECTED_ROUTE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "Invalid SELECTED_ROUTE_NAME: $SELECTED_ROUTE_NAME" >&2
  exit 1
}
LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
SELECTED_ROUTE_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME"
SELECTED_GAUGE_MANIFEST="$SELECTED_ROUTE_ROOT/gauge-seed7.csv"
OUTPUT_DIR="$SELECTED_ROUTE_ROOT/gate-seed7"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$SELECTED_GAUGE_MANIFEST" ]] || { echo "Missing selected gauge manifest: $SELECTED_GAUGE_MANIFEST" >&2; exit 1; }
[[ -d "$SELECTED_TENSOR_ROOT" ]] || { echo "Missing selected tensor root: $SELECTED_TENSOR_ROOT" >&2; exit 1; }
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not an empty directory: $OUTPUT_DIR" >&2
  exit 1
fi

cd "$GAVD6_ROOT"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality benchmark \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed 7 \
  --output-dir "$OUTPUT_DIR"
