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
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
ll_require_selected_route
: "${SELECTED_TENSOR_ROOT:?Point SELECTED_TENSOR_ROOT at the selected tensor root}"

SELECTED_ROUTE_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME"
SELECTED_GAUGE_MANIFEST="$SELECTED_ROUTE_ROOT/gauge-seed7.csv"
OUTPUT_DIR="$SELECTED_ROUTE_ROOT/gate-seed7"

ll_require_file "$SELECTED_GAUGE_MANIFEST" "selected gauge manifest"
ll_require_dir "$SELECTED_TENSOR_ROOT" "selected tensor root"
ll_require_empty_output "$OUTPUT_DIR"

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality benchmark \
  --gauge-manifest "$SELECTED_GAUGE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --seed 7 \
  --output-dir "$OUTPUT_DIR"
