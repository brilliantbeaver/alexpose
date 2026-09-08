#!/usr/bin/env bash

#SBATCH --job-name=ll-selected-gauge
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-ll-selected-gauge-%j.out
#SBATCH --error=slurm-ll-selected-gauge-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
ll_require_selected_route
: "${SELECTED_SOURCE_MANIFEST:?Point SELECTED_SOURCE_MANIFEST at the selected Core11 manifest}"
: "${SELECTED_TENSOR_ROOT:?Point SELECTED_TENSOR_ROOT at the selected tensor root}"

SELECTED_ROUTE_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME"
SELECTED_GAUGE_MANIFEST="$SELECTED_ROUTE_ROOT/gauge-seed7.csv"

ll_require_file "$SELECTED_SOURCE_MANIFEST" "selected source manifest"
ll_require_dir "$SELECTED_TENSOR_ROOT" "selected tensor root"
ll_require_absent "$SELECTED_GAUGE_MANIFEST"

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality build-manifest \
  --source-manifest "$SELECTED_SOURCE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --output-manifest "$SELECTED_GAUGE_MANIFEST" \
  --draws 1 \
  --seed 7
