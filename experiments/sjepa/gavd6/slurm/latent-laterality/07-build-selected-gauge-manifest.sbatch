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
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
: "${SELECTED_ROUTE_NAME:?Set SELECTED_ROUTE_NAME to amass-only or gavd-only}"
: "${SELECTED_SOURCE_MANIFEST:?Point SELECTED_SOURCE_MANIFEST at the selected Core11 manifest}"
: "${SELECTED_TENSOR_ROOT:?Point SELECTED_TENSOR_ROOT at the selected tensor root}"

[[ "$SELECTED_ROUTE_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || {
  echo "SELECTED_ROUTE_NAME may contain only letters, digits, underscores, and hyphens" >&2
  exit 1
}
LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
SELECTED_ROUTE_ROOT="$LATENT_LATERALITY_RUN_ROOT/selected-route/$SELECTED_ROUTE_NAME"
SELECTED_GAUGE_MANIFEST="$SELECTED_ROUTE_ROOT/gauge-seed7.csv"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$SELECTED_SOURCE_MANIFEST" ]] || { echo "Missing selected source manifest: $SELECTED_SOURCE_MANIFEST" >&2; exit 1; }
[[ -d "$SELECTED_TENSOR_ROOT" ]] || { echo "Missing selected tensor root: $SELECTED_TENSOR_ROOT" >&2; exit 1; }
[[ ! -e "$SELECTED_GAUGE_MANIFEST" ]] || {
  echo "Refusing to replace existing selected-route gauge manifest: $SELECTED_GAUGE_MANIFEST" >&2
  exit 1
}

cd "$GAVD6_ROOT"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality build-manifest \
  --source-manifest "$SELECTED_SOURCE_MANIFEST" \
  --tensor-root "$SELECTED_TENSOR_ROOT" \
  --output-manifest "$SELECTED_GAUGE_MANIFEST" \
  --draws 1 \
  --seed 7
