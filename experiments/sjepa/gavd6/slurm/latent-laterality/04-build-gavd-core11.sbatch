#!/usr/bin/env bash

#SBATCH --job-name=ll-gavd-core11
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-gavd-core11-%j.out
#SBATCH --error=slurm-ll-gavd-core11-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
: "${GAVD_FULL_ROOT:?Export GAVD_FULL_ROOT before submitting}"

GAVD_POSE_ROOT="${GAVD_POSE_ROOT:-$GAVD_FULL_ROOT/poses/latent-laterality}"
GAVD_POSE_MANIFEST="${GAVD_POSE_MANIFEST:-$GAVD_POSE_ROOT/full_gavd_pose_split_manifest.csv}"
OUTPUT_ROOT="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors"
OUTPUT_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv"
REJECTS="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/rejects.csv"

[[ -f "$GAVD_POSE_MANIFEST" ]] || {
  echo "Missing full-GAVD pose manifest: $GAVD_POSE_MANIFEST" >&2
  echo "The checked-in GAVD indexes are not a pose manifest; finish full-corpus pose extraction first." >&2
  exit 1
}
ll_require_absent "$OUTPUT_MANIFEST"

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface gavd convert-core11 \
  --pose-manifest "$GAVD_POSE_MANIFEST" \
  --output-root "$OUTPUT_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --rejects "$REJECTS"
