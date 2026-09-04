#!/usr/bin/env bash

#SBATCH --job-name=ll-amass-gauge
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-ll-amass-gauge-%j.out
#SBATCH --error=slurm-ll-amass-gauge-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
SOURCE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7.csv"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$SOURCE_MANIFEST" ]] || { echo "Missing source manifest: $SOURCE_MANIFEST" >&2; exit 1; }
[[ -d "$TENSOR_ROOT" ]] || { echo "Missing tensor root: $TENSOR_ROOT" >&2; exit 1; }
[[ ! -e "$OUTPUT_MANIFEST" ]] || {
  echo "Refusing to replace existing gauge manifest: $OUTPUT_MANIFEST" >&2
  exit 1
}

cd "$GAVD6_ROOT"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality build-manifest \
  --source-manifest "$SOURCE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --draws 1 \
  --seed 7
