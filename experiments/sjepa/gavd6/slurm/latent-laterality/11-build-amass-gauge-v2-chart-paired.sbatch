#!/usr/bin/env bash

# Protocol v2: each source draw is emitted as an exact complementary chart
# pair, and all true/pseudo boundaries receive the same short validity gap.
# This is a new artifact; it intentionally never replaces gauge-seed7.csv.

#SBATCH --job-name=ll-amass-gauge-v2
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-ll-amass-gauge-v2-%j.out
#SBATCH --error=slurm-ll-amass-gauge-v2-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
SOURCE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$SOURCE_MANIFEST" ]] || { echo "Missing source manifest: $SOURCE_MANIFEST" >&2; exit 1; }
[[ -d "$TENSOR_ROOT" ]] || { echo "Missing tensor root: $TENSOR_ROOT" >&2; exit 1; }
[[ ! -e "$OUTPUT_MANIFEST" ]] || {
  echo "Refusing to replace v2 gauge manifest: $OUTPUT_MANIFEST" >&2
  exit 1
}

cd "$GAVD6_ROOT"
exec uv run --no-sync python scripts/build_sequence_gauge_manifest.py \
  --source-manifest "$SOURCE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --draws 1 \
  --seed 7 \
  --chart-pairs \
  --boundary-mode gap \
  --boundary-radius-frames 1 \
  --nuisance-selection independent
