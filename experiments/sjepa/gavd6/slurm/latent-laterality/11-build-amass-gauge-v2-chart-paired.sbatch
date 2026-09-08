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
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"

SOURCE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"

ll_require_file "$SOURCE_MANIFEST" "source manifest"
ll_require_dir "$TENSOR_ROOT" "tensor root"
ll_require_absent "$OUTPUT_MANIFEST"

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality build-manifest \
  --source-manifest "$SOURCE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --draws 1 \
  --seed 7 \
  --chart-pairs \
  --boundary-mode gap \
  --boundary-radius-frames 1 \
  --nuisance-selection independent
