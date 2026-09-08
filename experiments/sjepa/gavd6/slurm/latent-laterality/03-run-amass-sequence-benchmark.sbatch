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
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"

GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7"

ll_require_file "$GAUGE_MANIFEST" "gauge manifest"
ll_require_dir "$TENSOR_ROOT" "tensor root"
ll_require_empty_output "$OUTPUT_DIR"

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality benchmark \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed 7 \
  --output-dir "$OUTPUT_DIR"
