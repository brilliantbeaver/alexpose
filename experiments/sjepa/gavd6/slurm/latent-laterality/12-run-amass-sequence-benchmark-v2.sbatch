#!/usr/bin/env bash

#SBATCH --job-name=ll-amass-gate-v2
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-amass-gate-v2-%j.out
#SBATCH --error=slurm-ll-amass-gate-v2-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"

GAUGE_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/gauge-seed7-v2-chart-paired.csv"
TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/amass-benchmark-seed7-v2-chart-paired"

ll_require_file "$GAUGE_MANIFEST" "v2 gauge manifest"
ll_require_dir "$TENSOR_ROOT" "AMASS tensors"
ll_require_empty_output "$OUTPUT_DIR"

# Exit status 2 is an intentional scientific stop: inspect the complete
# diagnostics written before the program exits.  Do not submit training unless
# gate_decision.json says ready_for_sg_jepa=true.
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality benchmark \
  --gauge-manifest "$GAUGE_MANIFEST" \
  --tensor-root "$TENSOR_ROOT" \
  --seed 7 \
  --output-dir "$OUTPUT_DIR"
