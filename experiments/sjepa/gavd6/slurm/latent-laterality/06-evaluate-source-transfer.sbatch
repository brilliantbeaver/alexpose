#!/usr/bin/env bash

#SBATCH --job-name=ll-source-readout
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-ll-source-readout-%j.out
#SBATCH --error=slurm-ll-source-readout-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"

GAVD_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv"
GAVD_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/source-transfer-readout-seed7"

checkpoints=(
  "amass-standard=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-only-standard_sjepa-seed7/stage-amass/seed-7_standard_sjepa_best.pt"
  "amass-equivariant=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-only-reflection_equivariant-seed7/stage-amass/seed-7_reflection_equivariant_best.pt"
  "gavd-standard=$LATENT_LATERALITY_RUN_ROOT/source-screen/gavd-only-standard_sjepa-seed7/stage-gavd/seed-7_standard_sjepa_best.pt"
  "gavd-equivariant=$LATENT_LATERALITY_RUN_ROOT/source-screen/gavd-only-reflection_equivariant-seed7/stage-gavd/seed-7_reflection_equivariant_best.pt"
  "staged-standard=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-to-gavd-standard_sjepa-seed7/stage-gavd/seed-7_standard_sjepa_best.pt"
  "staged-equivariant=$LATENT_LATERALITY_RUN_ROOT/source-screen/amass-to-gavd-reflection_equivariant-seed7/stage-gavd/seed-7_reflection_equivariant_best.pt"
)

ll_require_file "$GAVD_MANIFEST" "GAVD manifest"
ll_require_dir "$GAVD_TENSOR_ROOT" "GAVD tensors"
for item in "${checkpoints[@]}"; do
  path="${item#*=}"
  ll_require_file "$path" "source-screen checkpoint"
done
ll_require_empty_output "$OUTPUT_DIR"

args=(
  --gavd-manifest "$GAVD_MANIFEST"
  --gavd-tensor-root "$GAVD_TENSOR_ROOT"
  --output-dir "$OUTPUT_DIR"
  --split validation
  --device cuda
  --batch-size 64
)
for item in "${checkpoints[@]}"; do
  args+=(--checkpoint "$item")
done

exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality evaluate-source-transfer "${args[@]}"
