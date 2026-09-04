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
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
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

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
[[ -f "$GAVD_MANIFEST" ]] || { echo "Missing GAVD manifest: $GAVD_MANIFEST" >&2; exit 1; }
[[ -d "$GAVD_TENSOR_ROOT" ]] || { echo "Missing GAVD tensors: $GAVD_TENSOR_ROOT" >&2; exit 1; }
for item in "${checkpoints[@]}"; do
  path="${item#*=}"
  [[ -f "$path" ]] || { echo "Missing source-screen checkpoint: $path" >&2; exit 1; }
done
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not an empty directory: $OUTPUT_DIR" >&2
  exit 1
fi

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

cd "$GAVD6_ROOT"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality evaluate-source-transfer "${args[@]}"
