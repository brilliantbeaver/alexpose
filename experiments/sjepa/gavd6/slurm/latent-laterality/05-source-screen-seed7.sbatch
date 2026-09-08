#!/usr/bin/env bash

#SBATCH --job-name=ll-source-screen
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=slurm-ll-source-screen-%A_%a.out
#SBATCH --error=slurm-ll-source-screen-%A_%a.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
# shellcheck source=slurm/latent-laterality/ll-common.sh
source "$GAVD6_ROOT/slurm/latent-laterality/ll-common.sh"
ll_require_array_task 5

AMASS_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
AMASS_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
GAVD_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv"
GAVD_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors"

routes=(amass-only gavd-only amass-to-gavd)
variants=(standard_sjepa reflection_equivariant)
route="${routes[$((SLURM_ARRAY_TASK_ID % 3))]}"
variant="${variants[$((SLURM_ARRAY_TASK_ID / 3))]}"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/source-screen/$route-$variant-seed7"

if [[ "$route" != "gavd-only" ]]; then
  ll_require_file "$AMASS_MANIFEST" "AMASS manifest"
  ll_require_dir "$AMASS_TENSOR_ROOT" "AMASS tensors"
fi
if [[ "$route" != "amass-only" ]]; then
  ll_require_file "$GAVD_MANIFEST" "GAVD manifest"
  ll_require_dir "$GAVD_TENSOR_ROOT" "GAVD tensors"
fi
ll_require_empty_output "$OUTPUT_DIR"

args=(
  --route "$route"
  --variant "$variant"
  --seed 7
  --profile full
  --device cuda
  --num-workers 4
  --output-dir "$OUTPUT_DIR"
)
case "$route" in
  amass-only)
    args+=(--amass-manifest "$AMASS_MANIFEST" --amass-tensor-root "$AMASS_TENSOR_ROOT")
    ;;
  gavd-only)
    args+=(--gavd-manifest "$GAVD_MANIFEST" --gavd-tensor-root "$GAVD_TENSOR_ROOT")
    ;;
  amass-to-gavd)
    args+=(
      --amass-manifest "$AMASS_MANIFEST" --amass-tensor-root "$AMASS_TENSOR_ROOT"
      --gavd-manifest "$GAVD_MANIFEST" --gavd-tensor-root "$GAVD_TENSOR_ROOT"
    )
    ;;
esac

echo "Source screen task $SLURM_ARRAY_TASK_ID: route=$route variant=$variant output=$OUTPUT_DIR"
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface laterality train-source-transfer "${args[@]}"
