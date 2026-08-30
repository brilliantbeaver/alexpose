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
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as its declared job array}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
AMASS_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
AMASS_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
GAVD_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/manifest.csv"
GAVD_TENSOR_ROOT="$LATENT_LATERALITY_RUN_ROOT/gavd-core11/tensors"

routes=(amass-only gavd-only amass-to-gavd)
variants=(standard_sjepa reflection_equivariant)
route="${routes[$((SLURM_ARRAY_TASK_ID % 3))]}"
variant="${variants[$((SLURM_ARRAY_TASK_ID / 3))]}"
OUTPUT_DIR="$LATENT_LATERALITY_RUN_ROOT/source-screen/$route-$variant-seed7"

[[ -d "$GAVD6_ROOT" ]] || { echo "Missing checkout: $GAVD6_ROOT" >&2; exit 1; }
if [[ "$route" != "gavd-only" ]]; then
  [[ -f "$AMASS_MANIFEST" ]] || { echo "Missing AMASS manifest: $AMASS_MANIFEST" >&2; exit 1; }
  [[ -d "$AMASS_TENSOR_ROOT" ]] || { echo "Missing AMASS tensors: $AMASS_TENSOR_ROOT" >&2; exit 1; }
fi
if [[ "$route" != "amass-only" ]]; then
  [[ -f "$GAVD_MANIFEST" ]] || { echo "Missing GAVD manifest: $GAVD_MANIFEST" >&2; exit 1; }
  [[ -d "$GAVD_TENSOR_ROOT" ]] || { echo "Missing GAVD tensors: $GAVD_TENSOR_ROOT" >&2; exit 1; }
fi
if [[ -e "$OUTPUT_DIR" ]] && { [[ ! -d "$OUTPUT_DIR" ]] || [[ -n "$(find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; }; then
  echo "Output already exists and is not an empty directory: $OUTPUT_DIR" >&2
  exit 1
fi

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

cd "$GAVD6_ROOT"
echo "Source screen task $SLURM_ARRAY_TASK_ID: route=$route variant=$variant output=$OUTPUT_DIR"
exec uv run --no-sync python scripts/train_latent_laterality.py "${args[@]}"
