#!/usr/bin/env bash

#SBATCH --job-name=ll-amass-convert
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-ll-amass-convert-%j.out
#SBATCH --error=slurm-ll-amass-convert-%j.err
#SBATCH --export=ALL

set -euo pipefail

: "${GAVD6_ROOT:?Export GAVD6_ROOT before submitting}"
: "${AMASS_EXTRACTED_ROOT:?Export AMASS_EXTRACTED_ROOT before submitting}"
: "${AMASS_RUN_ROOT:?Export AMASS_RUN_ROOT before submitting}"
: "${AMASS_BODY_MODEL_ROOT:?Export AMASS_BODY_MODEL_ROOT before submitting}"
: "${AMASS_SUBJECT_SPLITS:?Export AMASS_SUBJECT_SPLITS before submitting}"

LATENT_LATERALITY_RUN_ROOT="${LATENT_LATERALITY_RUN_ROOT:-$AMASS_RUN_ROOT/latent-laterality}"
AMASS_INVENTORY="${AMASS_INVENTORY:-$AMASS_RUN_ROOT/manifests/amass_raw_inventory_eligible.csv}"
OUTPUT_ROOT="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/core11"
OUTPUT_MANIFEST="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/manifest.csv"
REJECTS="$LATENT_LATERALITY_RUN_ROOT/amass-neutral/rejects.csv"

for path in "$GAVD6_ROOT" "$AMASS_EXTRACTED_ROOT" "$AMASS_BODY_MODEL_ROOT"; do
  [[ -d "$path" ]] || { echo "Missing directory: $path" >&2; exit 1; }
done
for path in \
  "$AMASS_INVENTORY" \
  "$AMASS_SUBJECT_SPLITS" \
  "$AMASS_BODY_MODEL_ROOT/smplh/male/model.npz" \
  "$AMASS_BODY_MODEL_ROOT/smplh/female/model.npz" \
  "$AMASS_BODY_MODEL_ROOT/dmpls/male/model.npz" \
  "$AMASS_BODY_MODEL_ROOT/dmpls/female/model.npz"; do
  [[ -f "$path" ]] || { echo "Missing file: $path" >&2; exit 1; }
done

cd "$GAVD6_ROOT"
echo "Converting eligible AMASS sequences to gauge-neutral Core11 tensors"
echo "  inventory: $AMASS_INVENTORY"
echo "  tensors:   $OUTPUT_ROOT"
echo "  manifest:  $OUTPUT_MANIFEST"

# The converter validates compatible existing tensors, so an interrupted job
# can be resubmitted. It never overwrites incompatible output without the
# deliberately omitted --overwrite flag.
exec uv run --no-sync python -m gavd6_sjepa.command_line_interface amass convert \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$AMASS_INVENTORY" \
  --subject-splits "$AMASS_SUBJECT_SPLITS" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --rejects "$REJECTS" \
  --canonical-fps 30 \
  --batch-size 256 \
  --forward-policy gauge-neutral-travel \
  --allow-rejects \
  --device cuda
