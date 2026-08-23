#!/usr/bin/env bash
# Convert the complete eligible AMASS inventory to the frozen core11-v1 contract.
#
# IMPORTANT: override --time at submission using the measured two-sequence smoke
# throughput when the 24-hour default is not appropriate:
#   sbatch --account="$HAIC_ACCOUNT" --time=HH:MM:SS --export=ALL \
#     slurm/convert-amass-core11-full.sbatch

#SBATCH --job-name=gp-core11-full
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/hai/scratch/%u/alexpose/experiments/sjepa/gavd6/data/amass/outputs/logs/%x-%j.out
#SBATCH --error=/hai/scratch/%u/alexpose/experiments/sjepa/gavd6/data/amass/outputs/logs/%x-%j.err

set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_env() {
  local name=$1
  [[ -n "${!name:-}" ]] || die "Set ${name} before submitting with --export=ALL."
}

require_file() {
  local path=$1
  [[ -f "$path" ]] || die "Required file does not exist: $path"
}

require_dir() {
  local path=$1
  [[ -d "$path" ]] || die "Required directory does not exist: $path"
}

require_env GAVD6_ROOT
require_env AMASS_EXTRACTED_ROOT
require_env AMASS_RUN_ROOT
require_env AMASS_BODY_MODEL_ROOT
require_env AMASS_SUBJECT_SPLITS

GAVD6_ROOT=$(realpath "$GAVD6_ROOT")
AMASS_EXTRACTED_ROOT=$(realpath "$AMASS_EXTRACTED_ROOT")
AMASS_RUN_ROOT=$(realpath "$AMASS_RUN_ROOT")
AMASS_BODY_MODEL_ROOT=$(realpath "$AMASS_BODY_MODEL_ROOT")
AMASS_SUBJECT_SPLITS=$(realpath "$AMASS_SUBJECT_SPLITS")
export GAVD6_ROOT AMASS_EXTRACTED_ROOT AMASS_RUN_ROOT AMASS_BODY_MODEL_ROOT AMASS_SUBJECT_SPLITS

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

INVENTORY="$AMASS_RUN_ROOT/manifests/amass_raw_inventory_eligible.csv"
EXPECTED_SUBJECT_SPLITS="$AMASS_RUN_ROOT/manifests/amass_subject_splits.csv"
OUTPUT_ROOT="$AMASS_RUN_ROOT/core11"
OUTPUT_MANIFEST="$AMASS_RUN_ROOT/manifests/amass_core11_conversion.csv"
REJECT_MANIFEST="$AMASS_RUN_ROOT/manifests/amass_core11_conversion_rejects.csv"
RUN_LOG="$AMASS_RUN_ROOT/reports/amass_core11_conversion-${SLURM_JOB_ID}.log"

require_dir "$GAVD6_ROOT"
require_dir "$AMASS_EXTRACTED_ROOT"
require_dir "$AMASS_RUN_ROOT"
require_dir "$AMASS_BODY_MODEL_ROOT"
require_file "$GAVD6_ROOT/scripts/convert_amass_core11.py"
require_file "$INVENTORY"
require_file "$EXPECTED_SUBJECT_SPLITS"
require_file "$AMASS_SUBJECT_SPLITS"
require_file "$AMASS_BODY_MODEL_ROOT/smplh/female/model.npz"
require_file "$AMASS_BODY_MODEL_ROOT/smplh/male/model.npz"
require_file "$AMASS_BODY_MODEL_ROOT/dmpls/female/model.npz"
require_file "$AMASS_BODY_MODEL_ROOT/dmpls/male/model.npz"

[[ "$AMASS_SUBJECT_SPLITS" == "$(realpath "$EXPECTED_SUBJECT_SPLITS")" ]] || die \
  "Production conversion requires $EXPECTED_SUBJECT_SPLITS; refusing temporary or alternate split $AMASS_SUBJECT_SPLITS"

command -v uv >/dev/null 2>&1 || die "uv is not available on PATH."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available on this GPU job."
command -v flock >/dev/null 2>&1 || die "flock is required to prevent concurrent conversion writers."

mkdir -p \
  "$AMASS_RUN_ROOT/logs" \
  "$AMASS_RUN_ROOT/manifests" \
  "$AMASS_RUN_ROOT/reports" \
  "$OUTPUT_ROOT"

# Only one production converter may write this output and manifest set at once.
exec 9>"$AMASS_RUN_ROOT/.amass-core11-conversion.lock"
flock -n 9 || die "Another production core-11 conversion holds the run-root lock."

cd "$GAVD6_ROOT"

eligible_sequences=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$INVENTORY")
(( eligible_sequences > 0 )) || die "Eligible inventory contains no data rows: $INVENTORY"

{
  echo "Full AMASS core-11 conversion"
  echo "job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_time_limit=${SLURM_TIMELIMIT:-unknown}"
  echo "project_root=$GAVD6_ROOT"
  echo "amass_root=$AMASS_EXTRACTED_ROOT"
  echo "run_root=$AMASS_RUN_ROOT"
  echo "body_model_root=$AMASS_BODY_MODEL_ROOT"
  echo "inventory=$INVENTORY"
  echo "inventory_sha256=$(sha256sum "$INVENTORY" | awk '{print $1}')"
  echo "eligible_sequences=$eligible_sequences"
  echo "subject_splits=$AMASS_SUBJECT_SPLITS"
  echo "subject_splits_sha256=$(sha256sum "$AMASS_SUBJECT_SPLITS" | awk '{print $1}')"
  echo "converter_sha256=$(sha256sum scripts/convert_amass_core11.py | awk '{print $1}')"
  echo "output_root=$OUTPUT_ROOT"
  echo "output_manifest=$OUTPUT_MANIFEST"
  echo "reject_manifest=$REJECT_MANIFEST"
  if [[ -n "${AMASS_SMOKE_ELAPSED_SECONDS:-}" ]]; then
    [[ "$AMASS_SMOKE_ELAPSED_SECONDS" =~ ^[0-9]+([.][0-9]+)?$ ]] || die \
      "AMASS_SMOKE_ELAPSED_SECONDS must be a nonnegative number, received: $AMASS_SMOKE_ELAPSED_SECONDS"
    awk -v count="$eligible_sequences" -v smoke="$AMASS_SMOKE_ELAPSED_SECONDS" \
      'BEGIN { printf "linear_smoke_extrapolation_hours=%.3f\n", (count / 2.0) * smoke / 3600.0 }'
  else
    echo "linear_smoke_extrapolation_hours=not_recorded"
    echo "WARNING: set AMASS_SMOKE_ELAPSED_SECONDS at submission to record the measured estimate."
  fi
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
} | tee "$RUN_LOG"

uv run --no-sync python -c \
  'import torch; import human_body_prior; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))' \
  2>&1 | tee -a "$RUN_LOG"

started_epoch=$(date +%s)
set +e
uv run --no-sync python scripts/convert_amass_core11.py \
  --amass-root "$AMASS_EXTRACTED_ROOT" \
  --inventory "$INVENTORY" \
  --subject-splits "$AMASS_SUBJECT_SPLITS" \
  --body-model-root "$AMASS_BODY_MODEL_ROOT" \
  --canonical-fps 30 \
  --device cuda \
  --batch-size 256 \
  --verify-source-sha256 \
  --output-root "$OUTPUT_ROOT" \
  --output-manifest "$OUTPUT_MANIFEST" \
  --rejects "$REJECT_MANIFEST" \
  2>&1 | tee -a "$RUN_LOG"
converter_status=${PIPESTATUS[0]}
set -e
finished_epoch=$(date +%s)

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "elapsed_seconds=$((finished_epoch - started_epoch))"
  echo "exit_status=$converter_status"
  if [[ -f "$OUTPUT_MANIFEST" ]]; then
    echo "manifest_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$OUTPUT_MANIFEST")"
  fi
  if [[ -f "$REJECT_MANIFEST" ]]; then
    echo "reject_rows=$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$REJECT_MANIFEST")"
  fi
} | tee -a "$RUN_LOG"

if (( converter_status != 0 )); then
  echo "Conversion did not pass cleanly. Review $RUN_LOG and $REJECT_MANIFEST." | tee -a "$RUN_LOG"
fi

exit "$converter_status"
