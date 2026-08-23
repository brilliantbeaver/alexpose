#!/usr/bin/env bash
# Run notebook 07 non-interactively and preserve a job-specific executed copy.
#
# Submit from the gavd6 repository root with the required environment exported:
#   sbatch --account="$HAIC_ACCOUNT" --export=ALL slurm/run-notebook-07.sbatch

#SBATCH --job-name=gp-nb07
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
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
require_env AMASS_RUN_ROOT
require_env AMASS_BODY_MODEL_ROOT

GAVD6_ROOT=$(realpath "$GAVD6_ROOT")
AMASS_RUN_ROOT=$(realpath "$AMASS_RUN_ROOT")
AMASS_BODY_MODEL_ROOT=$(realpath "$AMASS_BODY_MODEL_ROOT")
export GAVD6_ROOT AMASS_RUN_ROOT AMASS_BODY_MODEL_ROOT

export AMASS_DATA_ROOT="${AMASS_DATA_ROOT:-$GAVD6_ROOT/data/amass}"
export AMASS_VIS_REPORT_ROOT="${AMASS_VIS_REPORT_ROOT:-$AMASS_RUN_ROOT/reports/amass-pose-visualization}"
export AMASS_VIS_DEVICE=cuda
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg

SOURCE_NOTEBOOK="$GAVD6_ROOT/notebooks/07_visualize_amass_smplh_poses.ipynb"
NOTEBOOK_OUTPUT_DIR="$AMASS_RUN_ROOT/reports/notebooks"
OUTPUT_NOTEBOOK_NAME="07_visualize_amass_smplh_poses-${SLURM_JOB_ID}.ipynb"
RUN_LOG="$AMASS_RUN_ROOT/reports/notebook-07-${SLURM_JOB_ID}.log"

require_dir "$GAVD6_ROOT"
require_dir "$AMASS_DATA_ROOT"
require_dir "$AMASS_BODY_MODEL_ROOT"
require_file "$SOURCE_NOTEBOOK"
require_file "$AMASS_BODY_MODEL_ROOT/smplh/female/model.npz"
require_file "$AMASS_BODY_MODEL_ROOT/smplh/male/model.npz"
require_file "$AMASS_BODY_MODEL_ROOT/dmpls/female/model.npz"
require_file "$AMASS_BODY_MODEL_ROOT/dmpls/male/model.npz"

FIRST_POSE_FILE=$(find "$AMASS_DATA_ROOT" -type f -name '*_poses.npz' -print -quit)
[[ -n "$FIRST_POSE_FILE" ]] || die "No *_poses.npz recordings found under $AMASS_DATA_ROOT"

command -v uv >/dev/null 2>&1 || die "uv is not available on PATH."
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is not available on this GPU job."

mkdir -p \
  "$AMASS_RUN_ROOT/logs" \
  "$AMASS_RUN_ROOT/reports" \
  "$AMASS_VIS_REPORT_ROOT" \
  "$NOTEBOOK_OUTPUT_DIR"

# Keep notebook/runtime caches out of unwritable home-directory locations.
JOB_CACHE_ROOT="$AMASS_RUN_ROOT/.job-cache/notebook-07-${SLURM_JOB_ID}"
export MPLCONFIGDIR="$JOB_CACHE_ROOT/matplotlib"
export IPYTHONDIR="$JOB_CACHE_ROOT/ipython"
export JUPYTER_CONFIG_DIR="$JOB_CACHE_ROOT/jupyter"
export XDG_CACHE_HOME="$JOB_CACHE_ROOT/xdg"
mkdir -p "$MPLCONFIGDIR" "$IPYTHONDIR" "$JUPYTER_CONFIG_DIR" "$XDG_CACHE_HOME"

cd "$GAVD6_ROOT"

{
  echo "Notebook 07 batch execution"
  echo "job_id=$SLURM_JOB_ID"
  echo "host=$(hostname)"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "project_root=$GAVD6_ROOT"
  echo "data_root=$AMASS_DATA_ROOT"
  echo "body_model_root=$AMASS_BODY_MODEL_ROOT"
  echo "source_notebook=$SOURCE_NOTEBOOK"
  echo "source_notebook_sha256=$(sha256sum "$SOURCE_NOTEBOOK" | awk '{print $1}')"
  echo "output_notebook=$NOTEBOOK_OUTPUT_DIR/$OUTPUT_NOTEBOOK_NAME"
  nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader
} | tee "$RUN_LOG"

uv run --no-sync python -c \
  'import torch; import human_body_prior; assert torch.cuda.is_available(); print("torch", torch.__version__, "cuda", torch.version.cuda, "device", torch.cuda.get_device_name(0))' \
  2>&1 | tee -a "$RUN_LOG"

started_epoch=$(date +%s)
set +e
uv run --no-sync jupyter nbconvert \
  --to notebook \
  --execute "$SOURCE_NOTEBOOK" \
  --output "$OUTPUT_NOTEBOOK_NAME" \
  --output-dir "$NOTEBOOK_OUTPUT_DIR" \
  --ExecutePreprocessor.timeout=3600 \
  2>&1 | tee -a "$RUN_LOG"
notebook_status=${PIPESTATUS[0]}
set -e
finished_epoch=$(date +%s)

{
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "elapsed_seconds=$((finished_epoch - started_epoch))"
  echo "exit_status=$notebook_status"
} | tee -a "$RUN_LOG"

if (( notebook_status == 0 )); then
  require_file "$NOTEBOOK_OUTPUT_DIR/$OUTPUT_NOTEBOOK_NAME"
  echo "Executed notebook: $NOTEBOOK_OUTPUT_DIR/$OUTPUT_NOTEBOOK_NAME" | tee -a "$RUN_LOG"
fi

exit "$notebook_status"
