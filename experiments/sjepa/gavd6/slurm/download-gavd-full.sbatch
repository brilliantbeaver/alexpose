#!/usr/bin/env bash

# Download a shard of the full GAVD YouTube cache.  The submitter must set the
# two paths below in their shell before submitting; raw video must live in
# approved durable project/scratch storage, not in this checkout.
#
# Single-job use:
#   sbatch slurm/download-gavd-full.sbatch
# Array use (recommended only after a successful pilot):
#   sbatch --array=0-3 slurm/download-gavd-full.sbatch

#SBATCH --job-name=gavd-download
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00

set -euo pipefail

: "${GAVD6_ROOT:?Set GAVD6_ROOT to this gavd6 checkout before submitting}"
: "${GAVD_FULL_ROOT:?Set GAVD_FULL_ROOT to durable approved GAVD storage before submitting}"

cd "$GAVD6_ROOT"
mkdir -p "$GAVD_FULL_ROOT"/{cache/xdg,cache/matplotlib,reports,logs}

export XDG_CACHE_HOME="$GAVD_FULL_ROOT/cache/xdg"
export MPLCONFIGDIR="$GAVD_FULL_ROOT/cache/matplotlib"

shard_index="${SLURM_ARRAY_TASK_ID:-0}"
shard_count="${SLURM_ARRAY_TASK_COUNT:-1}"
report="$GAVD_FULL_ROOT/reports/video_download_audit_shard-${shard_index}-of-${shard_count}.csv"

echo "GAVD full YouTube downloader"
echo "  manifest: $GAVD_FULL_ROOT/manifests/gavd_full_sequences.csv"
echo "  cache:    $GAVD_FULL_ROOT/youtube/all"
echo "  shard:    $shard_index/$shard_count"
echo "  report:   $report"

exec uv run --no-sync python scripts/download_gavd_full.py \
  --manifest "$GAVD_FULL_ROOT/manifests/gavd_full_sequences.csv" \
  --youtube-dir "$GAVD_FULL_ROOT/youtube" \
  --report "$report" \
  --shard-index "$shard_index" \
  --shard-count "$shard_count"
