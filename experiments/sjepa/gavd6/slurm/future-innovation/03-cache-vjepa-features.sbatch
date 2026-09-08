#!/usr/bin/env bash
#SBATCH --job-name=fi-teacher
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=fi-teacher-%j.out
#SBATCH --error=fi-teacher-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
[[ -f "$FI_RUN_ROOT/qc/alignment-review.json" ]] || {
  echo "Inspect overlays and record review-alignment before teacher extraction" >&2; exit 1;
}
fi_python cache-teacher --device cuda
