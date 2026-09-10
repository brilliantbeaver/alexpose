#!/usr/bin/env bash
#SBATCH --job-name=fi-report
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=fi-report-%j.out
#SBATCH --error=fi-report-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
if ! fi_python score-gate; then
  echo "Scoring unavailable or invalid; writing a STOP report from saved evidence" >&2
  fi_python build-report
  exit 0
fi
fi_python build-report
