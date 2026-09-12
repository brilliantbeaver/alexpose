#!/usr/bin/env bash
#SBATCH --job-name=fi-cached-init
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
: "${FI_PARENT_ROOT:?Export the original direct-v2 cache root}"
: "${FI_CALIBRATION:?Export the passing direct-v3 calibration JSON}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
fi_python init-cached-run --parent-root "$FI_PARENT_ROOT" --calibration "$FI_CALIBRATION"
fi_python cache-teacher --device cpu
fi_python audit-teacher --device cpu
