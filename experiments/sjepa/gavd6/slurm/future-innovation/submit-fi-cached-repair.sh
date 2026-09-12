#!/usr/bin/env bash
# CPU-only direct-v3: verified inheritance -> five fitting tasks -> report.
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
: "${FI_PARENT_ROOT:?Export FI_PARENT_ROOT}"
: "${FI_CALIBRATION:?Export FI_CALIBRATION}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
export FI_EXPERIMENT_PROTOCOL=direct-v3
submit_cached() {
  local label="$1" script="$2" dependency="$3" job suffix='%j'
  [[ "$label" != fit ]] || suffix='%A_%a'
  local options=(--parsable --export=ALL --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT"
    --output="$FI_RUN_ROOT/logs/cached-$label-$suffix.out" --error="$FI_RUN_ROOT/logs/cached-$label-$suffix.err")
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency")
  job="$(sbatch "${options[@]}" "$GAVD6_ROOT/slurm/future-innovation/$script")" || return 1
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $job" >&2; return 1; }
  printf '%s\t%s\t%s\n' "$label" "$job" "$dependency" >> "$FI_RUN_ROOT/logs/cached-submissions.tsv"
  printf '%s\n' "$job"
}
init="$(submit_cached init 07-initialize-cached-repair.sbatch '')"
fit="$(submit_cached fit 05-fit-gate-models.sbatch "afterok:$init")"
report="$(submit_cached report 06-build-gate-report.sbatch "afterany:$init:$fit")"
echo "Submitted CPU cached repair init=$init fit=$fit report=$report"
