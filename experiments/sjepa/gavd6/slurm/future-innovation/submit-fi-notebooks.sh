#!/usr/bin/env bash
# Independent notebook submission path; the original CLI jobs remain available.
set -euo pipefail
phase="${1:-}"
if [[ "$phase" != prepare && "$phase" != compute && "$phase" != all ]]; then
  echo "Usage: bash slurm/future-innovation/submit-fi-notebooks.sh prepare|compute|all" >&2
  exit 2
fi
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
if [[ "$phase" == compute && ! -f "$FI_RUN_ROOT/config/cohort-contract.json" ]]; then
  echo "No completed cohort at $FI_RUN_ROOT. Run prepare and wait, or use all." >&2
  exit 1
fi
submit_notebook() {
  local label="$1" script="$2" dependency="$3" suffix='%j' job
  [[ "$label" != fit ]] || suffix='%A_%a'
  local options=(--parsable --export=ALL --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT"
                 --output="$FI_RUN_ROOT/logs/notebook-$label-$suffix.out"
                 --error="$FI_RUN_ROOT/logs/notebook-$label-$suffix.err")
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency")
  # Explicitly propagate sbatch failure even when this function is command-substituted.
  if ! job="$(sbatch "${options[@]}" "$GAVD6_ROOT/slurm/future-innovation/$script")"; then
    echo "Notebook submission failed at $label; see logs/notebook-submissions.tsv for earlier jobs." >&2
    return 1
  fi
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $job" >&2; return 1; }
  printf '%s\t%s\t%s\n' "$label" "$job" "$dependency" >> "$FI_RUN_ROOT/logs/notebook-submissions.tsv"
  printf '%s\n' "$job"
}
teacher_dependency=''
report_jobs=''
if [[ "$phase" == prepare || "$phase" == all ]]; then
  setup="$(submit_notebook setup 10-notebook-00-setup.sbatch '')"
  cohort="$(submit_notebook cohort 11-notebook-01-cohort.sbatch "afterok:$setup")"
  echo "Submitted notebook setup=$setup cohort=$cohort."
  teacher_dependency="afterok:$cohort"
  report_jobs="$setup:$cohort:"
fi
if [[ "$phase" == compute || "$phase" == all ]]; then
  teacher="$(submit_notebook teacher 12-notebook-02-teacher.sbatch "$teacher_dependency")"
  fit="$(submit_notebook fit 13-notebook-03-fit.sbatch "afterok:$teacher")"
  report="$(submit_notebook report 14-notebook-04-report.sbatch "afterany:${report_jobs}$teacher:$fit")"
  echo "Submitted notebook teacher=$teacher fit=$fit report=$report."
fi
