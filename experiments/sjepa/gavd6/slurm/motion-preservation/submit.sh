#!/usr/bin/env bash
# Submit only the requested research stage. No final test is included in pilot.
set -euo pipefail
phase="${1:-}"
case "$phase" in
  pilot|inventory|pairs|cache|fit|evaluate|final|gavd) ;;
  *) echo "Usage: bash slurm/motion-preservation/submit.sh pilot|inventory|pairs|cache|fit|evaluate|final|gavd [--dry-run]" >&2; exit 2 ;;
esac
dry_run=0
[[ $# -le 2 ]] || { echo "Expected one phase and optional --dry-run." >&2; exit 2; }
if [[ "${2:-}" == --dry-run ]]; then dry_run=1; elif [[ -n "${2:-}" ]]; then echo "Unknown option: $2" >&2; exit 2; fi
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export GAVD6_ROOT
source "$GAVD6_ROOT/slurm/motion-preservation/common.sh"
if [[ "$phase" == final ]]; then
  export MP_EVALUATION_SPLIT=final
else
  export MP_EVALUATION_SPLIT=development
fi
# The executor places each notebook in notebook_runs/run-XX. An explicit
# MP_NOTEBOOK_OUTPUT_DIR remains available as an exact-path override.
echo "Notebook outputs: $MP_RUN_ROOT/notebook_runs/run-XX"
dependency="${MP_DEPENDENCY:-}"
[[ -z "$dependency" || "$dependency" =~ ^(afterok:)?[0-9]+(:[0-9]+)*$ ]] || {
  echo "MP_DEPENDENCY must be a job ID or afterok:jobid[:jobid]." >&2; exit 2;
}
[[ -z "$dependency" || "$dependency" == afterok:* ]] || dependency="afterok:$dependency"
case "$phase" in
  pilot) stages=(00 01 02 03 04) ;;
  inventory) stages=(00) ;; pairs) stages=(01) ;; cache) stages=(02) ;;
  fit) stages=(03) ;; evaluate|final) stages=(04) ;; gavd) stages=(05) ;;
esac

for number in "${stages[@]}"; do
  case "$number" in
    00) script=inventory.sbatch ;; 01) script=controlled-pairs.sbatch ;;
    02) script=cache-evidence.sbatch ;; 03) script=train-calibrate.sbatch ;;
    04) script=evaluate.sbatch ;; 05) script=gavd-stress.sbatch ;;
  esac
  options=(--parsable --export=ALL --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT"
           --output="$MP_RUN_ROOT/logs/$phase-$number-%j.out"
           --error="$MP_RUN_ROOT/logs/$phase-$number-%j.err")
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency")
  [[ -z "${MP_ACCOUNT:-}" ]] || options+=(--account="$MP_ACCOUNT")
  [[ -z "${MP_PARTITION:-}" ]] || options+=(--partition="$MP_PARTITION")
  if [[ "$dry_run" == 1 ]]; then
    printf 'sbatch'; printf ' %q' "${options[@]}" "$GAVD6_ROOT/slurm/motion-preservation/$script"; printf '\n'
    dependency="afterok:DRY_${number}"
  else
    job="$(sbatch "${options[@]}" "$GAVD6_ROOT/slurm/motion-preservation/$script")"
    job="${job%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "$phase" "$number" "$job" "$dependency" >> "$MP_RUN_ROOT/logs/submissions.tsv"
    echo "Submitted notebook $number as job $job ($MP_EVALUATION_SPLIT)."
    dependency="afterok:$job"
  fi
done
