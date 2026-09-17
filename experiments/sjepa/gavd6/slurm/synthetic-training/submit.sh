#!/usr/bin/env bash
# Source fitting never opens real reference outcomes. Real stages are explicit.
set -euo pipefail
phase="${1:-}"
case "$phase" in
  source|inventory|data|trials|fit|gavd|deploy|crossover|evaluate|confirmation|report) ;;
  *) echo "Usage: bash slurm/synthetic-training/submit.sh source|inventory|data|trials|fit|gavd|deploy|crossover|evaluate|confirmation|report [--dry-run]" >&2; exit 2 ;;
esac
[[ $# -le 2 ]] || { echo "Expected a phase and optional --dry-run." >&2; exit 2; }
dry_run=0
if [[ "${2:-}" == --dry-run ]]; then dry_run=1; elif [[ -n "${2:-}" ]]; then echo "Unknown option: $2" >&2; exit 2; fi
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
export GAVD6_ROOT
source "$GAVD6_ROOT/slurm/synthetic-training/common.sh"
if [[ "$phase" == confirmation ]]; then
  export ST_EVALUATION_SPLIT=confirmation
elif [[ "$phase" == evaluate ]]; then
  export ST_EVALUATION_SPLIT=early
fi

dependency="${ST_DEPENDENCY:-}"
[[ -z "$dependency" || "$dependency" =~ ^(afterok:)?[0-9]+(:[0-9]+)*$ ]] || {
  echo "ST_DEPENDENCY must be a job ID or afterok:jobid[:jobid]." >&2; exit 2;
}
[[ -z "$dependency" || "$dependency" == afterok:* ]] || dependency="afterok:$dependency"
case "$phase" in
  source) stages=(00 01 02 03 07) ;;
  inventory) stages=(00) ;; data) stages=(01) ;; trials) stages=(02) ;;
  fit) stages=(03) ;; gavd) stages=(04) ;; deploy) stages=(05) ;;
  evaluate|confirmation) stages=(06) ;; report) stages=(07) ;;
  crossover) stages=(08) ;;
esac

# Resolve independent students once. Even dry-run checks the configured roster.
student_count=0
if [[ "$phase" == source || "$phase" == trials || "$phase" == deploy || "$phase" == crossover ]]; then
  roster_role=source
  if [[ "$phase" == deploy || "$phase" == crossover ]]; then roster_role=deployment; fi
  roster_options=(--list-students "$roster_role" --run-root "$ST_RUN_ROOT")
  [[ -z "${ST_CONFIG:-}" ]] || roster_options+=(--config "$ST_CONFIG")
  [[ -z "${ST_STUDENT_ID:-}" ]] || roster_options+=(--student-id "$ST_STUDENT_ID")
  student_lines="$(st_python "$GAVD6_ROOT/scripts/research_directions/synthetic_training/execute_notebook.py" "${roster_options[@]}")"
  students=()
  while IFS= read -r student; do
    [[ "$student" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || { echo "Invalid configured student ID: $student" >&2; exit 2; }
    students+=("$student")
  done <<< "$student_lines"
  student_count=${#students[@]}
  [[ "$student_count" -gt 0 ]] || { echo "No configured students." >&2; exit 2; }
  parallel="${ST_MAX_PARALLEL:-8}"
  [[ "$parallel" =~ ^[1-9][0-9]*$ ]] || { echo "ST_MAX_PARALLEL must be a positive integer." >&2; exit 2; }
  export ST_STUDENT_ROSTER="$ST_RUN_ROOT/logs/students-$roster_role-$(date -u +%Y%m%dT%H%M%SZ)-$$.txt"
  printf '%s\n' "${students[@]}" > "$ST_STUDENT_ROSTER"
  echo "Student array ($roster_role): ${students[*]}"
  if [[ "$student_count" -gt 1 && -n "${ST_NOTEBOOK_OUTPUT_DIR:-}" ]]; then
    echo "Unset ST_NOTEBOOK_OUTPUT_DIR for arrays; each student needs its own output folder." >&2; exit 2;
  fi
fi

echo "Executed notebooks: $ST_RUN_ROOT/notebook_runs/run-NN/[student or split]/timestamp/"
for number in "${stages[@]}"; do
  case "$number" in
    00) script=inventory.sbatch ;; 01) script=prepare-data.sbatch ;;
    02) script=source-trials.sbatch ;; 03) script=fit-selectors.sbatch ;;
    04) script=prepare-gavd.sbatch ;; 05) script=deploy.sbatch ;;
    06) script=evaluate.sbatch ;; 07) script=source-report.sbatch ;;
    08) script=crossover.sbatch ;;
  esac
  options=(--parsable --export=ALL --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT")
  if [[ "$number" == 02 || "$number" == 05 || "$number" == 08 ]]; then
    options+=(--array="0-$((student_count - 1))%$parallel"
              --output="$ST_RUN_ROOT/logs/$phase-$number-%A_%a.out"
              --error="$ST_RUN_ROOT/logs/$phase-$number-%A_%a.err")
  else
    options+=(--output="$ST_RUN_ROOT/logs/$phase-$number-%j.out"
              --error="$ST_RUN_ROOT/logs/$phase-$number-%j.err")
  fi
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency")
  [[ -z "${ST_ACCOUNT:-}" ]] || options+=(--account="$ST_ACCOUNT")
  [[ -z "${ST_PARTITION:-}" ]] || options+=(--partition="$ST_PARTITION")
  if [[ "$dry_run" == 1 ]]; then
    printf 'sbatch'; printf ' %q' "${options[@]}" "$GAVD6_ROOT/slurm/synthetic-training/$script"; printf '\n'
    dependency="afterok:DRY_$number"
  else
    job="$(sbatch "${options[@]}" "$GAVD6_ROOT/slurm/synthetic-training/$script")"
    job="${job%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "$phase" "$number" "$job" "$dependency" >> "$ST_RUN_ROOT/logs/submissions.tsv"
    echo "Submitted notebook $number as job $job."
    dependency="afterok:$job"
  fi
done
