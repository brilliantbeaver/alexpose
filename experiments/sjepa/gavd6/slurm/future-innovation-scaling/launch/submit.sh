#!/usr/bin/env bash
# One public launcher for calibration, reservation and the existing jobs 20–23.
set -euo pipefail
mode="${1:-all}"
dry="${2:-}"
case "$mode" in all|prepare|compute|fit|report|status|verify|notebooks) ;; *)
  echo 'Usage: submit.sh {all|prepare|compute|fit|report|status|verify|notebooks} [--dry-run]' >&2; exit 2;; esac
[[ $# -le 2 && ( -z "$dry" || "$dry" == --dry-run ) ]] || exit 2
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "$mode" == status || "$mode" == verify ]]; then
  [[ -z "$dry" ]] || { echo '--dry-run applies to submissions only' >&2; exit 2; }
  exec "$FI_PYTHON" slurm/future-innovation-scaling/launch/initialize.py "$mode"
fi
if [[ "$mode" == notebooks ]]; then
  "$FI_PYTHON" slurm/future-innovation-scaling/launch/notebooks.py --run-root "$FI_RUN_ROOT" --check
else
  "$FI_PYTHON" slurm/future-innovation-scaling/launch/initialize.py preflight --mode "$mode"
fi
if [[ "$dry" != --dry-run ]]; then mkdir -p "$FI_RUN_ROOT/logs"; fi
dependency=""
submitted=""
submit_stage() {
  local script="$1" tag="$2" kind="${3:-afterok}" id
  local -a args=(--parsable --kill-on-invalid-dep=yes --export=ALL --chdir="$GAVD6_ROOT"
    --output="$FI_RUN_ROOT/logs/$tag-%A_%a.out" --error="$FI_RUN_ROOT/logs/$tag-%A_%a.err")
  if [[ -n "$dependency" ]]; then args+=(--dependency="$kind:$dependency"); fi
  if [[ "$dry" == --dry-run ]]; then
    printf '%q ' sbatch "${args[@]}" "$GAVD6_ROOT/slurm/future-innovation-scaling/$script"
    printf '\n'
    dependency="DRY_$tag"
  else
    id="$(sbatch "${args[@]}" "$GAVD6_ROOT/slurm/future-innovation-scaling/$script")"
    id="${id%%;*}"
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $id" >&2; exit 2; }
    printf '%s\t%s\t%s\n' "$tag" "$id" "$dependency" >> "$FI_RUN_ROOT/logs/submissions.tsv"
    printf 'Submitted %s=%s\n' "$tag" "$id"
    dependency="$id"
  fi
  submitted="${submitted:+$submitted:}$dependency"
}
if [[ "$mode" == all || "$mode" == prepare ]]; then
  submit_stage launch/19-initialize.sbatch initialize
  submit_stage 20-prepare-scaling.sbatch pose
fi
if [[ "$mode" == all || "$mode" == compute ]]; then submit_stage 21-cache-scaling.sbatch cache; fi
if [[ "$mode" == all || "$mode" == compute || "$mode" == fit ]]; then submit_stage 22-fit-scaling.sbatch fit; fi
if [[ "$mode" != prepare && "$mode" != notebooks ]]; then submit_stage 23-report-scaling.sbatch report; fi
if [[ "$mode" != prepare ]]; then
  # Wait for all submitted jobs, including failed/canceled jobs. Reading partial
  # evidence must not race still-running fits when a report dependency fails.
  dependency="$submitted"
  submit_stage launch/24-notebooks.sbatch notebooks afterany
fi
