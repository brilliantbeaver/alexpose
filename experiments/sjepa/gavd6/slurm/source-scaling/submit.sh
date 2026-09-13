#!/usr/bin/env bash
# One public launcher for calibration, reservation and data preparation, fitting and inspection.
set -euo pipefail
mode="${1:-all}"
dry="${2:-}"
case "$mode" in all|prepare|compute|fit|report|status|verify|check|notebooks) ;; *)
  echo 'Usage: submit.sh {all|prepare|compute|fit|report|status|verify|check|notebooks} [--dry-run]' >&2; exit 2;; esac
[[ $# -le 2 && ( -z "$dry" || "$dry" == --dry-run ) ]] || exit 2
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
if [[ "$mode" == status || "$mode" == verify || "$mode" == check ]]; then
  [[ -z "$dry" ]] || { echo '--dry-run applies to submissions only' >&2; exit 2; }
  exec "$FI_PYTHON" scripts/research_directions/source_scaling/initialize.py "$mode"
fi
if [[ "$mode" == notebooks ]]; then
  "$FI_PYTHON" scripts/research_directions/source_scaling/inspect_notebooks.py --run-root "$FI_RUN_ROOT" --check
else
  "$FI_PYTHON" scripts/research_directions/source_scaling/initialize.py preflight --mode "$mode"
fi
dependency=""
submitted=""
submit_stage() {
  local script="$1" tag="$2" kind="${3:-afterok}" id log_root="$FI_RUN_ROOT/logs"
  [[ "$tag" != notebooks ]] || log_root="$FI_INSPECTION_ROOT/logs/slurm"
  if [[ "$dry" != --dry-run ]]; then mkdir -p "$log_root"; fi
  local -a args=(--parsable --kill-on-invalid-dep=yes --export=ALL --chdir="$GAVD6_ROOT"
    --output="$log_root/$tag-%A_%a.out" --error="$log_root/$tag-%A_%a.err")
  if [[ -n "$dependency" ]]; then args+=(--dependency="$kind:$dependency"); fi
  if [[ "$dry" == --dry-run ]]; then
    printf '%q ' sbatch "${args[@]}" "$GAVD6_ROOT/slurm/source-scaling/$script"
    printf '\n'
    dependency="DRY_$tag"
  else
    id="$(sbatch "${args[@]}" "$GAVD6_ROOT/slurm/source-scaling/$script")"
    id="${id%%;*}"
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $id" >&2; exit 2; }
    printf '%s\t%s\t%s\n' "$tag" "$id" "$dependency" >> "$log_root/submissions.tsv"
    printf 'Submitted %s=%s\n' "$tag" "$id"
    dependency="$id"
  fi
  submitted="${submitted:+$submitted:}$dependency"
}
if [[ "$mode" == all || "$mode" == prepare ]]; then
  submit_stage initialize.sbatch initialize
  submit_stage prepare-data.sbatch pose
fi
if [[ "$mode" == all || "$mode" == compute ]]; then submit_stage cache-features.sbatch cache; fi
if [[ "$mode" == all || "$mode" == compute || "$mode" == fit ]]; then submit_stage fit-models.sbatch fit; fi
if [[ "$mode" != prepare && "$mode" != notebooks ]]; then submit_stage build-report.sbatch report; fi
if [[ "$mode" != prepare ]]; then
  # Wait for all submitted jobs, including failed/canceled jobs. Reading partial
  # evidence must not race still-running fits when a report dependency fails.
  dependency="$submitted"
  submit_stage inspect-notebooks.sbatch notebooks afterany
fi
