#!/usr/bin/env bash
# Explicit frozen root; dry-run prints shell-escaped submissions without writing.
set -euo pipefail
mode="${1:-all}"
dry="${2:-}"
case "$mode" in all|prepare|compute|fit|report) ;; *) echo 'Usage: submit-source-learning-curve.sh {all|prepare|compute|fit|report} [--dry-run]' >&2; exit 2;; esac
if [[ -n "$dry" && "$dry" != --dry-run ]]; then exit 2; fi
: "${GAVD6_ROOT:?}" "${FI_SCALING_ROOT:?}" "${FI_PARENT_ROOT:?}"
[[ "$GAVD6_ROOT" = /* && "$FI_SCALING_ROOT" = /* && "$FI_PARENT_ROOT" = /* ]] || { echo 'Use absolute roots' >&2; exit 2; }
[[ -f "$FI_SCALING_ROOT/config/study.json" ]] || { echo 'Freeze the source reservation first' >&2; exit 2; }
if [[ "$dry" != --dry-run ]]; then mkdir -p "$FI_SCALING_ROOT/logs"; fi
dependency=""
submit_stage() {
  local script="$1" tag="$2" id
  local -a args=(--parsable --kill-on-invalid-dep=yes --export=ALL
    --output="$FI_SCALING_ROOT/logs/$tag-%A_%a.out" --error="$FI_SCALING_ROOT/logs/$tag-%A_%a.err")
  if [[ -n "$dependency" ]]; then args+=(--dependency="afterok:$dependency"); fi
  if [[ "$dry" == --dry-run ]]; then
    printf '%q ' sbatch "${args[@]}" "$GAVD6_ROOT/slurm/future-innovation-scaling/$script"
    printf '\n'
    dependency="DRY_$tag"
  else
    id="$(sbatch "${args[@]}" "$GAVD6_ROOT/slurm/future-innovation-scaling/$script")"
    id="${id%%;*}"
    [[ "$id" =~ ^[0-9]+$ ]] || { echo 'Invalid sbatch response' >&2; exit 2; }
    printf '%s\t%s\t%s\n' "$tag" "$id" "$dependency" >> "$FI_SCALING_ROOT/logs/submissions.tsv"
    dependency="$id"
  fi
}
if [[ "$mode" == all || "$mode" == prepare ]]; then submit_stage 20-prepare-scaling.sbatch pose; fi
if [[ "$mode" == all || "$mode" == compute ]]; then submit_stage 21-cache-scaling.sbatch cache; fi
if [[ "$mode" == all || "$mode" == compute || "$mode" == fit ]]; then submit_stage 22-fit-scaling.sbatch fit; fi
if [[ "$mode" != prepare ]]; then submit_stage 23-report-scaling.sbatch report; fi
