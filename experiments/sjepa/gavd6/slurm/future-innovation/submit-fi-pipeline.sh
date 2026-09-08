#!/usr/bin/env bash
# Two phases make the guide's pre-feature visual inspection a real gate.
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
phase="${1:-}"
if [[ "$phase" != prepare && "$phase" != compute ]]; then
  echo "Usage: bash slurm/future-innovation/submit-fi-pipeline.sh prepare|compute" >&2
  exit 2
fi
submit() {
  local label="$1" script="$2" dependency="$3"
  local options=(--parsable --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT"
                 --output="$FI_RUN_ROOT/logs/$label-%A_%a.out"
                 --error="$FI_RUN_ROOT/logs/$label-%A_%a.err")
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency")
  local job
  job="$(sbatch "${options[@]}" "$GAVD6_ROOT/slurm/future-innovation/$script")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $job" >&2; exit 1; }
  printf '%s\t%s\t%s\n' "$label" "$job" "$dependency" >> "$FI_RUN_ROOT/logs/submissions.tsv"
  printf '%s\n' "$job"
}
if [[ "$phase" == prepare ]]; then
  first="$(submit cohort 01-build-gate-cohort.sbatch '')"
  second="$(submit poses 02-extract-poses.sbatch "afterok:$first")"
  echo "Submitted cohort=$first poses=$second. Review one alignment overlay per fold before compute."
else
  [[ -f "$FI_RUN_ROOT/qc/alignment-review.json" ]] || { echo "Missing recorded alignment inspection" >&2; exit 1; }
  third="$(submit teacher 03-cache-vjepa-features.sbatch '')"
  fourth="$(submit audits 04-run-causal-and-target-audits.sbatch "afterok:$third")"
  fifth="$(submit fit 05-fit-gate-models.sbatch "afterok:$fourth")"
  sixth="$(submit report 06-build-gate-report.sbatch "afterany:$fifth")"
  echo "Submitted teacher=$third audits=$fourth fit=$fifth report=$sixth"
fi
