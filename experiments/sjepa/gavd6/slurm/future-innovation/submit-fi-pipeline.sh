#!/usr/bin/env bash
# Resume either phase, or submit the full dependency chain unattended.
set -euo pipefail
phase="${1:-}"
if [[ "$phase" != prepare && "$phase" != compute && "$phase" != all ]]; then
  echo "Usage: bash slurm/future-innovation/submit-fi-pipeline.sh prepare|compute|all" >&2
  exit 2
fi
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
if [[ "$phase" == compute && ! -f "$FI_RUN_ROOT/config/cohort-contract.json" ]]; then
  echo "No completed cohort at $FI_RUN_ROOT. Run prepare and wait for poses, or use all." >&2
  exit 1
fi
submit() {
  local label="$1" script="$2" dependency="$3"
  local suffix='%j'
  [[ "$label" != fit ]] || suffix='%A_%a'
  local options=(--parsable --export=ALL --kill-on-invalid-dep=yes --chdir="$GAVD6_ROOT"
                 --output="$FI_RUN_ROOT/logs/$label-$suffix.out"
                 --error="$FI_RUN_ROOT/logs/$label-$suffix.err")
  [[ -z "$dependency" ]] || options+=(--dependency="$dependency")
  local job
  job="$(sbatch "${options[@]}" "$GAVD6_ROOT/slurm/future-innovation/$script")"
  job="${job%%;*}"
  [[ "$job" =~ ^[0-9]+$ ]] || { echo "Invalid sbatch response: $job" >&2; exit 1; }
  printf '%s\t%s\t%s\n' "$label" "$job" "$dependency" >> "$FI_RUN_ROOT/logs/submissions.tsv"
  printf '%s\n' "$job"
}
teacher_dependency=''
if [[ "$phase" == prepare || "$phase" == all ]]; then
  first="$(submit cohort 01-build-gate-cohort.sbatch '')"
  second="$(submit poses 02-extract-poses.sbatch "afterok:$first")"
  echo "Submitted cohort=$first poses=$second."
  teacher_dependency="afterok:$second"
fi
if [[ "$phase" == compute || "$phase" == all ]]; then
  third="$(submit teacher 03-cache-vjepa-features.sbatch "$teacher_dependency")"
  fourth="$(submit audits 04-run-causal-and-target-audits.sbatch "afterok:$third")"
  fifth="$(submit fit 05-fit-gate-models.sbatch "afterok:$fourth")"
  sixth="$(submit report 06-build-gate-report.sbatch "afterany:$third:$fourth:$fifth")"
  echo "Submitted teacher=$third audits=$fourth fit=$fifth report=$sixth"
fi
