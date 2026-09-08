#!/usr/bin/env bash
#SBATCH --job-name=fi-cohort
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=fi-cohort-%j.out
#SBATCH --error=fi-cohort-%j.err
#SBATCH --export=ALL
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
source "$GAVD6_ROOT/slurm/future-innovation/fi-common.sh"
: "${GAVD_FULL_ROOT:?Export GAVD_FULL_ROOT}"
: "${VJEPA2_ROOT:?Run 00-setup-environment.sh first}"
: "${FI_TEACHER_CHECKPOINT:?Set the V-JEPA 2.1 checkpoint path}"
: "${FI_POSE_MODEL:?Set the MediaPipe model path}"
: "${FI_CHANGE_REASON:?Describe this initial preregistration or version change}"
FI_ANNOTATION_ROOT="${FI_ANNOTATION_ROOT:-$GAVD_FULL_ROOT/annotations}"
# Match only the five official non-overlapping partitions; never mix 1.1/1.2 releases.
annotations=()
for part in 1 2 3 4 5; do
  path="$FI_ANNOTATION_ROOT/GAVD_Clinical_Annotations_${part}.csv"
  [[ -f "$path" ]] || { echo "Missing full-GAVD annotation partition: $path" >&2; exit 1; }
  annotations+=("$path")
done
if [[ ! -f "$FI_RUN_ROOT/config/run-contract.json" ]]; then
  fi_python init-run \
    --sequence-manifest "$GAVD_FULL_ROOT/manifests/gavd_full_sequences.csv" \
    --video-manifest "$GAVD_FULL_ROOT/manifests/gavd_full_videos.csv" \
    --annotations "${annotations[@]}" --youtube-dir "$GAVD_FULL_ROOT/youtube" \
    --pose-model "$FI_POSE_MODEL" --vjepa-root "$VJEPA2_ROOT" \
    --checkpoint "$FI_TEACHER_CHECKPOINT" --change-reason "$FI_CHANGE_REASON"
fi
fi_python build-cohort
