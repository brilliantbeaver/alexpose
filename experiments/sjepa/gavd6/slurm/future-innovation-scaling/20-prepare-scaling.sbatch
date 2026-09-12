#!/usr/bin/env bash
#SBATCH --job-name=fi-scale-pose
#SBATCH --account=mind
#SBATCH --partition=hai
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
set -euo pipefail
: "${GAVD6_ROOT:?}" "${FI_SCALING_ROOT:?}" "${FI_PARENT_ROOT:?}"
: "${FI_ANNOTATION_ROOT:?}" "${FI_VIDEO_ROOT:?}" "${FI_POSE_MODEL:?}"
: "${VJEPA2_ROOT:?}" "${FI_TEACHER_CHECKPOINT:?}"
cd "$GAVD6_ROOT"
"${FI_PYTHON:-.venv/bin/python}" scripts/research_directions/future_innovation/run_source_learning_curve.py prepare \
  --run-root "$FI_SCALING_ROOT" --parent-root "$FI_PARENT_ROOT" \
  --annotations "$FI_ANNOTATION_ROOT"/GAVD_Clinical_Annotations_{1,2,3,4,5}.csv \
  --video-root "$FI_VIDEO_ROOT" --pose-model "$FI_POSE_MODEL" \
  --teacher-root "$VJEPA2_ROOT" --checkpoint "$FI_TEACHER_CHECKPOINT"
