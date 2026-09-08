#!/usr/bin/env bash
# Run once on HAIC before submitting. Downloads only public model/code assets.
set -euo pipefail
: "${GAVD6_ROOT:?Export GAVD6_ROOT}"
: "${VJEPA2_ROOT:?Export VJEPA2_ROOT outside the checkout}"
: "${FI_TEACHER_CHECKPOINT:?Export the absolute V-JEPA 2.1 checkpoint path}"
: "${FI_POSE_MODEL:?Export the absolute pose_landmarker_lite.task path}"
readonly FI_REVIEWED_COMMIT=204698b45b3712590f06245fbfba32d3be539812
if [[ ! -e "$VJEPA2_ROOT" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/facebookresearch/vjepa2.git "$VJEPA2_ROOT"
  git -C "$VJEPA2_ROOT" checkout --detach "$FI_REVIEWED_COMMIT"
fi
[[ "$(git -C "$VJEPA2_ROOT" rev-parse HEAD)" == "$FI_REVIEWED_COMMIT" ]] || {
  echo "Existing VJEPA2_ROOT is not at reviewed commit $FI_REVIEWED_COMMIT" >&2; exit 1;
}
[[ -z "$(git -C "$VJEPA2_ROOT" status --porcelain)" ]] || { echo "Teacher checkout is dirty" >&2; exit 1; }
mkdir -p "$(dirname "$FI_TEACHER_CHECKPOINT")" "$(dirname "$FI_POSE_MODEL")"
if [[ ! -s "$FI_TEACHER_CHECKPOINT" ]]; then
  curl --fail --location --retry 3 \
    https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt \
    --output "$FI_TEACHER_CHECKPOINT.partial"
  mv "$FI_TEACHER_CHECKPOINT.partial" "$FI_TEACHER_CHECKPOINT"
fi
if [[ ! -s "$FI_POSE_MODEL" ]]; then
  curl --fail --location --retry 3 \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task \
    --output "$FI_POSE_MODEL.partial"
  mv "$FI_POSE_MODEL.partial" "$FI_POSE_MODEL"
fi
cd "$GAVD6_ROOT"
export UV_PROJECT_ENVIRONMENT="${FI_ENVIRONMENT:-$GAVD6_ROOT/.venv}"
uv sync --frozen --extra future-innovation
uv run --no-sync python -c 'import cv2, einops, mediapipe, timm, torch, torchvision; assert torch.__version__ == "2.6.0+cu124"; print("Future Innovation environment ready")'
uv run --no-sync python -m unittest discover -s tests -p 'test_future_innovation_*.py'
