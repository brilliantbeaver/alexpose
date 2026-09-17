#!/usr/bin/env bash
# Fetch public MMPose configurations and the five released pilot checkpoints.
# Package installation, AMASS assets and COCO data remain separate setup steps.
set -euo pipefail
: "${GAVD6_ROOT:=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)}"
: "${ST_MODEL_ROOT:=$GAVD6_ROOT/models}"
mkdir -p "$ST_MODEL_ROOT/pose"
if [[ ! -d "$ST_MODEL_ROOT/mmpose" ]]; then
  git clone --depth 1 --branch v1.3.2 https://github.com/open-mmlab/mmpose.git "$ST_MODEL_ROOT/mmpose"
elif [[ ! -d "$ST_MODEL_ROOT/mmpose/.git" ]]; then
  echo "Existing model directory is not an MMPose checkout: $ST_MODEL_ROOT/mmpose" >&2
  exit 1
fi

download() {
  local name="$1" url="$2" target="$ST_MODEL_ROOT/pose/$1.pth"
  if [[ -f "$target" ]]; then
    echo "Keeping existing $target"
    return
  fi
  curl --fail --location "$url" --output "$target.part"
  mv "$target.part" "$target"
}

download rtmpose-m 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth'
download hrnet-w32 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'
download rtmpose-s 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-s_simcc-coco_pt-aic-coco_420e-256x192-8edcf0d7_20230127.pth'
download hrnet-w48 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'
download vitpose-base 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'
echo "Model assets: $ST_MODEL_ROOT"
echo "Set student config/checkpoint paths in ST_CONFIG to these locations."
