**Role**: You are a world models expert specializing in Joint Embedding Predictive Architecture (JEPA) and pose estimation.

**Task**: Based on the slurm guide document in `slurm/synthetic-training-v2/`, carefully and systematically revise the instructions to be much more clear and simple with as few steps as possible. Currently, it is too complicated and prone to errors with transferring files and setting up configurations, environment variables, etc.

For references, below are the environment variables currently in my HAIC environment that were used for the original synthetic training study:

---
export SJEPA_ROOT="/hai/scratch/$USER/alexpose/experiments/sjepa"
export GAVD6_ROOT="$SJEPA_ROOT/gavd6"
export VJEPA2_ROOT="/hai/scratch/$USER/vendor/vjepa2"
export AMASS_ROOT="$GAVD6_ROOT/data/amass"
export GAVD_FULL_ROOT="$GAVD6_ROOT/data/gavd_full"

export ST_PYTHON="/hai/scratch/$USER/envs/synthetic-training/bin/python"
export ST_RUN_ROOT="$GAVD6_ROOT/outputs/synthetic-training/pilot-01"
export ST_CONFIG="$ST_RUN_ROOT/config.json"
export ST_AMASS_ROOT="$AMASS_ROOT/extracted"
export ST_BODY_MODEL_ROOT="/hai/scratch/$USER/body_models"
export ST_GAVD_VIDEO_ROOT="$GAVD_FULL_ROOT/youtube/all"
export ST_GAVD_RESERVATION="$GAVD6_ROOT/outputs/future-innovation/learning-curve/config/source-reservation.csv"
export ST_MODEL_ROOT="/hai/scratch/$USER/models"

export ST_COCO_IMAGE_ROOT="/hai/scratch/$USER/coco/train2017"
export ST_COCO_ANNOTATIONS="/hai/scratch/$USER/coco/annotations/person_keypoints_train2017.json"
export ST_CONTEXT_REPO="${VJEPA2_ROOT:-$ST_MODEL_ROOT/vjepa2}"
export ST_CONTEXT_CHECKPOINT="$ST_MODEL_ROOT/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
export ST_IMAGE_CHECKPOINT="$ST_MODEL_ROOT/pose/resnet50-11ad3fa6.pth"

export ST_DMPL_ROOT="$ST_BODY_MODEL_ROOT/dmpls"
export ST_TEXTURE_DIR="$ST_MODEL_ROOT/synthetic-rendering/smplitex/textures"
export ST_UV_PATH="$ST_MODEL_ROOT/synthetic-rendering/smplitex/smpl_uv.obj"
export ST_BACKGROUND_DIR="$ST_MODEL_ROOT/synthetic-rendering/coco-backgrounds"
---

Use the same method and structure as the document in `slurm/synthetic-training` to brainstorm how to make the synthetic training v2 study as easy as possible for the user to run.

Additionally, ultrathink on how to efficiently and thoughtfully check the source code and scripts for the experiments to ensure that they run properly and have no scientific or technical error.

Use independent adversarial review to comprehensively review your changes. Use fan out subagents with dynamic workflows.
