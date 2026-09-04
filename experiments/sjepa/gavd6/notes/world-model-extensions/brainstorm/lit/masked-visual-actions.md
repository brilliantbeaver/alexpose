# Masked Visual Actions for Unified World Modeling

## Problem

**Full-text fact.** Pretrained video models contain broad motion and interaction priors, but common action inputs are sparse, embodiment-specific, or misaligned with pixels. The paper seeks one visual interface for forward prediction and inverse action recovery across embodiments.

## Method in exactly five numbered sentences

**Full-text facts.**

1. A Masked Visual Action reveals the spatiotemporal pixels occupied by selected entities while leaving the rest of the video to be predicted.
2. The model receives this masked video plus the initial reference frame and learns conditional video completion.
3. Training combines about 1,000 DROID demonstrations processed through robot segmentation and calibrated translucent URDF rendering with 4,000 RoboCasa examples, including failures.
4. Revealing robot motion yields a forward scene-response query, while revealing desired object motion yields an inverse robot-motion query from the same checkpoint.
5. Only robot masks are used for training, so object-conditioned inverse prediction is a zero-shot behavior rather than a separately trained inverse mode.

## Headline numbers

**Full-text facts.** On DROID, MVA records LPIPS/SSIM/PSNR of 0.0945/0.887/23.74 versus Ctrl-World's 0.362/0.708/18.15; on unseen bimanual BEHAVIOR, it records 0.123/0.843/22.90 versus 0.196/0.837/18.39 (Table 1, Section 5.1). Best-of-10 planning adds 24, 26, and 21 percentage points on close-microwave, open-drawer, and open-dishwasher over the base policy, using 10 scenes per task (Figure 8, Section 5.2). Imagined versus simulator policy success correlates at $r=0.982$, with optimistic bias (Figure 9, Section 5.2). Zero-shot inverse video plus a separately trained inverse-dynamics model reaches 90 percent on CoffeeServeMug across 20 trials; that model and baselines use 100 demonstrations (Figure 11, Section 5.2).

## Base model and exact adaptation scale

**Full-text and artifact facts.** This did **not** train a world model from scratch. It applies two rank-256 LoRAs to the high-noise and low-noise DiT experts of `PAI/Wan2.2-Fun-A14B-Control`, split at timestep 0.358, using the unmodified DiffSynth-Studio trainer. The paper reports about 15 hours of data, 10,000 steps, batch 4, eight H200 GPUs, and four days. Each released BF16 LoRA is 2,453,763,192 bytes with 1,226,833,920 scalars across 800 tensors.

## What it enables here

**Repo inference.** Transfer the query design, not the 14B pixel generator: train one gait JEPA to predict object, center-of-mass, or contact futures from revealed human motion, and inverse human motion from a desired outcome. Dense person, limb, assistive-device, or object masks are a useful matched baseline against sparse Core11 joints under held-out bodies, devices, and views. Any force or clinical claim still needs measured kinetics and subject-disjoint tests.

## Limitations

**Full-text facts.** The authors explicitly state that the model learns interaction **correlations, not causal relationships**. It inherits the base model's speed and expressivity limits, overpredicts task progress, struggles with precise contact, can generate unnatural interactions and unseen-region artifacts, and needs camera calibration for rendered controls; segmentation can leak occluded scene content. **Inference.** No result establishes gait, human biomechanics, force, long-horizon stability, or intervention validity.

## Public access status

**Artifact facts checked 2026-09-03.** Apache-2.0 training and inference code is public. Two ungated LoRA checkpoints are public on Hugging Face, whose card declares MIT. A web gallery exposes 80 real/control/generated evaluation triplets, and the original DROID and RoboCasa sources are public, but no official download packages the selected 15-hour MVA corpus, processed masks/control videos, or labeled evaluation set; the repository requires a user-supplied CSV and says DROID URDF-rendering tools are coming soon.

## Sources

- Paper: https://arxiv.org/pdf/2607.19343 and https://arxiv.org/html/2607.19343v1
- Project: https://masked-visual-actions.github.io/
- Evaluation gallery: https://masked-visual-actions.github.io/assets/luey_gallery/index.html
- Code: https://github.com/HadiZayer/masked-visual-actions
- Checkpoints: https://huggingface.co/HadiZayer/masked-visual-actions/tree/main
- Base model: https://modelscope.cn/models/PAI/Wan2.2-Fun-A14B-Control
- Source data: https://droid-dataset.github.io/droid/the-droid-dataset and https://github.com/robocasa/robocasa/blob/main/docs/datasets/using_datasets.md
- Fetch date: 2026-09-03
