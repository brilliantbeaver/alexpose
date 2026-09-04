# Adding Conditional Control to Text-to-Image Diffusion Models

## Problem

A large pretrained diffusion model carries useful visual knowledge, but direct fine-tuning on a small new conditioning dataset can erase that knowledge or overfit. ControlNet asks how to add spatial controls such as pose, depth, edges, and segmentation while preserving the original generator.

## Method in five sentences

1. ControlNet freezes the original Stable Diffusion network and clones its twelve encoder blocks plus its middle block into a trainable branch.
2. A small four-layer convolutional encoder maps each conditioning image into the generator's latent resolution.
3. Zero-initialized 1 by 1 convolutions inject the condition into the trainable branch and return its features to the frozen network.
4. The zero initialization makes the augmented model exactly equal to the frozen model at the first step, after which the residual control path learns through the standard diffusion noise-prediction loss.
5. Multiple trained ControlNets can be composed by adding their residual outputs without retraining the frozen generator.

## Headline numbers

The paper reports 23 percent more GPU memory and 34 percent more time per training iteration than ordinary Stable Diffusion fine-tuning on one A100 40 GB GPU. In a 1 to 5 user ranking, full ControlNet scored 4.22 for image quality and 4.28 for sketch fidelity, versus 3.93 and 4.09 for the lighter branch in Table 1. On semantic-map conditioning, Table 2 reports reconstructed intersection-over-union of 0.35, versus 0.32 for the lighter branch and 0.26 for PITI. A depth-conditioned model trained on 200,000 pairs for five days on one RTX 3090 Ti was difficult for users to distinguish from an industrial model trained on more than 12 million images and thousands of GPU-hours: user precision was 0.52 plus or minus 0.17. Figure 10 shows recognizable conditioning behavior with 1,000 training images, while the paper reports that control often appears before 10,000 optimization steps.

## What it makes possible here

The key pattern is adaptation without foundation-model training. A frozen video or motion predictor can retain its normal-motion knowledge while a zero-initialized branch learns gait-specific inputs such as 3D pose, depth uncertainty, contact state, assistive-device tracks, or simulator residuals. Exact equality at initialization also creates a clean scientific control: any changed gait score must come from the new branch. This architecture does not itself make the score physical or clinical. Those properties require independent measurements and shortcut controls.

## Limitations

ControlNet generates single images. It does not model temporal dynamics, forces, pathology, calibration, or causal interventions. Its main quantitative studies use perceptual and conditioning-fidelity measurements. Copying a large encoder branch is more expensive than low-rank adaptation. The reported small-data robustness does not guarantee that 1,874 correlated gait sequences can support a deep video control branch.

## Access status

Full arXiv HTML and the official repository were read on 2026-09-03. The paper, Apache-2.0 code, and pretrained pose, depth, edge, and segmentation ControlNet weights are public. The original models target Stable Diffusion 1.5 or 2.1 and cannot be inserted into Wan2.2 without a new implementation and new training.

## Sources

- Abstract and version record: https://arxiv.org/abs/2302.05543
- Full paper: https://arxiv.org/html/2302.05543v3
- Official code and checkpoints: https://github.com/lllyasviel/ControlNet
