# Goal Force: Teaching Video Models To Accomplish Physics-Conditioned Goals

## Problem

**Full-text fact.** Text and images underspecify or overburden precise dynamic goals. Goal Force asks an image-to-video model to infer a plausible antecedent causal chain that realizes a user-specified target force.

## Method in exactly five sentences

1. Given a start image and text context, the user specifies the location, direction, and relative magnitude of a desired goal force.
2. A three-channel spatiotemporal tensor represents direct force and goal force as moving Gaussian blobs and optional mass as a static Gaussian blob.
3. Training uses 3,000 domino, 6,000 rolling-ball, and 3,000 carnation videos generated with Blender or PhysDreamer.
4. Randomly masking direct-force versus goal-force channels teaches both action-to-outcome simulation and goal-to-antecedent planning, while random mass masking permits privileged or inferred mass.
5. A ControlNet cloned from the first 10 high-noise-expert DiT blocks injects the physics signal through zero convolutions into a frozen Wan2.2 image-to-video backbone.

## Headline numbers

**Full-text facts.** In the 75-scene, 40-participant 2AFC study, Goal Force won goal-adherence preferences over zero-shot text in 70.5 to 74.5 percent of comparisons and over fine-tuned text in 56.8 to 67.0 percent across four categories (Table 1, Section 4.1). Filtered-valid planning accuracy spans 54.55 to 100 percent across 22 blocker scenes, versus an at-most 33.3 percent random baseline (Tables 2 and 5, Section 5.1). Diversity is 0.6577 over 26 seeds versus 0.3900 for a deterministic plan (Table 3, Section 5.2). Mass-conditioned speeds satisfy 4 of 4 in-distribution and 3 of 4 out-of-distribution inequalities, using 15 videos per mass combination (Figure 6, Section 5.3).

## What it enables here

**Repo inference.** Transfer the objective, not the pixel generator: give gait JEPA forward prompts such as an applied perturbation and inverse prompts such as desired contact, center-of-mass, or joint dynamics. A masked cause/effect curriculum could test whether latent rollouts recover antecedent adjustments, using force-plate or musculoskeletal targets only when measured. The released model is a qualitative video baseline, not validation of force from monocular Core11 skeletons.

## Limitations

**Full-text and project-page facts.** Force and mass are domain-relative, not calibrated in newtons or kilograms; clips are 81 frames at 16 FPS; training covers rigid collisions plus one flower; and planning accuracy discards degraded samples. The authors show spontaneous target motion and base-model melting or distortion despite goal completion. **Inference.** There is no gait, force-plate, long-horizon, closed-loop, or absolute-force validation, so tool-use videos show perceptual plausibility rather than measured mechanics.

## Base model and exact adaptation scale

**Paper and artifact facts.** The base is Wan2.2-I2V-A14B: two roughly 14B experts, 27B total and 14B active per denoising step. Only a 10-block ControlNet for the high-noise expert is trained for 3,000 steps at effective batch 4 on four 80GB A100s for under 48 hours; the base stays frozen. The released BF16 adapter is 7,552,980,728 bytes and its safetensors header contains 3,776,471,040 scalars across 292 tensors.

## Public access status

**Artifact facts checked 2026-09-03.** Code is public under MIT with training and inference scripts. The 12,000-video training set is public, ungated, MIT-licensed, and downloadable despite a broken viewer. The public ungated checkpoint has no model card or stated adapter license and requires the Apache-2.0 Wan base. The paper says benchmark data are released, but official repositories expose examples rather than a labeled complete benchmark download.

## Sources

- Full paper: https://arxiv.org/pdf/2601.05848
- Project: https://goal-force.github.io/
- Code: https://github.com/brown-palm/goal-force
- Checkpoint: https://huggingface.co/brown-palm/goal-force/blob/main/step-3000.safetensors
- Training data: https://huggingface.co/datasets/brown-palm/goal-force-training-datasets
- Base model: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B
- Fetch date: 2026-09-03
