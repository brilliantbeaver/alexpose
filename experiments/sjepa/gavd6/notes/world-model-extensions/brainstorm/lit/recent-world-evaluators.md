# Recent video-world-model evaluators

## SC3-Eval

**Full-text facts.** **Problem:** closed-loop robot-policy evaluation is costly, while video simulators drift, disagree across views, and face policy shift. **Method:** SC3-Eval adapts pretrained Cosmos3-Nano, injecting normalized 7D delta-EE actions through an MLP into every DiT block, and jointly trains shared forward-dynamics, inverse-dynamics, and cross-view-inpainting modes. At inference it retains 16 of 24 predicted frames and stops when commanded versus inverse-recovered action error exceeds 0.02. Training is 24,000 steps, batch 512, on 32 GB200s for 2.2 days; the paper does not report parameters or frozen fraction. **Headline:** across seven pi_0.5 checkpoints, closed-loop Pearson/MMRV is 0.929/0.119 (Table 2, Section 4.4). **Limitations:** 381 hours but one scene, 12 objects, at most 20-second trials; inference is 2.3 seconds per 24-frame chunk and uncertainty is uncalibrated.

**Repo inference.** For gait judgment, treat joint trajectories as actions, demand forward/inverse agreement, enforce synchronized-view consistency, and abstain on drifting imagined trials rather than score them clinically.

**Artifact check.** The official page links only the paper. No SC3 code, training data, or checkpoint is public there.

## GigaWorld-1

**Full-text facts.** **Problem:** determine what makes a video world model agree with real robot-policy rankings. **Method/base/compute:** GigaWorld-1 adapts pretrained Wan 2.1 I2V 1.3B (Nano) and Wan 2.2 TI2V 5B (Plus) with pixel-aligned controls, anchored hierarchical memory, Relative RoPE, and LoRA. Its 12,980-hour corpus feeds 13,000-step rank-128 foundation adaptation then 36,000-step rank-256 autoregressive adaptation, each batch 32 on 32 H20s; optional ODE is 3,759 steps and required DMD2 distillation 2,250 steps (Tables 6-8). **Headline:** WMBench covers 7 models, 4 action encodings, 8 tasks, and 324,000 rollouts; Plus scores 0.6834, 11.6% over Cosmos-Predict2.5 and 14.9% over Wan 2.2 (Table 9, Section 6.5.1). **Limitations:** eight task families, video-centric scope, VLM labels still need human checks, and contact failures remain optimistically scored.

**Repo inference.** Port WMBench's matched real/generated outcome agreement, checkpoint rank correlation, geometry and JEPA metrics, and explicit penalty for static gait predictions.

**Artifact check.** Apache-2.0 code and Stage-1 Nano/Pro full and LoRA weights are public. Distilled weights are pending; public data are a 1.95 GB toy set, while the 31.5 GB challenge set is gated and WMBench is partial. The 12,980-hour corpus is not released.

## Scalable Policy Evaluation with Video World Models

**Full-text facts.** **Problem/method:** replace repeated robot trials with autoregressive video rollouts plus Gemini-2.5-Pro success labels. The model adapts pretrained Cosmos-Predict2-Video2World-2B end-to-end, not from scratch, using Fourier-mapped action MLP embeddings added to time embeddings. Synthetic/Bridge training uses 240,000/200,000 steps, batch 128, learning rate 4e-4, and 32 H100s; RoboMimic grows from about 0.3M to 1.1M transitions with policy rollouts (Appendix VIII). **Headline:** Pearson/MMRV ranges from 0.879/0.015 on Lift to 0.833/0.217 on Tool Hang (Table I); real Bridge evaluation is 0.687/0.170 versus IRASim 0.610/0.213 (Appendix Table VI). **Limitations:** hallucination, view disagreement, static copying, accumulated error, and only 0.66-0.79 VLM accuracy (Appendix Table V).

**Repo inference.** Include abnormal and failed gait trajectories during adaptation, rank against blinded clinical outcomes, and retain human adjudication for long or ambiguous sequences.

**Artifact check.** The paper says it plans release. The official site currently provides videos and an appendix, but no implementation, evaluator data package, or adapted checkpoint.

## Sources

- SC3: https://arxiv.org/pdf/2606.18610 ; https://weichengtseng.github.io/sc3-eval/
- GigaWorld-1: https://arxiv.org/pdf/2607.02642 ; https://open-gigaai.github.io/giga-world-1/ ; https://github.com/open-gigaai/giga-world-1 ; https://huggingface.co/open-gigaai/Giga-World-1 ; https://huggingface.co/datasets/open-gigaai/Giga-World-1-Toydata ; https://huggingface.co/datasets/open-gigaai/CVPR-2026-WorldModel-Track-Dataset
- Scalable evaluation: https://arxiv.org/pdf/2511.11520 ; https://miscsubmission.github.io/world_model_policy_eval/ ; https://raw.githubusercontent.com/MiscSubmission/world_model_policy_eval/main/docs/appendix.pdf
- Fetch date: 2026-09-03
