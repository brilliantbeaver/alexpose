# Recent physical and human-motion scoring

All primary texts and release pages were fetched 2026-09-03.

## HumanScore

[Paper](https://arxiv.org/pdf/2604.20157) | [Project](https://cs.stanford.edu/~xtiange/projects/humanscore/)

**Full-text facts.** HumanScore addresses missing biomechanical evaluation of generated human video. It applies six hand-designed scores for extra limbs, bone-length stability, joint range, self-collision, velocity extremes, and smoothness, using pretrained HADM and pose/mesh recovery plus OpenSim fitting. HumanScore trains no evaluator, and reports no training compute. Across 102 prompts and 13 generators, Seedance 1.0 Pro Fast and HunyuanVideo 1.5 score 91.1 overall versus 94.3 for real videos (Table 1); about 1,200 pairwise responses give rank correlations close to 1.0 (Figure 7). **Gait inference.** Its segment stability, range, and temporal metrics form a useful gait-quality rubric. **Limits/access.** Monocular reconstruction is occlusion-, blur-, and depth-sensitive (Sections 4.3, 6.2). The project page's Code label is unlinked; no public data or checkpoint was found.

## PhyMotion

[Paper](https://arxiv.org/pdf/2605.14269) | [Project](https://phy-motion.github.io/) | [Code](https://github.com/h6kplus/PhyMotion) | [LoRA](https://huggingface.co/6kplus/PhyMotion-CausalForcing-1.3B) | [Prompts](https://huggingface.co/datasets/6kplus/PhyMotion-MotionX-Prompts)

**Full-text facts.** PhyMotion counters 2D rewards that miss body contact and dynamics. It uses pretrained GVHMR, retargets recovered SMPL motion to MuJoCo, and deterministically scores kinematics, contact/balance, and inverse-dynamics proxies. Thus the evaluator itself is not trained. Separately, rank-256 LoRA RL adapts Causal Forcing-1.3B and FastWan-1.3B on 21,348 prompts for 330 steps at 480x832x45, AdamW 1e-5, BF16, using 8 A100-80GB GPUs for 60.9 and 66 hours (Table 10). Correlation is 0.376 versus 0.262 best baseline (Table 1); feasibility rises 3.5% and 7.0% (Table 2). **Gait inference.** Contact, support, torque, and GRF proxies can judge or reward gait rollouts. **Limits/access.** [Inference] Scores inherit monocular recovery and simplified body-model errors. Code, prompts, and only the Causal Forcing LoRA are public.

## PP-Motion

[Paper](https://arxiv.org/pdf/2508.08179) | [Code](https://github.com/sarahz024/PP-Motion) | [Annotations](https://drive.google.com/drive/folders/1rWq5GJEE_Cnkh3hvUZ51EBUyp3aaS06P) | [Motions](https://drive.google.com/drive/folders/1A8x4o_xJxsVTVETJ2VspEjmg0wVtW29F) | [Checkpoint](https://drive.google.com/drive/folders/11s-eXweZZ23g9ZonJQVRul-r0fHZXRAu)

**Full-text facts.** PP-Motion bridges coarse human preference and physical feasibility for 3D motion, not RGB video. PHC plus IsaacGym first supplies continuous minimum-correction labels; the evaluator is then trained, rather than merely using a pretrained base: a 3-layer, 8-head DSTformer and 1024-channel MLP learn from 46,761 MDM pairs for 200 epochs, batch 64, initial LR 4e-5 with 0.995 decay, and correlation weight 0.3 (Appendix A). Hardware is unreported. It reaches MDM/FLAME PLCC 0.727/0.657 and preference accuracy 85.18%/68.82% (Table 3). **Gait inference.** It offers a learnable scalar target after gait-to-SMPL conversion. **Limits/access.** [Inference] MDM-only training and simulator-derived labels constrain domain transfer. Code, annotations, motions, and evaluator checkpoint are public via the repository's live Drive links.

## PhyWorldBench

[Paper](https://arxiv.org/pdf/2507.13428) | [Code](https://github.com/ashwin-333/phy-world-bench) | [Data](https://huggingface.co/datasets/phyworldbench/phyworldbench)

**Full-text facts.** PhyWorldBench tests physical realism across 1,050 prompts, ten categories, and 12,600 generated videos, including biomechanics and locomotion. Its SA/PC standards are binary; CAP applies two-stage context-aware prompting to eight frames using pretrained MLLMs. Headline CAP is zero-shot, not evaluator training: GPT-o1 reaches ROC-AUC 80.3 SA and 75.1 PC (Table 2), while Pika's best human-rated joint success is only 0.262 (Table 3). An optional 80/20 fine-tuning experiment reports gains but no hardware, hyperparameters, or weights (Appendix D). **Gait inference.** Locomotion, balance, and anatomy prompts can broaden qualitative stress tests. **Limits/access.** The authors note niche-coverage gaps and aesthetic bias (Section I). Code, prompts, standards, and over 10,000 experiment videos are public; no trained evaluator checkpoint is released.
