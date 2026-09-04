# Checkpoint inventory

Fetch date: 2026-09-03. Access, sizes, parameters, and licenses below come only from official repositories or model cards. Roles are proposed uses. `Unknown` means the official source did not state it.

| Component | Exact checkpoint | Public, ungated, direct | Explicit license | Verified size or parameters | Proposed role | Blocking caveat |
|---|---|---|---|---|---|---|
| Wan2.2 I2V | [`Wan-AI/Wan2.2-I2V-A14B`](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) | Yes; HF `gated:false` | Apache-2.0 | 126.21 GB repo; 27B total, 14B active | RGB rollout base | Official single-GPU path needs at least 80 GB VRAM. |
| GoalForce | [`step-3000.safetensors`](https://huggingface.co/brown-palm/goal-force/blob/main/step-3000.safetensors) | Yes; HF `gated:false` | Weight unknown; code MIT | 7.553 GB; 3.776B BF16 scalars | Force-conditioned rollout | Relative force only; requires Wan I2V. |
| Masked Visual Actions | [`masked_world_lora_high.safetensors` + `masked_world_lora_low.safetensors`](https://huggingface.co/HadiZayer/masked-visual-actions/tree/main) | Yes; HF `gated:false` | MIT model card | 2 x 2.454 GB; 2 x 1.227B BF16 scalars | Masked action rollout | Not compatible with Wan I2V as released; requires `PAI/Wan2.2-Fun-A14B-Control`. |
| V-JEPA 2 video | [`vitl.pt`](https://dl.fbaipublicfiles.com/vjepa2/vitl.pt) | Yes; direct HTTP 200 | MIT-majority repo; weight terms not separate | 5.128 GB; 300M encoder | Video representation and prediction | No gait-specific action interface. |
| V-JEPA 2-AC | [`vjepa2-ac-vitg.pt`](https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt) | Yes; direct HTTP 200 | MIT-majority repo; weight terms not separate | 11.761 GB; 1B encoder; predictor unknown | Latent planning template | Post-trained for robot 7D state/action, not human gait controls. |
| WHAM | [`wham_vit_w_3dpw.pth.tar`](https://drive.google.com/uc?id=1i7kt9RlCCCNEW2aYaDWVr-G778JkLNcB&export=download&confirm=t) | Yes; direct HTTP 200 | Weight unknown; code MIT | 527.3 MB | Monocular world-grounded SMPL motion | Full pipeline requires registered SMPL and SMPLify assets. |
| GVHMR | [`gvhmr_siga24_release.ckpt`](https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD?usp=drive_link) | Public folder; direct file URL not stated | Custom: educational, research, nonprofit only | Unknown | Monocular world-grounded SMPL-X motion | Requires registered SMPL and SMPL-X assets; commercial use needs permission. |
| SAM 2.1 | [`sam2.1_hiera_large.pt`](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) | Yes; direct HTTP 200 | Apache-2.0; one dependency BSD-3-Clause | 898.1 MB; 224.4M | Prompted person/device tracks | Promptable masks are not semantic gait labels. |
| Depth Anything V2 | [`Depth-Anything-V2-Small-hf`](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf) | Yes; HF `gated:false` | Apache-2.0 | 99.17 MB; 24.785M | Monocular depth cue | Produces relative, not metric, depth. |
| Qwen VLM | [`Qwen3-VL-8B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Yes; HF `gated:false` | Apache-2.0 | 17.534 GB; 8.767B | Vision and tool orchestrator | Clinical gait validity is unknown. |
| GaitDynamics | [`GaitDynamicsDiffusion.pt`](https://github.com/stanfordnmbl/GaitDynamics/blob/main/example_usage/GaitDynamicsDiffusion.pt) + [`Refinement.pt`](https://github.com/stanfordnmbl/GaitDynamics/blob/main/example_usage/GaitDynamicsRefinement.pt) | Yes; Git LFS direct | MIT | 143.3 MB + 8.40 MB | Kinematic and GRF prior | Needs metric Rajagopal OpenSim inputs, not RGB. |
| GaitEncoder | [`saved_models_final/vae_final.pth`](https://github.com/rdmagruder/GaitEncoder/blob/master/saved_models_final/vae_final.pth) | Yes; GitHub direct | CC BY-NC 4.0 | 1.881 MB | Clinical gait latent and DMU | Needs 32-channel, 24-point Rajagopal strides; noncommercial. |
| Human Motion Diffusion Model | [`humanml-encoder-512-50steps`](https://drive.usercontent.google.com/download?id=1cfadR1eZ116TIdXK7qDX1RugAerEiJXr&export=download&confirm=t) | Yes; direct HTTP 200 | Weight unknown; code MIT | 206.5 MB zip | HumanML3D motion prior | CLIP, SMPL, SMPL-X, PyTorch3D, and datasets have separate licenses. |
| PP-Motion | [`pp-motion_pretrained/checkpoint_latest.pth`](https://drive.google.com/drive/folders/11s-eXweZZ23g9ZonJQVRul-r0fHZXRAu?usp=sharing) | Yes via official `gdown` folder script | Unknown | Unknown | Physical-perceptual motion score | Expects 60-frame, 25 x 3 SMPL motion; no repository license. |

## Operator confirmations

On 2026-09-03, the user confirmed that registered AMASS SMPL+H assets are available on HAIC. This resolves the base body-model requirement for AMASS conversion. WHAM and GVHMR still require a setup check for their exact SMPL, SMPL-X, and SMPLify variants. The official PP-Motion repository and checkpoint route are accessible, so C04 will use PP-Motion first. Its missing repository and weight license remains a release constraint rather than an access blocker.

## Official source index

https://github.com/Wan-Video/Wan2.2, https://github.com/brown-palm/goal-force, https://github.com/HadiZayer/masked-visual-actions, https://github.com/facebookresearch/vjepa2, https://github.com/yohanshin/WHAM, https://github.com/zju3dv/GVHMR, https://github.com/facebookresearch/sam2, https://github.com/DepthAnything/Depth-Anything-V2, https://github.com/QwenLM/Qwen3-VL, https://github.com/stanfordnmbl/GaitDynamics, https://github.com/rdmagruder/GaitEncoder, https://github.com/GuyTevet/motion-diffusion-model, https://github.com/sarahz024/PP-Motion
