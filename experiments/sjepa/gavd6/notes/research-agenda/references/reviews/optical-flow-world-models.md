# Optical flow: world-model opportunities and novelty audit

Live primary-source review on 12 September 2026. No models were downloaded or run. A public file listing establishes an accessible release surface, not successful local loading, inference speed or accuracy on GAVD.

## Recommendation

Replace the constraint-conflict reserve with a carefully bounded **state-sufficiency study**: establish movement that the actual skeleton input cannot distinguish, then determine the smallest additional surface-motion observation that produces useful forecasting gains. This is closer to the repo's evidence and cheaper to falsify. The novelty is not combining flow and pose. A second, weaker option is a causal transport baseline that separates future-feature prediction from merely carrying old appearance forward.

The strongest additional inference from this reading is a potential failure of skeleton-guided flow: a constraint based on joint displacements may suppress real visible transport that leaves those joint positions unchanged. This is a hypothesis to test, not a result established by the cited papers.

## What flow does and does not measure

Optical flow estimates a two-dimensional correspondence field between images. Dense flow can retain surface motion between joints, including visible material movement during some axial or terminal rotations. It also contains camera motion, clothing motion, shadows, reflections, estimator error and occlusion failures. It is not 3D scene flow, muscle activation, force, contact pressure or a unique body rotation estimate.

An image patch with no useful texture may offer little information about tangential motion. A hidden surface has no direct correspondence in the next frame. Different 3D motions can project to the same flow. Learned confidence and forward-backward consistency are useful filters, not independent truth. Long-range tracking through occlusion makes predictions based on a prior; it does not turn invisible pixels into new measurements.

## Primary literature that defines the novelty boundary

| Work read | What already exists | Consequence |
| --- | --- | --- |
| [MC-JEPA, 2023](https://arxiv.org/pdf/2307.12698) | A shared encoder learns content through VICReg and motion through PWC-style flow estimation, warped-feature regression, reconstruction and cycle losses. | A joint flow/JEPA objective or feature warping alone is not novel. |
| [Pose from Flow and Flow from Pose, CVPR 2013](https://openaccess.thecvf.com/content_cvpr_2013/papers/Fragkiadaki_Pose_from_Flow_2013_CVPR_paper.pdf) | Body-part segmentation, pose and articulated flow improve each other, including unusual poses and fast limb movement. | Pose-guided flow and flow-guided pose are established. |
| [Optical Flow-based 3D Human Motion Estimation, 2017](https://arxiv.org/abs/1703.00177) | Fit human shape and motion by comparing computed flow with an artificial flow renderer. | Flow-based inverse rendering is a mandatory classical baseline, not the contribution. |
| [H-MoRe, 2025](https://arxiv.org/html/2504.10676v1) | Human-centric flow uses skeleton and boundary constraints, plus subject-relative flow. It evaluates gait recognition, action recognition and video generation. | Generic flow fusion, camera/body-relative flow and skeleton-conditioned human-motion representation collide directly. |
| [FOFPred, 2026](https://arxiv.org/html/2601.10781v1) | Forecasts dense future flow with a VLM-diffusion architecture and applies it to control and generation. | Predicting future flow as a world-model intermediate is already demonstrated. |
| [JOPAT, 2026](https://arxiv.org/html/2605.23856v1) | Joint diffusion of visual latents, tracks with visibility and actions. Its paper reports four H200s for about five days of action-free pretraining, then one H200 for a day of task adaptation. | Joint pixel/track prediction and occlusion-aware world models are crowded and not cheap to reproduce. |
| [Track4Action, August 2026](https://arxiv.org/html/2608.03727v1) | Distills a frozen 3D tracker's realized transition features into a current-observation policy; the tracker disappears at deployment. | Tracker-feature distillation alone is not new. |
| [Flow Equivariant World Models, ICML 2026](https://arxiv.org/abs/2601.01075) | Structured latent memory follows self-motion and object-motion flows, including long rollouts under partial observation. | A flow-transported memory or equivariance story needs a much narrower contribution. Mathematical flows here are not synonymous with estimated optical flow. |
| [HTD-Refine, CVPR 2026](https://arxiv.org/html/2605.26879v1) | PVA-Net estimates joint positions, velocities and accelerations from video; optimization refines global trajectories and targets both jitter and oversmoothing. | Proposal 1 must compare this kind of high-order refinement. Optical flow was not found in its method text, so do not call it a flow-based method. |

H-MoRe's method explicitly aligns body-point flow direction and magnitude with nearby skeletal offsets. This creates a concrete question: when a material surface moves but those offsets do not, does the guidance remove useful evidence? A failure is not guaranteed because other image and boundary terms can counteract the constraint. Audit the implemented objective and compare untouched flow with its guided version before claiming suppression.

The parent team also identified GaitMDF, DOI `10.1016/j.patcog.2026.113147`. The DOI did not resolve through this review's web fetch, so its detailed method and release status remain unverified here. Do not imply it was fully read.

## Public checkpoint and execution audit

| Model | Exact release evidence | Status and use |
| --- | --- | --- |
| RAFT | [Official TorchVision source](https://github.com/pytorch/vision/blob/main/torchvision/models/optical_flow/raft.py) defines `Raft_Small_Weights.C_T_V2` and the direct `raft_small_C_T_V2-01064c6d.pth` download URL. | Official public loader and binary URL verified, bytes not downloaded. Small robust fallback; no new flow training. |
| SEA-RAFT | [Official repo](https://github.com/princeton-vl/SEA-RAFT) explicitly loads `MemorySlices/Tartan-C-T-TSKH-spring540x960-M`. Its [actual model.safetensors page](https://huggingface.co/MemorySlices/Tartan-C-T-TSKH-spring540x960-M/blob/main/model.safetensors) lists 78.8 MB and SHA256 `cb8cfbf14c5e0f6734b64add383708b7ff68cc6089a0007c67165d4761346102`. | Strongest directly verified initial dependency. BSD-3-Clause release surface. First local smoke test still required. |
| WAFT | [Official repo](https://github.com/princeton-vl/WAFT) recommends an a1 adaptation checkpoint, whose [Drive page](https://drive.google.com/file/d/1CxzBQx0iSg6AyIgt6MF0ROlF_cAeZLPC/view?usp=drive_link) identifies `tar-c-t.pth`. | Public author download link verified, binary transfer untested. Optional second estimator after SEA-RAFT. Repo uses Python 3.12, PyTorch 2.7, CUDA 12.8 and xformers; do not confuse setup friction with model training. |
| CoTracker3 | [Official repo](https://github.com/facebookresearch/co-tracker) publishes `https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth` and `scaled_offline.pth`, with inference code. | Public binary URLs verified, bytes not downloaded. Use only prefix frames for forecasting inputs, even with an offline tracker. |
| AllTracker | [Official project](https://alltracker.github.io/) links [released code](https://github.com/aharley/alltracker), estimating dense query-frame correspondences to many frames. | Code/release claim verified; exact checkpoint binary not inspected in this bounded review. Optional tracking comparison, not a required dependency. |
| FOFPred | [Official project](https://fofpred.github.io/) links [Salesforce/FOFPred](https://huggingface.co/Salesforce/FOFPred). | Public model page verified; exact binary completeness and loading not checked. Not needed for either first experiment. |
| H-MoRe | Paper links `https://github.com/haku-huang/h-more`. | This exact official link returned 404 during review. Do not claim a usable checkpoint. Reproduce the published skeleton/boundary strategy as an explicitly labeled implementation if no alternative author release can be verified. |
| MC-JEPA | Paper and architecture inspected. | No official downloadable checkpoint was verified in this bounded search. Citation and method baseline, not a required pretrained dependency. |
| HTD-Refine | Current [official repository](https://github.com/ant-research/HTD-Refine) releases code and links `pvanet.pt` through its checkpoint table. | Code is released, not merely promised. This review verified the README link but not the Drive binary. Demo expects 30 FPS, camera intrinsics, initial GVHMR/TRAM motion and compatible licensed body assets. Full-sequence refinement must be restricted to the observed prefix when used in forecasting comparisons. |

SEA-RAFT's [paper](https://arxiv.org/abs/2405.14793) introduces a mixture-of-Laplace loss and rigid-motion pretraining. WAFT's [paper](https://arxiv.org/html/2506.21526v2) replaces cost volumes with high-resolution feature warping. Neither paper establishes superiority on GAVD. Compare an estimator swap on held-out examples and measure actual throughput before allocating the week.

## Candidate A: the smallest surface-motion state a skeleton is missing

**Question.** Under which observation conditions do exact joint-position histories fail to determine useful movement state, and can at most 16 added flow or track tokens close most of the resulting forecast gap?

Begin from AMASS whole-body rotations and the exact joint-extraction function used by S-JEPA. Construct or search for paired motions whose input joint histories match to a declared tolerance but whose projected material transport differs. Terminal rotation and compensated axial rotation are candidate constructions, not assumptions. Verify equality after skinning, regressed-joint extraction, normalization and time sampling; some supposedly invisible twists move the actual model's joints.

Use these pairs to establish the limitation, then test natural held-out AMASS motion separately. Render paired motions with fixed shape, camera, texture, lighting, duration and background. Vary texture and view only in held-out stress tests. Record exact surface correspondence from the renderer. Optical flow estimates are observations; renderer motion is the reference. Do not mistake cloth flutter or lighting changes for skeletal state.

Freeze SEA-RAFT and the existing S-JEPA backbone. Add a small residual state carrying selected surface tracks or local transport features, with budgets of 0, 4, 8 and 16 tokens. Select locations from past observations only. Compare uniform surface samples, fixed anatomical samples, uncertainty sampling and learned selection. Train a small forecasting head or adapter, not a new large backbone. The proposed mechanism preserves evidence in directions that joint motion fails to constrain, rather than penalizing all deviations from a pose-derived flow field.

The headline must combine a measured state-sufficiency gap, a budget curve and genuine future-state benefit. Use held-out surface trajectories and rotation errors, then joint forecasts where the extra state is actually relevant. Synthetic hidden-state classification alone is insufficient. Some terminal rotations need not affect future joint centers, so do not manufacture that claim. Include natural unedited motion and a held-out perturbation family; otherwise the method may only decode the edit generator.

Mandatory comparisons are raw joint history, full SMPL rotations as an information ceiling, full RGB/video features, direct flow-to-state prediction, flow-based mesh fitting, H-MoRe-style guidance, fixed versus selected tokens, and random-encoder controls. Give each visual method the same past frames. A full-rotation ceiling is not a deployable observation baseline. Full-frame flow extraction cost remains even when few tokens reach S-JEPA, so claim downstream representation efficiency separately from total compute.

**48-hour gate.** Within at most 24 H100-hours, obtain at least 100 verified matched pairs and show that estimated, not only oracle, transport resolves a useful fraction under two textures and two cameras. Require a forecast benefit beyond an equal-budget direct flow head on an unedited grouped-motion pilot. Stop or narrow the claim if equality fails, natural transfer vanishes, the signal is just silhouette area/flow energy, or full RGB mesh fitting makes the proposed compact state unnecessary. A week cap of 192 H100-hours is reasonable only after measured throughput and this gate: 24 pilot, 48 extraction/render comparisons, 72 adapters/baselines, 48 held conditions and seeds.

**Assessment.** This is a better replacement for the old Proposal 7 than a new generic fusion method. Novelty remains conditional. The plausible paper is about what a predictive state must preserve, demonstrated in a controlled setting and repaired with a small measurement budget. It is not a claim that optical flow uniquely recovers 3D gait or that changing the skeleton representation is itself new.

## Candidate B: distinguish a world model from transported appearance

**Question.** How much apparent future-feature prediction survives a baseline that transports past features using motion predicted only from the observed prefix, and does teaching only the remaining innovation improve actual S-JEPA forecasting?

Use frozen V-JEPA or frame-level vision features and frozen SEA-RAFT on observed past frames. Build three references: unchanged last features, constant-velocity transport, and a small flow-history predictor followed by feature transport. The true future flow is allowed only as an explicitly labeled oracle ceiling, never as a causal baseline or student input. Compare their future-feature scores with the existing teacher, including foreground-only and matched-background conditions.

If cheap causal transport explains the representation score, train the small S-JEPA adapter on the part of future features not predicted by that causal reference. Demand better future joint/surface prediction on changed cadence, turning and upper/lower-body coordination, rather than a larger residual target R-squared. Hold out camera motion and observation masks. Compare plain feature distillation, Jacobian/relative distillation, direct coordinate supervision and a transported-feature model with equal parameters and input visibility.

**48-hour gate.** At most 16 H100-hours should establish whether transport explains a substantial portion of the saved positive result and whether the remaining target has measurable student utility. Stop the method claim if direct coordinate or flow supervision matches it, or the decomposition is estimator-specific. Cap the optional week at 128 H100-hours.

**Assessment.** Lower priority than Candidate A. This is closely related to MC-JEPA warping, flow-equivariant memory and the repo's existing future-innovation work. Its defensible opening is a prospective, leakage-free accounting of predictive content plus a useful training decision. A transport audit alone is a valuable correction but unlikely to meet the requested conceptual bar.
