# World-model review after notebook 06

Reviewed 14 September 2026. This memo contains literature findings and proposed experiments, not new GPU results. It revises the earlier portfolio after the MoMask reconstruction failure. Public release links were inspected; no weights were downloaded or executed in this review.

## What the inspiration papers justify

- **Goal Force** changes the conditioning interface: one channel specifies an applied force, another specifies a desired later force, and a third can supply relative mass. It trains a ControlNet branch copied from ten Wan2.2 layers for 3,000 steps, reporting under 48 hours on four A100 80 GB GPUs. Its physical quantities are relative, not calibrated clinical measurements. Some planning analyses exclude visually degraded generations. These results support trying a small, well-aligned interface, not assuming that a video generator is a verified human biomechanics simulator. [Paper](https://arxiv.org/html/2601.05848v1), [official code](https://github.com/brown-palm/goal-force).
- **Masked Visual Actions** reveals part of a motion video to condition completion of the rest. Robot motion supports forward prediction; desired object motion supports inverse behavior. The often quoted 15 hours is the dataset size. The reported training uses rank-256 LoRA, eight H200s, and four days. The paper explicitly limits its interpretation to learned interaction correlations rather than established causal relationships. A gait adaptation needs its own outcome tests. [Paper](https://arxiv.org/html/2607.19343v1).
- **ControlNet** freezes a pretrained generator and adds trainable conditioning through initially zero connections. This is a useful implementation pattern with a base-model endpoint, but applying it to skeletons is not sufficient novelty. [Paper](https://arxiv.org/abs/2302.05543), [official code](https://github.com/lllyasviel/ControlNet).
- **V-JEPA 2.1** explicitly improves dense spatial and temporal features through dense prediction and supervision at multiple layers. That makes frozen local tokens a more defensible starting point than repeating the previous experiment with global pooled targets. It does not prove that those tokens contain clinically useful movement or that the native predictor is a human action-conditioned simulator. [Paper](https://arxiv.org/abs/2603.14482).

## Models and access

| Resource | What is established | Consequence |
| --- | --- | --- |
| V-JEPA 2.1 | Official repository lists B, L, g, and G checkpoints with 80M, 300M, 1B, and 2B encoders and public download links. | Start with already downloaded local B/L encoder features. Use larger models only after a measurable gain exists. |
| V-JEPA 2-AC | Official model is post-trained on robot interaction trajectories. | Do not describe it as a downloadable human-action predictor. |
| Distilled V-JEPA 2.1 B/L predictor | Official loader uses `ema_encoder` and `teacher_embed_dim=1664`. | Encoder features and predictor targets are not automatically the same space. Do not compute their raw difference as prediction error. |
| Goal Force | A public 7.55 GB `step-3000.safetensors` file and author download instructions are visible. | Available as an optional comparator, not a necessary gait dependency. |
| Masked Visual Actions | Official code links two LoRAs, a downloader, and the Wan-Fun-Control base. It says URDF rendering tools are forthcoming. | Availability of an adapter is not a complete reproduction pipeline. The one-week proposal should not depend on reproducing its robot dataset. |
| Human-JEPA and GOT-JEPA | Papers were readable; a working public binary path was not confirmed in this review. | Treat as related work, not a critical executable dependency. |

Sources: [V-JEPA official repository](https://github.com/facebookresearch/vjepa2), [V-JEPA backbone loader](https://raw.githubusercontent.com/facebookresearch/vjepa2/main/src/hub/backbones.py), [Goal Force binary page](https://huggingface.co/brown-palm/goal-force/blob/main/step-3000.safetensors), [MVA official implementation](https://github.com/HadiZayer/masked-visual-actions).

The fetched V-JEPA loader currently contains a localhost base URL even though the README supplies public links. This is an access detail to check against the user's local code, not a scientific failure or a reason to download a new model. The MVA repository currently identifies its code license as Apache-2.0. Earlier notes describing a model-card license should not substitute for the current code license.

## Novelty boundaries that change the shortlist

- **Human-JEPA** already adapts human-centric perception to forecasting while preserving dense representations through frozen anchors. A generic human JEPA with adapters or a dense-preservation loss is occupied. [Paper](https://arxiv.org/html/2608.21160v1).
- **GOT-JEPA** predicts tracking models from degraded observations using clean-observation teacher targets, with point-centric occlusion reasoning. A generic predictive tracking repair claim is also occupied. [Paper](https://arxiv.org/html/2602.14771v1).
- **MC-JEPA** jointly learns motion and content. **JOPAT** incorporates point tracking into world action models. Flow-plus-JEPA and track-plus-JEPA are not new by themselves. [MC-JEPA](https://arxiv.org/abs/2307.12698), [JOPAT](https://arxiv.org/abs/2605.23856).
- **ActiveMoCap** and **Pose-DRL** already choose observations in space or time to improve pose estimation. [ActiveMoCap](https://arxiv.org/abs/1912.08568), [Pose-DRL](https://arxiv.org/abs/2001.02024).
- **EDDI** chooses costly observations for information about designated target variables. **GSM-AFA** learns a generative surrogate for active feature acquisition. **REACT** jointly selects context and temporal measurements. Merely replacing uncertainty with downstream utility is not a new principle. [EDDI](https://arxiv.org/abs/1809.11142), [GSM-AFA](https://arxiv.org/abs/2010.02433), [REACT](https://arxiv.org/abs/2603.11370).
- **JEDi** uses JEPA features for video-quality evaluation. **WMReward** uses latent world models to score physical video consistency. A new latent plausibility score needs a more specific contribution than preferring good gait videos. [JEDi](https://arxiv.org/abs/2410.05203), [WMReward](https://arxiv.org/abs/2601.10553).
- **UWM-JEPA** already studies uncertainty carried through latent rollouts and explicitly separates encoding from predictor behavior. A vague claim that deterministic latent prediction loses uncertainty is insufficient. [Paper](https://arxiv.org/abs/2605.25313).

## Candidate A: transfer the value of an observation

**Question:** Can a small head on frozen video features learn which additional part of an already recorded motion prefix will improve a forecast, and reuse that selection policy with an unseen forecaster or future-motion readout?

**Concrete example:** The model has a coarse view and noisy joint tracks. It can inspect two high-resolution limb clips from the observed second. An uncertain arm may be irrelevant to the requested foot forecast; a moderately uncertain pelvis observation may settle an imminent turn. The method should choose the observation that improves the eventual answer, rather than the most uncertain joint by default.

**Method:** Define a small menu of prefix-only queries: higher-resolution body-part clips, denser frame samples, or a local flow/track refinement. Train a lightweight head on the actual reduction in held-out future-motion loss after each query is revealed to a fixed forecasting model. Query training examples use training people. At inference, the selector sees only the initial coarse context, skeleton, and query identity. The hidden crop is processed only after selection. Frozen V-JEPA 2.1 supplies temporal context and selected-crop features. This does not use a fictional pretrained human-action predictor.

**Critical experiment:** Freeze the selector and evaluate it with a second forecasting architecture and an unseen movement family. A stronger optional test changes the requested future readout while conditioning the selector on that readout. The primary endpoint is actual AMASS future-motion error at equal query cost, not a decrease in the selector's own uncertainty. GAVD adds visible 2D future tracking with a small independently checked visible-frame subset. It cannot establish 3D movement truth.

**Mandatory baselines:** Uniform and stratified queries, query the target limb, largest pose covariance, motion energy, flow disagreement, a matched learned attention selector, target-information acquisition in the style of EDDI, and oracle realized query gain. Compute the initial observation's cost too. A global high-resolution encoder cannot secretly see the pixels being counted as unqueried.

**Forty-eight-hour test:** First measure the oracle query gain on natural motion over strong extrapolation. If revealing more past information does not improve future prediction, stop. Then require a learned selector to outperform the best inexpensive acquisition rule. If it works only with the forecaster used to manufacture its labels, retain an engineering optimization rather than claiming a transferable world-model capability.

**Novelty assessment:** Conditional. The opening is reusable observation value across tasks or models, backed by a real dissociation between pose-uncertainty reduction and useful future prediction. Active feature acquisition itself is mature. This is affordable, but should not receive an unconditional high novelty score.

## Candidate B: preserve the motion missing between joints

**Question:** On natural AMASS motion, how much useful future surface and segment-orientation information is absent from 22 joint-position histories, and can 4 to 16 observed motion tokens recover it across new bodies and cameras?

This sharpens the existing `motion-beyond-keypoints.md`; it is not an independent invention. Rotating a limb around its long axis can leave joint centers almost unchanged while moving the surface. The existing forward-kinematic alias construction is a diagnostic. The actual scientific requirement is a meaningful forecast gain on unedited recordings.

**Method:** Freeze SEA-RAFT or an available point tracker and V-JEPA 2.1. Build a small state from joints plus a few local transport histories or track-aligned dense video tokens. Predict future material-point displacement and declared segment rotations. Compare 0, 4, 8, and 16 tokens at fixed dimensions. Keep raw transport as well as any residual from a skeletal flow approximation, because subtraction can remove real motion.

**Critical experiment:** Match current posture, root motion, body size, and basic speed, then test held-person future surface motion. Compare against off-axis landmarks, full rotations as a labeled information ceiling, angular and flow-velocity extrapolation, raw flow concatenation, and same-capacity coordinate models. GAVD supplies real visible surface-track stress tests, not true bone twist or skin motion reference.

**Forty-eight-hour test:** Verify a natural-motion benefit before training a specialized adapter. If exact renderer transport helps but estimated transport does not, the current measurement path fails. If simple fusion matches the adapter, do not call the method novel. If the benefit vanishes on natural motion after controlling for extrapolation, the synthetic alias remains an information example, not a paper result.

**Novelty assessment:** Moderate and conditional on connecting the ambiguity to real forecast loss, a useful observation budget, and a method beyond fusion. This has stronger physical motivation than the current MoMask gate and avoids needing a good repair proposal. It has a harder practical surface-ground-truth limitation on GAVD.

## Candidate C: test whether latent similarity makes the right motion decision

**Question:** Can a small, frozen-backbone motion metric preserve the ordering of meaningful trajectory errors under changes in appearance, then transfer that ordering to unseen motion tasks and model families?

**Method:** Render pairs of real AMASS trajectories under matched nuisance conditions, and render the same trajectory under different nuisances. Candidate trajectories should be real recorded alternatives or declared bounded edits. Specify a motion question such as future foot clearance or arm-leg timing before fitting. Compare global JEPA distance, local track-indexed distance, flow/geometry distances, and a small question-conditioned metric head. Evaluate ranking regret against independent motion references. A useful head should improve decisions across held-out readouts, not merely reproduce its own training label.

The cleanest initial test is representation distance between observed clips, which needs only encoders. A separate native-predictor test must use the correct checkpoint target space and prefix-only masking. Predicted features are not a probability density; raw latent error is not calibrated likelihood.

**Critical experiment:** At equal nuisance changes, does a physically worse candidate consistently become more attractive in latent space? Does the learned correction remove those reversals while retaining recognition-relevant representation quality? Separate pose-fitting mismatch, texture change, temporal disturbance, and camera motion. On GAVD, evaluate only questions with visible reference annotations, not an inferred clinical condition.

**Forty-eight-hour test:** Freeze the ordering task, sample hundreds of independent source windows, and establish failures across two existing encoders. If flow or raw coordinate distance fully resolves the task whenever the required measurements are available, the model has not earned an additional role. A metric trained and tested on the same synthetic edit family is insufficient.

**Novelty assessment:** Lower than it first appears because JEDi, WMReward, and metric learning already cover much of the method. A broad result about decision-relevant motion orderings could matter, but a new gait-quality score alone is weak. This is a useful low-compute diagnostic and reserve, not my first flagship recommendation.

## Candidate D: explain uncertainty with a recoverable motion alternative

**Question:** Can a frozen JEPA representation help find a second movement explanation that changes a specified conclusion, then identify which additional recorded observation rejects that particular alternative?

This sharpens the existing `motion-ambiguity.md`. Keep independently checked articulated geometry as the admissibility rule. Use video features to propose or rank search initializations, rather than treating a generation prior's score as proof of human feasibility. This avoids reinstating MoMask as an authority after its failed repair audit.

**Method:** Search under observed 2D trajectories, camera bounds, joint limits, and an acceleration allowance. Return two candidate trajectories that fit the observation but straddle a predeclared movement threshold. Then select an additional past frame or body crop by candidate disagreement and test it on the held observation. A rejected candidate does not establish uniqueness because a third explanation may remain.

**Critical experiment:** Improvement in valid witness recovery per unit compute over multistart geometric optimization, plus reduction in a downstream incorrect conclusion after an actual additional observation. An answer-dependent query can connect this proposal to Candidate A, but it should not be counted as an independent flagship if it uses the same claim and evidence.

**Forty-eight-hour test:** If geometric search already finds essentially every relevant ambiguity, learned search has no meaningful headroom. If candidate differences are only reflection, arbitrary depth, or severe articulation violations, stop. A native 3D AMASS reference does not prove a single-view observation is unambiguous.

**Novelty assessment:** Stronger explanatory object than a scalar uncertainty score, but the central algorithm may collapse to geometry and known active sensing. The result must show useful finite-budget search and a changed decision, not simply plausible animations.

## Recommendation for the portfolio

Run the cheapest natural-motion headroom tests for A and B before choosing a new flagship. B has a physically explicit missing-state hypothesis; A has a clearer route to testing a reusable decision capability. Neither has demonstrated the needed natural-motion benefit yet. C is an inexpensive diagnostic that may expose why existing latent objectives fail. D is a useful alternate only if geometric baselines leave headroom.

The new portfolio should explicitly demote the current MoMask gate. It should also avoid manufacturing seven independent high-confidence claims from related versions of the same observation problem. A convincing result from one pilot is worth more than committing the week to seven adapted models.
