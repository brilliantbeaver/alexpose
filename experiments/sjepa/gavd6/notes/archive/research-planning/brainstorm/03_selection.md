# Phase 3 selection

## Scoring rule

All scores run from 1 to 5, with 5 best. Promise asks whether a true result would change how gait models are built or used. Feasibility asks whether the decisive result fits two weeks, eight H100s, public checkpoints, and the available data. Novelty uses only the verified literature set. Shortcut resistance scores both the intrinsic measurement and the strength of its falsification controls. Clinical meaning asks whether the output could support a cautious gait interpretation rather than only a benchmark number.

The operational high-promise, high-feasibility frontier requires Promise at least 4 and Feasibility at least 4. Six selections come from that set. C07 is the one deliberate exception: Promise 5 and Feasibility 3, with a cheap 72-hour track-and-shuffle kill test. Secondary scores break ties. The score is about the research question, not the polish of its implementation plan.

## Full scoring table

| ID | Candidate | Promise | Feasibility | Novelty | Shortcut resistance | Clinical meaning | Decision |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| C01 | Horizon-resolved latent error atlas | 4 | 5 | 4 | 4 | 4 | Reserve. Fast and necessary, but anatomical diffusion error is a close prior. |
| C02 | Forward-inverse motion cycle | 4 | 5 | 4 | 4 | 4 | Reserve. Strong control study; risk of reducing to smoothness is high. |
| C03 | Multi-lift minimum normalizing edit | 5 | 4 | 5 | 4 | 5 | **Select, rank 3.** The output is an explicit, falsifiable motion edit. |
| C04 | Physical feasibility versus healthy typicality | 5 | 4 | 5 | 5 | 5 | **Select, rank 2.** It separates perception failure, mechanics, and typicality. |
| C05 | Motion-conditioned depth residual | 4 | 4 | 4 | 5 | 4 | **Select, rank 6.** It attacks the central 2D-to-3D confound directly. |
| C06 | Video-skeleton predictive agreement | 4 | 4 | 4 | 4 | 3 | Reserve. Appearance leakage weakens its direct clinical meaning. |
| C07 | Body-device relational dynamics | 5 | 3 | 5 | 4 | 5 | **Select, rank 7.** High-risk exception with a cheap relational kill test. |
| C08 | Lift-consensus abstention | 4 | 3 | 3 | 5 | 4 | Reserve as a required safety layer; two lift stacks and variant integration add risk. |
| C09 | Cross-prior disagreement abstention | 4 | 4 | 4 | 5 | 5 | Reserve as a shared evaluation layer rather than a standalone proposal. |
| C10 | Missing-modality predictive set | 3 | 3 | 3 | 5 | 4 | Cut as a robustness module. It does not define a strong new gait measurement. |
| C11 | Simulator-teacher residual forecasting | 5 | 4 | 5 | 4 | 4 | **Select, rank 5.** It tests a distinct adaptation mechanism and a sharp null. |
| C12 | Generative JEPA posterior | 4 | 4 | 4 | 4 | 3 | Reserve. Multiple futures matter, but their clinical interpretation is indirect. |
| C13 | Intervention response fingerprint | 5 | 4 | 5 | 5 | 5 | **Select, rank 1.** It turns a prior into a dynamic perturbation instrument. |
| C14 | Conditional coordination graph | 5 | 5 | 5 | 4 | 5 | **Select, rank 4.** It is cheap, anatomical, and independent of pixel generation. |
| C15 | Normal-manifold tangent decomposition | 4 | 4 | 4 | 4 | 4 | Reserve. Strong geometry, but phase and Rajagopal conversion add ambiguity. |
| C16 | View-counterfactual score consistency | 3 | 3 | 3 | 3 | 3 | Cut. Pose-conditioned rendering artifacts can become the measurement. |
| C17 | Repair-set degeneracy | 4 | 4 | 4 | 4 | 4 | Reserve as the uncertainty ablation for C03, not a separate first proposal. |
| C18 | Phase-resolved transient localizer | 3 | 5 | 3 | 3 | 3 | Cut. GAVD lacks enough gait events to validate the localization claim. |
| C19 | Predictive-fingerprint discovery | 4 | 4 | 4 | 3 | 3 | Reserve. Stable unsupervised phenotypes are useful but hard to name clinically. |
| C20 | Few-shot taxonomy from predictive summaries | 3 | 5 | 2 | 4 | 3 | Cut. GaitForeMer, CARE-PD, and Aggregate, Don't Adapt occupy the claim. |
| C21 | Qwen corruption-aware tool policy | 4 | 3 | 4 | 4 | 3 | Cut from seven. It tests routing more than whether a world model judges gait. |
| C22 | Robot-trained masked-action transfer | 3 | 2 | 3 | 3 | 2 | Cut. Robot mask semantics and Wan inference cost dominate the gait question. |
| C23 | Robot action-latent transplant | 3 | 2 | 4 | 3 | 2 | Cut. The public latent is specialized to robot state and action. |
| C24 | Pose-conditioned visual counterfactual | 3 | 3 | 3 | 2 | 3 | Cut. It is useful visualization only if a stronger measurement already works. |
| C25 | Plain MDM surprise transfer | 2 | 4 | 1 | 3 | 3 | Cut. The central claim is already published. Keep it as a baseline. |
| C26 | Static GaitEncoder taxonomy probe | 2 | 4 | 1 | 3 | 2 | Cut. It is a representation probe, not a predictive instrument. |
| C27 | PP-Motion scalar-only gait judge | 2 | 4 | 1 | 2 | 2 | Cut. A scalar physics score cannot separate lift failure from atypical gait. |

## Conceptual recommendation

**C13, Intervention response fingerprint, is the conceptually strongest proposal.** Existing approaches ask whether an observed motion is likely, reconstructable, physically plausible, or close to a normal latent. C13 asks a different kind of question: how does a learned dynamics model respond locally when the same controlled change is applied around this particular walk? The resulting response operator records where an edit propagates, where it is absorbed, and how quickly the model returns toward its learned motion distribution. This turns the world model from a passive density judge into an active measurement instrument.

The conceptual leap is stronger than C04 because it creates a new object rather than combining two existing scores. It is stronger than C03 because it measures a local response surface rather than one optimizer-selected repair. It is stronger than C14 because it can reveal nonlinear, phase-specific coupling instead of only conditional predictability among body regions. A positive result would make one-number anomaly methods look incomplete. A negative result would establish that a pretrained motion generator's local sensitivity is governed by editing mechanics, lift choice, or its population prior rather than by the observed walk.

The closest verified collisions do not take this object. GaitDynamics uses conditioning to generate modified speed and estimate associated force changes. GaitEncoder predicts an intervention outcome in its latent space. GoalForce asks a video model to plan toward a specified physical outcome. C13 instead applies a fixed bank of small virtual perturbations around each observed walk and treats the resulting local sensitivity field as the measurement. Its contribution disappears if the study reports only edited samples, a scalar anomaly score, or label accuracy.

The causal boundary must remain explicit. C13 measures the model's conditional response around an observation. It does not measure how the person would respond to a real perturbation, and it does not identify neuromuscular control. The decisive evidence is therefore repeatability across lifts, perturbation sizes, and opposite edit directions, plus separation from speed, phase, and physical-feasibility effects. C04 should serve as a compulsory validity analysis inside the C13 study: response fingerprints must be interpreted jointly with PP-Motion feasibility and GaitDynamics typicality so that an impossible lift is not mistaken for unusual coordination.

C04 remains the easiest clean calibration experiment. Use the accessible public PP-Motion checkpoint as its feasibility implementation. It accepts a direct 60-frame SMPL motion input and avoids assembling the longer GVHMR, retargeting, and MuJoCo scoring stack required by a PhyMotion-style implementation. Retain deterministic PhyMotion-style scoring only as a later sensitivity analysis if PP-Motion behaves like a smoothness detector or its unstated license blocks reuse.

## Ranked shortlist

### 1. C13: Intervention response fingerprint

This candidate replaces passive anomaly scoring with an active query. The model receives the same small, phase-specific motion edits for every walk, and its propagation or relaxation becomes the measurement. That response can expose coordination even when mean surprise is similar. The strongest null is equally useful: if response follows edit norm, speed, physical feasibility, or lift choice, then a normal-motion generator is not a stable probe of an observed walk. GaitDynamics makes the 72-hour test possible without training a generator. The output must be called a model-response fingerprint, not a biological response.

### 2. C04: Physical feasibility versus healthy typicality

This is the cleanest validity test for the direction lock. It asks whether a predictor of unimpaired dynamics and a simulator-derived feasibility judge measure different things. Either answer matters. Separation would create a two-axis instrument that distinguishes plausible atypical gait from a bad reconstruction. Collinearity would invalidate a broad class of physical-score claims on monocular gait. Use the public PP-Motion checkpoint because it is the shortest accessible path to a feasibility score. The main risks are Rajagopal retargeting, PP-Motion's unstated license, and dependence on lift quality.

### 3. C03: Multi-lift minimum normalizing edit

This is the most interpretable candidate. It returns a proposed joint-angle change and an agreement score rather than a class probability. The lift ensemble makes identification part of the claim: the method must say when one video cannot support a stable edit. The primary risk is that the optimizer corrects the lifter's training bias rather than the gait. Camera perturbations, cadence preservation, and known AMASS corruptions directly test that failure.

### 4. C14: Conditional coordination graph

This is the strongest low-compute S-JEPA direction. Masking one anatomical region and predicting another turns the local model into a directed coordination probe. With the existing Core11 checkpoint, it can reveal left-right and pelvis-to-distal dependencies without relying on a pixel generator or a scalar latent. The main rebuttal is that missing-joint confidence or simple covariance draws the same graph. Balanced queries, random encoders, and a covariance baseline make that rebuttal decisive.

### 5. C11: Simulator-teacher residual forecasting

This candidate tests whether a simulator-informed correction signal can change what a JEPA learns while preserving a separate typicality residual. It follows the GoalForce and ControlNet pattern: freeze the motion representation and train one zero-initialized route. MotionCritic and matched-jerk arms determine whether PP-Motion adds simulator-informed content beyond perception or smoothness. Its contribution is a learning mechanism, not another application of a frozen score.

### 6. C05: Motion-conditioned depth residual

Depth error is the largest threat to every AMASS-to-GAVD claim. This candidate makes it the object of study. AMASS provides exact relative depth, while two frozen target-domain witnesses expose ambiguity on GAVD. A stable temporal depth residual would make later 3D measurements more credible. Failure would justify staying in 2D. Its Promise score is lower than the top five, but it earns a slot because it can resolve a shared dependency for the entire program.

### 7. C07: Body-device relational dynamics

This is the high-promise, lower-feasibility exception. A cane, walker, or prosthesis is part of the moving system, not background. Predicting relative body-device trajectories could reveal coordination that body-only models erase. The 72-hour gate does not require a label result: tracks must be visually credible, and real trajectories must beat presence-only and phase-matched trajectory controls on forecast likelihood. External-aid claims also require 12 independent GAVD sources. Failure ends the idea before expensive training. GAVD's eight prosthetic source videos and ProGait's four participants limit the claim.

## Coverage matrix

| Shortlist item | L1 vision physics | L2 physical grounding | L3 S-JEPA | L4 simulator loss | L5 motion depth | L6 Qwen RL tools | L7 body-object |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| C04 Physics versus typicality | Yes | Yes |  | Yes |  |  |  |
| C13 Intervention response | Yes | Yes |  |  |  |  |  |
| C03 Minimum normalizing edit | Yes | Yes |  |  |  |  |  |
| C14 Coordination graph | Yes |  | Yes |  |  |  |  |
| C11 Simulator teacher |  | Yes | Yes | Yes |  |  |  |
| C05 Motion depth residual | Yes |  | Yes |  | Yes |  |  |
| C07 Body-device dynamics | Yes |  | Yes |  |  |  | Yes |

The shortlist covers six of seven requested lenses. It omits the Qwen reinforcement-learning lens. C21 has a public base and a cheap pilot, but its central question is tool routing rather than whether predictive dynamics can measure gait. That is not strong enough to displace a gait-first candidate.

## Portfolio dependencies and safeguards

- C03, C04, and C13 share the GaitDynamics Rajagopal bridge. Validate that bridge once on known AMASS projections before any GAVD label analysis. C14 and the 2D arm of C05 remain viable if it fails.
- Use PP-Motion as C04's first feasibility implementation. Keep a deterministic PhyMotion-style stack as a sensitivity analysis, not a day-one dependency.
- C08 lift consensus and C09 cross-prior abstention should become common safety evaluations, not separate headline proposals.
- C01 horizon maps, C02 cycle residuals, C17 repair dispersion, C25 MDM surprise, C26 GaitEncoder, and C27 scalar PP-Motion are required baselines or ablations for the selected questions.
- Lock source-video folds, corruption parameters, supported labels, and all S0 controls before fitting any label head.
- Do not report a full-GAVD result from the 18-source local cache. It is only a pipeline pilot.
- Treat `gait_pat` as an observed presentation label. Do not infer diagnosis, severity, laterality, treatment, or participant-level performance.

## Open questions before Phase 4

- All 348 GAVD source videos are confirmed decodable on HAIC and cover all 1,874 sequences. Which extractor, schema, and confidence fields should create the missing full-video pose manifest?
- Registered AMASS SMPL+H assets are available. Do WHAM and GVHMR require additional SMPL-X or SMPLify variants that are not present?
- BABEL annotations are downloaded but not extracted. Can extraction and AMASS path matching finish before the final locomotion pool is locked?
- Can PP-Motion's authors or release surface clarify checkpoint and code licensing?
- Can C13's response fingerprint remain stable across two lift routes, both perturbation directions, and at least three perturbation magnitudes?

## Stop gate

The user cleared the Phase 3 gate. Phase 4 writeups and verified figures now exist for all seven ranked proposals: C13, C04, C03, C14, C11, C05, and C07. The adversarial Phase 5 review has not started.
