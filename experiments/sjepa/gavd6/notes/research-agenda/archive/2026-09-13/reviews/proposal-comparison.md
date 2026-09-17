# Adversarial portfolio selection

**Historical selection review, before the optical-flow request.** The final [ranking](../../../README.md) replaces the former constraint candidate with [P7: motion beyond keypoints](../proposals/motion-beyond-joints.md). This memo preserves the earlier decision and objections; it is not the final recommendation.

This review evaluates the seven shortlisted directions as one-week bets. The numbers are candidate IDs from the orchestration message. The rank is a judgment about the combination of conceptual distinction and executable evidence. It is not an estimated probability of paper acceptance.

## Recommended ordering

| Rank | Candidate | Novelty prospect | First-result feasibility | Principal reason |
| ---: | --- | --- | --- | --- |
| 1 | 1. Evidence-preserving restoration | Medium to high if a new preservation-recovery frontier is demonstrated | High with a working public motion prior | Makes a foundation model answerable to the evidence it could otherwise erase |
| 2 | 6. Cross-activity motor personalization | Medium to high for actual out-of-activity future prediction | Medium, dependent on usable within-person AMASS coverage | Tests whether a few walking seconds contain transferable information about how this person moves |
| 3 | 2. Constructive ambiguity witnesses | Medium for an efficient, operational finding-level instrument | High for witnesses; lower for a strong new method | Gives a concrete alternative explanation rather than a confidence score |
| 4 | 5. Paired-edit transfer | Medium only with unseen-edit composition and actual student gain | Medium to high after rendering availability | Can teach responses to change, but Jacobian transfer and causal weighting are nearby |
| 5 | 3. Teach/no-teach prediction | Low to medium after new ATLAS and CCH comparisons | High for screening, medium for enough actual students | Best continuity with repo evidence, but its broad novelty claim is already occupied |
| 6 | 4. Distributional KD | Low to medium unless marginalization produces a new operational result | High for a toy, medium for convincing real motion | Correctly handles hidden information, but probabilistic KD and mean-versus-distribution distinctions are established |
| 7 | 7. Constraint conflict witnesses | Low to medium, with serious validity risk | Low for nontrivial exact claims | Nonconvex search failure is not infeasibility, and easy impossible conditions are solved by geometry |

Do not make all seven sound equally likely or equally novel. Candidates 1 and 6 merit most of the finite experimental budget. Candidates 2 and 3 are excellent diagnostic companions. Candidates 4, 5 and 7 need especially careful contribution boundaries.

## The hardest objection to each proposal, and the necessary revision

### 1. Restoration that preserves unusual evidence

**Objection:** The method may simply restore less aggressively. Preserving a planted deviation is easy if it leaves all noise untouched. Alternatively, ordinary hard data consistency already gives the claimed behavior.

**Required revision:** Compare at matched noise removal or matched recovery accuracy, and plot a preservation-versus-recovery frontier. Include hard observation clamping, confidence-weighted smoothing, robust filtering and ordinary posterior/inpainting guidance. Give true unusual motion and tracker artifacts matched duration, amplitude, frequency and joint support. Otherwise jerk, rarity or corruption magnitude solves the distinction. A paired RGB view can distinguish visible motion from skeletal tracker noise, but hidden true motion remains ambiguous. Make abstention part of that boundary.

**Headline worth pursuing:** A small adaptation recovers missing or noisy motion while retaining independently verified unusual movement that an ordinary prior systematically attenuates. This should be shown on real held-out mocap motion, not only synthetic disease-like edits. GAVD can establish observation robustness, not the patient's hidden 3D truth.

### 2. Constructive ambiguity witnesses

**Objection:** Everyone knows that monocular 3D pose is ambiguous. A reflected skeleton, global left-right relabel or unrestricted depth change could make a dramatic but trivial figure.

**Required revision:** Exclude these trivial gauge transformations. Hold body dimensions, permitted cameras, reprojection tolerance, timing and a declared smoothness/kinematic-validity envelope fixed. Seek two whole trajectories that support opposite values of a precisely defined *kinematic finding*. Prefer a finding relevant to a downstream decision over average joint-depth disagreement. Each returned pair is a witness of ambiguity under those constraints. A failed search is only unresolved.

**Headline worth pursuing:** Public monocular pipelines confidently disagree with verified alternative trajectories on finding-level conclusions, and the proposed search discovers those witnesses efficiently enough to withhold only the unsupported conclusions. Include a raw constrained optimizer without a learned prior; the foundation model must improve useful witness discovery, not only visual plausibility.

### 3. Predict the value of teaching

**Objection:** ATLAS already constructs student-observable targets, and CCH already predicts cross-modal KD benefit using actual student outcomes. A new spectral score with the same demonstration is incremental.

**Required revision:** Fix the student, predict the effect of temporal target and rank choices, and hold out teacher families or datasets. Compare CCH, residual CCA, LogME or task readouts, and a short student pilot. Include no-transfer and known unfavorable cases prospectively. The predictor's ground truth must be independent physical student utility, never its own accessible-target loss. Do not call spectral algebra a new theorem.

**Headline worth pursuing:** Temporal target accessibility predicts whether teaching improves or harms physical forecasting under data limits, where global teacher prediction and strong transfer scores fail. This is a potentially useful empirical contribution; it is not the highest-novelty idea in the portfolio.

### 4. Distributional KD with a less-informed student

**Objection:** A conditional mean misses multimodal futures by definition. Showing this with a hidden coin flip or a two-branch toy does not establish a new problem or method. Naively averaging teacher Gaussians is already mixture modeling.

**Required revision:** Define the deployment information boundary exactly. The teacher may see upper-body or video evidence absent from the student, but the student must predict the distribution conditional on what it can see. Evaluate proper scores, event probabilities and calibration against actual held-out future motion. Compare direct probabilistic forecasting with the same data and a simple mixture of teacher predictions. Show that privileged supervision helps under a sparse-outcome budget rather than merely increasing the amount of training information. Keep MSE as a diagnostic, not the principal measure.

**Headline worth pursuing:** Privileged world-model evidence teaches calibrated unresolved alternatives, with a demonstrable benefit over direct probabilistic forecasting when deployment sensing is restricted. Calling all teacher uncertainty irreducible would be wrong: model error and missing-information uncertainty need separate controls.

### 5. Distill the response to a motion edit

**Objection:** This is Jacobian distillation or finite-difference matching. [CAER](https://arxiv.org/abs/2608.30897) already contrasts action-conditioned and action-removed predictions to redirect supervision toward action-sensitive changes.

**Required revision:** Use paired *actual rendered input changes* with identical nuisance variables, and train the small model to preserve a declared response relation. Synthetic motion edits are not biological interventions. Hold out edit families and compositions, with equal total teacher queries and student gradient budgets. Compare coordinate-edit supervision, ordinary contrastive pairs, Jacobian distillation, and a CAER-style weighting baseline. A gain on trained edit magnitude alone is insufficient.

**Headline worth pursuing:** Learning a teacher's response relation transfers to unseen combinations of motion changes, while matching its feature state does not. Raw kinematic supervision is a particularly strong competitor because the intervention is already specified in coordinates.

### 6. A short walk predicts how the same person moves elsewhere

**Objection:** The personal code may only identify height, limb ratios, walking speed, motion-capture source or action. It may generate a person's style without predicting what that person actually does next.

**Required revision:** Personalize on a short walking support sequence; test true future motion in an activity absent from the support. Hold the test activity's current state fixed in every comparator. Normalize and control morphology, cadence and source. Use another person's support matched on these variables, static-pose-only support, identity retrieval, an unpersonalized backbone and ordinary few-shot adaptation. Split people before building support/query sets and exclude overlapping mocap constituents. Audit whether enough independent people have multiple activities before launching training.

**Novelty boundary:** [MetaGait's few-shot generation/reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC12886945/) already addresses personalization from 1–5 gait cycles. [STyMo](https://arxiv.org/abs/2609.04500) adapts motion style from seconds of paired data and separates static and temporal style. The new object must be predictive transfer of person-specific dynamics across activities, beyond morphology and style imitation. Do not label a latent factor as impairment, motor capability or diagnosis without independent measurement.

### 7. Conflicting motion conditions

**Objection:** A diffusion model cannot prove that conditions are mutually impossible. General minimal conflict sets and infeasibility handling are old optimization topics. A contrived pose that breaks a bone-length constraint is a trivial signal.

**Required revision:** Prefer **minimal verified constraint repair**: show a concrete original condition set, a small declared relaxation and a generated trajectory that demonstrably meets the relaxed set. Exact infeasibility certificates may be offered only for a stated tractable relaxation, with the logical direction correct. Learned priors may propose candidates, but cannot certify anatomy, physics or global impossibility. [MIC](https://arxiv.org/abs/2607.01990) already coordinates heterogeneous constraints during motion generation, so compare it where code is usable, plus simple constrained optimization.

**Honest recommendation:** Keep seventh as exploratory and lowest priority. I did not find a clearly stronger distinct replacement that survives both the prior three portfolios and the one-week availability constraint. Turning it into active sensing would overlap prior Adaptive Gait Examination; turning it into counterfactual normalization would overlap the earlier minimum-edit proposal. A new title should not conceal those duplications.

## Portfolio-level corrections

1. The meaningful unit of success is one strong contribution supported by one mechanism experiment, an independent transfer test and adversarial controls. Seven proposals are choices, not a seven-way training campaign.
2. A one-week benchmark should prioritize the public model and already stored AMASS over a heavy video generator unless paired rendering and measured throughput are already established. Implementation speed does not remove extraction and encoding costs.
3. Use the latest run only as motivation for moving beyond global feature-prediction scores. It does not validate erasure, ambiguity, personalization or distributional transfer.
4. The first pilot for candidate 1 must establish matched restoration strength; candidate 6 must establish usable support/query coverage. These are more informative day-1 gates than a long training run.
5. Novelty claims should name the closest collision explicitly. A proposal can be worth trying even when its final contribution depends on a surprising result rather than a new mathematical operation.
