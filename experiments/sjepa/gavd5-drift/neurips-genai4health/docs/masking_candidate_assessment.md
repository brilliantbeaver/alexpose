# Could gait-informed masking support a GenAI4Health paper?

Reviewed September 6, 2026. This is an author-facing assessment of a possible alternative paper. It does not report a new experiment or replace the current position-paper draft.

## Recommendation

**Yes as a research direction, but the current results do not yet support a strong empirical paper about the benefit of masking.** The notebooks implement and check a plausible anatomical prior. They do not compare models trained with different masking policies, so they cannot identify whether that prior improves the learned representation.

For the current submission, the evidence-grounded position paper remains the more defensible option. A masking-centered research paper would become substantially stronger after a matched policy comparison with useful downstream endpoints. A masking-centered position paper is possible, but would rely more heavily on an untested proposal than the current draft.

A suitable working title for the research direction is **Testing Gait-Informed Masking for Self-Supervised Pose Learning**. “Gait-informed” or “anatomy-informed” accurately describes the implemented selection. “Neurologically guided” would require a clearer, externally justified link between the selection rule and neurological measurements or clinically established impairments.

## What is implemented

The encoder receives the full 33-landmark pose layout. Its target sampler restricts hidden-feature prediction to 12 landmarks: both shoulders, hips, knees, ankles, heels, and foot tips. These locations can support investigation of body alignment and lower-limb movement. Retaining the full layout provides context outside the target subset.

This gives the learning task a body-structure prior, but the anatomical names do not establish that the resulting latent space represents neurological mechanisms. The internal motivating material discusses MS/PD gait features; the evaluated five-category subset does not include MS. A stronger paper would map each selected location to a specific externally supported gait quantity, without treating that motivation as demonstrated clinical validity.

The most defensible present claim is:

> We implemented and checked a target sampler that concentrates self-supervised prediction on 12 gait-relevant landmarks while retaining full-skeleton context. Its effect on downstream usefulness remains untested.

## What the saved results actually test

The masking notebook checks a synthetic batch. It confirms that no forbidden landmarks are sampled as targets, valid observations are respected, and seeded sampling is repeatable. In that example, 114 of 528 positions are hidden—about 22% of the whole layout and about 60% of eligible positions. These are software and denominator checks, not results from a trained masking-policy comparison.

The pretraining notebook uses the selected-landmark policy. Its optional ablation concerns a condition-label loss, not masking. No completed unrestricted, random-subset, or motion-aware masking comparison was found in the current notebooks or available result records.

The later classifier result also cannot answer the masking question. Learned features correctly classify six of 20 test videos, while direct pose summaries classify ten. Only one trained masking policy is represented. The difference could reflect many choices in training and feature extraction, so it neither proves that the selected landmarks help nor shows that they are harmful.

The exact nominal masking setting for the recorded trained run was not fully preserved. The synthetic notebook's 60% setting should not be presented as independently verified execution metadata for that run.

## How much novelty is available?

Guiding skeleton masking through body structure or movement is established:

| Primary research | Relevant precedent |
|---|---|
| [S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Predicts masked skeletal representations and uses motion-aware target sampling |
| [Masked Motion Predictors, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html) | Uses movement intensity to select prediction targets and learns through motion prediction |
| [Graph-based SkeletonMAE, ICCV 2023](https://arxiv.org/html/2307.08476v1) | Groups joints into body regions and compares body-part masking with random masking at matched masked-joint counts, including choices of masked regions |

The graph-based SkeletonMAE comparison is particularly important: choosing meaningful body parts instead of random joints, and controlling the number of hidden joints, are already explicit prior art. A general “first anatomy-guided masking” claim would be unsupported.

The potentially distinctive question is narrower:

**Does selecting gait-relevant prediction targets improve representations learned from noisy, video-estimated pose, compared with equally demanding generic masking?**

A result could concern annotation prediction, sample efficiency, or performance under specified visibility and recording conditions. A claim about clinical or neurological meaning would need an independently established clinical endpoint. [GaitForeMer](https://arxiv.org/abs/2207.00106), which combines forecasting and activity-classification pretraining and evaluates gait severity, provides relevant health precedent rather than evidence for our particular mask.

## The minimum informative comparison

| Training condition | Question it answers |
|---|---|
| Predefined gait-landmark targets | Performance of the proposed prior |
| Uniform targets from all valid landmarks, with the same number hidden | Whether selecting gait locations helps beyond generic masking |
| Several prespecified alternative 12-landmark sets | Whether the particular gait selection matters beyond restricting target locations |
| Motion-aware targets, if making a comparative claim against that approach | Whether the gait selection adds value beyond an established informed policy |

The first two are essential. Alternative landmark sets materially strengthen the attribution. Motion-aware masking is required if the manuscript claims an advantage over that relevant prior approach.

A fair comparison must match the **actual number of hidden valid tokens**, not merely the nominal percentage. Applying 60% to a 12-landmark subset and 60% to all 33 landmarks produces very different amounts of visible context. In the notebook's synthetic batch, the selected policy hides 114 positions; the analogous unrestricted calculation would hide 315. That is a hypothetical contrast explaining a confound, not a result of an executed ablation.

Keep pose preparation, eligible clips, encoder and predictor, loss, initial weights, training-video draws, and update budget aligned across policies. Prespecify how differences in landmark visibility affect target counts and coverage. Keep the downstream feature summary and fitting procedure fixed so that changing the mask does not also change the readout's anatomical prior.

Use the same video-level aggregation during classifier selection and testing, correcting the current mismatch in a new experiment. Include direct pose summaries, landmark availability, and the same encoder at its untrained starting weights. Repeat training across prespecified video splits and initializations; report paired uncertainty with videos as the resampling unit. Intervals obtained by resampling saved predictions are conditional on those fitted models, so variation across training initializations and splits should be reported separately. Repeated training does not turn clips into independent people.

## Which outcome should lead the paper?

A feasible first endpoint is held-out prediction of GAVD annotations, stated as annotation prediction rather than clinical diagnosis. A consistent improvement could support a task-specific representation-learning finding, including if its magnitude is modest.

For a stronger health contribution, evaluate a clinician-rated gait endpoint or independently annotated movement events. Pose-derived timing statistics may be useful exploratory diagnostics, but they are not automatically clinical ground truth. The existing temporal notebook has target-validity and timing limitations and should not be promoted to clinical evidence without correction and validation.

Training loss, masked-feature prediction error, visually appealing embeddings, and successful sampler checks are insufficient on their own. A lower loss can reflect an easier target set. The key test is whether the learned features improve a relevant held-out outcome under matched conditions.

A negative or null policy comparison can also be informative if its controls and uncertainty support the conclusion. The study should not be designed around obtaining a favorable masking result.

## Fit to this workshop and decision before submission

The [GenAI4Health call](https://genai4health.github.io/2026-NeurIPS/) requires empirical research submissions to support their central claims with current evidence; its position track allows evidence-grounded proposals. A completed gait-masking comparison could fit health-oriented representation learning, with the model presented as a possible perception component rather than an implemented agent. An evaluation involving future assistant interpretation would make the GenAI connection more direct, but it is a separate research question.

Current strengths are a concrete implementation, a curated health-related video source, and an interpretable hypothesis distinct from laterality. Current weaknesses are absent mask-policy controls, incremental methodological novelty, unvalidated neurological interpretation, and no evaluated generative system.

The deadline is September 9, 2026, 11:59 PM AoE. Completing the comparison requires new training and evaluation; it cannot be achieved by revising terminology or replotting the existing outputs. Original fold model artifacts are also missing from this checkout, so those must be recovered or the relevant runs repeated. Runtime and resource requirements have not been benchmarked, and no claim is made that the work fits the remaining time.

**Decision rule:** consider switching to the masking research paper only after a matched comparison provides an interpretable result and the full experiment can be documented. Until then, keep the current position paper's numerical evidence and treat masking as a clearly labeled future study. No new model training or masking experiment has been run during this assessment.
