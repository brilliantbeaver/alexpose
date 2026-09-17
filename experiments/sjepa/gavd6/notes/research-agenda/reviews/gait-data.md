# Gait data and literature audit for the revised portfolio

Checked 14 September 2026. This memo separates available evidence from proposed experiments. No training, checkpoint loading, or HAIC data access was performed for this audit.

## What the local data support

The GAVD sequence manifest contains **1,874 sequences from 348 video IDs**. It supplies sequence bounds, annotated frame counts, source height, recording URL, coarse presentation labels, and camera view. It does not supply verified participant identity, quantitative impairment severity, measured joint trajectories, forces, treatment response, or longitudinal disease outcomes. Its 12 presentation labels mix normal movement, exercise, style, broad abnormality, and named conditions. They must not be silently converted into confirmed diagnoses.

The `dataset_annotation` field has 254 normal and 1,620 abnormal rows, while `gait_pattern_annotation` has 291 normal rows. These are different annotation fields, not interchangeable binary labels. Grouping by video is the minimum sensible split. It is recording separation, not proven participant separation across videos. Any new quantitative GAVD landmark endpoint needs explicit annotation or independent validation of the reference trajectories.

The AMASS inventory contains 10,941 readable source files; 8,854 are eligible and converted to Core11. The converted set contains **189 unique audited identities: 151 training, 19 validation, and 19 test**. The subject-split CSV has 201 rows because some rows are aliases of the same person. Six eligible sources contribute: KIT 4,232 motions, BioMotionLab_NTroje 3,061, Eyes_Japan_Dataset 750, EKUT 349, ACCAD 247, and MPI_HDM05 215. Source diversity is strongly imbalanced and should not be mistaken for six equally powered external cohorts.

The raw records expose 156 pose parameters, translation, body shape, and dynamic shape coefficients. The existing motion-preservation loader resamples all 52 SMPL-H rotations. A 22-joint whole-body representation including shoulders, elbows, wrists, spine, and pelvis is therefore practical. Fingers and soft tissue should not become headline targets simply because parameters exist. Core11 is a useful ablation, not the only data available. The existing validation split supports nine calibration and ten development people under the motion-preservation role assignment. Reuse the protected final people instead of creating favorable replacement splits.

## Required inspiration papers and usable assets

| Work | Verified observation | Consequence |
| --- | --- | --- |
| [S-JEPA project](https://sjepa.github.io/) and [ECCV paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Learns masked latent joint representations for action recognition. The inspected project page exposes no author weights. | Use the repository's actual compatible student or a small head. Do not budget around an unverified official checkpoint or describe masked reconstruction as proven causal forecasting. |
| [SleepFM](https://www.nature.com/articles/s41591-025-04133-4) and [official code](https://github.com/zou-group/sleepfm-clinical) | Multimodal pretraining and frozen-feature heads use large sleep cohorts and EHR-linked outcomes. Base and downstream checkpoint paths are documented. | Borrow heterogeneous-signal pooling and efficient adaptation. The clinical prediction claims require labels absent from GAVD and AMASS. |
| [GaitDynamics](https://www.nature.com/articles/s41551-025-01565-8) and [official code](https://github.com/stanfordnmbl/GaitDynamics) | Public diffusion and force-refinement files resolve. The paper specifies an OpenSim Rajagopal model without arms and warns about skeletal mismatch. | Public does not mean compatible. SMPL/Core11 cannot be substituted directly, and inferred forces cannot serve as measured truth. |
| [GaitForeMer](https://arxiv.org/abs/2207.00106) and [official code](https://github.com/markendo/GaitForeMer) | Forecast pretraining precedes severity estimation. Clinical PD data are explicitly nonpublic. The author Box link returned a tool error today. | Generic forecasting for gait severity is established and its private endpoint cannot be reproduced here. Do not make this checkpoint a mandatory dependency. |
| [GAVD](https://arxiv.org/html/2407.04190v1) | Paper experiment details include a balanced training subset and an abnormal-only GAVD test subset; performance drops on unseen normal recordings and varies by view. | The headline 92/94% does not establish a solved, balanced external diagnosis task. Still honor the user's decision to exclude binary classification as a headline. |
| [V-JEPA 2/2.1 official releases](https://github.com/facebookresearch/vjepa2#models) | Public backbones and loaders include 2.1 ViT-B and ViT-L. The user already has V-JEPA checkpoints on HAIC. | Frozen feature extraction is a credible shared starting point. Exact local checkpoint and predictor compatibility still govern use. |

Direct GaitDynamics file pages: [diffusion](https://github.com/stanfordnmbl/GaitDynamics/blob/main/example_usage/GaitDynamicsDiffusion.pt), [refinement](https://github.com/stanfordnmbl/GaitDynamics/blob/main/example_usage/GaitDynamicsRefinement.pt). Resolution of a file page is not a successful local inference run.

## Recent collisions that rule out easy novelty claims

- [Human-JEPA](https://arxiv.org/abs/2608.21160) already introduces human-centric adaptation with anchored forecasting. Adapting JEPA to human video is not sufficient novelty.
- [A Gait Foundation Model Predicts Multi-System Health Phenotypes](https://arxiv.org/abs/2603.25283) uses 3,414 deeply phenotyped adults. Broad gait-health embedding claims are both occupied and unsupported by the local labels.
- [FoundationGait](https://arxiv.org/abs/2512.00691) spans multiple gait tasks and large pretraining data. Another generic gait foundation model is a weak one-week strategy.
- [PHASOR](https://arxiv.org/abs/2606.01851) factors phase and pose for cross-embodiment representations. A phase token alone is not a conceptual contribution.
- [WRBench](https://arxiv.org/abs/2606.20545) already diagnoses persistence of world state when observations disappear. An occlusion benchmark needs a different capability and a useful method, not a renamed persistence failure.
- [Active3DPose, ICLR 2023](https://openreview.net/pdf?id=CPIy9TWFYBG) already chooses camera views for human pose estimation. Active acquisition by itself is established.
- [Certified World Models as Sensing Clocks](https://arxiv.org/abs/2607.01537) already studies calibrated decisions about when to observe again. A threshold on prediction uncertainty would not establish novelty.
- [Beyond Motion Imitation](https://arxiv.org/abs/2603.12408) provides another warning that accurate kinematics do not establish accurate kinetics. Avoid force, loading, stability, and treatment claims without the corresponding reference measurements.

## Candidate A: Learn the value of a correction, independently of its generator

**Question.** Can a verifier trained on corrections from several simple repair methods predict which local changes improve motion measurements when the repair method itself is held out?

The key update after notebook 06 is that a reconstruction generator is not the scientific commitment. A useful correction family must be established first. Use exact identity, partial temporal filters, and a verified repair model only if their calibration curves demonstrate meaningful improvement. MoMask stays a documented adverse reconstruction comparison unless its focused audit restores usefulness.

Train a small head on frozen V-JEPA features, local flow, the observed trajectory, and the proposed displacement. Its target is the actual change in reference error after applying that displacement. Hold out an entire correction generator. Construct mixed cases requiring opposite local decisions in different limbs within the same clip. The main question is transfer of evidence-based judgment, not classification of clips into clean or noisy.

AMASS supplies measured motion and controlled corrupted observations. Add untouched natural motions before interpreting edited examples as important. GAVD supplies external visible-evidence tests; it cannot establish absolute 3D repair by agreement with another estimator.

**48-hour test.** Measure the oracle for the exact temporal or limb decisions available to the head. Continue only if the practical correction family has clear headroom above the declared repair target. Then compare a small flow-only verifier with flow plus frozen JEPA. Stop if practical directions have no headroom or inexpensive evidence rules already attain the useful frontier. Expected pilot budget: tens of GPU-hours after cached candidates, subject to measured throughput.

**Novelty condition.** Generalization to a new correction generator, difficult simultaneous preserve-and-repair cases, and actual real-motion benefit beyond measurement consistency. A gate rejecting bad MoMask reconstructions does not meet this condition. This is the strongest continuation of current assets, but it is not established by the pilot.

## Candidate B: Spend additional pixels where they change a motion conclusion

**Question.** Can a small head predict which extra body-part crop will improve a motion estimate, and can that selection rule transfer to a different motion question without retraining?

Begin with a low-resolution video and frozen JEPA features. Offer a fixed budget of one or two high-resolution crops from frames that have already been observed. The model chooses a foot, knee, trunk, arm, or no crop. Train it to predict the actual reduction in a target error after revealing that crop. This differs from selecting the region with the largest uncertainty: an uncertain hidden foot might remain unobservable, while an arm crop can reveal useful phase information.

The scientific object is the **value of added evidence**. A successful result would show that a learned selection rule transfers across tasks, such as from joint displacement estimation to evaluating a proposed local correction. Keep total pixels, video context, and task-model capacity matched. Freeze the selector on the second task. No active camera hardware is required.

AMASS renders provide controlled visibility with full-body references derived from real captured motion. For GAVD, use real visible 2D landmarks or event timing on a small explicitly annotated subset. Two-dimensional image annotation of existing videos is possible within the week; clinical data collection is unnecessary. If annotation cannot fit the schedule, do not pretend tracker output is ground truth. Report the GAVD result as observational only.

**48-hour test.** First run every allowed crop on a small sample to measure the oracle improvement and cost. Then compare learned selection with uniform crops, detector confidence, optical-flow magnitude, simple geometric visibility, and a matched uncertainty selector. Stop if oracle selection has little headroom or confidence/flow selects equally well. A first assay and small head should fit roughly 40 to 100 H100-hours, but time a representative extraction batch before fixing the budget.

**Novelty condition.** The transfer of measured evidence value across tasks is the potential contribution. Active camera movement, generic attention, and uncertainty-based sensing are prior work. This is an independent alternative to repair and avoids the requirement that MoMask be useful.

## Candidate C: Determine when the rest of the body changes the predicted future

**Question.** For natural motions with similar recent lower-body position and velocity, does upper-body motion explain a substantial part of the remaining variation in the recorded lower-body future?

The first experiment is an information test, not a new architecture. Match natural AMASS prefixes on lower-body pose history, velocity, cadence, morphology, source, and duration. Ask whether adding the shoulders, elbows, wrists, and trunk improves predictions of the actual future feet and knees. Repeat at increasing lower-body history lengths. This tests whether upper-body observations provide complementary state or merely compensate for an inadequate temporal baseline.

If the signal survives, add a small full-body evidence branch to a frozen JEPA representation and compare it with the equally expressive raw-trajectory branch. A useful further result would let the model request only the complementary body region, linking naturally to Candidate B. Avoid interpreting predictive dependence as a causal physiological mechanism or labeling natural coordination differences as impairment.

GAVD can test whether real upper-body video improves predictions of later visible lower-body image motion under viewpoint changes. Again, 2D references require validation. Its absent participant IDs prevent the same person-level claim as AMASS.

**48-hour test.** Run matched linear, nearest-neighbor, and small temporal baselines before expensive feature extraction. Stop if longer lower-body history removes the gain, matching destroys sample size, or static shape and source explain the improvement. Treat an upper-body benefit on hand-edited phase changes alone as insufficient. Pilot cost is low because the main first test uses existing motion data.

**Novelty condition.** Whole-body input and sparse-joint completion are crowded. The contribution would require a reproducible regime where observed state is insufficient in a precisely measurable way, plus a compact transferable remedy. This is a lower-priority mechanistic candidate, not a claim that adding arms to Core11 makes an ICLR paper.

## Recommended use in the final portfolio

Candidate A offers the closest useful continuation if candidate headroom is recovered. Candidate B is the more distinct alternative and should receive a cheap oracle assay rather than an unconditional training commitment. Candidate C is best used as an early information test or as support for B unless its natural-motion effect is unusually strong.

The one-week plan should be conditional on measurable early effects. The data and checkpoints make these experiments possible; they do not make a significant result or main-track acceptance highly probable by themselves.
