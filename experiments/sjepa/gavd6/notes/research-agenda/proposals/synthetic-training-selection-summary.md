# Selecting Synthetic Training Data from a Pose Estimator's Learning Response

*Condensed research proposal. No experimental results are claimed. [Full protocol](synthetic-training-selection.md).*

## Abstract

We propose to test whether a short training intervention reveals which synthetic examples will improve an unfamiliar pose estimator on real video. A fixed probe updates the estimator, and a selector observes the resulting prediction changes on unlabeled deployment clips. The selector learns lesson benefits from supervised synthetic adaptation trials and chooses further training without deployment reference labels. The central hypothesis is that target-video prediction changes add transferable information beyond current errors, static predictions, and learning progress on labeled source data. Evaluation uses independent GAVD landmark annotations and an excluded estimator architecture. This is an untested feasibility hypothesis, not an established adaptation method.

## 1. Introduction and Related Work

A pose estimator's current weaknesses need not determine which additional examples it can learn from. We investigate whether a controlled update reveals useful differences in learning behavior. Synthetic-data selection, failure-driven adaptation, learner examination, and prediction of training gains are established approaches [1–6]. The proposed contribution is narrower: prediction changes on **unlabeled target video** may improve lesson selection beyond source-training history and either the estimator's pre-update or post-update state. A single probe samples one update direction; it neither identifies complete learning dynamics nor establishes whether its changes improve accuracy.

## 2. Method

**Students and lessons.** Target visible 2D landmark estimation using pretrained RTMPose and HRNet source students, reserving ViTPose for architecture-transfer evaluation. Initially adapt prediction heads using source-selected recipes. Construct eight equal-size lessons from natural full-body [AMASS](https://arxiv.org/abs/1904.03278) motion, varying view, resolution, blur, occlusion, and pose coverage. Mix synthetic examples with fixed labeled COCO replay to retain existing competence. Align twelve visible body-landmark definitions and fix detector, crop, and coordinate conventions across methods.

**Probe and outcome collection.** Apply a common probe of budget $P$ to student $M_0$, obtaining $M_p$. On identical unlabeled context clips $U$, record

$$
r(U)=f_{M_p}(U)-f_{M_0}(U).
$$

Retain per-joint response structure, synthetic diagnostic errors, source-loss history, available model/optimizer descriptors, and frozen V-JEPA context features [7]. From the same $M_p$, independently train each lesson branch and replay-only branch for budget $B$. Define the supervised selection target on separate synthetic references:

$$
g_k(B)=E_{\mathrm{ref}}\!\left(A_{0,B}(M_p)\right)-E_{\mathrm{ref}}\!\left(A_{k,B}(M_p)\right).
$$

Here $A_0$ is replay alone and $A_k$ is replay plus lesson $k$; positive gain denotes improvement over replay. Match optimizer initialization, batch size, and updates. Synthetic branches share replay exposure; replay-only training fills synthetic slots with additional real examples. Thus equal update budgets do not imply equal real-data exposure.

**Selection and transfer.** Begin with nearest-neighbor gain prediction; a small neural selector is optional. Choose the highest predicted gain, allowing replay alone with gain zero. Fit normalization and selectors only on source episodes. Separate probe, lessons, diagnostics, contexts, and references; validate on excluded people, checkpoints, and domain combinations. ViTPose and its adaptation histories remain excluded from fitting and tuning. At deployment, select one lesson per collection from permitted context and diagnostics, then adapt a fresh copy of $M_p$ without carrying updates between collections.

## 3. Experimental Design

**Mechanism controls.** Compare selectors receiving only pre-probe snapshots, only post-probe snapshots, or pre-probe snapshots plus change. All select for the same post-probe student with matched capacity, context, outcome database, and remaining budget. The decisive comparator additionally receives source-loss history, both synthetic diagnostic snapshots, and model/optimizer descriptors, but only post-probe target predictions. The full teacher adds target prediction change. An advantage over this matched source-progress comparator is required; ordinary curriculum gains would not establish the proposed mechanism.

Include the original student, probe alone, replay from $M_p$ for $B$ updates, and replay from $M_0$ for the full $P+B$ budget. Compare random, balanced, fixed-best, context-matching, synthetic-weakness, hard-example, and scalar-response-magnitude selection. Replace V-JEPA with image features, simple descriptors, or no context features to assess its incremental value. Encoder comparisons cannot isolate the JEPA training objective.

**Real evaluation.** Use six [GAVD](https://arxiv.org/abs/2407.04190) collections spanning three view groups and two source-resolution ranges: 18 context recordings, 24 early-reference recordings, and 36 untouched confirmation recordings. Annotate four preselected frames per reference recording and twelve landmarks where visible, yielding up to 2,880 placements. Keep contexts and references disjoint, exclude protected recordings, and annotate independently of model failures and method preference.

Freeze lessons, selectors, adaptation recipes, and analysis before inspecting any GAVD outcome table. Early outcomes permit continuation or stopping only. Any outcome-guided revision must be disclosed as labeled development and evaluated on untouched confirmation data.

**Endpoints.** Measure visible-landmark Euclidean error, normalized by independently annotated person-box diagonal, averaged within recordings; also report pixel error. Fix visibility masks and missing-prediction handling. The primary endpoint is the paired recording-level error difference on held-architecture confirmation data against the source-selected strongest comparator without target prediction change. Report the matched source-progress contrast separately, full-budget replay, recording-clustered intervals, every collection, and damage to initially accurate landmarks. Synthetic selection regret and predeclared student–lesson crossover tests provide supplementary evidence. Six collections are six selection decisions per student, not thousands of independent decisions.

## 4. Feasibility, Limitations, and Expected Contribution

Limit initial feasibility work to two days: verify textured rendering, trainable estimators, replay access, independent annotations, useful real adaptation, and meaningful lesson-choice differences. Freeze a minimal source-fitted selector before opening early real outcomes. Do not scale if these prerequisites fail. Six source checkpoints, eight lessons, and two budgets would require 96 candidate adaptations plus controls; reuse each adapted checkpoint across reference settings without treating those evaluations as independent learners. Report measured runtime and annotation effort rather than assuming eight H100s ensure readiness.

Both synthetic-to-real adaptation and transfer of response-informed lesson rankings remain unverified. Limited student diversity and six deployment collections constrain generality. A successful study would demonstrate useful target-response information beyond static weaknesses and source progress on an excluded architecture, while improving on full-budget replay. Augmentation gains alone, smoother predictions, or lower latent loss would not establish that contribution. The evidence currently supports a bounded feasibility test, not a confident one-week paper commitment.

## References

1. [Task2Sim](https://rpand002.github.io/data/CVPR_2022_task2sim.pdf). CVPR, 2022.
2. [PoseExaminer](https://arxiv.org/abs/2303.07337). CVPR, 2023.
3. [Meta-Sim](https://arxiv.org/abs/1904.11621). 2019.
4. [Towards Black-box Iterative Machine Teaching](https://proceedings.mlr.press/v80/liu18b.html). ICML, 2018.
5. [Model-Based Meta Automatic Curriculum Learning](https://proceedings.mlr.press/v232/xu23a/xu23a.pdf). CoLLAs, 2023.
6. [AdaptPose](https://arxiv.org/html/2112.11593v2). CVPR, 2022.
7. [V-JEPA 2](https://arxiv.org/abs/2506.09985). 2025.
