# Continued ideation: independent review of synthetic teaching

**Status:** this document preserves the original independent reviews. The lead assessment has since been revised: the proposal is scientifically plausible but too speculative to recommend as the primary one-week route to a significant, novel paper result. See [the current recommendation](../proposals/synthetic-training-selection.md#9-final-research-judgment). This reassessment did not involve new experiments or a new independent review.

**Original review decision, 14 September 2026:** the probe-conditioned synthetic teacher was judged promising enough to pursue. That endorsement superseded the earlier search's conclusion that no new direction had cleared review. It did not change any measured result or establish a high probability of an ICLR paper.

Read the [final proposal](../proposals/synthetic-training-selection.md). No model was trained, no checkpoint was loaded for inference, no GAVD landmark panel was annotated, and no HAIC job was submitted during this ideation.

## How the search changed

The [first decision record](direction-search.md) rejected noise-corrected JEPA targets, a reusable visual observation likelihood and predictive feature caching as deadline recommendations. After the request to continue in alternative directions, the search expanded to event-defined prediction, anticipation from the environment, adaptable measurement conventions, online learning and synthetic teaching.

Event-defined prediction had a coherent temporal-abstraction argument, but close prior work and a strong direct physical predictor weakened its distinctive benefit. Environment-conditioned anticipation supplied a real source of information beyond skeleton history, but required new synchronized scene/motion data or transition annotations. These alternatives were considered; they were not silently presented as completed experiments.

Synthetic teaching became the preferred route for three reasons:

1. Existing real experiments show that targeted synthetic training can improve pose estimators, and that the same curriculum can help one deployment distribution while harming another.
2. A fixed training probe gives partial evidence about a model's learning behavior, beyond its current answers or confidence.
3. A small student–lesson matrix can produce measured teaching-utility targets cheaply, because each adapted checkpoint can be evaluated across many deployment settings.

The first point is published evidence. The second is a mechanistic argument. The third is a computational simplification. None is a local result from the proposed system.

## Independent reviews and resulting changes

Three agents reviewed methods and prior work, data and model feasibility, and scientific weaknesses. They received follow-up tasks as the candidate changed. Each then read the written proposal independently. Their final judgments supported a qualified decision to pursue it; none endorsed a numerical success probability.

| Critique | Revision in the final proposal |
| --- | --- |
| Task-to-simulator mapping and synthetic failure mining already exist. | Narrow the claim to cheap teaching of an unseen estimator, conditioned on unlabeled real video and measured learning response. Cite Task2Sim, Meta-Sim, PoseExaminer, PoseAug and AdaptPose. |
| A world model of a learner is also an existing curriculum-learning idea. | Cite black-box machine teaching and MM-ACL. Keep direct gain prediction as the core; do not force a latent learner predictor into the initial method. |
| Current predictions do not determine how an estimator learns. | Add a fixed short training probe. Describe its response as partial evidence of plasticity, not identification of the full learning dynamics. |
| The teacher would otherwise infer error from unlabeled consistency. | Add an explicitly labeled synthetic weakness profile, available in both training and deployment. Treat simple weakness matching and hard-example selection as strong baselines. |
| The probe itself could explain the benefit, or damage the comparison starting point. | Add original-model, probe-only and full-budget replay controls, and a teacher ablation removing the training response. Require net improvement over the original estimator. |
| The initial plan multiplied adaptations by the number of domains unnecessarily. | Reuse fixed student–lesson checkpoints. Six students, eight lessons and two budgets yield 96 candidate adaptations, plus controls. Treat repeated domain evaluations as correlated observations. |
| Different budgets would assign incompatible utility labels to the same inputs. | Include the update budget in the gain definition and teacher input. |
| The local renderer and tracker do not implement trainable RGB adaptation. | Identify textured rendering and trainable RTMPose/HRNet/ViTPose as new dependencies. Keep MediaPipe inference and simple pilot rendering separate. |
| PoseExaminer's paper assets and training recipe were being treated too broadly. | State the public release's one texture and two backgrounds, missing default texture path, substantial original-data replay, and different 3D objective. |
| Synthetic student improvement does not establish synthetic-to-real teacher transfer. | Name both transfer problems. Make real context-to-utility transfer a central pre-abstract test. |
| Crop resizing can erase a small-person condition. | Fix detector/crop processing and reduce source resolution before crop resizing. |
| Different pose heads have incompatible confidence scales. | Prefer common coordinate-response summaries or source-only confidence calibration. |
| Early real evaluation could invalidate the target-label-free claim. | Freeze source-selected configurations before opening GAVD outcome tables. Limit early references to the proceed-or-stop decision; disclose and narrow the claim if used for tuning. |
| The annotation count and effective sample size were unclear. | Specify 60 distinct reference recordings, 24 early and 36 final, plus 18 separate context recordings across six deployment collections. State that six settings are six teaching decisions per student, not thousands of decisions. |
| The proposal overlaps conceptually with P7. | Explain the difference between choosing auxiliary latent targets and selecting labeled source lessons for pretrained visual estimators. |

## What actually supports the recommendation

The [PoseExaminer paper](https://arxiv.org/html/2303.07337v2) supplies the clearest positive precedent. Its Table 5 contains curriculum-dependent real errors, including harmful transfers. The [official release](https://github.com/qihao067/PoseExaminer) also reports a practical multi-GPU training recipe. These findings make useful synthetic lessons credible, while leaving our reduced-data 2D recipe unverified.

The proposal's new scientific question is whether a teacher can predict useful lessons for an unfamiliar estimator on unlabeled real video. It must outperform a fixed best lesson, target matching, known synthetic-error selection and nearest-neighbor lookup using the same probe-response database. A model-specific crossover provides a direct test of the proposed capability.

The final review did not find a fatal logical contradiction. It did identify a demanding conjunction: real adaptation must help; lesson selection must improve on strong alternatives; and the teacher must transfer to the held architecture. A JEPA-centered paper additionally needs useful video-context evidence. Including V-JEPA in the pipeline does not establish that contribution.

## Remaining uncertainty

The teacher's synthetic context-to-utility relationship may fail on real video. Six source checkpoints may not span enough learning behaviors. A fixed globally useful lesson or simple response lookup may match the learned selector. Thirty-six final recordings and six deployment collections limit the precision and breadth of the real claim. Rendering, replay acquisition and annotation remain concrete execution dependencies.

These limits are preserved in the final recommendation. The reason to pursue this direction is its supported mechanism and a useful, testable capability, not a promise that a sufficiently elaborate pipeline must produce a paper.
