# Independent review of abstract version 09

19 September 2026. Three independent agent reviewers examined the scientific evidence, contribution relative to prior work, and clarity of the writing. Each first reviewed version 08 without seeing the root author's candidate, then challenged a shared candidate. The science and writing reviewers also checked the final wording. The root author reconciled their suggestions and retained the final editorial responsibility. This was a review of documents, source code and published research; it generated no new experimental observations.

The resulting draft is [abstract-v09-study-scope.md](abstract-v09-study-scope.md). Earlier abstract versions remain unchanged.

**Principal changes and their reasons**

| Reviewer concern | Revision and disposition |
| --- | --- |
| Version 08 makes small-panel diagnostic results dominate the abstract. | The new opening introduces the broader restoration question and implemented method. Preliminary evidence occupies one sentence, with its two-person and one-run limits beside the finding. |
| “Latent prediction,” “readout,” and “training-only spatial calibration” require explanation. | The abstract describes prediction of numerical features, an encoder that converts estimated poses into those features, a separately trained output network, and joint-position corrections fitted only on training data. |
| The reference encoder could be confused with the encoder later used for restoration. | The frozen encoder is explicitly the one processing estimated poses. The reference-feature teacher supplies training targets. |
| A simple correction matching a neural gain does not establish what the neural model learned. | The draft reports descriptive error comparisons. It leaves the mechanism unresolved and makes the contribution of feature learning the question for the planned main study. |
| “Match or exceed” could overstate the offset comparison for ViTPose or imply equivalence. | The wording is “similar or larger reductions,” with an explicit author note that this describes preliminary point estimates and is not a statistical equivalence claim. |
| “One training run per learned method” could include deterministic calibration fits. | The draft specifies one training run per neural method. This denotes the one-seed evidence and includes each method's configured training phases. |
| General “motion timing” could be mistaken for clinically validated gait events. | The planned endpoint is the peak timing of horizontal ankle separation in the image. Author notes explain the synthetic projection, reference support and scaling checks. |
| Planned expansion could be read as completed, or as guaranteed to establish significance. | The larger comparison is explicitly called the planned main study. Its objective is to test benefits and estimate uncertainty across people. |
| A calibration-first intervention is optional in the plan. | The abstract compares against spatial corrections without claiming that every learned model will be retrained on calibrated inputs. |
| The title should state the question without promising successful motion preservation. | The selected title names position accuracy and motion preservation as evaluation objectives. It avoids presenting preservation as an achieved property of JEPA. |

**Title decision**

The science reviewer preferred “Evaluating Predictive Feature Learning for Pose Restoration,” which makes the open outcome clear. The contribution and writing reviewers preferred including both position accuracy and motion preservation so the reader can see why the comparison matters. The chosen title is **Predictive Feature Learning for Pose Restoration: Position Accuracy and Motion Preservation**. Its outcome remains open in the abstract. A title such as “Motion-Preserving JEPA” would require evidence not yet available.

**Where a substantive contribution could emerge**

Predicting skeletal features, temporal pose refinement, and learning motion representations from corrupted observations already have precedents. [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) predicts masked skeletal representations for action recognition. [SmoothNet](https://arxiv.org/abs/2112.13715) learns temporal pose refinement, and [MotionBERT](https://arxiv.org/abs/2210.06551) learns from recovery of 3D motion from noisy partial 2D observations. These primary sources were checked during the review. Their existence rules out treating a generic synthetic denoiser or the use of a JEPA component as an established novelty claim; it does not by itself determine whether the proposed controlled study will be novel.

The intended contribution is an empirical answer about when reference-feature prediction adds useful movement information beyond coordinate supervision and systematic position correction. That answer needs comparisons that isolate the training objective and alignment of reference targets, while checking the behavior of the motion measurements independently of which model wins. The current implementation is a local adaptation for offline 2D restoration; it is not an official reproduction of S-JEPA or evidence of future-motion forecasting.

The proposed larger study should resolve four questions:

1. Does feature prediction improve movement measurements under the same training examples and comparable output networks? Coordinate pretraining, direct restoration and the initialized-encoder control address different explanations. Ordinary and shuffled-pair JEPA, static and temporal controls are also part of the implemented study and should be assessed from their full outcomes.
2. Does the answer persist across independent people and motions, repeated training runs, a held estimator family, and rendering conditions excluded from fitting? Repeated windows and extractors must remain grouped within people when uncertainty is estimated. More seeds do not create more independent people.
3. Do the endpoints measure the intended changes in movement? Establish reference support, check the effect of coordinate scaling, and retain missing and extra peaks alongside timing error. Synthetic hidden-joint references and visible-image scoring describe different evaluation populations.
4. Does any apparent advantage survive adequate training and strong spatial controls? The short initial schedule cannot rule out insufficient training, and a coordinate improvement alone does not establish preserved movement. A replicated absence of benefit could also be informative if it survives these checks and is scoped to the tested setting.

These are research questions, not a promise of positive significance or acceptance. Independent real-video references would be needed for claims about real movement analysis; the present abstract stays within synthetic restoration.

**Local evidence used**

The [post-run analysis](../results/postrun-analysis-20260919/README.md) supports the initial coordinate/displacement statements and records the limited timing coverage. The implementation separates [representation training and frozen-encoder readout fitting](../../../../src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py) and enforces [training-only calibration fitting](../../../../scripts/research_directions/synthetic_training_v2/diagnostics/calibration.py). The [one-week plan](../research/iclr-one-week-plan-20260919.md) specifies the proposed expansion. The downloaded diagnostics do not contain every original neural prediction array, so this review does not claim a fresh replay of all source inference.
