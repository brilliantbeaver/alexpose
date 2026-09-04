# References and access ledger

**Checked 3 September 2026.** Links favor official papers, repositories, and dataset records. “Public” means the artifact can be downloaded under its stated terms. It does not mean unrestricted commercial use.

## Core models and inspiration

| Work | Verified fact used here | Access consequence |
| --- | --- | --- |
| [S-JEPA paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) and [project page](https://sjepa.github.io/) | Predicts masked joint-time latents and evaluates skeleton action recognition | No author-linked official checkpoint was verified. Use local implementations only. |
| [V-JEPA 2 and 2.1 official repository](https://github.com/facebookresearch/vjepa2) | Public video encoders and predictors; V-JEPA 2.1 adds dense loss, deep supervision, and multimodal tokenizers | Official checkpoints are usable. Lock one checksum and preprocessing path before experiments. |
| [Human-JEPA](https://arxiv.org/abs/2608.21160) | Uses an anchored forecasting objective for human perception and anticipation | Generic human-video anticipation is occupied. No proposal depends on its predictor release. |
| [GoalForce](https://arxiv.org/abs/2601.05848) and [official code](https://github.com/brown-palm/goal-force) | Adapts a frozen video generator with explicit force-conditioned controls | Inspires explicit intervention conditioning. Gait force is never inferred from GAVD. |
| [Masked Visual Actions](https://arxiv.org/abs/2607.19343) and [official code](https://github.com/HadiZayer/masked-visual-actions) | A small public LoRA layer turns partial visual trajectories into forward and inverse controls | Inspires sparse, queryable action channels. Generic partial-motion conditioning is not claimed as new. |
| [ControlNet](https://arxiv.org/abs/2302.05543) | Zero-initialized conditional branches can adapt a frozen generator | Motivates safe small adapters, not a novelty claim. |
| [SleepFM](https://www.nature.com/articles/s41591-025-04133-4) and [official code](https://github.com/zou-group/sleepfm-clinical) | Channel-agnostic, missing-modality alignment supports small frozen-feature heads | Inspires body-region omission tests, not direct model reuse. |
| [GaitDynamics](https://www.nature.com/articles/s41551-025-01565-8) and [official code](https://github.com/stanfordnmbl/GaitDynamics) | Public diffusion and refinement checkpoints model laboratory gait kinematics and kinetics | Generic gait completion and counterfactual editing are occupied. Topology and domain mismatch must be measured. |
| [GaitForeMer](https://arxiv.org/abs/2207.00106) and [official code](https://github.com/markendo/GaitForeMer) | Skeleton future-coordinate pretraining has already been used for Parkinsonian severity transfer | Generic forecast pretraining is occupied. The clinical data are private. |

## Datasets

| Dataset | Verified content | Access and limitation |
| --- | --- | --- |
| [GAVD paper](https://arxiv.org/abs/2407.04190) and [annotations](https://github.com/Rahmyyy/GAVD) | 1,874 in-the-wild sequences, more than 400 subjects, presentation labels and boxes | Annotations and video URLs are public. Videos must be retrieved separately. No participant ID, severity, or affected side. |
| [AMASS](https://amass.is.tue.mpg.de/) | Large unified archive of full-body motion | Already present on HAIC. Use held-identity splits and respect component licenses. |
| [Georgia Tech ground-translation perturbations](https://repository.gatech.edu/entities/publication/73a7c133-6535-4a88-b81e-5c39df5efb3e) | Disturbance magnitude, direction, and onset vary; 3.69 GB dataset | CC BY 4.0. Present release emphasizes pelvis, feet, step placement, and whole-body angular momentum. |
| [Stanford balance-impairment perturbations](https://datadryad.org/dataset/doi:10.5061/dryad.cnp5hqch3) | 10 people, four impairment conditions, four directions, two magnitudes, repeated perturbations, OpenSim outputs | Public 72.93 GB release. Perturbation phase is fixed at 32.5% of the gait cycle. |
| [SAFER-Activities project](https://safer-activities.github.io/) and [dataset card](https://huggingface.co/datasets/SAFER-Activities/SAFER-Activities) | More than 66 hours, 46 participants, 30 frame-level actions, 5,406 simulated falls, look-alike activities, pose and frozen RGB features | Public after accepting contact-sharing terms; 173 GB; CC BY-NC-SA 4.0. Falls are simulated. |
| [GaitIntent article](https://www.nature.com/articles/s41597-026-07799-8) and [Figshare record](https://doi.org/10.6084/m9.figshare.31436731) | 11 people, 13 locomotion modes, eight transitions, 1,430 trials, raw full-body and processed sound-limb IMU signals at 96 Hz | Public data and code. Ten participants are healthy and one has transtibial amputation. |
| [AddBiomechanics](https://addbiomechanics.org/download_data.html) | Standardized OpenSim motions, contacts, torques, and center-of-mass variables | Optional biomechanics bridge. Not needed for any headline gate. |
| [BABEL](https://babel.is.tue.mpg.de/data.html) | Dense action labels aligned with AMASS | Optional transition pretraining. Access agreement applies. |

## Closest collisions

| Work | Occupied claim | Boundary for this portfolio |
| --- | --- | --- |
| [Human Motion Prediction Under Unexpected Perturbation](https://openaccess.thecvf.com/content/CVPR2024/html/Yue_Human_Motion_Prediction_Under_Unexpected_Perturbation_CVPR_2024_paper.html) | Generic reactive-motion prediction with latent differentiable physics | Proposal 1 predicts calibrated recovery summaries under known interventions and requires cross-protocol transfer. |
| [MotionMap](https://arxiv.org/abs/2412.18883) | Multimodal human-motion futures with uncertainty and rare modes | Proposal 4 studies earliest pre-impact recoverability beyond matched trivial kinematics and OOD shifts. |
| [MaskCLR](https://openaccess.thecvf.com/content/CVPR2024/papers/Abdelfattah_MaskCLR_Attention-Guided_Contrastive_Learning_for_Robust_Action_Representation_Learning_CVPR_2024_paper.pdf) | Robust action recognition under joint corruption and pose-estimator shifts | Proposal 3 detects and localizes failure with calibrated abstention. It does not claim corruption-robust classification. |
| [FutureHuman3D](https://openaccess.thecvf.com/content/CVPR2024/html/Diller_FutureHuman3D_Forecasting_Complex_Long-Term_3D_Human_Behavior_from_Video_Observations_CVPR_2024_paper.html) | Joint future action and 3D pose forecasting from video | Proposal 2 measures cross-modal future-innovation sufficiency instead of generic future prediction. |
| [MoML](https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_MoML_Online_Meta_Adaptation_for_3D_Human_Motion_Prediction_CVPR_2024_paper.pdf) | Online personalization of human motion predictors | Test-time adaptation is not a headline in this portfolio. |
| [Action Motifs](https://openaccess.thecvf.com/content/CVPR2026/html/Kinoshita_Action_Motifs_Self-Supervised_Hierarchical_Representation_of_Human_Body_Movements_CVPR_2026_paper.html) | Motion atoms, motifs, and masked latent prediction | Motif discovery was removed. |
| [Zero-Shot Skeleton-Based Action Anticipation](https://arxiv.org/abs/2608.14243) | Zero-shot anticipation from partial skeleton observations | Proposal 7 studies when terrain intent becomes identifiable and how much of the body is sufficient. It does not claim zero-shot anticipation. |
| [Heterogeneous Skeleton-Based Action Representation Learning](https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Heterogeneous_Skeleton-Based_Action_Representation_Learning_CVPR_2025_paper.html) | Learning across different skeleton definitions | Cross-topology transfer was removed as a headline. |

## Local evidence sources

- [HAI internship report](</Users/theodoremui/Downloads/HAI Internship Summer 2025 - Theodore Mui.pdf>)
- [S-JEPA research writeup](</Users/theodoremui/Downloads/S-JEPA Research Writeup - Theodore Mui copy.pdf>)
- [`gavd6/notebooks/foundations`](../../../notebooks/foundations)
- [`gavd6/outputs`](../../../outputs)
- [Previous proposal set 1](../proposals-01/)
- [Previous proposal set 2](../proposals-02/README.md)

## Checkpoint decision

Immediately usable: V-JEPA 2 and 2.1, GaitDynamics, MotionBERT, GoalForce, Masked Visual Actions, and the local S-JEPA checkpoints.

Do not schedule around: an official S-JEPA checkpoint, FoundationGait weights marked as coming soon, the gait health-phenotype foundation model with no public checkpoint, or private GaitForeMer clinical data.
