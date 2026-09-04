# Reference and access ledger

**Checked 3 September 2026.** This ledger records the evidence used, the closest conceptual collisions, and whether a proposal depends on a downloadable model. Detailed reading notes are in [`../brainstorm/lit`](../brainstorm/lit).

## Core data and model sources

| Work | Fact used | Access consequence |
| --- | --- | --- |
| [GAVD](https://arxiv.org/abs/2407.04190) and [annotations](https://github.com/Rahmyyy/GAVD) | 1,874 annotated in-the-wild gait sequences; strong view effects; binary recognition already in the 92 to 94 percent range | Public annotations and source links support the audit. Local manifests and HAIC videos define the experiment. The release has no participant ID, severity, affected side, or official source-held split. |
| [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) and [project](https://sjepa.github.io/) | Masked latent prediction over joint-time tokens | No author-linked official checkpoint was verified. Every proposal uses the repository checkpoint and states its limits. |
| [GaitForeMer](https://arxiv.org/abs/2207.00106) and [code](https://github.com/markendo/GaitForeMer) | Forecasting pretraining has already been used for few-shot Parkinsonian severity estimation | Code is public, but the clinical data are private and the linked checkpoint was unavailable when checked. Generic forecast transfer is not a new claim. |
| [GaitDynamics](https://www.nature.com/articles/s41551-025-01565-8) and [official code](https://github.com/stanfordnmbl/GaitDynamics) | A large public diffusion prior can complete laboratory gait kinematics and predict forces | Code and trained diffusion and refinement models are public. Predicted forces on GAVD would remain model outputs, never measurements. The seven proposals do not depend on this bridge. |
| [GaitEncoder](https://www.medrxiv.org/content/10.64898/2026.07.07.26357479v1) and [official code](https://github.com/rdmagruder/GaitEncoder) | A public compact clinical gait latent and normative deviation score now exist | Rules out static distance from normal as a main contribution. Laboratory stride-normalized inputs are not a direct GAVD route. |
| [CARE-PD](https://arxiv.org/abs/2510.04312) and [project](https://neurips2025.care-pd.ca/) | Public multi-site SMPL gait with Parkinsonian severity benchmarks | Optional external validation for dose axes. It does not add severity labels to GAVD. |
| [GAITGen](https://openaccess.thecvf.com/content/WACV2026/html/Adeli_GAITGen_Disentangled_Motion-Pathology_Impaired_Gait_Generative_Model_--_Bringing_Motion_WACV_2026_paper.html) | Severity-conditioned impaired-gait generation and motion-pathology separation are occupied | Not required. No public trained checkpoint was verified. |
| [Zero-shot gait classification with diffusion models](https://openreview.net/forum?id=L5xyzjMCwd) | Motion-diffusion denoising error has already been used as a Parkinsonian gait score | Rules out a generic scalar surprise proposal. |

## Frozen adaptation and world-model context

| Work | Fact used | Boundary here |
| --- | --- | --- |
| [ControlNet](https://arxiv.org/abs/2302.05543) and [official code](https://github.com/lllyasviel/ControlNet) | A frozen base with a zero-initialized residual route preserves the initial function | Motivates SourceSwap's rank-8 adapter. Image weights are not used. |
| [Goal Force](https://arxiv.org/abs/2601.05848) and [official code](https://github.com/brown-palm/goal-force) | Explicit intervention channels can make a frozen video prior queryable | Public code, data, and adapter establish the ambition level. Relative robot forces are inspiration, not gait evidence. |
| [Masked Visual Actions](https://arxiv.org/abs/2607.19343) and [project](https://masked-visual-actions.github.io/) | Dense visual controls adapt Wan 2.2 with LoRA for forward and inverse robot queries | Its 15 hours refers to video duration. Reported training used eight H200 GPUs for about four days, so this portfolio transfers the control-interface idea only. |
| [ViA](https://arxiv.org/abs/2209.00065) | Learns view-invariant 2D and 3D skeleton action features through motion retargeting | View-invariant skeleton representation is occupied. SourceSwap must validate paired source interventions and semantic retention, not claim generic view invariance. |
| [RobustGait](https://arxiv.org/abs/2511.13065) | Audits RGB, environmental, temporal, occlusion, and silhouette-extractor corruption in gait recognition | Gait corruption benchmarking is occupied. It motivates applying source operators before pose extraction and auditing their propagation. |
| [Causality-Driven Audits of Model Robustness](https://openaccess.thecvf.com/content/WACV2026/html/Drenkow_Causality-Driven_Audits_of_Model_Robustness_WACV_2026_paper.html) | Uses explicit imaging-process factors to audit model robustness under compound distortions | Factorized acquisition audits are occupied. SourceSwap's distinction is paired identical motion plus a separate semantic-edit axis. |
| [V-JEPA 2](https://arxiv.org/abs/2506.09985) and [official checkpoints](https://github.com/facebookresearch/vjepa2) | Public frozen RGB predictive representations are available | Admissible optional baseline. No proposal requires RGB video pretraining or a V-JEPA 2 download. |
| [SleepFM](https://www.nature.com/articles/s41591-025-04133-4) and [official code](https://github.com/zou-group/sleepfm-clinical) | Large frozen representations can support small clinical heads and missing-input alignment | Motivates the small-adapter pattern, not a gait model dependency. |

## Past-only prediction

| Work | Closest idea | Distinction |
| --- | --- | --- |
| [Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) | Learn representations by future latent prediction | Proposal 2 does not claim future prediction as new. It measures surplus beyond periodic gait and architecture placebos. |
| [Dense Predictive Coding](https://openaccess.thecvf.com/content_ICCVW_2019/html/HVU/Han_Video_Representation_Learning_by_Dense_Predictive_Coding_ICCVW_2019_paper.html) | Predict future video feature blocks | Supplies a direct forecasting reference. The current local S-JEPA was trained with bidirectional masks. |
| [CF-JEPA](https://arxiv.org/abs/2606.07031) | Trains multi-horizon forward prediction for time-series representations | Forward JEPA training is occupied. Proposal 2 audits whether a bidirectionally trained gait encoder contains surplus forward information. |
| [Latent Video Prediction Learns Better World Models](https://arxiv.org/abs/2605.15618) | Audits temporal direction, corruption, occlusion, and fine-grained cues in latent-prediction video models | Arrow-of-time evidence is occupied. Proposal 2 adds gait-periodicity ceilings and an equal-capacity raw-past target. |
| [Diverse Human Motion Prediction with STARS](https://arxiv.org/abs/2302.04860) | Accurate and diverse pose futures | Used as context and a possible baseline, not as proof that local S-JEPA predicts beyond gait periodicity. |

## Adaptive measurement

| Work | Closest idea | Distinction |
| --- | --- | --- |
| [EDDI](https://proceedings.mlr.press/v97/ma19c.html) | Choose costly variables by expected information gain | Proposal 3 prices full-body four-frame detector passes and evaluates downstream utility with a complete gait action table. |
| [Active feature acquisition via explainability-driven ranking](https://proceedings.mlr.press/v267/guney25a.html) | Instance-specific feature acquisition for medical decisions | The new claim is not active acquisition itself. It is S-JEPA-guided frame-block reprocessing and exact oracle regret. |
| [Active high-resolution pose refinement](https://doi.org/10.1016/j.eswa.2025.126550) | Select human-pose regions for high-resolution refinement | Rules out claiming pose refinement or region selection as new. Full-body frame blocks are used because a normal detector returns all joints together. |
| [MMPose RTMPose whole-body model index](https://github.com/open-mmlab/mmpose/blob/main/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose_coco-wholebody.yml) and [releases](https://github.com/open-mmlab/mmpose/releases) | Lists directly downloadable RTMPose checkpoints with body and foot keypoints; MMPose 1.3.2 is tagged at commit `5408bc7` | Proposal 3 locks RTMPose-L 384 by 288 as its stronger measurement route and records the downloaded weight's SHA-256 before inference. |

## Continuous axes and cycle innovation

| Work | Fact used | Distinction |
| --- | --- | --- |
| [Gait variability review](https://pmc.ncbi.nlm.nih.gov/articles/PMC1185560/) | Stride-to-stride fluctuations can contain clinically relevant temporal structure | Proposal 5 estimates short-video predictive innovation, not a long-range physiological variability biomarker. |
| [Parkinsonian gait variability meta-analysis](https://pubmed.ncbi.nlm.nih.gov/27445759/) | Several gait variability measures differ in Parkinsonian cohorts | Motivates the clinical question but does not validate GAVD phase-local residuals. |
| [Action Motifs](https://openaccess.thecvf.com/content/CVPR2026/html/Kinoshita_Action_Motifs_Self-Supervised_Hierarchical_Representation_of_Human_Body_Movements_CVPR_2026_paper.html) | Learns motion atoms and recurring motifs using masked latent prediction | Directly killed the draft motif dictionary. Cycle innovation discovers no segments or vocabulary. |
| [GenGait](https://arxiv.org/abs/2604.01997) | Uses a normative masked Transformer for joint-level gait anomaly localization and kinematic correction | Joint anomaly localization is occupied. Proposal 5 is restricted to self-referenced adjacent-cycle onset residuals with boundary-matched nulls. |
| [GAITGen](https://openaccess.thecvf.com/content/WACV2026/html/Adeli_GAITGen_Disentangled_Motion-Pathology_Impaired_Gait_Generative_Model_--_Bringing_Motion_WACV_2026_paper.html) | Conditions generated gait on impairment severity | Dose axes learn calibrated motion-expression amounts from controlled edits. They do not generate pathology or invent GAVD severity. |

## Unsigned predictive asymmetry

| Work | Closest idea | Distinction |
| --- | --- | --- |
| [Glide-reflection symmetry](https://openaccess.thecvf.com/content_cvpr_2017_workshops/w7/html/Wang_Measuring_Glide-Reflection_Symmetry_CVPR_2017_paper.html) | Continuous symmetry scores compare counterpart motion under a half-cycle shift | Mandatory raw baseline. Proposal 6 survives only if predictive-error parity adds localization beyond it. |
| [Patterson et al.](https://pubmed.ncbi.nlm.nih.gov/19932621/) | Reviews temporal gait symmetry measures after stroke | Supports careful symmetry terminology, not affected-side inference on GAVD. |
| [Fukino and Tachibana](https://arxiv.org/abs/2505.10869) | Measures gait asymmetry through inter-limb coordination | Makes generic coordination-based asymmetry an occupied contribution. Proposal 6 must add predictive-error localization or stop. |
| [VisionMD-Gait](https://www.nature.com/articles/s41598-025-34912-5) | Mirrors and reprocesses video, restores limb identity, and averages outputs to suppress viewpoint-driven pose bias while preserving physiological asymmetry | Supplies a direct mirrored-processing control. Reflection robustness itself is not new. |
| [Chirality Nets](https://arxiv.org/abs/1911.00029) | Builds reflection and left-right joint-swap equivariance into pose networks | Even-and-odd decomposition is not claimed as new. The reflection-equivariant local checkpoint is a control. |
| [Repository laterality proposal](../../latent-laterality/proposal.md) | Already defines even and odd channels and tests a correspondence mechanism | The executed probability mechanism was matched by fixed 50/50 uncertainty. Proposal 6 is a deliberately weaker unsigned error assay. |

## Structure-surrogate ladder

| Work | Closest idea | Distinction |
| --- | --- | --- |
| [Schreiber and Schmitz](https://arxiv.org/abs/chao-dyn/9909041) | Iterative surrogates preserve a marginal distribution and autocorrelation | Supplies the univariate IAAFT basis. |
| [Prichard and Theiler](https://doi.org/10.1103/PhysRevLett.73.951) | Multivariate phase-randomized surrogates preserve cross-correlation structure | Supplies the cross-series null logic. |
| [Keylock](https://doi.org/10.1029/2012WR011923) | Multivariate IAAFT preserves cross-correlation and higher-order properties better than simple independent randomization | Supplies the implementation route for rungs 2 and 3. |
| [Dingwell and Cusumano](https://pubmed.ncbi.nlm.nih.gov/20605097/) | Uses several surrogate gait time series, including cross-correlated stride-length and stride-time nulls, to test control interpretations | Direct gait precedent. Proposal 7 cannot claim that surrogate families first reveal required gait structure. |
| [Liégeois, Yeo, and Van De Ville](https://doi.org/10.1016/j.neuroimage.2021.118518) | Explains null models that preserve progressively richer temporal properties | Progressive null attribution is established. The proposed distinction is the anatomy-nested JEPA audit and controlled gait calibration. |
| [Progressive surrogate attribution](https://arxiv.org/abs/2606.11415) | Uses phase, IAAFT, and block-shuffle surrogates to attribute time-series predictability | Closest recent collision. Proposal 7 adds a source-only rung, within-leg versus cross-leg hierarchy, exact controlled gait tasks, S-JEPA placebos, and source-held GAVD accounting. |
| [STC-Net](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Leveraging_Spatio-Temporal_Dependency_for_Skeleton-Based_Action_Recognition_ICCV_2023_paper.pdf) | Skeleton models exploit spatial and temporal joint dependencies | A model-design reference. The ladder audits which dependency order is accessible rather than proposing another graph block. |

## Ideas removed by collision or identifiability review

| Removed idea | Decisive reason |
| --- | --- |
| Smallest sufficient and necessary witness | [Sufficient Input Subsets](https://proceedings.mlr.press/v89/carter19a.html) and [TimePNS](https://arxiv.org/abs/2607.21573) already occupy the core object. |
| Predictive motif dictionary | Action Motifs is a direct 2026 collision. |
| Transformation-stable conformal sets | Transformation smoothing, test-time augmentation, canonicalized conformal prediction, equivariantized conformal prediction, and worst-case robust conformal sets already occupy the method family. |
| Generic RGB, 2D, and 3D triangulation | Repeats earlier repository depth and cross-prior safety ideas, and the draft experiment lacked a real monocular-lift failure. |
| Predictive repertoire dimension | The draft benchmark paired identical visible pasts with different target ranks, making rank unidentifiable. [Gait intrinsic dimension](https://doi.org/10.3390/e27040447), [predictive dimensionality](https://doi.org/10.1073/pnas.2021860119), and [future-feature effective rank](https://arxiv.org/abs/2607.26657) are also occupied. |

Representative conformal collision sources are [Franco et al.](https://proceedings.mlr.press/v244/franco24a.html), [test-time augmentation conformal prediction](https://openaccess.thecvf.com/content/CVPR2025/papers/Shanmugam_Test-time_Augmentation_Improves_Efficiency_in_Conformal_Prediction_CVPR_2025_paper.pdf), [CP²](https://proceedings.mlr.press/v286/linden25a.html), [equivariantized conformal prediction](https://arxiv.org/abs/2602.03986), and [worst-case robust conformal prediction](https://proceedings.mlr.press/v235/h-zargarbashi24a.html).

## Artifact rule

The two-week program may use only:

- the local S-JEPA checkpoint with recorded state and data hashes;
- public AMASS data already present under the repository's existing access agreement;
- GAVD source videos already cached on the user's HAIC scratch space;
- small heads or rank-8 adapters trained inside each outer source fold;
- optional directly downloadable public baselines whose exact version and checksum are recorded before evaluation.

No proposal depends on the unavailable GaitForeMer checkpoint, private GaitForeMer clinical cohort, unreleased GAITGen model, force plates, new clinical collection, or a large-model fine-tuning run.
