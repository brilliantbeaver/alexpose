# Literature and claim ledger

Reviewed **18 September 2026**, with execution date checked using the clock tool. This is a bounded primary-source review, not an exhaustive priority search. Methods/evaluation were read for SmoothNet, PoseBERT, DeciWatch, S-JEPA, GaitForeMer, MotionBERT and PoseSyn. SynSP's publisher record and indexed primary-paper excerpts were accessible; direct PDF retrieval failed. No external model binaries were downloaded or executed. A working repository link is not verification of checkpoints, data access or compatibility with Torch 2.6.0+cu124.

The defensible research question is whether paired rendering interventions improve **same-camera 2D restoration while preserving independently referenced motion variation**, and whether latent prediction improves that tradeoff beyond equally informed coordinate learning. Generic temporal correction, masked MoCap pretraining, estimator transfer and accuracy–smoothness balancing already have direct precedents. An identical experiment was not found in this search; that does not establish priority.

## Eight closest records

### 1. SmoothNet

**Title/authors:** *SmoothNet: A Plug-and-Play Network for Refining Human Poses in Videos*, Ailing Zeng, Lei Yang, Xuan Ju, Jiefeng Li, Jianyi Wang, Qiang Xu. **Version/status:** arXiv 2112.13715v2, 21 July 2022; ECCV 2022. [Version record](https://arxiv.org/abs/2112.13715), [paper and supplement](https://arxiv.org/pdf/2112.13715).

**Method/supervision:** estimated 2D/3D coordinates or SMPL rotations; coordinate-wise shared temporal MLP; supervised position and reference-acceleration losses. Motion-aware variant also processes velocity and acceleration. **Evaluation:** FCN–Human3.6M, SPIN–3DPW and VIBE–AIST++ training; transfer across these and MPI-INF-3DHP/MuPoTS-3D. Human3.6M uses subjects 1/5/6/7/8 for training, 9/11 for testing. Metrics include MPJPE, PA-MPJPE, acceleration error and difficult-frame errors.

**Supported claim/limit:** transfer across estimators/modalities/datasets and synthetic-noise experiments precede this study. Better acceleration error does not guarantee better coordinates under distribution shift; clinical gait preservation is not demonstrated. Use a source-trained body-12 adaptation, labeled accurately.

**Artifacts/access/license:** [official code](https://github.com/cure-lab/SmoothNet) supplies training/evaluation and 8/16/32/64-frame checkpoint links. The Drive landing page resolves; binary contents unverified. [LICENSE](https://raw.githubusercontent.com/cure-lab/SmoothNet/main/LICENSE) is Apache-2.0, while README says non-commercial scientific research. Record that discrepancy before importing code; third-party assets have separate terms.

### 2. PoseBERT

**Title/authors:** *PoseBERT: A Generic Transformer Module for Temporal 3D Human Modeling*, Fabien Baradel, Romain Brégier, Thibault Groueix, Philippe Weinzaepfel, Yannis Kalantidis, Grégory Rogez. **Version/status:** arXiv 2208.10211v2, 19 October 2022; record states accepted to TPAMI 2022. [Record](https://arxiv.org/abs/2208.10211), [full text](https://arxiv.org/html/2208.10211v2).

**Method/supervision:** masked/noise-corrupted 3D keypoints or SMPL/MANO rotations, optionally projected 2D inputs; clean MoCap pose/translation reconstruction. **Evaluation:** approximately 11,000 AMASS training sequences at 30 Hz; body evaluation on 3DPW/MPI-INF-3DHP test sets, MuPoTS-3D and AIST. Metrics include MPJPE/PA-MPJPE, mesh error, PCK and acceleration error; hand experiments are separate.

**Supported claim/limit:** masked MoCap training already supports refinement, completion and forecasting. The paper reports occasional deterioration of already-good estimates. Its bone/orientation normalization and 3D output require separate justification for functional 2D gait measurement.

**Artifacts/access/license:** [official repository](https://github.com/naver/posebert), CC BY-NC-SA 4.0, provides demo and model archive links; archive retrieval unverified. README describes the 3DV 2021 predecessor, not complete TPAMI training reproduction. Separate SMPL assets are required. Do not copy its older Torch installation recipe into HAIC.

### 3. DeciWatch

**Title/authors:** *DeciWatch: A Simple Baseline for 10× Efficient 2D and 3D Pose Estimation*, Ailing Zeng, Xuan Ju, Lei Yang, Ruiyuan Gao, Xizhou Zhu, Bo Dai, Qiang Xu. **Version/status:** arXiv 2203.08713v2, 20 July 2022; ECCV 2022. [Record](https://arxiv.org/abs/2203.08713), [proceedings paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136650597.pdf).

**Method/supervision:** sparse estimated 2D/3D poses or body rotations; transformer DenoiseNet and RecoverNet; supervised L1 sampled-pose and full-sequence losses. **Evaluation:** Sub-JHMDB, Human3.6M, 3DPW and AIST++; benchmark train/test detections; complete split assignments not independently verified. PCK, MPJPE, acceleration error, FLOPs and inference time.

**Supported claim/limit:** noisy-coordinate denoising and recovery are established. Its efficiency comes substantially from skipping image-estimator calls; that is different from an equal-observation restoration comparison. No unusual-gait preservation claim follows.

**Artifacts/access/license:** [official code](https://github.com/cure-lab/DeciWatch), data and checkpoint links verified. A 2D checkpoint folder landing page resolves; binary contents unverified. [LICENSE](https://raw.githubusercontent.com/cure-lab/DeciWatch/main/LICENSE) is Apache-2.0, while README says non-commercial scientific research; record the discrepancy.

### 4. S-JEPA

**Title/authors:** *S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition*, Mohamed Abdelfattah, Alexandre Alahi. **Version/status:** ECCV 2024 chapter, published 23 November 2024; DOI 10.1007/978-3-031-73411-3_21. [Publisher record](https://link.springer.com/chapter/10.1007/978-3-031-73411-3_21), [paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf).

**Method/supervision:** 3D skeletons, motion-aware masking, EMA target encoder, centered/sharpened latent distributions and cross-entropy; rotations/translations/flips augment views. **Evaluation:** NTU60 cross-subject/cross-view, NTU120 cross-subject/cross-setup and subject-separated PKU-MMD. Action-recognition accuracy under linear evaluation, fine-tuning, limited labels and transfer.

**Supported claim/limit:** skeleton JEPA and its masking/stability mechanisms already exist. Recognition accuracy does not establish preservation of timing, anatomical side or precise coordinates. Aligned clean synthetic targets add privileged supervision to this study.

**Artifacts/access/license:** [author project](https://sjepa.github.io/) accessible. No usable official code/checkpoint link found on the inspected page; artifact license unverified. Local models are adaptations, not an official S-JEPA release.

### 5. GaitForeMer

**Title/authors:** *GaitForeMer: Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for Few-Shot Gait Impairment Severity Estimation*, Mark Endo, Kathleen L. Poston, Edith V. Sullivan, Li Fei-Fei, Kilian M. Pohl, Ehsan Adeli. **Version/status:** arXiv 2207.00106v1, 30 June 2022; MICCAI 2022, DOI 10.1007/978-3-031-16452-1_13. [Paper](https://arxiv.org/pdf/2207.00106), [conference record](https://conferences.miccai.org/2022/papers/230-Paper0398.html).

**Method/supervision:** 3D forecasting plus classification; NTU activity labels supervise a parallel pretraining branch. Clinical VIBE-derived skeletons support MDS-UPDRS gait classification. **Evaluation:** NTU RGB+D60 pretraining, private 54-participant clinical cohort, subject-level leave-one-out evaluation; F1/precision/recall, best reported F1 0.76.

**Supported claim/limit:** forecasting pretraining for gait-severity estimation is established, but this implementation is not wholly label-free. Clinical severity is neither pose accuracy nor identity recognition.

**Artifacts/access/license:** [official code](https://github.com/markendo/GaitForeMer), GPL-3.0. Weight link advertised; Box retrieval failed. Clinical data are not a verified available resource.

### 6. MotionBERT

**Title/authors:** *MotionBERT: A Unified Perspective on Learning Human Motion Representations*, Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, Yizhou Wang. **Version/status:** arXiv 2210.06551v5, 14 August 2023; ICCV 2023. [Full text/version](https://arxiv.org/html/2210.06551v5), [proceedings record](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html).

**Method/supervision:** DSTformer recovers 3D motion from corrupted/partial 2D skeletons, using heterogeneous coordinate supervision. **Evaluation:** projected Human3.6M/AMASS MoCap plus PoseTrack and InstaVariety pretraining; Human3.6M subjects 1/5/6/7/8 train and 9/11 test. Pose error, action recognition and mesh-recovery evaluations.

**Supported claim/limit:** corrupted 2D observations, MoCap targets and transferable coordinate pretraining precede this proposal. Three-dimensional lifting does not directly answer same-camera 2D restoration or gait asymmetry preservation.

**Artifacts/access/license:** [official code](https://github.com/Walter0807/MotionBERT), Apache-2.0; model/documentation links present, binaries unverified. Standard input expects 17 joints and three channels; body-12 needs a justified adapter, not silent padding.

### 7. SynSP

**Title/authors:** *SynSP: Synergy of Smoothness and Precision in Pose Sequences Refinement*, Tao Wang, Lei Jin, Zheng Wang, Jianshu Li, Liang Li, Fang Zhao, Yu Cheng, Li Yuan, Li Zhou, Junliang Xing, Jian Zhao. **Version/status:** CVPR, June 2024, pp. 1824–1833; separate DOI/arXiv revision not verified. [CVF record](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.html), [primary paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_SynSP_Synergy_of_Smoothness_and_Precision_in_Pose_Sequences_Refinement_CVPR_2024_paper.pdf).

**Method/supervision:** supervised 2D/3D/SMPL refinement; disagreement between precision/smoothness branches supplies quality cues, with optional synchronized views. **Evaluation:** Human3.6M, AIST++, 3DPW and CMU-MoCap; MPJPE, PA-MPJPE and acceleration error. Exact splits unverified.

**Supported claim/limit:** quality-aware restoration and the accuracy–smoothness tradeoff are explicit preceding contributions. Multi-view results use extra information. Direct full-PDF access failed; do not borrow an exact objective from this abbreviated review.

**Artifacts/access/license:** [official repository](https://github.com/InvertedForest/SynSP) has training/evaluation code but says under construction; checkpoints unverified. GitHub identifies Apache-2.0 while README says non-commercial; direct license retrieval failed.

### 8. PoseSyn

**Title/authors:** *PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data*, ChangHee Yang, Hyeonseop Song, Seokhun Choi, Seungwoo Lee, Jaechul Kim, Hoseok Do. **Version/status:** arXiv 2503.13025v1, 17 March 2025; ICCV 2025. [Full text](https://arxiv.org/html/2503.13025v1), [proceedings paper](https://openaccess.thecvf.com/content/ICCV2025/papers/Yang_PoseSyn_Synthesizing_Diverse_3D_Pose_Data_from_In-the-Wild_2D_Data_ICCV_2025_paper.pdf).

**Method/supervision:** identify difficult real 2D poses, synthesize neighboring 3D motions and image–pose pairs, adapt image-to-3D estimators; uses 2D/pseudo-3D/generated targets. **Evaluation:** initial training on Human3.6M/MuCo/MPI-INF-3DHP/MPII/COCO; synthesis from MPII; evaluation on 3DPW, EMDB, CMU, HuMMan, LSPET and JHMDB. MPJPE, PA-MPJPE, PCKh; exact identity-overlap accounting unverified.

**Supported claim/limit:** estimator-specific hard-example synthesis already exists; it does not establish paired rendering-intervention JEPA or gait preservation. **Artifacts/access/license:** [author project](https://seokhunchoi.github.io/PoseSyn/) links papers; no official downloadable code/checkpoint found there. Artifact licenses unverified.

## Citation trail and current contextual references

Backward from PoseBERT: [Skeletor, CVPR Workshops 2021](https://openaccess.thecvf.com/content/CVPR2021W/ChaLearn/html/Jiang_Skeletor_Skeletal_Transformers_for_Robust_Body-Pose_Estimation_CVPRW_2021_paper.html) already corrects noisy 3D skeletons; [Leveraging MoCap Data for Human Mesh Recovery, 3DV 2021](https://arxiv.org/abs/2110.09243) is PoseBERT's direct predecessor. Backward from DeciWatch: [Single-Shot Motion Completion with Transformer](https://arxiv.org/abs/2103.00776) distinguishes reliable-keyframe completion from noisy-estimator restoration. SmoothNet's supplement includes comparisons to RefineNet/filters and synthetic noise.

Forward: SynSP cites SmoothNet/DeciWatch; MotionBERT cites PoseBERT. [PS-Mamba, Dong and Lee, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Dong_PS-Mamba_Spatial-Temporal_Graph_Mamba_for_Pose_Sequence_Refinement_ICCV_2025_paper.pdf) compares spatial-temporal graph/state-space refinement against SmoothNet/SynSP. Its [official code](https://github.com/donghaoye/ps-mamba) links data/weights, untested; Apache-2.0 license file/non-commercial README discrepancy persists. [SynSP++ author code](https://github.com/InvertedForest/SynSP2) describes similar-pose retrieval; publisher record/full paper unverified. These are literature boundaries, not mandatory first-milestone dependencies.

| Lead | Verified source/status and boundary | Artifact access |
| --- | --- | --- |
| V-JEPA | [2404.08471v1](https://arxiv.org/abs/2404.08471), video feature prediction; recognition is not coordinate fidelity. | [Official code/models](https://github.com/facebookresearch/jepa) linked; binaries untested. |
| V-JEPA 2 | [2506.09985v1, 11 June 2025](https://arxiv.org/abs/2506.09985); robot action conditioning is a distinct post-training task. | [Official repository](https://github.com/facebookresearch/vjepa2) provides model links; runtime unverified. |
| V-JEPA 2.1 | [2603.14482v3, 11 June 2026](https://arxiv.org/abs/2603.14482); dense/intermediate-layer objectives, venue unverified. | Official V-JEPA 2 repository lists 80M ViT-B/16 at 384 pixels; binary untested. |
| LeJEPA | [2511.08544v3, 14 November 2025](https://arxiv.org/abs/2511.08544), Balestriero/LeCun; SIGReg is no preservation guarantee. | [Code](https://github.com/galilai-group/lejepa), CC BY-NC 4.0 verified; optional comparator. |
| seq-JEPA | [NeurIPS 2025 proceedings](https://proceedings.neurips.cc/paper_files/paper/2025/file/2f63d2963526bdd9ff1b8bcc2dc9905a-Paper-Conference.pdf); transformation-conditioned invariant/equivariant learning. | [Code](https://github.com/hafezgh/seq-jepa), MIT, model links present; binaries untested. |
| FSGait | [ACCV 2024](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html), Duan/Wan/Zhao; normal-gait reconstruction/prediction for anomaly detection. | Code/weights/license not verified; author lab lists paper/BibTeX. |
| Skeleton SSL scaling | [AAAI, 14 March 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37340), DOI 10.1609/aaai.v40i5.37340, Cosma/Catruna/Radoi. Identity objective. | [Full text](https://arxiv.org/html/2504.07598v1) uses private large-scale pretraining data; not a downloadable corpus assumption. |
| GaitPT | [Paper](https://arxiv.org/abs/2308.10623), FG 2024, DOI 10.1109/FG59268.2024.10581947; identity recognition. | [Code/weight links](https://github.com/AndyCatruna/GaitPT), CC BY-NC-ND 4.0; weights untested. |
| H-MoRe | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Huang_H-MoRe_Learning_Human-centric_Motion_Representation_for_Action_Analysis_CVPR_2025_paper.pdf), Huang/Liu/Kong; human world/local flow. | [Author page](https://zbhuang.com/h-more) says code/models will be released; advertised GitHub retrieval failed; license unverified. |
| SM-SGE | [Paper](https://arxiv.org/abs/2107.01903), ACM MM 2021; skeleton re-identification. | [MIT code](https://github.com/Kali-Hac/SM-SGE), processed IAS-Lab/KGBD links; KS20 requires license/request. |
| GaitJEPA | [University listing](https://www.uco.es/investiga/grupos/ava/publicaciones/) and [author IJCB 2026 acceptance announcement](https://www.linkedin.com/posts/mjmarin_avagroup-ijcb2026-gaitrecognition-activity-7483034746906669056-FcEG); silhouette identity recognition. | Publisher proceedings/code/weights/license unverified. Avoid first-JEPA-for-gait claims. |
| Task2Sim | [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Mishra_Task2Sim_Towards_Effective_Pre-Training_and_Transfer_From_Synthetic_Data_CVPR_2022_paper.pdf); task-conditioned simulation selection. | [Official code](https://github.com/samarth4149/task2sim) verified; full simulator assets/license unverified. |
| PoseExaminer | [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Liu_PoseExaminer_Automated_Testing_of_Out-of-Distribution_Robustness_in_Human_Pose_and_CVPR_2023_paper.html); failure search and corrective fine-tuning. | Author code URL identified in paper; artifact usability/license not checked. |
| MM-ACL | [PMLR/CoLLAs 2023](https://proceedings.mlr.press/v232/xu23a.html); predicts cross-task learning improvement conditioned on student status. | Code/weights/license unverified. |
| CARE-PD | [NeurIPS 2025 D&B](https://proceedings.neurips.cc/paper_files/paper/2025/file/bedc73979a95be7727af0c9a99c675ce-Paper-Datasets_and_Benchmarks_Track.pdf); derived SMPL motions from both RGB and MoCap. | [MIT benchmark code](https://github.com/TaatiTeam/CARE-PD), data links; raw clinical RGB release not established. Code license does not establish data license. |
| GAITGen | [Official WACV 2026 repository](https://github.com/TaatiTeam/GAITGen); motion/pathology generation; PD-GaM overlaps CARE-PD. Direct publisher retrieval failed this review. | MIT code; [raw README](https://raw.githubusercontent.com/TaatiTeam/GAITGen/main/README.md) explicitly leaves checkpoints/preprocessed representations unreleased. |
| DiffuseGaitNet | [TNSRE 2025 record](https://pubmed.ncbi.nlm.nih.gov/40658580/), DOI 10.1109/TNSRE.2025.3589074; clinical-feature-conditioned augmentation. | [MIT code](https://github.com/arshakRz/DiffuseGaitNet), checkpoint directory present, untested; original data explicitly private. |
| Factorized Latent Dynamics | [2605.17165v1, 16 May 2026](https://arxiv.org/abs/2605.17165), Santosh Premi; auxiliary-objective tradeoffs. | Code URL advertised; artifact usability/license and reviewed venue unverified. |
| Graph-JEPA information loss | [2608.20516v1, 20 August 2026](https://arxiv.org/abs/2608.20516), Rabby/Auer; document-graph diagnostics. | Preprint; methodological analogy only, not gait evidence or this pilot's collapse diagnosis. |

## Claim-to-source ledger and falsification

| Claim under consideration | Evidence/verdict | Smallest adequate study correction |
| --- | --- | --- |
| Estimator-independent temporal correction is new | Falsified broadly by SmoothNet/PoseBERT/DeciWatch. | Name the controlled intervention and functional preservation question. |
| Masked MoCap pretraining or corrupted-2D/clean-motion learning is new | Falsified broadly by PoseBERT/MotionBERT. | Compare matched clean-coordinate representation learning. |
| Latent prediction intrinsically preserves gait | Unsupported by recognition-focused S-JEPA. | Measure common coordinate and reference-based motion endpoints. |
| Paired JEPA beating noisy-target JEPA isolates its objective | Invalid: target quality changes. | Same clean pairs, masks, readout capacity, tuning and declared resource matching. |
| Lower acceleration error establishes preservation | Invalid inference; SmoothNet/PoseBERT separate smoothness from accuracy. | Test amplitude, anatomical side and timing against references, including accurate inputs. |
| Adaptive synthetic teaching is new | Task2Sim/PoseExaminer/PoseSyn/MM-ACL overlap. | Require new headroom and matched scene/state/source-progress controls before personalization. |
| First JEPA for gait | Unsafe given GaitJEPA author evidence. | Avoid priority wording; distinguish silhouette identity from pose measurement. |
| Current work establishes a novel effective method | Not established by implementation or literature search. | Keep empirical gates pending until supported artifact-derived comparisons exist. |

The strongest competing explanation for any future JEPA gain is **privileged clean targets, exposure or smoothing**, rather than latent prediction. The clean-coordinate representation control, practical temporal MLP, initialized readout and unchanged-input baseline directly challenge that explanation. Negative gates and direct-denoiser success must remain reportable independently.
