# Recent clinical gait and anomaly work

Fetch date: 2026-09-03.

## A Gait Foundation Model Predicts Multi-System Health Phenotypes from 3D Skeletal Motion

**Problem.** The paper asks whether a motion representation can expose health information beyond conventional gait summaries. It uses gait broadly to include five locomotor, postural, and functional tasks.

**Method.** A 3.2 million parameter dual-stream transformer processes 30-second sequences of 26 Azure Kinect joints at 30 Hz. Self-supervision masks about 70 percent of joints and frames, adds pose noise, and reconstructs position and velocity. Frozen features are pooled by anatomical group and then entered into ridge regression or logistic regression. The study contains 17,589 sequences and 351 hours from 3,414 adults. Training used one L40S GPU, but the paper does not report wall time or epochs.

**Headline evidence.** Learned features predicted age at Pearson r=0.69, body mass index at r=0.90, and visceral adipose tissue at r=0.82. They significantly predicted 1,980 of 3,210 targets. Nested subject-level evaluation used five folds and 15 seeds. These results show that long motion traces retain distributed health signals, but they do not establish pathology causation or in-the-wild transfer.

**Use here.** Long-range and body-region ablations are useful controls for any gait judge. The result also warns that age, body size, sex, and lifestyle can drive a clinical prediction. Each proposal must match or condition on those variables where possible. This model is not an admissible base for the two-week program because no public checkpoint exists.

**Limits and access.** The cohort is predominantly Ashkenazi Jewish, all capture is frontal and depth-based, and associations are cross-sectional. The authors name Azure Kinect transfer as an open problem. The Human Phenotype Project data require an application. The stated GitHub URL returned 404, and the paper says code is available on request during review. The full arXiv PDF was read.

Sources: https://arxiv.org/abs/2603.25283 and https://github.com/AdamGabet/GaitPredict

## GaitEncoder: A Foundation Model of Gait Kinematics for Diverse Clinical Applications and Pathologies

**Problem.** GaitEncoder asks whether one compact clinical motion space can support disease recognition, severity estimation, recovery tracking, and intervention prediction.

**Method.** A weakly supervised variational autoencoder maps 32 Rajagopal-model kinematic channels at 24 stride-normalized time points into 16 features. A variational autoencoder, or VAE, learns a continuous compressed distribution from which inputs can be reconstructed. Its loss combines reconstruction, latent regularization, and gait-speed prediction, without diagnosis labels. The main evaluation model uses 381 people from four pathology groups, with three disorders held out entirely. The released final model uses 657 people across seven pathologies.

**Headline evidence.** The main model reconstructed joints with 3.5 degree mean absolute error. Zero-shot balanced accuracy was 0.68 for myotonic dystrophy, and condition-specific fine-tuning raised it to 0.81. For facioscapulohumeral muscular dystrophy, fine-tuned accuracy was 0.75 versus 0.44 for hand-designed gait features. Frozen features predicted held-out Parkinson severity at r=0.63 and r=0.65 for patient-rated and clinician-rated scores. A small latent intervention model predicted post-hip-surgery kinematics at 4.7 degrees error versus 5.9 degrees for the average-treatment baseline. The final model obtained 84.8 percent balanced accuracy across control, stroke, Parkinson disease, and hip osteoarthritis under five-fold cross-validation.

**Use here.** The released checkpoint is an immediate metric-kinematics baseline and a public clinical comparator. Its deviation from mean unimpaired score is a direct prior for normality. GAVD still needs monocular lifting and Rajagopal retargeting. The most important differentiation opportunity is to measure predictive dynamics, counterfactual repair, or physical consistency rather than another static distance to a normative latent mean.

**Limits and access.** Inputs are stride-normalized laboratory or OpenCap kinematics, not raw video. Left strides are reflected to resemble right strides, so the checkpoint intentionally erases laterality. All trials exclude assistive devices. The post-stroke recovery result is one person. The full medRxiv JATS text, public GitHub code and checkpoints, and public processed data were read and verified.

Sources: https://www.medrxiv.org/content/10.64898/2026.07.07.26357479v1, https://github.com/rdmagruder/GaitEncoder, and https://simtk.org/projects/gaitencoderdata

## Zero-shot Gait Classification with Diffusion Models

**Problem.** This is the closest verified prior to using generative surprise as a clinical gait measure. It asks whether a pretrained text-to-motion diffusion model can score Parkinson gait without fitting disease labels.

**Method.** The authors lift video into 24-joint motion over 1.0 to 1.6 seconds at 25 Hz and convert it to the HumanML3D representation with body shape removed. They pass the motion through the public Human Motion Diffusion Model, or MDM, under the prompt `Walking`. At each of 1,000 diffusion noise steps, they compare the predicted latent to the input latent. Mean squared latent error becomes a subject score, while channel-wise error becomes an anatomical explanation. Both are normalized by an empty-prompt score.

**Headline evidence.** Evaluation used 400 clips from 62 people, balanced to 100 clips at each observed Movement Disorder Society Unified Parkinson's Disease Rating Scale gait score from 0 through 3. The subject anomaly measure reached Spearman rho=0.7011 with clinical score. A one-way analysis of variance gave F=186.14. The 263 motion channels localized differences to lower limbs, distal joints, head, hip, and spine, depending on whether position, rotation, or velocity was measured.

**Use here.** A proposal that only applies diffusion error to GAVD is already occupied. A defensible advance must change the scientific object. Examples include minimum normalizing interventions, disagreement between physical feasibility and healthy typicality, calibration under camera shift, or causal edit responses. The public MDM checkpoint makes replication feasible, although the paper does not release its clinical preprocessing or evaluation code.

**Limits and access.** The clinical data are private. The work is a three-page short paper with no external cohort, held-out model selection report, camera controls, or comparison to simple speed and pose-quality baselines. It selects the first 900 diffusion steps because they best separate prompts, which risks analysis-set tuning. Direct OpenReview download returned 403, but the complete three-page PDF text was fetched and read through the indexed official OpenReview attachment. The official MDM repository provides public HumanML3D checkpoints and code.

Sources: https://openreview.net/forum?id=L5xyzjMCwd, https://openreview.net/attachment?id=L5xyzjMCwd&name=pdf, and https://github.com/GuyTevet/motion-diffusion-model

## Autoregressive Denoising Score Matching is a Good Video Anomaly Detector

**Problem.** A likelihood or reconstruction model can assign a deceptively normal score to an anomaly that lies near a local mode of its learned distribution. This paper tests whether scene, motion, and appearance evidence can correct that failure.

**Method.** A noise-conditioned score transformer learns only from normal 8-frame, 160 by 160 object crops. A score function is the gradient that points toward higher data density. Scene labels condition the score, while first-to-last-frame differences weight motion. At test time, twenty increasing noise levels are applied autoregressively to each denoised result. The final anomaly score combines score magnitude with the peak signal-to-noise ratio between input and reconstruction.

**Headline evidence.** Macro area under the receiver operating characteristic curve reached 94.2 on Avenue, 93.2 on ShanghaiTech, and 78.1 on NWPU Campus. Each dataset-specific model trained for 100 epochs with batch size 20 on four RTX 4090 GPUs. The method trains its anomaly model from scratch and therefore does not satisfy this program's base-model constraint.

**Use here.** A single GaitDynamics, MDM, or video-model error is not enough. A gait judge should separate motion evidence from scene and reconstruction quality, report stability across noise or forecast horizons, and test local-mode failures with controlled synthetic edits.

**Limits and access.** Benchmarks contain public-safety events rather than subtle clinical motion. The score is normalized across each test video, which does not directly support standalone patient calibration. The official repository currently contains only a README, image folder, and license despite the paper's code statement. It has no implementation or checkpoint. The full official ICCV paper and repository were read.

Sources: https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_Autoregressive_Denoising_Score_Matching_is_a_Good_Video_Anomaly_Detector_ICCV_2025_paper.html and https://github.com/Bbeholder/ADSM

## ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users

**Problem.** Standard person models often omit a prosthetic limb or confuse it with background. ProGait provides paired video, body and prosthesis masks, 23 body and foot keypoints, and clinician-authored corrective descriptions.

**Method.** The dataset has 412 frontal and sagittal clips from 144 trials by four transfemoral prosthesis users. Trials vary prosthesis configuration and occur inside parallel bars or in a hallway. Human correction follows GroundingDINO, SAM2, and RTMPose annotation. The nine clinical categories include rotational, support-base, timing, knee, length, clearance, socket, ankle, and normal patterns. Subject-disjoint splits support segmentation, pose, and gait classification.

**Headline evidence.** Prosthesis-tuned RTMPose reached 0.947 whole-body average precision and 0.918 lower-body average precision, compared with 0.855 and 0.804 for the original model. Nine-class lower-body classification reached 0.812 balanced accuracy for sagittal video but only 0.413 across the full mixed-view dataset. A two-layer head on a pretrained GPGait encoder reached 0.457 top-1 accuracy. These gaps show that viewpoint and prosthesis visibility are central, not incidental.

**Use here.** ProGait is a public stress test for joint body-object modeling and for separating assistive-device perception from gait characterization. It cannot replace GAVD because it has four people, but it can test whether a body-only model fails systematically on prostheses and whether explicit device tracks repair that failure.

**Limits and access.** Four people are too few for broad clinical claims. Multiple clips share a trial and textual label. Data were collected in two controlled indoor layouts. The dataset and annotations are public, but the repository's baseline-model section remains marked `TBD`, so no task-specific checkpoint was verified there. The full arXiv paper, official repository, and Hugging Face dataset page were read.

Sources: https://arxiv.org/abs/2507.10223, https://github.com/pittisl/ProGait, and https://huggingface.co/datasets/ericyxy98/ProGait
