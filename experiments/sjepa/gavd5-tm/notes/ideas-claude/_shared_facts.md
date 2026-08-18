# Shared grounded facts for the notes/ideas proposals

This is the single source of truth for every number and claim used across the seven
proposals in `notes/ideas/`. It is distilled from notebooks 00-06, `plan/_shared/evidence-ledger.md`,
the project `README.md`, and the `worldmodels/gait` + `worldmodels/wiki` trees. Every proposal
must keep its quantitative claims consistent with this file. Do not use em-dashes anywhere.

## Data and independence
- The independent unit is the SOURCE VIDEO, not the extracted clip.
- Canonical cohort: 96 sequences from 18 unique YouTube source videos.
- Per-condition sequence counts: normal 12, Parkinson's 9, stroke 12, myopathic 47, cerebral palsy 16.
- Per-condition SOURCE-VIDEO counts: normal 1, Parkinson's 2, stroke 3, myopathic 10, cerebral palsy 2.
- All 12 normal sequences come from ONE video (id `3KnFt8bH3tE`). Condition label is therefore nearly
  collinear with source-video identity, especially for normal.
- Full curriculum: 159 sequences from 35 source videos (75 normal after adding accepted augmented-normal
  windows, plus 84 canonical non-normal). Larger clip count does not remove source or provenance dependence.
- Added-normal: 63 accepted of 64 candidate windows (one rejected at neurologic coverage 0.027).
- Provenance confound: most normal rows use the AUGMENTED extraction path; every abnormal row uses the
  CANONICAL path. A normal-vs-abnormal classifier can learn acquisition or extraction differences.

## Token geometry and model
- Each sequence resized to 64 frames; 4 adjacent frames form one time patch => 16 time positions.
- 33 BlazePose joints x 16 time positions = 528 possible joint-time tokens.
- Coordinate dim 3 (monocular BlazePose x, y, relative z). One token embeds a 4-frame x 3-coord = 12-vector
  via a linear layer to embed_dim (default 64; depth 2, 4 heads, GELU, pre-norm).
- View (online) encoder sees only visible tokens; target (EMA) encoder sees all 528 tokens and is not
  updated by backprop. Predictor is a 2-layer Transformer with a learned mask token, predicts hidden
  target features, returns predictions only at masked positions.

## Masking (notebook 00/03)
- Only 12 landmarks are maskable prediction targets: left/right shoulder (11,12), hip (23,24),
  knee (25,26), ankle (27,28), heel (29,30), foot index (31,32). Face and arm joints are visible
  context but never targets.
- Maximum possible GLOBAL mask fraction = 12/33 = 0.364 (far below JEPA/V-JEPA's 75-90%), because
  masking a face or arm joint is forbidden.
- Configured target 0.60 of eligible tokens; batch-safe sampler masks 60% of the least-visible sample's
  eligible count, same count for every sample, always leaving at least one eligible token visible.
- Realized mean eligible-token fraction fell from 0.551 (end of Stage 0) to 0.423 (end of Stage 4).
- The sampler never reads coordinate size, displacement, velocity, acceleration, or a learned motion score
  (MAMP motion-aware sampling and MTM motion-topology masking are explicitly forbidden).

## Loss and anti-collapse (notebook 00/04)
- L = L_JEPA + 0.05 * L_VICReg + 0.25 * L_group.
- L_JEPA is a latent cross-entropy: teacher target is centered (running EMA center, beta 0.9) then sharpened
  at temperature 0.06 and detached (stop-gradient); prediction uses temperature 0.10.
- EMA target encoder momentum schedule via cosine, recommended real profile starts 0.999 -> 1.0.
- VICReg (variance floor + covariance penalty) resists collapse. L_group is a label-aware condition-centroid
  compactness plus margin term, active only in Stages 1-4, so Stages 1-4 are SUPERVISED representation
  fine-tuning, not pure self-supervised learning.
- Geometric augmentation: small y-axis rotation (max 8 degrees), small translation. Laterality FLIP is OFF
  by default (flip_probability 0.0) because left-right identity matters for stroke.

## Training run and health (notebook 04)
- Five-stage curriculum: Stage 0 normal (300 epochs), Stages 1-4 add PD, stroke, myopathic, cerebral palsy
  (75 epochs each). 600 curriculum epochs, 11,400 optimizer updates. Final checkpoint fingerprint prefix
  `d0acc262`. A canonical (non-augmented) lineage prefix `dba24a` has also been observed locally; bind every
  result to ONE fingerprint before comparing.
- Final feature standard deviation 0.413745 (not total collapse), mean pairwise cosine 0.609342.
- Normal-anchor cosine fell 0.954 (after Stage 1) -> 0.594 (after Stage 4): substantial drift.

## Representation geometry (notebook 05, mean/std pooling to 256-d)
- A mean and a standard deviation are permutation-invariant, so this pooled readout discards temporal order
  by construction.
- Cosine silhouette 0.008975 (no clean five-group separation).
- Minimum centroid distance 0.036718 (smaller than mean within-condition distance 0.119521).
- Mean centroid distance 0.292119. Closest centroids: myopathic and cerebral palsy.
- Historical scalar-decoding claims (step amplitude R^2 ~0.719; asymmetry R^2 ~0.154) are
  unreproducible legacy claims: their archived model, target provenance, and split artifact are
  not available. Notebook 05 now contains a versioned scalar-readout audit; do not treat those
  numbers as current evidence until that audit is run and its state-hash-bound report is cited.

## Readouts and leakage (notebook 06)
- All results are TRANSDUCTIVE: the encoder saw every evaluation row. A held-out probe split is still
  transductive if the encoder saw that video's clips.
- All-96 stratified S-JEPA: accuracy 0.793, balanced 0.889, macro-F1 0.821 (all 16 test videos overlap
  training; all 29 test rows trained the encoder).
- All-96 missingness-only control: accuracy 0.448, balanced 0.466, macro-F1 0.429 (visibility only, no
  gait coordinates).
- Video-grouped binary readout: accuracy 0.849, balanced 0.874 (probe folds group videos, but encoder saw
  all 159 rows). Normal-vs-abnormal separability at the embedding level is very high (order 0.96 AUC), which
  is exactly the number the provenance confound puts at risk.
- Video-grouped five-class readout: accuracy 0.653, macro-F1 0.625 (two folds; encoder saw all rows).

## Extraction (notebook 02)
- MediaPipe pose_landmarker_lite, video mode, single pose, detection/presence/tracking confidence 0.45.
- Heels are the visibility weak link: left heel mean visibility ~0.699, right heel ~0.673, vs shoulders and
  hips ~0.988. Failed pose rows are kept on the timeline with zero visibility to preserve gait timing.
- Source video resolution 1280x720; measured FPS around 30 (some 23.976, 29.97). GAVD CSVs do NOT store FPS;
  notebook 02 probes it from the MP4 with a hard-coded 29.97 fallback. frame_num is 1-based absolute.

## Five claim boundaries (do not cross)
1. The independent unit is the source video, not the clip.
2. A held-out probe split is still transductive if the encoder saw that video.
3. Seed variation is not source variation.
4. The label-aware group loss makes Stages 1-4 supervised fine-tuning.
5. Folder labels (stroke, parkinsons) are dataset annotations, not diagnoses made by this project.

## Reviewer framing (verified)
- ICLR 2026 Reviewer Guide: state-of-the-art performance is NOT required; a well-motivated study that
  contributes new knowledge (including careful analysis or a negative result) is valued.
- ICML 2026 and NeurIPS 2026 reward originality-through-evaluation and informative null results that change
  understanding.
- Seven screening questions each proposal must pass: (1) falsifiable in one sentence; (2) source video is the
  unit before all fitting; (3) changes only the named factor or controls every extra change; (4) has a simple
  non-neural or nuisance baseline; (5) a null rules out a plausible belief; (6) decisive figure by Day 14;
  (7) matters beyond this repository.

## Verified citations (use these exact anchors)
- Abdelfattah and Alahi, S-JEPA, ECCV 2024, DOI 10.1007/978-3-031-73411-3_21.
- Assran et al., I-JEPA, CVPR 2023, arXiv:2301.08243.
- Bardes et al., V-JEPA "Revisiting Feature Prediction for Learning Visual Representations from Video", 2024,
  arXiv:2404.08471 (masked latent feature prediction, EMA target + stop-gradient, tube masking, frozen probe).
- Bardes, Ponce, LeCun, VICReg, ICLR 2022, arXiv:2105.04906 (variance / invariance / covariance; variance
  term maintains high effective rank).
- Assran et al., V-JEPA 2, 2025, arXiv:2506.09985 (action-free pretraining then action-conditioned predictor).
- Ranjan et al., GAVD, IEEE Access 2025, DOI 10.1109/ACCESS.2025.3545787.
- Kapoor and Narayanan, "Leakage and the Reproducibility Crisis in ML-based Science", 2022, arXiv:2207.07048
  (taxonomy of leakage: no independent test set / train-test contamination, temporal leakage).
- Varoquaux, "Cross-validation failure: small sample sizes lead to large error bars", NeuroImage 2018.
- Grishchenko et al., BlazePose GHUM, 2022, arXiv:2206.11678.
- Rousseeuw, silhouettes, 1987, DOI 10.1016/0377-0427(87)90125-7.
- Xu et al., "A Theory of Usable Information Under Computational Constraints" (V-information / predictive
  V-usable information), ICLR 2020, arXiv:2002.10689 (for proposal 07's per-stage information panel).

## Verified external multi-view / pose-validity cohorts (non-clinical; reach-tier arms only)
- Yu, Tan, Tan, "A Framework for Evaluating the Effect of View Angle, Clothing and Carrying Condition on
  Gait Recognition" (CASIA-B), ICPR 2006 (124 subjects, 11 camera views; non-clinical multi-view gait).
- Takemura et al., "Multi-view large population gait dataset and its performance evaluation for cross-view
  gait recognition" (OU-MVLP-Pose), IPSJ Trans CVA 2018 (~10,000 subjects, multi-view pose keypoints).
- Zhu et al., "Gait Recognition in the Wild: A Benchmark" (GREW), 2022, arXiv:2205.02692.
- Zheng et al., "Gait Recognition in the Wild with Dense 3D Representations and a Benchmark" (Gait3D),
  2022, arXiv:2204.02569.
- Ionescu et al., "Human3.6M: Large Scale Datasets and Predictive Methods for 3D Human Sensing in Natural
  Environments", IEEE TPAMI 2014, DOI 10.1109/tpami.2013.248 (pose validity against motion capture).
- Stenum et al., "Two-dimensional video-based analysis of human gait using pose estimation", PLoS Comput
  Biol 2021, PMID 33891585 (temporal MAE ~0.02 s/step, sagittal joint angles 4-7 deg; skeleton validity).

## Distinctness from the existing plan/ portfolio (state this in each README)
Existing plan proposals: 01 honest video-disjoint anomaly screening; 02 clinical threshold audit;
03 SIGReg effective-rank audit; 04 motion-vs-position TARGET ablation (retrains encoders);
05 temporal READOUT diagnostic (mean/std vs temporal head); 06 missingness/visibility confound control;
07 viewpoint/selective-invariance stress test.
- ideas/01 makes provenance the OBJECT (plan/06 treats it only as a nuisance control).
- ideas/02 makes the 2-D error IMAGE the object (plan/01 pools error into one scalar).
- ideas/03 changes only the inference SCORING target on the frozen encoder (plan/04 retrains with motion targets).
- ideas/04 makes the fixed-64-frame resize the object as a preprocessing-validity measurement (no plan item does this).
- ideas/05 makes signed asymmetry a decodable axis and tests learned reflection-equivariance (distinct from plan/05 and plan/07).
- ideas/06 makes mask GEOMETRY the treatment (plan/01 sweeps masks only as robustness; plan/04 fixes masks, varies targets).
- ideas/07 isolates the label-aware GROUP LOSS as the single factor (no plan item does this).
