# Literature verification report: Gait S-JEPA workshop paper (BrainBodyFM @ NeurIPS 2026)

Prepared by a literature-research subagent. Every claim below was checked against primary sources (paper PDFs, arXiv API metadata, Crossref, publisher/workshop pages, GitHub) fetched live on 2026-09-02. Items I could NOT verify are explicitly marked UNVERIFIED rather than guessed.

---

## 1. S-JEPA — VERIFIED (from the ECVA PDF, checked line-by-line)
"S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition", Mohamed Abdelfattah & Alexandre Alahi (EPFL), **ECCV 2024**, pp. 367-384, Springer. No arXiv version exists for this paper (confirmed by arXiv API title search returning 0 results) - cite the ECVA/Springer PDF/DOI.
- PDF: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf | project: https://sjepa.github.io/ | DOI: 10.1007/978-3-031-73411-3_21
- Pretext task: given a **partially masked skeleton sequence, predict the latent representations of the missing joints** of the SAME sequence (not coordinates, not motion). ✓
- Masking: motion-aware masking following MAMP [32]; "mask ratio r to 0.9 and segment length l to 4" ✓ (i.e., 90% motion-aware masking within temporal segments of 4 frames; sequence trimmed to fixed length T=120; random trim 0.5-1.0 train / 0.9 test).
- Architecture: "vanilla transformer"; view encoder Le = **8 layers**, predictor Lp = **5 layers**, embedding dim Ce = Cp = **256**, **8 heads**, FFN hidden 1024 ✓. Separate learnable spatial + temporal positional embeddings.
- Target encoder: **EMA** of view-encoder weights, lambda raised 0.9999 -> 1.0 on cosine schedule; predictor applied at masked positions via learnable mask tokens. ✓
- Loss: cross-entropy between **centered + softmax-sharpened** representations **over the feature dimension** (centering rate beta=0.9; sharpening temps tau_p=0.1, tau_t=0.06); centering is S-JEPA's contribution for training stability. ✓
- Pretraining 1200 epochs, batch 256, AdamW, 8x A100; eval: linear / fine-tune / semi-supervised / transfer.
- Reported numbers (Table 1, linear eval, joints-only input): NTU60 XSub **85.3**, XView **89.8**; NTU120 XSub **79.6**, XSet **79.9**; PKU-MMD I **92.2**, II **53.5**. Beats MAMP (84.9/89.1/78.6/79.1/92.2/53.8) and SkeletonMAE (74.8/77.7/72.5/73.5/82.8/36.1). Fine-tuning (Table 2): NTU60 XSub ~93.1, XView 97.6; NTU120 XSub 90.3, XSet 91.3 (website headline: "Fine-Tuning 97.6% XView"; +10 / +13.5 pts over from-scratch transformer on NTU60/120 XSub). Semi-supervised 1%-labels NTU60-XSub: 67.5%.
- Correction to your framing: masking is **motion-aware within l=4-frame segments at ratio 0.9**, and prediction targets are obtained by **masking the target encoder's OUTPUT** (not the input), matching I-JEPA's recipe.

## 2. I-JEPA / V-JEPA — VERIFIED (from paper text)
- **I-JEPA**: Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas. "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", **CVPR 2023**; arXiv:2301.08243; OpenAccess: https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html. Verified: predict representations of multiple target blocks from ONE context block of the same image; **4 target blocks**, scale ~(0.15,0.2), aspect (0.75,1.5), context block scale (0.85,1.0); targets computed by a **target encoder updated via EMA** of the context encoder (stop-gradient); **no negatives, no augmentation views**; multi-block masking ablations; predictor applied per target block with learnable mask tokens. ("masking 4 target blocks" ✓.)
- **V-JEPA**: Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas. "Revisiting Feature Prediction for Learning Visual Representations from Video", **TMLR 2024**; arXiv:2404.08471. NOTE: not literally titled "V-JEPA". Verified: feature-space prediction (no pixels, no text, no negatives, no human annotations); **multi-block masking**: short-range = union of 8 random blocks covering 15% of each frame, long-range = union of 2 blocks covering 70%; EMA target (y) encoder; ablates random-tube (90%) and **causal multi-block** (context = first p frames) masking; **frozen evaluation via attentive probe** (learnable cross-attention query + MLP -> linear classifier); ViT-H/16 frozen -> K400 81.9%, SSv2 72.2%, IN1K 77.9%. The causal-multi-block ablation is your best citation for D4's "causal latent forward prediction" design choice.

## 3. LeCun world-model framing — VERIFIED
- Yann LeCun, "A Path Towards Autonomous Machine Intelligence" (position paper, v0.9, June 2022). **Not on arXiv.** Canonical OpenReview forum (confirmed reachable, id resolves): https://openreview.net/forum?id=BZ5a1r-kVsf (PDF: https://openreview.net/pdf?id=BZ5a1r-kVsf). Proposes JEPA as core of an autonomous agent architecture: world model predicting **latent** representations, hierarchical planning by latent prediction, intrinsic cost modules - exactly the framing D4 (causal latent forward prediction + latent rollout) builds on.

## 4. Skeleton SSL / masked pose modeling (MAMP, Skeleton2vec, + newer)
- **MAMP** - "Masked Motion Predictors are Strong 3D Action Representation Learners", Mao, Deng, Zhou, Fang, Ouyang, Li, **ICCV 2023**, arXiv:2308.07092; https://www.openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html. Motion-aware masking: joints with large temporal motion masked with higher probability within segments; pretext = predict masked joints' explicit motion. (S-JEPA inherits MAMP's masking recipe and ablates coordinate- vs motion- vs latent targets.)
- **Skeleton2vec** - Xu, Huang, Wang, Hu, Deng, arXiv:2401.00921 (Jan 2024). Teacher encoder on unmasked samples produces contextualized target latent representations. CAVEAT: arXiv listing says only "Submitted to CVPR 2024"; I found **no confirmed publication venue** - cite as arXiv preprint.
- **SkeletonMAE (graph)** - Yan, Liu, Wei, Li, Lin, **ICCV 2023**, arXiv:2307.08476 (graph-based asymmetric MAE). There is also a DIFFERENT "SkeletonMAE" (Wu et al., arXiv:2209.02399). Disambiguate in your bib.
- **HumanMAC** - Chen et al., "HumanMAC: Masked Motion Completion for Human Motion Prediction", **ICCV 2023**, arXiv:2302.03665.
- **MotionGPT** - Jiang, Chen, Liu, Yu, Yu, Chen, "Human Motion as a Foreign Language", **NeurIPS 2023**, arXiv:2306.14795; https://papers.nips.cc/paper_files/paper/2023/hash/3fbf0c1ea0716c03dea93bb6be78dd6f-Abstract-Conference.html
- Newest skeleton-SSL relevant to mask geometry (2026): **ASMa** "Asymmetric Spatio-temporal Masking for Skeleton Action Representation Learning", arXiv:2602.06251 - masks "high-degree joints + low-motion" vs "low-degree joints + high-motion" asymmetrically; direct precedent for D3 mask-geometry ablations.
- NOT FOUND: any paper named "MotionMAE" or "PST" matching your description (no arXiv records by title). Do not cite them by name.

## 5. Gait-specific SSL / clinical gait works
- **GaitForeMer** ✓ - Endo, Poston, Sullivan, Fei-Fei, Pohl, Adeli, "Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for Few-Shot Gait Impairment Severity Estimation", **MICCAI 2022**, LNCS 13438, pp. 130-139, DOI 10.1007/978-3-031-16452-1_13. Pretrains a transformer by forecasting future gait motion on public action data (NTU), then estimates **MDS-UPDRS gait impairment severity** from video of PD subjects (F1 0.76). Caveat: its downstream clinical dataset was NOT disclosed in the paper (noted in its own reviews). Code: github.com/markendo/GaitForeMer.
- **FSGait** - CORRECTION: Duan, Wan, Zhao, **ACCV 2024**, "FSGait: Fine-Grained Self-supervised Gait Abnormality Detection" (CVF: https://www.openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html; DOI 10.1007/978-981-96-0960-4_19). "FS" = **Fine-Grained**, not few-shot; task is **gait-abnormality detection** (self-supervised), not few-shot recognition.
- **GaitPT** ✓ exists but is NOT clinical SSL: Catruna, Cosma, Radoi, "GaitPT: Skeletons Are All You Need For Gait Recognition", arXiv:2308.10623 - hierarchical **skeleton gait-recognition** transformer (82.6% CASIA-B; identity task).
- **Gait SSL benchmark**: Fan, Hou, Wang, Huang, Yu, "Learning Gait Representation From Massive Unlabelled Walking Videos: A Benchmark", **IEEE TPAMI 45(12):14920-14937 (2023)**, DOI 10.1109/TPAMI.2023.3312419.
- NOT FOUND by exact title: "GaitSSB", "GaitMAE", "GaitGPT", "GaitFormer" (a "GaitMA" exists but is a different multimodal gait-recognition fusion paper, arXiv:2407.14812). Drop these names from the related-work skeleton until you have exact citations.
- **Gait foundation models 2025-2026** (directly relevant to positioning):
  - "Silhouette-based Gait Foundation Model" (FoundationGait), Ye, Fan, et al., arXiv:2512.00691 (Dec 2025): first scalable self-supervised gait pretraining, ~0.13B params, 12 datasets / >2M walking sequences, silhouette modality.
  - "A Gait Foundation Model Predicts Multi-System Health Phenotypes from 3D Skeletal Motion", Gabet et al., arXiv:2603.25283 (Mar 2026, preprint): 3D-skeletal-motion FM from 3,414 adults; predicts age/BMI/VAT and 1,980/3,210 phenotypes; includes an **anatomical ablation** (legs dominate metabolic/frailty predictions, torso encodes sleep/lifestyle) - strong precedent for body-part ablations and health-phenotype probing.
  - "GaitEncoder: A Foundation Model of Gait Kinematics...", Magruder, Gilon, Falisse, Uhlrich, medRxiv 2026, DOI 10.64898/2026.07.07.26357479v1 (16-dim VAE latent of walking kinematics, 657 individuals / 7 pathologies, OpenCap smartphone pipeline; DMU deviation metric).
  - NONE of these (nor any 2025-2026 gait FM found) uses **video-grouped leakage auditing** - D1 is a genuine gap.

## 6. Subject/source identity leakage — MOSTLY CORRECTED
Your claim "work by the LaBraM authors" is NOT what I find. The identity-leakage literature on EEG FMs is produced by OTHER groups auditing LaBraM (and BIOT, EEGPT, CBraMod, REVE):
- **"The Identity Trap in EEG Foundation Models: A Diagnostic Audit"** - Lin, Wu, Jung (UCSD/Swartz Center), arXiv:2606.06647 (2026, code "FMScope"). The single best analog for D1: shows near-perfect clinical accuracy under subject-disjoint CV can be explained by **subject-identity features**; frozen-repr diagnostics (subject-variance 13-89x null in 12/12 model-dataset pairs; subject-axis erasure; within-subject direction consistency). https://arxiv.org/abs/2606.06647
- **"Pretrained, Frozen, Still Leaking: Auditing Cross-Encoder Attribute Transfer in EEG Foundation Models"** - Tai, arXiv:2606.09189 (2026). Audits BIOT/LaBraM/EEGPT jointly for reconstruction/membership/identity/attribute leakage; single-encoder ridge probes transfer to held-out-subject splits via a linear bridge.
- **"Subject Identity Confounds qEEG Emotion Recognition on DEAP and DREAMER"** - Pandilova, Stojmenski, Chorbev, Petrov, Kitanovski, Trajanov, **Sensors 2026, 26(17):5327**, DOI 10.3390/s26175327 (leakage-free re-evaluation: emotion effects shrink after identity is accounted for).
- **LaBraM itself**: Jiang, Zhao, Lu, "Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI", **ICLR 2024**, arXiv:2405.18765; https://iclr.cc/virtual/2024/poster/18658. (EEG2Rep, arXiv:2402.17772, is a masking-based EEG-SSL work worth citing for the biosignal-masking parallel.)
- **Video/pose analogs of identity leakage** (for the "movement analog" framing):
  - **EgoPrivacy** - Li et al., ICML 2025, arXiv:2506.12258 (identity leakage from first-person video).
  - **Activity-Biometrics** - Azad & Rawat, CVPR 2024, arXiv:2403.17360 (person identification from daily-activity videos).
  - **Identity Preserve Transform** - Lyu, Qiu, Wei, Zhang, Yuille, Zha, arXiv:1912.06314: shows **action-classification models trained on activity videos implicitly learn person identity** - arguably the closest published statement of the D1 phenomenon for body-motion models (venue UNVERIFIED).
  - Kinetics dedup across train/test (Kay et al., arXiv:1705.06950) is the closest mainstream-video acknowledgment of source/duplicate leakage.
- **General leakage methodology**: Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science", Patterns 4(9):100804 (2023), arXiv:2207.07048 - canonical leakage taxonomy (incl. group leakage); cite for the random-split vs grouped-split gap.

## 7. Probing / SSL-evaluation methodology
- Alain & Bengio, "Understanding intermediate layers using linear classifier probes", ICLR 2017 (arXiv:1610.01644). ✓
- Hewitt & Liang, "Designing and Interpreting Probes with Control Tasks", EMNLP 2019 (arXiv:1909.03368) - control-task baseline for probe selectivity (report probe-vs-control gap).
- Kornblith, Shlens, Le, "Do Better ImageNet Models Transfer Better?", CVPR 2019 (arXiv:1805.08974) - canonical linear-probing/transfer protocol.
- Ericsson, Gouk, Hospedales, "How Well Do Self-Supervised Models Transfer?", CVPR 2021 (arXiv:2011.13377).
- No paper titled exactly "A critical view of self-supervised learning" surfaced. If you mean Asano/Rupprecht/Vedaldi "A Critical Analysis of Self-Supervision, or What We Can Learn From a Single Image" (ICLR 2020, arXiv:1904.13132), VERIFY before citing (I did not re-check its metadata). Recommendation: pair a linear-probe protocol with frozen + fine-tune curves and a Hewitt-Liang-style control task.

## 8. Grouped/hierarchical CV — VERIFIED
- Roberts, Bahn, Ciuti, Boyce, Elith, Guillera-Arroita, Hauenstein, Lahoz-Monfort, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", **Ecography 40:913-929 (2017)**, DOI 10.1111/ecog.02881 ✓ - canonical citation for grouped CV under correlated/nested data; naive random CV overestimates accuracy when clusters share structure.
- Video-benchmark practice: subject-disjoint splits are standard in skeleton benchmarks (NTU RGB+D Cross-Subject/Cross-View: Shahroudy, Liu, Ng, Wang, CVPR 2016, arXiv:1604.02808). Your "sequence-split vs source-split gap" is the skeleton/gait instantiation of Roberts' block-CV logic and of the "subject-disjoint CV" your EEG counterparts use. No published gait-SSL work appears to report this gap -> novelty claim supportable.

## 9. Gait phase / biomechanical parameter estimation from pose
Verified citable anchors (monocular/smartphone-video pose):
- Kidzinski, Yang, Hicks, Rajagopal, Delp, Schwartz, "Deep neural networks enable quantitative movement analysis using single-camera videos", **Nature Communications 11:4054 (2020)**, DOI 10.1038/s41467-020-17807-z - clinical gait quantification from single-camera 2D pose.
- Uhlrich et al., "OpenCap: Human movement dynamics from smartphone videos", **PLOS Computational Biology 19(10):e1011462 (2023)**, DOI 10.1371/journal.pcbi.1011462 - 3D biomechanics from smartphone video (used by GaitEncoder).
- Natraj et al., "3D pose estimation for scalable remote gait kinematics assessment", **npj Digital Medicine** (2025/26), DOI 10.1038/s41746-025-02211-y.
- Liu, Wong, Chen, Antonelli, Guarin, "VisionMD-Gait: scalable clinical gait assessment from smartphone videos", **Scientific Reports** (2026), DOI 10.1038/s41598-025-34912-5.
- Stroke-clinical pose-gait proof-of-concept: PMC8832219 (journal UNVERIFIED - check before citing).
- CAVEAT / genuine gap: I did NOT find a clean, dedicated, well-known reference for **gait-cycle phase (e.g., heel-strike) estimation from 2D/3D pose landmarks treated as a periodic latent**. Defensible as novel territory for D2/D4 - say so explicitly ("phase estimation from pose remains under-standardized") rather than implying a rich literature. Linear-probe targets (speed, cadence, asymmetry, joint excursion, sway) are all derivable from pose time series (Kidzinski/OpenCap pipelines) and are standard clinical gait measures.

## 10. Latent world models for motor control / motion
- **DreamerV3**: Hafner, Pasukonis, Ba, Lillicrap, arXiv:2301.04104 (2023) ✓.
- **TD-MPC2**: Hansen, Su, Wang, arXiv:2310.16828 (2023) ✓.
- **MotionGPT** (NeurIPS 2023, sec. 4) - motion-language generation context, not a world model.
- Pose/motion world-model-flavored works found: **"Semantic Belief-State World Model for 3D Human Motion Prediction"** (Chaudhry, arXiv:2601.03517, Jan 2026) - closest pose-based world model; VLA-JEPA (arXiv:2602.10098) and a humanoid-locomotion latent world model (arXiv:2608.06375) were glimpsed but UNVERIFIED in detail - fetch before citing.
- D4 framing note: predicting the FUTURE is an established skeleton SSL pretext (GaitForeMer forecasts future motion in joint space; V-JEPA's causal-multi-block ablation predicts from first p frames), but predicting FUTURE *latents* with a JEPA-style predictor on gait skeleton data, with gait-phase-conditioned evaluation, is not published. Cite the causal-forecasting evidence and claim the latent-future-prediction-on-gait variant as new.

## 11. BrainBodyFM workshop: 2025 accepted papers + 2026 CFP — VERIFIED from the official site
- **2026 CFP** (https://brainbodyfm-workshop.github.io/call-for-papers.html): submissions open **Aug 3, 2026**; deadline **Sep 5, 2026 AoE** (papers + travel awards), demos Sep 19; notification Sep 29; camera-ready Nov 6; workshop **Dec 11 or 12, 2026, Sydney** (second edition). Format: **5 pages max excluding references and appendices, modified NeurIPS 2026 LaTeX style**, via **OpenReview**, **fully anonymized / double-blind**; **non-archival**; 9 spotlights; concurrent NeurIPS-2026 submissions allowed but withdrawn if accepted at main conf. Scope explicitly lists "pose extracted from video" as a biosignal and topics "Measuring whether pretraining helps: probing and downstream evaluation", "Reproducibility and standardized reporting" - your four directions map almost 1:1 onto the CFP.
- **2025 edition** (https://brainbodyfm-workshop.github.io/2025/): **57 accepted papers / 8 spotlights** (full list: repo file 2025/brainbodyfm_accepted_submissions.csv). Movement/pose-relevant items: **CPEP** "Contrastive Pose-EMG Pre-training Enhances Gesture Generalization" (Spotlight; OpenReview cr7jItQVAN); "Human Sensory-Musculoskeletal Modeling and Control of Whole-Body Movements" (Spotlight); "Imitation learning of dexterous hand control..." (Mathis et al.); "Handwriting decoding as a challenging Motor Imagery task for EEG FMs"; "Predictive Modeling of Brain-Body Association"; **PhysioJEPA** (JEPA for physiological signals; non-archival, no arXiv); "EEG Foundation Models: A Critical Review..."; "Are foundation models useful feature extractors for electroencephalography analysis?". **No gait-dataset/FM paper and no skeleton/pose-SSL paper was accepted in 2025** -> positioning as the first skeleton-JEPA *gait* FM with leakage audits + probing + mask ablations at this venue is cleanly novel; cite PhysioJEPA + CPEP + the EEG feature-extractor-probing paper as nearest neighbors.

## 12. GAVD dataset — VERIFIED (paper + GitHub + DOAJ)
- Ranjan, Ahmedt-Aristizabal, Ali Armin, Kim, "Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset", **IEEE Access 13:45321-45339 (2025)**, DOI 10.1109/ACCESS.2025.3545787; https://ieeexplore.ieee.org/document/10921672; annotations: https://github.com/rahmyyy/gavd.
- **1,874 video sequences ✓; >450 videos ✓** (annotation repository "for over 450 videos"); **over 400 subjects ✓**; RGB from public (YouTube) sources; normal/abnormal/pathological + per-sequence clinical gait-pattern labels; TSN 94% / SlowFast 92% abnormality detection baselines (their splits).
- IMPORTANT reproducibility caveats: repo ships **only annotations/URLs - no video files, no person-identity column, no official train/test protocol; dead links expected**; some videos contain multiple people (bbox-tracked). Consequences: (a) your "source-video ID" is the natural grouping key because **subject ID is not provided** - state this and treat video-grouped CV as the identity proxy; (b) re-download availability and YouTube ToS/ethics should be acknowledged (fits the workshop's reproducibility emphasis). Your 5-class subset (normal/PD/stroke/CP/myopathic) vs the paper's binary normal/abnormal + multi-class gait-pattern labels: whether "myopathic" exists among official gait_pat labels is **UNVERIFIED** (I did not inspect the CSV).
- The GAVD paper also validates on a "Clinical Abnormality Simulated Dataset (CASD)" (73%/87%) - a precedent for cross-dataset probing in D2.

---

## CORRECTIONS / UNCERTAINTY SUMMARY
1. S-JEPA: mask ratio 0.9 = motion-aware (MAMP-style, segment length l=4); targets = masked target-encoder OUTPUTS; no arXiv version exists.
2. V-JEPA is "Revisiting Feature Prediction for Learning Visual Representations from Video" (TMLR 2024); masking = 8 blocks@15% (short-range) U 2 blocks@70% (long-range); its **causal multi-block** ablation is your D4 citation.
3. FSGait = "Fine-Grained" (not few-shot) gait-abnormality detection, ACCV 2024 (Duan/Wan/Zhao) ✓ venue.
4. "Identity leakage by LaBraM authors" - not found; real works audit LaBraM (Identity Trap/FMScope; Tai 2026; DEAP/DREAMER 2026). For body motion: EgoPrivacy (ICML'25), Activity-Biometrics (CVPR'24), Lyu et al. arXiv:1912.06314.
5. "GaitSSB / GaitMAE / GaitGPT / GaitFormer / MotionMAE / PST" - could not be verified by exact title; do not cite by name.
6. Gait phase as a periodic latent from pose: thin published basis; position as a gap.
7. GAVD: no subject IDs -> "source-video grouping" is the correct identity proxy; 96-seq/5-class numbers are your subset, not the dataset headline stats.
8. "A critical view of self-supervised learning" - no such canonical paper found; use Kornblith linear-probe protocol + Hewitt-Liang control tasks, or verify Asano et al. 2020 first.

## TOP 15 REFERENCES (5-page BrainBodyFM paper: gait S-JEPA + leakage audit + probing + mask ablation + latent forward prediction)
1. Abdelfattah & Alahi, "S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition", ECCV 2024. https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf (DOI 10.1007/978-3-031-73411-3_21)
2. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture", CVPR 2023. arXiv:2301.08243
3. Bardes et al., "Revisiting Feature Prediction for Learning Visual Representations from Video", TMLR 2024. arXiv:2404.08471
4. LeCun, "A Path Towards Autonomous Machine Intelligence", 2022. https://openreview.net/forum?id=BZ5a1r-kVsf
5. Mao et al., "Masked Motion Predictors are Strong 3D Action Representation Learners" (MAMP), ICCV 2023. arXiv:2308.07092
6. Xu et al., "Skeleton2vec: A Self-supervised Learning Framework with Contextualized Target Representations for Skeleton Sequence", arXiv:2401.00921 (2024; venue unconfirmed)
7. Endo et al., "GaitForeMer: Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for Few-Shot Gait Impairment Severity Estimation", MICCAI 2022. DOI 10.1007/978-3-031-16452-1_13
8. Duan, Wan, Zhao, "FSGait: Fine-Grained Self-supervised Gait Abnormality Detection", ACCV 2024. https://www.openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html (DOI 10.1007/978-981-96-0960-4_19)
9. Gabet et al., "A Gait Foundation Model Predicts Multi-System Health Phenotypes from 3D Skeletal Motion", arXiv:2603.25283 (2026) - closest clinical 3D-skeleton gait FM; anatomical ablations.
10. Lin, Wu, Jung, "The Identity Trap in EEG Foundation Models: A Diagnostic Audit", arXiv:2606.06647 (2026) - D1's template.
11. Pandilova et al., "Subject Identity Confounds qEEG Emotion Recognition on DEAP and DREAMER", Sensors 2026, 26(17):5327. DOI 10.3390/s26175327
12. Kapoor & Narayanan, "Leakage and the Reproducibility Crisis in Machine-Learning-Based Science", Patterns 4(9):100804 (2023). arXiv:2207.07048
13. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure", Ecography 40:913-929 (2017). DOI 10.1111/ecog.02881
14. Alain & Bengio, "Understanding intermediate layers using linear classifier probes", ICLR 2017. arXiv:1610.01644 (pair with Hewitt & Liang, EMNLP 2019, arXiv:1909.03368)
15. Kidzinski et al., "Deep neural networks enable quantitative movement analysis using single-camera videos", Nature Communications 11:4054 (2020). DOI 10.1038/s41467-020-17807-z (pair with OpenCap, PLOS Comput Biol 2023, DOI 10.1371/journal.pcbi.1011462)

Honorable mentions to weave into the text: SkeletonMAE (ICCV'23, arXiv:2307.08476), HumanMAC (ICCV'23, arXiv:2302.03665), ASMa (arXiv:2602.06251), MotionGPT (NeurIPS'23, arXiv:2306.14795), GaitPT (arXiv:2308.10623), Fan et al. TPAMI 2023 (10.1109/TPAMI.2023.3312419), FoundationGait (arXiv:2512.00691), GaitEncoder (medRxiv 10.64898/2026.07.07.26357479v1), LaBraM (ICLR 2024, arXiv:2405.18765), EEG2Rep (arXiv:2402.17772), Tai (arXiv:2606.09189), EgoPrivacy (ICML'25, arXiv:2506.12258), Activity-Biometrics (CVPR'24, arXiv:2403.17360), Lyu et al. (arXiv:1912.06314), NTU RGB+D (CVPR 2016, arXiv:1604.02808), Kornblith et al. (CVPR 2019, arXiv:1805.08974), Ericsson et al. (CVPR 2021, arXiv:2011.13377), DreamerV3 (arXiv:2301.04104), TD-MPC2 (arXiv:2310.16828), Chaudhry (arXiv:2601.03517), Ranjan et al. IEEE Access 2025 (GAVD), VisionMD-Gait (10.1038/s41598-025-34912-5), Natraj et al. npj Digit. Med. (10.1038/s41746-025-02211-y).

---
*Practical note: with "today" = 2026-09-02 and the workshop deadline 2026-09-05 AoE, submissions close in ~3 days - recommend freezing the reference list now.*
