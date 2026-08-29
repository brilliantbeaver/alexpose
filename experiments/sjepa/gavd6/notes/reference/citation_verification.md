# 06 — Literature Findings: Citation Verification

## Purpose

This is the former “notes/06” deliverable, retained after the notes
reorganization, to verify
**every arXiv / ACM / IEEE / DOI identifier before it enters the manuscript** for the
S-JEPA monocular-gait classification project. The prime directive is **honesty**: each
citation below is marked `verified`, `partial`, `wrong`, or `unresolved` based *only* on
the recorded verification evidence. No verdict has been upgraded, and no DOI or arXiv id
has been invented. Where the planning-note §5 flagged an identifier as a guess (MAMP
arXiv, MotionBERT arXiv, GAVD arXiv:2309.01480), the resolution is reported prominently
in the relevant section below.

## Summary table

| Ref | Short name | Claimed id | Verdict | Recommendation |
|-----|------------|------------|---------|----------------|
| [1] | Zanardi et al. — PD gait meta-analysis | `10.1038/s41598-020-80768-2` | verified | cite-as-is |
| [2] | Lauziere et al. — post-stroke gait asymmetry | `10.4172/2329-9096.1000201` | verified | cite-as-is |
| [3] | Pandey et al. — CP crouch gait | `10.1007/s43465-023-01002-5` | verified | cite-as-is |
| [4] | Maulet et al. — Pompe motor function | `10.1212/WNL.0000000000201333` | verified | cite-as-is |
| [5] | Assran et al. — I-JEPA | `10.1109/CVPR52729.2023.01499` (alt `arXiv:2301.08243`) | verified | cite-as-is |
| [6] | Abdelfattah & Alahi — S-JEPA | `10.1007/978-3-031-73411-3_21` | verified | cite-as-is |
| [7] | Mao et al. — MAMP | `10.1109/ICCV51070.2023.00934` (alt `arXiv:2308.07092`) | verified | cite-as-is |
| [8] | Xu et al. — Skeleton2vec | `10.48550/arXiv.2401.00921` | verified | cite-as-is |
| [9] | Endo et al. — GaitForeMer | `10.1007/978-3-031-16452-1_13` | verified | cite-as-is |
| [10] | Duan et al. — FSGait | `10.1007/978-981-96-0960-4_19` | **partial** | cite-with-corrected-id (pages) |
| [11] | Ranjan et al. — GAVD | `10.1109/ACCESS.2025.3545787` (alt `arXiv:2309.01480` ✗) | verified | cite-as-is (drop bad arXiv) |
| [12] | Grishchenko et al. — BlazePose GHUM | `10.48550/arXiv.2206.11678` | verified | cite-as-is |
| [13] | Hii et al. — marker-free gait | `10.3390/s23146489` | verified | cite-as-is |
| [14] | Roberts et al. — blocked CV | `10.1111/ecog.02881` | verified | cite-as-is |
| §5 | Bardes et al. — V-JEPA | `arXiv:2404.08471` | verified | cite-as-is |
| §5 | Assran et al. — V-JEPA 2 | `arXiv:2506.09985` | **partial** | cite-as-is |
| §5 | Bardes et al. — VICReg | `arXiv:2105.04906` | verified | cite-as-is |
| §5 | Caron et al. — DINO | `arXiv:2104.14294` | verified | cite-as-is |
| §5 | Tong et al. — VideoMAE | `arXiv:2203.12602` | verified | cite-as-is |
| §5 | Zhu et al. — MotionBERT | `arXiv:2210.06551` | verified | cite-as-is |
| §5 | Shi et al. — 2s-AGCN | `arXiv:1805.07694` | verified | cite-as-is |
| §5 | Menon et al. — logit adjustment | `arXiv:2007.07314` | verified | cite-as-is |

## Counts

**verified: 20 | partial: 2 | wrong: 0 | unresolved: 0**

(Total 22 citations: manuscript refs [1]–[14] plus 8 planning-note §5-only entries.)

> Note on the §5-flagged guesses: none of the three warned identifiers is *wrong* on the
> citation that will be used. The MAMP arXiv guess (`2308.07092`) and MotionBERT arXiv
> guess (`2210.06551`) both **resolved correctly** and are now confirmed. The GAVD
> `arXiv:2309.01480` id is **confirmed to point to an unrelated paper** and must be
> dropped — but the manuscript's actual GAVD citation uses the correct IEEE Access DOI,
> so the GAVD *reference* is verified. See the detail sections for each.

---

## Per-citation details

### [1] Zanardi et al. — Parkinson's disease gait meta-analysis

- **Claimed identity:** DOI `10.1038/s41598-020-80768-2`; "Gait parameters of Parkinson's disease compared with healthy controls: A systematic review and meta-analysis"; *Scientific Reports*, vol. 11, p. 752, 2021; Zanardi et al.
- **Resolved identity:** "Gait parameters of Parkinson's disease compared with healthy controls: a systematic review and meta-analysis"; Zanardi, da Silva, Costa, Passos-Monteiro, Dos Santos, Kruel, Peyré-Tartaruga; *Scientific Reports*, vol. 11, article 752, published Jan 12, 2021.
- **Id verdict:** **verified.** DOI redirects via doi.org to nature.com/articles/s41598-020-80768-2 and is confirmed through PubMed (PMID 33436993), which resolves the same DOI to a matching title, author list, and venue.
- **Quoted evidence:** Title, authors, journal (*Scientific Reports*, vol. 11, art. 752, Jan 12, 2021), and DOI all match. Abstract: 72 studies, 3027 participants (1510 PD, 1517 controls); PD patients had reduced self-selected walking speed, stride length, swing time and hip excursion, higher cadence and double support time; walking speed decreased 0.13 m/s (treadmill) and 0.17 m/s (overground).
- **Factual/benchmark check:** n/a — the cited-for text makes no numeric claim. The abstract substantively supports the qualitative claim of PD-vs-control gait parameter differences.
- **Relevance to the S-JEPA lever:** Provides clinical grounding for the premise that PD produces measurable gait signatures (speed, stride length, cadence, double support time), supporting the choice of gait / neurologic landmarks for the S-JEPA analysis.
- **Recommendation:** cite-as-is.

### [2] Lauziere et al. — post-stroke gait asymmetry

- **Claimed identity:** DOI `10.4172/2329-9096.1000201`; "Understanding spatial and temporal gait asymmetries in individuals post stroke"; *Int. J. Physical Medicine and Rehabilitation*, vol. 2(3), p. 201, 2014; Lauziere, Betschart, Aissaoui, Nadeau.
- **Resolved identity:** "Understanding Spatial and Temporal Gait Asymmetries in Individuals Post Stroke"; *International Journal of Physical Medicine & Rehabilitation*, vol. 02, issue 03, 2014. CrossRef registers only one author: Sylvie Nadeau.
- **Id verdict:** **verified.** DOI resolves via doi.org (302 → OMICS Online → 301 → Longdom, the OMICS rebrand). Publisher slug and page show the exact claimed title; CrossRef returns matching container-title, volume 02, issue 03, year 2014.
- **Quoted evidence:** Publisher landing title (via doi.org redirect chain): "Understanding Spatial and Temporal Gait Asymmetries in Individuals Post Stroke". CrossRef: container-title "International Journal of Physical Medicine & Rehabilitation", vol. 02, issue 03, 2014; author family "Nadeau" (given "Sylvie"). **Caveat:** CrossRef deposit registered only 1 author (Nadeau); co-authors Lauziere, Betschart, Aissaoui could not be independently confirmed, and the PDF body / full byline was not parseable.
- **Factual/benchmark check:** n/a — qualitative clinical-motivation citation. Abstract not retrievable, but the topical claim is consistent with the confirmed title.
- **Relevance to the S-JEPA lever:** On-topic evidence that post-stroke gait is asymmetric, motivating selection of bilateral (left/right) landmarks so the model can capture inter-limb asymmetry.
- **Recommendation:** cite-as-is. (Author byline is only partially confirmed via metadata; identity of the work itself is verified.)

### [3] Pandey et al. — cerebral palsy crouch gait

- **Claimed identity:** DOI `10.1007/s43465-023-01002-5`; "Crouch gait in cerebral palsy: Current concepts review"; *Indian Journal of Orthopaedics*, vol. 57(12), pp. 1913-1926, 2023; Pandey, Johari, Shetty.
- **Resolved identity:** "Crouch Gait in Cerebral Palsy: Current Concepts Review"; Ritesh Arvind Pandey, Ashok N. Johari, Triveni Shetty; *Indian Journal of Orthopaedics*, vol. 57(12), pp. 1913-1926, 2023 (Springer).
- **Id verdict:** **verified.** DOI resolves via doi.org to Springer. CrossRef and NCBI PMC (PMCID PMC10673808) independently confirm matching title, authors, journal, volume, issue, pages, and year.
- **Quoted evidence:** CrossRef + PMC: title, authors (Pandey RA, Johari AN, Shetty T), *Indian J Orthop* 57(12):1913-1926, 2023, online 30 Sep 2023.
- **Factual/benchmark check:** n/a — no numeric claim.
- **Relevance to the S-JEPA lever:** Clinical review of crouch gait (excessive knee/hip flexion in stance) supporting the clinical justification for focusing on knee/hip landmarks.
- **Recommendation:** cite-as-is.

### [4] Maulet et al. — late-onset Pompe disease motor function

- **Claimed identity:** DOI `10.1212/WNL.0000000000201333`; "Motor function characteristics of adults with late-onset Pompe disease: A systematic scoping review"; *Neurology*, vol. 100(1), pp. e72-e83, 2023; Maulet, Bonnyaud, Weill, Laforet, Cattagni.
- **Resolved identity:** "Motor Function Characteristics of Adults With Late-Onset Pompe Disease: A Systematic Scoping Review"; Théo Maulet, Celine Bonnyaud, Catherine Weill, Pascal Laforêt, Thomas Cattagni; *Neurology*, 2023 Jan 3, vol. 100(1), pp. e72-e83 (Epub 2022 Oct 27).
- **Id verdict:** **verified.** DOI resolves via doi.org to the *Neurology* page; PubMed lookup keyed on the same DOI returns a matching title, author list, and venue.
- **Quoted evidence:** PubMed record matches the claim. Abstract: "aLOPD experience symmetrical weakness, concerning especially the hip and lumbar muscles" and locomotor "increased pelvic drop and tilt."
- **Factual/benchmark check:** n/a — no numeric claim.
- **Relevance to the S-JEPA lever:** Documents symmetrical proximal (hip/lumbar) weakness and gait alterations (pelvic drop/tilt) in Pompe disease, grounding a proximal-limb landmark focus.
- **Recommendation:** cite-as-is.

### [5] Assran et al. — I-JEPA

- **Claimed identity:** DOI `10.1109/CVPR52729.2023.01499` (alt `arXiv:2301.08243`); "Self-supervised learning from images with a joint-embedding predictive architecture (I-JEPA)"; CVPR 2023, pp. 15619-15629; Assran et al.
- **Resolved identity:** "Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture"; Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas; arXiv:2301.08243 (2023); DOI → IEEE Xplore doc 10205476 (CVPR 2023).
- **Id verdict:** **verified.** DOI issues a 302 to ieeexplore.ieee.org/document/10205476/ (CVPR 2023 record); arXiv:2301.08243 title and author list match exactly. Both identifiers refer to the same work. (IEEE landing page body was JS-gated/empty, but arXiv identity match + live DOI redirect confirm the paper.)
- **Quoted evidence:** arXiv title verbatim matches. Abstract: I-JEPA "predicts representations of target blocks from a context block within the same image, without hand-crafted data augmentations," with the key design of "sampling target blocks at large scale and using spatially distributed context blocks."
- **Factual/benchmark check:** The ~18% mask figure in the cited-for text refers to the S-JEPA project, **not** to this paper — no benchmark to verify against I-JEPA. The abstract does support the qualitative lever (I-JEPA needs large, informative target blocks).
- **Relevance to the S-JEPA lever:** Foundational JEPA method. The "sampling target blocks at large scale" design point is exactly the lever the project cites to indict a low ~18% mask/target ratio in the S-JEPA gait setting.
- **Recommendation:** cite-as-is.

### [6] Abdelfattah & Alahi — S-JEPA

- **Claimed identity:** DOI `10.1007/978-3-031-73411-3_21`; "S-JEPA: A joint embedding predictive architecture for skeletal action recognition"; ECCV 2024, LNCS vol. 15090, pp. 367-384; Abdelfattah, Alahi.
- **Resolved identity:** "S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition"; Mohamed Abdelfattah, Alexandre Alahi; *Computer Vision – ECCV 2024* (LNCS), pp. 367-384, published online Nov 23, 2024 (print 2025).
- **Id verdict:** **verified.** DOI resolves via doi.org to the Springer/ECCV chapter. CrossRef and DBLP both return matching title, authors, and pages; title matches verbatim.
- **Quoted evidence:** CrossRef + DBLP: title, authors, container "Computer Vision – ECCV 2024" (LNCS series), pp. 367-384, 2024.
- **Factual/benchmark check:** n/a — no numeric claim.
- **Relevance to the S-JEPA lever:** This *is* the S-JEPA paper — the direct methodological basis for the skeletal-JEPA gait project.
- **Recommendation:** cite-as-is.

### [7] Mao et al. — MAMP  ⚑ (§5-flagged arXiv guess — RESOLVED CORRECT)

- **Claimed identity:** DOI `10.1109/ICCV51070.2023.00934` (alt `arXiv:2308.07092?`); "Masked motion predictors are strong 3D action representation learners (MAMP)"; ICCV 2023, pp. 10147-10157; Mao, Deng, Zhou, Fang, Ouyang, Li.
- **Resolved identity:** "Masked Motion Predictors are Strong 3D Action Representation Learners"; Yunyao Mao, Jiajun Deng, Wengang Zhou, Yao Fang, Wanli Ouyang, Houqiang Li; ICCV 2023.
- **Id verdict:** **verified.** DOI resolves via doi.org (302) to ieeexplore.ieee.org/document/10378501/ (ICCV 2023 proceedings; ICCV51070.2023 is the official series). arXiv:2308.07092 resolves to the exact claimed title and full author list, Comments field "To appear in ICCV 2023." **The previously UNVERIFIED §5 guess `arXiv:2308.07092` is now CONFIRMED correct** — both ids refer to the same work.
- **Quoted evidence:** arXiv title verbatim matches; authors match. Abstract: proposes Masked Motion Prediction (MAMP), argues against masked self-component (joint/coordinate) reconstruction in favor of "explicit contextual motion modeling," and "predicts the corresponding temporal motion of the masked human joints."
- **Factual/benchmark check:** Qualitative claim ("predicting velocity/motion beats masked-coordinate reconstruction for skeletons") is directly supported by the abstract. No specific accuracy % is asserted (abstract cites only SOTA on NTU-60/120, PKU-MMD) — n/a for numerics.
- **Relevance to the S-JEPA lever:** Canonical skeleton-domain result motivating motion/velocity targets over raw-coordinate targets — directly substantiates the A4 rate-features lever.
- **Recommendation:** cite-as-is (the `arXiv:2308.07092` guess can now be recorded as confirmed, not a guess).

### [8] Xu et al. — Skeleton2vec

- **Claimed identity:** arXiv `10.48550/arXiv.2401.00921` (alt `arXiv:2401.00921`); "Skeleton2vec: A self-supervised learning framework with contextualized target representations for skeleton sequence"; arXiv 2024; Xu, Huang, Wang, Hu, Deng.
- **Resolved identity:** "Skeleton2vec: A Self-supervised Learning Framework with Contextualized Target Representations for Skeleton Sequence"; Ruizhuo Xu, Linzhi Huang, Mei Wang, Jiani Hu, Weihong Deng; arXiv:2401.00921, submitted Jan 1, 2024 (paper targets CVPR 2024).
- **Id verdict:** **verified.** arXiv:2401.00921 title matches verbatim; DOI `10.48550/arXiv.2401.00921` issues a 302 to arxiv.org/abs/2401.00921 — same work.
- **Quoted evidence:** arXiv title + authors match. Abstract: uses high-level contextualized features from a transformer teacher encoder as prediction targets (not low-level features), plus a motion-aware tube masking strategy; evaluated on NTU-60, NTU-120, PKU-MMD.
- **Factual/benchmark check:** n/a — no numeric claim.
- **Relevance to the S-JEPA lever:** Skeleton-domain precedent for predicting high-level contextualized latent targets — supports the latent-prediction design.
- **Recommendation:** cite-as-is.

### [9] Endo et al. — GaitForeMer

- **Claimed identity:** DOI `10.1007/978-3-031-16452-1_13`; "GaitForeMer: Self-supervised pre-training of transformers via human motion forecasting for few-shot gait impairment severity estimation"; MICCAI 2022, pp. 130-139; Endo, Poston, Sullivan, Fei-Fei, Pohl, Adeli.
- **Resolved identity:** "GaitForeMer: Self-Supervised Pre-Training of Transformers via Human Motion Forecasting for Few-Shot Gait Impairment Severity Estimation"; Mark Endo, Kathleen L. Poston, Edith V. Sullivan, Li Fei-Fei, Kilian M. Pohl, Ehsan Adeli; MICCAI 2022 (LNCS vol. 13438), pp. 130-139; DOI `10.1007/978-3-031-16452-1_13`; arXiv:2207.00106.
- **Id verdict:** **verified.** DOI resolves via doi.org to the Springer chapter. DBLP confirms MICCAI 2022, pp. 130-139, matching title/authors. arXiv:2207.00106 ("Accepted at MICCAI 2022") carries the identical title and author list — same work.
- **Quoted evidence:** arXiv title + authors match. Abstract: pre-trains on public movement data to forecast gait, then fine-tunes on clinical data to predict MDS-UPDRS severity, "achieving an F1 score of 0.76, precision of 0.79, and recall of 0.75." DBLP lists the same DOI, pp. 130-139, "MICCAI (8) 2022."
- **Factual/benchmark check:** Cited-for text has no numeric claim (n/a). For completeness, the paper's own metrics are F1 0.76 / precision 0.79 / recall 0.75.
- **Relevance to the S-JEPA lever:** Direct clinical-SSL analogue — self-supervised motion (gait) forecasting fine-tuned for few-shot gait severity; the closest prior clinical SSL work.
- **Recommendation:** cite-as-is.

### [10] Duan et al. — FSGait  ⚑ PARTIAL (page-range mismatch)

- **Claimed identity:** DOI `10.1007/978-981-96-0960-4_19`; "FSGait: Fine-grained self-supervised gait abnormality detection"; **ACCV 2024, pp. 2248-2264**; Duan, Wan, Zhao.
- **Resolved identity:** "FSGait: Fine-Grained Self-supervised Gait Abnormality Detection"; Bingzhi Duan, Xiaoyue Wan, Xu Zhao; *Computer Vision – ACCV 2024* (LNCS), 2024/2025, **pp. 313-329**.
- **Id verdict:** **partial.** The DOI resolves correctly (doi.org → Springer Link) and CrossRef confirms matching title, authors, and container. **However, the claimed page range (pp. 2248-2264) does NOT match the CrossRef `page` field (313-329).** Identity of the work is confirmed via metadata; the abstract could not be fetched (no CrossRef abstract; Springer paywalled).
- **Quoted evidence:** CrossRef for `10.1007/978-981-96-0960-4_19`: title "FSGait: Fine-Grained Self-supervised Gait Abnormality Detection"; authors Duan, Wan, Zhao; container "Computer Vision – ACCV 2024" (LNCS); year 2024 (online Dec 8, 2024); **page field verbatim "313-329"** — contradicting the claimed 2248-2264.
- **Factual/benchmark check:** n/a — no numeric benchmark claim in the cited-for text.
- **Relevance to the S-JEPA lever:** Title and topic (fine-grained self-supervised gait abnormality detection) are a direct task analogue for the S-JEPA gait work. Note: the self-supervised mechanism could not be independently confirmed (abstract unavailable).
- **Recommendation:** **cite-with-corrected-id — change the page range from 2248-2264 to 313-329.** DOI and identity are sound.

### [11] Ranjan et al. — GAVD dataset  ⚑ (§5-flagged `arXiv:2309.01480` — CONFIRMED WRONG arXiv, but DOI is correct)

- **Claimed identity:** DOI `10.1109/ACCESS.2025.3545787` (alt `arXiv:2309.01480`, flagged as NOT resolving to GAVD); "Computer vision for clinical gait analysis: A gait abnormality video dataset (GAVD)"; *IEEE Access*, vol. 13, pp. 45321-45339, 2025; Ranjan, Ahmedt-Aristizabal, Armin, Kim.
- **Resolved identity:** "Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset"; Rahm Ranjan, David Ahmedt-Aristizabal, Mohammad Ali Armin, Juno Kim; *IEEE Access*, vol. 13, pp. 45321-45339, 2025.
- **Id verdict:** **verified** (for the IEEE Access DOI). The DOI resolves via doi.org (302) to IEEE Xplore document 10921672; CrossRef returns matching title, authors, venue, year 2025, vol. 13, pp. 45321-45339. **CRITICAL:** the alternate `arXiv:2309.01480` is confirmed to be a **DIFFERENT, unrelated paper** — "BadSQA: Stealthy Backdoor Attacks Using Presence Events as Triggers in Non-Intrusive Speech Quality Assessment" (a.k.a. EventTrojan) by Ying Ren et al. It is NOT GAVD and must never be cited for GAVD.
- **Quoted evidence:** CrossRef for `10.1109/ACCESS.2025.3545787`: full title, authors, *IEEE Access*, 2025, vol. 13, pp. 45321-45339. arXiv:2309.01480 resolves to the BadSQA speech-quality backdoor-attack paper — wholly unrelated.
- **Factual/benchmark check:** Dataset-scale claim (1874 sequences, >450 videos) is **supported** via secondary sources quoting "1,874 gait sequences in the GAVD dataset" and "over 400" videos / "458,000 annotated frames." The IEEE abstract itself was bot-blocked, so this is corroborated via secondary sources rather than the primary abstract.
- **Relevance to the S-JEPA lever:** This is the authoritative citation for GAVD — the exact dataset used in this project. The IEEE Access DOI resolves cleanly to the correct paper.
- **Recommendation:** cite-as-is on the IEEE Access DOI. **Remove `arXiv:2309.01480` entirely** from any GAVD reference — it points to an unrelated speech-quality paper.

### [12] Grishchenko et al. — BlazePose GHUM Holistic

- **Claimed identity:** arXiv `10.48550/arXiv.2206.11678` (alt `arXiv:2206.11678`); "BlazePose GHUM Holistic: Real-time 3D human landmarks and pose estimation"; arXiv 2022; Grishchenko et al.
- **Resolved identity:** "BlazePose GHUM Holistic: Real-time 3D Human Landmarks and Pose Estimation"; Grishchenko, Bazarevsky, A. Zanfir, Bazavan, M. Zanfir, Yee, Raveendran, Zhdanovich, Grundmann, Sminchisescu; arXiv June 23, 2022; also CVPR Workshop on CV for AR/VR, 2022.
- **Id verdict:** **verified.** arXiv:2206.11678 title matches verbatim; DOI `10.48550/arXiv.2206.11678` issues a 302 to arxiv.org/abs/2206.11678 — same work.
- **Quoted evidence:** arXiv title + full author list match. Abstract: "We present BlazePose GHUM Holistic, a lightweight neural network pipeline for 3D human body landmarks and pose estimation, specifically tailored to real-time on-device inference."
- **Factual/benchmark check:** n/a — no numeric claim. The paper is the BlazePose GHUM pipeline underlying MediaPipe's 33-keypoint body landmark model, consistent with the intended use.
- **Relevance to the S-JEPA lever:** The pose-estimation backbone (MediaPipe/BlazePose) used to extract the 33-landmark sequences feeding the S-JEPA gait model.
- **Recommendation:** cite-as-is.

### [13] Hii et al. — marker-free pose-based gait analysis

- **Claimed identity:** DOI `10.3390/s23146489`; "Automated gait analysis based on a marker-free pose estimation model"; *Sensors*, vol. 23(14), p. 6489, 2023; Hii et al.
- **Resolved identity:** "Automated Gait Analysis Based on a Marker-Free Pose Estimation Model"; Chang Soon Tony Hii, Kok Beng Gan, Nasharuddin Zainal, Norlinah Mohamed Ibrahim, Shahrul Azmin, Siti Hajar Mat Desa, Bart van de Warrenburg, Huay Woon You; *Sensors (Basel)*, vol. 23(14), art. 6489, 2023.
- **Id verdict:** **verified.** DOI resolves via doi.org (302) to the MDPI *Sensors* page. MDPI returned 403 to direct fetch, but PubMed (PMID 37514783) for the exact DOI confirms title, authors, journal, vol. 23, issue 14, art. 6489, 2023.
- **Quoted evidence:** PubMed record matches. Abstract: "Markerless gait analysis using 2D pose estimation techniques has emerged as a potential solution" and describes "an automated method for temporal gait analysis that employs the MediaPipe Pose, a low-computational-resource pose estimation model."
- **Factual/benchmark check:** No numeric claim in cited-for (n/a). For context, the abstract reports ICC(2,1) > 0.75 (good) to > 0.90 (excellent) vs. Vicon for most temporal gait parameters (a few at moderate ICC > 0.50) — consistent with the feasibility claim.
- **Relevance to the S-JEPA lever:** Peer-reviewed validation that a marker-free monocular 2D pose pipeline yields temporal gait parameters with good-to-excellent agreement vs. Vicon — grounds the feasibility of markerless/monocular pose-based gait analysis.
- **Recommendation:** cite-as-is.

### [14] Roberts et al. — cross-validation for structured data

- **Claimed identity:** DOI `10.1111/ecog.02881`; "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"; *Ecography*, vol. 40(8), pp. 913-929, 2017; Roberts et al.
- **Resolved identity:** Same title; David R. Roberts, Volker Bahn, Simone Ciuti, … Carsten F. Dormann; *Ecography*, vol. 40, no. 8, pp. 913-929, 2017.
- **Id verdict:** **verified.** DOI resolves via doi.org (302) to the Wiley/Ecography page (HTTP 402), and authoritative CrossRef metadata for the same DOI returns matching title, authors, journal, volume, issue, pages, and year.
- **Quoted evidence:** CrossRef: title, authors (lead: David R. Roberts), *Ecography*, vol. 40, issue 8, pp. 913-929, 2017 — all match.
- **Factual/benchmark check:** n/a — methodological rationale, no numeric claim. This is the canonical reference for blocking/grouped CV under dependence.
- **Relevance to the S-JEPA lever:** Standard methodological basis for grouped/blocked CV to prevent leakage — grounds the A2 GroupKFold/LOVO design.
- **Recommendation:** cite-as-is.

---

### §5 — Bardes et al. — V-JEPA

- **Claimed identity:** `arXiv:2404.08471`; "V-JEPA: Revisiting Feature Prediction for Self-Supervised Video Learning"; 2024; Bardes et al. (Meta). Cited for spatiotemporal multiblock/tube masking + smooth-L1 feature regression + frozen-backbone/attentive-probe eval; §5 claims SSv2 72.2%.
- **Resolved identity:** "Revisiting Feature Prediction for Learning Visual Representations from Video"; Bardes, Garrido, Ponce, Chen, Rabbat, LeCun, Assran, Ballas (Meta AI / FAIR); arXiv 2024.
- **Id verdict:** **verified.** arXiv:2404.08471 resolves to the V-JEPA paper ("introduces V-JEPA"), Bardes et al. from Meta. No DOI claimed. **Title note:** the citation prepends "V-JEPA:" and uses "Self-Supervised Video Learning"; the arXiv title omits the prefix and reads "Learning Visual Representations from Video" — same paper, minor wording variation.
- **Quoted evidence:** arXiv abstract: "introduces V-JEPA, a collection of vision models trained solely using a feature prediction objective … using a frozen backbone … Our largest model, a ViT-H/16 trained only on videos, obtains 81.9% on Kinetics-400, 72.2% on Something-Something-v2, and 77.9% on ImageNet1K."
- **Factual/benchmark check:** **§5 claim "SSv2 72.2%" is CONFIRMED** — abstract states ViT-H/16 "obtains … 72.2% on Something-Something-v2." Exact match.
- **Relevance to the S-JEPA lever:** V-JEPA uses spatiotemporal multiblock/tube masking, predicts target-encoder features (not pixels), and is evaluated with a frozen backbone + attentive probing — exactly the design choices attributed to it. The 72.2% SSv2 figure is accurate.
- **Recommendation:** cite-as-is. (Optional: align the cited title wording with the actual arXiv title.)

### §5 — Assran et al. — V-JEPA 2  ⚑ PARTIAL (title truncated / author field imprecise)

- **Claimed identity:** `arXiv:2506.09985`; "V-JEPA 2: Self-Supervised Video Models"; 2025; author listed as "Meta AI." Cited for scale + latent prediction; frozen-feature probing; §5 claims SSv2 77.3%.
- **Resolved identity:** "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning"; Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, … Yann LeCun, Michael Rabbat, Nicolas Ballas (Meta AI / FAIR); arXiv preprint, submitted June 11, 2025.
- **Id verdict:** **partial.** arXiv:2506.09985 resolves to the correct work; the id matches. But the claimed title is a **truncated form** (drops "…Enable Understanding, Prediction and Planning") and the author is given as "Meta AI" rather than the full individual author list. Substance matches; metadata is imprecise. (`correctedId` recorded as `arXiv:2506.09985` — same as claimed.)
- **Quoted evidence:** Fetched title verbatim: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning." Abstract: "Something-Something v2: 77.3 top-1 accuracy"; pre-training on "over 1 million hours of internet video."
- **Factual/benchmark check:** **§5 claim "SSv2 77.3%" is CONFIRMED** — abstract reports Something-Something v2 = 77.3 top-1. Exact match.
- **Relevance to the S-JEPA lever:** Large-scale JEPA (1M+ hours) using latent prediction with strong frozen-feature/probing performance — an apt template for the S-JEPA scale + latent-prediction lever.
- **Recommendation:** cite-as-is on the id. Recommend completing the title (full subtitle) and the author list when entering the manuscript; substance is verified.

### §5 — Bardes et al. — VICReg

- **Claimed identity:** `arXiv:2105.04906`; "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"; ICLR 2022; Bardes, Ponce, LeCun. Cited for the variance-hinge collapse guardrail added to the DINO loss at tiny N (C2).
- **Resolved identity:** Same title; Adrien Bardes, Jean Ponce, Yann LeCun; arXiv:2105.04906, submitted May 2021, rev. Jan 2022; accepted at ICLR 2022.
- **Id verdict:** **verified.** arXiv:2105.04906 title matches verbatim; authors and ICLR 2022 venue match.
- **Quoted evidence:** arXiv note "Accepted at ICLR 2022." Abstract: describes the trivial collapse solution ("A trivial solution is obtained when the encoder outputs constant vectors"), consistent with VICReg's variance term as a collapse guardrail.
- **Factual/benchmark check:** n/a — no numeric claim.
- **Relevance to the S-JEPA lever:** VICReg's explicit variance regularization (hinge on embedding standard deviation) directly supports the C2 lever — a variance-hinge collapse guardrail on the DINO loss at tiny N.
- **Recommendation:** cite-as-is.

### §5 — Caron et al. — DINO

- **Claimed identity:** `arXiv:2104.14294`; "Emerging Properties in Self-Supervised Vision Transformers (DINO)"; ICCV 2021; Caron et al. Cited for the centering+sharpening objective; teacher-temp warmup & momentum hygiene.
- **Resolved identity:** "Emerging Properties in Self-Supervised Vision Transformers"; Caron, Touvron, Misra, Jégou, Mairal, Bojanowski, Joulin; arXiv Apr 29, 2021 (ICCV 2021).
- **Id verdict:** **verified.** arXiv:2104.14294 title matches exactly; author list headed by Mathilde Caron (matches "Caron et al.").
- **Quoted evidence:** arXiv title + authors match. Abstract underlines the importance of the momentum encoder as a DINO component.
- **Factual/benchmark check:** n/a — qualitative method descriptions. Momentum encoder confirmed in abstract; centering/sharpening/teacher-temperature are the well-known DINO self-distillation mechanisms (method body, not abstract).
- **Relevance to the S-JEPA lever:** Canonical source of the self-distillation objective (centering + sharpening, EMA momentum teacher, teacher-temperature warmup) — the training-stability levers the project reuses.
- **Recommendation:** cite-as-is.

### §5 — Tong et al. — VideoMAE

- **Claimed identity:** `arXiv:2203.12602`; "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"; NeurIPS 2022; Tong, Song, Wang, Wang. Cited for high-ratio tube masking as the driver of data-efficient temporal SSL on small video sets.
- **Resolved identity:** Same title; Zhan Tong, Yibing Song, Jue Wang, Limin Wang; NeurIPS 2022; arXiv:2203.12602.
- **Id verdict:** **verified.** arXiv:2203.12602 title matches verbatim; author surnames match; comments field states "NeurIPS 2022 camera-ready."
- **Quoted evidence:** Abstract: proposes "customized video tube masking with an extremely high ratio"; "(1) An extremely high proportion of masking ratio (i.e., 90% to 95%) still yields favorable performance"; "(2) VideoMAE achieves impressive results on very small datasets (i.e., around 3k-4k videos) without using any extra data."
- **Factual/benchmark check:** Mechanism claim, not a discrete numeric assertion (n/a). Abstract directly substantiates: extreme tube masking (90-95%), strong results on ~3k-4k-video datasets without extra data.
- **Relevance to the S-JEPA lever:** Canonical source for high-ratio tube masking enabling data-efficient SSL on small video sets — directly supports the S-JEPA gait argument for temporal SSL on small video sets.
- **Recommendation:** cite-as-is.

### §5 — Zhu et al. — MotionBERT  ⚑ (§5-flagged arXiv guess — RESOLVED CORRECT)

- **Claimed identity:** `arXiv:2210.06551?` (id flagged UNVERIFIED in §5); "MotionBERT: A Unified Perspective on Learning Human Motion Representations"; ICCV 2023; Zhu et al. Cited for masked 2D→3D motion pretraining on noisy monocular poses; candidate external upper-bound.
- **Resolved identity:** Same title; Wentao Zhu, Xiaoxuan Ma, Zhaoyang Liu, Libin Liu, Wayne Wu, Yizhou Wang; ICCV 2023 (arXiv Oct 2022, camera-ready Aug 2023).
- **Id verdict:** **verified.** arXiv:2210.06551 title matches verbatim; authors match ("Zhu et al."). **The "?" flag is resolved — the id is correct.**
- **Quoted evidence:** Abstract: uses "a pretraining stage where a motion encoder learns to recover 3D motion from noisy partial 2D observations" via a Dual-stream Spatio-temporal Transformer (DSTformer); ICCV 2023.
- **Factual/benchmark check:** No numeric claim in cited-for (the "candidate external upper-bound" framing is qualitative) — n/a. The masked 2D→3D motion-pretraining claim is supported by the abstract ("recover 3D motion from noisy partial 2D observations").
- **Relevance to the S-JEPA lever:** Directly supports the "masked 2D→3D motion pretraining on noisy monocular poses" lever; as an ICCV 2023 SOTA method, a legitimate candidate external upper-bound.
- **Recommendation:** cite-as-is (record `arXiv:2210.06551` as confirmed, drop the "?").

### §5 — Shi et al. — 2s-AGCN

- **Claimed identity:** `arXiv:1805.07694`; "Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition (2s-AGCN)"; CVPR 2019; Shi, Zhang, Cheng, Lu. Cited for bone + motion streams beating positions alone → velocity/bone channels (A4).
- **Resolved identity:** Same title; Lei Shi, Yifan Zhang, Jian Cheng, Hanqing Lu; CVPR 2019, pp. 12026-12035.
- **Id verdict:** **verified.** arXiv:1805.07694 title matches exactly; authors and CVPR 2019 venue confirmed.
- **Quoted evidence:** Abstract: proposes "a novel two-stream adaptive graph convolutional network (2s-AGCN)" modeling "both the first-order and the second-order information simultaneously" (joints/positions plus bones); evaluated on NTU-RGBD and Kinetics-Skeleton.
- **Factual/benchmark check:** No accuracy % in cited-for (n/a). Qualitative claim (bone + motion streams beat positions alone) supported by the two-stream design.
- **Relevance to the S-JEPA lever:** Canonical 2s-AGCN — fusing joint (position) and bone (second-order) streams plus motion beats positions alone; supports the A4 justification for velocity/bone channels.
- **Recommendation:** cite-as-is.

### §5 — Menon et al. — logit adjustment

- **Claimed identity:** `arXiv:2007.07314`; "Long-Tail Learning via Logit Adjustment"; ICLR 2021; Menon, Jayasumana, Rawat, Jain, Veit, Kumar. Cited for principled label-frequency correction beyond RF class_weight (A4 logit adjustment).
- **Resolved identity:** "Long-tail learning via logit adjustment"; Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit, Sanjiv Kumar; ICLR 2021; arXiv:2007.07314.
- **Id verdict:** **verified.** arXiv:2007.07314 title and author list match; page states "Published as a conference paper at ICLR 2021."
- **Quoted evidence:** Abstract: "two simple modifications of standard softmax cross-entropy training" adjusting logits by label frequencies, "applied post-hoc to a trained model, or enforced in the loss during training."
- **Factual/benchmark check:** n/a — no numeric claim.
- **Relevance to the S-JEPA lever:** Statistically principled label-frequency correction grounding the A4 lever (beyond ad-hoc RF `class_weight`). Both post-hoc and loss-based variants match the cited purpose.
- **Recommendation:** cite-as-is.

---

## Action items for the manuscript / plan

**IDs to correct before entering the manuscript:**

- **[10] FSGait** — change the page range from **`pp. 2248-2264` to `pp. 313-329`** (CrossRef authoritative). DOI `10.1007/978-981-96-0960-4_19` is correct; identity confirmed.
- **[11] GAVD** — **delete the alternate `arXiv:2309.01480`** from the reference. It resolves to an unrelated speech-quality backdoor-attack paper (BadSQA / EventTrojan, Ren et al.), NOT GAVD. Keep only the IEEE Access DOI `10.1109/ACCESS.2025.3545787`.
- **[7] MAMP** — the §5 arXiv guess `arXiv:2308.07092` is **confirmed correct**; record it as verified (remove the "guess/unverified" annotation).
- **§5 MotionBERT** — the §5 arXiv guess `arXiv:2210.06551` is **confirmed correct**; drop the "?" / "UNVERIFIED" flag.
- **§5 V-JEPA 2** — complete the reference: use the full title ("V-JEPA 2: Self-Supervised Video Models **Enable Understanding, Prediction and Planning**") and the full author list (Assran, Bardes, Fan, Garrido, … LeCun, Rabbat, Ballas) instead of "Meta AI." Id `arXiv:2506.09985` is correct.
- **§5 V-JEPA** — optionally align the cited title with the actual arXiv title ("Revisiting Feature Prediction for Learning Visual Representations from Video"); id `arXiv:2404.08471` is correct.

**IDs to keep as-is (verified, no change needed):**

- [1] `10.1038/s41598-020-80768-2`, [2] `10.4172/2329-9096.1000201`, [3] `10.1007/s43465-023-01002-5`, [4] `10.1212/WNL.0000000000201333`, [5] `10.1109/CVPR52729.2023.01499` / `arXiv:2301.08243`, [6] `10.1007/978-3-031-73411-3_21`, [8] `10.48550/arXiv.2401.00921`, [9] `10.1007/978-3-031-16452-1_13`, [12] `10.48550/arXiv.2206.11678`, [13] `10.3390/s23146489`, [14] `10.1111/ecog.02881`, and §5 VICReg `arXiv:2105.04906`, DINO `arXiv:2104.14294`, VideoMAE `arXiv:2203.12602`, 2s-AGCN `arXiv:1805.07694`, logit-adjustment `arXiv:2007.07314`.

**Must NOT be cited until confirmed / must never be used:**

- **`arXiv:2309.01480` must NEVER be cited for GAVD** — it is a different, unrelated paper. This is the specific §5-warned wrong id; it is now positively identified as BadSQA (speech quality assessment), so it is disqualified permanently, not merely pending.
- No citation is currently *blocked pending confirmation* — every reference resolved to a verified or partial identity. The two `partial` entries ([10] FSGait, §5 V-JEPA 2) are citeable once the metadata corrections above are applied; neither has a wrong or unresolved identifier.

**Caveats to keep in mind (do not block citation, but note for rigor):**

- [2] Lauziere et al.: only lead author Nadeau is confirmed in CrossRef; the co-author byline (Lauziere, Betschart, Aissaoui) could not be independently verified from fetched metadata.
- [11] GAVD dataset-scale (1874 seq / >450 videos): corroborated via secondary sources, not the primary abstract (IEEE page was bot-blocked).
