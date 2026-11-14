# Venue, rubric and literature evidence

Initially verified 2026-10-08; the live workshop call and official author guide were rechecked 2026-10-09. The regular-paper deadline has been extended as recorded below. This report supports manuscript preparation; no submission has been made.

## Venue requirements and positioning

The [Vision4Neuro call](https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html) describes a track within Vision for Biomedical Discovery. Markerless behavioural phenotyping, transfer across subjects/sessions, domain shift, reproducibility and ethics are explicit interests. The source-grouped gait experiment fits behavioural phenotyping and evaluation under recording heterogeneity. It contains no neural recordings and should not be presented as neural decoding.

| Item | Workshop requirement |
|---|---|
| Regular paper | 8 pages excluding references; archival proceedings |
| Other formats | 4-page short paper or 2-page extended abstract, non-archival |
| Template | Official WACV 2027 author kit |
| Review | Double-blind for all formats |
| Supplement | Optional; outside page limit |
| Submission | OpenReview; select Vision4Neuro |
| Regular deadline | October 16, 2026, 23:59 AoE; extended from October 9 |
| Regular notification | October 30, 2026 |
| Non-archival deadline | November 6, 2026, 23:59 AoE |
| Archival camera-ready | November 20, 2026, 23:59 PT |
| Attendance | At least one registered author presents on site |

The official kit is saved under `../template`; its URL, hash and minor README inconsistencies are recorded in `../template/PROVENANCE.md`. The live [author guide](https://wacv.thecvf.com/Conferences/2027/AuthorGuides) requires an anonymous WACV-format PDF and discusses human-data provenance, consent and approval status. It welcomes writing tools while retaining author responsibility for correctness; tools that materially affect methodology should be described. These are main-conference instructions, read alongside the workshop-specific call rather than substituting main-conference deadlines for the workshop timeline.

## Editorial scoring rubric

**This is a task-specific rubric, not an official numeric workshop rubric or an estimate of acceptance probability.** Its priorities derive from the workshop's scope and the [WACV reviewer guide](https://wacv.thecvf.com/Conferences/2027/ReviewerGuidelines). The latter distinguishes application contributions from algorithmic novelty and explicitly values scientifically sound benchmarking and auditing. The numerical weights below are editorial choices.

Rate each category on 0–5; decimal ratings are allowed. Weighted total = sum(weight × rating / 5). Anchors: 0 absent or materially false; 1 severe unresolved deficiency; 2 substantial revision needed; 3 defensible with important limitations; 4 strong and clearly supported; 5 unusually complete for the claimed scope. Scores must reflect evidence, so successive versions can plateau or decline.

| Category | Weight | Evidence required for a high score |
|---|---:|---|
| Vision4Neuro relevance and practical contribution | 15 | A focused behavioural-vision question and an explicit, useful contribution that the experiment can support. |
| Empirical correctness and uncertainty | 25 | Every result reconstructable from its stated artifact; consistent denominators, metrics and uncertainty; no substituted experiments. |
| Evaluation design and generalization scope | 20 | Source grouping enforced through preprocessing/training/selection; paired comparisons; clear limits on person/domain generalization. |
| Reproducibility and methodological completeness | 15 | Input construction, architecture, sampling, optimization, selection and readout sufficiently specified; provenance and defects disclosed. |
| Clinical interpretation, provenance and ethics | 10 | Collection labels distinguished from diagnosis; identity uncertainty and data-use/consent/approval gaps stated; no unearned clinical readiness. |
| Literature, clarity and WACV presentation | 15 | Verified relevant references, fluent precise writing, readable figures, faithful equations and compliant anonymous layout. |

High total scores cannot override a material unsupported result, hidden leakage, unsupported clinical claim or broken format. This study's unresolved clinical ground truth and data-use documentation cap the clinical-evidence category regardless of prose improvements. The paper can be well written and reproducible while the underlying experiment remains exploratory.

## Verified retained literature and safe claim scope

The bibliography keys below exist in `../references.bib`. Each link is an original paper, publisher/conference page or author-deposited primary manuscript. The broader 57-item source audit is in `reference-audit.md`; it distinguishes existence/metadata verification from support for a specific assertion.

| Key | Primary evidence | Claim supported; boundary |
|---|---|---|
| `blazepose` | [Bazarevsky et al., 2020](https://arxiv.org/abs/2006.10204) | BlazePose estimates 33 landmarks; its paper does not validate this corpus's clinical labels or physical gait quantities. |
| `stenum2021` | [Stenum, Rossi and Roemmich, 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008935) | OpenPose gait parameters compared with simultaneous motion capture in healthy adults; camera geometry matters. Validation is workflow-specific. |
| `stenum2024` | [Stenum et al., 2024](https://journals.plos.org/digitalhealth/article?id=10.1371/journal.pdig.0000467) | Video gait validation across perspectives and clinical populations; does not establish diagnostic classification from web labels. |
| `shin2021` | [Shin et al., 2021](https://journals.sagepub.com/doi/10.3233/JPD-212544) | Video-derived gait measures in PD with prospective walkway comparison; includes specific calibration, exclusions and consent, unlike the current web collection. |
| `filli2018` | [Filli et al., 2018](https://www.nature.com/articles/s41598-018-22676-0) | Heterogeneous MS gait deficits studied with clinical assessment and 3D gait analysis; supports clinical motivation, not a validated disease-specific 12-joint mask. |
| `plotnik2020` | [Plotnik et al., 2020](https://www.nature.com/articles/s41598-020-68263-0) | MS gait asymmetry and bilateral coordination during a six-minute walk; those quantities were not directly measured by this classifier. |
| `sjepa` | [Abdelfattah and Alahi, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | Existing latent masked skeleton prediction method. Motion-aware masks sample high-motion joints with greater probability, rather than deterministically choosing the largest motions. The study adapts this method and cannot claim to originate S-JEPA. |
| `ijepa` | [Assran et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Assran_Self-Supervised_Learning_From_Images_With_a_Joint-Embedding_Predictive_Architecture_CVPR_2023_paper.html) | Image feature prediction with context and momentum target encoders; not evidence about skeletal masking or this clinical application. |
| `vjepa` | [Bardes et al., 2024](https://arxiv.org/abs/2404.08471) | Video feature prediction; TMLR publication confirmed by the journal's [August 2024 announcement](https://groups.google.com/g/tmlr-announce-weekly/c/Bs02CFGTU6g). |
| `skeletonmae` | [Yan et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.html) | Graph-based masked reconstruction of joints/edges. Do not call this cited method a transformer. |
| `motionbert` | [Zhu et al., ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.html) | Transformer motion representation learning through recovery of 3D motion from corrupted 2D observations. Its benchmarks do not validate the present classifier. |
| `varoquaux2018` | [Varoquaux, 2018](https://www.sciencedirect.com/science/article/pii/S1053811917305311) | Small-sample cross-validation uncertainty and dependent folds; supports avoiding independent-fold significance claims. |
| `varma2006` | [Varma and Simon, 2006](https://link.springer.com/article/10.1186/1471-2105-7-91) | Selection inside an evaluation procedure can bias error estimates; selection requires separation from the test fold. |
| `roy2007` (optional) | [Roy and Vetterli, 2007](https://infoscience.epfl.ch/bitstreams/2907ab8a-23f5-481d-bb07-1d56a3f3511f/download) | Effective rank from normalized singular-value entropy; no implication of label information. |
| `han2026` (optional) | [Han et al., npj Digital Medicine](https://www.nature.com/articles/s41746-025-02150-8) | PD video severity assessment with clinical ratings and held-out participants. Published online December 13, 2025; volume 9, article 98 is assigned 2026. Both dates are legitimate in their proper roles. |

## Interpretation constraints

Source separation tests transfer to held-out video IDs within this collection. It does not establish transfer to unseen people, sites, cameras or a clinical deployment distribution. “Normal,” MS and PD are collection labels. Public availability of a source does not establish permission for the current use, clinical consent or research approval. No approval, waiver, consent record or cleared redistribution license was provided in the audited materials; the manuscript should describe this absence without inventing a determination. No clinical diagnostic use is supported.

Similarity to a mean-pose baseline does not identify what the learned representation uses. Temporal shuffling, random-encoder and masking ablations would answer different questions and must remain future work unless actually performed. Known training and RF defects affect the scope of the reported experiment; disclosure does not make their effects disappear. A repair changes the experimental condition and cannot silently inherit cached scores.
