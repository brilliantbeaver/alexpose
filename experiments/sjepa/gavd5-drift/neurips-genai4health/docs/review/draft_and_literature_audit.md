# Draft, literature, and venue audit

Audit date: 2026-09-05. This review distinguishes facts checked in the local documents from recommendations and from findings requiring the separate notebook/artifact audit. The source package is preserved.

## Recommendation

The strongest defensible GenAI4Health submission is an evidence-grounded position paper about evaluating movement representations before using them in ambient or agentic health systems. The worked GAVD fold supplies a concrete empirical case: the learned representation did not exceed a direct kinematic baseline, and current evidence does not establish clinical validity, functional retention, or a predictive world model. A research-track submission is possible if its central contribution is explicitly a bounded audit and reproducibility case study. It should not promise that incomplete experiments will substantiate its central claim later.

This is a judgment about fit and evidence, not an acceptance prediction. The health relevance lies in preventing unsupported sensor-to-clinical inferences. There is no demonstrated agent, clinical workflow, user study, prospective patient monitoring system, or clinical outcome in the audited drafts.

## Documents examined and version conflicts

| File | Finding |
|---|---|
| `neurips-brain-body/README.md` | Current protocol-v2 cohort, split, and artifact boundary; one executed fold/seed. Explicitly archives earlier metrics. |
| `neurips-brain-body/docs/bbfm2026_paper_draft.tex` | Canonical current manuscript; five main-text pages in its compiled PDF. |
| `neurips-brain-body/docs/bbfm2026_paper_draft.md` | Current Markdown mirror with the same principal counts and worked-fold results. |
| `neurips-brain-body/docs/bbfm2026_paper_draft.pdf` | Six pages: five main-text pages, then appendix/references. Extracted text matches the current manuscript's main scientific content. |
| `neurips-brain-body/docs/neurips-brain-body.md` | September 5 readiness guide, correctly distinguishing protocol-v2 evidence from historical results. |
| `neurips-brain-body/docs/neurips-brain-body.pdf` | **Stale September 3 guide**, despite sharing the current guide's basename. Contains the old 626-sequence/93-source cohort, label-aware 300+75-epoch training, 0.2966 anchor cosine, and transductive macro-F1 0.899. It must not supply current submission results. |
| `docs/references.bib` | Shared bibliography checked for the manuscript's central references; see verification below. |

Both PDFs were text-extracted in full. An eleven-page rendered montage was inspected at `tmp/pdfs/draft_audit/source_pdf_montage.png`; this is a content/version review, not final production QA of the new submission. The old manuscript renders without a conspicuous global layout failure, but the figures should be rebuilt for the new scientific emphasis and venue.

The stale guide is useful historical evidence of earlier shortcomings. Its result values must not be pooled with protocol-v2 values, and its old percentage estimates of submission readiness should be discarded: they were subjective judgments, not measurements.

## Verified workshop requirements

The official [GenAI4Health 2026 call](https://genai4health.github.io/2026-NeurIPS/) lists research papers (up to nine pages), demonstrations (up to five), and position papers (up to five). It does **not** list an extended-abstract track. Page limits exclude acknowledgments, references, and appendices. Use anonymous `\usepackage{neurips_2026}` **without options**; no NeurIPS checklist is required. Review is double blind, with no rebuttal. The deadline is September 5, 2026, 23:59 AoE, equivalent to September 6, 04:59 Los Angeles time. The authorship list must be settled by the deadline. Accepted work is non-archival; concurrent submission must also satisfy the other venue's rules. The page emphasizes complete, supported research claims and evidence-grounded positions that address counterarguments.

The strongest topic alignment is its trust/evaluation area, with ambient and embodied health as the intended application context. Its audience includes ML researchers, healthcare professionals, policy experts, and clinical practitioners. Accordingly, present the requested extended abstract as a companion synopsis or compact position-paper variant, not as a separately advertised submission category. Do not submit near-duplicate variants as separate papers without a justified policy-compliant reason.

## Scientific claim audit

### Highest-priority corrections

1. **Do not call this a generative or agentic system.** I-JEPA explicitly describes itself as non-generative; S-JEPA predicts representations of masked skeleton regions. The connection to GenAI4Health is the evaluation of a possible perception component and the evidentiary risks of downstream use. The application bridge is a reasoned proposal, not an experimental finding.
2. **Do not call masked prediction a demonstrated world model.** No current future-mask-trained checkpoint, action-conditioned dynamics, rollout accuracy, intervention prediction, or planning evaluation is reported. A future-direction paragraph may state what additional evidence would be needed.
3. **Narrow “label-free.”** The encoder loss can omit condition labels while the experiment uses them to define the normal-first curriculum, cumulative stages, stratified partitions, or replay. Say “no condition-label term in the representation loss,” and declare any label-informed scheduling/sampling. The notebook audit must verify the precise replay and checkpoint-selection implementation.
4. **Do not claim a causal effect of fixing leakage.** The current conclusion says the discipline “changes the scientific conclusion,” but the old and new runs differ in cohort, training, label use, and splitting. Without a matched comparison, report that the audited fold does not support learned-latent superiority. Do not calculate a leakage penalty by subtracting historical and current scores.
5. **Separate a registered protocol from completed evidence.** The draft's minimum comparison set includes an untrained encoder, joint training, continued-normal training, multiple seeds, condition order variation, alignment, and functional retention. The manuscript must mark each as completed, unperformed, or non-comparable. Listing a desirable control in present tense can falsely imply it was executed.
6. **Do not equate cosine retention with normal-gait function.** Stage-to-stage raw cosine is coordinate sensitive. Held-out inputs do not change that fact. A normal-anchor value of 0.850 is not 85% retained function or a clinical safety guarantee. Use “representation similarity,” pending alignment-aware and task-level checks.
7. **Keep the estimand modest.** Fold 0/seed 42 and 20 test sources support a descriptive observed contrast. They do not establish expected performance across training seeds, folds, people, institutions, or clinical settings. State the actual uncertainty calculation, if added, and its conditional nature.
8. **Report the local subset, not all of GAVD.** The paper's 666 sequences/103 sources are this project's selected five-category inventory. The full published dataset contains 1,874 sequences. Do not imply that GAVD itself contains only five categories or 103 sources.

### Additional wording and interpretation fixes

| Current wording or implied claim | Recommended treatment |
|---|---|
| “Independent source videos” | “Source-video groups” or “source-level evaluation units”; independence cannot be established from upload IDs alone. |
| Split-figure headline “zero cross-fold leakage” | “Zero source-ID overlap”; residual person overlap and correlated uploads remain possible. |
| “Honest generalization” in the title | Prefer a precise trust/evaluation title that does not imply other work is dishonest. |
| “The same person may cross uploads” | Retain as an unresolved possibility; do not claim repeated identities were measured. |
| “After selection was frozen” / “evaluated once” | Attribute to recorded execution state. Hashes and a checkpoint boolean document lineage; they cannot establish that no investigator previously saw test-derived information. |
| “All thresholds only from training data” | Distinguish data-fitted quantities from fixed per-sequence deterministic transformations and predeclared QC rules; state which actually apply. |
| “Raw kinematics outperformed the latent” | “Raw kinematics had higher observed source-level macro-F1 in this fold”; infer statistical superiority only if an appropriate paired analysis supports it. |
| Missingness-only macro-F1 0.251 | Potential sensor/visibility signal, not proof that the latent exploits missingness or that the performance is above a valid chance/null distribution. Macro-F1 has no universal 0.2 chance baseline under class imbalance. |
| Negative phase-lag R-squared | Evidence that this fitted probe predicts poorly under its implemented metric; not proof that the encoder contains no phase information. |
| Weak timing scores explained by fixed-length resampling | Plausible hypothesis, not causal explanation without a native-duration/resampling ablation. |
| “Clinical labels” | Acknowledge GAVD's clinician-observed annotation process while clarifying that this project did not verify diagnoses or outcomes. Avoid implying that annotations are random folder names. |
| Public skeletons are anonymous | False assurance. Publish aggregate results and assess the release of individual trajectories separately. |

## Contribution and narrative revision

Grouped evaluation and leakage prevention are established methods. “We split by source” alone is insufficient methodological novelty. A defensible contribution is the combination of a dated public-video data boundary, fold-local representation learning, sensor-only comparison, and a claim ledger, demonstrated on an actual small health-related representation-learning pipeline. Make the audited failure to beat a simpler control the empirical reason these distinctions matter.

The argument should progress from a realistic proposed use to an evidence gap. An ambient health assistant might receive a movement representation alongside other information; a convincing latent plot or strong sequence-level score would not tell its designer whether the signal reflects transferable movement, acquisition conditions, or label-informed training. This study makes that uncertainty inspectable. The hypothetical assistant should occupy a small motivation paragraph, not a fabricated system diagram with unevaluated performance claims.

Suggested main-paper flow:

1. State the position and concrete health implication in the introduction.
2. Explain the local GAVD subset, public-availability/decoding/QC gates, and limits of source grouping.
3. Describe the actual representation loss, label use, curriculum, source roles, and readout selection.
4. Present the main source-level comparison, its uncertainty, and the data provenance needed to reproduce it.
5. Use retention and temporal probes only when they support a distinct lesson and their target definitions are clear.
6. Engage counterarguments: a tiny undertrained model is not a test of all JEPAs; missingness may itself correlate with real impairment; source grouping can be conservative while still missing identity overlap; extra reporting costs are justified proportionally to claims.
7. Close with measured implications and specific missing evidence for person-level or clinical use.

Avoid spending most of the paper on download engineering. The data funnel matters because it changes which population is evaluated; technical platform retry details belong in the supplement. Likewise, an unused supervised group-loss equation should not compete visually with the actual method. State the historical distinction briefly unless a current matched ablation is available.

## Results and visual selection

The worked-fold source-level readout comparison is the strongest empirical centerpiece. Retain raw kinematics, the learned latent, and missingness together. Show exact feature/readout definitions, test-source counts per class, macro-F1 and balanced accuracy, and source-level predictions or a reproducible aggregate table. Any post hoc interval calculation must be described as such and tied to frozen predictions.

The corpus funnel is useful but should be compact: 666/103 raw, 657/100 metadata-public, 655/98 decoded, 639/97 pose-QC. Explain that 656/99 is a candidate upper bound before the remaining retryable acquisition failure, not another completed cohort. Counts, hashes, and dates should come from the separate artifact audit.

A two-panel main figure combining the funnel and the baseline comparison may convey more than the current four-panel graphic. Put training-loss telemetry and conditional temporal probes in the appendix unless a precise claim needs them. Do not show a selected heatmap without explaining its rows, target construction, source aggregation, and negative scores.

The policy/evidence visual should distinguish supported claims from prerequisites for stronger ones: observed source-held-out label readability; unresolved transfer across people; untested clinical endpoints; and untested action-conditioned prediction. Do not connect these with arrows suggesting that the present study implements the later stages.

Exclude the stale PDF's 0.899 macro-F1, 0.2966 anchor cosine, historical laterality, AnchorGuard, margin, and predictive-surprise results. Exclusion follows their different or invalid evidentiary boundary, not whether their numerical outcomes are favorable.

## Primary-source reference verification

| Reference | Verification and appropriate use |
|---|---|
| Abdelfattah and Alahi, S-JEPA, ECCV 2024, pp. 367-384 | **Correct**, including the existing title and authors. Verified against the [ECVA conference PDF](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf). Original method uses motion-aware target sampling and centered/sharpened cross-entropy predictions. Fixed landmark eligibility and VICReg are project modifications. |
| Ranjan, Ahmedt-Aristizabal, Ali Armin, and Kim, GAVD, IEEE Access 13:45321-45339, 2025, DOI 10.1109/ACCESS.2025.3545787 | **Correct**. Verified in the [original repository](https://github.com/Rahmyyy/GAVD), [authors' preprint](https://arxiv.org/abs/2407.04190), and [UNSW author publication record](https://research.unsw.edu.au/people/mr-rahm-ranjan/publications). Preserve the compound surname `Ali Armin` in structured BibTeX, e.g. `Ali Armin, Mohammad`. |
| Assran et al., I-JEPA, CVPR 2023 | Verified against the [authors' paper](https://arxiv.org/abs/2301.08243). Supports latent feature prediction and the distinction from reconstruction/generative modeling; does not support a healthcare efficacy claim. |
| Bardes et al., V-JEPA, 2024 | Title/authors and feature-prediction formulation verified in the [authors' paper](https://arxiv.org/abs/2404.08471). Existing TMLR citation is consistent with the publication record; OpenReview's forum currently presents a browser challenge to this tool. |
| Bardes, Ponce, and LeCun, VICReg, ICLR 2022 | Verified in the [authors' paper](https://arxiv.org/abs/2105.04906), which records ICLR 2022 acceptance. It motivates regularization; it does not establish that this implementation cannot collapse. |
| Grishchenko et al., BlazePose GHUM Holistic, 2022 | Correct title/authors/arXiv identifier, verified in the [authors' paper](https://arxiv.org/abs/2206.11678). Distinguish an estimator output from measured clinical kinematics. |
| Roberts et al., Ecography 40:913-929, 2017, DOI 10.1111/ecog.02881 | Established grouped/block cross-validation precedent, verified from the [author-posted paper](https://www.researchgate.net/profile/David-Roberts-52/publication/311523792_Cross-validation_strategies_for_data_with_temporal_spatial_hierarchical_or_phylogenetic_structure/links/5cae1607a6fdcc1d498af333/Cross-validation-strategies-for-data-with-temporal-spatial-hierarchical-or-phylogenetic-structure.pdf). This is prior art, not a reason to label the present evaluation design new in isolation. |

Recommended additions:

- Kapoor and Narayanan, *Leakage and the reproducibility crisis in machine-learning-based science*, Patterns 4:100804, 2023, [primary article](https://doi.org/10.1016/j.patter.2023.100804). Ground the need to enclose preprocessing and representation learning within evaluation boundaries; acknowledge prior model-information documentation.
- Zech et al., *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs*, PLOS Medicine 15:e1002683, 2018, [primary article full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6219764/). Provides direct health precedent for acquisition/site signals masquerading as transferable disease prediction. Do not claim its specific mechanisms were demonstrated in gait data.
- Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025, [authors' paper](https://arxiv.org/abs/2506.09985). Use only to explain why action-conditioned planning requires additional training/evaluation beyond this study's masked representation objective.
- Collins et al., *TRIPOD+AI statement*, BMJ 385:e078378, 2024, [primary guidance](https://www.bmj.com/content/385/bmj-2023-078378). Optional transparency context. Do not claim formal compliance or imply it certifies quality: the guidance itself says it is a reporting guideline, primarily concerns non-generative prediction models, and is not a quality-appraisal instrument.

The [GAVD repository's data-use statement](https://github.com/Rahmyyy/GAVD) explicitly limits its distribution to annotations/metadata and places independent retrieval and compliance responsibilities on researchers. In the manuscript, state the actual project ethics determination if one exists; otherwise accurately say it was not supplied in the reviewed evidence. Do not invent approval, consent, legal permission, clinical collaborators, or released datasets.

## Adversarial acceptance-risk assessment

The central reviewer objection is that the paper could be routine hygiene plus a weak model on a small convenience sample. The best response is specificity: clearly identify what the audit exposed, what an attractive but unsupported interpretation would have been, and how a reader can apply the evidence boundary to a proposed health system. A modest, testable position with a transparent case study is stronger than branding ordinary representation learning as an agentic clinical world model.

The second objection is insufficient health/GenAI relevance. Address it with a limited, concrete sensor-to-decision argument and established clinical shortcut literature. Avoid padding with agents, care planning, or trust language that has no role in the demonstrated evaluation.

The third objection is missing experimental coverage. The current evidence can support a scoped empirical observation and an evaluation position. It cannot support performance rankings for an architecture family, a forgetting mechanism, a repair benefit, or readiness for deployment. State these limits in the abstract, results, and conclusion consistently.

Before upload, choose one track and one canonical PDF; recheck the exact current call; confirm actual authorship, anonymity, external-work overlap rules, data-use status, page count, and legibility. Those are author submission decisions. This audit does not submit or communicate externally.
