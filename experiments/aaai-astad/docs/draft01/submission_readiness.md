# Submission readiness and a critical assessment

**Assessment date: November 15, 2026.** The real experiments, notebook execution and anonymous manuscript are complete. The draft is technically prepared for a workshop review. Its scientific contribution is an exploratory audit of a prediction-context score. It does not currently support a strong new-detector claim, clinical use, participant-level generalization, or an identified neurological mechanism.

The most useful lesson is concrete: comparing a calibrated regional profile with raw error can produce an appealing point estimate without establishing a benefit from inter-joint subtraction or learned representations. The matched controls, unchanged-coordinate random-feature comparison, learning-adequacy diagnostics and contextual-target probe make that interpretation check reproducible. This is a modest methodological contribution, not a new JEPA architecture.

## Scores and what they mean

These are editorial judgments on a five-point scale: 1 is very weak, 3 is adequate with material limits, and 5 is exceptional. They are not acceptance probabilities. The independent reviews scored the initial real-results manuscript; their scores are preserved rather than silently upgraded after revision.

| Criterion | Claude Opus review | GPT-6 Astra review | Current assessment after revision |
| --- | ---: | ---: | --- |
| Novelty | 2.5 | 2 | 2.5. The controlled combination is useful, but its ingredients and target-context limitation are established ideas. |
| Methodological rigor | 4 | 3 | 3.5. Source roles, matched targets, fixed primary analysis and verified artifacts are strengths. Small references and learning inadequacy remain. |
| Evidence strength | 2 | 3 for the audit | 3 for the bounded audit; 1 for a superior detector or neurological claim. Actual adverse results support the former only. |
| Clarity | 3.5 | 4 | 4. The revision foregrounds the operational finding, unchanged-coordinate control, attrition and uncertainty. |
| Workshop fit | 4.5 | 4 | 4. The question directly concerns spatial and temporal anomaly detection, but the paper must make the methodological lesson useful beyond this one experiment. |
| Technical readiness | 3 | Not scored separately | 4.5. The PDF, references, execution record and vector figures are prepared. Final author and submission-system checks remain. |

The overall conclusion is **a defensible exploratory workshop paper with a substantial risk of a weak-novelty or insufficient-evidence review**. There is no supported basis for calling it a strong learned detector. Finishing more formatting cannot remove that distinction.

## What is complete

| Requirement | Completed evidence |
| --- | --- |
| Fresh GAVD inventory and actual acquisition | 348 annotated sources; 332 decodable videos; category-specific attrition recorded. All video files remain under `data/videos-gavd`. |
| Real normal-only model fitting | Four folds for 512 pretraining plus 512 head updates; a separate four-fold 4,096-head-update sensitivity. |
| Meaningful controls | Own/patch/full contexts, coordinate/latent targets, matched calibrated losses, random encoders with fitted heads, fixed ridge, target-budget and reference sensitivities, motion and observation features. |
| Separate supervised comparison | Normal, Parkinsons, stroke, cerebral palsy and myopathic recognition on the same source folds. The Random Forest has higher macro-F1 in every fold. |
| Artifact integrity | Both real runs pass all eight grouped checks covering 173 required artifacts each. Tensor audit confirms identical pretrained backbones across budgets. |
| Notebooks and tests | All 12 notebooks execute in fresh kernels with real data; 73 code cells and zero error outputs. The experiment test suite passes 105 tests. |
| Real model criticism | Actual Claude CLI responses and independent GPT critiques are archived. A response log verifies their claims rather than treating model agreement as evidence. |
| Manuscript and slides | Anonymous official-style PDF, editable source, bibliography, vector figures and the 40-slide illustrated deck. |

See the [execution validation](../current_execution_validation.md), [visual review](visual_review.md) and [response to reviews](response_to_reviews.md) for the evidence behind these checks.

## What prevents a stronger claim

**The limiting sample is normal gait.** Only 23 normal sources survive the observation rules, and all were present in earlier project development. Each fold uses five reference normals and five or six test normals. More abnormal videos broaden coverage but do not supply an independent normal distribution. Different uploads may also contain the same person.

**The biological question has not received an adequate test.** Full-context coordinate MSE is worse than own-context MSE in all four primary folds and three longer-run folds. Restricted contexts are re-encoded, so nested raw observations do not imply nested frozen representations. These finite predictors may not use the available information well. The result cannot establish that relationships between joints are intrinsically unhelpful.

**The clinical motivation is not clinical validation.** GAVD folder labels are observational. The binary endpoint includes exercise and other circumstances that are not uniform diagnoses. Only five fitting-normal sources retain complete profiles across all edited versions. The teacher diagnostic demonstrates contextual target behavior, not physiology. No qualified human movement review, lesion information, forces, electromyography or medication measurements were obtained.

**The comparative claim is incomplete.** A learned advantage is not established against calibrated constituent scores, a common-coordinate random encoder or direct motion features. SeeKer and STG-NF are close published precedents but neither has a validated matched-support reproduction here. Their released layouts and missing-observation assumptions cannot be repaired by a silent zero-fill conversion. The [feasibility audit](../published_baseline_feasibility.md) explains the actual checks and remaining work.

**Uncertainty remains large and conditional.** The primary difference is 0.101 with an interval from -0.035 to 0.224. Intervals condition on the trained models, references, folds and observed sources. They do not include model retraining, new participants or analyst adaptation. Neither a null effect nor equivalence is established. Reference omission changes latent AUROC from 0.585 to 0.673 without retraining.

## Highest-value improvements before the deadline

| Priority | Concrete action | Completion criterion | Effect on claims |
| --- | --- | --- | --- |
| 1 | Validate a published pose-anomaly comparator on a declared common observation cohort. | Correct native execution, tested joint mapping and missingness handling, source-separated fitting and complete source predictions. | Makes the audit more relevant to existing methods. It cannot establish superiority unless the new evidence supports it. |
| 2 | Test a nested or residual predictor on processes with known dependencies, then normal development data. | The own-context solution remains available, and the learner recovers known added-context information. | Addresses whether the estimator can test the proposed question. Any new architecture remains a separately declared exploratory study. |
| 3 | Secure new normal recordings with participant identity and a frozen evaluation plan. | No fitting or method selection on these participants before the declared test. | Provides the missing independent normal population. This is valuable even if it cannot finish before November 20. |
| 4 | Complete independent movement review and source-identity/view checks. | Qualified reviewers, hidden labels where feasible, written rubric, agreement, uncertainty and exclusions recorded. | Tests visible movement interpretation. It does not establish a disease cause. |
| 5 | Improve reporting without changing outcomes. | Verified citations, anonymous PDF, consistent units, readable figures and preserved primary schedule. | Increases clarity and credibility. This work is already incorporated in the revised draft. |

Do not spend the remaining time trying many seeds, reversing an unfavorable score, promoting the longer schedule, or adding stochastic latent variables until one comparison looks positive. These actions would add selection risk without resolving the main uncertainty. The original stochastic-positional and latent-variable ideas remain future ablations rather than unsupported contributions of this paper.

## The submission decision

If no new validated baseline or independent data can be added, submit only with the present audit framing and explicit limits. The manuscript has a coherent workshop contribution, but the reviewers may reasonably consider its novelty and evidence too limited. A stronger publication program should establish estimator adequacy and independent participant evaluation before pursuing physiological explanations.

The [official ASTAD instructions](https://ismart.ece.mcgill.ca/astad4/) list a November 20, 2026 deadline, double-blind review, and seven content pages plus two reference pages. The public page does not state a timezone. Before submission, the authors still need to verify the system cutoff, author metadata, conflicts, originality, applicable AI-use disclosure and the exact anonymous files being uploaded. These are final author responsibilities, not missing experimental results. No external submission has been performed.
