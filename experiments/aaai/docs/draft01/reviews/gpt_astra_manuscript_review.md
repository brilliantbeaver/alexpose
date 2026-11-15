# Final adversarial manuscript review

Reviewer: GPT-6 Astra. Reviewed November 15, 2026. This review covers `manuscript.md`, `context_study_results.md`, `context_study_sensitivity.md`, the earlier protocol critique, and selected completed-run metric artifacts. I did not change the manuscript, train models, or generate new experimental outcomes. Suggested explanations are hypotheses unless supported by the cited completed evidence. Line numbers below refer to the version read at the start of this review; the coordinating author is revising some passages concurrently.

**Judgment:** this is an honest, potentially useful exploratory audit. It is not evidence of a strong novel detector, and it does not establish that relational gait information lacks value. The manuscript now preserves the primary schedule, discloses historical source reuse, distinguishes contextual targets from local measurements, and correctly treats its intervals as conditional. I found no numerical contradiction in the principal AUROCs or paired intervals. The most important remaining reporting omission is a completed common-coordinate-target comparison that is more interpretable than the highlighted random-latent comparison.

## Prioritized actionable findings

### 1. High priority: show the random-coordinate control, not only the changed-target latent control

**Evidence:** manuscript lines 77 and 112 correctly acknowledge that a random teacher changes the latent prediction target. Nevertheless, the principal table at lines 97–105 includes random latent performance and omits random coordinate performance. The saved primary [ranking metrics, line 44](/Users/pmui/dev/alexpose/experiments/aaai/artifacts/context_study/gavd/3044566091261819/ranking_metrics.csv:44) give random-coordinate profile AUROC **0.700992**, versus trained-coordinate **0.685736** at line 16. The longer-run [ranking metrics, line 44](/Users/pmui/dev/alexpose/experiments/aaai/artifacts/context_study/gavd/eba2bd5a6e05efa1/ranking_metrics.csv:44) give random-coordinate **0.693364**, versus trained-coordinate **0.689550** at line 16.

These are already completed outcomes, not a proposed additional experiment. They hold the observable target fixed and therefore provide a cleaner control for the pretrained system than comparing two different latent teachers. They still do not isolate encoder effects perfectly: the heads are separately fitted and their optimization can differ.

**Correction:** add a clearly named random-encoder coordinate-profile row, show the point estimates without claiming a significant ordering, and state that both target families fail to establish a learned-system advantage. A supporting paired interval may be reported if explicitly labeled additional; do not select the primary target based on these outcomes. The coordinating author has accepted this correction and reports that it is being added.

### 2. High scientific priority: make the successful audit claim distinct from an unsuccessful test of the biological hypothesis

**Evidence:** manuscript line 118 and the sensitivity table show full-context coordinate MSE worse than own-context MSE in all four primary folds and three of four longer-run folds. The primary improvement interval is **[-0.035, 0.224]** at manuscript line 89. The timing probe has only five scorable fitting-normal sources and does not demonstrate its intended selective response at line 130. The teacher-only diagnostic at line 132 confirms a property expected from contextual targets, not a newly established biological explanation.

These facts support a useful warning about this implemented estimator. They do not constitute an adequately powered falsification of the underlying proposition that additional joints contain diagnostic information. The manuscript mostly respects that distinction, but “falsification controls” at line 23 and the broad biological setup can make the contribution seem more decisive than the evidence permits.

**Correction:** state one operational conclusion early and repeat it consistently: “An apparent point-estimate gain over raw latent error did not establish a relational or learned-representation advantage after matched calibration, random-feature, and learning-adequacy checks.” Describe the controls as diagnostic checks, not proof that the biological hypothesis failed. Preserve the strong explanation at lines 51–55 that nested raw observations do not imply nested frozen representations. If space is limited, replace some clinical background or the separate five-class comparison with a compact table linking each proposed interpretation to its observed diagnostic result.

No prose revision can turn this into a strong detector result. That would require better-supported prediction, independent normals, and stronger comparative evidence. The appropriate present contribution is the bounded audit itself.

### 3. Medium priority: report differential observation selection in the manuscript, not only total attrition

**Evidence:** manuscript lines 33 and 39 state overall filtering and the strict twelve-joint support rule. The detailed results table shows normal sources falling from **28 to 23** at the profile gate, exercise from **92 to 59**, abnormal from **103 to 92**, and inebriated from **6 to 2**. Exercise is also the strongest displayed category, with AUROC **0.815** and 59 retained sources at manuscript line 124.

The gate is independent of model errors, which avoids score-driven exclusion, but it selects different observable subsets across categories. A generic statement that visibility can matter does not convey the magnitude of that selection. The observation-only controls' poor performance cannot establish that visibility, view, activity, or source selection is harmless; the manuscript correctly avoids that conclusion at line 114.

**Correction:** include the existing folder-level source attrition table, or a compact main-text version with the complete table in supplementary material. State that the target population is recordings meeting this extraction/support rule. Keep normal versus every other folder as the authoritative primary endpoint. The focal-condition restriction is a secondary task, not a replacement intended to make the endpoint appear more clinical. The coordinating author reports that an attrition table is being added.

### 4. Medium priority: include the closest dependency and masked-gait precedents

**Evidence:** manuscript line 23 discusses S-JEPA, MAMP, FSGait, SeeKer and STG-NF, but omits two particularly close references already identified during planning. [GenGait](https://arxiv.org/abs/2604.01997) studies normal-only masked gait reconstruction and joint inconsistency. [Beyond Marginals](https://arxiv.org/abs/2509.15033) explicitly distinguishes marginal, temporal and inter-variable structure in anomaly detection. Both primary sources were checked in the preceding protocol review.

The manuscript makes restrained priority claims, so this is not a demonstrated false claim. Nevertheless, omitting these works weakens the novelty argument for a paper whose central question is whether relations add information beyond individual motion.

**Correction:** add two concise comparisons and their references. Identify the contribution as a source-level gait audit using explicitly restricted contexts, matched constituent calibration, and contextual-target checks. Do not claim novelty for masking gait, conditional joint prediction, cross-variable dependence, or the four-value algebraic decomposition. No new numerical reproduction is required to make the related-work statement accurate; the lack of a validated published baseline must remain explicit.

### 5. Medium priority: keep the population and uncertainty limits attached to every headline result

**Evidence:** manuscript lines 71 and 137 establish five references per fold, only five or six test normals per fold, and **all 23 normal sources reused from earlier development**. Lines 75 and 139 correctly exclude retraining, reference-estimation, and unknown participant dependence from the reported interval. Source IDs establish upload grouping, not independent people or sites. New abnormal sources do not repair the lack of fresh normal participants.

The present language is substantially correct. This is a preservation requirement for the abstract, figures, slides, and submission edits, rather than an allegation of an existing leakage bug. The overall sample count of 251 can otherwise conceal the limiting normal sample. Four cross-fitted models are not four independent replications, and the reference-removal extrema are not confidence bounds.

**Correction:** keep “exploratory,” the 23-normal count, the historical overlap, and “conditional interval” in the main result presentation. A forest plot should label differences and comparators explicitly, use one consistent positive direction, and distinguish the primary comparison from unadjusted supporting comparisons. Do not describe 0.673 versus 0.641, or random versus trained point estimates, as established superiority. The planned paired-contrast figure is an appropriate use of existing evidence.

### 6. Low priority, factual: correct the supporting report's blanket coordinate-loss sentence

**Evidence:** `context_study_sensitivity.md` line 32 says coordinate full-context MSE remains above own-context MSE without qualification. Its line 29 gives longer-run fold 2 full MSE **0.034**, below own MSE **0.036**. Manuscript line 118 already gives the accurate count.

**Correction:** say “Full-context coordinate MSE exceeds own-context MSE in all four primary folds and three of four longer-run folds.” The coordinating author reports that this sentence has now been corrected.

### 7. Low priority, factual: SeeKer uses keypoint-detector confidence

**Evidence:** manuscript line 23 says SeeKer incorporates “prediction confidence.” The [SeeKer primary abstract](https://arxiv.org/abs/2506.18368) describes weighting conditional log densities using the confidence of the underlying keypoint detector. That differs from calibrated predictive uncertainty of the sequence model.

**Correction:** replace this with “keypoint-detector confidence.” This avoids implying that the study has compared its uncertainty handling with a probabilistically calibrated predictor.

## Findings from the protocol review that are now adequately handled

The manuscript reports paired comparisons against constituent distances and does not attribute the raw-error contrast uniquely to loss subtraction. It keeps 512 updates primary and describes the normal-development motivation for 4,096 updates. It explains shared body normalization, restricted pre-encoder access, finite-predictor limitations, changed random latent targets, the limited fixed ridge baseline, and the unequal precision of four-patch scoring. It labels interventions as fitting-source diagnostics, reports their severe support loss, and keeps raw teacher decomposition separate from the nonlinear anomaly score. It also separates supervised recognition from normal-only detection and explicitly declines clinical, disease-causal, participant-independent and state-of-the-art claims.

These improvements materially strengthen credibility. They do not increase statistical power or methodological novelty by themselves.

## Scores

The scale is 1 for very weak, 3 for adequate but materially limited, and 5 for exceptional. These are editorial assessments, not acceptance probabilities.

| Criterion | Score / 5 | Reason |
| --- | ---: | --- |
| Novelty | 2 | The ingredients and interpretation hazards are established. Their controlled combination provides a useful case study, but neither a new mechanism nor a successful detector is demonstrated. |
| Rigor | 3 | Source roles, matched support, constituent controls, traceability and restrained claims are strong. Historical selection, small references, uncertain participant independence, and incomplete learning adequacy remain substantial. |
| Evidence | 3 | Completed real runs support a limited audit, including adverse results. They do not support a strong negative result about relational information or a positive detector claim. Evidence for either stronger claim would rate 1. |
| Clarity | 4 | The estimator and most boundaries are unusually explicit. A sharper operational takeaway and the common-coordinate random control would improve the narrative. |
| Workshop fit | 4 | A transparent anomaly-detection audit and reproducible failure analysis are relevant. Its value depends on presenting a concrete methodological lesson rather than offering an underpowered leaderboard or a neurological world-model claim. |

The defensible final positioning is an exploratory empirical audit of when predictive-context scores are interpretable. Keep the completed adverse evidence visible. Do not promote the longer schedule, coordinate score, favorable category, or a new primary contrast after seeing the results.
