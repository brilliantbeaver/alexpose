# GenAI4Health: submission strategy and critical assessment

Reviewed September 6, 2026. This is an author-facing document, not part of the anonymous submission. The recommended manuscript is **Before Health Agents Interpret Movement: Lessons from a Gait Representation Study**.

## Recommended route

Use the position-paper track, with trustworthy evaluation as the primary topic and future ambient health systems as the secondary motivation. The paper presents a specific, evidence-grounded position: movement results passed to a health assistant should retain their measured quantity, recording weights, model reference, and tested scope. The empirical case shows why these details can change interpretation.

This is a stronger fit than claiming a new clinical world model or a working health agent. Neither has been evaluated. The [current workshop call](https://genai4health.github.io/2026-NeurIPS/) accepts position papers up to five main-text pages and research papers up to nine. Its deadline is **September 9, 2026, 11:59 PM AoE**, an extension from the earlier date. It lists no separate extended-abstract track. The two-page companion is therefore a shorter alternative or synopsis, not a second independent contribution to submit alongside the paper.

The revised paper makes no laterality contribution. Its question concerns interpretation of movement evidence, using recording-level weighting and an exploratory classifier comparison. The existing BrainBodyFM paper remains unchanged.

## What the current evidence supports

The requested alternative on neurologically guided masking is assessed separately in [the masking candidate review](masking_candidate_assessment.md). It is a plausible future research paper, but the notebooks currently supply sampler checks rather than a trained mask-policy comparison. Established body-part and motion-aware masking literature also narrows its novelty. The current submission has therefore not been silently reframed around an untested masking benefit.

| Finding | Numerical evidence | What can be concluded |
|---|---|---|
| Averaging changes which recordings a summary represents | The same 64 normal-validation clips give mean cosine 0.89 with equal clip weights and 0.70 with equal video weights | Changing only the weights changes the summary; this does not measure deterioration or loss of predictive function |
| One upload explains much of that difference | One video contributes 60 clips; four other videos contribute one each. Its weight is about 94% or 20%, respectively | A clip average is dominated by one recording. Neither weighting rule establishes equal patient weighting |
| Simple inputs are a necessary comparison | Pose summaries correctly classify 10/20 test videos; learned features and availability-only inputs each classify 6/20. Balanced accuracy is 0.44, 0.26, and 0.25 | This particular learned-feature pipeline has not demonstrated an advantage over the simpler input; a general pretraining effect is not identified |
| An annotation score can hide category failures | Every classifier misclassifies all three stroke-annotated test videos | Aggregate performance cannot support diagnosis or reliable category-specific claims in this sample |

The weighting example is the cleanest comparison because recordings and recorded similarities stay fixed. It is an exact reaggregation of a finite set of records, not an intervention on training. The classifier result is supporting evidence, with one split, one initialization, and a mismatch between validation and test aggregation. No significance claim, broad architecture ranking, or clinical threshold is attached to either result.

The numerical supplement retains full-precision values for calculation. The manuscript uses two decimals for similarity and classification scores, whole percentages for illustrative weights, and exact counts. A cosine is not a retained-information percentage. No interpretation relies on rounding a value across a decision threshold.

## How the experiment works

1. Select a five-category subset from GAVD's curated gait annotations, then apply recorded acquisition and pose-quality checks. The usable cohort contains 639 clips from 97 videos. These are subset counts, not the size of GAVD.
2. Keep every clip from a video in the same role. The recorded evaluation uses 59 training, 18 validation, and 20 test videos. Person separation across uploads remains unverified.
3. Prepare video-estimated pose for the encoder. Four-frame coordinate means at 33 landmarks enter a compact two-layer Transformer. The estimates are not calibrated physical measurements.
4. Train by predicting hidden features from a slowly updated target encoder, with variance and covariance regularization. Category annotations determine a normal-first, cumulative exposure order but do not appear as prediction labels in this loss.
5. Compare fixed normal clips across two training stages. Recalculate the feature-similarity average using clip weights and then video weights.
6. Fit separate classifiers to learned features, pose summaries, and landmark availability. Use held-out videos for the reported annotation predictions, while acknowledging the differing aggregation used in selection and testing.
7. State what these results do and do not establish, then propose an explicit way to test their interpretation by a future health assistant. That assistant evaluation has not been performed.

Full preparation and readout details are in the paper appendix. Low-level run identifiers and notebook numbering are confined to internal verification records.

## What is distinctive, and what is established practice?

The contribution is a concrete application with a reproducible numerical example, not a new foundation-model architecture or general theory of clinical trust.

| Element | Assessment of novelty | Workshop relevance |
|---|---|---|
| Same-records weighting demonstration | Specific empirical observation from this gait representation study; weighted averages themselves are established | Shows how repeated observations can dominate evidence that a health system might summarize |
| Simpler-feature and availability comparisons | Useful negative case, but incremental and underpowered for broad conclusions | Encourages checking whether a proposed movement input adds predictive value |
| Evidence record accompanying a movement summary | A worked specialization of existing reporting practice, not a replacement for Model Cards or clinical validation | Makes model reference, recording unit, and untested uses available where an assistant or clinician interprets a result |
| Proposed interpretation experiment | A testable future study, not a demonstrated safety method | Directly connects the position to human–AI collaboration and unsupported clinical interpretations |

[Model Cards](https://arxiv.org/abs/1810.03993) already address intended uses and evaluation limitations. The [V3 digital-measurement framework](https://www.nature.com/articles/s41746-020-0260-4) distinguishes sensor, analytical, and clinical evidence. [DECIDE-AI](https://www.nature.com/articles/s41591-022-01772-9) provides clinical-evaluation context, including human factors. These are acknowledged in the manuscript so that ordinary validation principles are not presented as inventions.

A positive comparison from the literature is also retained. [GaitForeMer](https://arxiv.org/abs/2207.00106) reports improved gait-severity estimation after pretraining that combines motion forecasting with activity classification. Our small, different experiment does not refute that result or the broader promise of self-supervised movement learning.

## What was revised

The main paper and companion now share one title, position, and numerical interpretation. The rewrite makes the weighting observation primary and the annotation classifier secondary. A new vector figure shows the weights separately from the mean cosines, avoiding a truncated similarity axis or a visual suggestion of improved retention. A simple table reports correct-video counts and balanced accuracy.

A second table gives a worked evidence record for the actual weighting analysis. It is explicitly proposed, not a screenshot of a deployed system or a validated safety intervention. The paper describes how to evaluate it against a bare-score condition with the same assistant and cases, including blinded review of unsupported mobility claims.

The previous manuscript's many local processing gates, artifact hashes, notebook references, repeated caveats, and supplementary exploratory plots have been removed from submission-facing prose. Necessary limitations remain near the relevant results and in the methods appendix. The abstract identifies the normal-validation subset and distinguishes coordinate similarity from useful prediction and health change.

Method corrections separate visibility-based clip eligibility from finite-coordinate encoder validity; specify that test aggregation averages category probabilities; remove an unrecorded masking fraction; and define cosine as directional agreement. Current verification is described as reconstruction from retained records, without claiming renewed verification of absent checkpoints.

## Notebook-by-notebook selection

All 14 notebooks were reviewed through code and saved outputs; no notebook was executed wholesale. The source files were preserved.

| Notebook | Evidence and decision |
|---|---|
| 00: first principles | Teaching model and synthetic examples. Useful for concepts; its architecture is not substituted for the trained model |
| 01: manifest and video acquisition | Supports the recorded cohort and fixed video-role assignments; current availability is not inferred |
| 02: pose extraction and inspection | Supports pose eligibility and measurement limitations; no new extraction was performed |
| 03: landmark masking | Explains valid target selection. Synthetic mask checks are not evidence of clinical validity or the executed run's exact masking setting |
| 04: pretraining | Determines the compact learner, loss, staged exposure, and recorded training history |
| 05: latent inspection | Descriptive clusters and retrieval examples do not establish generalization or clinical usefulness; no headline figure reused |
| 06: annotation classifiers | Retained predictions support the secondary result and all three input comparisons |
| 07: temporal readout | Heuristic pose targets, changing readouts, and no matched untrained or pose baseline. Excluded from the submission's evidence |
| 08: normal-reference comparisons | Supplies the recorded similarities and source counts for the main reaggregation; no forgetting or repair claim |
| 09: predictive surprise | No qualifying forecast result. Prospective code also permits future context; the paper retains only a general requirement for genuinely past-only forecasting |
| 05a: signed laterality | Excluded because of subject overlap and archived experimental status |
| 05b: reflection and futures | Simulated illustrations and incomplete real-data paths; excluded |
| 05c: equivariant readout | Laterality-focused archived evidence; excluded |
| 05d: equivariant encoder | Laterality-focused archived evidence; excluded |

The separate laterality experiment's five folds, five seeds, and matched initializations are not imported. The classification result already appears as supporting material in the BrainBodyFM paper, so the two manuscripts are not wholly disjoint in data or findings. Authors should disclose relevant overlap through the venues' prescribed anonymous process and confirm concurrent-submission policies. A different title or position does not make reused observations independent.

## Current verification, including its limits

The portable verifier recalculates all three classifier scores and both weighting averages. A separate read-only check matches all 60 prediction rows to the current classifier notebook, reproduces cohort totals from available annotations and ledgers, and confirms recording-role consistency. Both weighting CSV copies agree.

Original fold checkpoints, cached embeddings, and the full split registry are absent from this checkout. Historical records describe an earlier checkpoint-backed review; they are preserved but are not labeled as checks rerun now. The current report is [evidence/current_verification.json](evidence/current_verification.json). The earlier [verification manifest](evidence/verification_manifest.json) is historical. Matching records does not reproduce training, prove historical test secrecy, or establish clinical validity.

The original numerical verifier failed because its recorded file checksums used Windows line endings while this checkout uses Unix line endings. The values were identical. Verification now explicitly normalizes line endings in memory, preserves the historical checksums, and checks the unchanged numerical content. This fixes portability without changing any result.

## Honest assessment of submission strength

| Criterion | Current assessment | Main residual risk |
|---|---|---|
| Position clarity | Stronger: a concrete recommendation, a worked example, and a proposed test | The recommendation may still be judged familiar |
| Empirical grounding | Good for a descriptive weighting example; limited for classifier claims | Five videos in the weighting subset; one classifier split and initialization |
| Technical correctness | Retained-output arithmetic and relevant implementation details checked | Full encoder reproduction remains unavailable |
| Health relevance | Plausible for future movement interpretation; clinically grounded prior work | No independently measured mobility outcome or demonstrated care workflow |
| GenAI relevance | Explicit perception-to-assistant motivation | No generative model or agent experiment; this is the main track-fit objection |
| Presentation | Focused findings, readable tables, one new vector graphic | Authors should still review the rendered submission as a reader |
| Submission and data-use readiness | Draft artifacts prepared locally | Ethics/data-use determination, authorship, overlap review, and release decisions require authors |

This is a defensible position-paper submission, not a demonstrated high-performing health AI system. No acceptance probability is justified. Strong prose can clarify the contribution but cannot replace missing empirical breadth or a direct GenAI experiment.

## Improvements to prioritize before September 9

**Required before upload.** Confirm the actual institutional ethics and data-use determination, author list, concurrent-submission obligations, and supplement release decision. The user has confirmed GAVD's MIT licensing for research use; the manuscript states that clearly while distinguishing the source videos. Review the final PDF, choose one manuscript and track, and verify portal declarations. No external submission has been made.

**Highest-value empirical extension, if feasible without rushing.** Test the evidence-record proposal with a fixed assistant and a small prespecified set of cases whose interpretations can be independently judged. Vary recording composition and encoder reference without changing the claimed patient state, and include genuinely measured movement-change cases if such data exist. Compare bare scores with the record under matched prompts and blinded assessment. Do not fabricate clinical cases or call an unvalidated synthetic scenario a patient outcome. This would address the strongest current objection: the GenAI connection is proposed, not evaluated.

**Next representation experiment.** Recover the original run artifacts, then specify a fresh comparison that uses the same aggregation during classifier selection and testing, includes an untrained encoder, and repeats across video groups and seeds. Standardize missing-data handling and serialize all runtime settings. Any reanalysis on the already inspected test videos is exploratory; it cannot be relabeled as a new untouched confirmation.

**Longer-term clinical work.** Use identity-aware external data and independently measured mobility endpoints. Evaluate task performance across model updates before calling coordinate similarity retention. Forecasting and action-conditioned planning need separate designs and evidence. These are research priorities, not prerequisites that have been silently completed for this position paper.

## Review record

- [Core notebook review](review/core_notebook_audit.md)
- [Advanced notebook review](review/advanced_notebook_audit.md)
- [Workshop, prior literature, and novelty assessment](review/draft_and_literature_audit.md)
- [Adversarial manuscript review](review/adversarial_draft_review.md)
- [Final core claim crosscheck](review/core_draft_crosscheck.md)
- [Advanced evidence boundaries](review/advanced_draft_crosscheck.md)
- [Numerical supplement](numerical_supplement/README.md)
- [Final verification and unresolved author decisions](review/final_verification.md)
