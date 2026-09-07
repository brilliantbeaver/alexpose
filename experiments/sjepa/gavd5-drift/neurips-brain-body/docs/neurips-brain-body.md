---
title: "BrainBodyFM 2026: submission-readiness guide"
subtitle: "Current two-study manuscript and remaining decisions"
date: "Updated September 5, 2026"
---

## Current position

The revised short paper combines a source-held-out classification pilot with a completed laterality study. Its central contribution is an empirical evaluation of what learned gait features preserve, how reflection augmentation changes token consistency, and what an explicitly imposed sign constraint can guarantee. The laterality study supplies the stronger controlled evidence; the classification pilot provides supporting context.

The editable submission source is [bbfm2026_paper_draft.md](bbfm2026_paper_draft.md), with generated [LaTeX](bbfm2026_paper_draft.tex) and [PDF](bbfm2026_paper_draft.pdf). The [explanatory paper](explanatory_paper.pdf) provides the longer methods tutorial and evidence assessment. It is an author resource rather than the main submission file.

Submission and release remain blocked by the laterality project's recorded governance gate. Ethics determination, data-use review, and derived-pose release review are all unresolved in [governance/status.json](../../neurips-laterality/governance/status.json). The implemented rule requires each review to be resolved with an internal reference and date. This is a documented project condition, not a claim about an automatic rejection rule of the workshop. A weighted readiness score cannot override it.

## Two studies, with different evidence

| Property | Study A: classification pilot | Study B: laterality |
|---|---|---|
| Accepted cohort | 639 sequences / 97 sources | 625 sequences / 93 sources |
| Evaluation | Outer fold 0, seed 42 | Five outer folds, five seeds, two training variants |
| Source counts | 59 train, 18 validation, 20 test | 74–75 outer train, 18–19 outer test per fold |
| Main comparison | Learned features, raw pose statistics, missingness | Trained versus recorded initial features, reflection augmentation, odd/even readouts |
| Current artifacts | Saved notebook outputs; current checkpoint/evaluation bundles absent | Cohort, splits, 50 trained checkpoints, and 100,000 prediction rows across 16 lanes inspected |

The cohorts overlap and use different eligibility rules and models. Their counts should not be added, and Study B does not replicate Study A's classifier experiment. In both studies, source-video separation is weaker than person separation because one individual may appear in multiple uploads.

Study A's dated census begins with 666 annotated sequences from 103 uploads. Its September 4 metadata gate retains 657 sequences/100 sources, the media-span gate 655/98, and pose quality control 639/97. Source roles are frozen before the later exclusions. These figures describe the saved acquisition snapshot, not current video availability.

Study B starts with a frozen inventory of 642 pose archives and retains 625/93 after quality and target-computability checks. The local verification loaded its recorded lineage and predictions. Independent in-memory recomputation of the primary predictive and token-equivariance bootstrap tables reproduced the saved results within $10^{-16}$. That checks the stored evidence and report calculations; no independent retraining was performed.

## The results to emphasize

Study A's source-level macro-F1 is 0.440513 for raw pose statistics, 0.292424 for learned features, and 0.251111 for missingness. All three lanes miss all three stroke-annotated test sources. The comparison concerns 20 test sources in one fold and has no matched untrained encoder, so it does not isolate a general effect of pretraining.

Study B provides the following paired contrasts. Intervals are pointwise 95% source-bootstrap intervals, conditional on the fitted models and fixed split.

| Contrast | Estimate [95% interval] | Supported reading |
|---|---|---|
| Vanilla trained minus initial token error | +0.03055 [0.01576, 0.04763] | Training increases error under the specified token action |
| Reflection-augmented minus vanilla token error | −0.00843 [−0.01020, −0.00687] | Augmentation improves this geometric consistency measure |
| Native learned minus initial predictive $R^2$ | −0.01798 [−0.03851, 0.00248] | A predictive benefit is unestablished; equivalence is also unestablished |
| Constructed odd learned minus initial $R^2$ | −0.05874 [−0.09549, −0.01740] | Trained features perform worse through this constrained readout |
| Reflection-augmented minus vanilla native $R^2$ | +0.00408 [−0.00556, 0.01277] | Predictive improvement remains uncertain |

An odd feature construction and origin-preserving linear readout enforce output sign reversal for any encoder, including its initialization. The saved study verifies that property for every seed. Its value is as a controlled transformation rule; useful learned information still requires a predictive comparison. The novelty is the specific empirical separation of these outcomes, with matched controls, rather than new symmetry algebra or a discovery of anatomical laterality.

Study B's primary predictive score uses sequence rows weighted by inverse source size. For each seed, it pools out-of-fold predictions before calculating $R^2$, then averages the five seed scores. It does not average source targets first or score a prediction ensemble. Study A's temporal diagnostic does average targets and predictions within source before computing its separate $R^2$.

## Method descriptions that must remain accurate

Study A uses four-frame coordinate averages, a pooled-context MLP predictor, and SmoothL1 latent prediction plus variance and covariance penalties weighted 0.10 and 0.01. Its optional 0.10 condition cross-entropy term is disabled. Labels still determine the cumulative normal-first curriculum. Classifier selection uses validation sources, followed by a scaler/classifier refit on all 77 development sources. The encoder receives gradient updates from the 59 training sources only.

Study B flattens four-frame coordinate patches into 96-dimensional tokens, uses a Transformer predictor, and trains with centered latent cross-entropy plus a 0.05-weighted VICReg-style objective. Five folds, five seeds, and two variants give 50 trained fits. These implementations should not share one undifferentiated loss equation or configuration description.

Study A's temporal and drift notebooks interpolate short gaps, unlike its training preparation. Its classifier also averages probabilities at test time while validation uses source-mean features. Those limitations remain even after their descriptions are corrected. Normal-anchor cosine measures same-clip geometric change; functional forgetting and a successful consolidation repair remain untested. The forecasting branch lacks a separately trained checkpoint and requires a prefix-only context and baseline implementation before a causal forecasting claim is possible.

## Internal readiness assessment

Scores run from 1 to 5, with 3 representing an adequate limited contribution and 4 a strong one. The revised index assesses the integrated paper with the factual corrections applied and the evidence limitations retained. It is an author-side heuristic, not a workshop rubric or acceptance probability.

| Dimension | Weight | Revised score / 5 |
|---|---:|---:|
| Workshop fit | 15% | 4.0 |
| Contribution and novelty | 15% | 3.0 |
| Methods correctness | 20% | 4.0 |
| Empirical strength | 20% | 3.5 |
| Reproducibility and traceability | 15% | 3.5 |
| Claim discipline | 5% | 4.5 |
| Clarity and presentation | 5% | 4.0 |
| Data-use reporting and submission readiness | 5% | 2.0 |

Movement-representation evaluation gives the paper a direct workshop connection, while its methods remain established. The repeated laterality study and matched controls support the methods score; empirical strength remains limited by one observational dataset and no external person-level test. Reproducibility is scored for the whole paper, including the missing pilot bundles. The claim and clarity scores assume the revised two-study framing is retained. Unresolved governance keeps the final dimension low.

With percentage weights $w_j$, the index is $\sum_j w_j s_j/5=72/100$. The historical uncorrected pilot-only draft scored 48/100; a hypothetical corrected pilot without additional evidence scored approximately 57/100. The increase to 72 reflects the integrated existing laterality evidence and verification. It does not mean the pilot gained additional folds or that acceptance odds are 72%.

The scientific case is now more credible as a bounded methodological workshop contribution. Its novelty remains incremental, and no neural measurements, broad foundation-model transfer, clinical validation, or interactive control results are available. Independently of this assessment, the governance gate remains blocked.

## Remaining priorities

1. Resolve the three recorded governance reviews through the responsible authors or institution and record the required references and dates. Do not mark them resolved on the basis of manuscript edits.
2. Preserve the verified short-paper format in any final revision. The September 5 build has five main-text pages, two appendix pages, and one reference page in the shared NeurIPS 2026 double-blind workshop style. All eight pages were visually inspected; author identity is absent from the PDF metadata, and the vector figure uses labels of at least 9.3 pt at the official 5.5-inch text width. Recheck the PDF after any further edit.
3. Check that every abstract, table, and caption distinguishes the two cohorts and estimands. Preserve the token-error benefit of augmentation, the inconclusive native predictive contrast, and the negative constructed learned-minus-initial result.
4. Confirm that any linked reviewer-facing artifact extract complies with the recorded release determination. The local evidence contains source-linked records and is not automatically authorized for public redistribution.
5. Recover Study A's original bundles if available. If they remain unavailable, retain the explicit saved-output limitation and avoid upgrading the pilot's evidence status.
6. Verify the live submission portal and required author declarations before upload. The published deadline is September 5, 2026 AoE; the [call for papers](https://brainbodyfm-workshop.github.io/call-for-papers.html) and [OpenReview venue](https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/BrainBodyFM) are the relevant submission references.

New architecture changes, pilot preprocessing fixes, or forecasting experiments would create new evidence. They should not inherit the current scores without rerunning the affected pipeline. The completed laterality results can support the present paper while these broader experiments remain follow-up work.
