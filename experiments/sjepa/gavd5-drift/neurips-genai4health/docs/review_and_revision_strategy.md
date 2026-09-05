# GenAI4Health: research assessment and revision strategy

Reviewed September 5, 2026. This is an internal author-facing analysis; the anonymous manuscript is `genai4health_paper_draft.tex`. The source notebooks and BrainBodyFM drafts were preserved. No model training or external submission was performed.

Final consistency note: five notebook files received concurrent updates during the review. Comparing their cell source text found no code/prose changes, and all manuscript evidence-input hashes remained identical. The initial snapshot and the update check are retained in the audit records; this work did not overwrite those updates.

## Recommended paper and why

The strongest defensible submission is a **position paper with a concrete empirical audit**: *Before Gait Models Inform Care: Evidence Boundaries for Predictive Health AI*. Its position is that evidence for a movement representation must remain attached to its population, model version, measured task, and unresolved failures when that representation is used downstream. Source transfer, functional retention, forecasting, and clinical utility require different evaluations.

The current work does not support a strong claim of a new clinical world model, an effective agent, superior JEPA performance, or successful consolidation. It does support a specific and useful argument about evaluation. The best results are two observations that can be independently reconstructed from existing artifacts:

| Observation | Verified evidence | Defensible interpretation |
|---|---|---|
| A simpler feature baseline has the higher observed readout score | Raw-kinematic macro-F1 0.440513, skeleton-JEPA 0.292424, missingness 0.251111 on the same 20 test sources | This run provides no demonstrated advantage for the learned representation; it does not rank architecture families |
| Aggregation changes a representation-similarity summary | Identical 64 normal-validation embeddings give clip-weighted cosine 0.889061 and source-weighted cosine 0.701058 | A clip average mostly describes the dominant upload; the result changes the estimand, not the model |
| The dominant upload explains the weighting contrast | 60 of 64 clips; source mean cosine 0.904994; other four sources supply one clip each | 93.75% clip weight versus 20% source weight is an explicit, testable analytic choice |
| The evaluated population is smaller than the annotation inventory | 666/103 raw → 657/100 metadata-public → 655/98 decoded → 639/97 pose-eligible | Report distinct validity gates and the actual evaluated sample |

These observations are more convincing than a broad claim built from appealing historical plots. The weighting analysis is especially valuable: it changes only the averaging rule, so the inference does not depend on comparing incompatible runs. The raw-minus-latent macro-F1 difference is 0.148089 **before rounding**. Neither observation is a significance test, a patient-level estimate, or a measured clinical effect.

## Workshop fit and submission choice

The official [GenAI4Health 2026 call](https://genai4health.github.io/2026-NeurIPS/) emphasizes trustworthy evaluation and ambient/embodied health alongside frontier models. Our best fit is evaluation and trust, motivated by a possible ambient perception component. The mixed ML/health audience needs clear separation of algorithmic proxies and clinical endpoints.

The current call permits research papers up to nine pages and position papers up to five; it lists no separate extended-abstract track. The companion abstract is therefore an alternative synopsis, not a second submission. The deadline is September 5, 2026, 23:59 AoE (September 6, 04:59 Los Angeles). Use the specified anonymous style and confirm the live requirements before uploading. The supplied main draft fits the position limit.

A research-track case study would be possible if its central claim remains the audit itself. I recommend the position track because the empirical coverage is one fold/seed, the model and raw feature comparison have unresolved procedural weaknesses, and no new architecture or clinical endpoint is established. A demonstration submission would be a poor fit: the notebook collection is not evidence of a functioning care-facing agent.

The central acceptance risk is modest novelty over standard evaluation practice. The response is specificity: foreground what the audit exposed and why those boundaries matter when a downstream system turns movement features into explanations. Do not claim that source grouping, raw baselines, or provenance hashes are individually novel. No acceptance probability can be inferred from this review.

## What the notebook collection actually establishes

All 14 notebooks were inspected through their code, markdown, and retained outputs. Embedded media and historical artifacts were distinguished from current real-mode results. Detailed audits are linked below; this table is the editorial selection map.

| Notebook | Scientific purpose and current status | Use in this submission |
|---|---|---|
| 00, first principles | Tutorial and synthetic execution; architecture and objective differ from the current trained model | Conceptual distinction between masked prediction and forecasting; no performance result or reused architecture diagram |
| 01, manifest/video | Dated availability, decoding gates, fixed source-role registry | Cohort and provenance methods; compact appendix counts |
| 02, skeleton extraction | Audits 655 available legacy caches; 639 sequences pass the selected-landmark coverage gate | QC and geometry limitation; no claim that geometry-complete caches were regenerated |
| 03, target masking | Tests the valid 12-joint whitelist and nominal mask fraction | Exact masking description; not clinical validation of the whitelist |
| 04, training | Current fold-0/seed-42 cumulative training and hash-bound checkpoints | Canonical model/loss description and training provenance |
| 05, latent inspection | Descriptive geometry, training-only retrieval gallery, role-specific diagnostics | Exclude from headline results; negative silhouette is not a diagnosis of collapse or information absence |
| 06, classifiers | Three feature lanes on the same 20 held-out source groups | Main negative comparison; disclose readout refit and inconsistent selection/test aggregation |
| 07, temporal readout | Ridge probes of pose-derived clip statistics; one fold, different preprocessing, early test materialization | Appendix only; retain all nine values, including negative phase-lag R² |
| 08, normal anchors | Current coordinate drift and a single available final candidate; source-equal summaries | Main weighting reanalysis and a supplementary trajectory; no repair/forgetting/health claim |
| 09, predictive surprise | No current future-trained checkpoint; exploratory code also permits future information | Static code-audit finding and future evidence requirement; no surprise score or forecast result |
| 05a, signed laterality | Archived transductive experiment; fold-local gate blocked | Exclude empirical scores |
| 05b, reflection/futures exploration | Includes simulations and illustrative values; some labeled R² values do not match the generated statistic | Exclude numerical evidence entirely |
| 05c, equivariant readout | Archived construction and transductive results | Exclude; oddness constraints do not establish useful signed prediction |
| 05d, equivariant encoder | Archived encoder/augmentation experiments; current gate blocked | Exclude; do not mix with the protocol-v2 checkpoint |

Two distinctions are essential. First, a notebook that executes successfully can still produce only a status message or a simulated example. Second, a current figure exporter can display historical data. File modification time, a polished plot, and a zero-error execution are not sufficient provenance.

## Required corrections to the BrainBodyFM drafts

The new drafts implement the corrections below. They have not been written back into the original source package.

| Existing claim or presentation | Why it is inaccurate or risky | Correction in the GenAI4Health draft |
|---|---|---|
| Primary loss `JEPA + 0.05 VICReg`; group term `0.25` | Describes an older model | Smooth-L1 JEPA + 0.10 variance + 0.01 covariance; no separate two-view invariance term |
| Group loss means within-label compactness/separation | Current optional head uses classification cross-entropy | State it is disabled; do not report a current ablation that was not run |
| Fully label-free learning pipeline | Condition annotations select the normal-first cumulative stages and source stratification | Loss has no condition-label term; scheduling is annotation-informed |
| First-principles tutorial describes the executed model | Tutorial flattens coordinate patches and uses other masking/prediction mechanics | Specify the checkpoint's four-frame means, 64-D tokens, 2-layer transformer and pooled MLP predictor |
| Shared short-gap interpolation in all notebooks | Training/readout zero-fill nonfinite values; normal-anchor/temporal diagnostics interpolate short gaps | State the distinct preprocessing paths and why the same-vector weighting comparison remains valid |
| Readout fits only training sources | Final scaler/classifier refit uses train plus validation | Explain tuning followed by 77-source refit; encoder remains frozen |
| One identical source aggregation throughout | Selection classifies mean source features; test averages per-clip probabilities | Disclose the mismatch as an unresolved limitation of the saved result |
| Globally unopened test set until final evaluation | Training excludes test, but later notebooks inspect it; notebook 07 loads it before ridge selection | Report inspected training access and role separation; do not certify historical test secrecy |
| Independent or subject-held-out videos | Upload IDs are not reliable person identities | Source-held-out evaluation; cross-upload people and correlated sources unresolved |
| Cosine measures retained normal function | Coordinate changes, scale/basis effects and collapse can make that inference wrong | Cross-checkpoint embedding cosine; functional retention needs an actual fixed task |
| Test cosine 0.850 shows repair from 0.701 validation | Different sources; only one candidate exists | Distinct descriptive samples, no intervention or repair gain |
| Predictive surprise demonstrates a world model | No qualifying current checkpoint; visible future joints and full-context baseline compromise the proposed test | No current forecasting result; require prefix-causal context, preprocessing, and baselines |
| Revised evaluation caused a large performance reduction | Old and current runs differ in data, objective, model, and protocol | No causal “leakage penalty”; compare only the contemporaneous three lanes |
| Full GAVD has 666 clips and five categories | These describe the local subset | Explicitly label the subset and preserve the verified original dataset citation |
| Current ready-to-submit guide PDF | `neurips-brain-body.pdf` is a stale guide with old results | Use current source documents and the newly verified package |

The canonical old manuscript PDF matches its current `.tex`; the separate readiness-guide PDF does not match its September 5 Markdown counterpart. The latter contains the older 626/93 cohort and transductive values, including macro-F1 0.899. Those are not alternate estimates of the present experiment.

## Argument and result selection

The paper begins with a narrowly specified proposed use: movement summaries could become inputs to an ambient health assistant. This motivates an evidence problem without pretending such a system was implemented. The introduction then states the four independent claim types and acknowledges prior work.

The methods identify the actual sample, evaluation units, executed model, and annotation use before presenting any scores. The results give both contemporaneous controls and the direct weighting comparison. The proposed evaluation requirements follow from these findings. Counterarguments acknowledge that a larger model could fare better, different weights target different populations, and missingness can reflect actual movement difficulty.

Figure 1 carries the empirical argument. Its first panel shows all three readouts and both macro-F1 and balanced accuracy. Its second compares weights on identical normal-validation vectors. It does not display error bars that were never estimated. The cohort table, role composition, model details, all temporal metrics, and coordinate trajectory appear in the appendix. An aliased numerical supplement reproduces the main scores using only standard Python.

Do not add a speculative agent architecture diagram, diagnostic pipeline, generated patient example, or illustrated clinical benefit. They would make the implemented scope less clear. Likewise, an attractive latent visualization adds little to the central position unless its specific hypothesis, comparison, and caveat are stated.

## What remains experimentally unresolved

The following changes are needed for stronger empirical claims. They are not completed experiments and do not need to be pretended complete for this bounded position paper.

1. **Make a new experiment version before repairing the evaluation.** Standardize validation and test source aggregation; centralize preprocessing; settle invalid-token treatment; fully serialize runtime configuration and the optimizer state. Preserve the existing run as historical evidence. Do not quietly regenerate a better score and call it the original test result.
2. **Strengthen sensing provenance.** Re-extract with resolution/crop-aware geometry where feasible, bind actual video and annotation content, and report attrition by source. Assess missingness and acquisition strata. Current cache contracts do not independently establish every scientific input byte.
3. **Finish comparative evaluation.** Include an untrained encoder, raw pose summaries, missingness, and matched learning controls. Retrain within each fold and multiple predetermined seeds. Do not describe a five-fold registry as five-fold results.
4. **Separate curriculum effects.** Compare continued-normal training with matched updates, cumulative/joint training, and alternative condition orders. Rewinding selected model weights while retaining later optimizer moments also needs an explicit, consistent policy.
5. **Use uncertainty appropriate to the claim.** Paired source resampling can characterize the fixed predictions, but cannot recover seed or split variability. Report both when available. Do not use sequence bootstrapping as if 131 clips were 131 independent people.
6. **Test retained function.** Add fixed movement tasks and alignment-aware analyses. A stable latent and a drifting latent can each be compatible with either useful or poor behavior. Avoid clinical “normality” thresholds derived from a small annotation subset.
7. **Redesign forecasting before training it.** Hide every future input to context and persistence baselines, audit preprocessing for future dependence, preserve physical time, and define prefix/horizon splits. Train a dedicated temporal objective and compare simple causal predictors. Action-conditioned planning would be a further study.
8. **Study clinical claims in an appropriate cohort.** External, adjudicated, identity-aware data and a specified workflow are necessary for person-level or clinical assertions. A public-video convenience sample cannot provide those by renaming a metric.

An inexpensive matched aggregation reanalysis would be informative, but it would be a post hoc follow-up, not a new untouched confirmatory test. The current task prioritized an accurate manuscript and an audit over additional model development.

## Adversarial review and disposition

Three parallel reviews covered core training/data/readouts, laterality/retention/forecasting, and literature/venue/narrative. A second pass reviewed both new drafts against the evidence. Reviewers independently confirmed the headline scores, counts, objective, parameter dimensions, and weighting equations.

The second pass led to explicit normal-anchor interpolation and 12-joint pooling descriptions, a trained normal-only reference checkpoint, clearer local-registry language, unrounded-gap disclosure, the correct V-JEPA 2 author spelling, and a portable numerical supplement. Figure sizing and labels were also revised during PDF inspection. No new experimental success was invented to resolve a criticism.

Remaining scientific weaknesses are visible in the manuscript: one fold/seed, inconsistent readout aggregation, legacy geometry, incomplete runtime metadata, no person key, no calibrated clinical outcome, no functional retention control, and no current forecasting result. The final claim is intentionally narrower than these unresolved issues.

## Author decisions before upload

Review the final PDF and its cited scope. Establish the actual project-specific institutional ethics/data-use determination; none was supplied in the inspected record, and the manuscript explicitly says so. Do not substitute public availability for that determination. Confirm authorship and any concurrent-submission obligations associated with the BrainBodyFM package. Choose one submission track and one manuscript; the extended abstract is a companion, not a second empirical work.

The drafting, numerical reconstruction, and adversarial review are complete. Submission, institutional determinations, and any new experiments are separate actions that have not been performed.

## Supporting audits

- [Core notebooks 00–07](review/core_notebook_audit.md)
- [Advanced notebooks 08–09 and 05a–05d](review/advanced_notebook_audit.md)
- [Original drafts, workshop, and verified literature](review/draft_and_literature_audit.md)
- [Adversarial review of both new drafts](review/adversarial_draft_review.md)
- [Core manuscript crosscheck](review/core_draft_crosscheck.md)
- [Advanced manuscript crosscheck](review/advanced_draft_crosscheck.md)
- [Recomputed values and input hashes](evidence/verification_manifest.json)
- [Portable reviewer numerical verification](numerical_supplement/README.md)
