# BrainBodyFM 2026: research and submission-readiness guide

**Status date:** September 3, 2026

**Target:** NeurIPS 2026 Workshop on Foundation Models for the Brain and Body

**Deadline:** September 5, 2026 AoE

**Current recommendation:** submit only as a clearly labeled work in progress, or postpone the strong forgetting and repair claims.

## 1. The short version

The project now has a larger, cleaner primary run that uses only the GAVD path. It does **not** use the optional project-labeled normal-video cohort.

The verified result is simple:

- after normal-only training, the same model learns four condition groups in sequence;
- the mean cosine between each normal sequence's current representation and its own Stage-0 representation falls from 0.700 after the first later stage to 0.297 after the fourth; and
- reloading the saved checkpoints reproduces that curve within `4.51e-7`.

This shows a substantial **raw latent-coordinate change in this run**. It does not yet prove that the model forgot how to represent normal gait. A representation can rotate its coordinate system while keeping the same information.

The old paper claimed more: a drift mechanism, an AnchorGuard repair result, and a forecasting result. Those claims came from old or mixed-lineage artifacts and have been removed from the current paper.

## 2. Why the workshop is a good fit

The [official call for papers](https://brainbodyfm-workshop.github.io/call-for-papers) explicitly includes pose extracted from video as a behavioral signal. It also lists self-supervised learning, continual learning, movement, interpretability, evaluation protocols, and reproducibility.

This project fits those topics well. Its strongest workshop angle is not clinical classification. It is **how to audit a continually updated movement representation without confusing coordinate drift, label readability, and generalization**. The paper must call the model a project-specific, S-JEPA-inspired variant because its target sampler and auxiliary losses differ from the published method.

The workshop especially encourages generalization and scaling. That makes the missing held-out-video and multi-seed experiments important review risks.

## 3. Current experiment, in plain language

### 3.1 Data

The local GAVD inventory starts with 666 sequences from 103 source videos. Nine unavailable sources are removed, leaving 642 sequences from 94 videos. Pose-quality filtering removes 16 more sequences.

The model uses 626 sequences from 93 source videos:

|Group|Sequences|Videos|
|---|---:|---:|
|Normal|270|29|
|Parkinson's|41|9|
|Stroke|74|18|
|Myopathic|183|28|
|Cerebral palsy|58|9|

These names are GAVD folder annotations. They are not new diagnoses made by the project.

The 642 availability-filtered pose caches carry mixed extraction labels: 546 `gavd5`, 95 `gavd3`, and one `gavd4`. The modeled 626 rows contain 530, 95, and 1, respectively. Their pose-model hash agrees, but version history could still correlate with sources or labels and should be controlled in a generalization experiment.

### 3.2 Model

MediaPipe converts each video segment into 33 moving landmarks. Each sequence is centered, scaled, and resized to 64 frames. The S-JEPA-inspired model hides valid tokens from a 12-landmark gait whitelist and predicts their teacher-encoder features.

The model has three training signals:

1. JEPA latent prediction;
2. VICReg anti-collapse regularization; and
3. a label-aware group term after Stage 0.

The label-aware term matters to interpretation. The full curriculum is not purely self-supervised.

### 3.3 Curriculum

One model trains for 300 normal-only epochs, then for 75 epochs after each new group is added. Balanced replay keeps earlier groups in every later stage. The complete run has 40,800 optimizer updates and one seed, 42.

## 4. What the current results show

### 4.1 Verified anchor curve

|After adding|Raw normal-anchor cosine|
|---|---:|
|Parkinson's|0.7002|
|Stroke|0.5021|
|Myopathic|0.3962|
|Cerebral palsy|0.2966|

The checkpoint reload differs from the saved curve by at most `4.51e-7`. This rules out a simple logging mistake.

This value averages 270 matched, per-sequence cosines. It is not a cosine between two normal-cohort centroids.

It is also sequence-weighted: the top two normal videos supply 105 of 270 rows, and the top three supply 137. A strong paper needs per-video values, an equal-video-weighted curve, and source-cluster uncertainty.

### 4.2 In-corpus structure

The final 384-D features have cosine silhouette 0.362. The minimum distance between condition centroids is 0.086, and the mean within-condition distance is 0.078.

This is label-informed in-corpus geometry because the group loss used those condition labels. It must not be compared numerically with the retired augmented run as if the cohort and training contract were fixed.

### 4.3 Descriptive classifiers

The all-row sequence split reaches 0.920 accuracy and 0.899 macro-F1. The exact historical 47/21 split reaches 0.857 accuracy and 0.861 macro-F1.

These high values are not held-out performance:

- every classifier test row was used to train the encoder;
- 64 source videos occur on both sides of the all-row classifier split; and
- all 9 exact-split test videos occur in classifier training.

A missingness-only classifier reaches 0.441 accuracy and 0.355 macro-F1, showing that pose-estimator behavior contains some label signal.

### 4.4 Timing and laterality probes

The temporal readout study is current and fingerprint-matched. Its pre-specified temporal-moment lane does not pass the two-part improvement rule. Cadence and stride-time are weakly decoded, which is compatible with resizing every clip to a fixed length without giving the model native duration, but does not prove the cause.

The signed-laterality study finds a small learned advantage over an untrained encoder, but source-level sign consistency and mirror-equivariance gates fail. Its raw-feature score near 1 is only a construction check because the target is built from those raw signed features.

That laterality probe uses all 642 availability-filtered rows, including the 16 low-coverage rows excluded from the main training cohort. It is useful as a secondary probe but is not directly cohort-matched to the 626-row primary analysis.

Both executed ridge-probe notebooks contain many numerical ill-conditioning warnings and lack repeated-split intervals. Their small differences are inconclusive until rerun with a stable solver and sensitivity analysis.

## 5. What the current results do not show

### Raw anchor drift is not automatically forgetting

Cosine compares coordinates directly. If every feature rotates together, raw cosine can fall while downstream information remains intact. Use "raw coordinate drift" until an aligned comparison and a functional-retention test agree.

### Source-video grouping is not person grouping

The dataset does not provide a reliable person identifier for this analysis. The same person may appear in different videos. Even a video-disjoint result would not automatically be person-disjoint.

### The model is not clinically validated

The data are public videos with dataset folder labels, variable cameras, and markerless pose estimates. There is no external clinical cohort, independent diagnosis review, severity label, or prospective evaluation.

## 6. Artifact problems found by adversarial review

The audit found four result families that must stay out of the current paper.

|Artifact family|Problem|Action|
|---|---|---|
|Margin ablation|Notebook 08 loads `sjepa_normal_augmented.pt`; the with-margin arm does not reproduce the current Stage-1 result|Fix checkpoint selection and rerun|
|AnchorGuard|Cached checkpoint lacks current dataset and parent lineage; stale augmented fingerprint remains|Fix metadata, invalidate cache, rerun|
|Grouped Lane C|CSV still names the 159-row augmented run|Regenerate from the current checkpoint, then label it encoder-transductive|
|Forecasting and surprise|Notebook 09 loads the old augmented checkpoint while scoring current rows|Fix selector and rerun|

The AnchorGuard gate code also treats non-inferiority as an absolute difference. It should use a one-sided rule: an improvement should pass, not fail for being more than 0.05 away from baseline.

## 7. Experiments needed before strong claims

### P0: needed for a strong drift or forgetting claim

1. **Multiple seeds.** Run at least 3, preferably 5, full curricula and show the stage-wise distribution.
2. **Source-weighted anchor.** Report per-video values, an equal-video-weighted curve, and source-cluster uncertainty; test source-balanced replay.
3. **Alignment-invariant drift.** Report orthogonal-Procrustes-aligned normal drift and linear CKA or SVCCA.
4. **Matched optimization controls.** Compare with continued-normal and joint-training runs using the same number of updates.
5. **Functional retention.** Reserve source-grouped normal data before training, retrain, and test JEPA loss and a normal-versus-perturbed ranking task at every checkpoint.
6. **Order control.** Repeat at least one different condition order, or randomize order across seeds.

### P1: needed for mechanism and repair claims

1. Rerun group-weight on/off from the current Stage-0 checkpoint with identical RNG streams.
2. Require the on arm to reproduce the saved Stage-1 endpoint before interpreting the ablation.
3. Fix AnchorGuard lineage and one-sided gates.
4. Test a small anchor-weight sweep and compare feature spread, aligned retention, and held-out function.
5. Add label-permutation and raw, untrained, missingness, extraction-version, and source-aggregation controls.

### P1: needed for performance claims

1. Split source videos before fitting preprocessing choices.
2. Train the full five-stage encoder inside every outer training fold.
3. Fit the downstream model only on outer-training features.
4. Open each outer test fold once.
5. Report per-source predictions and uncertainty across folds or repeated grouped splits.

### P2: needed for a clinical direction

Add person identifiers, external sites and cameras, adjudicated labels, demographic and severity information, and an external validation cohort.

## 8. Submission readiness

### Venue requirements

The workshop requires:

- at most five main-text pages, excluding references and appendices;
- the modified NeurIPS 2026 LaTeX style;
- full anonymization for double-blind review;
- OpenReview submission; and
- a September 5, 2026 AoE deadline.

Accepted papers are non-archival but will appear on OpenReview and the workshop website.

### Readiness score

|Area|Status|Reason|
|---|---|---|
|Topic fit|Strong|Movement, continual learning, representation audit, and reproducibility fit the call|
|Current data lineage|Strong for the primary run|Checkpoint, fingerprint, row counts, and drift curve agree|
|Core causal claim|Weak|No alignment, continued-normal, order, or multi-seed controls|
|Repair claim|Blocked|AnchorGuard artifact is not current-lineage evidence|
|Generalization|Blocked|No fold-local encoder retraining|
|Clinical claim|Blocked|No clinical validation design|
|Ethics and data use|Blocked|No institutional ethics determination or completed data-use review is recorded in the workspace|
|Paper format|Draft ready|The official-style anonymous PDF has four main pages; appendices and references begin on page 5|
|Anonymization|Passed for current build|The PDF has template-anonymous authors, no author metadata, and no local identity strings; recheck the exact upload file|

**Overall:** about **30% ready for a strong empirical paper**, **60% ready in scientific content for an honest work-in-progress paper**, and **60% ready as an actual submission package**. The package score improved after conversion to the official anonymous style; unresolved ethics review and final release checks still block upload. These numbers are judgment calls, not venue acceptance probabilities.

## 9. Best submission strategy

With roughly two days until the deadline, there are two defensible choices.

### Submit a preliminary audit paper

Use the current paper title and verified evidence only. Make the main contribution the distinction between coordinate drift and forgetting. State that the P0 experiments are ongoing. Do not include AnchorGuard, margin attribution, Lane C, or forecasting results.

### Postpone the strong paper

Run the P0 and P1 experiments, then submit a later version with a defensible mechanism and intervention. This is the better scientific choice if the goal is a durable empirical paper rather than workshop feedback.

## 10. Final pre-submission checklist

1. Rebuild `bbfm2026_paper_draft.tex` with the included official `neurips_2026.sty` and confirm the `dblblindworkshop` option remains active.
2. Confirm that the main paper remains within five pages; the current build uses four.
3. Replace informal references with complete citations.
4. Remove author names, affiliations, repository names, local paths, PDF author metadata, and acknowledgments.
5. Check every number against the Appendix A ledger.
6. Confirm that no `_augmented` file supplies a current result.
7. Confirm that no mixed-lineage AnchorGuard, Lane C, or surprise value remains.
8. Use "coordinate drift," not "forgetting," in the title, abstract, figures, and conclusion.
9. Record the institutional ethics determination, complete the data-use review, and retain the paper's data-use limitation statement. The [official GAVD repository](https://github.com/Rahmyyy/GAVD) distributes annotations and URLs rather than raw videos and places compliance responsibility on users.
10. Inspect the final PDF at actual size before upload.
