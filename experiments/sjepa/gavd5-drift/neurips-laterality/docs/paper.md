# Build the Geometry In, Don't Hope It Emerges: A Reflection-Symmetry Audit of a Skeleton World Model

**Anonymous submission draft — NeurIPS 2026 Workshop on Physical World AI: Geometry, Characteristics, and Multimodal Sensing. Under double-blind review; do not distribute.**

> **Evidence status.** This document describes the registered laterality v2.1 protocol. The earlier transductive experiment, in which an encoder was evaluated on sequences that had also entered its training, is superseded. Its numerical results are not evidence for the v2.1 study and are intentionally omitted here. The v2.1 paper run is not complete, so this draft makes no empirical performance claim. Results may be added only after all registered folds, seeds, variants, lineage checks, and governance requirements are complete.

## Abstract

The human body has an approximate left–right structure: reflecting a pose and exchanging anatomically corresponding left and right landmarks produces another valid pose representation. We use this structure to ask whether a self-supervised skeleton model learns a representation that behaves predictably under reflection. The study uses pose sequences derived from the Gait Abnormality in Video Dataset (GAVD) and a Skeleton Joint-Embedding Predictive Architecture (S-JEPA). Its target is a coordinate-derived comparison of left-side and right-side motion, not a diagnosis or validated clinical measurement.

The evaluation is designed around held-out source videos. The quality-controlled cohort contains 625 pose sequences from 93 source videos. Five outer cross-validation folds are created at the source-video level, so all sequences originating from one video remain together. In each fold, the encoder is trained from initialization on approximately 80% of the source videos and tested on the remaining approximately 20%. Four inner folds, built only from the outer-training sources, select the ridge read-out penalty; the outer-test sources are not used for model training, preprocessing, read-out fitting, or parameter selection. Source-video bootstrap intervals and repeated optimization seeds represent different sources of uncertainty and are reported separately. This design supports a narrow claim about post-development performance on held-out GAVD source videos. It does not establish generalization to unseen people, clinical validity, population prevalence, or treatment effects.

## 1. Introduction

A world model learns useful structure by predicting or reconstructing aspects of its input without relying on the final task label. For a model of an articulated body, geometry supplies a natural test of what the representation has learned. Here we focus on bilateral reflection: the transformation that flips a skeleton horizontally and exchanges every named left body landmark with its right partner.

The central question is deliberately limited. We ask whether a representation trained without the laterality target contains useful information about a signed left-versus-right motion contrast and whether that information changes correctly when the input is mirrored. We separate three statements that can otherwise be confused. First, a mathematical wrapper can force a read-out to change sign. Second, a trained encoder followed by a probe can exhibit useful sign-changing behavior. Third, the encoder tokens themselves can satisfy a direct reflection-equivariance test. The first statement is a construction; the latter two require empirical evidence.

The revised protocol also separates model development from evaluation. Earlier exploratory work used a transductive design, meaning that the encoder had seen the evaluated sequences during representation training. That design was useful for generating hypotheses but could not support a held-out-video claim. Laterality v2.1 replaces it with fold-local training: a fresh encoder is trained for each outer fold, and the corresponding outer-test videos remain unavailable until evaluation.

## 2. Concepts and terminology

A **pose sequence** is a series of video frames represented by estimated body-landmark coordinates rather than by the original images. Each sequence contains 33 BlazePose landmarks, and each landmark has horizontal, vertical, and depth coordinates. A **source video** is the original video from which one or more pose sequences were extracted. Because several sequences can come from the same video, treating sequences as independent during splitting would allow nearly related material to appear in training and testing.

An **encoder** converts the pose sequence into learned numeric features. A **read-out**, also called a probe, is a simpler model fitted on those features to predict the laterality target. The encoder and read-out have different training procedures and therefore different uses of validation data. The encoder follows a fixed, pre-registered training schedule and does not use an early-stopping validation set. The read-out uses inner cross-validation to select its ridge penalty.

**Cross-validation** rotates which data are held out. An **outer fold** measures final held-out performance. An **inner fold** makes a training-only choice without consulting the outer test set. **Stratification** keeps the source counts for the dataset annotations reasonably balanced across folds. Those annotations are used for balancing and for an explicitly named confounding control; they are not treated as diagnoses or primary prediction labels.

## 3. Dataset inventory and cohort construction

GAVD is the Gait Abnormality in Video Dataset [ranjan2025gavd, gavdRepo2026]. Its official distribution supplies annotations and public video links rather than raw media files. This study operates on derived pose archives and does not redistribute source videos or identity-bearing frames.

The frozen inventory contains 666 annotation files describing 103 source videos. A matching derived-pose archive is available for 642 annotations; 24 annotations have no matching pose archive. The quality-control procedure then retains 625 pose sequences from 93 source videos and excludes 17 sequences. Exclusion can occur when the target cannot be computed, when too few selected landmarks are valid, or when too few complete four-frame model patches remain. The inclusion rules were fixed before the later model results were examined, and inclusion never depends on whether the target is positive, negative, large, or small.

Every retained sequence carries provenance information: the source annotation, pose archive fingerprint, extraction version, pose-model identity, and visibility threshold. A SHA-256 content digest, which acts like a fingerprint, binds the cohort table and arrays together. Later stages reject a cohort whose content or provenance does not match the frozen protocol.

## 4. Coordinate-derived laterality target

The target compares movement on the left and right sides of the pose representation. It uses five anatomical pairs: shoulders, knees, ankles, heels, and foot-index landmarks. The hips define the pelvis-centered coordinate system but are excluded from the target because pelvis centering makes their two velocity magnitudes algebraically identical.

For each pair, the calculation keeps only **paired-valid transitions**. This means that both the left and right landmark must be observed at both ends of the same frame transition. Let $m_{L,k}$ and $m_{R,k}$ denote the median left and right speeds for pair $k$. The normalized contrast is

$$
c_k = \frac{m_{L,k}-m_{R,k}}{m_{L,k}+m_{R,k}+\epsilon},
\qquad y=\operatorname{mean}_k c_k.
$$

The small constant $\epsilon$ prevents division by zero. Every registered pair must provide at least eight shared valid transitions, and all five pairs must be usable. Positive values mean relatively greater left-side motion under this formula, negative values mean relatively greater right-side motion, and zero indicates balance under the formula. These meanings are mathematical, not clinical.

Short missing stretches may be interpolated for encoder input, but interpolated coordinates never define the target. Invalid coordinates are accompanied by a validity mask and cannot contribute as numeric placeholders. Automated tests confirm that changing invalid placeholders leaves the target unchanged, that mirroring negates the target, and that mirroring twice restores the original coordinates and validity mask.

## 5. Training, validation, and testing design

### 5.1 Why the source video is the splitting unit

All sequences derived from one source video are assigned together. Mirrored inputs and any training augmentations inherit that assignment. This prevents direct video-level leakage: no clip, mirror, or augmentation from an outer-test source can enter the corresponding outer-training set.

The dataset does not supply a persistent person identifier, and this project does not infer identity. Two different source videos could therefore contain the same unidentified person. The split supports **held-out-source-video evaluation**, not **held-out-person evaluation**.

### 5.2 Five outer folds

The 93 eligible source videos are divided into five stratified outer folds. In every fold, approximately four fifths of the sources train the encoder and fit the final read-out, while approximately one fifth provide the untouched test set. Sequence counts vary because source videos contribute different numbers of pose sequences.

| Outer fold | Training sources | Test sources | Training sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

This is not one permanent 80/20 split. Each source serves as an outer-test source exactly once and as an outer-training source in the other four folds. Combining the out-of-fold predictions therefore gives one held-out prediction for every eligible sequence.

### 5.3 Four inner validation folds

Within each outer-training set, the source videos are divided again into four inner folds. One inner fold contains 18 or 19 validation sources, while the other 55 to 57 sources fit a candidate read-out. The validation role rotates until every outer-training source has served as inner validation exactly once.

Because sources contribute different numbers of sequences, an inner fitting partition contains 266 to 450 sequences and an inner validation partition contains 80 to 197 sequences. These are rotating read-out partitions, not additional encoder-training or final-test sets. Their exact memberships are stored in the split artifact.

These inner folds select the ridge penalty for the read-out. They do not select an encoder checkpoint, stop encoder training, or change the registered encoder schedule. After the penalty is selected, the read-out is fitted again on all 74 or 75 outer-training sources. Only then is it applied to the corresponding 18 or 19 outer-test sources.

The nesting is essential. If the outer-test targets helped choose the read-out penalty, the final performance estimate would no longer describe untouched held-out data. Likewise, preprocessing quantities such as feature scaling, the target scale, and the neutral band are estimated from outer-training data only.

## 6. Fold-local representation training

The representation model is an S-JEPA: a Skeleton Joint-Embedding Predictive Architecture. It receives 64 time steps with 33 landmarks, groups time into four-frame patches, and uses a 96-dimensional embedding, a four-layer encoder, a two-layer predictor, and four attention heads. During self-supervised training, portions of the sequence representation are masked and the model learns to predict the hidden latent content. The laterality target and dataset annotation do not enter this objective.

Training is **fold-local**. For each combination of outer fold, optimization seed, and training variant, a new encoder starts from the recorded seed-matched initialization and sees only sequences belonging to that fold's outer-training sources. Sampling first chooses a source video uniformly and then a sequence from that source, which prevents videos with many extracted sequences from dominating training.

The paper profile registers five optimization seeds, numbered 42 through 46, and two training variants. The `vanilla` variant never reflects a training sample. The `reflection_augmented` variant reflects a sample with probability 0.5. Within each seed, the variants share the same initialization, source draws, masks, geometric views, optimizer schedule, and number of updates; reflection decisions use a separate random stream. The complete design contains $5$ outer folds $\times 5$ seeds $\times 2$ variants, or 50 fold-local encoders. Each encoder follows the fixed 300-epoch paper schedule. A final report is incomplete if any registered checkpoint is missing.

## 7. Held-out evaluation

For a given fold and seed, feature extraction uses the encoder trained without that fold's test sources. The primary lane fits a free linear ridge read-out to the learned representation. Additional lanes distinguish several questions:

- an identically structured untrained encoder provides an initialization floor;
- a target-component self-consistency oracle confirms that the saved pair contrasts reconstruct the target; it is a check, not a learned baseline;
- pooled and missingness features test measured nuisance explanations;
- constructed one-pass and two-pass read-outs test whether antisymmetry can be guaranteed analytically;
- direct token comparisons test whether the encoder itself is reflection-equivariant.

The principal performance measure is source-balanced $R^2$, the proportion of target variation explained after giving each source video equal total weight. Mean absolute error, source-level sign accuracy, and mirror errors are secondary measures. Each checkpoint is evaluated separately. Metrics and squared mirror errors are computed within a checkpoint before registered seeds are averaged; signed errors are never allowed to cancel across seeds.

An analytically antisymmetric read-out and an empirically equivariant representation are not the same result. Exact output oddness proves that the wrapper was implemented correctly, but useful learned content additionally requires positive held-out predictive utility and improvement over the paired untrained initialization. A strict representation claim separately requires low direct token-equivariance error and improvement over initialization.

## 8. Uncertainty and decision rules

Two types of variation are kept separate. Optimization-seed variation asks how results change when training begins from a different registered random seed. Source-sampling uncertainty asks how the estimate changes when the 93 observed source videos are resampled. The latter uses 2,000 bootstrap repetitions at the source-video level, preserving all sequences from a sampled source together. These intervals are conditional on the already fitted cross-validation pipeline; they do not turn the observed videos into a random sample of all people or all gait videos.

Decision thresholds are frozen in `config/protocol.json`. Native probe behavior requires useful absolute held-out prediction, sufficiently low mirror error, and improvement over the paired untrained encoder. Strict representation behavior requires sufficiently low token-equivariance error and a favorable learned-minus-initialized comparison. A null or adverse result remains a result and does not authorize retuning the protocol after inspection.

## 9. Evidence status and reporting boundary

The v2.1 cohort and source split have been audited, but the registered paper computation and held-out evaluation are not yet complete. Consequently, this document reports the method but no v2.1 performance number or empirical conclusion. The older transductive estimates answer a different question and must not be inserted into the tables, abstract, figures, or conclusion of this protocol.

When the run is complete, every reported number must trace to artifacts carrying matching protocol, context, cohort, split, checkpoint, and implementation digests. A paper table must include all five outer folds and all five registered seeds for the relevant variant. Smoke-profile output uses generated data to test software plumbing and must never enter an empirical table.

## 10. Limitations

The design prevents overlap at the source-video level, but it cannot guarantee that people are distinct across videos. The supported claim is therefore post-development, within-GAVD cross-validated performance on held-out source videos. It is not a claim about unseen people, a new dataset, clinical validity, prevalence, diagnosis, or treatment.

The target captures one signed left-versus-right motion contrast. It does not represent the full kinematic structure of walking and has not been validated as a clinical gait index. All learning claims remain conditional on the BlazePose landmark schema, pose-extraction system, preprocessing choices, architecture, registered seeds, and measured controls. Unmeasured acquisition and missingness effects may remain.

## 11. Ethics and data use

Public availability is not the same as authorization for research use or redistribution. The suite stores no raw videos or identity-bearing frames, but derived poses, source identifiers, embeddings, predictions, and checkpoints may remain linkable or sensitive. The institutional ethics determination, data-use review, and derived-pose release review are currently unresolved. Submission and artifact release remain blocked until all three reviews carry genuine dated internal references. Statistical safeguards cannot replace those determinations.

## 12. Reproducibility sequence

The notebooks are intended to be read and run in numeric order. Notebook 00 freezes the protocol and reports governance status. Notebook 01 audits cohort construction and the target. Notebook 02 creates the source-level outer and inner folds. Notebook 03 trains the fold-local encoders. Notebook 04 fits read-outs and produces held-out predictions. Notebook 05 aggregates the registered checkpoints and uncertainty estimates. Notebook 06 is a separate gate for a possible external dataset; the GAVD review cannot authorize that dataset.

The exact split assignments live in `artifacts/paper/protocol_<digest>/splits/source_splits.json`. The registered rules live in `config/protocol.json`, and the interpretation boundary is summarized in `PROTOCOL.md` and `RUNBOOK.md`.
