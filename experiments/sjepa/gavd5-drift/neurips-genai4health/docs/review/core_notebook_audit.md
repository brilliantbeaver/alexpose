# Core notebook evidence review for GenAI4Health

Review updated September 6, 2026. This is an internal evidence report, not submission prose. The review read the current source and saved numerical outputs of notebooks 00–06, the split, pose-cache, and crop-geometry modules, the existing paper and extended abstract, and the earlier verification records. Cell numbers are zero-based. No training, pose extraction, model inference, or source-notebook execution was performed.

## What the current evidence supports

The strongest completed finding from these notebooks is a small source-held-out comparison between learned pose features and two direct controls. Across the same 20 test uploads, normalized pose summaries classified 10 correctly, whereas learned features and missingness summaries each classified 6 correctly. Their macro-F1 scores were 0.44, 0.29, and 0.25, respectively. These numbers agree between notebook 06's complete saved prediction table, the portable numerical supplement, and independently reconstructed metrics.

This is an exploratory result for one trained model and one split. It does not establish a general disadvantage of JEPA, a causal missingness shortcut, diagnostic performance, or an agentic health capability. Its value for a health-AI submission is a concrete example of why a learned sensing component needs a direct-data comparison before it is interpreted or incorporated into a clinical system. The source-weighting result from notebook 08 is complementary and is reviewed separately.

The newer notebook 00 is clearer about these boundaries. It now has 29 cells and explicitly distinguishes its teaching architecture from the executed model in notebook 04. It adds explanation, not a new empirical health result. Notebooks 04–06 still report one outer fold and one model initialization. The five-fold design must not be described as five completed evaluations.

## What was verified again, and what remains historical

The original fold checkpoint, stage checkpoints, and full source registry are absent from this checkout. Consequently, the original full-artifact verifier fails on its missing protocol inputs, and the September 5 report's full checkpoint verification cannot be presented as a new September 6 check. Legacy checkpoint files elsewhere in the artifact folder belong to different experiments and must not be substituted.

A targeted read-only checker, [verify_current_core_exports.py](verify_current_core_exports.py), now verifies the available evidence directly:

- It scans the raw annotation CSVs and checks the retained public-metadata manifest.
- It verifies cohort and role counts in the retained pose-QC ledger, including source-role disjointness and the 0.50 coverage rule.
- It compares all 60 saved notebook-06 prediction rows with the 60 aliased supplement rows and independently reconstructs the three readout scores.
- It checks that shared setup, data-loading, preparation, and model cells in notebooks 04–06 are identical.
- It identifies the missing original checkpoint explicitly instead of treating numerical reconstruction as training reproduction.

The current public manifest has 657 rows, while its decoding field still says pending. The decoded-cohort total is therefore reconstructed from the 655 locked rows in the QC ledger and agrees with notebook 01's saved output; it is not obtained by claiming that the current public manifest contains a completed decoding decision or by downloading media again.

The portable numerical verifier originally failed on raw-file checksums because the recorded files had Windows CRLF endings and the current copies have LF endings. Converting the current files to CRLF in memory reproduces both historical checksums. All values are unchanged. Verification now uses an explicit canonical-LF convention, retaining the original raw checksums as historical metadata. All 60 prediction rows and all five weighting rows were crosschecked before refreshing that record.

Run from the experiment root:

```bash
.venv/bin/python neurips-genai4health/docs/review/verify_current_core_exports.py
.venv/bin/python neurips-genai4health/docs/numerical_supplement/verify.py
```

The first command also needs the retained annotation and QC files. The second requires only the small numerical supplement and Python's standard library.

## Cohort and evaluation units

| Gate | Clips | Source videos | Annotated frame rows |
|---|---:|---:|---:|
| Raw selected annotation folders | 666 | 103 | 140,641 |
| Public metadata in the dated snapshot | 657 | 100 | 137,690 |
| Locked decoded cohort in the QC ledger | 655 | 98 | 135,804 |
| Pose-coverage eligible | 639 | 97 | 134,259 |

Frame totals sum the annotated rows of each sequence; they are not demonstrated counts of globally unique frames when excerpts overlap. An upload can contain several people, and the same person can appear in different uploads. Use “source videos” or “uploads,” not “patients” or “independent people.”

| Annotation | Raw clips / sources | Public metadata | Decoded cohort | Pose eligible |
|---|---:|---:|---:|---:|
| Normal | 291 / 32 | 291 / 32 | 290 / 31 | 276 / 30 |
| Parkinson's | 47 / 11 | 47 / 11 | 47 / 11 | 46 / 11 |
| Stroke | 76 / 19 | 75 / 18 | 75 / 18 | 74 / 18 |
| Myopathic | 188 / 30 | 184 / 29 | 183 / 28 | 183 / 28 |
| Cerebral palsy | 64 / 11 | 60 / 10 | 60 / 10 | 60 / 10 |

Two public-metadata sources fail the recorded acquisition/span check: one downloaded video ends before the annotated span, and another has a bounded, retryable acquisition failure. The latter is not proven permanently unavailable. Sixteen decoded clips fail the coverage threshold: 14 normal, one Parkinson's, and one stroke. The excluded extra cache is outside the locked cohort.

| Role after pose QC | Clips | Sources | Normal | Parkinson's | Stroke | Myopathic | Cerebral palsy |
|---|---:|---:|---:|---:|---:|---:|---:|
| Training | 377 | 59 | 156 / 18 | 28 / 7 | 44 / 11 | 111 / 17 | 38 / 6 |
| Validation | 131 | 18 | 64 / 5 | 9 / 2 | 15 / 4 | 35 / 5 | 8 / 2 |
| Test | 131 | 20 | 56 / 7 | 9 / 2 | 15 / 3 | 37 / 6 | 14 / 2 |

Condition cells give clips / sources. Source roles were assigned before later acquisition and QC attrition. Their post-QC sizes differ from the original 60 / 20 / 20 allocation. The available test set retains 20 videos, but that does not imply all other folds have the same retention.

## Notebook-by-notebook findings

### 00: a teaching model and synthetic arithmetic checks

The saved forward/backward trace uses two hand-authored 32-frame walks. Its prediction and target arrays both have shape `(2, 57, 32)`; the loss is finite and the target encoder has zero gradients. The mask hides 57 of 96 eligible positions per input, about 59%, or about 22% of all 264 positions. These are software checks, not gait-recognition results.

This model flattens four frames into 12 coordinate values, removes masked tokens before encoding, uses a Transformer predictor, and compares centered soft-target distributions. The trace does not call an optimizer, target-update, or center-update step. Its updated prose correctly says that these choices differ from notebook 04's trained model. Neither the synthetic trace nor its conceptual graphics should supply architecture or performance claims for the paper.

### 01: source grouping and separate acquisition gates

The split module assigns uploads, rather than clips, to roles. It uses exact per-condition source quotas differing by at most one, then optimizes clip-count balance. Five outer folds are specified. Each development set is partitioned four ways, but only one validation partition is selected for a given outer fold. This is not an executed full nested cross-validation.

The decoding gate checks opening, positive frame rate, sufficient duration, and first/last required frame decoding. Full extraction is a separate check. Source grouping is useful rigor, but notebook phrases such as “opened once” describe intended within-workflow use, not a global access-control system or a retrospective guarantee that analysts never saw test outcomes.

Manifest fingerprints bind selected identifiers and frame-span counts. They do not bind all annotation boxes, pose arrays, source-media bytes, or research decisions.

### 02: fixed pose extraction and limitations of old caches

A fixed MediaPipe Lite model processes annotated walker crops in video mode, with 15% padding and confidence thresholds of 0.45. It stores image-normalized horizontal and vertical coordinates, estimated relative depth, and visibility. These are monocular estimates, not calibrated 3D motion-capture measurements.

The crop helper normalizes the annotation box using its declared source dimensions, then projects it into the decoded rendition. The documented preview bug involved drawing 640 × 360 crop pixels unchanged on a 1280 × 720 frame. Correcting that preview does not show that all prior trajectories were recomputed or independently validated.

The saved run reused all 655 locked caches. Their present ledger entries all lack the newer frame-size and normalized-crop fields. Cache checks verify identity, timeline, shape, frame rate, model identity, and recorded split metadata, but do not verify every trajectory against expert motion capture or reconstruct original media bytes. Legacy metadata migration preserves the scientific arrays.

Pose QC is mean visibility coverage across 12 selected landmarks, retaining clips at or above 0.50. This is a detector-coverage rule. It is not a clinical gait-quality rating. One remaining notebook-02 sentence says notebook 04 interpolates short gaps; its actual code does not, so submission methods must follow the code.

### 03: a fixed target set, with no efficacy ablation

The eligible landmarks are shoulders, hips, knees, ankles, heels, and foot tips: indices 11, 12, and 23–32. All 33 joints can provide context. The sampler uses validity alone and draws uniformly within the eligible set, without motion scores. A common number of targets is based on the least-covered sample in the batch.

The synthetic invariants pass: no forbidden targets, the same seed repeats the mask, and each sample has the same target count. The nominal 0.60 fraction applies to valid eligible targets, not to every token. A different, separately implemented Torch function provides this mechanism in notebook 04.

The anatomical choice is a plausible engineering prior, informed by the project's gait-feature mapping. No completed unrestricted-mask comparison or clinical validation establishes that this particular selection improves health representation learning. Do not describe its physiological meaningfulness as an empirical discovery.

### 04: the executed compact learner

The recorded model configuration and source define 64 resampled frames and 16 segments. Each joint's four-frame coordinate mean is linearly projected into a 64-dimensional token; two pre-normalized Transformer blocks use four heads and feed-forward width 128. Masked content is replaced with a learned mask vector while the position stays in the sequence. Invalid positions are excluded from targets and pooling but are not excluded from contextual attention by a padding mask.

The predictor averages visible valid contextual tokens, adds a learned target position, and applies an MLP. The target encoder is an exponential-moving-average copy. The executed objective is

$$
L = L_{\mathrm{SmoothL1}} + 0.10L_{\mathrm{variance}} + 0.01L_{\mathrm{covariance}}.
$$

There is no separate two-view invariance term, teacher centering, coordinate decoder, or geometric-view augmentation in this run. “VICReg-inspired variance and covariance regularization” is more precise than implying that the full VICReg objective is used. The optional condition-classification term is disabled.

The cumulative training order introduces normal, Parkinson's, stroke, myopathic, and cerebral-palsy annotations while retaining all previously introduced groups. The loss excludes condition labels, but the schedule uses them. This is neither disease progression nor a no-replay continual-learning experiment. Updates sample a source uniformly and then a clip from it; this does not equalize category counts.

The source defaults are 20 epochs per stage, 100 updates per epoch, batch size 32, AdamW learning rate 0.001, weight decay 0.0001, gradient clipping 1, EMA decay 0.996, and masking fraction 0.60. Several can be overridden by environment variables and were not persisted in the retained checkpoint metadata. The saved stage output confirms selected epochs 7, 0, 0, 4, and 0, with objectives approximately 0.151, 0.130, 0.122, 0.114, and 0.126. The historical verification read a 100-row training history, but that original history is absent here.

Checkpoint selection averages per-batch validation objectives, rather than weighting sources equally. Masks change between epochs. After each stage, the best weights are restored without restoring the optimizer moments from that epoch. These details matter for a future confirmatory run; the falling objectives across changing stage populations do not establish improved health function.

### 05: descriptive geometry without clinical separation

The EMA encoder supplies 256 features: means and standard deviations over all valid tokens and over the selected 12 joints. The normal reference and feature scale are constructed using training-source summaries. Same-source neighbors are excluded from retrieval.

Saved condition silhouettes are approximately −0.11 for training, −0.26 for validation, and −0.19 for test. These do not support a positive class-separation claim. Two displayed nearest-neighbor examples, including a cerebral-palsy query retrieving a myopathic source, are illustrations rather than a retrieval-performance benchmark. High cosine alone does not validate a clinically meaningful latent space.

### 06: the main completed comparison

| Input to logistic regression | Feature count | Correct / test sources | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Learned EMA features | 256 | 6 / 20 | 0.30 | 0.26 | 0.29 |
| Validity/missingness only | 97 | 6 / 20 | 0.30 | 0.25 | 0.25 |
| Normalized pose summaries | 144 | 10 / 20 | 0.50 | 0.44 | 0.44 |

Raw summaries concatenate coordinate means and standard deviations, mean absolute frame-to-frame differences, and difference standard deviations over the 12 joints. Resampling and image normalization mean these differences are not physical velocities. Missingness concatenates mean validity per joint and per resampled frame.

Each readout uses standardized features and class-weighted logistic regression. Validation selects among regularization values 0.1, 1, and 10; selected values are 1 for learned features, 1 for missingness, and 10 for pose summaries. Both scaler and classifier are then refit on all 77 training-plus-validation sources.

An important asymmetry remains: fitting and validation use one mean feature vector per source, while testing averages probabilities computed for individual clips. Those operations are not equivalent. The same asymmetry applies to all three methods but may affect them differently; call this a same-source exploratory comparison rather than a perfectly matched performance experiment.

The raw-minus-learned macro-F1 difference is about 0.15. All three methods incorrectly label all three stroke-annotated test videos. These observations should be reported without a significance or clinical-risk estimate. The two methods with 6 correct videos do not necessarily make the same mistakes, and matching accuracy does not establish a shared causal shortcut.

Higher scores from the notebook's archived experiments must remain excluded. They involve different cohorts, architectures, objectives, and test exposure; comparing them with this split cannot isolate a numerical “leakage penalty.”

## Recommended contribution and evidence selection

Use the source-weighting result and the source-level baseline comparison to support a position about evidence needed before a learned movement component informs health decisions. This is distinct from the laterality paper: the target here is five dataset annotations and the central methodological concern is how evaluation unit and measured endpoint affect interpretation.

One concise method diagram and one figure containing the two main observations are sufficient. A cohort table can sit in supplementary material. Training-loss curves, high-cosine retrieval examples, and synthetic teaching graphics would add apparent activity without demonstrating additional health capabilities.

The position should remain proportionate. Standard source grouping and averaging are established methods, and this small case does not introduce a new JEPA algorithm. The contribution is the concrete, reproducible demonstration of how a real movement-learning pipeline can support narrower conclusions than its intended health application. Whether that position is sufficiently novel for the workshop remains a reviewer judgment.

## Improvements that require new experiments

A stronger empirical paper needs repeated folds and seeds, consistent source aggregation during tuning and testing, a matched random-encoder baseline, and controls for the cumulative training order. Functional retention needs a fixed held-out task, not cross-checkpoint cosine alone. Forecasting needs past-only inputs and preprocessing, a defined horizon, and relevant baselines. A health assistant would additionally need a defined workflow, uncertainty handling, and clinical evaluation.

None of these are completed by editing the manuscript or reconstructing old scores. Since test results have already been inspected, any confirmatory continuation should freeze decisions transparently and preferably use a new external evaluation cohort. The current submission can be stronger through clarity and precise scope, without claiming new validation that was not performed.
