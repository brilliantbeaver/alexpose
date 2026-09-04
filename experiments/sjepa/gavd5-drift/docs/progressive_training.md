# Normal-first progressive skeleton-JEPA training

This document describes the completed **GAVD-only** run in notebook 04. It is a project-specific S-JEPA-inspired variant, not a reproduction of the published S-JEPA. The optional added-normal cohort is off. Files with `_augmented` in their names describe an older run and are not part of this contract.

## 1. Research question

The model first learns only from normal gait. It then continues through four condition-labeled stages while replaying earlier groups. The audit asks:

1. Does training remain numerically stable?
2. Does the representation avoid total constant collapse?
3. How much do raw normal coordinates change after Stage 0?
4. What can and cannot be inferred from the final in-corpus representation?

The current run answers these implementation and representation questions. It does not test unseen-person or clinical performance.

## 2. Current data contract

The availability-filtered GAVD set has 642 sequences from 94 source videos. A neurologic-landmark coverage threshold of 0.50 removes 16 rows before training or frozen-feature analysis.

|Condition|Available rows / videos|Training rows / videos|Rows removed|
|---|---:|---:|---:|
|Normal|284 / 30|270 / 29|14|
|Parkinson's|42 / 9|41 / 9|1|
|Stroke|75 / 18|74 / 18|1|
|Myopathic|183 / 28|183 / 28|0|
|Cerebral palsy|58 / 9|58 / 9|0|
|**Total**|**642 / 94**|**626 / 93**|**16**|

The report of dropped normal rows is `normal_low_coverage_dropped.csv`. The full extraction and classifier coverage tables provide the remaining row-level audit.

Across the 642 availability-filtered pose caches, the recorded extraction labels are 546 `gavd5`, 95 `gavd3`, and one `gavd4`. The modeled 626 rows contain 530, 95, and 1, respectively. The pose-model hash agrees, but a source-disjoint experiment should still control this version history.

## 3. One continuing model

The view encoder, EMA target encoder, predictor, target center, and VICReg projector continue across all stages. They are not reinitialized at stage boundaries.

Each stage starts a fresh AdamW optimizer and learning-rate schedule. Thus the model weights continue, but optimizer moments, warmup, and schedule position restart.

AdamW uses betas (0.9, 0.95) and weight decay 0.05. The learning rate is $10^{-3}$ for Stage 0 and $3\times10^{-4}$ later. Each optimizer batch draws four rows from every active condition. The saved run reports seed 42 and the MPS backend; the exact hardware model and full deterministic-computation settings were not recorded.

|Stage|New group|Active rows|Active source videos|Epochs|Optimizer updates|Checkpoint|
|---:|---|---:|---:|---:|---:|---|
|0|Normal|270|29|300|20,400|`sjepa_normal.pt`|
|1|Parkinson's|311|38|75|5,100|`sjepa_stage_01_parkinsons.pt`|
|2|Stroke|385|56|75|5,100|`sjepa_stage_02_stroke.pt`|
|3|Myopathic|568|84|75|5,100|`sjepa_stage_03_myopathic.pt`|
|4|Cerebral palsy|626|93|75|5,100|`sjepa_stage_04_cerebralpalsy.pt`|
|**Total**||||**600**|**40,800**|`sjepa_curriculum_final.pt`|

Balanced replay draws the same number of samples from every active condition in each batch. This balances the optimization stream; it does not change the stored cohort imbalance.

## 4. Pose preprocessing

Each raw pose record contains 33 landmarks, x/y/z coordinates, and visibility.

The preprocessing sequence is:

1. mark a point observed when visibility is at least 0.45 and coordinates are finite;
2. interpolate only short internal gaps of at most four frames;
3. center each frame on the pelvis;
4. divide by a sequence-level shoulder/hip scale;
5. resize each sequence to 64 frames;
6. preserve a validity mask so missing points cannot become targets; and
7. remove sequences with less than 0.50 observed fraction over the 12 target landmarks.

Resizing every clip to 64 frames does not preserve native duration as a model input. Absolute timing claims therefore require a separate treatment of timestamps or frame rate.

## 5. Token and mask rule

A token contains one landmark over four consecutive frames. The recommended configuration has:

```text
64 frames / 4 frames per patch = 16 time patches
16 time patches * 33 landmarks = 528 tokens
```

Only 12 landmark identities may become prediction targets: shoulders, hips, knees, ankles, heels, and foot tips.

For each batch, the sampler:

1. counts valid eligible tokens per sample;
2. takes the smallest count in the batch;
3. multiplies that count by the configured mask fraction, 0.60;
4. takes the floor; and
5. samples that same number of eligible targets from every sample.

The 0.60 value is therefore not an exact per-sample masking fraction. Samples with more valid eligible tokens realize a smaller fraction.

## 6. Three loss components

The optimized objective is

$$
\mathcal L = \mathcal L_{\mathrm{JEPA}}
+ 0.05\,\mathcal L_{\mathrm{VICReg}}
+ 0.25\,\mathcal L_{\mathrm{group}}.
$$

### JEPA prediction

The student sees visible tokens. The EMA teacher sees the complete cleaned sequence. The predictor estimates centered, sharpened teacher features at the hidden tokens. This is same-clip latent infilling, not future forecasting.

### VICReg

VICReg compares projected student features from two geometric views. It combines invariance, variance-hinge, and covariance terms. It uses no condition labels and discourages, but does not guarantee against, collapse.

### Group term

The group term is zero in Stage 0. In Stages 1 through 4 it uses condition labels to reduce within-condition distance and penalize condition centroids that are closer than the configured margin.

The short epoch log is easy to misread:

- `VICReg` is the inner combined value before the outer 0.05 weight;
- `group` prints only the separation component, not compactness plus separation; and
- `std` is a post-epoch teacher-feature diagnostic, not a loss term.

## 7. Stage-end measurements

|Stage|JEPA loss|VICReg loss|Feature std|Mean pair cosine|Min normalized Euclidean centroid distance|Normal-anchor cosine|
|---:|---:|---:|---:|---:|---:|---:|
|0|0.700|16.791|0.291|0.283|not defined|1.000 by definition|
|1|0.743|12.910|0.219|0.726|0.883|0.700|
|2|0.498|10.775|0.229|0.668|0.704|0.502|
|3|0.451|9.332|0.226|0.569|0.616|0.396|
|4|0.370|8.237|0.229|0.440|0.534|0.297|

These values come from `training_history.csv` and `curriculum_stage_summary.csv`.

The stage diagnostics use a 96-D validity-weighted mean of target-encoder tokens at the 12 authorized landmarks. The centroid column is Euclidean distance after normalizing those 96-D vectors. The normal anchor is the mean of 270 matched, per-sequence cosine similarities in the same 96-D space. For every normal row, its representation at the current stage is compared with that same row's Stage-0 representation. It is not the cosine between two cohort means.

This mean weights rows equally. Normal videos contribute from 1 to 60 rows, with a median of 4. The top two videos supply 105 of 270 rows and the top three supply 137. A population-level analysis needs per-video and equal-video-weighted summaries, source-cluster uncertainty, and preferably source-balanced replay.

## 8. How to interpret the run

Training stayed finite and the final features were not all identical. That is evidence against total constant collapse, not proof that the learned dimensions are useful.

The raw normal-anchor cosine fell sharply, and reloading the saved seed-42 checkpoints reproduces the logged curve within `4.51e-7`. It is correct to say that matched normal-sequence coordinates drifted on average in this run. It is not yet correct to say that the model forgot normal gait, because the current metric is not invariant to a global feature rotation and no held-out normal function was measured.

The separate frozen-feature geometry and classifiers use a 384-D concatenation of global mean, global standard deviation, authorized mean, and authorized standard deviation. Their cosine distances are not numerically comparable with the stage-end 96-D Euclidean centroid distances. The final label-informed representation has in-corpus structure. That is expected in part because the group term used the same folder labels during training. Downstream probes on these rows remain transductive.

## 9. Reproducibility contract

```text
final checkpoint: sjepa_curriculum_final.pt
fingerprint:      7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2
file SHA-256:     64008d77689cefa4beb51a0dcf5ed6cae743454134c163e9087f66510af4e7ad
seed:             42
augmented normal: false
```

Every checkpoint records its completed stage, parent fingerprint, sequence IDs, condition order, loss settings, model configuration, and masking whitelist. A consumer should reject a checkpoint when any of those fields do not match its expected run.

## 10. Experiments still needed

Before calling the curve forgetting or attributing its cause, run:

1. at least three full seeds;
2. per-video and equal-video-weighted anchors with source-cluster uncertainty;
3. Procrustes-aligned cosine plus linear CKA or SVCCA across stages;
4. a same-duration normal-only continuation control;
5. at least one alternate condition order;
6. a held-out normal-function test at every checkpoint;
7. a current, exact group-weight ablation whose with-group arm reproduces Stage 1; and
8. fold-local full retraining for any unseen-video performance claim.

Notebook 08's current ablation and AnchorGuard files do not meet the lineage requirements above and must be rerun before they are reported.
