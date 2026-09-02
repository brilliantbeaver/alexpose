# Normal-first progressive S-JEPA training

This is the plain-language contract implemented by notebook 04. It describes the completed augmented-normal real run and explains what its diagnostics do and do not show.

![Cohort and curriculum](figures/cohort_curriculum.svg)

## Why train in stages?

Pooling every condition from the first update would make it hard to answer a basic question: what did the model learn from normal gait before condition labels influenced the representation?

The curriculum creates a visible baseline. Stage 0 starts from fresh weights and sees only normal sequences. Each later stage adds one condition while replaying every earlier condition. The model state continues across stages. This order gives us an anchor for measuring change, but it does not guarantee that the anchor will be retained.

## The two normal data sources

Stage 0 used 75 normal sequences from 18 videos:

- 12 canonical GAVD sequences from one source video;
- 63 accepted self-annotated windows from 17 additional YouTube videos.

The added windows use automatic MediaPipe bounding boxes. They are not canonical GAVD annotations and are not independently clinically verified. The extraction report contains 64 candidates. Notebook 04 accepts candidates whose neurologic-landmark coverage is at least 0.45. One 44-frame candidate had coverage 0.027 and was rejected.

This selection is now enforced from `augmented_pose_extraction_report.csv`. The loader checks that every accepted pose file exists and ignores rejected candidates. The cohort is therefore defined by a recorded rule rather than by the files that happen to remain in a folder.

## One continuing model

|Stage|New group|Active sequences|Source videos|Epochs|Optimizer updates|Checkpoint|
|---:|---|---:|---:|---:|---:|---|
|0|Normal|75|18|300|5,700|`sjepa_normal_augmented.pt`|
|1|Parkinson's|84|20|75|1,425|`sjepa_stage_01_parkinsons_augmented.pt`|
|2|Stroke|96|23|75|1,425|`sjepa_stage_02_stroke_augmented.pt`|
|3|Myopathic|143|33|75|1,425|`sjepa_stage_03_myopathic_augmented.pt`|
|4|Cerebral palsy|159|35|75|1,425|`sjepa_stage_04_cerebralpalsy_augmented.pt`|

The Stage 4 state is also saved as `sjepa_curriculum_final_augmented.pt`. Total training was 600 epochs and 11,400 optimizer updates.

The view encoder, predictor, EMA target encoder, target center, VICReg projector, and normal reference continue across stages. Model parameters are not reinitialized. Each stage deliberately creates a fresh AdamW optimizer and learning-rate schedule, so optimizer moments, warmup, and schedule position restart at the stage boundary. Each batch draws the same number of samples from every active condition. This balances the training stream even though the stored cohort is highly imbalanced.

The final experiment fingerprint is:

```text
ea59fea055f0230bcf236deb1d1e8bbf08033766e7cd95a98f28210b3042c4e4
```

This SHA-256 value identifies the experiment and data payload saved in the checkpoint. It is not the checksum of the checkpoint file itself.

## Pose preprocessing

Each pose sequence has shape `[time, 33, 4]`: three coordinates plus visibility. Notebook 04 applies these steps:

1. mark a joint valid when visibility is at least 0.45;
2. interpolate internal coordinate gaps of at most four frames while preserving the original validity mask;
3. subtract the pelvis center;
4. divide by a shoulder-and-hip width scale;
5. replace remaining invalid coordinates with a zero sentinel;
6. resize coordinates and validity to 64 frames;
7. join every four frames into one time patch.

The result has 16 time patches and 33 joints, or 528 possible joint-time tokens.

## The target whitelist and the actual mask rule

Only these BlazePose indices may become hidden prediction targets:

```python
MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
```

They are the left and right shoulders, hips, knees, ankles, heels, and foot indices. The list is expanded and de-duplicated from `experiments/multiple-sclerosis/mapping-data/ms-pd-mapping.md`. Applying it to this five-group experiment is a project rule, not a validated diagnostic rule.

The configuration says `mask_fraction = 0.60`, but 0.60 is an upper target rather than a promise for every sample. The sampler works as follows:

1. count valid eligible tokens in each sample in the batch;
2. take the smallest count;
3. multiply that count by 0.60 and round down;
4. cap the result so at least one eligible token remains visible;
5. sample that common number uniformly without replacement in every sample.

This batch-minimum rule gives tensors a common target count. A sample with more valid tokens therefore has a realized fraction below 0.60. Averaged across all logged epochs, the realized eligible fraction was 0.549 during Stage 0 and 0.421 during Stage 4. These are stage averages, not final-epoch values; the final Stage 4 epoch was 0.427. The sampler never reads position, motion magnitude, velocity, acceleration, or a learned motion score. Assertions reject every forbidden target.

## Three loss terms with three jobs

\[
\mathcal L_{total}
=\mathcal L_{JEPA}
+0.05\mathcal L_{VICReg}
+0.25\mathcal L_{group}.
\]

`L_JEPA` trains the predictor to match hidden target-encoder features. The target encoder receives the full sequence and no gradient. It follows the view encoder through a cosine EMA schedule that starts at 0.999.

`L_VICReg` compares two geometric views of a sequence. Its invariance term brings paired views together. Its variance term resists constant dimensions. Its covariance term reduces redundant dimensions. In the code, the usual VICReg components are `25 * invariance + 25 * variance + covariance` before the outer weight of 0.05 is applied.

`L_group` is zero at Stage 0. In later stages it combines within-label compactness with a squared penalty when normalized condition centroids are closer than margin 1.0. Its outer weight is 0.25. Because this term reads folder labels, Stages 1 through 4 are label-informed representation fine-tuning.

### VICReg step by step

VICReg and the group term operate on related but different tensors:

1. Create two independent geometric views of every sequence.
2. Run both through the trainable view encoder.
3. Pool valid tokens belonging to the 12 authorized landmark identities into one vector per sequence and view.
4. Pass both pooled vectors through the VICReg projector.
5. Compute VICReg on those **projected** vectors.
6. Compute the group terms separately from the **unprojected** pooled vector for the first view.

For projected batches $z_a$ and $z_b$, the invariance term is their elementwise mean squared error. The variance term computes the population standard deviation of every feature dimension in each view and penalizes only shortfalls below 1:

$$
L_{var}=\frac{1}{2}\left[
\operatorname{mean}_d\max(0,1-\sigma_d(z_a))
+\operatorname{mean}_d\max(0,1-\sigma_d(z_b))
\right].
$$

The covariance term centers each view, forms its feature covariance matrix, squares its off-diagonal entries, and averages them. Diagonal entries describe each feature's own variance; off-diagonal entries describe features changing together. Penalizing only the off-diagonal entries discourages duplicated information without directly suppressing each feature's own spread.

These terms are combined inside VICReg as `25 * invariance + 25 * variance + covariance`. The result printed as `VICReg` is this inner, unscaled value averaged over the epoch. The total optimizer objective multiplies it by 0.05. VICReg never reads folder labels and does not directly create condition clusters.

### What the centroid penalty does

For the group objective, each unprojected pooled sequence vector is scaled to unit length. Vectors sharing a condition label are averaged to form a condition centroid, and the centroid is normalized again. For each pair of centroids with Euclidean distance $d_{ij}$, the separation term is

$$
L_{sep}=\operatorname{mean}_{i<j}\left[\max(0,1-d_{ij})\right]^2.
$$

Distances at or above margin 1.0 contribute zero. Distances 0.9, 0.8, and 0.5 contribute 0.01, 0.04, and 0.25. On the unit sphere, distance 1.0 corresponds to a 60-degree angle and cosine similarity 0.5. The companion compactness term is the mean squared distance from each normalized sequence vector to its own normalized centroid. The optimized group loss is `compactness + separation`; one term tightens each labeled cloud and the other discourages centroid overlap.

### Decode the abbreviated epoch line

For example:

```text
JEPA 0.5337  VICReg 13.0367  group 0.0008  std 0.4134
```

- `group 0.0008` is only the epoch-mean $L_{sep}$, averaged over centroid pairs and balanced batches. It is not `compactness + separation`, so it is not the full group contribution to total loss. A small value says that batch centroids usually met or nearly met the margin; it does not prove individual samples form clean clusters.
- `std 0.4134` is computed after the epoch from unprojected, authorized-pooled **EMA target-encoder** vectors for the entire active corpus. The code takes the population standard deviation of every feature dimension and then averages across dimensions. It is not the VICReg variance term, does not use the VICReg projector, is not backpropagated, and has no required target of 1. A nonzero value is evidence against every sequence mapping to exactly the same vector, but it does not identify what information created the variation.

The printed losses are epoch means over optimizer batches, whereas `std` is one whole-corpus diagnostic measured after those updates. Their numeric scales should not be compared directly.

## Measured stage endpoints

|Stage|JEPA loss|VICReg loss|Compactness|Margin penalty|Feature std|Pair cosine|Minimum training centroid distance|Normal-anchor cosine|
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|0|0.526|16.990|0|0|0.445|0.417|not applicable|reference|
|1|0.534|13.037|0.0856|0.0008|0.413|0.509|0.610|0.959|
|2|0.673|10.584|0.0723|0.0019|0.379|0.632|0.470|0.849|
|3|0.597|9.472|0.0764|0.0292|0.360|0.678|0.323|0.729|
|4|0.487|8.312|0.0761|0.0357|0.363|0.660|0.259|0.617|

![Completed training diagnostics](figures/training_health.svg)

## Step-by-step interpretation

### 1. Did training diverge?

No. Losses stayed finite through all stages, all checkpoints were written, and the parent lineage is complete.

### 2. Did every feature collapse to one constant?

No, they did not, but that is a narrow win. It helps to keep four things apart: what we measured, what those measurements support, what they do not support, and what to do next.

**What we observed.** At the end of Stage 4 the mean feature standard deviation was 0.363 and the mean pairwise cosine was 0.660. The normal-anchor cosine had fallen to 0.617. On the canonical 96 sequences, notebook 05 measured a cosine silhouette of 0.054, and the two closest condition centroids were only 0.026 apart.

**What those numbers support.** The features did not all become the same vector. A standard deviation of 0.363 is well above zero, and a pairwise cosine of 0.660 is well below one, so the representation still tells sequences apart. At the same time, the normal reference moved a long way from where Stage 0 left it, and five clean condition groups did not form. A silhouette of 0.054 is close to zero, which means the condition clouds overlap instead of sitting apart.

**What those numbers do not support.** They say nothing about what the surviving variation is about. Feature spread and pairwise cosine only describe how different the vectors are from each other. They cannot separate gait from source video, from the person walking, from the extraction path used for the added normal clips, or from pose-detector behavior. A healthy spread is just as consistent with a model that has learned camera style as with one that has learned walking.

**What the next valid step is.** Two separate follow-ups are needed, and they answer different questions. To improve this training result, vary the replay schedule or the loss weights and read feature spread and normal-anchor retention together, because a change that protects the anchor can also flatten the features, and a change that raises spread can also increase drift. Watching one of those numbers alone hides that trade. To say anything about generalization, hold out complete source videos before any preprocessing or representation training, retrain all five stages inside each outer training fold, and open the sealed videos once. No amount of tuning measured on the current run can substitute for that.

### 3. Did replay preserve the normal representation?

Only partly. Normal-anchor cosine fell from 0.959 after Stage 1 to 0.617 after Stage 4. That is substantial drift. Balanced replay reduced the chance of forgetting, but it did not preserve the original normal geometry.

### 4. Did the training margin produce clean groups?

No. The final margin penalty remained above zero, so the margin was not fully met. More importantly, notebook 05 found a canonical-96 cosine silhouette of 0.054 and a minimum centroid distance of 0.026, between the myopathic and cerebral palsy centroids. That gap is smaller than the mean distance inside a single condition, which is 0.104, so those two classes are not cleanly separated.

### 5. Why do the training and inspection distances differ?

The Stage 4 table reports the Euclidean centroid diagnostic used during training on the active 159-sequence corpus. Notebook 05 reports cosine distances on pooled 384-dimensional target-encoder vectors for the canonical 96 sequences. They answer different questions, use different vectors, and use different corpora. Their numerical values should not be compared as if they were the same metric.

![Canonical geometry audit](figures/representation_geometry.svg)

## What every checkpoint records

- mode, stage, ordered conditions, epoch counts, and optimizer updates;
- parent experiment fingerprint and full sequence membership;
- mapping path, mapping SHA-256, and the 12-keypoint whitelist;
- preprocessing and mask configuration;
- model, EMA target, predictor, VICReg projector, and target-center states;
- optimizer update counts, while optimizer and scheduler states themselves are not saved;
- JEPA, VICReg, and group-loss settings;
- whether the curriculum is complete and label-informed.

Notebook 05 and notebook 06 reject a missing or incomplete curriculum, the wrong stage order, a different whitelist, or an artifact from the wrong run mode.

## Other runs executed under this same contract

This contract has been executed more than once. Recording that here keeps the training document honest about what else exists in the artifact tree, so that a reader who finds extra checkpoints knows where they came from.

One follow-up experiment, Idea 9 arm 2, re-ran this exact curriculum contract to test an optional fourth, label-free loss term that asks the encoder to respect an anatomical mirror, meaning the operation that reflects the body and swaps left and right landmark identities. Its endpoint is a label-free mirror residual called rho, on a scale where 0 is mirror equivariant and 4 is mirror blind.

What was re-run, and how:

1. **The contract was unchanged.** Every rung used 600 epochs and 11,400 optimizer updates, with per-stage updates `[5700, 1425, 1425, 1425, 1425]`, width 96, and depth 4, exactly as in the table above.
2. **One number changed between the two rungs.** Rung D0 set the equivariance weight to 0.0 and is the control. Rung E1 set it to 0.02 and is the treatment. The three terms of this document are present and unchanged in both.
3. **Both rungs ran once per seed for seeds 0, 1, and 2**, giving six completed curricula. That is 3 seeds run against 5 registered; the reduction is recorded as a protocol deviation in the result bundle.

Where those checkpoints live, and what they are not:

- They are written under the `idea9_arm2/` subdirectory of the artifact root, one checkpoint and one JSON per rung and seed.
- They are **separate from** `sjepa_curriculum_final_augmented.pt`. That main checkpoint carries no equivariance term at all, and every result elsewhere in this document and in the paper comes from it.
- Rung D0 is a control for that experiment only. It is a fresh run at its own seeds, not a copy of the main curriculum checkpoint, so its diagnostics are not interchangeable with the stage endpoint table above.

This subsection deliberately reports no verdict. Section 15 of `staged_details.md` gives the full step-by-step treatment, the preregistered credit rule, the guardrail that failed, the guardrail that was not evaluable, and the competing explanation; the paper states the same conclusion in short form.

## What the run can support

The completed run supports these narrow statements:

- a five-stage S-JEPA curriculum executed to completion on the recorded 159-sequence corpus;
- the final representation did not totally collapse;
- later stages caused substantial normal-anchor drift;
- canonical five-condition geometry remained weak, at cosine silhouette 0.054 with the closest centroid pair 0.026 apart;
- frozen features contain class-related structure inside the already-seen corpus.

![Current classifier readouts. These are descriptive scores inside an encoder-exposed corpus.](figures/readout_results.svg)

It does not establish generalization to an unseen video, person, camera, or clinic. The current classifier lanes use an encoder that already saw every evaluated sequence. A valid estimate needs a source-video split before all preprocessing and representation training, followed by a fresh five-stage model inside each outer training fold.

![The three evaluation lanes and the claim each one can support](figures/evidence_ladder.svg)

![Required fold-local evaluation](../images/11_nested_evaluation.svg)
