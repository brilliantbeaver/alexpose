# Detailed audit of the current staged S-JEPA run

This report describes only the current GAVD-only experiment. The optional added-normal dataset is disabled. Historical files with `_augmented` in their names do not supply results in this report.

## 1. Evidence boundary

The local raw inventory has 666 sequences from 103 source videos. Availability checks retain 642 sequences from 94 videos. Requiring at least 0.50 observed coverage over the 12 target landmarks removes 16 rows, leaving 626 sequences from 93 videos.

|Folder annotation|Available rows / videos|Modeled rows / videos|
|---|---:|---:|
|Normal|284 / 30|270 / 29|
|Parkinson's|42 / 9|41 / 9|
|Stroke|75 / 18|74 / 18|
|Myopathic|183 / 28|183 / 28|
|Cerebral palsy|58 / 9|58 / 9|
|**Total**|**642 / 94**|**626 / 93**|

These are dataset folder annotations, not diagnoses made by this project. Source video is the strongest available grouping key; it is not a person identifier.

Across the 642 availability-filtered pose caches, 546 carry extraction label `gavd5`, 95 carry `gavd3`, and one carries `gavd4`. After coverage filtering, the 626 modeled rows contain 530, 95, and 1, respectively. The recorded pose-model hash agrees, but extraction provenance can still be a shortcut. Any generalization analysis should stratify or control for it.

## 2. Training contract

One model continues across five stages. The weights, target encoder, target center, and VICReg projector continue. AdamW state and the learning-rate schedule restart at each stage.

|Stage|New group|Active rows|Active videos|Epochs|Updates|
|---:|---|---:|---:|---:|---:|
|0|Normal|270|29|300|20,400|
|1|Parkinson's|311|38|75|5,100|
|2|Stroke|385|56|75|5,100|
|3|Myopathic|568|84|75|5,100|
|4|Cerebral palsy|626|93|75|5,100|
|**Total**||||**600**|**40,800**|

Balanced replay contributes the same number of rows from each active condition to an optimizer batch. It does not make the stored dataset balanced.

AdamW uses betas (0.9, 0.95) and weight decay 0.05. The learning rate is $10^{-3}$ in Stage 0 and $3\times10^{-4}$ in Stages 1--4. Each batch contains four rows from every active condition. The saved run reports seed 42 and the MPS backend. The exact hardware model and full deterministic-computation settings are not recorded, so runtime and exact numerical replication across devices remain documentation gaps.

The model uses 64 resized frames, 33 landmarks, four-frame temporal patches, and 96-dimensional tokens. Only shoulders, hips, knees, ankles, heels, and foot tips can be prediction targets. Missing and low-visibility tokens cannot become targets.

The objective is

$$
\mathcal L = \mathcal L_{\mathrm{JEPA}}
+0.05\mathcal L_{\mathrm{VICReg}}
+0.25\mathcal L_{\mathrm{group}}.
$$

The group term is disabled in Stage 0. Later it uses the folder labels to encourage within-group compactness and condition-centroid separation. The later curriculum is therefore label-informed, not fully self-supervised.

## 3. Current artifact identity

```text
checkpoint:  sjepa_curriculum_final.pt
fingerprint: 7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2
file SHA256: 64008d77689cefa4beb51a0dcf5ed6cae743454134c163e9087f66510af4e7ad
seed:        42
```

The final checkpoint contract contains all 626 sequence IDs, curriculum order, masking rule, loss settings, and checkpoint lineage. Source-video IDs are audited separately in the pose-coverage and split-manifest artifacts.

## 4. Training diagnostics

|Stage|JEPA|VICReg|Feature std.|Mean pair cosine|Min. normalized Euclidean centroid distance|Normal anchor|
|---:|---:|---:|---:|---:|---:|---:|
|0|0.6998|16.7905|0.2907|0.2833|not defined|1.0000|
|1|0.7430|12.9099|0.2188|0.7264|0.8831|0.7002|
|2|0.4981|10.7752|0.2295|0.6683|0.7041|0.5021|
|3|0.4509|9.3316|0.2263|0.5688|0.6160|0.3962|
|4|0.3699|8.2374|0.2291|0.4404|0.5344|0.2966|

`VICReg` is the unweighted inner value. `group` in the epoch log is only the separation component, not the full group term. `std` is a post-epoch teacher-feature diagnostic, not a loss.

Finite loss and nonzero spread argue against numerical failure and total constant collapse. They do not establish useful or clinical features.

## 5. Exact definition of the normal anchor

For each row, the stage audit first forms a 96-D validity-weighted mean of target-encoder tokens at the 12 authorized landmarks. For every one of the same 270 normal rows, it compares this current summary with that row's Stage-0 summary. It then averages the 270 cosine similarities:

$$
a_t=\frac{1}{270}\sum_{x\in N}\cos(z_t(x),z_0(x)).
$$

This is not the cosine between two cohort centroids. Reloading the five checkpoints reproduces the saved curve with a maximum absolute gap of `4.51e-7`.

The mean is sequence-weighted. Normal videos contribute 1 to 60 rows, with a median of 4. The top two videos supply 105 of 270 rows (38.9%), and the top three supply 137 (50.7%). The paper therefore needs per-video anchor values, an equal-video-weighted curve, and source-cluster uncertainty before making a population-level claim.

The decline proves raw coordinate change on encoder-exposed normal rows. It does not prove forgetting because a shared basis rotation can change raw cosine without removing information. The current experiment also lacks a continued-normal control and has only one seed.

## 6. Frozen-feature geometry

The final 384-dimensional summaries over the 626 training rows have:

|Metric|Value|
|---|---:|
|Cosine silhouette|0.361717|
|Minimum between-centroid cosine distance|0.086261|
|Mean between-centroid cosine distance|0.621765|
|Mean within-condition cosine distance|0.078339|

These are in-corpus, label-informed measurements. The group labels helped train the encoder, so the geometry is descriptive rather than independent validation.

The stage-end minimum centroid distance and the frozen-feature minimum distance are different statistics. The first is Euclidean distance between normalized centroids of 96-D authorized token means. The second is cosine distance between centroids of 384-D vectors that concatenate global mean, global standard deviation, authorized mean, and authorized standard deviation. Do not compare them as if they were the same scale.

## 7. Downstream probes

|Probe|Accuracy|Balanced accuracy|Macro-F1|
|---|---:|---:|---:|
|All 626 rows, sequence split|0.9202|0.9001|0.8985|
|Exact historical 47/21 split|0.8571|0.8800|0.8607|
|Missingness only, all-row split|0.4415|0.4269|0.3547|
|Missingness only, exact split|0.3333|0.3638|0.3361|

The all-row split has 438 classifier-training rows and 188 test rows. All 188 test rows were exposed to the encoder. Sixty-four source-video IDs cross the classifier split, and 181 of 188 test sequences come from those shared videos. In the exact split, all 21 test rows were encoder-exposed and all nine test videos overlap classifier training.

These results show that labels are readable in the fitted corpus. They do not estimate new-video, new-person, or clinical performance. The file name `all_96_stratified_video_confounded` is a legacy label; the current table contains 626 rows.

The saved Lane C file is not usable: it names the old 159-row augmented encoder. No grouped Lane C number is current evidence.

## 8. Secondary probes

The temporal-readout study compares the deployed pooling with signed moments, time-bin pooling, and learned attention. The pre-specified temporal-moment lane does not clear its rule of at least 10% lower error and improvement in at least 75% of source videos. Target counts vary: 626 for phase targets, 578 for cadence and stride time, and 539 for energy ratio. Cadence and stride-time decoding are weak, which is compatible with omitted native duration but does not identify the cause.

The signed-laterality study gives $R^2=0.241$ for learned tokens and 0.190 for an untrained encoder. The raw score near 1.0 is a construction sanity ceiling because the target is derived from those raw signed-excursion features. Sign consistency is 55.3%, and the mirror slope is -0.627 rather than the required range of -1.25 to -0.80. It uses 642 rows, 94 videos, and five grouped folds, including the 16 rows excluded from primary training.

The executed temporal notebook contains 3,369 numerical `LinAlgWarning` records, and the laterality notebook contains 151. Neither probe has repeated grouped-split uncertainty. Rerun them with a stable solver, regularization sensitivity checks, and repeated grouped splits before interpreting small differences.

## 9. Invalid or historical result families

Do not use these results in the current paper:

1. files with `_augmented` in the name;
2. the notebook 08 margin ablation, which loads the old Stage-0 checkpoint and fails to reproduce current Stage 1;
3. the cached AnchorGuard result, which lacks complete current parent and cohort lineage;
4. Lane C, which still identifies the 159-row encoder; and
5. notebook 09 predictive-surprise results, which combine the old checkpoint with current rows.

The AnchorGuard non-inferiority gate must also change from `abs(delta) <= 0.05` to the one-sided rule `delta >= -0.05` before a rerun.

## 10. Claim ledger

|Claim|Current status|What is missing|
|---|---|---|
|The raw normal coordinates change|Supported for seed 42|Multiple seeds for repeatability|
|The model forgets normal gait|Not supported|Alignment and held-out normal function|
|The group term causes the drift|Not supported|Valid matched ablation|
|AnchorGuard repairs retention|Not supported|Current-lineage rerun and corrected gates|
|Labels are readable in-corpus|Supported descriptively|None if kept transductive|
|The encoder generalizes to new sources|Not supported|Fold-local full retraining|
|The model has clinical value|Not supported|Clinical and external validation design|

## 11. Data-use boundary

GAVD's official distribution provides annotations and public video URLs rather than raw video files. Users retrieve media independently and are responsible for YouTube terms, institutional ethics review, and applicable copyright, privacy, and data-protection rules. This analysis uses derived pose sequences and does not infer identity. A public artifact should not redistribute raw video or identity-bearing frames. The workspace does not currently contain a recorded institutional ethics determination or completed data-use review; both must be resolved before submission.

## 12. Required next experiments

The minimum high-priority set is:

1. three to five full seeds;
2. per-video and equal-video-weighted anchors with source-cluster uncertainty;
3. orthogonal-Procrustes alignment plus linear CKA or SVCCA;
4. same-update continued-normal and joint-training controls;
5. held-out normal JEPA loss or ranking at every checkpoint;
6. at least one alternate curriculum order; and
7. preprocessing, encoder training, and readout fitting inside outer source-video folds.

See [the workshop readiness guide](../neurips-brain-body/docs/neurips-brain-body.md) for the submission decision and [the paper draft](../neurips-brain-body/docs/bbfm2026_paper_draft.md) for the compact argument.
