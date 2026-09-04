# Normal-first skeleton JEPA for gait: current technical summary

## Summary

We trained one project-specific, S-JEPA-inspired skeleton model first on normal-labeled gait and then on four condition-labeled GAVD groups. It is not a reproduction of the published S-JEPA: it uses a fixed 12-landmark target whitelist, VICReg, and a later label-aware group term. The current run does not use the optional added-normal dataset. After availability and pose-quality checks, it contains 626 sequences from 93 source videos.

Across the 270 matched normal rows, the mean cosine between each later representation and its own Stage-0 representation falls from 0.700 after Stage 1 to 0.297 after Stage 4. Reloading the saved seed-42 checkpoints reproduces this raw-coordinate curve within `4.51e-7`. It is not yet evidence of functional forgetting.

The final representation has nonzero spread and label-informed in-corpus structure. An all-row frozen-feature classifier reaches 0.899 macro-F1, but every classifier test row was used by the encoder and 64 source videos cross the classifier split. The score is descriptive, not an unseen-source result.

## Data and curriculum

|Annotation|Rows|Videos|
|---|---:|---:|
|Normal|270|29|
|Parkinson's|41|9|
|Stroke|74|18|
|Myopathic|183|28|
|Cerebral palsy|58|9|
|**Total**|**626**|**93**|

Stage 0 uses 300 epochs on normal rows. Each later stage adds one group, replays all active groups in balanced optimizer batches, and runs for 75 epochs. The final run has 600 epochs, 40,800 optimizer updates, and one seed.

Weights, the EMA target encoder, target center, and VICReg projector continue across stages; the AdamW optimizer restarts. AdamW uses betas (0.9, 0.95), weight decay 0.05, learning rate $10^{-3}$ at Stage 0, and $3\times10^{-4}$ later. A batch draws four rows per active group. The saved run reports MPS, but the exact hardware and full determinism settings are not recorded.

The model predicts teacher features at masked gait-landmark tokens. VICReg discourages collapse. A label-aware group loss is off during Stage 0 and on later. Results after Stage 0 are therefore not purely self-supervised.

## Main results

|Stage|Active rows|Feature std.|Min. normalized Euclidean centroid distance|Normal anchor|
|---:|---:|---:|---:|---:|
|0|270|0.291|not defined|1.000|
|1|311|0.219|0.883|0.700|
|2|385|0.229|0.704|0.502|
|3|568|0.226|0.616|0.396|
|4|626|0.229|0.534|0.297|

Checkpoint reload reproduces the anchor curve to within `4.51e-7`. The stage diagnostics and anchor use 96-D authorized target-token means; the centroid column is normalized Euclidean distance. A separate frozen 384-D vector concatenates global and authorized means and standard deviations. It has cosine silhouette 0.362, minimum between-centroid cosine distance 0.086, and mean within-condition cosine distance 0.078. These values are computed on the encoder's training corpus with labels that also shaped the encoder.

|Five-class readout|Accuracy|Balanced accuracy|Macro-F1|
|---|---:|---:|---:|
|All-row sequence split|0.920|0.900|0.899|
|Exact historical 47/21 split|0.857|0.880|0.861|
|Missingness-only all-row control|0.441|0.427|0.355|

## Interpretation

The supported result is raw normal-coordinate drift. A global rotation can lower raw cosine while preserving useful relationships, so the paper should not call the curve catastrophic forgetting. A forgetting claim needs alignment-aware comparisons and loss or ranking on held-out normal data.

The anchor is sequence-weighted. The top two normal source videos contribute 105 of 270 rows, and the top three contribute 137. Population-level language requires per-video results, equal-video weighting, and source-cluster uncertainty.

The classifiers show that the fitted representation contains information associated with dataset folder labels. They do not show new-video, new-person, or clinical performance. Camera, source, pose missingness, and mixed extraction-version history may all contribute.

## Required controls

Before making a strong claim, run three to five seeds, equal-video-weighted anchors with source-cluster uncertainty, Procrustes plus CKA or SVCCA, matched continued-normal and joint-training controls, held-out normal-function tests, and at least one order control. Any unseen-source performance claim also requires full encoder retraining inside each outer source-video split.

Historical AnchorGuard, margin-ablation, grouped Lane C, and predictive-surprise outputs are excluded because their checkpoint or cohort lineage does not match the current experiment.

GAVD provides annotations and public video URLs rather than raw videos. Before public release, the authors must record the institutional ethics determination, complete the data-use review, and confirm that no raw video or identity-bearing frame is redistributed.

For full details, see [staged_details.md](staged_details.md). For the workshop argument and readiness assessment, see [bbfm2026_paper_draft.md](../neurips-brain-body/docs/bbfm2026_paper_draft.md) and [neurips-brain-body.md](../neurips-brain-body/docs/neurips-brain-body.md).
