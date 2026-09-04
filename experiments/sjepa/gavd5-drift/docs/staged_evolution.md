# How the staged S-JEPA evidence changed

This document records the change from the retired augmented-normal experiment to the current GAVD-only run. It is an evidence history, not an invitation to combine results across runs.

## Current versus retired experiment

|Item|Retired experiment|Current experiment|
|---|---|---|
|Optional added-normal data|Enabled|Disabled|
|Modeled cohort|159 sequences / 35 videos|626 sequences / 93 videos|
|Final checkpoint|`sjepa_curriculum_final_augmented.pt`|`sjepa_curriculum_final.pt`|
|Final fingerprint prefix|`d0acc262...`|`7d13841a...`|
|Anchor after Stages 1–4|0.954 / 0.839 / 0.707 / 0.594|0.700 / 0.502 / 0.396 / 0.297|
|All-row five-class probe|Old 96-row result|626-row result: 0.920 accuracy, 0.899 macro-F1|
|Grouped Lane C|Available for old encoder|Not yet regenerated for current encoder|
|AnchorGuard and surprise|Old or mixed lineage|Excluded|

The numbers in one column cannot validate a model in the other column. A checkpoint fingerprint, cohort identity, sequence list, and preprocessing contract must agree before results are combined.

## What did not change

The core S-JEPA design remains a masked latent-prediction model with a view encoder, EMA target encoder, predictor, VICReg regularizer, and a later label-aware group term. One model continues through normal, Parkinson's, stroke, myopathic, and cerebral-palsy stages with replay.

The downstream 384-dimensional vector is still a concatenation of four 96-dimensional summaries. It is not the encoder width.

## What changed scientifically

The larger current run shows a much steeper raw anchor curve and stronger in-corpus label geometry. Those two changes do not establish better or worse generalization. The geometry is label-informed, and the anchor is a raw-coordinate measure over encoder-exposed normal rows.

The current anchor is

$$
a_t=\frac{1}{|N|}\sum_{x\in N}\cos(z_t(x),z_0(x)),
$$

the average of matched per-sequence cosines. It is not a cosine between cohort means.

The current all-row classifier is also larger than the legacy `all_96` name suggests. It uses 626 rows. Every one of its 188 test rows was seen by the encoder, and 64 source videos cross its classifier split. It is an in-corpus readout.

## Why some apparently recent results were removed

The workspace retained cached outputs from different experiment generations. The audit found:

- notebook 08 loading the augmented Stage-0 checkpoint for a current ablation;
- an AnchorGuard checkpoint without a complete current parent/cohort record;
- a grouped-classifier CSV that still names the 159-row encoder; and
- notebook 09 loading an old augmented checkpoint while evaluating current rows.

These are lineage failures, not small labeling errors. Their outputs are excluded until rerun end to end.

## Current evidence ladder

1. **Reproduced:** the seed-42 checkpoint lineage and raw anchor curve.
2. **Descriptive:** training stability, in-corpus geometry, frozen-feature classifiers, temporal readouts, and laterality probes.
3. **Not established:** functional forgetting, the cause of drift, a repair effect, unseen-source performance, and clinical utility.

## Rules for future updates

1. Put the checkpoint fingerprint and SHA-256 in every derived-result contract.
2. Record the exact sequence and source-video IDs.
3. Reject cached results when cohort, parent checkpoint, masking rule, or preprocessing version differs.
4. Never use `_augmented` results as evidence for the current run.
5. Split sources before any fitted preprocessing for a generalization claim.
6. Keep raw coordinate drift, functional retention, and downstream performance as separate endpoints.

The current numerical record is in [staged_details.md](staged_details.md). The model implementation is explained in [tutorials/sjepa_model_internals.md](tutorials/sjepa_model_internals.md).
