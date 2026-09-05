# Evidence ledger for the BrainBodyFM pipeline

This ledger separates verified corpus/protocol facts from pending model evidence. A saved output describes only the population, split, code, and lineage that produced it.

## Current data evidence

|Gate|Sequences|Source videos|Annotated frames|Evidence status|
|---|---:|---:|---:|---|
|Raw annotations|666|103|140,641|Counted from the five manifest folders|
|Metadata-public|657|100|137,690|Dated live-metadata snapshot, local date 2026-09-04|
|Decoded-span candidate upper bound|656|99|137,232|Theoretical maximum if the one retryable acquisition failure is recovered|
|Decoded-frame eligible, current audit|655|98|135,804|Measured after all public sources were attempted|
|Pose-QC eligible, fold 0|639|97|134,259|Measured at neurologic-joint observed fraction >= 0.50|

Metadata-public counts by manifest condition:

|Condition|Sequences|Source videos|Annotated frames|
|---|---:|---:|---:|
|Normal|291|32|41,340|
|Parkinson's|47|11|10,426|
|Stroke|75|18|32,930|
|Myopathic|184|29|33,992|
|Cerebral palsy|60|10|19,002|
|**Total**|**657**|**100**|**137,690**|

At the dated check, `sf5X4YYkWUA` and `YjRoLtP1di0` were private, while `yULxvDc9e8c` was unavailable. This accounts for nine annotated sequences across three sources. `n93bgWhLZk4` downloaded but had only 228 frames for annotations through frame 458 and is terminal attrition. `hGNKzkCF4J8` remained a retryable acquisition failure after four bounded client/format strategies. The measured decoded-frame cohort is 655 sequences / 98 sources / 135,804 annotated frames; 656 / 99 / 137,232 remains only the candidate upper bound if the retryable source is recovered. Notebook 02's fold-0 audit found all 655 locked caches valid and retained 639 sequences / 97 sources / 134,259 frames after pose QC.

## Current independence contract

Protocol v2 freezes five source-grouped outer folds from the 100 metadata-public sources. Each fold contains 60 training, 20 validation, and 20 test sources. Every source is test exactly once and no source crosses roles within a fold.

|Artifact identity|SHA-256|
|---|---|
|Metadata-public input manifest|`7fd559e5105b11011a3e5c194b7ccc29729c56491c424745834df39884123b5a`|
|Protocol-v2 split registry|`ff3518b87b1d1fa7d95efb1aea1711773137a21699967cb8015edb8d845ccbe1`|

These are deterministic hashes from the dated snapshot and current split module. The fold-0 pose-QC bundle has been generated; fold-local checkpoint and model-result hashes remain pending.

## Model and training facts

- A 64-frame sequence is divided into 16 four-frame positions.
- Thirty-three joints across 16 positions produce 528 possible joint-time tokens.
- Only the 12 shoulder, hip, knee, ankle, heel, and foot-index landmarks may become prediction targets.
- The primary representation objective is label-free JEPA plus VICReg.
- The condition-label group term is a supervised ablation, not part of the primary self-supervised claim.
- The full encoder and predictor, preprocessing, selection, and readout must be refitted independently inside every outer fold and seed.

These are implementation or protocol facts, not evidence that the model generalizes.

## Model-result status

No protocol-v2 model metric is currently supported by a fold-local run bundle. Earlier anchor, geometry, classifier, temporal, laterality, forecasting, and repair results are archived because they came from older cohorts, mixed lineages, or encoders exposed to evaluation inputs. They are not comparable with future protocol-v2 results and must not be reused as baselines without a complete fold-local rerun.

Required result artifacts include per-fold manifests, decoded-media and pose-QC attrition, parent and checkpoint hashes, seeds, configurations, per-source predictions, uncertainty inputs, and a claim ledger mapping every reported value to its file.

## Evidence levels

|Level|Currently supported statement|
|---|---|
|Direct corpus fact|The raw and dated metadata-public counts above|
|Direct protocol fact|The deterministic five-fold 60/20/20 source-role registry and hashes above|
|Measured decode cohort|655 sequences from 98 sources pass the current container/FPS/last-annotated-frame gate|
|Candidate upper bound|At most 656 sequences from 99 sources if the retryable acquisition failure is recovered|
|Pending run fact|Pose-QC cohort, training stability, retention, and readout values|
|Unsupported|Unseen-source model performance before fold-local rerun|
|Unsupported|Unseen-person, cross-clinic, causal, diagnostic, surveillance, or deployment claims|

## Primary literature anchors

- Abdelfattah and Alahi, [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4755_ECCV_2024_paper.php), ECCV 2024.
- Assran et al., [I-JEPA](https://arxiv.org/abs/2301.08243), CVPR 2023.
- Bardes et al., [Revisiting Feature Prediction for Learning Visual Representations from Video](https://arxiv.org/abs/2404.08471), 2024.
- Bardes, Ponce, and LeCun, [VICReg](https://arxiv.org/abs/2105.04906), ICLR 2022.
- Ranjan et al., [GAVD](https://arxiv.org/abs/2407.04190), IEEE Access 2025.
- Kapoor and Narayanan, [Leakage and the Reproducibility Crisis in Machine-Learning-Based Science](https://arxiv.org/abs/2207.07048), *Patterns* 2023.
- Varoquaux, [Cross-validation failure: Small sample sizes lead to large error bars](https://pubmed.ncbi.nlm.nih.gov/28655633/), *NeuroImage* 2018.
