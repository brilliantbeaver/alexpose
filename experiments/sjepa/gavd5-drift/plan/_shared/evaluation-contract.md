# Shared evaluation contract

Every proposal in `plan/` follows this contract. It fixes the population, source-level split, fitting boundary, controls, and claim language before model results are inspected.

## Data gates

Never use one unqualified "dataset size." Report sequences, unique source videos, and annotated manifest frames at the relevant gate.

|Gate|Sequences|Source videos|Annotated frames|Status|
|---|---:|---:|---:|---|
|Raw annotations|666|103|140,641|All five GAVD manifest folders|
|Metadata-public|657|100|137,690|Live platform metadata on local date 2026-09-04|
|Decoded-span candidate upper bound|656|99|137,232|Theoretical maximum if the one retryable acquisition failure is recovered|
|Decoded-frame eligible, current audit|655|98|135,804|Last annotated frame decodes; one terminal-short and one retryable acquisition failure|
|Pose-QC eligible, fold 0|639|97|134,259|Neurologic-joint observed fraction >= the predeclared 0.50 threshold|

The metadata-public population is normal 291 sequences/32 sources/41,340 annotated frames, Parkinson's 47/11/10,426, stroke 75/18/32,930, myopathic 184/29/33,992, and cerebral palsy 60/10/19,002. At the dated metadata check, `sf5X4YYkWUA` and `YjRoLtP1di0` were private and `yULxvDc9e8c` was unavailable.

"Metadata-public" means that the platform returned public metadata without authentication. It does not establish download success, valid FPS, sufficient decoded frames, or acceptable pose coverage. The current decode audit measures 655 eligible sequences from 98 sources: `n93bgWhLZk4` is terminal-short and `hGNKzkCF4J8` is retryable. The 656/99 row remains a **candidate upper bound** if the latter source is recovered. Notebook 02 validated all 655 locked pose caches, then retained 639 sequences from 97 sources at the separate pose-QC gate. Failures are recorded against the frozen split; they do not trigger a convenient re-split.

## Frozen protocol-v2 split

The independent unit is the canonical YouTube source-video ID. Protocol v2 deterministically assigns the 100 metadata-public sources to five outer folds. In every fold:

- 60 sources are training sources and may receive gradient updates;
- 20 sources are validation sources and may select preprocessing, hyperparameters, stopping, and checkpoints; and
- 20 sources are test sources and remain sealed until the pipeline is frozen.

Each source is outer-test exactly once, and every sequence inherits its source's role. Folder labels balance the folds but never allow a source to cross roles.

- Input-manifest SHA-256: `7fd559e5105b11011a3e5c194b7ccc29729c56491c424745834df39884123b5a`
- Split SHA-256: `ff3518b87b1d1fa7d95efb1aea1711773137a21699967cb8015edb8d845ccbe1`

These are deterministic hashes from the 2026-09-04 metadata-public snapshot and protocol-v2 module. Fold-0 pose-QC artifacts are current; the corresponding fold-local checkpoints and model-result artifacts are pending.

## Fit boundary

For each outer fold, training and validation sources are the only sources allowed to influence:

- decode and pose-QC policy decisions that are data-dependent;
- imputation, normalization, scaling, augmentation, or feature selection;
- the complete encoder/predictor curriculum and checkpoint selection;
- all readout fitting, threshold choice, calibration, and model selection; and
- exploratory decisions promoted into the confirmatory analysis.

Open test sources once after the analysis is frozen. A source-grouped readout over embeddings from an encoder trained on all sources is encoder-transductive and is not held-out generalization.

## Primary and supervised-ablation models

The primary representation objective is label-free JEPA plus VICReg. Any condition-label group objective is a supervised ablation and is trained, selected, and reported separately. Label readability in that ablation cannot be described as structure independently discovered by self-supervision.

## Required controls and reporting

Where relevant, compare against raw pose, an untrained encoder, pose missingness/coverage, continued-normal training with matched updates, joint training, and the label-aware ablation. Test curriculum-order sensitivity and use multiple training seeds.

Report sequence-level and equal-source-weighted endpoints, per-source predictions, fold and seed dispersion, source-cluster uncertainty, every exclusion, and all deviations from the frozen registry. Treat folds and checkpoints from the same training run as correlated. Predeclare the primary endpoint and report effect sizes with uncertainty rather than selecting the friendliest metric after inspection.

## Evidence and claim hygiene

All model metrics from the earlier 96/18, 159/35, 642/94, and 626/93 cohorts or encoder-exposed probe lanes are archived. They are not current baselines and are not comparable with protocol-v2 results until regenerated with fold-local preprocessing, representation learning, selection, and readout fitting.

- Say "source-held-out," not "subject-held-out": GAVD does not supply a verified person identifier.
- A folder label is a dataset annotation, not a diagnosis produced or adjudicated by this project.
- Prediction error indicates model surprise, not danger, pathology, or physical impossibility.
- Compact embeddings are not private by default; gait trajectories can remain identifying.
- Unusual but real gait is not the same as invalid motion.
- No result establishes new-patient, cross-clinic, diagnostic, causal, surveillance, or deployment performance without an appropriate external study.

## How proposals reference this contract

Each `plan/<NN>-<slug>/README.md` should cite `plan/_shared/evaluation-contract.md` and reuse the frozen protocol-v2 registry rather than construct its own favorable split. If a proposal needs a different population or grouping unit, it must version that contract, justify the change before results are inspected, and report the new manifest and split hashes.
