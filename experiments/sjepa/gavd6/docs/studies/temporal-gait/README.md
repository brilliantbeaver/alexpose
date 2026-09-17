# Temporal gait: observation-preserving JEPA and observable forecasting

This is the implementation of the [47 improvement plan](../../../../gavd5-drift/notes/47-improvement-plan.md), not a revision of the historical paper's results. **Real GAVD/HAIC experiments have not run.** No private source paths were searched, no data were downloaded, and no Slurm jobs were submitted.

Start with the [HAIC runbook](execution/haic.md) and [frozen scientific protocol](protocol/protocol.md). The [notebooks](../../../notebooks/temporal_gait/README.md) are thin, output-free teaching stages; explicit execution writes separate run artifacts. [Source code](../../../src/gavd6_sjepa/research_directions/temporal_gait), [tests](../../../tests/temporal_gait), and [Slurm launchers](../../../slurm/temporal-gait) share one validated configuration.

The current milestone implements explicit-manifest inventory, full-bout pose/PTS validation, uncapped window indexing, original-time causal preparation, an index-resized common-window control, masked-feature JEPA, future-feature JEPA, wrong-source and wrong-horizon-order controls, strong direct predictors, frozen readouts, source-group uncertainty and sealed-test mechanics. It uses small skeleton models from scratch—not pretrained foundation-model transfer.

E3 dense × intermediate-layer supervision, released RGB/video feature extraction, raw-video pose extraction, uncertainty calibration, physical-unit or clinical endpoints, and an exact historical checkpoint replay are **not implemented by this milestone**. They are not silently replaced. The plan's first E0–E2 causal/measurement comparisons take precedence; no extension should be activated merely because its notebook renders.

Evidence and decisions:

- [Historical evidence ledger](development/evidence-ledger.md): what is retained, reproducible, inferred and missing.
- [Primary-source refresh, 2026-09-15](development/literature-refresh.md): methods, sampling, negative evidence and transfer limitations.
- [Decision and dependency board](development/decisions.md): scoped ownership, changed interfaces and implementation deviations.
- [Adversarial review dispositions](development/review-dispositions.md): supported defects, fixes and outstanding limits.
- [Verification and result status](results/README.md): software checks versus unrun empirical claims.

Historical evidence remains unchanged: 625 sequences, 93 sources, five folds × five seeds, 125 encoders with 1,200 updates each; expanded initialized R²≈0.222544 exceeds all trained teachers≈0.100777–0.114209. Missing historical raw predictions/checkpoints prevent independent inference/bootstrap replay. The separate latent-laterality uniform-control stop remains a valid stop, not a failed job to restart.
