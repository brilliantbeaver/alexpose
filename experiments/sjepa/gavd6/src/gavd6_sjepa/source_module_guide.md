# Python code by responsibility

The package separates study methods from shared data and artifact handling. [Study overviews](../../docs/studies/README.md) explain the scientific questions; [code organization](../../docs/repository/code-organization.md) explains the refactor and replay boundaries.

| Responsibility | Modules to start with |
| --- | --- |
| Preserve movement during tracking repair | [motion_preservation/workflow.py](research_directions/motion_preservation/workflow.py), [pretrained_models.py](research_directions/motion_preservation/pretrained_models.py), [repair_models.py](research_directions/motion_preservation/repair_models.py), [preservation_metrics.py](research_directions/motion_preservation/preservation_metrics.py) |
| Predict future video features | [future_prediction](research_directions/future_prediction/README.md) |
| Measure learning as recording count grows | [source_scaling/cohort.py](research_directions/source_scaling/cohort.py), [learning_curve.py](research_directions/source_scaling/learning_curve.py) |
| Measure student target accessibility | [target_accessibility/cached_panel.py](research_directions/target_accessibility/cached_panel.py), [inspection.py](research_directions/target_accessibility/inspection.py), [verification.py](research_directions/target_accessibility/verification.py) |
| Known anatomical reflection and probes | [reflection_equivariance/jepa.py](research_directions/reflection_equivariance/jepa.py), [amass_training.py](research_directions/reflection_equivariance/amass_training.py), [gavd_probe.py](research_directions/reflection_equivariance/gavd_probe.py), [swap_probe.py](research_directions/reflection_equivariance/swap_probe.py) |
| Infer unknown left/right correspondence | [latent_laterality/inference.py](research_directions/latent_laterality/inference.py), [benchmark.py](research_directions/latent_laterality/benchmark.py), [training.py](research_directions/latent_laterality/training.py), [evaluation.py](research_directions/latent_laterality/evaluation.py) |
| Load, identify and convert shared data | [data_foundations](data_foundations/) |
| Write artifacts and locate experiment results | [artifact_io.py](shared_infrastructure/artifact_io.py), [result_catalog.py](shared_infrastructure/result_catalog.py), [result_migration.py](shared_infrastructure/result_migration.py) |
| Validate notebook startup | [workspace_validation/notebooks.py](workspace_validation/notebooks.py) |
| Route commands lazily | [cli.py](cli.py) |

The movement study uses whole-body AMASS motion. The historical `core11-v1` schema describes an 11-landmark lower-body representation and retains its serialized spelling. File organization does not rename schema fields, random seeds, arms, checkpoint tensors or experimental protocols.

The benchmark and GAVD-probe modules now contain their small command handlers. Larger training and inference modules remain separate because their responsibilities differ. Integrity-only target inspection also remains separate from verification that reconstructs model fits.

Older command/import adapters remain under [archive](archive/); they resolve to current implementations. Exact historical software execution instead uses the [archived source snapshot](../../scripts/archive/code_layout_20260913/README.md). These are distinct purposes.
