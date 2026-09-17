# Cluster execution by study

Choose a study launcher; each Sbatch file retains its resource request and operation-specific arguments. Dependency order is defined by the submission scripts.

| Study | Entry point | Execution boundary |
| --- | --- | --- |
| Synthetic training selection | [synthetic-training](synthetic-training/README.md) | Source student arrays and frozen selection; separate GAVD deployment, crossover, and human-reference evaluation. |
| Motion preservation | [motion-preservation](motion-preservation/README.md) | Pilot inventory through evaluation; final family and GAVD require explicit selections. |
| Future-feature prediction gate | [future-prediction](future-prediction/README.md) | Versioned direct-prediction gates with separate CLI, notebook and cached-repair submission paths. |
| Source scaling | [source-scaling](source-scaling/README.md) | Calibration, source reservation, fitting and inspection of an identified parent cache. |
| Latent laterality | [latent-laterality](latent-laterality/README.md) | Paired AMASS, source-transfer and legacy jobs; confirmation remains stopped. |
| Reflection controls | [reflection-equivariance](reflection-equivariance/README.md) | AMASS training and the historical frozen swap probe. |
| Shared data | [shared-data](shared-data/README.md) | GAVD acquisition and AMASS landmark conversion. |

Python orchestration is in [scripts](../scripts/navigation_guide.md). Source-scaling `submit.sh` performs initialization and appends inspection; `submit-stages.sh` submits processing/fitting stages for an already-frozen run. They intentionally have different input contracts.

The code refactor changes software identity. Preserve historical run receipts and use the [pre-refactor replay snapshot](../scripts/archive/code_layout_20260913/README.md) when exact old software is required. These launchers do not authorize opening reserved evaluation data or resuming a stopped scientific comparison.
