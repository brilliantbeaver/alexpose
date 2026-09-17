# Research commands and scripts

Start from the [study index](../docs/studies/README.md). Numerical methods live in `src/gavd6_sjepa`; scripts own notebook generation, experiment orchestration and artifact inspection.

| Study or responsibility | Scripts | Entry point |
| --- | --- | --- |
| Synthetic training selection | [synthetic_training](research_directions/synthetic_training/) | `build_notebooks.py`, `execute_notebook.py`; follow the [teaching experiment guide](../notebooks/synthetic_training/README.md). |
| Motion preservation | [motion_preservation](research_directions/motion_preservation/) | `build_notebooks.py`, `execute_notebook.py`; follow the [current experiment guide](../notebooks/motion_preservation/README.md). |
| Future-feature prediction gate | [future_prediction](research_directions/future_prediction/README.md) | Build, execute and verify notebooks; inspect fitted models and residual-head failures. |
| Source scaling | [source_scaling](research_directions/source_scaling/README.md) | Calibration, initialization, stage execution and read-only notebook inspection. |
| Student accessibility | [target_accessibility](research_directions/target_accessibility/) | Cached-panel execution and notebook, evidence, figure and manuscript generation. |
| Reflection equivariance | [reflection_equivariance](research_directions/reflection_equivariance/) | Separate encoder, GAVD, AMASS, replication and extension lessons. |
| Signed laterality | [signed_laterality](research_directions/signed_laterality/) | Probe and extension lessons. |
| Result administration | [workspace_management](workspace_management/) | Locate, inventory and verify installed run bundles. |
| Historical command adapters | [archive](archive/) | Retained command entry points backed by the current package. |

`uv run gavd6 --help` lists shared-data and historical study commands. The motion-preservation workflow has a dedicated notebook executor and Slurm launcher. The three future-prediction diagnostic scripts perform different analyses: saved-fit inspection, synthetic residual-head probes and forensic failure evaluation.

The September 2026 code refactor changes software identity. Existing sealed runs must reject incompatible code for resume or fitting. The pre-refactor source and environment contract are preserved in [the replay snapshot](../scripts/archive/code_layout_20260913/README.md); historical run receipts are not rewritten. See [notebook ownership](research_directions/research_notebook_builder_guide.md) and [layout conventions](../docs/repository/layout.md).
