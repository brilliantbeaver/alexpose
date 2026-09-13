# Source-scaling orchestration

These scripts support the [source-scaling study](../../../docs/studies/future-feature-prediction/scaling/README.md). Numerical cohort, fitting and reporting methods live in the [source package](../../../src/gavd6_sjepa/research_directions/source_scaling/).

| Operation | Owner |
| --- | --- |
| Calibrate and initialize | [calibrate.py](calibrate.py), [initialize.py](initialize.py): synthetic calibration, exposure checks, source reservation and initialization receipts. |
| Execute stages | [run.py](run.py) invokes the package CLI; [run_stage.py](run_stage.py) retains per-attempt logs and operational status. |
| Build and inspect the notebook | [build_notebook.py](build_notebook.py), [inspect_notebooks.py](inspect_notebooks.py): output-free source generation and read-only evidence inspection. |
| Known exposure inventory | [known-exposure.csv](known-exposure.csv), [provenance](known-exposure.json): original checksummed source reservations, unchanged by the move. |

Use the [Slurm guide](../../../slurm/source-scaling/README.md) for the public submission workflow. Software and launcher receipts distinguish this refactored implementation from historical runs. An initialization receipt or successful process exit is not a scientific advance.
