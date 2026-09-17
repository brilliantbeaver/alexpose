# Temporal gait notebooks

Nine output-free teaching notebooks explain the full-bout timing, information audit, small JEPA comparisons and locked evaluation. Numerical work lives in `src/gavd6_sjepa/research_directions/temporal_gait`; notebook cells call its workflow. See the [HAIC execution guide](../../docs/studies/temporal-gait/execution/haic.md).

| Notebook | Stage | Purpose |
| --- | --- | --- |
| [00](00_inventory_and_split.ipynb) | `inventory` | Explicit manifests, exposure, independent groups and roles |
| [01](01_full_bout_timing_and_windows.ipynb) | `prepare` | Complete-bout PTS, pose alignment and window coverage |
| [02](02_information_and_baseline_audit.ipynb) | `audit` | Measurement support and past-only baselines |
| [03](03_time_faithful_masked_jepa.ipynb) | `masked` | One `masked_index` or `masked` arm/seed task |
| [04](04_causal_future_jepa.ipynb) | `future` | One future or mismatched-target arm/seed task |
| [05](05_optional_dense_and_video_transfer.ipynb) | `extensions` / `cache-video` | Explain currently gated E3/RGB branches |
| [06](06_development_comparison.ipynb) | `evaluate` | Complete expected grid, paired results and expansion decision |
| [07](07_locked_calibration_and_test.ipynb) | `calibrate` / `test` | Freeze choices; explicit separate test access |
| [08](08_aggregate_and_claim_audit.ipynb) | `aggregate` | Retained evidence and claim limits |

Open each notebook with the environment containing the study dependencies. Set `GAVD6_ROOT`, `TG_CONFIG` and an explicit `TG_RUN_ROOT` consistent with that configuration. The default is **plan inspection**: no workflow stage runs unless `TG_EXECUTE=1`. Synthetic fixtures require a configuration whose mode is explicitly `synthetic`; missing real assets never activate them.

Use `TG_PHASE=pilot`, `develop` or `confirm` and, for training, one `TG_TASK_ID` from the immutable task plan. Notebook 07 defaults to `calibrate`; `TG_STAGE=test` is a separate choice and remains subject to the package's saved-lock checks. Notebook 05 reports unsupported extensions honestly; it is not a trained dense/video comparator.

The runner creates a temporary kernel using its own Python interpreter, saves after cells and on failures, and refuses to overwrite an executed copy. Output metadata labels mode and separates `plan_only`, `synthetic_software`, and `real_execution` from completion/failure status. Default outputs are `notebook_runs/run-NN-PHASE-STAGE[-task-ID]/`. Change `TG_NOTEBOOK_OUTPUT_DIR` or runner `--output-dir` for a new attempt. Do not edit source notebooks to insert new results.

Regenerate canonical notebooks after editing their builder:

```bash
.venv/bin/python scripts/research_directions/temporal_gait/build_notebooks.py
```

Generated cells have deterministic IDs and no stored outputs. Synthetic execution verifies software mechanics; real GAVD results require the separately configured HAIC run and complete saved predictions.

For a complete bounded CPU check after freezing code, run `.venv/bin/python scripts/research_directions/temporal_gait/verify_software.py --run-root /ABSOLUTE/NEW/LOCAL/OUTPUT` from `gavd6/`. It executes the stage sequence in fresh kernels and retains a `verification.json` report. It also verifies real-mode plan inspection with nonexistent input manifests; it does not open real data or submit Slurm jobs.
