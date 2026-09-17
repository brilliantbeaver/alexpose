# Temporal gait on HAIC

Use the [execution guide](../../docs/studies/temporal-gait/execution/haic.md) for exact manifests, configuration, commands, outputs and resume behavior. The [notebooks](../../notebooks/temporal_gait/README.md) explain the same stages.

```bash
bash slurm/temporal-gait/submit.sh pilot --dry-run
bash slurm/temporal-gait/submit.sh pilot
```

`pilot` submits inventory → preparation → information audit → parallel masked/future arrays → development evaluation. The final evaluation depends on **both complete arrays**. Pilot never opens test. `develop` and `confirm` require a saved positive expansion decision; individual stages still enforce their runtime data/receipt gates.

`common.sh` chooses the explicit interpreter, deterministic CUDA workspace and thread limits. Individual `.sbatch` files retain visible resource requests. The thin shell submission wrapper delegates scheduler argument construction to `scripts/research_directions/temporal_gait/submit.py`, which records job IDs/dependencies, handles cluster-suffixed responses and never invokes a shell to construct scheduler arguments.

Dry runs parse configuration and construct task mappings without opening manifests or media, writing a task grid, or submitting jobs. Actual submission freezes `manifests/task-grid.json` first. The default account/partition are `mind` / `hai`; GPU jobs request one H100. `TG_ARRAY_CONCURRENCY` defaults to two tasks per array, so the parallel masked and future arrays may together use up to four GPUs. Adjust it to your allocation; these scripts do not implement distributed multi-GPU training.

The pilot template uses audited `historical_overlap` sources for the first matched timing/support contrast. `full-cohort.example.json` expands to `full_allowed` in a new run root while keeping the remaining recipe fixed. Every exposure row explicitly supplies historical membership; no old source list is imported. Phase expansion adds seeds, not cohorts.

Real data require supplied full videos and complete, causally prepared pose caches. There is no raw-pose extractor fallback. Dense/deep and frozen-video extension submissions return `gated_not_implemented` without allocating a job and cannot be interpreted as completed experiments.
