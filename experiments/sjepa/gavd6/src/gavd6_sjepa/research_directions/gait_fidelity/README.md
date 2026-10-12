# Gait Fidelity implementation

The implementation lives in the single Python package `gavd6_sjepa.research_directions.gait_fidelity`. Source code and notebooks use `gait_fidelity`; documentation and Slurm scripts use `gait-fidelity`. Run `python -m gavd6_sjepa.research_directions.gait_fidelity --help` with this checkout's `src` on `PYTHONPATH`.

The [HAIC guide](../../../../slurm/gait-fidelity/README.md) starts with the installed synthetic-training CUDA environment. The [notebooks](../../../../notebooks/gait_fidelity/README.md) use the same command interface, so tutorial execution and batch execution share their configuration, models and evaluator.

| Module | Responsibility |
| --- | --- |
| `config.py`, `spec.py`, `recipes.json` | Discover HAIC assets, save an explicit protocol, and enumerate the focused core or the full 34-recipe matrix across three seeds. |
| `cohort.py` | Join the full eligible AMASS inventory to audited identities and original splits; freeze nonoverlapping candidate intervals and explain every exclusion. |
| `data.py`, `preparation.py` | Check people, clocks and references; render changed bodies, extract fresh estimator tracks, and publish memory-mapped arrays in complete source-family chunks. |
| `masking.py`, `measurements.py` | Sample observed-token masks and compute signed projected knee excursion. |
| `training.py` | Fit shared pretraining phases and independent readouts, including all supervision controls. |
| `scheduler.py` | Reserve allocation cost, launch independent Slurm workers and retain dependency/attempt identities. |
| `evaluation.py` | Reconstruct physical-time coordinate and movement metrics, response interactions and person/seed uncertainty. |
| `confirmation.py` | Bind original AMASS test people to reviewed exposure evidence, frozen checkpoints and a declared final comparison. |
| `gavd.py` | Index full GAVD annotations, group related videos, extract real pose observations, and evaluate training-only gait-label probes with recording-level uncertainty and acquisition controls. |
| `visualization.py` | Review actual videos with aligned skeletons and trajectories. |
| `cli.py` | The common notebook and shell entry point. |

Historical synthetic-training-v2 source is imported without modification. Generated data, fitting attempts, checkpoints, predictions and receipts belong in the selected output directory. A software fixture is explicitly labelled and cannot become source evidence by changing its status string.

## Data and claims

Full-manifest AMASS preparation is the default for a new source run. The `named_walking` preset scans the entire eligible inventory for candidate walking motions; `treadmill_walking` provides a narrower speed-stratified cohort. `all_eligible` includes other actions and must be described as a heterogeneous motion experiment. An interval review file can explicitly include or exclude motion intervals. The old retained bundle remains available through `legacy_roster` for reproducibility.

Each original AMASS identity keeps its original split. Ordinary preparation uses training and validation people; original test people require a separate confirmation declaration. Missing prior-exposure information supports exploratory development only. Technical geometry checks do not enforce normal or symmetric gait, and filename labels do not certify walking quality. Review the retained contact sheets, videos and exclusion tables before interpreting results.

The GAVD branch uses the full sequence inventory and raw annotated person boxes. It retains actual decoded timestamps and admits complete fixed-length windows without repeating or padding video frames. Related clips, supplied person links and exact duplicate media share a group. Historical reservations remain protected. The default model comparison resolves every declared seed of the primary AMASS comparison from verified completion receipts.

GAVD gait labels train a fixed ridge classifier on training recordings only; they never train the pose restorer. Its features summarize normalized joint trajectories and projected knee motion. A camera-view/source-height classifier measures acquisition confounding. Comparisons retain matched populations and resample whole recording groups, with crossed recording/seed intervals for the declared neural comparison. This measures retained label information, not pose accuracy, disease diagnosis or affected-limb correctness. Short, low-frame-rate, missing and failed recordings remain visible in extraction coverage.

## Scale and reproducibility

Source arrays use hashed, memory-mapped `.npy` files. Preparation retains complete source-family chunks for recovery; merging and prediction export write bounded batches. Training in the core protocol samples people, motions and windows hierarchically. The full matrix retains matched endpoint batches for its re-pairing control, and records that different sampling protocol explicitly. Evaluation averages windows within motions, then motions within people, and keeps coverage alongside errors.

The saved cohort plan reports derived render/track counts and estimated array storage. Preparation chunks, shard bundles and the merged bundle intentionally coexist for verification and recovery, so budget for the reported retained copies. Eight GPUs bound concurrency, not the total amount of work: measured throughput and the saved allocation allowance determine feasibility.

The [validation record](../../../../slurm/gait-fidelity/validation/README.md) retains the tested code identity, automated checks, executed-notebook receipt and independent review findings, including the remaining HAIC verification requirements.
