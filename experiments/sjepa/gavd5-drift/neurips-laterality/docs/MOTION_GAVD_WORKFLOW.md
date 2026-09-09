# Running notebooks 15–18 on GAVD

The four notebooks now start with real GAVD. They share one input workflow and
one explicit fold/seed declaration. Notebook 17 trains the declared grid;
Notebook 18 opens those same saved jobs in a fresh kernel. Generated data is
available only through an explicit software-check mode.

Long-running cells use `NotebookTaskProgress` from `notebook_progress.py`.
Each task has one updating display with elapsed time, estimated remaining time,
and its active stage. Data preparation reports individual fold loads; mask
audits show the experiment, fold, seed and sampled-clip counts. Training shows
mask-schedule preflight and paired optimizer updates, then feature encoding,
ridge selection, predictor diagnostics, cache validation and pooled reporting.
Stage costs differ, so early ETA is approximate. Completed cache hits, disabled
training, absent checkpoints, optional skips and interruptions have distinct
statuses. A missing evaluation grid is never shown as a successful evaluation.

The wrappers observe the existing computations and restore their function
aliases on success, error or kernel interruption. Progress reporting itself
does not change training/readout computations or their identities. Use one wrapped task at a time within a
kernel, as with the existing workflow wrappers in the suite. Re-running a cell
starts a fresh progress display and reuses scientifically compatible artifacts.

## 1. Understand what was inherited from the existing notebooks

| Existing notebook or reference | Rule reused by 15–18 |
|---|---|
| 01, cohort and target audit | Locked pose/annotation inventories, extraction provenance, visibility and interpolation rules, normalized inputs and coordinate-derived target |
| 02, source-level splits | Five outer folds at source-video level; the same fold assignments for every training seed |
| 08, matched-budget masking | The tracked 1,200-update recipe, width 96, four encoder layers, two predictor layers, four heads, batch 20 |
| 11, masking patterns | Anatomical region graph and explicit target/context validity |
| 12, controlled pretraining | Paired initialization and exposure, train-only encoder fitting, explicit real-training enablement and source-balanced held-out reporting |
| 13–14, evaluation and forecasting | Separate encoder/predictor questions; distinguish missing raw observations, prepared-token completion and past-only forecasting |
| `docs/TUTORIAL.md`, Direction A | Test useful learned movement information against initial features before broadening masking mixtures |

The extension's own runner uses dense per-clip target loss. It keeps the current
fold's training tensors and validated masks resident on the selected device,
generates shared views there, and uses fused AdamW on CUDA. When periodic paired
resume is enabled, it can continue from a checksum-validated shared-arm optimizer
boundary; otherwise an interrupted job restarts from its seed. It does not claim
identical wall-clock performance across backends. Old protocol settings and
empirical results are preserved.

For Notebook 17's CUDA kernel setup, measured speedups, optional BF16 mode,
content-checked score/tensor reuse and the now-connected 100-update recovery
interval, follow [the performance review](MOTION_PRETRAINING_PERFORMANCE.md).
Keep `PRECISION` identical in Notebooks 17 and 18. BF16 and FP32 are separate
numerical modes and use different training/evaluation identities.

## 2. Prepare the real inputs

Run the setup, configuration and input cells in any of the four notebooks.
`DATA_MODE` defaults to `gavd`. The input helper:

1. Loads the paper context explicitly, regardless of the default smoke profile.
2. Reuses a complete cohort after checking protocol, context and file hashes.
3. If the cohort is absent, invokes the original `prepare_cohort` / `save_cohort`
   functions used by Notebook 01. Partial artifacts cause an explicit error;
   they are not overwritten.
4. Reuses verified source splits, or invokes the original `build_source_splits`
   / `save_splits` functions used by Notebook 02 when the split file is absent.
5. Loads `load_learning_dataset(real=True, fold=fold)` for every selected fold.
   It checks consistent cohort/split identities, disjoint train/test roles and
   once-only test coverage across the five folds.

On this checkout, the configured pose root is `../work/artifacts/real/poses`
and the annotation root is `../data-gavd`, relative to `neurips-laterality`.
The local cache contains 656 pose archives, while the protocol locks 642.
The original extraction generations select exactly those 642 archives, and
their inventory hash matches
`765851b9ddbc7f4534010cf5dfbd415c94c2824cc43f45b2e68b54893743654b`.

The helper copies that exact subset into
`artifacts/paper/protocol_6f7baefbda07/inputs/poses_765851b9ddbc/`, verifies the
copied bytes, and records the 14 excluded additions in `inventory.json`. The
source archives stay intact. Selection by provenance alone is insufficient:
the locked count and inventory hash must both match. The original loader then
checks the annotation identities, source mapping and per-archive provenance.
No extraction or downloading is performed.

The default prepared outputs are:

```text
artifacts/paper/protocol_6f7baefbda07/
  cohort/cohort.npz
  cohort/manifest.csv
  cohort/metadata.json
  splits/source_splits.json
```

`LATERALITY_ARTIFACT_ROOT` can override the paper artifact root, as in the
existing suite. The notebook prints the resolved directory and digests.
`CREATE_MISSING_INPUTS=False` permits reuse only. An unavailable or mismatched
cache stops explicitly; there is no automatic generated-data fallback.

## 3. Verify the five-fold, five-seed census

The reproduced cohort has 625 accepted clips from 93 videos, after excluding
17 of the locked 642 archives under the existing QC rules.

| Fold | Training videos | Training clips | Test videos | Test clips |
|---:|---:|---:|---:|---:|
| 0 | 74 | 436 | 19 | 189 |
| 1 | 74 | 443 | 19 | 182 |
| 2 | 74 | 553 | 19 | 72 |
| 3 | 75 | 548 | 18 | 77 |
| 4 | 75 | 520 | 18 | 105 |

Every row is repeated for seeds **42, 43, 44, 45 and 46**. The resulting census
has 25 rows. Seeds change stochastic training, not the source partitions.
`inputs["memberships"]` retains each sequence, video, fold, seed and train/test
role. Every video is test data once per seed. All clips from that video inherit
the same role. This is video separation, not verified person separation.

Condition counts and attrition are printed for audit, but dataset condition
annotations do not enter the mask sampler or encoder objective. Prepared model
inputs are `[625, 64, 33, 3]`, with a matching validity array and one
coordinate-derived bilateral movement target per clip.

## 4. Follow the notebooks in order

**15: motion weighting.** Audit uniform, MAMP-convention and robust motion masks
on every training clip for every fold/seed. The full declaration has 37,500
clip/arm draws. Inspect actual clip IDs, mask plots, realized counts, motion
enrichment and whether both legs receive targets. Read descriptive averages
as video-weighted coverage, not learned performance. The nominal 50% budget
comes from eligible gait tokens; it is not a 50% all-landmark mask ratio.

**16: structured masks.** Audit regions, trajectories and interior completion,
each with its own count-matched scattered reference: 75,000 draws. Missing cells
reduce supervised counts without trimming the declared structure. Compare
temporal brackets and visible graph neighbors. Regions and trajectories each
nominally hide 48 tokens; completion hides 132 and is a separate task. Only
regions are in the default new training grid.

**17: paired training.** Inspect the 25-row census, full recipe, arm definitions
and 50-row job table. Set `RUN_TRAINING = True` in its configuration cell and
run the subsequent cells. Alternatively, before starting the Jupyter kernel:

```powershell
$env:LATERALITY_RESEARCH_RUN_REAL = '1'
$env:LATERALITY_MOTION_DATA_MODE = 'gavd'
$env:LATERALITY_DEVICE = 'auto'
```

The study-specific alias `LATERALITY_MOTION_RUN_REAL=1` takes precedence over
the shared training flag. Run All with training disabled still uses real inputs
and displays readiness. It does not launch the 150,000-update workload.

Notebook 17's hardware preflight distinguishes the operating system's NVIDIA
inventory from the capabilities of the active Python kernel. An NVIDIA adapter
visible to Task Manager or `nvidia-smi` cannot be used when that environment has
a CPU-only PyTorch wheel: `torch.version.cuda` is then absent and
`torch.cuda.is_available()` is false, so `auto` resolves to CPU. Install a
CUDA-enabled PyTorch build or select the installed **GAVD5 CUDA (PyTorch 2.13)**
kernel, then rerun the configuration and preflight cells. Real auto-device
training now stops if NVIDIA hardware is visible but CUDA is unavailable.
An explicit unavailable accelerator also stops with an error.

The default `PRECISION="fp32"` retains FP32 computation. Optional `"bf16"` enables
native CUDA autocast for transformer/projector operations, with FP32 weights,
loss reductions and evaluation. The numerical mode has a distinct artifact
identity and must match between 17 and 18. `torch.compile` remains disabled.
`RESUME_INTERVAL=100` enables checked paired optimizer recovery by default.
The configuration deliberately does not accept FP8, INT8, or an 8-bit optimizer.
FP8 is reserved for a separate measured numerical experiment; 8-bit Adam would
save only about 12 MiB across the three motion arms, and INT8 belongs to a
separate inference/deployment evaluation. See the
[precision decision](MOTION_PRETRAINING_PERFORMANCE.md#fp8-and-8-bit-quantization-decision).
Synthetic mode always uses one CPU update in its separate artifact root, even
on a CUDA machine.

The motion experiment trains three encoders in each of 25 paired jobs. Regions
train two encoders in each of 25 paired jobs. Total: **125 encoders**, each for
**1,200 updates**. Shared initial parameters, source draws, views, objective and
per-clip hidden counts isolate masking within each experiment. Every scheduled
mask is validated before optimization. The progress display reports fold, seed,
experiment and update. Outer-test tensors never enter the resident training
bank. Completed compatible jobs and evaluations can be reused only after their
identities and contents validate.

**18: frozen evaluation.** Restart the kernel and use the same configuration.
The loader reconstructs the expected training identities; it does not train
an encoder. If a job is missing, the notebook lists it and withholds pooled
performance. If all jobs are present, it loads compatible readout tables or
computes new readouts from the saved encoders. It reports initial/online/teacher
features, two temporal summaries, direct pose and training-mean controls,
ridge selections, effective rank and predictor correspondence diagnostics.

The optional generated amplitude control appears after the GAVD workflow and
is labeled separately. The optional retained Notebook 12 reanalysis also stays
separate and does not substitute one older job for the new grid.

## 5. Read scores and choose the next investigation

The ridge grid spans 0.01–10,000. The comparative readout helper uses three
source-separated inner groups formed from sorted training-video IDs. This is
readout-only selection, not the registered protocol's four-fold inner encoder
selection. All scaler/ridge fits exclude the outer test sources. Features may
have been pretrained on all outer-training videos.

For each seed, pool all five held-out folds before computing source-balanced
R² and MAE. Each source video receives equal total weight. Every full-grid
arm/representation/seed therefore has 625 predictions from 93 videos. Eight
representations, five trained arms across two experiments and five seeds give
125,000 per-clip prediction rows. Never average fold R² values or treat repeated
seeds as independent participants.

First inspect trained-minus-initial contrasts under the same summary. Then
compare temporal summaries against means for both trained and initial features.
Finally compare masking arms against their own uniform reference. Bootstrap
intervals resample complete videos with paired conditions and seeds retained;
they condition on the fitted models. Seed variation is reported separately.

A readout gain shared with initial features does not establish a JEPA gain.
A trained-over-initial gain under temporal summaries supports the hypothesis
that averaging obscured learned information. Lower own-teacher prediction loss
alone is insufficient, because teacher scale and content change across arms.
For mismatch tests, compare errors on the same eligible control clips.

The notebook includes a conditional inference table for favorable, unfavorable
and ambiguous outcomes. The full new empirical grid remains to be run; input
and software verification do not establish masking benefits.

## 6. Find artifacts and recover work

`LATERALITY_MOTION_OUTPUT_ROOT` overrides the study output root in all four
notebooks. Defaults are `artifacts/motion_structured` for GAVD and
`artifacts/motion_structured_synthetic` for explicit generated checks.

```text
motion_structured/
  motion/.<training identity>.resume.pt, .<training identity>.resume.pt.sha256
  motion/<training identity>/manifest.json, *.pt, *_training.csv, *_schedule.npy
  regions/.<training identity>.resume.pt, .<training identity>.resume.pt.sha256
  regions/<training identity>/...
  evaluations/<readout identity>/manifest.json, predictions.csv, selection.csv,
                                  diagnostics.csv, predictor_diagnostics.csv
  grids/<grid identity>/manifest.json, jobs.csv, census.csv, memberships.csv,
                        predictions.csv, per_seed.csv, summary.csv, paired_intervals.csv, ...
```

Notebook 17 prints the exact expected complete and resume paths for every
experiment/fold/seed job. A completed directory counts only after manifest,
identity, file-inventory, SHA-256, tensor-shape, finite-value, schedule, and
control checks pass. A resume file and its checksum are only recovery candidates;
they are validated with the expected identity and full paired optimizer state
when resume is enabled, and never count as completed evidence. A missing half of
that pair fails closed. The completed directory is published atomically only
after training finishes.

`jobs.csv` maps every experiment/fold/seed to its training and evaluation paths.
Model/data/code/backend changes produce a different expected identity rather
than overwriting the earlier artifact. Missing completed jobs must be run in 17.
Without an enabled, valid paired resume candidate, an interrupted job restarts
from its seed. Keep fold, seed and experiment scope identical in 17 and 18;
deliberate subsets are pilots.

## 7. Verify without launching the full real training grid

From the repository root with its Python environment:

```text
python -m unittest discover -s neurips-laterality/tests -p test_motion*.py -v
python neurips-laterality/scripts/build_research_notebooks.py --check --only 15 16 17 18
python neurips-laterality/scripts/verify_motion_notebooks.py --execute
python neurips-laterality/scripts/verify_motion_notebooks.py --execute --data-mode gavd
```

The first execution command forces explicit generated mode in fresh kernels,
using one fold/seed and one update in an isolated output directory. The second
uses real GAVD with training disabled, runs the complete mask audits and checks
the declared real grid. Executed notebooks and figures are retained under
`executed/motion_structured`. The separate integration test actually optimizes
125 tiny encoders over all five folds and seeds on generated fixtures, then
checks train/test membership, complete coverage, cache reuse and corruption
rejection. These are software checks, not a new empirical masking comparison.
