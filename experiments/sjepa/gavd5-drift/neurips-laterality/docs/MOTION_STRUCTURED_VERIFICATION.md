# Verification of notebooks 15–18

Verified on 8 September 2026 with Python 3.12.10 and PyTorch 2.13.0+cpu on
Windows. The revised notebooks default to real GAVD, and their explicit
synthetic mode remains a separate software check. The full 1,200-update real
training grid has not been launched.

## Progress-wrapper follow-up

Notebooks 15–18 now use the existing `notebook_progress.py` display for input
preparation, fold loading, mask audits, checkpoint inspection, paired training,
frozen readouts, reporting and optional controls. Mask audits expose fold/seed
passes and clip batches; training exposes mask-schedule checks and optimizer
updates. Readout stages identify encoding, ridge selection and predictor
diagnostics. Cache hits, disabled training, missing jobs, skipped optional work
and interruptions remain distinct terminal states.

Seven new tests in `test_motion_notebook_progress.py` and ten existing tests in
`test_notebook_workflow_progress.py` pass. The new tests compare wrapped and
unwrapped mask tables and trained/evaluated results, exercise cache reuse,
check nested progress, and verify restoration after errors and keyboard
interrupts. Completed cache hits are removed from ETA's remaining-computation
count. The scientific source files, training/readout identities and notebooks
00–14 match the recorded pre-wrapper hashes.

All 29 code cells execute in fresh kernels with progress enabled, retained at
`executed/motion_structured/synthetic_tw34buwj/`. Real-input execution of 17–18
is retained at `executed/motion_structured/gavd_wly3264r/`, with real training
disabled. Their rendered HTML contains the updating progress displays and
correct disabled/missing-grid/optional-skip states. No full real grid or new
scientific masking comparison was launched for this display change.

```text
python -m unittest discover -s neurips-laterality/tests -p test_motion_notebook_progress.py -v
python -m unittest discover -s neurips-laterality/tests -p test_notebook_workflow_progress.py -v
python neurips-laterality/scripts/verify_motion_notebooks.py --execute
python neurips-laterality/scripts/verify_motion_notebooks.py --execute --data-mode gavd --only 17 18
```

## Real GAVD inputs and masking audits

The available source cache contains 656 pose archives. Extraction provenance
recovers exactly the protocol's original 642-archive inventory, including its
registered SHA-256. Verified copies were made without modifying the source
cache. The original Notebook 01 preparation rules produce **625 accepted clips
from 93 source videos**. Notebook 02's source splitter reproduces all five
reference train/test counts: 436/189, 443/182, 553/72, 548/77 and 520/105.

All four notebooks display the real 25-row census for folds 0–4 and seeds
42–46. Fresh-kernel execution completed the real-data code paths: 15 and 16
perform full training-mask audits; 17 displays the complete training workload
with training disabled; 18 reports the missing real training jobs without
substituting a synthetic model or an incomplete aggregate.

The audits inspect 37,500 motion draws and 75,000 structured/reference draws,
for **112,500 real training-clip mask draws**. Realized hidden counts match
within each comparison. Descriptive video-weighted means across the fold/seed
audits show:

| Comparison | Mean realized targets | Temporal brackets, structure / reference | Visible neighbors, structure / reference |
|---|---:|---:|---:|
| Connected region / its uniform reference | 47.30 | 0.000 / 0.699 | 0.512 / 0.988 |
| Full trajectories / their uniform reference | 45.49 | 0.000 / 0.706 | 0.995 / 0.989 |
| Interior completion / its uniform reference | 120.97 | 0.000 / 0.482 | 0.000 / 0.941 |

These findings establish which specific cues the masks remove. In particular,
full trajectories remove same-landmark temporal brackets but leave graph
neighbors almost always visible. They do not demonstrate a learned masking
benefit. Motion-weighted draws show positive enrichment under the common robust
motion diagnostic; uniform draws remain close to zero enrichment. This also
concerns selection behavior, not downstream performance.

The real executed notebooks and vector/PNG figures are retained locally at:

- `executed/motion_structured/gavd_5yc3ve5h/`: notebooks 15 and 16. Both finished;
  the initial verification command subsequently stopped because builder metadata
  was updated during that command. Their cell sources were checked against the
  final builders separately.
- `executed/motion_structured/gavd_wag7vh63/`: notebooks 17 and 18, with the
  completed verifier's `verification.json`.

All seven real-mode figure outputs were checked for readable axes and labels.
Notebook 18's generated amplitude figure is explicitly labeled as a separate
teaching control. Source notebooks remain output-free.

## Actual real-data training/evaluation check

A separate bounded GAVD check trained the three motion arms and two region
arms on **fold 0, seed 42**, for **one update at width 16**. It used 436 real
training clips from 74 videos and evaluated 189 held-out clips from 19 videos.
The readouts produced 7,560 per-clip prediction rows across the five arms and
eight representations. Every prediction is marked `synthetic=False`.

Both paired jobs completed, persisted their checkpoints and readout tables,
and reopened successfully without encoder training. The check and its scope
are retained at `executed/motion_structured/gavd_training_check_nq9klam6/`,
including `verification.json`. This verifies optimization, missingness handling,
readout fitting and held-out prediction on actual GAVD arrays. One update with
a small model is not the planned 1,200-update experiment and supplies no
scientific ranking of masking policies.

## Automated software checks

**All 19 extension tests pass** (`test_motion_structured.py` and
`test_motion_gavd.py`). The six new workflow tests cover:

- Exact-inventory recovery from an expanded cache, byte-preserving source
  handling, rejection of a wrong inventory and refusal to overwrite partial
  cohort artifacts.
- Real mode as the default and failure without synthetic fallback when its
  inputs cannot be prepared.
- Complete training-clip coverage, exclusion of outer-test sources from each
  fold's mask audit, matched budgets and label-blind masks.
- Actual tiny optimization of **125 encoders in 50 paired jobs over every fold
  and seed** on generated fixtures. Saved schedules contain training videos
  only, and predictions contain exactly the correct held-out videos.
- Complete pooled prediction coverage, reopening all jobs without encoder
  training or repeated readout fitting, and corruption rejection even when
  compatible evaluation tables exist.
- The full real recipe (1,200 updates, width 96, 125 encoders, 150,000 updates)
  and rejection of a modified/misreported workload.

The existing 13 tests continue to check official MAMP log-weight agreement,
motion bias, stationary fallback, missingness isolation, mask geometry,
label/test-content isolation during training, gradient-free teachers, fixed
hidden-content isolation, paired exposure, checkpoint reuse, source-only ridge
selection, amplitude recovery, strict aggregation and retained-encoder reuse.

All four notebooks also executed in separate fresh kernels in explicit
synthetic mode. The main execution is retained at
`executed/motion_structured/synthetic_1azgyh3g/`; the final optional retained-job
cell in Notebook 18 was additionally executed in
`executed/motion_structured/synthetic_8hu7f9bi/`. The current four notebooks
contain **29 code cells**. Source/builder checks and `git diff --check` pass.
The completed verifier reports 57 protected files unchanged; notebooks 00–14,
core model code and protocol configuration remain untouched.

The user's previously executed Notebook 16 was preserved before rewriting:
`executed/motion_structured/before_gavd_revision_cx2cx_ty/16_structured_masking_and_context.ipynb`,
SHA-256 `eca6bd2419c6b987faa8dfecab796c85fe27c7c3c546dd35fc2d836dfe3f086f`.

## Commands and remaining scope

```text
python -m unittest discover -s neurips-laterality/tests -p test_motion*.py -v
python neurips-laterality/scripts/build_research_notebooks.py --check --only 15 16 17 18
python neurips-laterality/scripts/verify_motion_notebooks.py --execute
python neurips-laterality/scripts/verify_motion_notebooks.py --execute --data-mode gavd
```

The previous full-suite run, before this GAVD revision, passed 219 of 221 tests.
Its two existing environment-dependent failures were the missing old Notebook
08 reference grid and the CRLF-versus-LF raw implementation hash checked by
`test_training_progress.py`. That entire older suite was not repeated for this
revision; its compatibility rules were not relaxed. The 19 affected extension
tests were run together and passed.

GPU execution and the full new real-data model comparison remain unverified.
The tutorial's previous empirical conclusions and PDF are unchanged; its
Markdown navigation now points to the real workflow. The
[GAVD run guide](MOTION_GAVD_WORKFLOW.md) explains how to launch Notebook 17's
full grid and evaluate those same saved jobs in Notebook 18.
