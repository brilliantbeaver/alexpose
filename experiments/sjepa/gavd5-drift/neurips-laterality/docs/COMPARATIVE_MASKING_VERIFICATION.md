# Verification of comparative tutorials 11–14

The four tutorials and their new helpers are implemented. Their small teaching examples use generated movement. As checked on 8 September 2026, Notebook 12 also retains a completed real-data gait-target versus all-landmark comparison: 25 paired jobs, 50 encoders, and 1,200 updates per encoder, evaluated on 625 clips from 93 videos. The results do not establish a masking advantage or improved laterality prediction over the matched initial encoder. Other masking-family training comparisons and real-data forecasting remain unrun. [TUTORIAL.md](TUTORIAL.md#3-what-the-updated-notebooks-help-us-answer) gives the updated interpretation.

## What was checked

All **208 tests** in the laterality suite passed. The checks include unequal target counts within a batch, isolation of withheld content, source-separated fitting, paired training randomness, exact CPU continuation from a shared two-arm checkpoint, rejection of incomplete optimizer state, row-selected feature extraction, impossible structured budgets, corruption feasibility, and validation of saved results. Tests of a deliberately informative synthetic signal establish that the readout can recover information supplied to it; no test requires a proposed masking method to win.

All four notebooks executed in fresh kernels and rendered vector figures during the recorded software verification. Working copies of notebooks 11–14 now retain inspected outputs and result-specific commentary. Notebook 12's final output records the completed real grid, including recovery of an interrupted job. Notebook 14's current working copy and an earlier retained teaching copy contain the annotated synthetic results. The commentary review preserved all code and outputs. The older 07–10 verifier retains its original scope and passed its source checks and 60 extension tests during the earlier verification.

A content comparison confirmed that the **469 files present in the protected research set before implementation remain unchanged**, including notebooks 00–10, their editable tutorial sources, existing helper implementations, protocol and configuration files, retained artifacts, and earlier executed copies. The new comparison code resides in separate modules.

Independent reviews checked mask construction and training controls, evaluation boundaries, forecasting aggregation, and the tutorial explanations. Corrections included recording coverage after reflection, validating cached checkpoints and sampling schedules, identifying infeasible evaluation corruptions, and rejecting mixed checkpoints or timing recipes during aggregation. An independent numerical example with unequal clips per source confirmed the forecast aggregation's source weighting and placement of the square root after pooled squared errors.

## Real-data checks performed without training

The 25 saved masking-comparison manifests agree on the reference settings. The two-condition full plan displays 50 encoders and 60,000 optimizer updates before training is enabled.

| Outer fold | Training clips | Test clips | Training sources | Test sources |
|--:|--:|--:|--:|--:|
| 0 | 436 | 189 | 74 | 19 |
| 1 | 443 | 182 | 74 | 19 |
| 2 | 553 | 72 | 74 | 19 |
| 3 | 548 | 77 | 75 | 18 |
| 4 | 520 | 105 | 75 | 18 |

The existing gait/all-landmark plan passed input and mask-feasibility checks for all five folds. Separate fold-0 preflights also passed for motion weighting, two whole trajectories, a three-landmark region over three time blocks, and a two-block interior gap, each with an appropriate scattered reference. These checks establish feasible configurations, not completed training results or a performance ranking.

Forecast preparation passed for both twelve- and 33-landmark inputs across all five source partitions. Both versions retain **611 clips, 1,814 clip–horizon examples, and 93 sources**, with identical future landmark endpoints, endpoint times, and validity. The repeated horizons share recordings and are not independent participants. Source separation, availability of mismatched training futures, and full declared evaluation coverage were checked before training.

## Evaluation scope

The automatic masking runner evaluates frozen online and teacher features, initial features, direct-pose summaries, and the training-source mean. It retains per-clip predictions and readout validation results. The JEPA predictor is examined against its own teacher, with feature-variation and mismatched-target diagnostics. Each teacher defines its own latent coordinate system, so these feature errors do not provide a common ranking of movement usefulness.

Its automatic missing-observation comparison concerns masking prepared coordinates. Notebook 13 separately implements and tests raw observation removal before interpolation and normalization. A dataset-wide raw-missingness experiment requires a declared corruption plan and application of that helper to the selected recordings.

The forecasting runner fits a decoder on observed future-teacher features from training sources and applies that same decoder to observed and predicted future features. It saves future coordinates and prediction coverage and pools outer-fold squared errors within each seed before calculating RMSE. Paired source intervals preserve clips and seed predictions together. These intervals condition on fitted models and exclude full retraining uncertainty.

Controlled-masking training checkpoints both arms at one shared boundary and resumes only an identical, verified fold/seed request. An exact interrupted-versus-uninterrupted CPU test covers state restoration; MPS and CUDA use the same state-complete path without a claim of bitwise equality across accelerator kernels. Forecasting jobs still remain incomplete after interruption and have no automatic resume path. Completed results are reused only after compatibility and content checks. Synthetic execution does not measure accelerator performance or demonstrate that the full real-data training budget is sufficient.

## Reproduce the software checks

Run from the repository root:

```bash
.venv/bin/python -m unittest discover -s neurips-laterality/tests -p 'test_*.py'
```

The test count above records the implementation verification; this documentation review did not rerun training or the full test suite. Executed copies now contain added interpretation cells that are intentionally absent from the editable synthetic sources. Builder equality checks therefore do not apply to these annotated copies. Preserve them before regenerating output-free notebooks or running the synthetic execution verifier.

The [experiment specification](COMPARATIVE_MASKING_PLAN.md) describes the comparisons, and [TUTORIAL.md](TUTORIAL.md#9-copy-ready-implementation-prompt-for-the-next-notebook-suite) links the implemented notebooks and retained implementation prompt.
