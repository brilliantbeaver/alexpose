# Independent review before the fresh study

This review inspected the existing implementation, data artifacts and released results on November 16, 2026. The reviewer did not author the model, data adapters, prior reports or proposed fresh implementation. This report concerns the historical study and requirements for its successor. It does not certify a future implementation that has not yet been reviewed.

The strongest reason to restart is scientific rather than a newly discovered implementation failure. The released primary JEPA classifier does not establish a pretraining advantage, and the forecasting controls do not establish useful ordered temporal dynamics. A defensible successor should test whether a deliberately small observation model uses gait history, with informative failures retained. A larger model or a new split alone would not repair the evidential limits.

## Findings and required dispositions

| Severity | Finding | Concrete evidence | Repair or limitation required in the successor |
| --- | --- | --- | --- |
| High | The old final sources are already exposed for methodological redevelopment. | `docs/final_scientific_review.md`, `docs/results_tables.md` and the README report their outcomes. `docs/test_release.json` also records older automatically computed test outputs. | Label analyses on these sources as exploratory redevelopment or internal cross-validation. Do not call a newly shuffled split an untouched independent test. Freeze a new protocol before fresh fits and seek external or genuinely new source groups for confirmation. |
| High | Existing JEPA results do not identify a useful temporal mechanism. | Static-prefix residual forecasting improves rather than worsens error. Shuffling the earlier prefix frames changes error little. The untrained classifier probe ties the primary JEPA probe in balanced accuracy. | Make the incremental value of ordered history the primary empirical question. Include persistence, static-context and a competent regularized dynamics baseline on identical support. Retain failures rather than choosing the favorable secondary anomaly score as a new confirmatory primary. |
| High | The current inputs use very little of the available walking footage. | `worldmodels/pose.py:extract_sequence` truncates with `max_frames`; `worldmodels/extended_data.py:prepare` explicitly takes the first 32 stride-two annotation rows. The cache scan found no full sequences in those caches. | Extract further annotated frames through Ambient with provenance, or clearly call a cache-based successor a bounded-prefix study. More windows improve observation coverage but do not increase independent participants. |
| High | Video source groups do not establish participant or site independence. | Source IDs are YouTube-style media identifiers. The existing data schema contains no verified participant identity or clinical adjudication. Only 29 normal source groups survive in the clean long cohort. | Report the independent unit as a source video. Audit repeated uploads and participants where evidence permits. Restrict generalization to this observational benchmark and require external patient-disjoint evaluation for clinical claims. |
| High | Person containment is not person tracking. | `gate_person` rejects an outside-box pelvis but preserves confidence on unresolved geometry. Existing independent visual review found identity switches and scene cuts in two whole stroke sources. | Preserve both existing source exclusions. Re-evaluate identity over newly extracted intervals. Record unresolved geometry and inspect representative trajectories and discontinuities. Do not infer patient/helper roles or diagnosis from appearance. |
| Medium | Frame-count horizons vary in physical time. | `make_windows` computes the sampling rate from native FPS and frame spacing. Predictions and temporal positions operate at frame indices. The reported clean cohort spans different forecast durations. | Prefer an explicit time grid or report duration distributions and restrict the timing claim. If resampling, define gap handling and never promote interpolated coordinates into observed target evidence. |
| Medium | Quality eligibility conditions on future observability. | `make_windows` accepts a window by the full-window valid fraction, and forecast support requires future observed coordinates. | This is appropriate for an observable-target benchmark but does not establish prospective coverage. Report prefix eligibility, future attrition and target counts separately. Assess whether quality attrition varies by label and source. |
| Medium | Normalization changes the prediction question. | Each frame is centered on its own pelvis, with scale estimated from the prefix. For random masking, normalization precedes masking and can use subsequently hidden hip/shoulder observations. | Call the output body-relative 2D articulation. Do not claim global displacement, physical speed, raw hidden-coordinate imputation or measured 3D anatomy. For any masked-completion claim, retain the documented preprocessing information boundary. |
| Medium | Numerical improvement in energy score does not establish useful uncertainty. | The stochastic arm has additional parameters and lacks unconditional-noise or equal-capacity controls. Prefix acquisition quality beats dispersion for error-based deferral. | Treat stochasticity as a secondary extension after the dynamics test. If studied, include these controls and a source-level calibration or coverage endpoint with enough calibration groups. |
| Medium | A larger family of exploratory scores creates selection risk. | Existing reports include several arms, seeds, horizons and raw, energy and excess-error scores. The favorable direct-model excess score is secondary and the same model forecasts normal motion poorly. | Predeclare the main contrast and score direction. Keep all planned controls, seeds and conditions in the outputs. Describe intervals as descriptive when they are not simultaneous or multiplicity-adjusted. |

No direct future-coordinate leak was found in the current forecasting predictor. Hidden coordinates are zeroed before the encoder, the residual anchor is formed from prefix observations, the prior is used at inference, and future visibility does not select predictor queries. Teacher access to the observed outcome constructs a training or scoring target. That distinction is deliberate and covered by regression tests. This conclusion concerns the inspected predictor, not every downstream interpretation of its errors.

## Cache and cohort evidence

The following counts were obtained by reading the actual local NPZ files, not by copying the README.

| Existing cache | Files | Unique sequences | Frames stored per file |
| --- | ---: | ---: | --- |
| `artifacts/poses/` | 1,391 | 652 | 1,383 files have 16 frames; four have 14; two have 13; two have 12. |
| `artifacts/runs/extended-input-v1/sequence_poses/` | 595 | 595 | Every file has 32 frames. |

Repeated short-cache files reflect different extraction or gating identities. They must not be counted as independent sequences or merged without checking the provenance fields. This audit establishes the scope of these two worldmodels caches. It does not claim that no longer pose cache exists anywhere else in the repository.

| Clean cohort | Windows | Sources | Train / validation / test sources | Normal training windows / sources |
| --- | ---: | ---: | --- | --- |
| `pilot-input-clean-v1` | 405 | 89 | 53 / 19 / 17 | 92 / 17 |
| `extended-input-clean-v1` | 407 | 91 | 53 / 19 / 19 | 84 / 17 |

The long test set contains six normal, two Parkinson's, four stroke, one cerebral-palsy and six myopathic source videos. A single cerebral-palsy source cannot support a stable condition-specific generalization estimate. The short and long cohorts differ, so their endpoint metrics are not matched cross-task comparisons.

## Existing controls worth preserving

The existing forecast training schedule samples a source, then a sequence, then a window. A batch does not repeat a source. Source-level reports average windows within a sequence before averaging sequences within a source. These are sensible protections against long or densely sampled videos dominating the result. Preserve them when expanding to complete sequences.

The current protocol also preserves the declared primary model despite a more favorable secondary model, keeps the failed validation criterion visible, separates an objective-weight sensitivity from the primary fit, and records checkpoint inheritance when training tensors did not change. These are useful practices for the restart. The released file identities passed independent verification.

## Recommended fresh design

1. Use all eligible annotation sequences and as much valid annotated footage as the declared extraction budget permits. Deduplicate frame requests within each source and use the shared Ambient RGB extractor. Store source, sequence, frame and time mappings, pixel dimensions, raw confidence and gate outcomes. Keep the source directory and previews under `data/videos-gavd`.
2. State a falsifiable neuromotor hypothesis about visible temporal coordination. For example, ordered bilateral history may predict future relative leg articulation better than a matched static state, and the history benefit may vary across annotated conditions. These are observational consequences compatible with several mechanisms, not diagnoses or causal lesion localization.
3. Fit a compact regularized dynamics model and a matched state-only model. Use the same target channels, horizons, quality support, normal-training groups and training-fold preprocessing. Include persistence and a cautious velocity reference. A neural model is an additional arm only when it answers a specified question beyond these baselines.
4. Evaluate with source-grouped outer folds and training-only hyperparameter choice, or a protocol with a fully fixed regularization value. Keep every window of a source in its assigned fold. Average windows within sequences and sequences within sources. Treat this as exploratory cross-validation because the source population has informed the redesign.
5. Pair the temporal model with controls that preserve the latest observed state. A shuffled prefix tests order sensitivity but can also create an out-of-distribution input; a separately fitted static-state model is therefore a necessary complementary control. Report effects on normal forecasting separately from abnormality ranking.
6. Report source counts, quality attrition, view and duration distributions, model fit budgets and all negative results. Bootstrap source groups after within-source aggregation. Do not treat folds, model seeds or many windows as extra independent patients. If pooled out-of-fold scores come from separately calibrated models, state the comparability assumption and include foldwise or training-calibrated sensitivity analyses.
7. Use a manuscript framing that survives either outcome: an audit of what an observable gait dynamics model learns and what it cannot identify. Workshop fit should come from dynamic prediction and its validation boundary, not from rebranding a static diagnostic classifier as treatment simulation.

## Checks actually run

The reviewer ran this focused command from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 experiments/aaai-worldmodels/.venv/bin/python -m pytest \
  -c experiments/aaai-worldmodels/pyproject.toml \
  experiments/aaai-worldmodels/tests/test_revision.py \
  experiments/aaai-worldmodels/tests/test_model.py \
  experiments/aaai-worldmodels/tests/test_pose_timing.py \
  experiments/aaai-worldmodels/tests/test_evaluation.py \
  -q -p no:cacheprovider
```

The result was **19 passed in 4.99 seconds**. These are software-invariant checks, not a fresh model-training experiment. A separate read-only SHA256 verification checked the joint release, its frozen files, every completion-record source CSV and all four result JSON files. **All 127 checks passed with no mismatches.** The review did not rerun heavy extraction or training and did not modify existing implementation or result files.

The fresh protocol, implementation, notebook executions, manuscript and rendered slides require a separate review after they exist. Acceptance of historical provenance does not certify future changes.
