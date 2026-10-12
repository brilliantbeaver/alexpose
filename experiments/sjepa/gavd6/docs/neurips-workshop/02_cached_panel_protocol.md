# A frozen development test of student-accessible teacher information

Version: `student-accessibility-v1`, 11 September 2026. This document is written before fitting the new comparison. Its frozen copy and digest belong to `outputs/iclr-bridge-cached-20260911/config/`. Subsequent changes to a scientific choice require a new experiment. This is an exploratory change of question, not a new version of the completed direct-v3 gate.

## 1. Question and information boundary

The completed Experiment 0 asks whether skeleton history adds predictive information after RGB features and recording variables. A skeleton-only student cannot use that RGB reference. This follow-up asks whether a fixed representation of the first 32 skeleton frames predicts the existing contextual teacher target after current posture and observation quality are accounted for.

The primary comparison is **real history minus no-skeleton within the posture panel**. The support panel and comparisons with baseline, shuffle and mismatch are secondary, descriptive comparisons. We will not choose a primary panel, feature family or target after reading outer-fold results. A positive result is a lead for a new target and distillation study; it does not establish clinical utility, causal dynamics, a better learned representation, or an independent confirmation.

Current posture means frame-31 coordinates in the existing prefix-normalized representation. Its torso scale uses frames 0–31. Thus this reference is a function of the observed history, not an independently measured single image. Neither panel uses RGB or background optical flow. Observation support and confidence use the complete allowed prefix.

## 2. Data and target held fixed

Reuse `outputs/future-innovation-direct-v3-dev-20260911` and its verified, read-only lineage to `outputs/future-innovation`. Retain all 50 clips, 43 source videos, five outer source folds, three inner source folds, matching metadata, projected person target and original teacher audits. These sources have already informed development. Splitting by recording does not establish participant separation.

Each history is a 32 × 33 × 4 array: x, y, confidence, validity. The target has 256 projected teacher dimensions from frames 38–39, encoded using the full 64-frame video. It can therefore contain information from frames later than the named target frames. The measured outcome is prediction of contextual teacher features. No teacher inference, raw-video mount or GPU is needed; future-only targets, mirrored RGB targets and future pose are unavailable in this cache.

## 3. Fixed feature families

| Block | Frozen construction | Dimension |
|---|---|---:|
| Support reference X | In bins [0,8), [8,16), [16,24), [24,32), mean raw confidence and validity for all 33 joints; frame-31 confidence and validity; decoded frames per second | 331 |
| Posture reference X | Support reference plus frame-31 x and y for all joints; invalid coordinates are missing values | 397 |
| Added history S | Existing `ordered-bins-v1`: four bins × 33 joints × x, y, valid adjacent-frame x/y displacement, valid confidence, support and transition support | 924 |

Within each bin, joint index precedes channel index. Confidence and validity in X are bounded fractions. Posture coordinates use the parent body's normalized units; FPS is a continuous timing input. History displacement is per sampled frame, not physical speed. Invalid coordinates do not create artificial velocities. No learned projection, feature selection or new augmentation is introduced.

## 4. Training, selection and exact rejection

Use `joint-ridge-v1`, `supported-input-v1` and `training-target-v1` without changing their historical implementations. The fitted prediction is an unpenalized intercept plus X W_x + S W_s. Minimize summed source-weighted squared error plus λ_x ||W_x||²_F + λ_s ||W_s||²_F. The squared Frobenius norm is the sum of squared entries. Normalize fitting weights to sum to the number of training windows; give each source the same total weight. Solve in float64.

Choose each positive penalty from {0.1, 1, 10, 100, 1,000, 10,000}. First select one shared reference X model from six penalties on the three inner source partitions. Each arm then receives the same 36 joint penalty pairs and one exact `baseline_only` candidate. This candidate predicts the selected reference without a correction. Every arm has identical opportunities, but supports can differ after controls.

Pool inner validation squared-error sums and source-weight totals in each inner training partition's standardized target units, using its training-variance mask. Tie tolerance is 10^-10 + 10^-8 × the relevant loss magnitude; prefer baseline for a tied correction, otherwise larger X and then S penalty. The reference also prefers the larger tied penalty. All candidates, rejected candidates, failure records, inner partitions and selection reasons remain saved. A required numerical candidate failure makes measurement incomplete; a failed required reference aborts. Outer scores cannot select anything.

Input statistics are fitted only on the applicable training sources. Unsupported columns have exactly zero influence everywhere. A usable column requires two observed windows, two observed sources and standard deviation exceeding the frozen threshold. Threshold/scale-floor pairs are: fractions (10^-6, 0.1), coordinates and displacement (10^-6, 0.01), continuous variables (10^-8, 0.001). Missing values use training means; imputed variation contributes zero. Target scaling is separate and preserves raw-unit inverse transforms and training-derived masks.

The primary model is deterministic. Seed 0 identifies its single solution; it is not a repeated optimization trial. Do not duplicate results under three seed names or claim stochastic stability.

## 5. Matched controls and source boundaries

Apply the existing controls to raw histories before feature construction. Real uses the original prefix. Shuffle permutes four-frame blocks with coordinates, confidence and validity together. Mismatch uses a different-source donor from the same training, validation or test partition under the existing context rule, which does not use targets. No-skeleton zeros coordinates and confidence while retaining original time-varying validity. Each arm retains the recipient's X reference.

All clips from a source remain together. Save every inner and outer fit identity, preprocessing identity and donor identity. Partition-local matching is distinct from learned preprocessing, which uses training sources only. Shuffle and mismatch can change observation patterns, so their effects are not pure interventions on motion order. The no-skeleton comparison controls validity under the selected finite model family; it does not remove all possible recording confounding.

## 6. Scoring and interpretation fixed in advance

Pool exactly one held-out prediction per panel, window, arm and target feature. Use the intersection of outer-training valid target dimensions. For each dimension, predictive R² is 1 minus source-weighted prediction error divided by error of its outer-training-mean reference. Average across dimensions. This is not the laterality paper's test-mean-centered R².

Use 2,000 paired whole-source bootstrap draws with seed 260905; preserve repeated-source multiplicities, clips within source and alignment across arms. Report percentile 95% intervals and positive fractions for real-minus-reference and real-minus-each-arm. These intervals condition on saved models: they exclude repeated fitting, selection uncertainty and adaptive redesign of this cohort.

The descriptive status is `development_lead` only when the primary real-minus-no-skeleton interval has lower endpoint above zero and real beats both shuffle and mismatch in point estimates. Otherwise a complete run is `no_supported_temporal_lead`. Any required invalid comparison is `incomplete_measurement`. `scientific_advance` is always false: neither status authorizes full training or changes the original direct-v3 STOP. The historical +0.05 direct-gate criterion belongs to its RGB-conditioned question and is not silently transferred to this changed comparison.

## 7. Verification, preservation and recovery

Before fitting, copy this protocol, schemas and configuration into the child root; record source/parent digests, byte sizes and modification times, original contracts, code fingerprints and runtime. The source and parent trees must remain unchanged. A changed contract, cache, schema or implementation rejects reuse. Keep all new artifacts in the child tree. A stage lock rejects concurrent writers; completed folds can resume through verified receipts. A completed run verifies instead of fitting again.

The read-only numerical check reloads and refits selected linear models and preprocessing from their declared training sources; reconstructs every standardized and raw prediction, mask, score, bootstrap draw and selected identity; and compares reports. It checks candidate loss pooling and winner selection, but does not independently refit every rejected candidate. Tolerances are 10^-10 absolute/relative for prediction reconstruction, 10^-12 absolute and 10^-10 relative for scores. Hashes alone do not establish arithmetic correctness.

## 8. Calibration and commands

Before the real run, run the focused tests in `tests/test_iclr_bridge.py`: feature dimensions and missingness, training-only scaler behavior, a source-held planted earlier-motion signal through the full finite search, exact no-skeleton fallback, exact-zero paired bootstrap and updated-hash coefficient-tamper rejection. Separate symmetry fixtures test reflection, time reversal, valid velocities and the failure of symmetry alone to establish usefulness. These are software evidence, not experimental gains.

From the repository root:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_iclr_bridge*.py' -v
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py freeze --source-run outputs/future-innovation-direct-v3-dev-20260911 --output-root outputs/iclr-bridge-cached-20260911 --protocol-document docs/studies/iclr/02_cached_panel_protocol.md
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py run --output-root outputs/iclr-bridge-cached-20260911
.venv/bin/python scripts/research_directions/iclr_bridge/run_cached_panel.py verify --output-root outputs/iclr-bridge-cached-20260911
```

Re-run `run` to recover an interrupted stage or verify completed-stage reuse. Freeze only into a new directory. Do not delete locks without identifying their owner. The final validation report records which commands actually ran, their outcomes and any amendment; this prospective document contains no revised held-out result.
