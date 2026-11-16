# Fixed-time dynamics model contract

The new implementation is `worldmodels/restart_experiment.py`. It consumes the full-duration, nominal 10 Hz cohort created by `restart_data.py`. Its run identity is `medwm-dynamics-audit-v1`, and all new evaluation is exploratory because earlier outcomes from this dataset were known during design.

## Representation and inference

Each input is a 24-frame, 16-joint, two-coordinate array with separate validity. The observed prefix has 16 samples, and the eight future samples extend to approximately 0.8 seconds after the last prefix sample. Left targets are BlazePose joints 25, 27, 29 and 31. Right targets are 26, 28, 30 and 32. Coordinates are relative to the shared pelvis reference and a prefix-only torso scale. No physical units, neural states, muscle activations or treatments are inferred.

`features()` and `predict_checkpoint()` are the inference boundary. They accept the prefix alone and never consult future coordinates or validity. Missing prefix coordinates are filled only from earlier observations of that joint, with zero before the first observation. Observation indicators remain features. The latest observed coordinate is the persistence anchor. The model predicts residuals from that anchor.

Eight selected prefix positions are 0, 2, 4, 6, 9, 11, 13 and 15. Own-leg inputs contain 64 relative coordinates, 32 observation indicators and eight anchor coordinates, giving 104 features. Bilateral inputs double those numbers, giving 208 features. The static control replaces prefix coordinates with the latest observed coordinates and indicators with prefix-known indicators. All arms predict 64 outputs per leg, covering eight future times, four joints and two coordinates.

The shifted control first selects the same eight times as the bilateral arm. For each window, it circularly shifts the first seven opposite-leg observations and their validity indicators by a stable offset between one and six. The latest selected observation and own-leg history remain fixed. The offset is `1 + int(SHA256(window_id)[:16], 16) % 6`. This varies the alignment across windows. A globally identical feature permutation would be invertible by linear ridge and would therefore be an inadequate temporal control. The exact selected values, missingness, dimensions, target and anchor remain matched. The control is an information perturbation, not a biological intervention.

## Fitting and selection

The outer split stratifies source IDs into five folds with seed 2701. All windows of a source stay together. Only normal sources outside each held-out fold contribute to fitting. Three inner source folds select hyperparameters independently for each leg and arm. Feature means and scales are fitted on the current inner training observations, with equal source mass, equal sequence mass within source and equal window mass within sequence. Training weights have mean one.

Each output coordinate uses only its observed training targets with a known prefix anchor. Outputs with identical observation masks share the same matrix decomposition. The solver minimizes weighted squared error with an unpenalized intercept and ridge penalty. For each leg, target-loss weights additionally multiply the hierarchy weight by 64 divided by the number of observed output coordinates in that window. Thus a window has the same total coordinate-loss mass despite missing targets. Feature scaling retains the original hierarchy weights. Ridge strengths are 0.1, 1, 10 and 100. The primary arms fix the residual multiplier to one. A declared prospective sensitivity jointly selects the same ridge grid and residual multiplier 0, 0.25, 0.5 or 1 for every learned arm. Exact ties favor smaller multipliers and then larger ridge penalties. The zero multiplier is persistence and cannot be interpreted as learned dynamics.

The separately trained shifted arm and the inference-only shifted perturbation are distinct outputs. Persistence and a fixed half-velocity extrapolation are additional baselines. Velocity uses the last and fourth-last prefix samples, and is set to zero for a joint unless both endpoints were observed.

## Evaluation and retained artifacts

Every method is scored on the same target mask. A target must be observed and its joint must have at least one prefix observation. Both legs need at least four observed joint-time targets in the first four future samples and eight in the complete future. Exclusions are saved. Coordinate errors are averaged within each leg, the two leg errors are averaged equally within a window, then windows are averaged within sequence and sequences within source. Inner validation uses the same source aggregation separately per leg.

The run saves fold membership, candidate validation scores, selected parameters, 80 arm/leg/fold checkpoints, predicted coordinate trajectories, window scores, sequence scores, source scores, source-level anomaly rankings, execution time, dependency versions and artifact hashes. Some shrinkage checkpoints reuse the same fitted ridge coefficients, so 80 checkpoints do not mean 80 statistically independent models. Source bootstrap intervals condition on these fixed cross-validation fits and do not account for uncertain patient identity or all training-set variation.

Raw bilateral error, persistence error, excess bilateral error and bilateral-minus-own error have a fixed larger-is-abnormal direction. Prefix missingness is an acquisition control. Abnormal labels never choose forecasting hyperparameters. No observed outer outcome can change a score direction, model menu or threshold under this identity.

## Verification status

Focused tests cover future-value and future-validity invariance, missing-input invariance, prefix-only inference, masked weighted ridge agreement with an independent scikit-learn solver, source and sequence weighting, fold grouping, matched shifted values and masks, and missing-target loss invariance. A bounded real-data fit is a software and resource check only. The full run remains subject to independent implementation review and coordinator release of the heavy-computation slot.

## Completed real execution

On November 16, 2026, after independent software review and complete extraction verification, the following command completed in 21.18 seconds:

```bash
cd experiments/aaai-worldmodels
.venv/bin/python -m worldmodels.restart_experiment \
  --input artifacts/runs/restart-full10hz-v1 \
  --output artifacts/runs/restart-dynamics-v1
```

The model support gate removed three of 2,818 prepared windows. The evaluated cohort has 2,815 windows from 456 sequences and 91 sources, including 29 normal sources. All five outer folds completed. Each normal-source model trained on 23 or 24 sources outside its held-out fold. The run retained 80 checkpoints, complete coordinate predictions and all declared comparisons. The author verified all 94 artifact hashes and agreement with the frozen source-code and protocol hashes. Independent review of the real results is a separate acceptance step.

At the complete eight-sample horizon, bilateral forecasting reduced normal source-average MSE from 0.12126 for persistence to 0.08063, a 33.5% reduction. Its paired source-bootstrap gain was 0.04063 with interval [0.02399, 0.05911]. The separately fitted shifted control had MSE 0.08654, and the bilateral gain was 0.00591 [0.00141, 0.01134]. However, the static model had a lower MSE of 0.07164. The complete dynamics criterion therefore did not pass. Incremental gain over own-leg history was also uncertain.

The prespecified excess-error score had source AUROC 0.7931 [0.6802, 0.8960]. Raw bilateral error had AUROC 0.3042, persistence error 0.1935, bilateral-minus-own error 0.5495 and prefix missingness 0.4266. These are exploratory pooled out-of-fold rankings with conditional source-bootstrap intervals. A favorable excess-error ranking does not identify neurological causation or show that temporal prediction beats static posture. Residual shrinkage did not improve the bilateral result, and its outcomes remain in the saved tables. No score reversal, model expansion or outcome-driven refitting followed this run.

Before the full fit, nine focused model tests passed in 1.42 seconds, and a bounded fit on eleven real pilot windows checked all four arms and both legs in 0.091 seconds. The independent reviewer additionally replayed the complete synthetic pipeline to inspect all checkpoint predictions, source folds, aggregation and hashes. Synthetic checks were not used in any efficacy table.
