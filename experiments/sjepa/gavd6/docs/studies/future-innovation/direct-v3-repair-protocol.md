# Experiment 0: calibrated joint prediction on the inspected development cache

Specified 11 September 2026, before revised real-data fitting. This protocol is
`direct-v3`, model `joint-ridge-v1`, input preprocessor `supported-input-v1`, target
standardizer `training-target-v1`, and skeleton schema `ordered-bins-v1`. Its
primary predictor is deterministic joint linear ridge. There is no nonlinear
candidate or sequential residual fit. The completed `gate-v2` / `direct-v2` STOP
and its post hoc diagnostic interventions remain unchanged.

The question is whether prefix skeleton coordinates and confidence help predict
contextual teacher features after current video, recording conditions and
observation support have been accounted for. The inspected 50 clips from 43
sources are development data. A different directory does not restore an untouched
test set. Inputs remain frames 0–31. The projected person-region target at frames
38–39 was encoded with all 64 frames and may reflect later observations. This
experiment predicts that contextual feature vector, not isolated future gait.

## Predictor and reference

Let X contain safe RGB/nuisance features, S fixed skeleton summaries, and Y the
target in the current training partition's standardized units. The fitted model
is `intercept + X W_x + S W_s`. It minimizes the **sum** of source-weighted squared
errors plus `lambda_x ||W_x||² + lambda_s ||W_s||²`. Each matrix penalty sums its
squared entries (the squared Frobenius norm). The intercept is unpenalized.
Within a fit, each source has equal total weight and weights sum to the number
of training windows. A float64 Cholesky solve of the sample-sized positive-definite
kernel gives the unique regularized coefficient solution. No arbitrary weights
remain in unsupported training directions. Disabled S reproduces weighted
RGB-only ridge for the same RGB penalty and preprocessing.

Both penalties use the finite grid `[0.1, 1, 10, 100, 1000, 10000]`. Each outer
fold first selects one RGB-only reference from that RGB grid using three inner
source partitions. All four arms share that reference. Each arm then evaluates
36 joint penalty pairs and an explicit `baseline_only` candidate equal to the
selected reference. S is truly disabled for that candidate. It does not mean a
large penalty or a neural head with zero updates. A saved typed fallback reloads
the same baseline and returns exactly the same prediction.

The joint model may choose a different RGB penalty. Consequently gain over RGB
alone cannot establish skeleton benefit. Real minus no-skeleton is separately
required. The latter control retains time-varying validity and the same search
opportunities. Nominal and supported feature counts are reported because equal
coefficient counts do not imply equal statistical flexibility.

## Fixed representation and preprocessing

The same constructor follows each raw control transformation. Four ordered bins
are `[0,8)`, `[8,16)`, `[16,24)`, `[24,32)`. Within each bin, for each of 33 joints,
feature order is x, y, velocity x, velocity y, confidence, valid-frame fraction,
and valid-adjacent-transition fraction: 924 columns in bin/joint/channel order.
Coordinates use the cached prefix-only body normalization. Velocities are
coordinate changes per frame, using only adjacent valid observations within a
bin. Missing endpoints never create movement. Means with no observations are
missing values, not zeros inferred as observed. Fractions use 8 frames or 7
possible within-bin transitions. Frame rate remains a nuisance input.

All learned input statistics use only the applicable training sources. Weighted
observed means supply imputation. Variances use mean-imputed training entries
and the same source weights. A column needs observations from at least two
sources and standard deviation strictly above its declared minimum. Otherwise
its transformed values are exactly zero everywhere. Column positions remain
stable. The following values are in each feature's own units:

| Input kind | Minimum standard deviation | Scale floor |
|---|---:|---:|
| RGB feature | 1e-8 | 0.01 |
| Other continuous nuisance | 1e-8 | 0.001 |
| Normalized coordinate | 1e-6 | 0.01 |
| Velocity per frame | 1e-6 | 0.01 |
| Confidence, support, missingness, view fractions | 1e-6 | 0.1 |

The input schema, kinds, training window/source IDs, normalized weights,
observation counts, source counts, weighted support, means, variances, scales and
mask are saved. Held-out unsupported observations and transform ranges are
recorded without changing the mask or excluding difficult clips. Input masking
never changes Y. Y keeps source-weighted training means, scales floored at 1e-8,
inverse transforms and a training variance mask `variance > 1e-10`. The scoring
mask is the intersection across outer folds. Historical scaler classes retain
their original methods and loading semantics.

## Source separation, controls and selection

Reuse the exact parent cohort, outer labels, cache arrays, teacher projection,
target, nuisance columns and control definitions. Keep five outer and three
inner source folds. All clips of a source stay together. Input/target scalers are
fit inside each training partition. Partition-local mismatch matching uses
context metadata, never targets; it is separate from prediction preprocessing.
The controls retain direct-v2's four-frame block shuffle, different-source donor
matching inside each current partition, and zero x/y/confidence with original
validity for no-skeleton. Shuffle moves all channels together. Both shuffle and
mismatch can change observation support as well as coordinates.

Inner losses are mean squared error over the same training-valid target
coordinates for all candidates in a partition, in that partition's target units.
Validation windows have weight `1 / clips_from_this_source`; pool weighted error
sums and source-weight totals before dividing. Do not average fold means.
Within `1e-10 + 1e-8 * max(abs(best_loss), abs(baseline_loss))`, prefer baseline.
Joint ties prefer larger RGB penalty, then larger skeleton penalty. RGB reference
ties use the same absolute/relative tolerance and prefer larger penalty.
Selection never reads outer-test losses. A selected addition can still lose on
outer sources; baseline eligibility gives no guarantee there.

Every candidate retains its parameters, three inner records, fitted-state train
loss, validation loss, pooled loss, improvement, status and selection reason.
A numerical failure rejects that candidate, is retained visibly, and makes the
scientific comparison incomplete even if a fallback is fitted. A baseline failure
aborts the fold. Interrupted unsealed folds can refit; completed folds are verified
and reused. Fold ownership is protected by the existing OS stage locks. A complete
negative measurement is different from an incomplete execution.

## Calibration and prospective seed amendment

Before the real run, exact fixtures check unsupported/near-constant/all-missing
columns, independent target units, serialization, zero correction, weighted ridge
equivalence, inferior candidate fallback, numerical failures and source/donor
boundaries. These fixtures also replay the parent inner 1/1 and 3/0 input defect.
Exact invariants use 1e-10 absolute/relative tolerance for numerical predictions
and exact equality for masks, schemas, fallback and fitted-statistic immutability.

The statistical calibration uses 80 synthetic sources with two windows each,
fixed source folds, the full production selection grid and all four controls.
RGB-only fixtures use seeds 1101–1108. Mean real-minus-RGB and mean
real-minus-no-skeleton across those fixed draws must each be at most 0.03 R²;
individual finite samples need not show zero gain. Planted skeleton and temporal
fixtures use seeds 2201 and 3301 respectively: real gain must exceed 0.10 and
matched increment must exceed 0.10. For temporal calibration, the designated
block shuffle must reduce real R² by at least 0.10. Fixture equations and all
observed scores are retained with the calibration implementation. Synthetic
positive controls validate the software; they cannot authorize ADVANCE.

This is a prospective change from stochastic seeds 7/19/31. A deterministic
solution is fitted once per fold; numeric seed 0 is solely a row identifier and
is labeled deterministic. Repeating it three times would supply no optimization
stability evidence. The stochastic three-seed conditions are **inapplicable**,
not passed. Retain every effect/control threshold and the paired source-bootstrap
stability rule. A passing development comparison requires independent-source
confirmation before any larger JEPA comparison is authorized.

## Scoring, verification and decision

For each target coordinate, predictive R² is one minus source-weighted prediction
squared error divided by the source-weighted error of the outer-training mean.
That reference is zero in training target units. Pool outer predictions first,
then average featurewise R²; never center on the held-out mean or average fold
scores. Keep one held-out prediction per window, arm and target coordinate.

Use the existing 2,000 paired source-bootstrap draws with seed 260905. Draw whole
sources with replacement, retaining all their clips and multiplicities. Align
draws across arms. Report 95% intervals separately from the positive-draw rule.
Intervals are conditional on saved models; they omit repeated training, selection
and the adaptive redesign of this inspected cohort. Bootstrap draws add no new
sources, and source separation does not prove participant separation.

| Criterion | Required result |
|---|---|
| Real gain over shared RGB ridge | at least +0.05 R² |
| Shuffle | real gain at least twice max(shuffle gain, 0) |
| Mismatch | gain at most +0.01 R² |
| Matched increment | real minus no-skeleton strictly positive |
| Paired source-bootstrap stability | real gain and matched increment each positive in at least 90% of draws |
| Measurement validity | complete cache, audit, split, preprocessing, candidate, checkpoint, prediction and numerical evidence |

Invalid/missing evidence produces incomplete STOP without a predictive claim.
Complete failed point criteria produce scientific STOP. Passed point criteria
with failed bootstrap stability produce INCONCLUSIVE. Passing everything is
ADVANCE **as development evidence**, with independent-source confirmation as the
next step. No S-JEPA training, adapters or distillation are launched or authorized.

A read-only numerical verifier reloads typed models, reconstructs predictions,
checks training target statistics and the shared baseline, and recomputes
featurewise/aggregate scores and every bootstrap row. Float64 prediction tolerance
is 1e-10 absolute/relative; score/interval tolerance is 1e-10 relative, 1e-12
absolute. Identities and masks require exact equality. Hashes alone cannot excuse
wrong units, duplicated rows, invalid checkpoint types or incorrect arithmetic.

## Cache inheritance and compatibility

The new root must be a sibling, not a descendant, of its parent. Before fitting,
snapshot every parent file's SHA-256, size and nanosecond modification time,
including historical executed notebooks in a separate preservation snapshot.
Check copied index basenames, per-window receipts, original embedded bindings,
array schema/shape/finiteness, original run/config/cohort/projection/audit hashes
and readiness arithmetic. Resolve absolute HAIC cache entries inside the supplied
parent root. Do not rewrite the index, manifests or embedded bindings. Record all
lineage and new provenance in the child. Raw frames and teacher weights are not
opened. Teacher encoding and pixel QC are explicitly reused, not rerun.

Changes to cohort, pose processing, nuisance definitions, temporal boundaries,
projection or teacher invalidate the corresponding derived cache evidence and
require a separate cache/audit plan. This repair changes only prediction-side
features, scaling and fitting. `legacy-v1` and `direct-v2` remain explicit loading
paths. Later scientific amendments require a separate run and versioned record.
