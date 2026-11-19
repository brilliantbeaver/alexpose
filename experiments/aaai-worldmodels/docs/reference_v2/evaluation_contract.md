# Evaluation contract for reference-aware gait study v2

This contract was prepared before new outer-fold model outcomes were inspected. It implements the evaluation requirements in the research instruction and the coordinator's `protocol.md`. The study remains exploratory because earlier GAVD results were already known. No new anomaly performance was used to select a family, score direction, calibration design, or comparison.

## Populations and prediction support

The frozen primary candidate is the compact posture-plus-velocity ridge family. Forecast regularization is chosen on permitted normal training groups only, from the fixed grid. The neural masking comparison is a separate forecast experiment and does not select the primary anomaly detector.

The new cohort adapter preserves 2,815 common-support windows from 91 source videos, including 29 normal sources. Its verified source-relation components are the minimum grouping unit. All current primary components contain one source, but evaluation code averages scores across linked source videos before calculating component-level endpoints if verified links are added under a new cohort identity. A component must have one reconciled annotation and outer fold. Different class labels or folds inside one component cause an error rather than a silent reassignment.

Every forecast is scored against exactly the same observed future coordinates. Coordinate error is averaged within each leg, legs receive equal weight in a window, windows receive equal weight within a sequence, sequences receive equal weight within a source, and linked sources receive equal weight within a component. The primary forecast horizon averages all eight future samples, nominally 0.1 through 0.8 seconds. The four-step horizon is secondary. Missing targets do not become measured zeros. Source covariates must be available for the common cohort; absent motion support causes an explicit error rather than a hidden change in the analysis population.

## Prefix covariates

`window_covariates` receives measurements and support plus actual decoder timestamps. It uses only the first sixteen samples. For each distal joint, it divides adjacent observed displacement magnitude by the measured time difference. It averages those rates within each leg and then gives the legs equal weight. A missing adjacent pair is excluded rather than assigned zero velocity. A leg with no measured adjacent pair is invalid for this proposed reference model and requires an explicit cohort amendment.

Prefix missingness is the fraction of the sixteen declared joints that was unobserved over the sixteen prefix samples. Both quantities follow the sequence/source hierarchy above. The nuisance design contains exactly `log(prefix_motion + 1e-8)` and prefix missingness. These are relative 2D image-articulation quantities; they are not measured physical walking speed. Changing any future coordinate, future visibility, or future timestamp cannot change these covariates.

## Disjoint calibration rotations

For outer fold `k`, let `T` be the normal source components outside `k`. `calibration_schedule` deterministically partitions `T` into three calibration groups using shuffled K-fold partitioning with seed `4701 + k`. For each rotation, one group is `C` and its complement is `R`. The outer evaluation groups, `R`, and `C` are disjoint.

In folds zero through three, `T` contains 23 normal sources. Its rotations have `(R, C)` sizes `(15, 8)`, `(15, 8)`, and `(16, 7)`. The final outer fold has 24 normal training sources and all three rotations have sizes `(16, 8)`. These counts were checked from the cohort identities, without fitting new models.

Each rotation follows this sequence.

1. Split `R` into three source-held-out nuisance-reference folds, using seed `4801 + 10*k + rotation`. Fit a forecast for each nuisance-reference fold using its permitted normal training subset. Select its forecast regularization only inside that subset. Predict its held-out `R` sources and aggregate one MSE per source.
2. Fit a small ridge regression `g` on those out-of-fit `R` source rows. Its response is log source MSE; its inputs are the two prefix nuisance covariates. Select `g`'s ridge strength from `{0.1, 1, 10, 100}` by leave-component-out error within `R`, refitting feature standardization in every leave-out fit. Exact ties prefer larger regularization. Refit `g` on all permitted `R` reference rows.
3. Select the final forecast's regularization using only `R`, then refit it on all `R`. Apply that identical forecast and `g` to every source in `C` and the outer evaluation fold. No `C` source may enter any forecast fit, intermediate reference prediction, scaling operation, or nuisance fit contributing to this rotation.
4. Calculate each source's scalar residual `r = log(MSE + 1e-8) - g(log(prefix_motion + 1e-8), prefix_missingness)`. Use `C` residuals as the reference distribution. If a component contains several sources, average their scalar residuals before constructing that component's reference contribution.
5. Compute the endpoint-smoothed empirical mid-CDF `F_C(r) = (0.5 + count(reference < r) + 0.5*count(reference == r)) / (n_C + 1)`. The two-sided score is `abs(2*F_C(r) - 1)`. A large value indicates a source that was unusually easy or unusually difficult to predict relative to the permitted normal reference. The midpoint gives zero, and endpoint smoothing keeps the score below one.
6. Average the three rotation scores equally for each held-out source. Every source must be present in every rotation. No rotation, direction, or tail is selected after examining abnormal labels.

`fit_source_calibrator` validates source membership and per-row forecast provenance. Every out-of-fit `R` row must name the exact training sources from its nuisance-reference fold. Every `C` row must name exactly `R` as its forecast training pool. Missing, duplicate, or conflicting provenance causes refusal. The caller must retain actual fitted checkpoint memberships and hashes, so a recorded list is not mistaken for proof that an arbitrary fit obeyed it.

Each calibrator retains the complete nuisance-reference and calibration source rows for numerical replay. The nuisance model also records its two covariate ranges and log-error response range, calculated only from its permitted normal reference rows. These are diagnostics and do not clip, rescale, or alter a prediction. External reports must show how many observations fall outside those training ranges and how often empirical reference ranks reach their endpoints. Such counts characterize domain shift; they do not justify choosing a new transform after outcomes are seen.

This score uses reduced-training predictors. Separate core forecasting models may use all 23 or 24 permitted normal outer-training sources; they cannot replace the reduced-training predictors while retaining the same calibration reference. A model fit on all `T` would have seen `C` and invalidate the intended isolation.

The nuisance-response forecasts are trained on 10 or 11 normal sources, while the refitted predictor uses 15 or 16. This creates a training-size shift in the nuisance reference. Calibration and evaluation nevertheless share the same final forecast and `g`; the normal calibration residuals expose their common bias. This is a practical exploratory reference, not a proof that conditioning has removed nuisance effects. Seven or eight reference sources also produce coarse, partly tied ranks. There is no conformal coverage, clinical operating-point, or finite-sample false-alarm guarantee.

## Endpoints and uncertainty

The primary anomaly comparison is the calibrated two-sided score versus the locked negative-persistence-MSE score. A larger value indicates abnormality for both. For each outer fold, AUROC is the fraction of abnormal-normal component pairs with the higher score on the abnormal component, awarding half credit for ties. Average fold AUROCs with weights equal to the number of abnormal-normal pairs. Compute the difference between candidate and reference under that same weighting. This excludes every cross-fold pair.

`paired_within_fold_bootstrap` resamples independent components with replacement within each outer fold and binary-label stratum. The two methods receive identical bootstrap multiplicities. The implementation uses pair-comparison matrices and multinomial component counts; it does not treat pair entries or repeated windows as independent. The main runner specifies 5,000 draws and seed 5701. The resulting interval conditions on the saved fitted models. It omits retraining variation, model-development uncertainty, and unknown participant overlap across uploads. Fold means, seeds, and windows do not multiply the independent participant count.

Also report every fold's class counts and AUROC, equal-fold averaging, pooled calibrated ranking, and average precision with prevalence. Pooled raw-score comparisons require the additional assumption that separate fold models produce comparable score scales. Condition-specific slices share the same normal references and are descriptive. Their counts and uncertainty must be visible. No confirmatory p-value or multiplicity-adjusted clinical claim is produced by this exploratory evaluation.

`normal_forecast_contrast` averages normal-component forecast errors and calculates the paired difference `reference MSE - method MSE`, with a positive value favoring the method. It also reports the relative reduction from the reference mean and its conditional percentile interval. The main forecast uncertainty uses 5,000 draws and seed 5702. Optimization seeds are shown individually and, where prespecified, their predictions are averaged before scoring; their count is not added to the source count.

## Nuisance and sensitivity controls

The required reference score is negative persistence error. Secondary controls retain their declared directions: raw forecast MSE, excess MSE above persistence, high prefix missingness, prefix motion with its stated direction, and normal-only distances from current posture or prefix nuisance features. `fit_normal_distance` standardizes permitted normal source features and uses a covariance with a fixed 20% identity shrinkage. It does not select a distance transform or shrinkage by abnormal ranking. All reference fits save their source memberships.

Normal-calibrated marginal and one-sided scores are secondary, not replacement primaries selected after outcomes. View strata, motion matching, and observation-quality strata must state their sample support, retained population, and selection rules. Results in a sparse or changed population cannot be substituted for the main endpoint. A supervised nuisance-only classifier, if included, must be declared label-assisted and fully nested separately.

External-dataset transfer must keep verified participants and repeated views together. A frozen GAVD calibration component scores external source errors without using external labels or normal examples for adaptation. The same stored `g` and empirical reference are used. A separate external compatibility contract must define video-to-participant aggregation and the ensemble over fifteen GAVD outer-fold/rotation components. External clinical sensitivity is not inferred from a normal-only external set. A force-domain study would test a separate adapted method rather than validate the image pipeline.

## Checks actually executed before release

The focused evaluation tests validate deterministic source schedules and related-component isolation, exact out-of-fit row provenance, future/missing-value invariance of prefix covariates, actual-time scaling, source/sequence/component aggregation, the nuisance ridge against an independent scikit-learn solver, empirical-CDF ties, component score ensembles, within-fold AUROC without cross-fold pairs, paired uncertainty, forecast contrasts, and normal-only distance behavior. No new outer-fold model was fitted for these tests. Synthetic test data are software evidence only.

The real-cohort prefix check found 91 finite source covariate rows. Prefix motion ranges from approximately 0.365 to 3.963 normalized articulation units per second, and source-average missingness ranges from zero to approximately 0.252. These are input-description checks and cannot establish disease association or model performance.
