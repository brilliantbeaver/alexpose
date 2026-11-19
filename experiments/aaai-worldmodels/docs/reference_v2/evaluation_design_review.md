# Independent evaluation design review before outer fitting

The reviewer is the data specialist. The reviewer authored `v2_data.py` and its cohort audit, but did not author the reviewed evaluation implementation, evaluation tests, coordinator protocol, or evaluation contract. This is an independent review of the latter artifacts only, not independent acceptance of the reviewer's own cohort implementation. No new outer-fold forecasting outcome or anomaly ranking was inspected during this review.

## Disposition

The evaluation design is accepted for the internal exploratory run at the exact identities listed below. The frozen primary model family is posture-plus-velocity ridge, and the primary score is a normal-reference two-sided residual score compared with negative persistence MSE. The neural masking study does not select this detector. Calibration rotations retain separate predictor-fit R, calibration C, and outer evaluation roles, with normal-only nested selection. The held-out C sources never enter the nuisance or forecasting fits that construct their reference distribution.

Every source's error is aggregated under common coordinate/leg/window/sequence/source support. The residual is explicitly a scalar log source MSE after adjustment for log prefix motion and prefix missingness. The smoothed mid-CDF and its two tails are completely defined. Bootstrap pairing operates at verified source-relation components within outer fold and binary-label cells, excluding cross-fold ranking pairs. Current singleton components retain the historical folds and 29 normal source groups.

## Finding resolved before release

**E1: related-source aliases could bypass a scoring guard.** The initial scoring API checked only source IDs against the calibration/predictor pools. A differently named upload sharing a known component could therefore bypass the guard. This did not affect the current 91-singleton cohort, but it violated the API's intended grouping contract. The evaluation owner added a query-component overlap check against both R and C and a synthetic alias test. The owner also requires exact R forecast provenance when a query provides the membership field. The reviewer inspected the correction and reran the tests.

## Checks actually executed

The reviewer independently ran:

```bash
experiments/aaai-worldmodels/.venv/bin/python -m pytest -c experiments/aaai-worldmodels/pyproject.toml experiments/aaai-worldmodels/tests/test_v2_evaluation.py -q
```

The final recheck returned 12 passed. The earlier measured run completed in 1.42 seconds. Tests include grouped schedule isolation, exact out-of-fit forecast provenance, actual-time and future-value invariance, the nuisance ridge against an independent scikit-learn solution, mid-CDF ties and symmetry, equal rotation averaging, explicit within-fold pairs, paired bootstrap arithmetic, source/component hierarchy, and normal-distance controls. These are software checks on synthetic fixtures, not model-performance findings.

A separate real-input check constructed all 91 finite source covariate rows from the frozen cohort. The reviewer independently calculated the first real window's observed adjacent-pair motion rates with explicit joint/time loops and obtained the same value as the vectorized API. All 15 real calibration schedules passed R/C/outer isolation. In outer folds 0–3, normal training sources number 23 and the rotations use R/C sizes 15/8, 15/8, 16/7. In outer fold 4, training sources number 24 and every rotation uses 16/8. No forecast model was fitted for these checks.

The final inspected implementation also stores nuisance-training covariate ranges and response ranges, and retains complete out-of-fit reference rows for replay. Those are diagnostic/provenance additions and do not change the scoring equations.

## Limits and subsequent checks

The out-of-fit forecasts used to train the nuisance model have fewer training sources than the refitted R predictor. The contract states this training-size shift. C and outer evaluations nevertheless share the same final predictor and nuisance model. Small C groups produce coarse ranks, and averaging rotations does not create independent patients or a conformal guarantee. The bootstrap conditions on saved predictions and omits retraining and uncertain participant overlap. These limitations must accompany the results.

Caller-provided provenance lists are checked for agreement with the schedule, but a list is not proof that the fitted predictor obeyed it. The full-run implementation review must replay actual fit memberships, scaling and hyperparameter selection before accepting efficacy outputs. Normal-only static-posture and acquisition controls also depend on the caller fitting them inside the permitted source pool.

Before external scoring, freeze an explicit video-to-participant aggregation order and the 15-component ensemble rule. Averaging source errors before the nonlinear tail transform is generally different from averaging per-video tail scores. A separate external contract must identify which is intended and its changed measurement unit. Actual invalid external files and pose/tracking quality must remain visible. This condition affects external release; it does not prevent the defined internal comparison.

## Reviewed identities

- `worldmodels/v2_evaluation.py`: `f01d2ebd1087fdd910f79b2442ca6a1c1325739c7e4dc0e9b35d496ce301ebff`.
- `tests/test_v2_evaluation.py`: `7930b232a840fc10f2e64248f3fdfd13945f64a77d5f8857abf73c622b911cf5`.
- `docs/reference_v2/protocol.md`: `02591980f6e4e535a08bcc631fa155d0459100e5136f588ad2a5f886cd685103`.
- `docs/reference_v2/evaluation_contract.md`: `df0f147f0a5b24d5785fac5bb695f77812d4f3e56fa598e7942c2edd768f798c`.
