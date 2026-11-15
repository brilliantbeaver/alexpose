# Descriptive score-calibration audit

This audit uses the saved outer-test normal scores from every fold of the completed GAVD U/U and external angle studies. It was requested after outcomes were known. It does not fit models, change reference groups, calibrate scores, choose a favorable fold, or revise the original anomaly result. The [manifest](audit_manifest.json) identifies the producer and every input by SHA256.

The main finding is a difference in score distributions, especially their tails. It does not establish why fusion reduced anomaly performance. M and R measure different distances and are not probabilities. Dividing each distance by its dimension does not guarantee comparable distributions on new normal examples.

## All outer-test normal folds

The table shows empirical quantiles. Every GAVD row uses seven normal recording sources. Every external row uses 32 healthy participants. No row used the relational fallback. With these sample sizes, especially seven sources, p99 is close to the observed maximum and is not a precise population-tail estimate.

| Dataset | Fold | Normal units | M median | M p90 | M p99 | R median | R p90 | R p99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GAVD U/U | 0 | 7 | 1.152 | 2.110 | 2.539 | 0.548 | 0.943 | 1.351 |
| GAVD U/U | 1 | 7 | 0.620 | 2.324 | 2.326 | 5.804 | 13.263 | 19.114 |
| GAVD U/U | 2 | 7 | 0.858 | 1.765 | 2.219 | 2.191 | 5.341 | 7.346 |
| GAVD U/U | 3 | 7 | 0.783 | 2.146 | 2.267 | 0.984 | 5.216 | 8.148 |
| External angles | 0 | 32 | 0.719 | 1.790 | 3.027 | 0.310 | 1.596 | 16.934 |
| External angles | 1 | 32 | 1.003 | 2.077 | 3.071 | 0.154 | 0.715 | 6.299 |
| External angles | 2 | 32 | 0.940 | 1.743 | 2.452 | 0.720 | 2.791 | 7.170 |
| External angles | 3 | 32 | 0.686 | 1.401 | 3.986 | 0.795 | 1.946 | 33.939 |

External R has a lower median than M in three folds but a higher p99 in every fold. GAVD R is smaller on all three displayed quantiles in fold 0 and larger in folds 1 through 3. Therefore, neither “R always dominates M” nor “the scores have comparable scales” describes all observed folds. The [complete quantile table](normal_score_quantiles.csv) also retains R_raw and each maximum.

## Reference dependence

External R uses separate shrunken variances for left and right benefit, without a covariance term. The twenty reference normal participants in each fold show positive left/right associations.

| Fold | Reference normal participants | Pearson correlation | Spearman correlation |
| --- | ---: | ---: | ---: |
| 0 | 20 | 0.823 | 0.379 |
| 1 | 20 | 0.767 | 0.414 |
| 2 | 20 | 0.572 | 0.559 |
| 3 | 20 | 0.544 | 0.550 |

These statistics describe the actual reference profiles, rather than a hypothetical mismatch. They do not prove that a full-covariance reference or a different fusion would improve the held-out result. Pearson correlations are stronger than rank correlations in the first two folds, so a few large profile values may affect the linear association. The [reference table](external_reference_correlations.csv) includes means, variances, covariance and fitted variances for inspection.

## Native prediction errors

The following means weight the six angles equally within each participant and then weight participants equally. Values are squared degrees. Each row uses only the indicated outer-test group, not fitting or reference participants.

| Fold | Group | Participants | Baseline f | Own augmentation f+h | Both-leg augmentation f+g |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | Healthy | 32 | 204.295 | 199.995 | 188.950 |
| 0 | Stroke | 13 | 131.916 | 126.580 | 136.570 |
| 1 | Healthy | 32 | 213.246 | 206.398 | 195.217 |
| 1 | Stroke | 13 | 139.703 | 132.436 | 134.835 |
| 2 | Healthy | 32 | 198.880 | 193.485 | 183.650 |
| 2 | Stroke | 12 | 127.945 | 117.425 | 117.388 |
| 3 | Healthy | 32 | 208.627 | 204.273 | 194.517 |
| 3 | Stroke | 12 | 148.672 | 134.467 | 156.195 |

The stroke group's absolute completion error is lower than the healthy group's for every displayed predictor and fold. This is an observed group difference, not evidence that stroke gait is intrinsically easier or more coordinated. Motion amplitude, trial selection, preprocessing, clinical heterogeneity and the fitted predictor family can affect squared error. The relational score measures departure from reference benefit and is not simply raw reconstruction error. [Full participant-balanced summaries](external_heldout_mse.csv) include medians as well as means.

The defensible manuscript statement is that normal score scales, tails and reference dependencies differ, while the completed fixed fusion performs worse than M. A causal explanation of that loss would require a separately specified calibration experiment. This audit supplies no such experiment and makes no claim that recalibration repairs the result.
