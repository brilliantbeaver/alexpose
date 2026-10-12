# Version 08 evidence audit

This analysis uses the completed 14-person, three-seed development panel. All new penalty sensitivities, condition summaries, and figure intervals are exploratory and unadjusted. Original primary endpoints and the 720-degree scoring rule are retained.

## What the compact exports support

Not reconstructable from the compact exports: signed reference_change in response-curves-person.csv is already averaged across source windows/motions, naming, extractor and seeds. abs(mean reference_change) is not mean(abs(reference_change)). No per-pair magnitude-bin analysis is manufactured.

Report all clear/occluded by seen (5 and 10) or held (15) intervention groups, each with its saved zero-response MAE. Nominal edits are never labeled true response magnitudes.

## Failure accounting

The person-balanced rates differ from raw pair fractions because the published scores average within windows, motions, and people. Conditional errors are reported separately from unconditional contributions.

| Method | Failure rate (%) | Successful contribution (degrees) | Failure contribution at 720 (degrees) | Total (degrees) | Conditional error under original weights (degrees) |
|---|---:|---:|---:|---:|---:|
| F-response-jepa_delta_v1-graph_time-paired_change | 0.524783 | 6.209596 | 3.778439 | 9.988035 | 6.242355 |
| F-response-jepa_endpoint_v1-graph_time-paired_change | 0.563210 | 6.305974 | 4.055115 | 10.361088 | 6.341691 |
| P-direct-none-base | 0.343762 | 5.069067 | 2.475088 | 7.544155 | 5.086552 |

E_w[absolute error times successful indicator] / P_w(success). Conditioning can change population weights; this is not the independently averaged per-family conditional export.

Existing exporter averages successful errors within each source family before the motion/person hierarchy. Retained separately; it is not additive with the failure contribution.

## Sensitivity without changing the primary

Interior projected angles are in [0,180], per-leg excursion in [0,180], A in [-180,180], response in [-360,360], and a valid prediction's absolute response error is bounded by 720 degrees.

| Failure cost (degrees) | Delta-versus-endpoint improvement (degrees) | Crossed 95% interval | Learned methods below zero-response mean |
|---|---:|---|---:|
| 0 | 0.096378 | [-0.022562, 0.239265] | 2 |
| 180 | 0.165547 | [-0.176717, 0.526907] | 1 |
| 360 | 0.234716 | [-0.472937, 0.941314] | 0 |
| 720 | 0.373054 | [-1.110272, 1.760337] | 0 |

- Cost 0: Accounting lower bound assigning zero error to failures; not a useful operating score and not a conditional-success mean.
- Cost 180: Supplementary quarter-cost stress test; 180 is an angle-error bound, not a response-error bound.
- Cost 360: Half-cost stress test and worst-case zero-response error; not the maximum error of a valid restored response.
- Cost 720: Inherited canonical maximum response-error bound; original declared analysis unchanged.

The same predictions and failure indicators are rescored; no alternative cost is selected as preferable. A zero cost assigns failed cases a free score and must not be interpreted as success-only accuracy. All intervals in the delta/endpoint sensitivity remain descriptive.

## Version 07 accuracy assessment

The main v07 numerical claims reproduce. Its geometric assignment is post hoc, its primary response interval remains unresolved, and its 93% repair ratio and 74% failure fraction are descriptive. The 0.0013% versus 10% initialization gradient influence is verified from the calibration receipt. Important improvements achievable in v08 are to expose rates separately from contributions, make the zero-response benchmark central, show all supported observation strata, and separate jointly trained direct models from frozen readouts. These do not supply independent confirmation.

## Limits requiring new artifacts or experiments

Per-pair true response magnitudes and wrong/ambiguous/missing assignment components are absent from the compact packet. Raw predictions, learning curves, and feature arrays are unavailable locally. Fourteen repeatedly inspected people and three fitted seeds do not become additional evidence through new strata or bootstrap draws. External methods were not fitted; independent population, natural-observation, anatomical, and clinical validation remain future work.

## Reproduction

Run `.venv/bin/python docs/iclr/versions/v08/scripts/analyze_evidence.py`. The script modifies only version-08 evidence files, verifies the complete person/seed panels, checks additive decompositions and exact condition reaggregation, and reconstructs all 16 saved comparison entries. CSV intervals label their estimand and procedure. Source hashes and definitions are in `provenance.json`; the claim-to-artifact map is `claim-to-artifact.json`.

## An informative lower-bound distinction

Delta and endpoint have unconditional successful contributions of 6.209596 and 6.305974 degrees, both exceeding the zero-response mean of 5.810830 degrees. Since failure rates are nonnegative, S(C) is at least this successful contribution for every nonnegative cost C. Thus no nonnegative failure cost reverses their observed pooled ranking against zero response. This is an accounting lower-bound argument, not a comparison of conditional successful accuracy. Direct differs: its successful contribution is 5.069067 degrees and its ranking against zero does depend on cost. The argument concerns these development means, not a universal claim about the representation families.


## Framing-revision additions

The original estimates and source exports are unchanged. The new participant
illustration is bound by case-figure-provenance.json, case-figure-seeds.csv and
case-figure-person.csv. It retains all 126 person/seed/panel values, all 14 people,
and the original failure cost and condition weights. framing-claims.json maps
the new clinical/context, tensor, split, notebook and null-formalization claims
to primary sources and audited implementation evidence. No new training,
clinical labels, future-state evaluation or independent participants were added.
