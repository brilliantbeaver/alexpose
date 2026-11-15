# Normal-reference accounting and scale audit

This audit addresses H1, M1 and M4 in the final Claude review. It uses source-role records, frozen normal-reference parameters and the already saved fitting-normal development probes. No abnormal ranking results were used, and no reference parameter, score, threshold or observation was changed.

## H1: exact reference reuse

Within every outer fold, normal fitting, normal reference and outer-test sources are disjoint. The counts are 12/5/6 for folds 0–2 and 13/5/5 for fold 3. Across folds, the 20 reference assignments use 15 distinct normal source IDs: 10 sources are referenced once and 5 twice. The remaining 8 normal sources are never references. Each normal source is an outer-test source exactly once.

| Fold pair | Shared reference sources |
| --- | ---: |
| 0, 1 | 2 |
| 0, 2 | 1 |
| 0, 3 | 1 |
| 1, 2 | 1 |
| 1, 3 | 0 |
| 2, 3 | 0 |

The critic's claim that five references per fold must necessarily overlap heavily is incorrect. The actual overlaps are modest and explicitly measurable. The valid concern is that normal fitting and reference sources are reused across models, so the folds are not independent replications. The current fixed-fit bootstrap does not propagate uncertainty from refitting those overlapping training/reference sets. This is distinct from within-fold source leakage, which the role records exclude.

Suggested manuscript sentence: “Normal fitting, reference and test sources are disjoint within each fold; across folds, 20 reference assignments use 15 distinct normals, with pairwise reference overlaps of 0–2, so fold estimates are not independent replications.”

Exact identities and counts are in [normal_source_accounting.csv](../normal_source_accounting.csv) and [normal_reference_overlap.csv](../normal_reference_overlap.csv).

## M1: reference scale and floor activity

The audit checked all frozen normal references for both budgets, all three model families, both observation budgets and every reference field. This gives 960 regional variance denominators. None is affected by the `1e-8` floor. The smallest pre-floor shrunken variance is `1.3871711954101562e-5`, approximately 1,387 times the floor.

For the trained model's all-support benefit profile, the ranges below span the 12 fold/region entries in each row. Means describe the reference benefit, not individual-source extrema.

| Head budget | Target | Reference mean range | Sample variance range | Shrunken variance range | Floor bindings |
| --- | --- | --- | --- | --- | ---: |
| 512 | Coordinate | −0.0916703 to −0.0006340 | 1.62903e−5 to 0.00605884 | 8.21763e−5 to 0.00404840 | 0 / 12 |
| 512 | Latent | −0.223260 to 1.10556 | 0.0461392 to 0.362444 | 0.0628448 to 0.288577 | 0 / 12 |
| 4,096 | Coordinate | −0.0396620 to 0.0118185 | 5.53797e−6 to 0.00153688 | 1.38717e−5 to 0.00108843 | 0 / 12 |
| 4,096 | Latent | 0.205924 to 0.875577 | 0.0179735 to 0.981894 | 0.0448062 to 0.818940 | 0 / 12 |

The fixed 0.5 shrinkage remains a modeling choice, and five references give only four degrees of freedom for each sample variance. An inactive numerical floor does not remove reference uncertainty or validate the shrinkage strength. The leave-one-reference-out results cannot establish that shrinkage is the dominant cause of instability without a separate comparison. No shrinkage sensitivity was tuned or substituted after inspection.

The full normal-only parameter audit is [normal_reference_scales.csv](../normal_reference_scales.csv), with source hashes and grouped summaries in [normal_reference_audit.json](../normal_reference_audit.json).

## M4: temporal shift and score units

The critic confuses a temporal shift with a spatial translation and a standardized profile difference with coordinate MSE.

`normalize_control_windows` freezes per-frame pelvis centering and a single torso scale computed over long-window frames `4:20`, the first 16 frames of the original crop. `paired_edits` uses `x[:, 4:36]` for the original and `x[:, 8:40]` for `whole_body_shift`. This advances the complete body by four frames, or approximately 0.267 seconds at 15 Hz. It does not add a constant spatial displacement. Every fixed-index coordinate target now contains later measurements, and the contextual latent target also changes. Pelvis centering does not remove changes in gait phase or target measurements.

The reported coordinate median `+0.7653677` and maximum `+12.0052975` are changes in the frozen-reference standardized squared regional-benefit distance. They are not raw coordinate prediction errors. Recomputing that distance directly from each saved profile and the unchanged reference reproduces the stored scores. For the same five scorable normal fitting sources, the median change in raw full-context coordinate MSE is `+0.00660562`, and its maximum is `+0.0318891`.

The objection therefore does not identify a normalization or leakage bug. It does not justify capping or winsorizing the observation. The legitimate limitations remain the small reused development sample, changing temporal content, and the absence of a held-out biological-selectivity result. The JSON audit records the code hashes, normalization anchor, target change, score interpretation and direct score-recomputation check.
