# Additional real-data evidence

The 512-head-update experiment remains primary. The 4,096-update experiment was motivated by normal-validation head adequacy before the new outer-test scores were inspected. It uses the same frozen source roles, masks and 512-update pretrained backbones. Neither budget establishes a learned-encoder advantage.

| Method | Primary512 | Sensitivity4096 |
| --- | --- | --- |
| Latent benefit profile | 0.641 | 0.667 |
| Raw full latent error | 0.541 | 0.529 |
| Full-error reference distance | 0.673 | 0.666 |
| Own-error reference distance | 0.657 | 0.590 |
| Random encoder, latent heads | 0.651 | 0.714 |
| Coordinate benefit profile | 0.686 | 0.690 |
| Random encoder, coordinate heads | 0.701 | 0.693 |
| Ridge coordinate profile | 0.607 | 0.607 |
| Latent profile, four patches | 0.542 | 0.644 |
| Reference-free contrast | 0.527 | 0.398 |

The primary latent profile minus raw error difference is 0.101, with a conditional 95% interval from -0.035 to 0.224. The corresponding longer-budget difference is 0.138, with a conditional 95% interval from -0.008 to 0.280. Both intervals include zero. The random encoder's prediction heads are fitted in both budgets.

## Normal validation

| head_updates | fold | own_coordinate | full_coordinate | own_latent | full_latent | effective_rank | embedding_std |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 512 | 0 | 0.023 | 0.057 | 1.433 | 1.208 | 8.393 | 0.537 |
| 512 | 1 | 0.044 | 0.063 | 1.756 | 1.466 | 8.145 | 0.570 |
| 512 | 2 | 0.043 | 0.058 | 1.057 | 0.692 | 5.193 | 0.488 |
| 512 | 3 | 0.046 | 0.074 | 1.333 | 0.856 | 5.560 | 0.564 |
| 4096 | 0 | 0.018 | 0.021 | 1.325 | 1.223 | 8.393 | 0.537 |
| 4096 | 1 | 0.038 | 0.040 | 1.788 | 1.165 | 8.145 | 0.570 |
| 4096 | 2 | 0.036 | 0.034 | 0.851 | 0.517 | 5.193 | 0.488 |
| 4096 | 3 | 0.040 | 0.047 | 1.182 | 0.740 | 5.560 | 0.564 |

Coordinate full-context MSE is above own-context MSE in all four primary folds and three of four longer-budget folds. Fold 2 reverses that ordering after longer training. This remaining learner-adequacy limitation is not evidence that additional body information is intrinsically harmful.

## Development probes

| target_type | intervention | n_sources | median_paired_change | minimum | maximum |
| --- | --- | --- | --- | --- | --- |
| coordinate | relative_limb_shift | 5 | -0.001 | -0.145 | 0.060 |
| coordinate | whole_body_shift | 5 | 0.765 | -0.231 | 12.005 |
| coordinate | coordinate_noise | 5 | 0.007 | -0.023 | 0.273 |
| coordinate | observation_dropout | 5 | 0.049 | -0.171 | 0.304 |
| latent | relative_limb_shift | 5 | -0.014 | -0.041 | 0.279 |
| latent | whole_body_shift | 5 | -0.037 | -0.192 | 0.118 |
| latent | coordinate_noise | 5 | 0.004 | -0.026 | 0.065 |
| latent | observation_dropout | 5 | 0.068 | -0.050 | 0.768 |

Twelve fitting-normal sources supplied long windows; five remain scorable after common support across every edit, including dropout. These are signed paired score changes on that small development set. They do not validate a neurological mechanism or held-out generalization. Geometry reports separately measure damage at limb-to-torso connections.

For the contextual-teacher diagnostic, 12 fitting-normal sources contribute 87 source-joint summaries. The own-predictor-only change is exactly zero when its local history is unchanged. The median across sources of the mean absolute teacher-only own-KL change is 0.0351. This verifies that a latent target can change because other joints changed, even when the local joint and its own-context prediction are fixed. It is an expected property of contextual targets, not a neural mechanism.

## Traceability

The [manuscript evidence file](draft01/evidence.json) records source artifact hashes. The primary results remain in [context_study_results.md](context_study_results.md). The longer-budget and nuisance comparisons are supporting analyses, not replacements selected for a favorable outcome.
