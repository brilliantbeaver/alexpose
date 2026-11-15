# Final implementation and artifact audit

Audit date: 15 November 2026. The primary run is `3044566091261819`; the longer-head sensitivity is `eba2bd5a6e05efa1`. This review compared the methods draft, implementation, checkpoint tensors, training histories, source roles, frozen support selections and manuscript evidence. It did not modify the manuscript or any module included in the run identity.

No implementation blocker was found under the declared observational completion protocol. This conclusion concerns consistency of the implemented experiment. It does not establish convergence, useful neurological representations, participant independence or clinical validity.

## Direct checkpoint comparison across training budgets

The comparison loaded both `trained.pt` and `untrained.pt` for all four folds with `torch.load(..., weights_only=True)`. It compared every saved `backbone.*` tensor using exact `torch.equal`, including the EMA target encoder and target center. This is stronger evidence than matching random seeds or configuration fields.

| Fold | Backbone | Equal backbone tensors | Equal teacher/center tensors | Changed head tensors | Shared backbone SHA-256 prefix |
| --- | --- | ---: | ---: | ---: | --- |
| 0 | Trained | 84 / 84 | 31 / 31 | 66 / 66 | `c3c3398944fcfca1` |
| 0 | Random | 84 / 84 | 31 / 31 | 66 / 66 | `4c9f977eb3f7d160` |
| 1 | Trained | 84 / 84 | 31 / 31 | 66 / 66 | `9e1d573e0403cc29` |
| 1 | Random | 84 / 84 | 31 / 31 | 66 / 66 | `797f9a989682b776` |
| 2 | Trained | 84 / 84 | 31 / 31 | 66 / 66 | `fc2334b101c9d960` |
| 2 | Random | 84 / 84 | 31 / 31 | 66 / 66 | `2949b4d3ac79eb9f` |
| 3 | Trained | 84 / 84 | 31 / 31 | 66 / 66 | `51cc7a38a44766e5` |
| 3 | Random | 84 / 84 | 31 / 31 | 66 / 66 | `c180892e08b590bf` |

Full hashes are retained in the corresponding checkpoint provenance. The pretraining histories match exactly between budgets for all eight fold/backbone pairs. The primary prediction-head history also equals the longer run's entire history prefix through update 512, including its diagnostics. The implementation uses separate fixed random streams for head-window sampling and mask/joint sampling. The longer run therefore extends the same fitting trajectory rather than changing the representation or the initial 512-update schedule.

The source-fold registry, retained window index, profile-eligibility CSV/JSON and every method's four-patch selection are identical between budgets. Both run configurations use 512 pretraining updates, batch size 16, width 64, two encoder blocks, four attention heads, learning rate 0.0003, weight decay 0.05 and EMA momentum 0.996. Only the head-update budget changes from 512 to 4,096.

## Target, context and loss semantics

The implementation matches the declared target distribution:

\[
q=\operatorname{softmax}((z^T-c)/0.06),\qquad
p=\operatorname{softmax}(u/0.1).
\]

`ContextBundle.targets` uses the frozen EMA teacher over all observed tokens in the complete window and the frozen target center. The pretraining objective uses cross-entropy plus 0.1 times VICReg. The six later heads have separate optimizers and use coordinate MSE or KL from teacher to prediction. Tiny negative KL values caused by floating-point arithmetic are clamped to zero. The output distribution is normalized over the embedding channels; it is not a distribution over health conditions.

Restricted context is encoded by the frozen online view encoder. Tokens are selected before its first attention layer, with invalid keys excluded at every layer. Own and patch heads do not receive globally contextualized memory. All heads use identical target identities and common eligibility. Mask sampling is uniform across the 128-token grid and does not inspect movement or validity. Eligibility then requires an observed target and at least two observed visible own-joint patches, without redrawing masks.

The measurements are already expressed in a shared pelvis/torso reference before context restriction. Consequently, the experiment isolates additional observations conditional on that reference, not all information derived from the rest of the body. Visible context can follow the hidden patch. These are completion experiments, not causal forecasting experiments.

The optional teacher decomposition is correctly limited to unchanged arm/right-leg readouts when the left leg is shifted. Its four factors separate changes in the fitted predictor and contextual target on common raw-KL support. Their additive identity does not imply an additive decomposition of the nonlinear regional anomaly score.

## Sampling and aggregation corrections

The methods draft now explicitly distinguishes the following operations.

1. Training samples a normal source uniformly and then a window within that source. A pretraining batch containing only one distinct window has its last draw replaced by a different window for the variance regularizer. Head fitting averages eligible target losses within each window and then averages active windows. Sampling is source-balanced in expectation; eligibility can still affect which draws contribute.
2. Head diagnostics reuse four fixed source-sampled minibatches with fixed random joint/mask draws. They are sampled window-level diagnostics, not complete source-level test or calibration summaries.
3. Evaluation first averages all eligible patch-mask exposures for a joint within a window. Mask compression preserves exposure counts. Therefore, a patch observed under more eligible masks carries more weight within that window.
4. Windows receive equal weight within a sequence, sequences within a source, and the four joints within each region. Physical-patch deduplication governs support counts and four-patch selection. It does not remove all prediction exposures from overlapping windows. Different annotation sequences remain different support identities even when their underlying physical intervals might overlap.

The earlier wording could suggest equal weight for patch means after mask averaging. The corrected wording matches `compress_loss_rows` and the `target_count` weighting in `aggregate_profiles`.

## Normal-development adequacy evidence

These values are the final trained-head diagnostics, not outer-test prediction errors.

| Fold | 512 own coordinate MSE | 512 full coordinate MSE | 4,096 own coordinate MSE | 4,096 full coordinate MSE |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.023187 | 0.056799 | 0.018206 | 0.021079 |
| 1 | 0.044139 | 0.063058 | 0.038461 | 0.040379 |
| 2 | 0.042509 | 0.058327 | 0.036110 | 0.034134 |
| 3 | 0.046041 | 0.074266 | 0.040348 | 0.047248 |

Full-context coordinate error exceeds own-context error in all four primary folds and in three longer-budget folds. Fold 2 reverses this ordering after longer fitting. The methods draft now states that exception. Neither schedule establishes convergence. Frozen re-encoding can also produce nonnested memories despite nested raw observations, so these differences do not prove that additional body information is intrinsically harmful.

Checkpoint comparisons establish computational comparability across budgets. The claim that the longer budget was motivated before inspecting its abnormal rankings depends on the project decision record; tensors alone cannot establish when a human inspected results.

## Source roles and leakage boundaries

The source partitions were checked against both runs' saved registries and `window_roles.csv` files.

| Fold | Normal fitting sources | Normal reference sources | Normal test sources | Abnormal test sources | Auxiliary abnormal sources |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 12 | 5 | 6 | 57 | 171 |
| 1 | 12 | 5 | 6 | 57 | 171 |
| 2 | 12 | 5 | 6 | 57 | 171 |
| 3 | 13 | 5 | 5 | 57 | 171 |

Within each fold the four roles are disjoint. Anomaly-model fitting and reference fitting use normal sources only. Auxiliary abnormal sources are excluded from anomaly fitting and enter only the separately supervised condition task. Five-class evaluation distinguishes unique sources from source-label rows and divides each source's total evaluation weight among its labels. Repeated windows are averaged within sequences before source-label evaluation.

No direct source-ID fitting/test leakage was found. This is not evidence of participant independence: different source IDs can contain the same person. The existing limitations correctly retain analyst adaptation, camera/activity confounds, pose/missingness selection, overlapping annotations, small normal samples and prior cohort reuse. All 23 retained normals and 58 retained abnormal sources occur in the earlier index; 170 abnormal sources do not. Normal development also informs the later budget sensitivity while supplying the reference, so it is not untouched validation for that sensitivity.

## Data and evidence reconciliation

The refreshed inventory has 1,874 sequences and 348 source IDs: 32 normal and 316 abnormal. The media ledger contains 184 downloaded, 148 verified-present and 16 failed source videos, giving 332 available videos. Pose QC retains 9,474 windows, 1,389 sequences, 311 sources and 28 normal sources. Profile support retains 8,950 windows, 1,303 sequences and 251 sources, including 23 normal sources. These figures match the methods draft.

All 28 source-artifact hashes listed in `docs/draft01/evidence.json` matched their files during this audit. Its normal-validation entries matched the checkpoint histories. Both real runs already have passing `verification.json` reports covering code/data identities, checkpoint provenance, common target counts, reference hashes, fixed-budget selections and complete paired out-of-fold source coverage. Their verification timestamps are `2026-11-15T10:08:47.058073+00:00` for the primary run and `2026-11-15T10:11:57.478700+00:00` for the sensitivity run.

No result in this audit turns the study into an external replication or a validated clinical classifier. The manuscript's inconclusive paired comparisons and absent published anomaly baseline remain material limitations. The implemented stochasticity is uniform masking and the specified augmentation. It does not implement learned stochastic latent-variable inference.

## Full test suite

The suite was run once from `/Users/pmui/dev/alexpose/experiments/aaai` using that folder's uv-managed `.venv/bin/python`, CPython 3.12.13:

```text
.venv/bin/python -m pytest tests -q
105 passed in 16.07s
```

The complete captured pytest output is in [final_pytest.log](final_pytest.log). This test result is separate from fresh-kernel notebook execution. No hashed model, data or evaluation module was changed by this audit.

## Follow-up reference review

The subsequent [normal-reference review response](normal_reference_review_response.md) supplies exact cross-fold reuse and scale checks. Twenty reference assignments use 15 distinct normal sources, with pairwise overlaps of 0–2 and no within-fold fitting/reference/test overlap. None of the 960 frozen regional denominators across both runs binds the numerical variance floor. It also verifies that the whole-body probe is a four-frame temporal crop shift and that the quoted `+0.765` is a standardized profile-distance change, not raw MSE. No test ranking, retuning or winsorization was used for these checks.
