# Independent review of the controlled revision

This review covers the implementation of the binary endpoint correction, the extended observation cohort, and the three-arm forecasting study. The reviewer authored the literature synthesis and proposed the scientific protocol, but did not author the reviewed Python implementations. This is an independent implementation review, not an independent assessment of the reviewer's own protocol design.

At this review stage, real test metrics and test source predictions remain uninspected. Cohort membership and attrition are permitted design information. Synthetic test partitions used in software fixtures are unrelated to the real held-out sources.

## Extended observation cohort

Reviewed `worldmodels/extended_data.py` SHA256 `609de01497b1b661907952257df37c358f87cf15c0c66f6f7060c1e355416ea5`. The completed `extended-input-v1/preparation.json` records the same adapter identity, 414 windows from 93 sources, and 262.41 seconds of preparation. The prepared array SHA256 is `806077a81400a249726a3f8d1e4d3146378bc8ab01a83985a757b3cd4c9ebf6b`; metadata SHA256 is `c24c016c3c498f61dfaa8629d8a47d09781c45d716aa6abfcd77c3e6515f681e`.

The implementation rebuilds the current annotation inventory and retains the established source split. It samples the first 32 available annotation frames at stride two. The selected annotation sequences were separately audited for consecutive native-frame spacing: sufficiently long sequences had the expected spacing of two. This supports the recorded sampled FPS calculation for this cohort, but does not make the observation duration constant across sources.

Raw pose reuse requires matching annotation, video, pose-model, Ambient extractor, original adapter, and sampling identities. Reused cache file hashes are recorded. Additional requested frames are decoded once per source. Full-image extraction remains the same operation, so the revision does not confound a crop-scale change with its longer observation horizon.

The annotation-box rule scales coordinates only when image geometry is compatible. A confidently located pelvis outside the box invalidates the frame; absent pelvis support also invalidates it. Incompatible geometry is retained with an unresolved flag. This catches clear mismatches but cannot establish that the skeleton belongs to the intended person when people overlap. Retained unresolved cases still require visual review before any clinical interpretation.

One naming limitation needs to remain explicit: `prefix_mean_confidence` and `future_mean_confidence` average all selected-joint confidences, including zeros after invalidation. They are not means restricted to observed joints. This is a usable acquisition-quality feature, but reports should call it zero-inclusive mean selected-joint confidence. Prefix and retrospective future features must remain separate.

Acceptance: suitable for a documented source-grouped methodological experiment. It is not a patient-verified clinical cohort. The longer window and person gate jointly change eligibility, so comparisons against the earlier 16-frame cohort cannot isolate the effect of horizon length.

## Binary endpoint correction

The implementation filters training and validation rows before fitting or selecting any classifier. It loads the same saved representations and earlier classifiers for the endpoint comparison. Source aggregation precedes validation selection. The explicit binary labels and balanced source/class weights address the mismatch between a five-class objective and a pooled abnormal decision.

The seal pins arrays, metadata, baseline checkpoint/classifier bytes, implementation dependencies, protocol text, classifier bytes, and runtime versions. The revised final evaluator checks those identities before opening real test rows. The mixed-adaptation probe is the primary JEPA comparator based on validation selection; other readouts remain secondary. Direct-versus-derived endpoint differences use paired source predictions.

The initial implementation inherited a 1,000-resample bootstrap default despite the protocol specifying 2,000. This was reported before final evaluation; the reviewed revision now passes `repetitions=2000` explicitly. Because this changes the recorded code identity, the coordinator must retain the old development record and create or reseal a clearly identified revision before evaluation.

A binary seal alone does not enforce the cross-study release rule. The coordinator must freeze both supervised and forecasting decisions before opening either study's held-out metrics, because their source partitions overlap. Passing one implementation's checks is not permission to use its test findings to tune the other experiment.

## Forecast implementation and review fixes

The reviewed design hides all future tokens in the predictor, while loss support includes future-observed joints that were also observed in the prefix. The teacher may construct training targets from the full sequence. Deployment predictions use the context-only latent prior. This separates permissible training supervision from forbidden future inputs.

All arms clone one normal-only pretrained state per seed and share source/sequence/window schedules. The residual head initially reproduces persistence. Static and shuffled controls preserve the original-prefix persistence anchor, avoiding an accidental change of anchor for joints missing in the final prefix frame. Coordinate and latent losses average within each sampled source example before averaging the batch.

The energy score uses distinct sample pairs with the appropriate Monte Carlo denominator. It uses the complete fixed sample set rather than selecting the sample nearest the outcome. Source aggregation averages windows within sequence, then sequences within source. The displayed primary paired interval averages seed effects within source before bootstrapping sources, so seeds are not treated as extra patients.

Two discrepancies were raised before the substantial training run. First, pretraining initially omitted latent MSE despite the sealed protocol specifying it. The model author corrected the isolated revision trainer to use CE, source-balanced latent MSE, and variance/covariance regularization, with zero coordinate weight. Second, the uncertainty summary initially reported AURC but omitted the declared retained-risk values at 50%, 75%, and 100%; the author was asked to add these values with consistent tie treatment.

The hard freeze checks completed finite executions and artifact identities. The scientific persistence criterion limits claims, but does not suppress unfavorable arms or prevent their final evaluation. This distinction is essential for an honest negative result.

Independent focused verification command:

```bash
uv run --project experiments/aaai-worldmodels --no-sync pytest \
  -c experiments/aaai-worldmodels/pyproject.toml \
  experiments/aaai-worldmodels/tests/test_revision.py -q
```

The first independent replay passed all six tests in 2.23 seconds. These cover persistence initialization, hidden-future invariance, common support, fixed control anchors, energy-score arithmetic, source scheduling, time-aware velocity, insufficient partial-correlation degrees of freedom, validation-only development, and rejection of an invalid freeze. This is software evidence, not real-data scientific evidence.

Final code acceptance before the substantial run: `worldmodels/revision.py` SHA256 `5dcf6e3165ff04e313e48cde39ac2638a1c24fcc0a2d9be3cbd56ad3397b2b02`. This version includes the corrected pretraining objective, retained-risk values with tie averaging and actual coverage, finite inference checks, runtime version checks including SciPy, and the preparation-manifest hash.

An additional independent in-memory synthetic replay used nine generated sequences, three disjoint source partitions, four frames, a two-frame context, two updates per stage, one seed, all three arms, and two prior samples. It completed development, freeze, and final evaluation, verified every prescribed retained-risk field, and rejected a repeated final-access attempt. The temporary fixture was deleted after completion. No real test outputs were involved. These checks authorize the declared real-data run; they do not establish any scientific hypothesis.

## Boundaries on interpretation

The study tests relative 2D articulation at a fixed number of sampled frames. Physical durations vary by source. A positive persistence comparison would support this forecasting task; it would not establish causal neurological understanding, future disease progression, or treatment simulation. Static and shuffled controls determine whether temporal ordering materially contributes.

The stochastic arm has additional trainable parameters. Any gain is therefore evidence for the complete conditional latent model, not a parameter-count-isolated causal effect of stochasticity. Source counts, especially six normal validation sources, limit uncertainty analysis. A heavily adjusted partial correlation is unavailable when its residual degrees of freedom are too small.

Real training results, real test performance, and manuscript claims remain pending until the frozen experiments complete and their outputs receive a separate review.

## Secondary anomaly audit declared before test release

The coordinator requested this extension after implementation acceptance and before inspecting the held-out outcomes. It does not change the frozen model training or primary H1 endpoint. The coordinator will pin a separate pretest amendment and include it in the joint release manifest; this review does not silently rewrite the frozen protocol.

Evaluate three prespecified source scores: future-coordinate MSE, predictive energy score, and excess MSE defined as model MSE minus persistence MSE. Fix the direction in advance: larger values indicate greater abnormality. Excess MSE asks whether the model's relative advantage over local persistence deteriorates on abnormal examples. It is not a likelihood ratio, and may remain negative for both labels. Do not divide by persistence MSE, whose small denominator can create unstable scores. Energy scores and MSE have different units and must not be subtracted from each other.

Use identical source membership and valid-coordinate support when comparing models with persistence. Compute source scores using the existing window, sequence, then source aggregation. Report all declared arms and seeds, with any seed-mean score explicitly identified. Never flip the score sign or select the best arm, seed, horizon, or score after viewing test AUROC. The full sixteen-frame horizon remains primary within this secondary audit; shorter horizons remain descriptive.

Report AUROC and average precision with normal/abnormal source counts and abnormal prevalence. Use paired source resampling with 2,000 replicates for differences where both labels remain supported, recording any rejected single-label bootstrap draws. Very small support and mixed neurological labels preclude condition-specific clinical sensitivity claims.

Include persistence error and prefix acquisition-quality controls on the same observations. A quality-only classifier trained on abnormal labels and a normal-only predictive surprise score use different supervision. Their comparison is a confounding diagnostic, not a matched-learning-regime superiority test. Keep future-quality predictors in a separately labeled retrospective audit. A strong quality baseline can challenge a motion-based interpretation even when the predictive model has favorable discrimination.

This amendment increases the number of descriptive comparisons. Its purpose is transparent assessment of the link between forecasting and abnormality, rather than a new opportunity to claim whichever metric is favorable. H1, normal-source forecast improvement over persistence, remains the primary endpoint.

## Validation-stage assessment of one possible sensitivity analysis

Before joint test release, the coordinator asked whether increasing the coordinate-loss weight from 1 to 10 would be justified. The reviewer inspected only training histories and the permitted validation progress report from `revision-development-v1`. In the final 50 recorded updates, residual coordinate MSE was approximately 0.075 to 0.086, while latent CE was approximately 1.16 to 1.42 and latent MSE approximately 0.28 to 0.34. Direct-prediction coordinate error decreased substantially during training, from approximately 0.50 near the beginning to 0.085 to 0.103 near the end. Thus, the coordinate head demonstrably learns; a completely broken coordinate objective is not supported.

All three deterministic residual models had slightly negative normal-validation gains over persistence. Their temporal-shuffle effects were very small. These results do not support H1 under the frozen training budget and task. They must remain visible regardless of any additional diagnostic experiment.

Scalar loss magnitudes alone cannot establish gradient imbalance. The coordinate head is supervised only by its coordinate objective, and Adam scaling plus gradient clipping complicate an argument based on loss units. The reviewer therefore advised against tuning merely to obtain a favorable result on six normal validation sources.

A single fixed weight-10 residual sensitivity can be justified only as an explicitly exploratory optimization audit if measurements on predeclared training batches show that shared-trunk coordinate gradients are materially weaker than the competing gradients. Such a run should reuse the exact pretrained states, source schedules, 300-update budget, and three seeds; retain weight 1 as the primary analysis; report both settings regardless of sign; and prohibit additional weight searches or test-based selection. It would test sensitivity of this hybrid objective, not establish a new medical-world-model method. The gradient diagnosis and the coordinator's decision must be recorded before test release.

The model author subsequently measured the final deterministic residual checkpoints on the original training-schedule batches 0, 149, and 299 for each of seeds 27, 41, and 73, without optimizer updates. The reported shared-trunk coordinate-to-auxiliary gradient norm ratios ranged from 0.22% to 1.69%, with median 0.77%. Auxiliary loss combined CE, balanced latent MSE, variance, and weighted covariance. Gradient cosines had mixed signs. These measurements support the plausibility of an objective-weight sensitivity, while not proving that a larger weight will generalize; a small gradient can also reflect local fit. Based on this training-only evidence, the reviewer accepted exactly one weight-10 deterministic residual audit under the constraints above. The original weight-1 study remains primary. These diagnostic values are attributed to the model author's measurement, not to an independently repeated gradient calculation.

Improved stochastic energy score, if observed, would support the complete sampled predictor relative to its deterministic comparator. It would not by itself establish calibrated clinical uncertainty or prove that conditional stochasticity is necessary: this three-arm design does not include an unconditional-noise comparator or an equal-parameter deterministic control. This limitation should remain explicit without adding post hoc baselines to obtain a preferred conclusion.

## Independent review of the objective-weight audit

Reviewed `worldmodels/objective_audit.py` SHA256 `7dbe67067c4621b4d3cc0d8a1511d988c463f14d9180bb472349e75ca538fa3a`. The gradient evidence is recorded in `docs/training_gradient_audit.json`, SHA256 `1e7504d8e2cd62c5ed0922bbbb64fc08849b34520c25d0c8907ebc7616d79006`. The frozen primary implementation remains SHA256 `5dcf6e3165ff04e313e48cde39ac2638a1c24fcc0a2d9be3cbd56ad3397b2b02`.

The audit explicitly asserts that the sole changed protocol field is coordinate weight 1 to 10. It reconstructs the original pretrained model state, applies the same deterministic residual clone and zero coordinate-head initialization, loads the original source schedule, resets optimizer and random seed, and runs the same 300 updates for each of the three seeds. It does not continue from a trained weight-1 forecast checkpoint. Validation scoring and control masks are unchanged. All audit seeds are retained independently of their result signs.

The implementation pins primary artifacts, prepared arrays and metadata, runtime versions, gradient evidence, and its own code. Final evaluation requires a completed primary final result matching the original development identity, and it rejects repeated audit test access. A focused synthetic fixture also verifies parent-checkpoint and source-schedule hashes, refusal before the primary release, successful complete execution, and the repeat-access guard.

Independent replay of `tests/test_objective_audit.py` passed both tests in 2.76 seconds using the experiment's `uv` environment and pytest configuration. It exercised synthetic partitions only and did not access real held-out outcomes.

Two provenance requirements are handled at the coordinator's joint release boundary rather than by changing code during the active audit run: verify the primary accepted freeze and current protocol hash, and pin the exact primary per-seed source-score CSVs plus persistence CSV consumed by the audit comparison. The reviewer independently verified that the real primary freeze references the current development record and that its protocol hash still matches. The joint release must retain these checks and consumed-output identities.

Acceptance: the audit implements the intended single controlled objective change. No additional hyperparameter search is authorized by this review. Real final outcomes remain unopened at this stage.

## Pretest visual quality correction

Before final test access, the coordinator produced a contact sheet for the two sources with retained unresolved annotation geometry. The reviewer independently opened the rendered `artifacts/research_analysis/person_review.png`, SHA256 `69caf82bdacdece92655ead86ab8862ef0a81d03d52d9b7d81f09effb3c5c2a4`, containing 21 frames from seven retained windows. The following observations concern visible track continuity, not inferred disease, age, clinical role, or participant identity:

| Source | Partition | Independently visible failure |
| --- | --- | --- |
| `eBvzFkcvScg` | Training | At frame 478, the selected keypoints follow the person wearing burgundy. At frames 510 and 540 they follow the different person wearing white. |
| `CMls0LwK5OU` | Validation | At frames 758, 790, and 820 the selected keypoints alternate between the person wearing white/red and the person wearing black. Frames 493 to 525 and 1667 to 1699 also contain clear scene changes within the proposed windows. |

These failures violate the assumed continuous single-person motion observation. The reviewer supports conservative exclusion of both whole source groups from the short and extended cohorts. This does not assert that every frame from either source is incorrect. It prevents reliable-looking windows from a failed source track from being used selectively after observing outcomes. The rule applies to all sequences and windows from each excluded source, without replacing them or changing partitions.

The correction occurred after training and validation development but before final test inspection. It is therefore a documented data-quality amendment, not an untouched original preregistration. Preserve the original data and results under their existing identities and publish the attrition and reason for the derivative cohorts.

The minimum provenance contract is:

1. Pin the rendered review image, sampled source/frame IDs, independent observation record, and exact source exclusion ledger.
2. Create new immutable cohort directories, with a mapping from every retained row to its original row and recorded before/after source and sequence counts.
3. Verify exact equality of retained coordinate arrays, validity masks, labels, source/sequence identities, frame/time metadata, preprocessing settings, and row order. In particular, prove that normal training, normal validation, and all held-out test observations are unchanged.
4. Reuse a frozen normal-only forecast checkpoint only through explicit inherited-artifact lineage, with byte hashes and the equality proof above. The same rule applies to the weight-10 audit. These exclusions affect abnormal training/validation sources, so unchanged normal-only training can justify reuse; a generic statement that the cohort changed only slightly cannot.
5. Preserve the existing validation window predictions for unchanged observations and recompute aggregates after filtering the excluded sources. The current sampling and shuffle seeds depend on batch positions. Re-running a filtered validation array can therefore alter Monte Carlo draws or shuffled controls even when individual normal windows are unchanged. Such a rerun must not be mislabeled an identical stochastic evaluation.
6. Refit components that learned from the excluded abnormal training source: staged/mixed encoders, supervised RF/probes, the five-class comparator used for the endpoint comparison, and quality-only classifiers. Use the same declared search grids and budgets. Retain the mixed-adaptation JEPA probe as the primary supervised comparator; do not open a new search for a favorable representation.
7. Freeze a new joint-release manifest that pins corrected cohorts, inherited forecast lineage, refitted components, validation decisions, and exact test-set equality before opening any test outcomes.

The independent visual review confirms the need for this correction. Acceptance of the resulting derivative artifacts still depends on the coordinator's equality proofs and new manifests. No real test performance was inspected to motivate the exclusions.

## Independent acceptance of the corrected observations

Reviewed `worldmodels/clean_data.py` SHA256 `85e8fadfeb18d01d01ee71b97811a42f89f961fa2a313e345d3805a68d752f22`, exclusion ledger SHA256 `98a234fbc4c6bd1948afd6e1799aee2eafd3ebddd27e6e2d9fbf35830cf6ed60`, and forecast reuse certificate SHA256 `cb3c39757c85ce89c06f7abe8a77302ae103b9f0fad8458de81e1fd8c2034d9a`.

The reviewer independently loaded the saved parent and derivative arrays and metadata. All 407 retained extended windows from 91 sources match the parent rows exactly, including coordinate arrays, validity masks, every metadata field, and row order. The saved row map also matches exactly. Normal training is unchanged at 84 windows from 17 sources; normal validation is unchanged at 27 windows from six sources; the complete extended test input is unchanged at 59 windows from 19 sources. This is input verification only, not test-outcome access.

All 82 inherited validation CSVs were independently checked against their parent hashes and filtered parent rows. Every retained value and row order matches exactly, with no numeric round-trip differences. The recorded lineage hashes also match. This preserves the original stochastic samples and temporal-control realizations. Both excluded sources are absent from the corrected extended and short cohorts, and each retained source belongs to exactly one partition.

The short cohort requires a different description. It is freshly gated from provenance-verified raw poses, rather than merely a row-deleted copy of the earlier short cohort. The reviewer independently verified 641 raw-to-gated pose lineages: parent and derivative file hashes, annotation hashes, unchanged raw coordinates and frame/time/dimension fields, and the re-applied annotation-box confidence gate. Reconstructing windows from those gated poses reproduced all 405 saved short coordinate arrays and masks exactly. No pose estimator was rerun, and no test metrics were read during this audit.

The short cohort has 405 windows from 89 sources. Its training, validation, and test support is 53, 19, and **17 sources**, respectively. The test partition has 57 windows: six normal sources, one cerebral-palsy source, six myopathic sources, two Parkinson's sources, and two stroke sources. The fixed person gate additionally removes test source `sChOLBghpAc`; therefore the short test cohort is not identical to the earlier 18-source test cohort. Training support also decreases from 58 to 53 sources. This attrition must be disclosed, and all short-cohort comparators must be refitted and evaluated on the same corrected observations.

Acceptance: the extended cohort supports explicit inheritance of the frozen normal-only forecast and objective-audit models and validation predictions. The short cohort is accepted as a newly gated real-data cohort requiring fresh supervised/adaptation fitting. Claims of unchanged test observations apply only to the extended forecasting cohort. The coordinator must use the corrected short-cohort counts in final tables and notebook execution evidence.

## Prerelease acceptance of refitting and analysis

The clean binary runner, `worldmodels/binary_clean.py` SHA256 `d5530e7a2fcec2570726319bf6132ea4b6b026caf7e2d5bc0be8064faa3876ac`, retains the fixed mixed-adaptation primary comparator irrespective of the new validation ranking. It rejects the two excluded source IDs, verifies the refitted baseline lineage, limits model selection to training/validation rows, and retains the declared grids and seed. Independent replay passed all three focused tests in 2.05 seconds. No real test metrics were inspected in this check.

Accepted release runner SHA256: `e183d2754fee6614e1f8d1a5e5d3acc8c9c03353c5b7198e6e37348f8a842033`. Accepted analysis runner SHA256: `30795fb6cdf8d55352fec407d1f24877b8612ac5d23739c6342780a95dba5960`.

Before acceptance, the reviewer requested three changes that are present in these versions. The joint release directly pins the relevant implementation modules. The completion record pins every final result JSON and consumed source/window CSV, and the analysis checks them before computing summaries. The anomaly analysis reports paired AUROC and average-precision differences against persistence, using aligned source IDs/labels and the same resampling draws, instead of relying only on separate confidence intervals.

The reviewer extracted only the bootstrap function from the analysis source and tested it on synthetic inputs. Identical candidate/reference scores produced exactly zero paired effects and zero-width intervals; a perfectly ranked score against its reversed reference produced the expected AUROC difference of one; the fixed-threshold balanced-accuracy check also passed. This isolated check did not execute the analysis script or read any real final outputs.

The 17-source classifier cohort and 19-source forecast/quality cohort remain separate. Their point estimates cannot be treated as a matched cross-task comparison. The release explicitly records that legacy comparison code had already computed quarantined test outputs, while the authors report not inspecting metric values for development decisions. Accordingly, the final study is an internally held-out exploratory experiment, not an independently preregistered untouched clinical validation.

Prerelease implementation acceptance is complete. A separate review of empirical claims follows only after the coordinator confirms completed joint release.
