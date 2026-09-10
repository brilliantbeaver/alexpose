# Independent evidence audit: notebooks 00–07 and the original paper

Audit date: 9 September 2026. This review inspected the original `docs/paper.md`, the current protocol and implementation, the canonical notebooks, retained figure data, and the saved cohort and splits. Paths below are relative to `neurips-laterality/` unless explicitly prefixed by `../`. No encoder was trained and no existing artifact was changed. Applicable writing guidance was read from the workspace `AGENT.md`.

The core experiment is a useful controlled audit of a specific pose representation, but the original manuscript turns several unsuccessful tests into stronger conclusions than the evidence permits. The strongest revised story will distinguish a known coordinate transformation, a learned representation's utility, and a downstream construction that guarantees the desired output transformation. The current workspace also has an important provenance gap: the original run's report and checkpoints are absent, although a small aggregate-number file survives. This limits what can be independently verified during this revision.

## Evidence availability and independent checks

The retained original-run directory is `artifacts/paper/protocol_6f7baefbda07/`. It contains `cohort/`, `splits/`, `inputs/`, and `protocol_snapshot.json`. It has no `report/`, original-run checkpoint directory, or original-run held-out prediction directory. A whole-workspace filename search found no `checkpoint_source_bootstrap.csv`, `strict_representation_equivariance_source_bootstrap.csv`, or `native_output_symmetry_source_bootstrap.csv`. The only canonical `05_aggregate_statistics.ipynb` has no retained code-cell outputs; the same holds for notebooks 00–07. The suite's executed directory currently contains `motion_structured/`, rather than executed copies of the original registered evaluation.

The surviving primary-number source is `docs/figures/v21_figure_numbers.json`. It supplies values rounded to three decimals, without the complete original report metadata or prediction rows. `docs/figures/make_v21_figures.py:19` identifies the now-absent report directory as the source from which those figures were generated. `docs/TUTORIAL.md:152` independently repeats two of those aggregate comparisons in prose, but is not an independent statistical replication.

The manuscript's opening claim that all fifty jobs and hard integrity gates passed is therefore a historical report claim, not a fact that this review could verify from surviving primary artifacts. A revised paper can attribute the original numbers to the retained report summary, disclose the missing prediction/checkpoint provenance, and prioritize the newer results that can be recomputed. It should not say this revision reproduced original confidence intervals or revalidated all original checkpoints.

Read-only verification using the suite's own loaders succeeded:

- `laterality.data.load_cohort` validated the cohort's manifest, array hashes, profile, and content digest.
- `laterality.splitting.load_splits` validated its stored lineage. Rebuilding the split in memory using the frozen parameters reproduced every saved outer and inner partition.
- `laterality_extensions.diagnostics.read_registered_results` returned `registered report unavailable`, with no result rows.
- `laterality.governance.submission_readiness` returned `ready: false` and listed all three reviews as unresolved.

The retained lineage is:

| Item | Digest |
|---|---|
| Protocol | `6f7baefbda07c8e6899bc5fbb82b4651995b4b0089e0a1d1b36ae04480657088` |
| Effective context | `abc7a3d26a60fd479641b2ca76f7caab16bba127aeb05f5ef2be6eb56fd9d143` |
| Cohort | `bc23447824d2bc2bbe62dfdc7df5da8db7d1e73291d1675f2556597eeeefe2e3` |

Local configuration files establish a versioned computational contract. They do not, by themselves, establish an independently timestamped public preregistration. Use “frozen post-development protocol” unless the authors can supply a dated registration record demonstrating the stronger chronology. In particular, “before any result was seen” should not conceal the earlier transductive development or the later exploratory iterations on the same sources.

## Cohort, units, and the actual shape of the data

The inventory contract records 666 annotation files, 103 original source videos, and 642 available pose archives. Quality control accepts 625 sequences from 93 sources. Three finite targets were excluded for insufficient coverage: the saved target-contract record has 628 finite targets checked, 625 retained sequences, zero maximum mirror-target residual, and zero maximum invalid-coordinate-sentinel residual.

| Dataset annotation | Sources | Sequences |
|---|---:|---:|
| Cerebral palsy | 9 | 58 |
| Myopathic | 28 | 183 |
| Normal | 29 | 270 |
| Parkinson's | 9 | 39 |
| Stroke | 18 | 75 |
| Total | 93 | 625 |

These are observed source/video and annotation counts, not independent patients or confirmed diagnoses. Sources contribute between 1 and 60 sequences, with median 4. The smallest condition groups have only nine source videos. This supports motivation by clinical gait literature but does not support discovered condition-specific asymmetry, treatment response, or diagnostic performance.

The tensor description should be explicit because the user asks whether training preserves data shape. Raw archives contain `T × 33 × 4`: three estimated coordinates plus visibility, with retained raw sequence lengths from 24 to 1,576 frames. After preprocessing, saved arrays have these exact shapes:

| Artifact | Shape | Meaning |
|---|---|---|
| `model_xyz` | `625 × 64 × 33 × 3` | Prepared model coordinates |
| `model_valid` | `625 × 64 × 33` | Separate observation-validity mask |
| `pair_contrasts` | `625 × 5` | Original-target components |
| `missingness` | `625 × 10` | Original-lane bilateral coverage summaries |
| Encoder tokens per batch | `B × 16 × 33 × 96` | Four prepared time steps per joint token |
| Main probe features per batch | `B × 960` | Five pairs, each with 96-channel differences and sums |

No honest revision should claim that preprocessing preserves the exact original trajectory or time grid. It preserves the 33-landmark schema and a consistent tensor layout, while temporal interpolation and resizing change observations. Within training, a target mask retains the allocated joint/time token axes; selected target coordinate embeddings are replaced by zero before adding position embeddings. Invalid tokens are masked in attention and zeroed at the output. The predictor replaces target-position context vectors with learned mask tokens. See `laterality/model.py:44`, `:51`, `:110`, and `:165`.

Both “gait-focused” and more broadly masked variants can receive all 33 joints. In the original run, twelve joints are eligible prediction targets and are pooled by the VICReg regularizer: shoulders, hips, knees, ankles, heels, and foot-index landmarks on both sides. The five laterality pairs omit hips. It would be wrong to illustrate the encoder input as only ten or twelve landmarks, or to imply the broad-target comparison removed every anatomical prior.

## Coordinates and target definition

The target uses pelvis-centered, body-scale-normalized coordinates only where landmarks were originally observed. `laterality/geometry.py:171` explicitly separates that lane from model interpolation. Missing target coordinates remain invalid; target pelvis centering has no fallback. Input preprocessing may interpolate a gap of at most four missing samples and use a pelvis fallback, then linearly resize by relative sample index to 64 steps. Validity is resized separately and thresholded at 0.999. The target is never recalculated on that imputed/resized input in the registered evaluation.

Target frame times are `(frame_numbers - first_frame_number) / fps`. For each bilateral pair, both landmarks must be valid at both endpoints of a transition and the elapsed time must be positive. The code divides each displacement norm by that elapsed time, takes each side's median speed on the shared support, then forms `(left - right) / (left + right + 1e-8)`. Every pair must supply at least eight transitions and all five pairs must be usable. The final target averages the five contrasts. See `laterality/geometry.py:217` and `laterality/data.py:453`.

The target's physical units need careful explanation. Retained extraction code in `../work/nb_extracts/02_extract_and_watch_skeletons.md:512` uses MediaPipe `pose_landmarks`, not calibrated world landmarks:

```text
x = (crop_x0 + landmark.x * crop_width) / image_width
y = (crop_y0 + landmark.y * crop_height) / image_height
z = landmark.z * crop_width / image_width
```

Thus the coordinate norm combines image-relative horizontal and vertical quantities with inferred depth. Image aspect ratio affects the relative x/y scaling; pelvis normalization does not calibrate a metric 3D camera frame. “Coordinate-derived motion contrast” is accurate; “physical gait speed,” “metric 3D biomechanics,” and “real-world dynamics” would overstate this measurement.

The target's mirror sign reversal remains an exact algebraic property of this representation. Horizontal sign inversion preserves coordinate norms and exchanging paired landmark identities exchanges left/right speed summaries. That property holds for symmetric and asymmetric trajectories alike; it does not require a patient's actual gait to be symmetric, or a mirrored recording to have equal prevalence in nature.

Fresh calculation from `cohort/manifest.csv` gives target mean −0.006074563028935445, sample standard deviation 0.059148206385182124, minimum −0.1948212340422903, and maximum 0.2146815707124385. These are sequence-descriptive values. The saved components reconstruct the target to maximum absolute disagreement `9.93129189996722e-17`. That is a self-consistency check, not evidence of what a learned encoder preserves.

## Split and training audit

| Outer fold | Train sources | Test sources | Train sequences | Test sequences |
|---:|---:|---:|---:|---:|
| 0 | 74 | 19 | 436 | 189 |
| 1 | 74 | 19 | 443 | 182 |
| 2 | 74 | 19 | 553 | 72 |
| 3 | 75 | 18 | 548 | 77 |
| 4 | 75 | 18 | 520 | 105 |

The split is stratified at the source-table level by dataset annotation. Every source is an outer test source exactly once and appears in the other four training folds. All clips, reflections, and other augmentations inherit the source assignment. A dataset annotation is used for split balance and one nuisance lane; it does not enter the self-supervised objective. Independent person identifiers are unavailable, so shared people or related material across source IDs remain possible.

Each outer training set has four inner readout folds: 55–57 fitting sources and 18–19 validation sources, with 266–450 and 80–197 sequences, respectively. `laterality/evaluation.py:534` calls `_fit_readout` separately inside every inner fitting partition. Imputation, weighted feature centering, and scaling are fitted there; candidate ridge penalties are scored on the corresponding inner validation sources. The selected penalty is refitted on all outer training sources before outer testing. This implementation is more precise than the paper's blanket statement that scaling is fitted on outer training data: inner selection itself also avoids leaking validation targets or feature-scaling statistics into its fitted readout.

The encoder is trained on all outer training sources, including those subsequently assigned to inner readout validation. That is the stated design for tuning a readout of a fixed, label-free representation; it is not fully nested encoder selection. The outer test sources remain excluded from both encoder updates and supervised fitting. Per-sequence pelvis/body scaling uses the observed sequence itself and no cohort-wide learned statistics, which is appropriate for full-clip representation evaluation but should not be transplanted into a past-only forecasting claim.

The original model uses a four-layer, 96-channel transformer encoder, two-layer predictor, and four heads. It starts a new model for each outer fold, seed 42–46, and variant. Its self-supervised loss is a centered, temperature-scaled cross-entropy on masked teacher features plus a VICReg term, rather than a generic unspecified latent mean-square loss. The teacher is updated by an exponential moving average of student weights. `laterality/training.py:363` provides the actual training loop, and `laterality/model.py:308` provides the loss.

Each epoch samples one sequence from every training source, then adds source-uniform draws to reach 80 samples, or four batches of 20. Three hundred epochs therefore mean 1,200 updates per encoder, with 24,000 sample draws; they do not mean 300 complete passes through all 436–553 available training sequences. Vanilla uses no reflections; reflection augmentation uses probability 0.5, with a separate RNG stream. Source sampling, masking, initialization, and update budgets are matched across variants. The registered source and implementation checks are strong design safeguards, though absent original checkpoints prevent verifying that every historical job complied.

“Mask fraction 0.6” is also easy to misstate. `uniform_authorized_mask` computes a common hidden count from 60% of the minimum number of eligible valid tokens in the current batch, capped to leave at least one eligible token visible. It does not mask 60% of all `16 × 33 = 528` allocated tokens. Bilateral semantics enter this recipe through the supplied landmark schema, eligible target joints, regularizer pooling, optional anatomical reflection, and later probe construction. The audit's signed target never supplies an encoder training label. An explicit symmetry penalty belongs to the later experimental extension and must not be illustrated as part of the original training objective.

## Exact estimand and inferential limits

The primary predictive quantity is **not** the average of 25 fold-specific R² scores. For a given seed, the implementation joins each sequence's prediction from its held-out outer fold, obtaining a full out-of-fold pass over all 625 sequences. It computes source-balanced R² on that complete pass and then averages the five seed-specific scores. Each sequence receives weight `1 / clips_in_its_source`, so each source has equal total weight. The comparison between trained and paired initial representations is computed with the same source/sequence/seed alignment. See `laterality/reporting.py:409`–`:613`.

The source bootstrap resamples 93 sources with replacement 2,000 times, carrying all corresponding sequences and registered seed predictions together. It forms each seed's metric within the resample before averaging seeds; it does not average predictions first. The alternative mean-prediction ensemble is separately stored. Confidence intervals are percentile source-resampling intervals conditional on the fitted cross-validation pipeline. They do not incorporate new optimization seeds, new split allocations, encoder refitting, model selection over the many extensions, unmeasured repeated people, or arbitrary new acquisition domains.

The original manuscript inaccurately treats the high-coverage result as though it were another primary-estimand check. `laterality/reporting.py:1180`–`:1197` builds high-coverage metrics from the **seed-mean prediction ensemble**. The high-coverage subset itself is independently verified as 600 sequences from 91 sources. Its predictive result should either be explicitly labeled ensemble sensitivity or omitted until a matching primary-estimand calculation can be recovered.

Native output error first squares `(prediction + mirrored_prediction) / (2 × outer-training target SD)` within sequence and seed, then aggregates within source and across sources/seeds before taking the square root. It is not a signed average that can cancel between seeds. The strict token statistic averages sequence-level ratios with equal total source weight. Different error constructions answer different questions and should have separate axis labels.

The declared positive-claim gates are conjunctions of useful absolute prediction, favorable comparison with initialization, and a small transformation error. Failure of these gates does not prove that the corresponding population quantity is exactly zero. The protocol has no demonstrated prospective power analysis or an equivalence margin for learned-versus-initial predictive performance. “Fully powered null,” “no effect,” and “augmentation does not help” are therefore inappropriate summaries. Prefer “the interval did not establish a predictive gain under this protocol.” The registered augmentation interval is compatible with small benefits or harms.

## Primary numbers that survive

All entries in the next table are retained three-decimal summaries from `docs/figures/v21_figure_numbers.json`, not freshly recomputed confidence intervals.

| Comparison | Estimate | 95% interval | Interpretation |
|---|---:|---|---|
| Learned primary R² | 0.060 | [−0.025, 0.126] | Positive absolute utility gate unmet |
| Learned minus initial primary R² | −0.018 | [−0.039, 0.002] | Positive training gain unestablished |
| Reflection augmented minus vanilla primary R² | 0.004 | [−0.006, 0.013] | Neither superiority nor equivalence established |
| Learned constructed odd/zero R² | 0.043 | [−0.044, 0.113] | Exact parity does not establish useful prediction |
| Learned minus initial constructed odd/zero R² | −0.059 | [−0.095, −0.017] | Adverse conditional training contrast in this lane |
| Learned minus initial odd/free R² | −0.071 | [−0.108, −0.033] | Adverse conditional training contrast in this lane |
| Learned minus initial even/free R² | 0.024 | [0.009, 0.042] | Positive exploratory contrast, not a paradox |
| Native output antisymmetry error | 0.215 | [0.194, 0.236] | Above the 0.1 registered margin |
| Strict token error, learned vanilla | 0.114 | [0.095, 0.138] | Upper-bound gate unmet; interval straddles 0.1 |
| Strict token error, initial | 0.083 | [0.075, 0.094] | Lower error under this specified test |
| Learned minus initial strict token error, vanilla | 0.031 | [0.016, 0.048] | Increase under the identity-channel test |
| Learned minus initial strict token error, reflection augmented | 0.022 | [0.008, 0.039] | Increase under the same test |

Three-decimal reporting matters: rounding the primary R² contrast's upper bound 0.002 to 0.00 or the token-error lower bound 0.095 to 0.10 can obscure whether a threshold is crossed. The learned q interval includes values below 0.1; the honest statement is failure to demonstrate an upper bound below the threshold, not conclusive evidence that the population q exceeds 0.1.

## What the strict token test does and does not establish

`laterality/symmetry.py:91` compares reflected teacher tokens to anatomically permuted original tokens with the **identity transformation in the 96 feature channels**. It allows no fitted rotation, channel permutation, sign transformation, or centering. For common-valid tokens it divides squared residual energy by the total energy of both token arrays. This is a clear, reproducible test of one chosen representation action, not a general test of every possible reflection-equivariant encoding.

A representation could store horizontal vectors in channels that should change sign on reflection, or store an action expressed in another channel basis, and fail this identity-channel check while carrying useful geometry. The revised manuscript should use “strict identity-channel joint-swap error” near every important interpretation, rather than burying the restriction in limitations.

The mathematical ratio lies between 0 and 2 when the denominator is positive. A value around 1 indicates approximately uncorrelated vectors under the particular raw-energy comparison; “unrelated representations equal one” is not universal in the presence of shared offsets. Rejecting zero-energy tensors only excludes an all-zero degeneracy. A constant nonzero, mirror-insensitive encoding can obtain zero error while retaining no useful motion. Learned positional embeddings, common token offsets, and representation sensitivity matter, so a low initial q does not establish a random model's understanding of human geometry. Pair q with held-out utility, feature-variation or rank diagnostics, and a clear account of its fixed action.

The constructed odd readout is a valid algebraic control. With `z_minus = (z(x) - z(Mx))/sqrt(2)`, no feature centering, and zero intercept, a linear map is odd. The guarantee holds for arbitrary encoders, including random or constant ones. Thus “the only route” in the abstract is an unjustified universal claim. Poor trained-minus-initial performance also does not show that every useful component comes solely from the wrapper: it compares two complete representation/probe pipelines and cannot isolate all causes of their absolute prediction.

The positive even-feature contrast is not “the opposite of what laterality requires” in the strong sense asserted by the original paper. An exactly even predictor cannot represent an exactly odd target on a mirror-balanced population except trivially, but the observed dataset need not be mirror-balanced. Even features may correlate with signed labels through source distribution or measured/unmeasured nuisance structure. This is a distributional limitation or control result, not evidence that self-supervision inherently learns the wrong physical quantity.

## Notebook 07: an independently recomputed diagnostic worth retaining

The read-only `reconstruct_processed_target` diagnostic applies the same formula to prepared model inputs, with uniform relative-time steps. It can reveal disagreement introduced by the complete measurement path, but it neither identifies a particular preprocessing operation as the cause nor estimates a decoder's best achievable performance.

Fresh calculation using the retained cohort and the existing helper gives:

| Quantity | Value |
|---|---:|
| Original input | 625 sequences / 93 sources |
| Finite original/recomputed overlap | 623 sequences / 92 sources |
| Excluded from overlap | 2 sequences |
| Direct calculation-agreement R² | 0.21818988696299113 |
| Source-balanced correlation | 0.6518182554344372 |
| Source-balanced mean absolute difference | 0.04089209804204162 |
| Original target SD on weighted overlap | 0.05984303514794512 |
| Source-balanced sign agreement | 0.7041029622551362 |
| Maximum absolute discrepancy | 0.17660128616160597 |

The 70.4% quantity weights sequences equally within each source and sources equally; it is not a percentage of source medians with matching sign. This illustration is particularly valuable because it exposes a real limitation rather than assuming that normalization and resizing preserve a motion target. Label it a newly recomputed descriptive diagnostic, not a new learned result or an information ceiling. No uncertainty interval is supplied by the existing helper.

Notebook 07 also provides a clean constructed counterexample: left positions `(0,1,0,1)` and right positions `(0,0,1,1)` have the same average position but different median speeds. Swapping the trajectories reverses their contrast while preserving the means. This demonstrates a limitation of raw time averaging; it does not prove that averaged contextual encoder features lose temporal information, because an encoder can encode motion before averaging. Reversing time preserves a median speed contrast, so successful prediction of this particular target alone cannot demonstrate knowledge of temporal direction or causal dynamics.

## Prioritized revision requests

1. Replace the universal title and conclusion with a conditional claim about geometry-aware evaluation of skeleton predictive representations. A stronger Physical World AI connection explains why known coordinate actions provide controlled tests of articulated-body representations, while acknowledging the absence of calibrated 3D dynamics, interventions, external participant validation, and multimodal sensing in this experiment.
2. Disclose original-run artifact availability and distinguish historically retained aggregates from results recomputed from prediction rows. Restore original report/checkpoint lineage before submission if available; do not fabricate full precision or infer absent confidence intervals from differences of marginal bounds.
3. State the hypothesis as whether training improves source-balanced laterality prediction and transformation consistency over the paired initialization, with useful absolute prediction required. Remove “fully powered null” and equivalence language.
4. Put preprocessing and probe limitations into the main methods/results. The newly recomputed input-agreement diagnostic and the transparent toy temporal counterexample are suitable failure illustrations, with their evidentiary limits in the captions.
5. Show the true training graph: original coordinates and validity branch into the target measurement lane and the prepared input lane; all 33 joints enter the student/teacher; anatomical selection controls eligible hidden targets; optional reflection enters before training views; the audit label is used only by the fitted readout after encoder training. Explicit symmetry-loss experiments need a separately marked branch.
6. Correct the estimand description to five full out-of-fold seed passes, distinguish the ensemble sensitivity analysis, and explain source bootstrap conditioning. Keep source counts visible when citing large evaluation-row totals in later notebooks.
7. Narrow direct q claims everywhere to the joint-permutation/identity-channel action and explain why neither parity nor low q alone proves information retention. Remove the inference that the target-component oracle locates the failure inside the encoder.
8. Use authoritative gait literature for how conditions can alter bilateral coordination, but do not derive clinical condition differences from the dataset annotations or these signed targets. A population can have substantial absolute asymmetry and near-zero average signed asymmetry because affected sides differ.
9. Retain the unresolved governance status. This audit does not justify public submission, pose release, embeddings release, or external data use; it does support local analysis and manuscript revision within the user's request.

The core methods have strong source-separation and paired-control structure. The paper will be more credible if it makes these concrete safeguards legible while treating the failed and incomplete tests as evidence that refines the next question, rather than as proof that a general predictive-learning approach cannot encode symmetry.
