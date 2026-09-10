# Evidence audit of notebooks 08–18

Reviewed 9 September 2026. This audit inspects retained notebook outputs, implementations, the tutorial, and saved experiments. It does not retrain models or alter existing research artifacts. Paths below are relative to `neurips-laterality` unless otherwise stated.

The strongest extension finding is a discrepancy between successful optimization and useful movement representation: every trained arm learns clip correspondence under a held-out predictor diagnostic, while its motion-sensitive frozen readout performs worse than the matched initial encoder. Motion weighting changes the target distribution and connected regions remove measured local cues, but neither establishes a laterality-readout benefit. This supports a paper about testing whether geometric predictive learning preserves a physical observable. It does not support a claim of improved clinical diagnosis, successful gait forecasting, or successful explicit symmetry training on GAVD.

## What was independently verified

The latest complete grid is:

`artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57`

Its manifest marks the grid complete. The audit independently verified all ten grid-table SHA-256 hashes, 125,000 prediction records, their complete pairing and coverage, the 200 seed-specific pooled R²/MAE rows, and all 125 training histories with exactly steps 1–1,200. The largest discrepancy between recomputed and saved scores was `9.194034422677078e-17` for R² and `9.71445146547012e-17` for MAE. Each experiment/arm/representation/seed contains 625 clips from 93 source videos. A source appears in only one outer test fold. The 50 paired fold/seed jobs contain 125 encoder arms and 150,000 arm-specific optimizer updates.

The companion [recomputation script](../../verify_physworld_evidence.py) performs these checks without importing the experimental metric code, prints its numerical audit as JSON, and writes no experimental files. It also independently reproduces the three existing source-bootstrap intervals. It checks grid-table hashes and training-history completeness; it does not claim to have independently rehashed all 125 model checkpoints and every large training manifest.

The exact output and its analysis-status annotations are retained in [physworld_evidence_recomputed.json](../../physworld_evidence_recomputed.json).

The audit additionally computes new paired trained-versus-initial intervals from the retained predictions. These are explicitly identified below as an exploratory analysis added on 9 September, rather than a result present in the original registered study.

## Evidence status by notebook

| Notebook(s) | Retained evidence and scope | Appropriate use in the paper |
|---|---|---|
| 08 | Current source notebook has no retained code outputs. The dated real-data result survives in `docs/figures/tutorial_masking_summary.json`: 25 fold/seed jobs, 50 encoders, fixed ridge penalty 1. | Historical masking comparison, clearly separated from the later implementations. Its raw research-masking artifacts are absent from this local workspace. |
| 09 | Symmetry-loss implementation and a default synthetic teaching experiment. The tutorial describes eight updates on generated data; the current source notebook has no inline code outputs. No retained real-data symmetry-loss comparison was found. | Explain the proposed geometric objective and its software checks; do not present a clinical or real-data training benefit. |
| 10 | Past-only forecasting implementation and documented synthetic demonstrations; current source has no inline code outputs. | Explain the information boundary and controls. Its evaluation regresses coordinates from past features, rather than decoding the predictor's forecast latent. |
| 11 | Executed generated examples for mask construction, equal counts, anatomical connectivity, motion selection and reflection of validity/masks. | Explain controlled interventions. Its generated selection counts are not GAVD performance evidence. |
| 12 | Inline output records completion of 25 real GAVD jobs and 50 encoders, alongside clearly labeled synthetic examples. `docs/figures/tutorial_comparative_masking_summary.json` retains the result snapshot. | A completed earlier real-data comparison of gait versus all-landmark target eligibility. Raw `artifacts/comparative_masking` is absent locally, so this audit does not claim a fresh reconstruction of its raw predictions. |
| 13 | Inline output is a generated three-update demonstration, including a positive control and raw-removal example. Larger real predictor and mild-corruption findings in the tutorial come from Notebook 12's saved evaluations. | Keep the synthetic software checks distinct from those empirical summaries. |
| 14 | Executed synthetic decoder comparison: eight generated clips/eight generated sources, six training/two test, four updates per arm, one seed. The real-data plan is displayed with loading/training disabled. | A failed short-run synthetic forecasting demonstration and a disciplined next experiment. No real-data forecasting gain. |
| 15 | Current source has no inline outputs, but `executed/motion_structured/gavd_5yc3ve5h/15_motion_weighted_masking.ipynb` retains the real mask audit. | Positive evidence that the sampling intervention changes motion enrichment. |
| 16 | Inline real GAVD coverage audit: 75,000 repeated mask draws over the same 625 clips/93 videos. | Positive evidence that geometries remove different local cues. Whole trajectories and interior completion have no trained downstream comparison. |
| 17–18 | The source 17 lacks inline code outputs, but 18 and the complete saved grid establish all 50 jobs/125 encoders. | The central empirical comparison. Use its raw predictions and exact scope rather than inferring completeness from workload declarations. |

There is a documentation freshness conflict: `docs/MOTION_STRUCTURED_MASKING.md` still says that new full-grid checkpoints are absent, whereas the dated 8 September notebook interpretation, Notebook 18, and actual saved grid establish completion. The manuscript should use the completed artifacts as the authority and cite dates when distinguishing earlier experimental plans.

## What the latest encoder was actually trained to do

All 33 anatomical landmarks remain in the prepared input. A clip has shape `[64,33,3]`; validity has shape `[64,33]`. Four consecutive prepared steps of one landmark form a token, giving 16 time blocks × 33 landmarks, with a 96-dimensional vector per token. A token is eligible only if all four required observations are valid. The online encoder zeros the hidden patch embeddings while retaining anatomical/time positions, so masking preserves the token array and sample identity. The teacher receives the full prepared clip. Naturally absent observations are excluded from targets and attention support.

The use of a fixed input tensor does not mean preprocessing preserves the original shape, timing or every geometric quantity. Temporal resizing changes the original number of observations; interpolation and normalization can change displacement-based measurements. The safe claim is that masking preserves the prepared joint/time indexing, bilateral identities and validity contract. The manuscript should explicitly distinguish the original target-calculation lane from the normalized, resized model-input lane.

The latest recipe uses an encoder depth of four, predictor depth of two, four attention heads, batch size 20, 1,200 updates, AdamW learning rate 0.001, weight decay 0.05, EMA momentum 0.999, and gradient clipping at 1.0. It ran on CUDA with BF16 computations and FP32 weights/loss reductions/frozen evaluation. The objective is centered teacher-distribution cross-entropy plus `0.05 × VICReg`, not a plain latent MSE. VICReg here combines agreement between geometric views, a feature-variance penalty and a covariance penalty. The geometric views use rotations within ±8 degrees and small translations. See `laterality/model.py`, `laterality_extensions/motion_structured_training.py` and the first motion-job manifest.

The trained extension is anatomy informed in three concrete places. Learned landmark embeddings distinguish left from right positions. The original gait-mask conditions select targets from shoulders, hips, knees, ankles, heels and foot tips, while the all-landmark conditions change target eligibility. The shared auxiliary regularizer pools the same twelve gait landmarks even when prediction targets can come from all 33. The subsequent readout explicitly combines five bilateral pairs—shoulders, knees, ankles, heels and foot tips—using sums and signed differences; hips are excluded from that readout. The five pairs and twelve-landmark set should not be described as neurologically validated.

Explicit reflection training is a separate intervention. In the latest motion/region grid, saved `reflection_probability=0` and `symmetry_weight=0`, and `train_mask_study` raises an error if either is nonzero. Notebook 09's proposed loss aligns anatomical token positions under reflection, without forcing the whole-body feature to be identical. A training-pipeline figure should use a distinct proposed/synthetic inset for this loss, rather than attach it to the completed GAVD training path. Side-sensitive evaluation is also not equivalent to imposing a reflection loss on the encoder.

Matched experimental arms share initial model/projector weights, source schedules, geometric views and update counts. Masks use separate controlled random streams. Regions determine a per-clip feasible count and receive a random reference at the same count; a dense reduction averages hidden-target losses within each clip and then across clips, preserving sample identities when counts vary. Both compared arms retain the full-input regularizer, so the masked predictor's input isolation should not be overstated as a prohibition on the encoder ever seeing full training clips.

## Held-out evaluation and its limits

The same five outer source partitions are reused across seeds 42–46. Their training/test clip counts are 436/189, 443/182, 553/72, 548/77 and 520/105. They have 74/19, 74/19, 74/19, 75/18 and 75/18 training/test source videos respectively. All clips from each outer test video are excluded from encoder training and readout fitting. Source identity is a recording-level grouping; there is no verified cross-video participant identity, so person-independent generalization is unestablished.

Readouts use training-source-only imputation, scaling and ridge regression. Three inner groups choose among penalties `[0.01,0.1,1,10,100,1000,10000]` by pooled source-balanced validation MSE, with smaller penalties breaking ties. Those groups tune the readout only: the encoder has already seen all outer-training sources. Choosing a whole pretraining recipe would require keeping the pilot validation sources outside that encoder's training too.

For each seed, all five outer-test folds are pooled. A clip from source s receives weight proportional to `1/n_s`, where `n_s` is the number of evaluated clips from that source. R² is computed against the weighted mean of the pooled evaluation targets, and MAE uses the same weights. The paper then averages the five seed scores. Averaging fold R² values or averaging predictions across seeds before scoring would define different estimands. The training-source-mean control is deployable; the pooled evaluation mean in the R² denominator is a metric reference.

The source bootstrap resamples 93 complete source videos with replacement, moving their clips and all paired seed predictions together. It conditions on the fitted models and existing source partitions. It excludes retraining uncertainty, uncertainty from selection after repeated inspection of this cohort, and unknown relationships between participants across videos. Seeds are repetitions of stochastic fitting on the same recordings; they do not create five independent cohorts.

## Latest reproducible movement-readout results

All rows below use 625 clips, 93 source videos and five seeds. R² and MAE are means of pooled per-seed scores; the spread shown for R² is the standard deviation across seeds.

| Representation | Mean R² ± seed SD | Mean MAE |
|---|---:|---:|
| Training-source mean | −0.010609 ± 0 | 0.0461723 |
| Direct pose summary | 0.034541 ± 0 | 0.0449596 |
| Initial encoder, mean summary | 0.070827 ± 0.018574 | 0.0443442 |
| Initial encoder, mean-motion summary | 0.222544 ± 0.026808 | 0.0415461 |
| Motion uniform teacher, mean-motion | 0.113725 ± 0.011007 | 0.0436583 |
| MAMP-convention teacher, mean-motion | 0.112890 ± 0.030590 | 0.0437511 |
| Robust-motion teacher, mean-motion | 0.114209 ± 0.021036 | 0.0435951 |
| Region uniform teacher, mean-motion | 0.109446 ± 0.008849 | 0.0440373 |
| Connected-region teacher, mean-motion | 0.100777 ± 0.009239 | 0.0437557 |

The mean-motion summary adds temporal standard deviations, mean absolute feature increments and observation-support fractions to bilateral sums/differences of means. At width 96, the mean summary has 960 features; mean-motion has 2,890, including ten support fractions. The initial encoder improves by 0.151716 R² when these components are added. Their combined addition improves trained teachers by only 0.027022–0.046815. The experiment does not isolate ordered motion from amplitude or missingness. In particular, standard deviation does not encode temporal order, and support fractions may encode recording/pose-estimation properties.

Every trained teacher has lower R² and higher MAE than its matched initial encoder under the same mean-motion summary in every seed. Final online mean-motion scores are also lower: 0.080493, 0.080923, 0.073961, 0.065700 and 0.063632 in the corresponding five-arm order. The disadvantage therefore cannot be attributed solely to choosing the EMA teacher instead of the online encoder.

The three existing paired mask contrasts are:

| Final teacher, mean-motion contrast | Mean ΔR² | 95% source-bootstrap interval |
|---|---:|---:|
| MAMP convention − motion uniform | −0.000834 | [−0.020372, 0.015296] |
| Robust motion − motion uniform | 0.000485 | [−0.015497, 0.015173] |
| Connected region − region uniform | −0.008669 | [−0.033290, 0.017666] |

These intervals include zero and permit effects in either direction. They support neither superiority nor equivalence. Region and motion references differ in mask budget and must remain separate.

The new exploratory analysis applies the same 2,000-resample source bootstrap, seed 812, directly to trained teacher versus matched initial mean-motion predictions:

| Teacher arm − matched initial | Mean ΔR² | 95% interval | Mean ΔMAE | 95% interval |
|---|---:|---:|---:|---:|
| Motion uniform | −0.108819 | [−0.170093, −0.041219] | 0.0021122 | [0.0004937, 0.0038781] |
| MAMP convention | −0.109653 | [−0.165071, −0.051351] | 0.0022050 | [0.0007061, 0.0038627] |
| Robust motion | −0.108334 | [−0.170640, −0.042307] | 0.0020490 | [0.0003668, 0.0039266] |
| Region uniform | −0.113098 | [−0.166192, −0.059623] | 0.0024912 | [0.0010502, 0.0040473] |
| Connected region | −0.121766 | [−0.182787, −0.055281] | 0.0022096 | [0.0005845, 0.0040972] |

All intervals exclude zero in the unfavorable direction, conditional on the fitted models. They are exploratory, marginal intervals without a familywise multiplicity correction. The five contrasts share controls and recordings. This direct uncertainty calculation strengthens the specific observed trained-versus-initial comparison; it does not establish that JEPA necessarily destroys movement information or that no other decoder could access it.

## What succeeded, and what remains unresolved

The archived Notebook 15 audit shows equal average target counts of 80.771 across motion arms, approximately 17% of valid all-landmark tokens. The configured 0.5 fraction derives from the smaller gait-landmark budget. Mean target-minus-eligible motion is −0.000105 for uniform, 0.01749 for MAMP convention and 0.03784 for the robust mixture. These are source-balanced summaries of repeated mask draws. The common enrichment diagnostic resembles the robust sampler's own score; it is not an independent measure of clinical movement or tracking quality.

Notebook 16 shows that connected regions reduce the fraction of targets with immediate two-sided temporal neighbors from 0.699 to zero and the fraction with a visible anatomical graph neighbor from 0.988 to 0.512. Regions hide about 9.9% of valid tokens. Whole trajectories hide about 9.5% and interior gaps about 25.3%; these two mask families have coverage audits but no trained comparison. Zero immediate brackets does not mean all temporal context is absent. The 75,000 draws are repeated uses of 625 clips, not independent observations.

All five arms substantially reduce their training objective. Mean masked prediction loss at steps 1 and 1,200 is 15.0778→0.8493 for motion uniform, 15.0418→0.8355 for MAMP, 15.0239→0.8644 for robust motion, 15.2323→0.8393 for region uniform, and 15.1253→0.9479 for connected regions. Each arm has its own moving teacher and feature coordinates, so comparing these losses does not rank semantic usefulness.

The predictor diagnostic is more informative when its own matched and mismatched targets are compared on the same control clips. Every one of the 375 trained fold/seed/evaluation-mask rows has greater error for a target taken from another source video. Mean gaps are 0.435368 for motion uniform, 0.432875 for MAMP, 0.428370 for robust motion, 0.418305 for region uniform and 0.344790 for connected regions. Each arm has 75 such rows. The initial control mean is −0.003065 and is positive in only 33/75 rows; the two experiment copies reuse that control. These diagnostic gaps live in different learned teacher spaces and should not be ranked across arms or interpreted as 375 independent replications. They establish clip-related information, which may include posture, viewpoint or recording characteristics as well as movement.

The retained standardized readout diagnostics flag none of 875 rows as nearly constant. The raw predictor token/clip diagnostics also do not flag constant representations. Complete constant-feature collapse is therefore an unsupported explanation. The standardized diagnostic is insufficient to exclude weak raw feature variation or selective loss of endpoint-relevant directions.

Regularization remains a concrete concern. The largest candidate penalty, 10,000, is selected for 49/125 teacher mean-motion fits, 96/125 online mean-motion fits and 109/125 teacher mean fits, versus 0/125 initial mean-motion fits. Several controls are repeated across arms in these counts. Boundary selection warrants a wider common training-only search; it does not establish that widening the grid will restore learning benefit.

## Earlier results that should not be blended with this grid

Notebook 12's retained summary reports teacher R² −0.022633 for gait targets and −0.010449 for all-landmark targets; initial encoder R² is 0.048160, direct pose 0.129978, and the training mean −0.010609. The all-landmark-minus-gait contrast is 0.012184 with interval [−0.018303,0.050207]. The trained encoders again trail initialization. Its ridge grid ends at 100, and its feature/implementation details differ from Notebook 18. Notebook 08's fixed-alpha values are −0.540136 for gait, −0.422757 for uniform, −0.351389 for initial, and −0.003843 for direct pose. Numerical movement across these separate experiments cannot be attributed solely to masking or regularization.

The mild prepared-coordinate corruption results from Notebook 12 remove about four tokens, generally less than 1% of valid support. Left/scattered gaps evaluate 622 clips and right gaps 621, all from 93 sources. The unaltered laterality target remains the reference. Any paired corruption effect should compare unaltered and altered predictions on the same available clips. These masks are applied after interpolation/normalization, so they test sensitivity of prepared features; a real missing-measurement claim requires removal before preparation. Notebook 13 demonstrates that stronger information boundary synthetically.

Notebook 14's observed-future decoder diagnostic yields displayed coordinate RMSE 0.011, 0.013 and 0.025 at horizons 0.25, 0.50 and 0.75 seconds. Decoding predicted future features gives 2.51, 2.61 and 2.64. Each horizon has 24 common landmark endpoints from two synthetic test clips. Four training updates do not test an adequately trained forecasting system, and access to observed future features makes the first condition a diagnostic rather than a deployable forecast. Several forecasts from one prefix are not a recursive rollout. The tutorial's 611 real clips/1,814 clip–horizon eligibility count is a preparation audit, not trained forecasting evidence.

## Concrete recommendations for the revisions

1. Center the paper on a physical-representation evaluation question: does a learned representation preserve a side-sensitive observable under sensible anatomical transformations and controlled context removal? Make the trained-versus-initial discrepancy and predictor/readout separation the empirical contribution.
2. Distinguish mathematical guarantees from learning. A sign-constrained readout can satisfy reflection by construction; zero symmetry error is then a correctness check. Its prediction accuracy and gain over a matched initial encoder remain separate tests.
3. State successive nulls as retrospective descriptions where they were not preregistered: no trained-over-initial gain; no target-eligibility advantage; no motion/region-mask advantage; no correct-clip predictor preference; no demonstrated forecasting benefit. Do not imply the whole notebook sequence was planned before inspecting results.
4. Make clinical conditions motivate different kinds of bilateral coordination, while identifying the actual endpoint as a coordinate-derived displacement contrast. There are no affected-side labels, calibrated clinical laterality ground truth, or demonstrated condition-specific diagnostic outcomes in these extension experiments.
5. Add the new trained-minus-initial uncertainty table only with its analysis date, fixed-model conditioning and exploratory status. Retain unfavorable outcomes in every version. Avoid framing a zero-spanning mask interval as evidence that all masks are equivalent.
6. Use three complementary figures: the exact training/evaluation separation; a mask audit showing a successful change in available information; and the initial/trained readout comparison with paired contrasts. A small clearly labeled synthetic future-decoding failure can belong in an appendix, without competing with the real-data result.
7. Prioritize expanded ridge regularization, support-only/direct-coordinate controls, summary-component ablations and a stagewise input/target agreement audit using existing checkpoints. These should be declared before running and preserved alongside the original scores. Reserve new sources or an external cohort for stronger confirmation because the present outer folds have repeatedly informed development.
8. Defer a larger training grid until a focused training-source pilot identifies a discriminating question. An explicit reflection penalty, changed regularizer, motion target or forecasting objective each merits its own matched comparison. No present result justifies bundling them into a claimed successful physical world model.

The multi-stage story is credible when its decisions follow the evidence: anatomical consistency checks establish the measurement contract; target-eligibility and motion/structure experiments probe the learning intervention; initial controls expose the weak learned contribution; and predictor/readout disagreement identifies the next unresolved mechanism. The paper becomes stronger by making those inferential boundaries visible.
