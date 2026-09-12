# Saved-model evaluation and revised next steps

11 September 2026. Run `gate-v2`, protocol `direct-v2`, notebook batch
`haic-xaGGjyT7`. Local artifact root: `outputs/future-innovation/`.

**The failed comparison is reproducible, and its dominant mechanism is now
confirmed on the real models. About 92% of the real head's loss to ridge disappears
when unsupported baseline-input weights are removed. The remaining loss tracks
an unregularized linear fit. A separate scaling defect dominates two inner
validation partitions. Zero initialization alone does not restore ridge's
performance.**

Keep the completed STOP. The next step is a separately versioned, calibrated
predictor experiment on the existing development cache, with safe preprocessing,
controlled RGB regularization and an explicit baseline-only candidate. Student,
S-JEPA adapter and full-data training should wait for a successful measurement.

This evaluation supersedes the artifact-access limitations and next-step ordering
in the [initial investigation](residual-head-root-cause-and-next-experiment.md).
That document retains the earlier synthetic reproduction as historical evidence.

## Evidence inspected and independently reproduced

The model directory contains five ridge baselines, 60 residual-head checkpoints,
60 fit summaries, five inner-selection tables, five split audits and five fold
receipts: **140 files**. The subsequently supplied cache, configuration,
predictions, reports and readiness records made full numerical evaluation possible.

- Verified the available frozen configuration, cache, model/prediction receipts,
scoring artifacts, readiness binding and audit-file hashes, and final report seal.
- Loaded every checkpoint, checked finite weights and identities, and reconstructed
  all 60 held-out predictions. Maximum difference from stored float32 corrections
  was **5.13 × 10⁻⁶**.
- Checked all **153,600 prediction rows**, including source/fold identity,
  training-only target scaling, shared baselines and valid-feature masks.
- Recomputed all 12 arm/seed scores and **8,000 source-bootstrap rows**
  (2,000 draws × four arms), reproducing the saved numerical evidence.
- Reconstructed all 60 selection grids from the **1,440 inner-fit records**.
- Replayed the two unstable inner fits and 15 real-skeleton outer fits for
  controlled, in-memory interventions. No diagnostic checkpoint was written into
  the experiment.

The run/configuration hashes match notebook 04, including run-contract hash
`f1bb01466fe89f9ba40040e6a66afc96d5b21388793c733a86e6a284b38ae3da`.
The production fingerprint remains
`3e2156ecb0845bc4fb9d87a352409d218b9f4b4f285f81a970595ca9cfa87ca2`.
Local computation used Torch 2.13.0; the original environment recorded
2.6.0+cu124. Saved-output reconstruction and inner replay independently check
the numerical compatibility needed here. Raw teacher inference and pixel audits
were not rerun; the retained audit evidence was checked.

## 1. The headline result is correct

| Predictor | Mean source-held-out R² | Gain over ridge |
| --- | ---: | ---: |
| RGB + nuisance ridge | 0.271699 | Reference |
| Real skeleton full head | 0.028456 | −0.243242 |
| Shuffled skeleton full head | 0.028473 | −0.243226 |
| Mismatched skeleton full head | 0.028645 | −0.243054 |
| No-skeleton full head | 0.028523 | −0.243176 |

The matched real-minus-no-skeleton increment remains **−0.00006651**, with
95% source-bootstrap interval **[−0.00027454, +0.00008082]**. There is no supported
skeleton improvement in these fitted models. The failure is not a notebook
display error, missing fold, corrupted checkpoint, target-unit mismatch or
incorrect aggregation of the saved predictions.

## 2. Unsupported weights explain most of the outer loss

The correction has the form `W_x x + W_s h(s) + b`. The direct baseline-input
branch has 609,792 weights out of 664,192 total parameters. It receives 2,382
features despite only 39–41 training clips per outer fold.

The actual centered training designs have ranks **38, 39, 39, 39 and 40**.
Consequently, 2,342–2,344 feature directions lie outside each training row space.
Across the 60 heads, **98.53%–99.91% of squared baseline-input weight magnitude**
lies in those unsupported directions.

I projected out only this component and reevaluated the original held-out
predictions. The intervention changes training predictions by at most
**6.78 × 10⁻¹⁴** in the float64 decomposition. It retains the learned skeleton
branch, bias and supported baseline-input component.

| Real-head diagnostic | Mean held-out R² | Interpretation |
| --- | ---: | --- |
| Original saved full head | 0.028456 | Frozen result |
| Remove unsupported baseline-input weights | 0.252867 | Recovers 92.26% of the loss to ridge |
| Keep only baseline-input correction, omit skeleton branch and bias | 0.028436 | Almost reproduces the entire failure |
| Remove the entire baseline-input correction | 0.271715 | Remaining trained skeleton branch adds only about 0.000016 R² |

These are post hoc interventions on the real held-out data, not synthetic
results or newly validated models. The 92.26% figure describes recovered R²
loss under the intervention; it is not a general causal variance partition.
Removing a branch after training also does not tell us how a skeleton-only
predictor would learn if trained independently.

## 3. The supported correction also erases useful ridge regularization

The residual head is trained on errors from the same samples used to fit ridge.
Because the wide baseline design spans all centered training samples, it can fit
those residuals using baseline inputs alone. No skeleton is necessary to obtain
an almost perfect training fit.

I computed the minimum-norm linear map from those same baseline inputs to the
training residuals. Its training error is below 1.1 × 10⁻²⁹. Adding it to ridge
effectively produces an unregularized interpolating predictor:

| Diagnostic | Mean held-out R² |
| --- | ---: |
| Ridge | **0.271699** |
| Ridge plus minimum-norm linear training-residual fit | 0.252822 |
| Actual head with unsupported weights removed | 0.252867 |
| Real head retrained with zero output initialization | 0.253120 |

The close agreement is strong evidence that the remaining correction mainly
undoes ridge shrinkage rather than contributing a useful skeleton prediction.

The zero-initialized refits used the original outer training sources, seeds and
selected hyperparameters, changing only output initialization. Their R² values
were **0.253204, 0.253031 and 0.253125**. Every seed still lost to ridge. These
15 fits are diagnostic interventions; inner selection was not rerun, so they
are not a fresh nested-validation result.

The original baseline penalties were 1000, 100, 1000, 100 and 1000. Its actual
training residual MSE was approximately 0.08959, 0.003782, 0.09477, 0.003450 and
0.09116, respectively. Held-out residual MSE ranged from 0.528 to 1.230.
Thus residual suppression is real, especially in folds 1 and 3, but the baseline
was not uniformly a nearly perfect fit. This refines the earlier hypothesis.

## 4. A scaling defect dominates two inner validation partitions

The scaler divides by `max(training_std, 1e-8)`. Several per-joint missingness
features are identically zero in an inner training partition but nonzero on its
validation sources. Ordinary fractions such as 0.21875 or 1 then become
**21,875,000 or 100,000,000**. The ridge coefficient for a constant training
column is zero. The randomly initialized residual-head coefficient need not be.

The affected partitions are:

| Outer / inner fold | Affected missingness columns | Maximum scaled magnitude |
| --- | ---: | ---: |
| 1 / 1 | 19 | 100,000,000 |
| 3 / 0 | 20 | 100,000,000 |

These are legitimate changes in observation quality, not a reason to remove the
clips. For example, `joint_missingness_25` in outer 1 / inner 1 was always zero
in training but reached 1 on validation. Outer 3 / inner 0 similarly exposed
`joint_missingness_26` and other landmarks. Exact columns and affected windows
are retained in the diagnostic JSON.

The two seed-7 real-head replays reproduced saved validation squared-error sums
to within **9 × 10⁻⁹ relative error**. Keeping each trained head fixed and setting
only those unsupported validation columns to zero produced:

| Outer / inner fold | Original replay MSE | Neutralized columns MSE | Ridge MSE |
| --- | ---: | ---: | ---: |
| 1 / 1 | 1.72401 × 10¹¹ | 1.33434 | 1.03726 |
| 3 / 0 | 3.07456 × 10¹¹ | 1.42125 | 1.15823 |

This identifies the scaling mechanism directly. It also shows that fixing it
alone does not make the head useful. The catastrophic values occur in inner
selection; all outer training partitions contain variation in those features.
The outer test inputs therefore avoid this particular 10⁸ scaling failure,
which is why the final R² values look moderate while inner selection is broken.

## 5. Every selected candidate was already worse than ridge

All **480 pooled candidate configurations** lose to ridge on inner validation.
The grid nevertheless must choose one because it contains no zero correction.
Every final head selected **200 updates and weight decay 0.1**, the largest
available values of both parameters.

| Outer fold | Ridge inner MSE | Selected head inner MSE across arms/seeds |
| --- | ---: | ---: |
| 0 | 0.957728 | 1.523–1.577 |
| 1 | 0.842812 | 5.578–5.691 × 10¹⁰ |
| 2 | 0.885443 | 1.451–1.499 |
| 3 | 0.947036 | 1.054–1.352 × 10¹¹ |
| 4 | 0.938115 | 1.332–1.354 |

Selecting the least harmful candidate is not evidence that a correction is
worth adding. If a baseline-only candidate had been eligible under these same
inner measurements, all 60 comparisons would have selected it. The outcome
would still be no evidence for a skeleton gain, with the baseline's performance
preserved.

![Real-model outer-fold interventions and inner-fold scaling failure](../../../work/artifacts/future-innovation-model-evaluation-2026-09-11/saved-model-diagnostics.png)

## Revised experiment plan

The earlier proposal to start with zero initialization is insufficient. The
recommended order is now:

1. **Repair the measurement contract in a new diagnostic run.** Fit a training-only
   variance/support mask and prevent unobserved feature variation from being
   divided by 1e-8. Apply the frozen mask consistently to training and held-out
   inputs; record unsupported observations. Keep the original clips and source
   folds. Do not estimate the mask from validation data or modify `gate-v2`.

2. **Preserve the baseline's regularization.** Use a small reference predictor
   that jointly fits the target from RGB/nuisance inputs and fixed skeleton
   summaries, with separately controlled regularization for the two blocks.
   Include the exact RGB-only baseline as a candidate. Fixed temporal-bin
   coordinates/velocities can provide an interpretable first skeleton feature
   set, with the same construction for real, shuffled, mismatched and no-skeleton
   controls. Freeze that construction before the diagnostic comparison.

   For any subsequent nonlinear residual model, avoid an unrestricted second
   linear map from all RGB inputs to the residual. Either freeze the RGB route
   or explicitly regularize it jointly with the baseline. Use zero initialization
   as a calibration safeguard, not as the entire repair.

3. **Allow no correction during inner selection.** Score ridge alone alongside
   every correction candidate using identical source weights, target masks and
   units. Record rejection when every correction is worse. Check selected losses
   for scale explosions and expose per-inner-fold results in notebook 03.

4. **Calibrate before drawing another scientific conclusion.** Require tests for
   a newly missing joint on a held-out source, a perfect-baseline/zero-residual
   case, noisy residuals without skeleton signal, a planted skeleton signal with
   competing RGB inputs, and a planted temporal signal removed by the designated
   shuffle. The current five-frame receptive field followed by global averaging
   makes demonstrated temporal sensitivity especially relevant.

   Cross-fitted residual targets remain a secondary candidate if a sequential
   skeleton-only correction is retained. They are no longer the first repair:
   stabilize preprocessing and remove the redundant RGB refit first. Cross-fit
   only inside the current training sources, restoring raw teacher units before
   combining targets from differently standardized subfits.

5. **Use the current 50 clips for development only.** Cache reuse makes this a
   small CPU experiment; another teacher run is unnecessary for these repairs.
   Compare the calibrated predictors under a separately identified model/scaler
   contract. These inspected outer folds are not an untouched confirmation set.
   Freeze the design and decision rules before evaluating reserved source videos.
   If the calibrated comparison remains flat, stop this target/horizon claim
   rather than escalating student capacity.

6. **Attempt distillation only after predictive evidence survives.** Compare raw
   skeletons, frozen pretrained S-JEPA and a frozen random S-JEPA under matched
   heads. Separately establish that the proposed innovation target is predictable
   from skeletons alone. Only then train and evaluate a student against matched
   non-distilled and raw-skeleton references. Adapter training remains conditional
   on evidence for the frozen representation.

No amended protocol, new scientific ADVANCE, or production-model fix is claimed
here. The current result remains a valid negative result for its frozen
implementation, and a poor basis for judging the ultimate value of skeleton
distillation. The full-clip contextual teacher target and source-versus-person
independence limitations also remain.

## Reproduction and retained artifacts

```bash
.venv/bin/python scripts/research_directions/future_innovation/diagnose_future_innovation_fits.py \
  --run-root outputs/future-innovation \
  --output work/artifacts/fi-fit-diagnosis-new.json

.venv/bin/python scripts/research_directions/future_innovation/evaluate_future_innovation_failures.py \
  --run-root outputs/future-innovation \
  --output work/artifacts/fi-evaluation-new.json \
  --replay-inner --zero-init-outer
```

The first command only reads fitted artifacts. The second reproduces scores and
post hoc interventions; its optional flags fit temporary diagnostic models in
memory. Both require new output files outside the input run. Neither rewrites
production provenance, checkpoints, notebooks, scores or gate decisions.

- [Full fitted-model decomposition](../../../work/artifacts/future-innovation-model-evaluation-2026-09-11/full-fit-diagnosis.json)
- [Verified scores, selection grids, interventions and replay evidence](../../../work/artifacts/future-innovation-model-evaluation-2026-09-11/final-evaluation.json)
- [Evaluation implementation](../../../scripts/research_directions/future_innovation/evaluate_future_innovation_failures.py)
- [Regression checks](../../../tests/test_future_innovation_failure_evaluation.py)

The evaluation tests reconstruct a complete synthetic run, verify that inspection
does not change its files, check the newly missing-feature case, and reject
incorrect score arithmetic even when its file checksum has been updated to match.
All nine focused checks passed; see the [test log](../../../work/artifacts/future-innovation-model-evaluation-2026-09-11/verification-tests.log).
All 337 input files retained their contents and modification times during the
completed analysis, and the production fingerprint was unchanged.
