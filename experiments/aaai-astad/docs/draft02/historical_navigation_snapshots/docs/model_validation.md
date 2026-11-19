# Model and workflow validation

The delivered code has passed executable software checks, a complete synthetic capstone run, and a small real-video extraction and normal-training pilot. These checks establish specific implementation properties. They do not establish accuracy on GAVD, clinical validity, or superiority over Random Forest. Full video extraction, full-cohort training, and externally validated performance remain work for the experiment runs.

## What was executed

The following command passed all 23 model and workflow tests in the existing Python 3.12 environment:

```bash
PYTHONPATH=experiments/aaai experiments/multiple-sclerosis/.venv/bin/python \
  -m pytest experiments/aaai/tests/test_model.py experiments/aaai/tests/test_workflow.py -q
```

That environment used Python 3.12.13, PyTorch 2.13.0, NumPy 2.4.6, scikit-learn 1.9.0, and pandas 3.0.5. The run completed without warnings. From the new experiment environment, the equivalent command is `uv run pytest tests/test_model.py tests/test_workflow.py -q` after `uv sync`.

A separate end-to-end check used the current GAVD manifest inventory, then explicitly selected the synthetic corpus. It trained the normal model and all four progressive stages, fitted seven comparison methods, selected readout settings using validation data, wrote aligned test predictions, calculated paired recording-cluster bootstrap summaries, and ran anomaly calibration. This used 60 synthetic windows and one training epoch per stage. The synthetic patterns have no physiological interpretation, so their classification scores are not reported as research findings.

The real-media pilot processed eight annotated sequences. The original rule requiring every coordinate throughout a window to be observed retained no windows. The final quality rule retained 12 windows from six sequences and six source videos. Each window had at least 75% fully observed joint patches, with at least 50% coverage in both halves. These windows contained 7.91% missing XY coordinates and 87.11% valid joint patches overall. A patch is valid only if all its four frames have both coordinates. Missing coordinates remain NaN in the corpus.

One normal-only training epoch used the seven normal training windows from this pilot. The loss and gradients remained finite. All 12 pilot windows produced finite residual scores, and invalid target patches received zero scoring coverage. The observed-feature linear-probe path also ran with finite probabilities on the available normal and myopathic training labels. These are integration checks on an incomplete cohort, not performance estimates. Parkinsons and cerebral palsy were absent from the pilot, and the full classifier's five-class coverage guard remains mandatory.

The repository's older Python 3.14 environment also completed the integration checks, but its NumPy 2.2.6 and scikit-learn 1.8.0 combination emitted matrix-multiplication runtime warnings. Coefficients and exported probabilities remained finite in the checked runs. The cause was not established. The tested Python 3.12 environment avoids these observed warnings, and no warning suppression was added to the implementation.

## Verified model properties

| Property | Executable challenge | Observed outcome |
|---|---|---|
| The whitelist is fixed. | Attempt to instantiate a 33-joint model and inspect the declared indices. | The model rejects other joint counts and accepts the specified 16-joint input order. |
| Masking does not use motion. | Draw 10,000 independent masks and inspect token inclusion frequencies. | Every draw has the required hidden count, and inclusion frequencies remain close to the uniform expectation. The sampler accepts no coordinates or velocities. |
| Hidden observations cannot enter context attention. | Add 100 to every hidden coordinate while keeping visible coordinates unchanged. | Predictor outputs remain identical. Teacher outputs change because the target legitimately sees the full window. |
| Missing observations cannot become zero-valued evidence. | Keep one coordinate missing in a patch and replace its remaining finite components by very different values. | Predictor outputs, target outputs at valid positions, and pooled embeddings remain unchanged because invalid patch keys are excluded. |
| Missing targets are never supervised. | Introduce missing patches in both halves and independently calculate the valid-target loss and center. | Loss and center match the observed masked patches only. Loss and gradients remain finite. |
| Future missingness cannot alter prediction. | Remove observations only in the future suffix. | Predictor outputs remain identical because all hidden queries remain present. The supervision mask changes, as it should. |
| Forecasting cannot inspect future observations. | Replace the entire future suffix with unrelated coordinates. | Predictor outputs remain identical. The loss rejects masks that expose any position in the configured future suffix. |
| Normalization and forecast cutoff agree. | Use an odd number of temporal patches and request a context shorter than half a window. | The cutoff rounds upward to cover all frames used to estimate scale. Context fractions below one half are rejected. |
| The teacher receives no gradient. | Backpropagate and take an optimizer step before invoking EMA. | All target gradients remain absent and target weights remain unchanged until the explicit EMA update. |
| EMA runs after the optimizer. | Compare target parameters with their exact old-target/new-online weighted sum. | The update matches the specified momentum. The teacher remains in evaluation mode. |
| VICReg reaches the online representation. | Differentiate its variance term near a collapsed representation. | Both view projections receive nonzero variance gradients. Exactly constant features incur a large penalty. |
| Scoring uses a fixed mask bank. | Score examples together, separately, and in repeated calls. | Scores agree to floating-point tolerance across batching, and repeated calls use identical masks. |
| Missing targets have no residual attribution. | Score partially observed windows and inspect coverage. | Invalid patches have zero coverage and NaN attribution. Valid scored patches and window scores remain finite. |
| Empty observed context or targets fail clearly. | Remove all context, all targets, or the entire input. | The relevant operation raises an explicit error rather than returning a misleading zero score or NaN loss. |
| Stochastic positions affect predictor requests only. | Change the SPE random seed with fixed pose and masking. | Predictions change while teacher features remain identical. Evaluation disables SPE noise. |
| SPE does not change mask draws. | Compare successive loss calls with SPE enabled and disabled. | Masks and mask-generator states remain identical because positional noise has its own random stream. |
| Checkpoints identify their data. | Load with a changed manifest hash and attempt to overwrite the normal anchor. | Both operations fail by default. Checkpoints include online weights, teacher weights, and target center. |
| Training excludes held-out coordinates. | Replace every held-out pose with NaN before training. | Normal-only training yields the same losses and weights. A recording spanning splits is rejected. |
| Progressive training preserves the anchor. | Compare all normal-model parameters before and after adaptation. | The supplied normal model remains unchanged because progressive training operates on a copy. |

## Verified workflow properties

The readout-selection test replaces test embeddings with NaN and changes every test label. Validation selection and learned coefficients remain unchanged. This tests actual code paths rather than relying only on a written protocol.

The capstone test verifies identical sequence identifiers, source identifiers, labels, split labels, and ordering across all seven exported prediction tables. It also checks that probabilities are finite and sum to one. This alignment is necessary for a valid paired comparison.

The anomaly-calibration test creates 20 normal validation recordings, each represented by multiple sequences and windows. Calibration still counts 20 recordings. At a nominal 5% rate, its finite-sample threshold is the expected highest calibration score. The small synthetic tutorial has only one normal validation source, so its 5% threshold correctly becomes infinite. This exposes insufficient calibration data rather than silently reporting a finite threshold with an unsupported guarantee.

The workflow review found and corrected five integration issues: an ignored frame-number-base setting, progressive checkpoint names that lacked anchor-weight identity, an ambiguous bootstrap estimand label, duplicate mean-pose readout fitting, and CPU/device assumptions in anomaly scoring. The paired output now explicitly names its sequence-weighted estimand. Source-weighted point estimates are reported separately.

The review also corrected pandas read-only-array handling in sample weights and applied source weights to the linear probe's scaler. Class balancing uses source-weighted class totals, which prevents raw window counts from determining the class correction. A separate test checks that the probe's imputer learns medians only from training observations, even when held-out values are extreme, and handles a completely missing feature without leaking held-out values.

## Limits that remain scientifically important

This model adapts the 2024 Skeleton-JEPA architecture. It uses uniform masking, a smaller transformer, VICReg, and an online-encoder downstream readout. It is not an exact reproduction of the published benchmark. The literature review distinguishes these decisions and the optional StoP-inspired predictor noise from the separate 2026 stochastic-latent S-JEPA work.

The model predicts latent features for hidden joint patches. It does not generate visible skeleton coordinates. A coordinate decoder would require a separate, explicitly trained readout and a reconstruction evaluation. Future suffix prediction must be trained as a separate forecasting task. A model trained with arbitrary masks should not be described as a validated forecaster.

VICReg discourages collapsed dimensions and redundant features. It does not guarantee separation between health-condition classes. An exactly constant representation can have zero variance gradient even though its variance penalty is large. Inspect `representation_diagnostics` on encoder embeddings and compare downstream probes rather than treating a finite loss as proof of useful representation learning.

Latent residuals and per-token attribution identify prediction mismatch. They do not prove disease, etiology, or causal explanation. Pose errors, occlusion, viewpoint, assistive devices, and camera motion can also produce high residuals. A normal-only anchor must remain separate from the model adapted on abnormal examples.

The valid-token mechanism avoids treating missing joints as measured zeros. It does not remove missingness as a possible confound. Attention can still depend on which joints are observed, and different conditions may have different occlusion patterns. The capstone therefore includes a pose-availability-only classifier. The coverage thresholds are engineering choices that must be fixed before comparing performance and accompanied by exclusions per condition and source. Retaining more windows after this pilot is not evidence that the new threshold improves clinical accuracy.

Source-video separation does not establish participant separation. Recording-cluster bootstraps account for dependence within known source videos but cannot correct unknown repeated participants across videos. The current bootstrap intervals describe sequence-weighted evaluation sampling, and they exclude and count resamples missing a prespecified class. They do not measure training-seed uncertainty. A small number of valid resamples produces no interval.

SPE training-time noise is an ablation of predictor regularization. It is not an uncertainty posterior or evidence of diverse plausible motion futures. The stochastic-latent-variable extension is deferred until an explicit conditional prior, target-informed posterior, regularized training objective, and calibration evaluation can be implemented and tested.
