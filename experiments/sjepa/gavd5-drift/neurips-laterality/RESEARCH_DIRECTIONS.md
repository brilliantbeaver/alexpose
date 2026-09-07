# From laterality checks to useful predictive representations

Research assessment and tutorial extension, September 7, 2026. The completed protocol and notebooks 00–06 are unchanged. The new notebooks are exploratory experiments and teaching material, with synthetic demonstrations clearly separated from evidence on gait recordings.

## Recommended direction

The most promising research question is **which geometric constraints help a JEPA preserve movement information that improves prediction**. The completed study offers a useful starting point because it separates response to reflection from useful prediction, and the two outcomes did not improve together. A stronger contribution would explain that separation and show when a proposed training change helps an observable movement task.

Start by checking the input and read-out in Notebook 07. Then use the matched comparisons in 08 and 09 to test a small number of learning choices. Notebook 10 creates a genuinely past-only forecasting task, which provides the clearest next step toward temporal world modeling. An external study becomes especially valuable when it tests the selected method against an independently rated or measured outcome.

## What the completed experiment supports

The study contains 625 accepted clips from 93 source videos, five source-held-out folds, five training initializations, and two paired training recipes. The existing saved reports, rather than stronger language in some older manuscript passages, determine the following interpretation.

| Finding | Numerical evidence | Consequence for the next study |
|---|---|---|
| Useful prediction from the current learned features remains uncertain | Native R² 0.060, 95% interval −0.025 to 0.126; learned minus initial −0.018, interval −0.039 to 0.002 | Retain matched initial encoders and direct movement measurements; locate the weak link before increasing scale |
| Mirrored training improves the specified feature-consistency measure | Error difference about −0.0084, interval −0.0102 to −0.0069 | Geometric consistency responds to the recipe, but should be evaluated alongside useful prediction |
| The predictive effect of mirrored training is uncertain | R² difference 0.004, interval −0.006 to 0.013 | Do not describe augmentation as either a demonstrated predictive success or a universal failure |
| An exact sign rule can be supplied by construction | Algebraic checks pass for the constructed odd read-outs | Enforcing a property is insufficient evidence that pretraining learned useful content |

The intervals resample source videos conditional on the fitted pipeline. The strict feature test also fixes a particular transformation: anatomical token exchange with no change of feature channels. Its outcome does not rule out a different latent-channel action. Temporal pooling and linear decoding introduce further choices, so poor prediction cannot be attributed to the encoder alone.

## A new, inexpensive diagnostic

We applied the registered target formula to the saved, resized model inputs and compared it with the original target computed from observed poses. The finite overlap contains 623 clips from 92 source videos. Source-balanced direct agreement is R² 0.218, correlation 0.652, mean absolute error 0.041, and sign agreement about 70%.

This calculation does not fit a model. It is a newly performed, exploratory comparison of two measurement paths. It does not establish irreversible loss or an upper bound on a learned predictor: a different feature calculation could recover information that this repeated formula misses. It does show that input preparation deserves attention before a larger encoder is assumed to be the main remedy. Notebook 07 makes the calculation optional and reproducible, reports the excluded overlap, and explains why time averaging can also hide a movement distinction in a constructed example.

## The new notebook sequence

| Notebook | Executable work | Scientific question |
|---|---|---|
| [07 — Research questions and diagnostics](07_research_questions_and_diagnostics.ipynb) | Saved-result review, temporal-pooling counterexample, optional read-only input reconstruction | Is the measurement available through the input and read-out being used? |
| [08 — Matched-budget masking](08_matched_budget_masking.ipynb) | Gait-target versus uniform-target JEPA, common realized target counts, paired controls, optional local-data grid | Does target selection improve useful representations when the hidden-token budget and training recipe are matched? |
| [09 — Symmetry-aware JEPA](09_symmetry_aware_jepa.ipynb) | Base objective, reflection augmentation, and explicit token-alignment loss; prediction, consistency, and feature-variation checks | Does directly encouraging geometric consistency preserve information useful for signed movement prediction? |
| [10 — Past-only movement prediction](10_past_only_movement_prediction.ipynb) | Timestamp-aware preparation, leakage tests, future-feature JEPA, mismatched-future control, and observable forecasting baselines | Does temporal pretraining improve future movement prediction using only available past information? |

Each notebook provides connected step-by-step explanations, runnable code, inline figures, and guidance for interpreting favorable, inconclusive, or adverse outcomes. Source files under `tutorials/` generate the output-free canonical notebooks; helper code under `laterality_extensions/` implements the repeated mechanics and is independently tested.

### Notebook 08: make the masking comparison informative

Its strongest near-term contribution is an adequately controlled experiment rather than a new masking concept. Both target policies hide the same actual number of valid tokens. Merely applying the same percentage to 12 and 33 landmarks would change how much context remains and confound the comparison. The runner holds initialization, source draws, optimizer updates, anatomy-based pooling, and read-out fitting fixed. Random landmark subsets can test whether the particular anatomical selection matters, although several prespecified subsets are needed before generalizing beyond one draw.

The synthetic run is deliberately small. The optional local-data grid creates new models on the existing source partitions, or validates and reuses an exact completed comparison. Its 1,200 updates per encoder match the completed paper budget of 300 epochs times four updates per epoch. With five folds, five seeds, and two recipes, both studies contain 50 encoders and 60,000 optimizer updates. The real runner selects CUDA or MPS when available, keeps fold tensors resident on the device, batches the two VICReg views through one encoder invocation, and restores a multithreaded CPU fallback. These execution changes retain the full model, batch, fold, seed, and update design. A separately labeled `single_seed_pilot` retains all five folds but is conditional on seed 42 and does not replace the full analysis. This controls optimizer exposure, but it does not reproduce the original training schedule: the exploratory runner uses a constant learning rate, fixed teacher momentum, and a 0.5 mask fraction, whereas Notebook 03 used cosine learning-rate decay, teacher momentum increasing toward 1.0, and a 0.6 mask fraction. The [matched-budget parameter reference](docs/MATCHED_BUDGET_MASKING.md) documents every visible setting, its rationale, runtime policy, and limitation.

A fixed ridge penalty keeps test outcomes out of selection, but it is not claimed to be optimal. A final comparative study should specify training-only tuning, repeat initializations, and distinguish equal updates from equal computational cost. The 1,200-update value is a defensible matched budget, not a demonstrated minimum; any shorter or longer stopping rule should be selected using prespecified training-only evidence rather than outer-test performance.

### Notebook 09: judge symmetry by what it preserves

An explicit loss compares the reflected encoder tokens with the anatomically exchanged original tokens. The comparison includes masks, so unavailable observations do not become targets or false evidence of agreement. Feature variation and effective rank help reveal representations that look consistent because they are nearly constant. Prediction from the learned and initial encoders remains a separate test.

An additional low-compute direction, proposed but not implemented here, is to fit an orthogonal, involutive feature-channel reflection on training sources and evaluate it on held-out sources. That could reveal dependence on the chosen latent coordinates. It must keep the original strict identity-channel result unchanged, include a mismatched-pair control, and avoid fitting the transformation on test examples.

Equivariance training is established in [SIE](https://proceedings.mlr.press/v202/garrido23b.html) and [Soft Equivariance](https://proceedings.mlr.press/v202/kim23p.html); [seq-JEPA](https://proceedings.neurips.cc/paper_files/paper/2025/file/2f63d2963526bdd9ff1b8bcc2dc9905a-Paper-Conference.pdf) already studies invariant/equivariant JEPA world models. The potential novelty here is a controlled explanation of useful signed dynamics under noisy pose observation, rather than the first symmetry-aware JEPA.

### Notebook 10: establish a real information boundary

The forecasting tutorial rebuilds preparation from original timestamps and uses only the observed prefix to determine its reference position, scale, and available input. Changing coordinates, visibility, or timestamps of observations that remain after the prediction boundary cannot change that input. Moving an observation across the boundary would change what is available and is a different test. Future observations enter a separate teacher branch and scoring, without being passed into the context encoder or predictor.

A read-only feasibility check found 611 clips contributing 1,814 eligible clip/horizon examples from all 93 sources under the tutorial settings. These counts describe a new selected forecasting sample, not additional independent participants. They depend on duration and visibility, and some rejection counts concern individual horizons. No empirical forecasting model was trained during this development.

The executable comparison evaluates the usefulness of frozen **context** features after future-feature pretraining. Its read-out predicts future coordinates and is compared with persistence, constant velocity, direct past-pose regression, a matched initial encoder, and a model trained against a different source's future. It does not decode the JEPA's predicted latent vector or test an autoregressive rollout. Those would require further experiments. Likewise, observational prediction does not establish action effects, intervention planning, or a calibrated physical simulator. [V-JEPA 2](https://arxiv.org/abs/2506.09985) provides relevant precedent, not evidence that these capabilities are already implemented here.

## What would make the research contribution convincing?

Prespecify a small comparison, with useful movement prediction as the primary outcome and geometric consistency as an explanatory measurement. A favorable result needs to exceed meaningful baselines under matched conditions and remain interpretable across source splits and training initializations. If a geometric penalty lowers its own error without improving prediction, report that trade-off directly. If changing the input or read-out resolves much of the gap, that finding should shape the research question before more pretraining is added.

Do not collect a large number of internal variants and then present the best outer-test result as confirmation. The current dataset has already informed the hypotheses. Newly explored results should remain labeled as development evidence, and an independently specified external evaluation can provide a stronger subsequent test.

## Is expanding Notebook 06 worthwhile?

Yes, if the next work produces an actual subject-held-out evaluation with an independently useful endpoint. The existing notebook already checks configuration, dataset-scoped records, source files, and participant separation. More readiness displays offer limited scientific return.

Keep 06 intact and select the external dataset and outcome first. The [external evaluation assessment](docs/external_evaluation_assessment.md) compares CARE-PD for clinical gait ratings with original BMCLab recordings for measured dynamics, including their representation mismatches, dataset overlap, and terms to clarify. A later `11_external_subject_evaluation.ipynb` should begin with a checked numerical adapter and simple baselines, rather than a broad training grid. No external dataset was downloaded and no external evaluation was implemented during this extension.

## Running and preserving the work

Run the new notebooks in order. Their default training uses synthetic data only; real training requires explicit switches described in each notebook. Notebook 07 may read available aggregate results, and its optional reconstruction is a read-only analysis. Missing real artifacts are never replaced with synthetic scores labeled as real.

```bash
.venv/bin/python neurips-laterality/scripts/build_research_notebooks.py --check
.venv/bin/python neurips-laterality/scripts/verify_research_notebooks.py --execute-smoke --save-executed
```

The second command runs the extension tests and executes each notebook in a fresh kernel, retaining separate output-bearing review copies and graphics under `executed/research_extensions/`. Regenerate edited tutorial sources with `build_research_notebooks.py`. The original builder continues to own only 00–06.

The new package deliberately sits outside `laterality/`, whose source contents determine compatibility with the existing checkpoints. Existing notebooks, registered code, protocol, and result artifacts were preserved. Saved extension comparisons use private, separate directories under `artifacts/research_extensions/`; the forecasting pilot currently retains its objects in memory and explains the additional recording requirements for a full study. Synthetic results are software demonstrations, and local exploratory results do not become submission-ready or clinically validated by completing these notebooks.
