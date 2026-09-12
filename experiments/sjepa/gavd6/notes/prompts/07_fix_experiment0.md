**Role**: You are an AI/ML researcher and research engineer specializing in world models, Joint Embedding Predictive Architecture (JEPA), statistical evaluation, and reproducible experiments on Slurm/HPC systems.

**Task**: Carefully and systematically repair Experiment 0 for the future-innovation distillation study. Use the executed notebooks in `notebook_runs/`, the matching artifacts in `outputs/future-innovation/`, and the verified investigation to understand the failures, implement the necessary changes, and run a separately versioned feasibility experiment using the existing teacher cache. Carry the work through implementation, calibration, evaluation, notebook updates, and a clear account of the result.

The goal is to establish whether past skeleton motion improves prediction of future teacher features after current video information and recording conditions have been accounted for. A successful repair produces a reliable answer to that question. The answer may be positive, negative, or inconclusive; full S-JEPA training and distillation depend on the evidence obtained afterward.

Work from the repository's current state. Reuse its existing models, data interfaces, scoring functions, artifact checks, notebook generator, and Slurm conventions wherever they remain correct. Preserve existing user changes. Resolve routine implementation choices using the code and evidence, and document consequential design decisions before running the revised real-data comparison.

**Scientific purpose and current evidence**

The frozen video teacher, V-JEPA 2.1, turns video into feature vectors. Experiment 0 predicts a future feature vector from the first 32 frames of a clip. Its reference model uses RGB features and nuisance variables, which describe recording conditions, framing, and observation quality. The experiment asks how much skeleton coordinate/confidence history adds beyond that reference and a matched model retaining joint-validity information.

The reference is ridge regression: a linear predictor with a penalty that restrains its coefficients. The current implementation then trains a correction on its residuals, the errors left by that reference. That correction receives both skeleton history and the original RGB/nuisance inputs. The investigation identified failures in this construction and in its preprocessing and selection.

Before interpreting another real-data result, check the repaired pipeline on deliberately constructed examples where the presence or absence of skeleton information is known. These calibration checks establish whether the method can detect the signal it is intended to measure and avoid inventing an improvement when that signal is absent.

Treat the following as verified starting evidence from `gate-v2`, using the `direct-v2` protocol. Check the saved files before relying on the numbers, and investigate any discrepancy rather than adjusting results to match this summary.

| Observation | Meaning for the repair |
| --- | --- |
| 50 clips from 43 source videos, with 39–41 training clips per outer fold | Model complexity and regularization must suit the available training data. |
| RGB+nuisance ridge R² = 0.271699; real-skeleton full head R² = 0.028456 | The added predictor substantially damages held-source performance. |
| Real-minus-no-skeleton increment = −0.00006651, with 95% interval [−0.00027454, +0.00008082] | The saved models provide no supported skeleton improvement. |
| Removing unsupported RGB-input weights raises real-head R² to 0.252867 | Those weights account for about 92.26% of the observed loss to ridge under this diagnostic intervention. |
| Refitting with zero output initialization gives R² = 0.253120 | Initialization alone does not repair the model. |
| Two inner validation partitions contain scaled missingness values up to 100,000,000 | Ordinary changes in joint visibility trigger a preprocessing failure. |
| All 480 pooled head candidates lose to their inner ridge reference | Selection needs an explicit baseline-only candidate. |

The saved prediction reconstruction and score aggregation were verified. Preserve the completed `STOP` as the result of that implementation. The weight-removal and zero-initialization results are diagnostic interventions performed after inspecting the results; they are not a fresh nested-validation result or evidence that the repaired model will pass.

**Read and trace the experiment before editing**

Start with these sources in order, then follow their relevant implementation references:

1. [Saved-model evaluation and revised next steps](../../docs/studies/future-innovation/saved-model-evaluation.md): the confirmed mechanisms and numerical evidence.
2. [Current direct-v2 protocol](../../docs/studies/future-innovation/direct-gate-protocol.md): the scientific question, data boundaries, controls, metric, and decision rules.
3. `outputs/future-innovation/`: inspect the frozen configuration, manifests, teacher cache, models, predictions, QC records, and reports. The copied local root contains the `gate-v2` artifacts despite its directory name.
4. `notebook_runs/`: inspect the executed results. The editable source notebooks are in `notebooks/experiments/future_innovation/`.
5. [Fitted-model diagnostic](../../scripts/research_directions/future_innovation/diagnose_future_innovation_fits.py) and [numerical failure evaluation](../../scripts/research_directions/future_innovation/evaluate_future_innovation_failures.py): reuse their verified calculations and relocated-cache handling where appropriate.
6. [HAIC execution guide](../../slurm/future-innovation/README.md), [notebook execution guide](../../slurm/future-innovation/NOTEBOOKS.md), and the relevant tests: understand execution, resumption, and artifact ownership.

The [initial root-cause note](../../docs/studies/future-innovation/residual-head-root-cause-and-next-experiment.md) is historical. Its early emphasis on zero initialization was superseded by the saved-model evaluation. The older experiment guide and distillation proposal also contain teacher-selectivity requirements that direct-v2 explicitly removed. Identify such differences rather than combining incompatible protocols.

Trace the complete path:

`CLI/Slurm → configuration → verified cache → source splits → controls and preprocessing → candidate fitting and selection → saved model → held-out predictions → scores and uncertainty → notebook/report decision`.

For each confirmed failure, identify where it originates, where its effect becomes visible, and the test that will demonstrate its repair. Keep confirmed causes, additional hypotheses, and proposed design choices clearly distinguished.

**1. Establish a separate, reproducible development experiment**

Create a new run root and explicit model/scaler/protocol versions for the repair. Preserve the original run, checkpoints, cache, reports, and executed notebooks. Capture a checksum and modification-time snapshot of the parent artifacts so their preservation can be verified afterward.

Before real-data refits, write down the chosen primary predictor, required reference models, feature construction, preprocessing rules, finite search spaces, tie-breaking, candidate-failure policy, seed treatment, metric, and decision thresholds. Use calibration fixtures to resolve model choices. Freeze these choices before inspecting the revised held-out scores, and record any later amendment as a separate experiment.

For the first repaired comparison, reuse the current 50 clips, source folds, cached inputs, teacher projection, target, and control definitions. This keeps the comparison focused on the implementation changes. The inspected sources are development data: a new run ID does not make them an untouched test set.

**2. Repair input scaling without changing target meaning**

`TrainingScaler.fit()` currently divides by `max(training_std, 1e-8)`. When a feature is always zero during training, a held-out value of 1 becomes 100 million. The old residual head can attach an arbitrary coefficient to that feature because training supplied no information about it.

Implement a versioned input preprocessor with the following behavior:

- Fit means, missing-value imputation, observed support, variances, and feature masks on the applicable training sources only, using the declared source weights.
- Define explicit behavior for entirely unobserved, constant, and near-constant columns. Columns without usable training support must have zero model influence across all partitions. Keeping their positions while setting transformed values to zero preserves a stable feature layout.
- Specify near-constant thresholds and scale floors with attention to feature units. A policy for bounded confidence/missingness fractions may differ from the policy for RGB features. Freeze the policy before real-data fitting.
- Save feature names/order, training-window IDs, support counts, means, variances, scales, masks, and the preprocessing version. Validate shapes, weights, finite values, and schema compatibility when loading.
- Record held-out observations in unsupported columns and transformed-value ranges. Retain legitimate difficult clips; evaluation observations must not be used to refit the scaler or decide which clips to exclude.

Separate input preprocessing from target standardization. The current scaler serves both purposes, and `assemble_oof()` reconstructs target scaling during verification. Preserve training-derived target means/scales, inverse transforms, and target-variance masks. An input mask must never silently erase held-out target variation.

Maintain historical loading behavior. Existing joblib objects refer to the original scaler class, so changing that class globally could change predictions from old checkpoints. Introduce explicit version dispatch or separate implementations and test that historical predictions remain reproducible.

Reproduce the newly missing-feature cases from outer/inner partitions 1/1 and 3/0, together with all-missing and near-constant fixtures. Show that the repaired transform prevents the old explosion and that changing held-out values cannot change any fitted statistic.

**3. Replace the redundant RGB residual fit with controlled regularization**

The current predictor has the form:

```text
prediction = ridge(x) + W_x x + W_s temporal(skeleton) + bias
```

The second RGB/nuisance mapping contains 609,792 parameters. Its 2,382 input features have centered training rank only 38–40, leaving many coefficient directions unconstrained by training. Even after removing those directions, fitting ridge's in-sample residuals from the same inputs largely reverses the original regularization.

Implement a jointly regularized linear reference:

```text
prediction = intercept + X W_x + S W_s

objective = sum_i w_i ||Y_i - prediction_i||²
            + lambda_x ||W_x||² + lambda_s ||W_s||²
```

Explain the implementation clearly: `X` is the safely transformed RGB/nuisance block; `S` contains fixed skeleton summaries; `Y` is the target in the current training partition's units; `w_i` balances sources; and the two penalties restrain the input blocks separately. Each matrix penalty is the sum of its squared coefficient entries, called the squared Frobenius norm. Leave the intercept unpenalized.

Use a numerically stable float64 solve. Specify summed versus averaged loss and source-weight normalization because these determine the effective penalty. With the skeleton block disabled and the RGB penalty matched, reproduce the RGB-only ridge solution under the new run's preprocessing. Keep a separately selected, shared RGB-only reference for all reported comparisons.

Construct interpretable skeleton features that preserve temporal position, such as ordered bins of valid coordinates, valid adjacent-frame velocities, confidence, and observation support. Freeze bin boundaries, units, feature order, and missing-joint behavior. Compute velocities only from valid observations; imputation must not manufacture movement. Fit any learned scaling or dimension reduction within training partitions.

Choose both penalties from finite, positive values by inner source validation. Include an explicitly disabled skeleton block rather than approximating it with an extremely large penalty. Decide before the real run whether this joint model is the primary feasibility predictor or a calibration reference for a separately specified nonlinear candidate.

If retaining a nonlinear candidate, remove the unrestricted second RGB map or regularize the RGB and skeleton routes jointly. A frozen-baseline, skeleton-only correction requires its own residual-target calibration. Zero output initialization can protect its starting prediction but does not address all the confirmed causes.

Cross-fitted residuals are optional follow-up work if sequential residual learning remains. Construct them using fits and hyperparameter selection confined to the current training partition. Convert subfit predictions to raw teacher units before combining residuals from different target scalers, and verify their relationship to the final baseline.

Report deterministic models honestly. Running the same linear solution under three seed labels does not measure optimization stability. Specify a prospective seed-policy amendment if a deterministic model becomes the primary predictor; otherwise, keep its results explicitly diagnostic. Preserve the effect, control, and source-bootstrap thresholds described below.

**4. Let inner selection reject every correction**

Add an explicit `baseline_only` candidate whose full prediction is the selected RGB-only reference. This is essential because every existing correction candidate lost during inner validation.

Evaluate all candidates on the same inner source partitions, target units, valid target dimensions, and source weights. Pool weighted error sums and weight totals consistently. Freeze tie-breaking and numerical tolerance, preferring the baseline when scores tie within that tolerance. Outer-test results must never select a penalty, feature set, checkpoint, or candidate.

Persist each candidate's identity, hyperparameters, per-inner-fold loss, pooled loss, improvement relative to baseline, validity/failure status, and selection reason. Keep failed and rejected candidates visible. Record numerical failures explicitly and define their effect on completeness in advance; a missing required comparison cannot produce a complete scientific result.

Save baseline-only winners as typed model artifacts that reference the correct baseline and reproduce an exact zero correction after reload. Zero training updates on a randomly initialized neural head would still produce a random correction. Extend model loading, completeness checks, and prediction generation to handle this legitimate candidate type.

Compute displayed training and validation losses at the same recorded model state, or label their timing. The current history records training loss before an optimizer update and validation afterward. Correct that reporting inconsistency without attributing the headline R² failure to it.

Verify that selection chooses the baseline when every addition is inferior. Also make clear that a nonzero candidate selected on inner validation may still lose on unseen sources; baseline eligibility is not a guarantee about outer performance.

**5. Preserve source isolation and make the temporal controls meaningful**

Retain five outer source folds and three inner source folds. An outer fold measures prediction on sources excluded from its training; the inner folds select settings using only the outer-training sources. Keep all clips from a source together at every boundary and save the identities used to fit every learned transform.

Construct the four controls from raw skeleton history before applying the common feature constructor:

| Arm | Required construction |
| --- | --- |
| Real skeleton | Original prefix coordinates, confidence, and validity. |
| Time shuffle | Permute four-frame blocks, moving coordinates, confidence, and validity together. |
| Clip mismatch | Use a different-source donor from the same current training, validation, or test partition, following the recorded context-matching rule. |
| No skeleton | Zero coordinates and confidence while preserving the original time-varying validity channel. |

Give every arm the same model family, feature-construction rules, search opportunities, and applicable training budget. Record nominal dimensions and supported features; parameter counts alone do not establish equal statistical flexibility. The no-skeleton arm retains validity information, so evaluate real-minus-no-skeleton separately from improvement over RGB-only ridge.

Validate donor identities and source exclusions. Preserve the distinction between partition-local donor matching based on context metadata and learned prediction preprocessing fitted only on training sources. Donor selection must not use targets. Interpret shuffle and mismatch results with their validity changes in mind: these transformations affect observation support as well as coordinates.

The current two kernel-3 convolutions have a five-frame receptive field followed by global averaging. Test whether the revised representation can learn a known temporal-order signal and whether the actual designated shuffle reduces that signal. Use this calibration to establish temporal sensitivity before interpreting a negative real-data comparison.

**6. Verify predictions and scores independently**

Preserve the declared predictive R². For each target feature, it is:

```text
1 - source-weighted prediction squared error
    / source-weighted error of the outer-training-mean reference
```

The reference is zero in training-standardized target units. Replacing it with a test-mean-centered metric would change the experiment.

Maintain one expected held-out prediction per window, arm, seed where applicable, and target feature. Verify identities, target units, shared RGB-only predictions, training-derived target masks, and the model artifact that generated each prediction. Preserve the declared intersection of valid target features across outer folds.

Pool held-out predictions across outer folds within each seed, balance source videos, and average featurewise R². Average seed scores afterward. Estimate uncertainty by repeatedly resampling whole source videos with replacement, keeping each source's clips together. This is the source bootstrap. Keep these paired draws aligned across arms and seeds, preserving source multiplicity when a source is sampled more than once. Retain 2,000 draws and report intervals plus the fractions with positive real gain and positive matched skeleton increment.

Extend the existing verification into a read-only numerical check that reloads selected models, reconstructs predictions, recomputes scores and paired bootstrap summaries, and compares them with saved reports. Reuse the forensic evaluation's verified calculations. Hash checks establish file integrity; independent recomputation also checks arithmetic.

Reject wrong units, altered masks, missing or duplicate rows, mismatched baselines, invalid checkpoint types, and incorrect aggregates even when their checksums were updated. Declare suitable numerical tolerances. The earlier local reconstruction differed from saved predictions by at most approximately 5.13e-6 under a different Torch runtime; this is context for validation, not a universal tolerance to copy without checking.

Explain that bootstrap intervals are conditional on saved fitted models. They do not include repeated fitting, model selection, or the adaptive redesign of this inspected cohort. Repeated seeds and bootstrap draws add no independent source videos. Source separation also does not establish participant separation across sources.

**7. Integrate versioning, cache reuse, and execution**

Make the new protocol work through configuration, CLI, notebook execution, model loading, readiness/report dispatch, and Slurm wrappers. A name such as `direct-v3` must be supported by code and contracts before it is offered as a runnable option. Preserve explicit loading paths for `legacy-v1` and `direct-v2`.

Record the primary model, feature schema, preprocessing version, candidate types, penalty grids, selection rules, seed policy, target units, thresholds, source splits, runtime, and code fingerprint. Keep checkpoint and report identities consistent with that contract.

Implement verified, read-only reuse of the parent teacher cache. Its binding includes the original run contract, and its copied index retains absolute HAIC paths. Resolve a supplied local parent root while checking each file's identity, digest, schema, shape, and original binding. Record parent run/config/cohort/cache/projection/audit hashes in the child run. Preserve the parent's manifests and embedded bindings.

The repaired CPU experiment must be runnable from the available cached arrays and audit records without loading the teacher, mounting unavailable raw-video paths, or allocating a GPU. Report reused teacher evidence as reused. If teacher encoding, temporal boundaries, projection, nuisance features, pose processing, or cohort changes, explicitly determine which cache and audit evidence must be regenerated.

Keep outputs and provenance writes in the child run. Test completed-stage reuse, interrupted-stage recovery, stage ownership, duplicate-writer protection, and rejection of changed or incompatible parent artifacts. Provide exact commands for local cached execution and HAIC/Slurm execution, identifying which commands were actually exercised.

**8. Revise the notebooks into a coherent explanation of the experiment**

Update the [canonical notebook generator](../../scripts/research_directions/future_innovation/build_future_innovation_notebooks.py) and regenerate the source notebooks in `notebooks/experiments/future_innovation/`. Retain historical executed notebooks and generator snapshots under `notebook_runs/`.

Build a clear progression from the scientific question through data alignment, teacher features, predictor fitting, and the final decision. Explain what each stage establishes and how the next stage uses its outputs.

| Notebook | Required improvements |
| --- | --- |
| 00 — question and worked example | Explain the prediction task, reference model, skeleton increment, and exact baseline-only outcome. Show run/protocol/root identity and distinguish local file presence, verified evidence, and scientific criteria. |
| 01 — cohort and alignment | Show clip/source counts by fold, input and target frame boundaries, pose confidence/validity, missingness, and the development status of inspected sources. |
| 02 — teacher features and validity | Explain target construction and prefix isolation. Show whether cache/QC were reused or rerun and their lineage. |
| 03 — matched predictors and controls | Expose every candidate's inner losses, baseline comparison, selected type/reason, fallback frequency, scaling diagnostics, model dimensions, and aligned training/validation losses. |
| 04 — results and next decision | Present all arm scores, real-minus-no-skeleton increments, seed or deterministic status, paired uncertainty, numerical verification, thresholds, and development/confirmation status. |

Use concise tables and explanatory plots where they improve understanding. Large inner losses may need a logarithmic axis; retain their numerical values in tables. Show small increments with enough precision to preserve their sign and scale. Keep full numerical precision in machine-readable artifacts and use defensible significant digits in prose.

Keep inspect mode read-only. Preserve the inventory helper's `present_locally` meaning, and label separately whether integrity or numerical reconstruction was checked. Clearly identify synthetic teaching examples so they cannot be mistaken for real experimental evidence.

**9. Demonstrate calibration and compatibility with meaningful tests**

Run focused regression tests for the confirmed failures, calibration tests that establish the predictor's intended behavior, and the relevant existing integration checks. Define synthetic fixtures, random seeds, and tolerances before observing their outcomes.

| Test | Evidence required |
| --- | --- |
| Constant, near-constant, and entirely missing inputs | Safe scaling, stable schemas, serialization round trips, and no held-out influence on fitted statistics. |
| Perfect baseline / zero residual | The exact fallback survives fitting, selection, saving, reload, and prediction without adding an arbitrary correction. |
| RGB-only signal with irrelevant skeletons | Fixed repeated synthetic draws show no systematic manufactured skeleton advantage; do not require every finite sample to have zero gain. |
| Planted skeleton signal with competing RGB inputs | The complete source-held fitting/selection/scoring path detects the specified additional signal. |
| Planted temporal-order signal | The chosen representation learns it and the designated shuffle reduces it. |
| All corrections inferior or a candidate fails | Selection, failure reporting, artifact completeness, and gate outcomes follow the frozen policy. |
| Source and donor isolation | Every inner/outer transform and donor assignment obeys its declared boundaries. |
| Analytical scoring and tampered reports | Metric identities, masks, seed treatment, bootstrap multiplicities, and independent numerical rejection behave correctly. |
| Relocated cache and resumption | CPU reuse succeeds without parent writes; changed artifacts, incompatible contracts, and unsafe concurrent writes are rejected. |
| Historical compatibility | Existing checkpoints retain their predictions and historical sealed results remain readable. |

Where applicable, show that a regression test reproduces the old failure before passing with the repair. Use small deterministic fixtures for exact invariants and repeated fixed-seed fixtures for statistical behavior. Synthetic positive controls can validate software but cannot authorize a scientific `ADVANCE`.

Run the relevant existing training, metrics, controls, causality, resumption, notebook, direct-protocol, and Slurm tests, plus appropriate syntax/import checks. Report actual commands and results. Distinguish local verification from HAIC execution and record any environment limitation with its concrete effect on completion.

**10. Evaluate the repaired run and state what follows**

Run the bounded development comparison on the existing cache after calibration and integration checks pass. Retain candidate outcomes, selected models, predictions, source/fold audits, preprocessing diagnostics, verification results, uncertainty, and the final decision. Confirm that the parent artifact snapshot is unchanged.

Preserve these direct-v2 scientific effect, control, and uncertainty thresholds for the repair:

| Criterion | Required result |
| --- | --- |
| Mean real gain over ridge | At least +0.05 R². |
| Shuffle comparison | Real gain ≥2 × max(shuffled gain, 0). |
| Clip-mismatch gain | At most +0.01 R². |
| Matched skeleton increment | Mean real-minus-no-skeleton strictly positive. |
| Stochastic seed stability | Real gain positive in all three seeds and ≥0.05 in at least two; matched increment positive in every seed. |
| Paired source-bootstrap stability | Real gain and matched increment each positive in at least 90% of draws. |
| Measurement validity | Complete valid input, target, teacher, control, split, model, and artifact evidence. |

A deterministic primary model requires the prospective seed-policy treatment specified earlier. The reported 95% interval remains separate from the 90%-positive rule. Keep incomplete/invalid execution distinct from a complete scientific `STOP`; retain `INCONCLUSIVE` when validity and point criteria pass but stability fails.

Label a passing result on the inspected cohort as development evidence and identify the independent-source confirmation needed before scaling. A repaired negative result concerns the specified predictor, target, and data regime. Use it to assess whether to stop this comparison or formulate a separate target, representation, or data study.

Keep the temporal claim precise: inputs use frames 0–31, while the projected person-region target at frames 38–39 was encoded using the full 64-frame clip and can reflect later observations. Report prediction of contextual teacher features. Full S-JEPA training, adapter training, and skeleton-only student distillation remain subsequent experiments requiring their own evidence and implementation.

**Deliverables and completion criteria**

Complete the implementation and retain a reviewable record of what changed and what was measured:

- Updated production code, tests, CLI/Slurm integration, canonical notebook generator, and regenerated source notebooks.
- A new protocol document under `docs/studies/future-innovation/` describing the chosen model, preprocessing, controls, selection, seed policy, thresholds, and compatibility decisions before the revised real-data run.
- A separate run directory containing frozen configuration, parent-cache lineage, selection records, selected models, held-out predictions, diagnostics, scores, uncertainty, and the final report.
- A repair/validation report under `docs/studies/future-innovation/` connecting each confirmed cause to its implementation change and verification evidence, with commands for reproduction and resumption.
- Updated execution documentation covering local cache reuse, HAIC submission, output locations, inspection, and recovery.
- A final assessment stating which repairs are verified, the observed scientific result, its limitations, and the specific evidence or work needed next.

Use descriptive filenames consistent with the repository. Keep machine-readable artifacts linked from the reports. A plan, synthetic smoke run, or checkpoint count alone does not complete the repair: the available cached development experiment must also be fitted and evaluated. If a concrete external blocker prevents a required step, complete the unaffected work and identify the remaining step and failed command without claiming it ran.

**Review the implementation and the conclusions critically**

After implementation, challenge whether an apparently positive result could arise from RGB refitting, unsupported input columns, target-unit mistakes, unequal search opportunities, source leakage, altered feature masks, duplicated seed results, or selection using outer-test outcomes. Trace each possibility through the saved artifacts and tests. Check that a correct negative result remains a valid outcome of the pipeline.

Review for duplicated logic, stale protocol assumptions, brittle absolute paths, incompatible checkpoint loading, misleading notebook labels, and weakened verification. Address concrete findings, rerun affected checks, and revise the documentation to describe the final implementation. Base scientific claims on the saved measured results and distinguish them from diagnostic interventions and hypotheses.

**Writing style**

Write for an advanced high-school reader with basic machine-learning familiarity. Explain technical terms when first introduced, then use them consistently. Keep the narrative connected: motivation, hypothesis, observed failure, confirmed cause, repair, validation, measured result, and implications for the next experiment.

Use natural, fluent paragraphs with concrete verbs and enough explanation to make each decision understandable. State what a method does, why it is needed here, and which evidence supports the resulting claim. Give exact effect sizes when they matter, and round displayed values to a defensible precision. Use tables for comparisons and tests rather than repeating the same explanation throughout the notebooks and report.

Avoid staccato slogans, rhetorical-question headings, repetitive cautionary statements, repeated “not X, but Y” constructions, inflated claims, and unexplained jargon. Avoid formulaic transitions and a mechanical audit voice. Explain uncertainty where it changes the interpretation, and keep implementation details close to the decisions they justify.

**Implementation reference**

All `fi_*.py` files below are under `src/gavd6_sjepa/research_directions/future_innovation/`.

| Area | Starting points |
| --- | --- |
| Input scaling and model types | `fi_residual_models.py`; introduce a focused joint-model module if useful. |
| Nested selection and model reload | `fi_nested_training.py` |
| Protocol, configuration, and cache/QC lineage | `fi_contracts.py`, `fi_feature_cache.py`, `fi_readiness.py`, `fi_validity_audits.py` |
| Controls and temporal features | `fi_controls.py` and the revised feature constructor |
| Metrics, verification, reporting, and decisions | `fi_metrics.py`, `fi_reporting.py`, `fi_direct_reporting.py`, `fi_gate_decision.py` |
| Notebook execution and inspection | `fi_notebook_workflow.py`, `fi_tutorial_inspection.py`, and the canonical generator linked above |
| CLI and scheduled execution | `fi_entrypoint.py`, `slurm/future-innovation/` |
| Existing coverage | `tests/test_future_innovation_*.py` and the diagnostic/calibration fixtures |
