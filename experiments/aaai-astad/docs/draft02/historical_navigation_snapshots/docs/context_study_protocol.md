# Implemented protocol for the context study

This document describes the code in `aaai_gait/context_prediction.py`, `predictive_profile.py`, `study_data.py`, `linear_context.py`, `study_baselines.py`, `study_runner.py` and the optional `teacher_decomposition.py` diagnostic as reviewed on November 15, 2026. It specifies an implemented experiment, not a claim that its hypotheses succeeded. Consult [current results](context_study_results.md) and the [implementation review](context_study_review.md) for execution status. The [earlier five-class results](results.md) belong to a different experiment.

The question is whether the predictive help provided by other joints has a more useful normal-reference profile than raw prediction discrepancy. The primary endpoint is source-level normal-versus-other-folder ranking. The task is masked completion, not forecasting, diagnosis or identification of a biological cause.

![The comparison from normal fitting to source evaluation](draft00/images/neuromechanics_experiment.svg)

## 1. Fix the endpoint and the observation unit

Read all current per-sequence CSVs in twelve `data/gavd` folders, excluding `source/`. Assign `normal` to binary label 0 and every other folder to label 1. The latter includes `abnormal`, `antalgic`, `cerebral palsy`, `exercise`, `inebriated`, `myopathic`, `parkinsons`, `pregnant`, `prosthetic`, `stroke` and `style`.

This is a folder-label endpoint. It does not assert that exercise, pregnancy or a walking style is necessarily pathological. Parkinsons, stroke, cerebral palsy and myopathic gait motivate selected movement hypotheses, but they do not exhaust the binary positive class. Myopathic gait is muscle-related rather than one neurological etiology. Folder labels are not verified clinical diagnoses.

The fresh annotation inventory contains 1,874 sequences, 458,116 rows and 348 unique source IDs, including 32 normal and 316 abnormal sources. These are pre-extraction counts. A source can occur in several abnormal folders. A normal/abnormal source conflict raises an error. Source IDs are the assignment and evaluation units; they do not establish independent participants across uploads.

The manifest records CSV hashes, frame ranges, camera views, source IDs and whether a source appeared in the earlier retained cohort. That prior-use flag supports an exposure audit. A newly defined analysis on previously examined sources remains exploratory, even when its source folds are internally separated.

## 2. Recover observations without manufacturing trajectories

`ingest_study.py` reads fresh annotations and reuses decodable local media. Optional downloads follow normal-first, then source-ID order with explicit time and worker budgets. Downloads and acquisition logs stay under `data/videos-gavd/`. Failures and unattempted sources remain visible in the ledger.

The adapter uses `ambient.pose.keypoint_extractor.SequenceKeypointExtractor` with annotated-person crops. It retains raw full-image XY, confidence, source FPS, frame indices and provenance. Inspect overlays to confirm person identity and timing. A decoded frame and a plausible pose do not prove that a current video rendition still matches the original annotation interval.

The shared window rules are:

| Setting | Implemented rule |
| --- | --- |
| Coordinate input | Relative image-plane XY for the fixed sixteen joints. |
| Confidence | Values below 0.5, or nonfinite confidence, make coordinates missing. |
| Sample rate | 15 Hz. Exclude source FPS below 15 instead of upsampling. |
| Missing frames | Resampling uses adjacent valid source frames and does not bridge a missing-frame gap. |
| Window | 32 frames, stride 16, within one annotated sequence. |
| Translation | Center each frame on its pelvis. |
| Scale | Use observed finite torso measurements from the first sixteen frames. |
| Token | Four frames of one joint, containing eight XY numbers. |
| Token validity | Every XY value in the four-frame patch must be finite. |
| Window admission | At least 75% valid tokens overall and 50% in each temporal half. |

NaNs remain in the arrays. The model excludes invalid keys, targets and pooled contributions. Numerical zeros used inside token projection are not filled pose observations. Full teacher targets also exclude invalid keys. An infinite coordinate is corrupt input and raises an error.

The normalization is shared by all methods. Own-joint prediction is therefore conditional on a body reference computed from pelvis and torso observations; it is not independent of other joints at preprocessing time. Neither normalized speed nor coordinate error is physical velocity or force. Quality filtering is a selection mechanism and must be reported by condition, source and view.

## 3. Fix joints, training masks and scoring support

The complete model allowlist is:

```python
(11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
```

The [deduplicated joint table](joint_table.md) names their feature uses. No face or hand-tip point enters the model. All sixteen joints remain eligible for training masks. Eight time patches times sixteen joints gives 128 tokens.

Training samples exactly 64 hidden positions uniformly without replacement. Its sampler reads no coordinates, velocities, confidence or labels. Observation validity is evaluated separately afterward. The study changes the original Skeleton S-JEPA's motion-aware masking to the requested uniform scheme; it is a paper-based adaptation rather than an exact reproduction. [Original S-JEPA project](https://sjepa.github.io/)

Scoring uses the same bank of sixteen uniform masks for every window, with mask seed 17. The bank is independent of observations, window order, batching and model fitting. No mask is redrawn to obtain a more favorable or more visible target.

For a requested hidden target to be eligible, the target patch must be valid and at least two valid visible patches of its own joint must remain. All three branches and both target types use that same eligibility. A valid target with a nonfinite loss is a model failure, not an exclusion.

The primary readout uses three groups:

| Region | Original BlazePose indices | Meaning |
| --- | --- | --- |
| Left leg | 25, 27, 29, 31 | Knee, ankle, heel and foot index. |
| Right leg | 26, 28, 30, 32 | Knee, ankle, heel and foot index. |
| Arms | 13, 14, 15, 16 | Both elbows and wrists. |

Hips and shoulders remain training and context joints. They are excluded from the primary readout because pelvis centering and torso scaling create strong geometric relationships. The three-region choice is an experimental design, not a clinically validated partition.

### Freeze profile eligibility before assigning folds

`freeze_profile_eligibility` applies the fixed mask bank to observation-validity arrays. It counts distinct eligible physical patches for each of the twelve readout joints. A physical patch is identified by source, sequence, original joint and its time interval. Repeated masks and duplicate coverage from overlapping windows do not increase this count.

A source must provide at least four distinct eligible patches for every readout joint. Sources failing that rule are excluded before fold assignment, and their reasons and per-joint counts are saved. All accepted windows from a retained source then enter the relevant source role. This gate uses finite-coordinate validity, time identities and fixed masks. It does not use coordinate magnitude, model errors or abnormal performance. It can still induce label-related selection through visibility.

## 4. Separate four outer folds and five inner normals

`make_outer_folds` assigns each retained source to exactly one of four outer-test folds. It balances normal and abnormal counts and spreads abnormal category memberships through a deterministic assignment. A multi-category source keeps one assignment. The default seed is 42.

For each fold, exactly five non-test normal sources are selected by a deterministic hash order for the inner development/reference role. Remaining non-test normal sources fit the anomaly models. Other non-test abnormal sources are auxiliary and do not fit encoders, context heads, ridge models or normal references.

| Role in one fold | Use |
| --- | --- |
| Normal fitting sources | Fit S-JEPA, matched heads and ridge predictors. |
| Five inner normal sources | Read-only training diagnostics and normal profile estimation. |
| Outer-test sources | Evaluate normal-versus-other-folder rankings after fitting. |
| Non-test abnormal sources | Available only to the separately labeled supervised classifier task. |

Every sequence and window inherits its source role. Save the registry and its hash before fitting. A source cannot be in both fitting and reference sets or in both fitting and test sets. No encoder pretrained on all four folds is reused across outer folds.

Five reference sources are small. They do not establish a reliable clinical normal range or a useful finite-sample 5% alarm threshold. The study therefore emphasizes ranking and paired uncertainty, not an operational screening cutoff.

## 5. Pretrain on normal gait with a fixed update budget

The default `ContextConfig` uses a 64-dimensional, two-layer, four-head encoder; batch size 16; AdamW learning rate 0.0003; weight decay 0.05; and 512 pretraining updates. Sampling first chooses a normal fitting source uniformly, then a window from that source. This reduces domination by sources with many windows.

The online encoder reads visible tokens. The full-input teacher supplies detached centered target distributions. Teacher temperature is 0.06 and predictor temperature is 0.1. Pretraining minimizes latent cross-entropy plus 0.1 times the implemented VICReg objective. The teacher updates after optimization with EMA momentum 0.996. Target centering also updates during pretraining.

VICReg discourages collapse and redundant dimensions. It does not force health-condition clusters apart or prove useful features. Histories record loss components, embedding/projector variation, effective rank and teacher drift. Normal development diagnostics are recorded at update zero and every 64 updates, including the final update. The current recipe runs a fixed budget; it does not automatically select the best abnormal test result or establish convergence.

After pretraining, freeze the online encoder, teacher, target center and their parameters. Positional noise is off in this primary experiment. The older StoP-inspired positional-noise ablation and learned stochastic latent-variable proposals are separate questions.

## 6. Fit six matched heads without sharing prohibited information

For each target type, initialize the own, patch-local and full heads identically. Fit each head with its own optimizer for 512 updates, using shared source-balanced batches, uniform masks and randomly selected training joints. Training joints are selected independently of mask, observation and label. An entirely unsupported training draw raises an error rather than being redrawn.

Each head uses a positional query, cross-attention and a small output network. The frozen encoder re-encodes only the permitted measurements for that target. The restriction is applied before the first contextual attention layer and remains effective through padding masks. Globally contextualized features are not passed to a restricted head.

| Branch | Target joint's other visible patches | Other joints at the target time patch | Other joints at other times |
| --- | --- | --- | --- |
| Own | Allowed | Blocked | Blocked |
| Patch-local | Allowed | Allowed | Blocked |
| Full | Allowed | Allowed | Allowed |

Patch-local context includes four frames, so it includes short motion as well as geometry. The same hidden target values are unavailable to every branch. Visible own-joint patches may lie before or after the target, which makes this completion rather than forecasting.

Coordinate heads predict the eight normalized XY values of the local patch and use mean squared error. Latent heads predict the frozen teacher distribution and use `KL(teacher || prediction)`. The teacher has full observed context, so its answer can encode neighboring movement. Coordinate targets test this alternative explanation. Never subtract coordinate MSE from latent KL or compare their magnitudes as if they shared units.

A frozen encoder pretrained with broad context may behave differently with only one joint. The access comparison measures this predictor's use of information; it is not a direct measure of biological coupling. Ridge and random-backbone controls help investigate that distinction.

## 7. Aggregate matched errors into a source profile

For each joint and target type, use identical eligible targets for own, patch-local and full errors. Average target exposures within a window, windows within a sequence, then sequences within a source. Repeated masks contribute to the window estimate but do not create independent observations or extra physical support.

For source `s` and joint `j`, define:

```text
benefit(s,j) = mean_own_error(s,j) - mean_full_error(s,j)
```

Average the four joint benefits in each region equally to produce a three-number profile. Form the nonlinear reference distance only after this source aggregation. Missing regions are not imputed as zero.

![The reference describes signed benefits in three regions](draft00/images/support_profile.svg)

For each target type, fit a separate normal reference from the inner normal source profiles. With sample variance `v_g` in region `g`, use:

```text
reference_variance_g = max(0.5*v_g + 0.5*mean(v), 1e-8)
score(s) = mean_g((benefit(s,g) - normal_mean_g)^2 / reference_variance_g)
```

This two-sided score increases for unusually low or high help. It is a regularized distance, not a probability, likelihood or validated clinical range. At least four scorable reference sources are required by the reference API; the runner reserves five and fails if its frozen support rules are violated.

Retain the signed profile, its constituent own/patch/full losses and the fixed one-sided contrast `raw_full - raw_own`. Score direction is not changed after inspecting abnormal results. The primary contrast compares the latent profile distance with raw full-context KL over the same twelve readout joints and identical aggregation.

## 8. Keep the controls matched and accurately named

| Control | Implemented purpose and boundary |
| --- | --- |
| Own/full reference distances | Test whether subtracting losses adds information beyond a constituent loss profile. |
| Patch-local branch | Test how much benefit is already explained by other joints in the target four-frame interval. |
| Coordinate target | Investigate contextual teacher effects using an observable local target. |
| Ridge context predictors | Three normal-only coordinate predictors with the same information restrictions, targets and mask identities. |
| Random backbone | Skip JEPA pretraining, freeze its random encoder/teacher, and still fit matched heads on normal sources. |
| Four-patch budget | Select exactly four eligible physical patches per joint using a fixed hash order and reuse the same selection across methods. |
| Leave-one-reference-out | Refit only the benefit reference with one inner normal omitted, keeping models, losses and support fixed. |

Ridge uses source-balanced training draws, fixed `alpha=10`, allowed coordinate values and explicit availability indicators. Its normal training draw budget is 256. Before scoring, every readout joint/time position must have at least eight eligible training examples and all three fitted branches. Failure raises an error. The primary neural cohort is not shrunk to accommodate an under-supported ridge fit.

The random-backbone run uses the same head initialization policy and head-fitting draws. It does not compare against an entirely unfitted head. Its latent targets differ because its teacher is random, so the coordinate comparison is also important.

Reference sensitivity runs do not retrain models and are not additional independent studies. No edited observation replaces a lost fixed-budget patch with a more favorable one. A published SeeKer or STG-NF adaptation remains unvalidated; see [baseline feasibility](published_baseline_feasibility.md). Neither RF nor ridge is a numerical reproduction of those methods.

## 9. Evaluate within folds and pair uncertainty by source

Compute AUROC and abnormal-class average precision inside each fitted outer fold. The primary summary is:

```text
sum_f(normal_test_sources_f * AUROC_f) / sum_f(normal_test_sources_f)
```

The code also reports normal-count-weighted average precision, fold prevalence, fold support and category summaries. Do not pool uncalibrated scores from different fitted models into one ROC curve. A category with only a few sources is descriptive, and category summaries reuse normals and sometimes multi-category sources.

The primary paired interval uses 1,000 bootstrap draws with seed 42. Within every fold and binary label, resample source IDs with replacement and carry both methods' scores through the same draw. Recompute the same weighted AUROC difference as the point estimate. Every source contributes once to outer-test evaluation.

This interval conditions on the fitted models, reference parameters, fold assignments and observed source cohort. It excludes retraining uncertainty, alternative reference selection and unknown participant dependence across uploads. An interval over windows would overstate independence. An interval spanning zero does not establish improvement, and a negative result must not trigger a sign reversal or favorable split search.

The runner exports a common-source table and checks target support across trained, random-backbone and ridge methods. A partial fold run is labeled partial in its summary. All four folds and the declared controls are needed for a complete protocol result. All comparisons remain exploratory because the research direction was developed after earlier results on overlapping sources.

## 10. Keep five-class recognition separate

`condition_capstone` uses the same accepted poses and outer folds for normal, parkinsons, stroke, cerebral palsy and myopathic labels. It fits on normal fitting sources plus non-test focal abnormal sources, excluding all inner normal reference sources. Thus it receives abnormal labels that the anomaly models do not receive.

RF uses 36 measured descriptors with training-only median imputation and fixed settings: 300 trees, maximum depth 8, minimum leaf size 2 and square-root feature sampling. The frozen JEPA probe uses training-fitted scaling and logistic regression with `C=1`. Training weights start with inverse source window counts and then balance focal classes. No test score selects these hyperparameters.

Average probabilities within sequences and then within source-label pairs. If one source has several focal labels, divide its total evaluation weight equally across those rows. Report per-fold macro-F1, balanced accuracy, confusion matrices and unique-source support. These supervised scores belong in a separate table and cannot establish that a normal-only detector is superior or inferior under equal supervision. Saved classifier artifacts record the fitted RF, scaler, probe and encoder checkpoint identity.

## 11. Use interventions as diagnostics, not virtual diagnoses

`run_study_controls.py` is separate from the main runner. It uses normal fitting sources in a chosen fold, obtains longer 40-frame windows, and selects one per source by sequence/time order. It fixes the original central crop's pelvis/torso reference across edits. A four-frame limb shift, a whole-body shift, coordinate noise and observation dropout use common interior support without circular wrapping.

All versions are scored with the same masks, then restricted to identical valid target identities. Geometry summaries track limb, pelvis and torso changes and donor-relative limb articulation. The previously fitted normal reference is applied without refitting to edited data. An edit that loses support remains unscorable. No geometry threshold certifies physiological validity.

### Optional predictor and teacher decomposition

Adding `--teacher-decomposition` enables the implemented four-factor latent-loss diagnostic. It compares the original input with the relative left-leg shift, using only joints `(13,14,15,16,26,28,30,32)`. All local coordinates and visible own-joint histories for these eight arm and right-leg joints must remain identical across the inputs. The command verifies that condition before scoring.

Let `p0` and `p1` be predictions from original and edited context, and let `y0` and `y1` be the original and edited full-input frozen teacher distributions. For each of the own, patch-local and full branches, compute:

```text
L00 = KL(y0 || p0)       L10 = KL(y0 || p1)
L01 = KL(y1 || p0)       L11 = KL(y1 || p1)
predictor change = L10 - L00
teacher change = L01 - L00
interaction = L11 - L10 - L01 + L00
joint change = L11 - L00
```

The last change equals the sum of the other three. All four factors use identical requested targets, common eligibility, the same fixed masks, unchanged temperatures and a frozen teacher center. The module checks the additive identity and rejects nonfinite eligible losses. The command also records full-minus-own component contrasts and aggregates raw components per source and unchanged joint.

Outputs are stored under `fold_<k>/development_controls/teacher_decomposition/`, including four factor tables, `raw_kl_components.csv.gz`, `source_joint_components.csv` and a protocol record. These are raw-loss diagnostics on eight unchanged joints. **They are not a decomposition of the nonlinear three-region anomaly score.** The missing left-leg group is not imputed or reconstructed. A future profile-distance decomposition would require four complete matched source profiles scored against one fixed reference before taking differences.

This optional module has dedicated tests and records its own implementation and input hashes. It does not alter the main model's run identity or refit the model. A measured teacher component indicates a contextual target effect in this computational comparison, not an effect of a nervous system or disease. Its real-data outcomes remain a separate evidence item until the command completes.

Neither the paired controls nor the decomposition provide blinded phenotype agreement or causal validation. A label-blinded annotation template exists, but videos can still reveal diagnoses through titles, captions or appearance. Real independent review, blinding checks and reviewer agreement remain separate work.

## 12. Preserve provenance and bound the manuscript claim

Run from `experiments/aaai`:

```bash
uv run python scripts/ingest_study.py --download-missing --workers 4
uv run python scripts/run_context_study.py --pretrain-updates 512 --predictor-updates 512 --masks 16 --ridge-draws 256
uv run python scripts/run_study_controls.py --run-dir artifacts/context_study/gavd/<run_id> --fold 0
# Include the optional raw-loss teacher diagnostic when required:
uv run python scripts/run_study_controls.py --run-dir artifacts/context_study/gavd/<run_id> --fold 0 --teacher-decomposition
```

Use `--demo` on the study runner only for artificial software checks. New results are written under `artifacts/context_study/gavd/<run_id>/`; synthetic runs use the separate `synthetic/` path. The run identity binds the corpus, profile eligibility, implementation hashes, software versions and settings. Freeze ingestion before training. Checkpoints and loss caches are accepted only with matching provenance.

The main files are `protocol.json`, `source_folds.csv`, `profile_eligibility.csv`, per-fold `source_roles.json`, model histories and checkpoints, loss tables, frozen references, support tables, source scores, ranking metrics and the paired interval. Keep historical outputs under their existing paths. Root `.env` settings and notebook aliases are documented in the [main guide](../README.md).

No result supports a clinical mechanism simply because its latent variable has a physiological name. A useful claim needs adequate normal learning, benefit beyond matched controls, sensitivity analysis and observation-quality evidence. External participant- and site-separated validation remains necessary for generalization. The [paper plan](paper_plan.md) and [draft manuscript workspace](draft01/README.md) must reflect the results actually obtained, including null or adverse findings.
