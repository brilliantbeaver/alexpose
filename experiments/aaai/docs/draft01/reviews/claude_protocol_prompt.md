Act as an independent, demanding ASTAD workshop reviewer and research advisor. This is first-party research critique. Use no tools. Evaluate whether this design can support a strong manuscript, identify scientific failure modes and overstated claims, and propose concrete fixes feasible in days. Distinguish correctness from novelty and clinical validity. Do not invent results, diagnoses, source identities, or acceptance probabilities. Use plain sentences without em dashes. Give severity, evidence from the supplied protocol, and the exact narrowing or experiment needed for each major issue. Score novelty, methods, evidence readiness, clarity, and workshop fit from1to5 with reasons. Current real corpus:348annotatedsources32normal;332videos decodable;9474QCwindows from311sources28normal;profileeligibility8950windows1303sequences251sources23normal228abnormal. All23normal sources previously explored;58abnormal sources previous and170new. Fourfold testnormal6/6/6/5 and57abnormal each. Five innernormal references per fold. Primary512pretrain+512headupdates is currently scoring; no primarytest results are supplied. On normal validation, coordinatefull-context MSE exceeds own-joint MSE in all4folds. A fixed4096headupdate sensitivity with same512pretrain was launched on that basis before newtestinspection, retaining the primaryrun. Real developmentnormal controls:12sources selected, common support aftertiming/noise/dropout leaves5scorable sourceprofiles. Relativeleg timing edit does not show a stronger medianprofilechange than whole-bodyshift in this tinydevelopment sample. Teacher diagnostic on unchangedarm/rightleg localmeasurements is implemented; it establishes model-targetcontexteffects, notbiology. Provide a defensible manuscript angle for positive,null,ornegativeforthcomingresults. Raw advice will be separately archived and verified.

FILE docs/context_study_protocol.md
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


FILE docs/published_baseline_feasibility.md
# Feasibility of a published skeleton anomaly baseline

Checked on November 15, 2026. **Neither official implementation is currently a validated comparator for our 16-joint, partially observed GAVD protocol.** STG-NF is easier to execute locally. SeeKer is the closer scientific comparison because it predicts conditional keypoint distributions. Their main shared obstacle is the treatment of missing coordinates. A quick zero-fill adapter would change the statistical model and would not establish a published-method reproduction.

This review inspected the official repositories, read the primary papers, and ran limited mechanical checks on unmodified model code. It did not train either method, change our model, install dependencies, or produce GAVD detection scores. The current study's matched ridge predictor is a useful control, but it does not substitute for a validated published anomaly detector.

## What was inspected and tested

| Item | Pinned version or basis | Observed status |
| --- | --- | --- |
| [Official SeeKer repository](https://github.com/adelic99/seeker/tree/7e7ab66a703188616b0efd482947ff2bf2bb26ca) | `7e7ab66a703188616b0efd482947ff2bf2bb26ca` | Source inspected. The unmodified model constructor fails on this laptop because its masks are moved directly to CUDA. |
| [Official STG-NF repository](https://github.com/orhir/STG-NF/tree/edb5f3220332e160e4d20ce258787d5e2d7e0200) | `edb5f3220332e160e4d20ce258787d5e2d7e0200` | Source inspected. An unmodified 18-joint model produced finite CPU outputs on synthetic finite input. |
| Local runtime | Apple arm64; Python 3.12.13; PyTorch 2.15.0; NumPy 2.5.4 | CPU checks only. No CUDA, MPS, optimizer, checkpoint or training validation was performed. |
| Our observation protocol | [data.py](../aaai_gait/data.py), [study_data.py](../aaai_gait/study_data.py), and [research roadmap](draft00/research_roadmap.md) | Sixteen selected joints, 32 frames at 15 Hz, four-frame tokens, confidence filtering, retained NaNs and source-held-out evaluation. |
| Historical completeness audit | [full_window_index.csv](../artifacts/full_window_index.csv) | Read-only count of the completed earlier five-class cohort. This is not the final expanded twelve-folder study cohort. |

Temporary repository checkouts were inspected outside the project. They were not vendored into the tutorial package. The historical window index had SHA-256 `53a08a8dffdfe0a691879587076dd6b04e1668811895cfd84cdf329483284142` at review time.

The STG-NF check used `pose_shape=(2,32,18)`, eight flow steps, one level, zero hidden channels, its default permutation and affine coupling, `R=3`, and CPU execution. This constructed 744 trainable parameters. A batch of two finite synthetic windows returned two finite negative log-likelihoods. After initialization, replacing one coordinate in the first window with NaN made that window's score nonfinite while the second score stayed finite. Reconstructing the same model with 16 joints failed because its graph still had 18 nodes. These checks establish a narrow execution fact, not training correctness, convergence or anomaly performance.

SeeKer's parser check confirmed that `droppout` exists and `dropout` does not. Static inspection confirmed that the trainer does not define `joint_lnp_full_window`. Importing its trainer in the current tutorial environment stopped at the missing `tqdm` package. Constructing its model directly, without the trainer, stopped at the hard-coded CUDA operation. No patches were applied to hide these failures.

## Why both methods matter scientifically

SeeKer models keypoint locations with autoregressive conditional Gaussians and combines keypoint log scores with detector-confidence weights. It already occupies the space of conditional skeletal prediction, compositional anomaly scoring and confidence-aware scoring. Therefore, predicting a hidden joint from other joints, reporting per-joint surprise, or using confidence cannot by itself establish our novelty. Our proposed distinction is the controlled comparison of own-joint, patch-local and full context, including a two-sided normal-reference profile and tests of contextual teacher effects. That distinction still needs evidence. [SeeKer paper](https://arxiv.org/html/2506.18368v1)

STG-NF directly learns a distribution over pose sequences using graph-conditioned normalizing flows. It supports training on normal examples and scores sequence likelihood. Normal-only skeleton anomaly detection and learning normal motion distributions are therefore established prior art. Our study must test a specific information hypothesis rather than claim novelty from normal-only fitting. [STG-NF paper](https://arxiv.org/pdf/2211.10946)

The two published methods are different kinds of baselines. SeeKer predicts ordered keypoints. STG-NF gives a whole-window density. Neither is a natural replacement for the exact matched raw-KL control inside the own-versus-full experiment. Their role is an external anomaly-detection comparison on a declared common source cohort.

## SeeKer: close scientific fit, incomplete released execution path

The released model is a masked fully connected autoregressive network. The masks express a fixed ordering, which is different from the uniform random target masks used by our S-JEPA. Adapting the published baseline does not require replacing its autoregressive factorization with our masking task.

The following issues are visible at the pinned version:

| Area | Verified issue | Required action before a result is credible |
| --- | --- | --- |
| Installation | The README references a `requirements.txt` that is absent. Its clone command contains a placeholder repository name. | Record an explicit dependency environment and use the actual official repository. |
| Entry point | The parser defines `droppout`; `seeker.py` accesses `args.dropout`. | Repair the argument mismatch and test the command-line path. |
| Device handling | The model, connectivity masks and confidence scoring contain direct CUDA calls. | Make device handling explicit and register masks as buffers. Check numerical agreement on an available reference device. |
| Test path | `test()` calls an undefined `joint_lnp_full_window`. It also unpacks three values from a scorer that returns one unless its separate return flag is enabled. | Resolve the intended inference path against the paper and a reference case. Do not guess which method name was intended and immediately report results. |
| Skeleton layout | The entry point uses 18 joints. The causal-mask context boundary subtracts `18*2`; validation reshapes scores with 18 joints. | Parameterize the joint count and audit the full flattening, ordering, output alignment and scoring path. |

These findings come from separate source files: [README](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/README.md), [arguments](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/args.py), [entry point](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/seeker.py), [masked model](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/models/made/made_partial.py), [training and inference](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/training.py), and [evaluation](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/validation.py).

Several methodological details also need adaptation:

- Its dataset loader converts COCO-17 detections into an 18-joint layout by deriving a neck point and reordering joints. Our allowlist cannot be represented by applying that conversion to a 16-joint tensor. The shoulders, hips, heels and foot indices must keep their actual identities. [Dataset conversion](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/dataset.py)
- Its normalization uses the mean position and vertical standard deviation across the whole input window. Our normalization uses each frame's pelvis and a prefix torso scale. Use the shared normalization for a matched-input comparison, and report the change. [Normalization](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/utils/data_utils.py)
- Its sequence utility interpolates missing frames. Our protocol preserves missing observations and never bridges a missing-frame gap. Reuse our accepted windows rather than silently importing that interpolation policy. [Sequence processing](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/utils/sequence_utils.py)
- The released trainer selects epochs by validation AUROC. For datasets other than UBnormal, its loader assigns test data to the validation role. Neither choice can be reused for our normal-only model selection. Our outer test sources must never choose a checkpoint. [Dataset roles](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/dataset.py), [checkpoint selection](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/training.py)

For a finite, fully observed input, a feasible layout adapter would use `[batch,time,16,2]` in the fixed numerical joint order. The flattened dimension would be `32*16*2=1024`; the context boundary would depend on `2*16`, not 36. Both XY coordinates of one target joint must be excluded from the conditioning set for that target. Verify this with numerical perturbation or Jacobian tests for every target position. In particular, audit the trainer's one-keypoint output offset before choosing an inference implementation. A changed order is an adaptation and should be fixed before inspecting abnormal performance.

The confidence weights need a separate decision. MediaPipe confidence values have not been established as calibrated probabilities of a correct joint location in this corpus. A weighted score can still be reported as the published scoring heuristic, with an unweighted sensitivity analysis, but it must not be described as calibrated uncertainty or as a solution to missing conditioning variables.

## STG-NF: executable core, incompatible graph and missingness assumptions

The official setup targets Ubuntu, Python 3.8 and a CUDA GPU. Its environment file pins PyTorch 1.10.1, NumPy 1.23.3 and CUDA 10.2, along with Linux-specific packages. That environment should not replace the tutorial's working Python 3.12 environment. The successful CPU core check suggests that a small modern environment is feasible, but it does not validate the legacy data and training scripts. The loader still contains the removed NumPy alias `np.int`. [Official setup](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/README.md), [environment](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/environment.yml), [segmentation code](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/utils/pose_utils.py)

The model constructs the default OpenPose graph inside each flow step. Passing a new input shape does not change its adjacency matrix. The existing headless option is also not our 16-joint skeleton. A real adapter must introduce an explicit layout and pass it through every flow block. For our numerical order `[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32]`, a documented candidate is the induced body graph with these undirected edges in original BlazePose indices:

```text
11-12, 11-13, 13-15, 12-14, 14-16,
11-23, 12-24, 23-24,
23-25, 25-27, 27-29, 29-31, 27-31,
24-26, 26-28, 28-30, 30-32, 28-32.
```

Add the 16 self connections and convert indices through the fixed order above. Preserve the declared adjacency normalization and hop settings. Do not pad to 18 with duplicate joints or invented face points. Check graph symmetry, reachability, normalized adjacency, gradient finiteness and a finite forward/inverse round trip. Using this graph is an explicit BlazePose adaptation, not an exact OpenPose benchmark reproduction. [Graph construction](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/graph.py), [flow graph use](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/model_pose.py)

The flow likelihood and its data-dependent normalization operate on every coordinate. They expose no observation mask or marginal likelihood over an observed subset. The confidence option in the trainer is not an invalid-coordinate exclusion mechanism. Replacing NaN with zero makes the flow model the filled observation, while dropping selected latent likelihood terms does not generally marginalize missing input coordinates because the flow mixes coordinates. [Likelihood and normalization](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/STG_NF/modules_pose.py), [training confidence handling](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/models/training.py)

## Missing observations are the deciding constraint

Our accepted windows need at least 75% valid four-frame tokens overall and 50% in each half. A valid token has finite XY for its joint in all four frames. Confidence below 0.5 creates missing coordinates. The model excludes invalid keys, targets and pooled contributions. A published baseline must state how it uses that same available information.

Multiplying a final likelihood term by zero does not undo the effect of a missing or filled value on other predicted terms. In SeeKer, a missing early keypoint may be part of later keypoints' conditioning sets. In STG-NF, missing inputs enter coupled transformations. Both need more than a loss mask.

The tempting complete-case fallback was checked against the saved historical cohort:

| Historical folder | Accepted windows | Accepted sources | Fully observed 32-frame windows | Sources with a fully observed window |
| --- | ---: | ---: | ---: | ---: |
| Normal | 482 | 28 | 11 | 5 |
| Parkinsons | 184 | 10 | 70 | 5 |
| Stroke | 779 | 15 | 143 | 9 |
| Cerebral palsy | 390 | 10 | 114 | 6 |
| Myopathic | 669 | 28 | 105 | 18 |
| Total | 2,504 | 91 | 443 | 43 |

The count selects `valid_token_fraction == 1` in the saved window index. The five remaining normal sources cannot support our four outer folds with five separate inner normal reference sources and a remaining normal fitting set. The new all-folder extraction must be audited separately, but adding abnormal folders does not by itself add normal sources. A complete-case result would also describe a strongly selected visibility population. It cannot stand in for the primary partially observed cohort.

There are three honest options:

1. **A fully observed secondary cohort.** Preserve the published density model and explicitly restrict all compared methods to the same complete cases. Proceed only if the final corpus supplies sufficient normal sources for a prespecified fitting and evaluation design. The historical count fails that gate. Shortening windows or lowering confidence thresholds would change the experiment and requires a separately declared sensitivity analysis.
2. **An observation-aware adaptation.** For example, a conditional predictor could read finite values plus explicit availability indicators and score observed targets. This is technically plausible, but changes SeeKer's conditioning model and requires missingness noninterference, calibration, coverage and nuisance tests. Label it a SeeKer-inspired adaptation until validated against the published method. It is not an exact marginal likelihood simply because missing targets are excluded.
3. **A later density-method study.** Proper integration over missing coordinates or a validated conditional-flow construction is a substantive extension. It is not a safe one-line repair before this deadline. For this submission, report the missing published comparator and keep claims limited to the matched internal controls.

Using low-confidence raw coordinates that our primary protocol excludes is another possible sensitivity analysis, but it changes the information supplied to the method. It cannot be presented as the same-observation comparison.

## Dependencies and a bounded validation plan

For an isolated SeeKer audit, direct imports require PyTorch, NumPy, SciPy, scikit-learn, `tqdm` and TensorBoard. STG-NF's model core requires PyTorch and NumPy; its surrounding scripts also use `tqdm`, TensorBoard, SciPy, scikit-learn, Pillow and Matplotlib. Most are already available in the tutorial environment, but `tqdm` and TensorBoard were absent during this review. These are observed import requirements, not a tested replacement lockfile. Create a separate `uv` environment and record resolved versions before a reproduction attempt. Avoid upgrading the running experiment environment to repair another repository.

The release licenses also differ: SeeKer is MIT, while STG-NF is distributed under Attribution-NonCommercial 4.0. Preserve source attribution and the relevant license if code is redistributed. This review did not redistribute either implementation. [SeeKer license](https://github.com/adelic99/seeker/blob/7e7ab66a703188616b0efd482947ff2bf2bb26ca/LICENSE), [STG-NF license](https://github.com/orhir/STG-NF/blob/edb5f3220332e160e4d20ce258787d5e2d7e0200/LICENSE)

The following is a planning estimate, not a measured completion schedule:

| Priority | Work package | Gate for moving forward |
| --- | --- | --- |
| First half day | Recompute completeness and source support on the final frozen twelve-folder corpus; choose the observation policy. | Sufficient independent source roles without changing the primary comparison after viewing results. Stop the complete-case route if normal support remains inadequate. |
| Next one to two days | Reproduce a small native-layout reference case in an isolated environment. Prefer STG-NF for immediate execution; use SeeKer if the inference alignment and released defects can be resolved against the paper. | Finite forward/backward passes, deterministic evaluation, correct score direction, stable checkpoint reload, and a reference result or independently checked analytical case. |
| Next one to two days | Add only the declared 16-joint layout and shared preprocessing, keeping a written change log. | Target-order or graph tests pass; no artificial joints; no silent missing-coordinate fill; no test-based checkpoint selection. |
| Remaining time | Fit normal-only folds and aggregate on the common eligible source set. | All source roles and exclusions are recorded, normal development diagnostics are adequate, and paired source-level intervals use the same metric as the point estimate. |

Executing and adapting a small model core within a week is plausible. **A validated published baseline on the full partially observed cohort is not currently a defensible promise for the November 20 deadline.** The missingness policy and scarce complete normal sources are the limiting evidence, even if code repairs are fast. The workshop date is verified on the [official ASTAD4 call](https://ismart.ece.mcgill.ca/astad4/); this review does not infer an unstated cutoff timezone.

For the present paper, prioritize the matched own/full/patch-local comparisons, local-coordinate target control, matched raw KL, ridge predictor and source-level uncertainty. If a published comparator fails the gates above, say so plainly. A supported study of predictive information can remain useful without claiming superiority over SeeKer, STG-NF or the wider literature. No current result should be called state of the art, and the absence of a runnable adapter does not diminish the prior-art relevance of either method.


FILE docs/draft01/primary_source_claim_map.md
# Verified references and allowed claims

Verified November 15, 2026. The accompanying `references.bib` contains **16 references**. Experimental runs are still in progress: none of these references supplies a new GAVD result, validates our implementation, or establishes superiority of the proposed score. This is a citation and claim map, not a literature-review chapter.

## Dataset and methods

| BibTeX key / primary source | Claim the manuscript may make | Inference this source does not license |
| --- | --- | --- |
| `ranjan2025gavd` — Ranjan, Ahmedt-Aristizabal, Ali Armin & Kim. **Computer Vision for Clinical Gait Analysis: A Gait Abnormality Video Dataset.** _IEEE Access_ 13:45321–45339, 2025. [DOI](https://doi.org/10.1109/ACCESS.2025.3545787); [author's preprint](https://arxiv.org/abs/2407.04190). | GAVD collects public-source gait videos with expert observational annotations, across normal, abnormal and pathological movement patterns. The paper reports an inventory of 1,874 sequences and visual screening; acknowledge this curation. | A folder name is not a verified medical diagnosis, participant identifier, force/EMG record, demographic ground truth, or measured disease mechanism. Published inventory counts and RGB-model results do not describe our usable pose cohort or establish a matched baseline. |
| `gavd2026catalog` — GAVD authors' [catalog documentation, pinned revision](https://github.com/Rahmyyy/GAVD/tree/a87859c881603443f200bcd640663d2c3d7a8136). | The documented fields include sequence/frame IDs, view, limited gait events, observed gait classification, bounding boxes and video URLs/IDs. The repository distributes annotation files and links; original videos can become unavailable. | A video ID is not a unique participant ID. Do not infer synchronized multiview recordings, paired medication states, verified laterality, or ground-truth kinetics from these fields. Compute local availability, grouping and exclusions from retained artifacts. |
| `abdelfattah2024sjepa` — Abdelfattah & Alahi. **S-JEPA: A Joint Embedding Predictive Architecture for Skeletal Action Recognition.** ECCV 2024. [Primary paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf); [DOI](https://doi.org/10.1007/978-3-031-73411-3_21). | Prior skeletal self-supervised learning predicts representations of missing joints from partial sequences, with teacher-target centering and action-recognition evaluation. | Latent skeleton prediction, masking, or a transformer alone is not novel here. An implementation using different targets, loss, landmarks or data is an adaptation, not an exact reproduction. Action-recognition results do not validate gait anomaly detection. |
| `mao2023mamp` — Mao, Deng, Zhou, Fang, Ouyang & Li. **Masked Motion Predictors are Strong 3D Action Representation Learners.** ICCV 2023. [Primary paper/metadata](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html); [DOI](https://doi.org/10.1109/ICCV51070.2023.00934). | MAMP predicts temporal motion of masked skeleton joints and uses motion-aware masking for 3D action representation learning. | Uniform masking is not its motion-aware sampler. Coordinate prediction and motion prediction are different objectives. Its action benchmarks do not establish our normal-only performance or clinical validity. |
| `bardes2022vicreg` — Bardes, Ponce & LeCun. **VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning.** ICLR 2022. [Official repository and citation](https://github.com/facebookresearch/vicreg); [paper](https://openreview.net/forum?id=iWpcWZ8phD). | VICReg combines agreement with variance and covariance regularization to discourage collapsed/redundant representations. | Including the loss does not prove that our encoder avoided collapse, learned useful dynamics, or yields independent physiological variables. Verify representation statistics and task outcomes in our runs. |
| `delic2025seeker` — Delić, Grcić & Šegvić. **Sequential Keypoint Density Estimator: An Overlooked Baseline of Skeleton-Based Video Anomaly Detection.** ICCV 2025. [Primary paper/metadata](https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html); [DOI](https://doi.org/10.1109/ICCV51701.2025.01077). | SeeKer autoregressively models conditional keypoint distributions and weights anomaly scores using keypoint-detector confidence. It is close prior art for conditional skeletal prediction. | Its use of “causal” describes an ordered prediction factorization, not identified biological causation. A local conditional-Gaussian baseline without the original model, scoring and confidence handling must be called inspired/adapted, not reproduced SeeKer. |
| `hirschorn2023stgnf` — Hirschorn & Avidan. **Normalizing Flows for Human Pose Anomaly Detection.** ICCV 2023. [Primary paper/metadata](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html); [DOI](https://doi.org/10.1109/ICCV51070.2023.01246). | STG-NF models pose-graph sequences with normalizing flows and uses likelihood for anomaly scoring; the paper studies normal-only and supervised settings. | Density estimation differs from reconstruction/prediction error. Changing joint topology, input dimensions, normalization or aggregation requires documented adaptation. Published surveillance results do not transfer numerically to GAVD. Pose input alone does not guarantee anonymity or freedom from bias. |
| `duan2024fsgait` — Duan, Wan & Zhao. **FSGait: Fine-Grained Self-supervised Gait Abnormality Detection.** ACCV 2024. [Primary proceedings version](https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19); [open paper](https://openaccess.thecvf.com/content/ACCV2024/papers/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.pdf). DOI: 10.1007/978-981-96-0960-4_19. | FSGait trains reconstruction and prediction modules on normal gait, separates postural and temporal anomaly analyses, and reports sequence/frame/joint outputs. | Normal-only gait prediction, spatial/temporal decomposition and joint-level visualization already exist. Its visualizations do not independently validate a pathological cause, and our paper cannot claim superiority without an appropriate implemented comparison. |

## Clinical motivation: empirical findings and limits

These sources motivate hypotheses about observed movement organization. They do not determine the sign of an own-joint versus full-context score, demonstrate that abnormal gait is less predictable, or validate a disease detector trained on GAVD labels.

| BibTeX key / primary study | Measured finding that may motivate the paper | Do not infer |
| --- | --- | --- |
| `clark2010modules` — Clark et al., 2010. [DOI 10.1152/jn.00825.2009](https://doi.org/10.1152/jn.00825.2009); [PubMed](https://pubmed.ncbi.nlm.nih.gov/20007501/). | Surface EMG from 55 participants with post-stroke hemiparesis and 20 controls supported fewer, sometimes merged, modules associated with walking-performance measures. | A pose token is a neural or EMG module; fewer modules imply universally greater/lesser kinematic predictability; every stroke gait is caused by spasticity. |
| `awad2020drive` — Awad, Hsiao & Binder-Macleod, 2020. [DOI 10.1097/NPT.0000000000000299](https://doi.org/10.1097/NPT.0000000000000299); [PubMed](https://pubmed.ncbi.nlm.nih.gov/31834220/). | Instrumented gait, dynamometry and stimulation in 40 people after stroke linked reduced plantarflexor central drive with propulsion deficits. Some fast walkers showed greater nonparetic contribution, consistent with compensation. | Speed alone measures recovery; projected asymmetry measures propulsion; this cross-sectional association proves a person deliberately adopted a strategy or that an intervention would reverse it. |
| `morris1994hypokinesia` — Morris et al., 1994. [DOI 10.1093/brain/117.5.1169](https://doi.org/10.1093/brain/117.5.1169); [PubMed](https://pubmed.ncbi.nlm.nih.gov/7953597/). | Experiments in 34 PD participants and 34 matched controls distinguished stride-length regulation from cadence; speed-controlled PD walking used shorter strides and higher cadence. | Every PD presentation has this pattern; image-plane excursion equals metric stride length; regular small movement implies normal gait or measured muscle weakness. |
| `sterling2015armswing` — Sterling et al., 2015. [DOI 10.3233/JPD-140447](https://doi.org/10.3233/JPD-140447); [PubMed](https://pubmed.ncbi.nlm.nih.gov/25502948/). | Among 16 PD participants and 17 controls, medication increased the slower arm's angular-velocity amplitude and reduced asymmetry; bilateral cross-correlation did not improve significantly. | No significant change proves no coordination effect; wrist displacement and angular-velocity amplitude are identical; medication state or transmitter status is identifiable from an unlabelled clip. |
| `steele2015synergies` — Steele, Rozumalski & Schwartz, 2015. [DOI 10.1111/dmcn.12826](https://doi.org/10.1111/dmcn.12826); [PubMed](https://pubmed.ncbi.nlm.nih.gov/26084733/). | Retrospective gait EMG in 549 people with CP and 84 unimpaired participants found reduced activation complexity associated with functional/clinical impairment measures. | CP has one mandatory gait signature; pose-only dependencies equal activation complexity; an observational association establishes a causal path in a GAVD sequence. |
| `steele2012crouch` — Steele et al., 2012. [DOI 10.1016/j.jbiomech.2012.07.028](https://doi.org/10.1016/j.jbiomech.2012.07.028); [PubMed](https://pubmed.ncbi.nlm.nih.gov/22959837/). | Model-based weakening experiments using six children with CP and three typically developing children showed demand tradeoffs: moderate crouch required more knee-extensor capacity but less hip-abductor capacity; mild crouch required less plantarflexor capacity. | Crouch is always beneficial or globally weaker; the model proves which impairment caused an individual's posture; estimated muscle forces are directly observed measurements. |
| `vandekerckhove2025dmd` — Vandekerckhove et al., 2025. [DOI 10.1186/s12984-025-01718-5](https://doi.org/10.1186/s12984-025-01718-5). | In 31 boys with DMD and 152 observations, strength, range and 3D gait measurements related hip-extension weakness to trunk/pelvic features and hip-abduction weakness to wider steps. | DMD defines all myopathic gait; visible sway uniquely identifies abductor weakness; initial-value associations necessarily describe within-person progression. Myopathic labels denote heterogeneous muscle-related patterns, not one brain lesion. |
| `kutch2012synergies` — Kutch & Valero-Cuevas, 2012. [DOI 10.1371/journal.pcbi.1002434](https://doi.org/10.1371/journal.pcbi.1002434); [PubMed](https://pubmed.ncbi.nlm.nih.gov/22570602/). | Cadaveric/model counterexamples show that biomechanical and task constraints can generate low-dimensional muscle patterns without fixed neural grouping. | Neural synergies never exist; a compact pose representation has discovered a neural controller. |

## Metadata decisions and checks

- **GAVD version:** cite the 2025 journal article, not the 2024 preprint as if no published version existed. The author-supplied repository citation confirms the surname **Ali Armin**, given name **Mohammad**. The catalog entry uses the same attribution and explicitly labels 2026 as the access year. The pinned revision was resolved from the official repository on the verification date.
- **S-JEPA/FSGait year:** `references.bib` follows the **2024 conference/first-online year**, also returned by DOI content negotiation and Crossref's publisher-deposited records. Springer's suggested-citation widgets display 2025 for the print/copyright year. This is a documented version convention, not two different studies. Springer page ranges are used: 367–384 and 313–329. FSGait's CVF version instead prints 2248–2264.
- **ICCV pages:** CVF's official BibTeX/PDF pagination is retained for MAMP (10181–10191), STG-NF (13545–13554) and SeeKer (11579–11589). IEEE's DOI-deposited records use different page ranges for MAMP (10147–10157) and STG-NF (13499–13508); titles/authors/DOIs agree. Do not silently treat the pagination difference as a different paper.
- **VICReg:** the author-maintained citation verifies ICLR 2022 and all three authors. A conference DOI was not established and is intentionally omitted. The 2021 arXiv record is an earlier version, not a separate selected reference.
- Clinical titles, authors, journal years, volumes, pages and DOIs were rechecked using PubMed/Europe PMC primary-publication records. Their findings were checked against the earlier clinical ledger and source abstracts/full text as recorded there. This is targeted verification, not a systematic clinical review.

## Claims awaiting our own artifacts

Local usable counts, pose exclusions, source grouping, training convergence, implementation fidelity, anomaly AUROCs, confidence intervals, controlled-edit effects and category/view results must come from the completed retained runs. Source-video separation cannot by itself establish participant independence across videos. No existing citation converts an observational movement label into diagnosis, demographic ground truth, causal localization, treatment response or deployment readiness.

The [official ASTAD call](https://ismart.ece.mcgill.ca/astad4/), checked November 15, 2026, specifies **November 20, 2026**, **up to seven content pages plus two reference-only pages**, the **AAAI-27 author kit**, and **double-blind review**. The page does not specify a deadline timezone; check the submission system before submission. This administrative source is recorded here rather than spending a manuscript bibliography entry on it.
