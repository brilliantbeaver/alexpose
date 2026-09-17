# Improving S-JEPA for temporal gait analysis

Research and implementation plan · 15 September 2026

## 1. Recommendation and deliverables

The strongest next study is **a controlled test of whether preserving observation time and predicting future movement makes JEPA representations more useful**. Establish the measurement and information boundary first, then compare a small number of objectives. Another large masking sweep on the current resized clips is unlikely to resolve the main uncertainty.

The current negative finding is valuable: pretraining improves within-clip feature correspondence while reducing the performance of the specified frozen movement readout. Improving this work means explaining that result, testing a useful temporal endpoint, and retaining reproducible evidence. It does not mean selecting settings until a positive result appears.

This planning package contains:

| Document | Purpose |
| --- | --- |
| [Implementation and result audit](47-improvement-audit.md) | Notebook-by-notebook evidence, code locations, numerical findings, and weaknesses |
| [Literature review](47-improvement-literature.md) | Verified primary research, publication status, limitations, and experiment implications |
| [Experiment specification and HAIC runbook](47-improvement-experiments.md) | Proposed modules, notebook stages, configuration, Slurm dependencies, artifacts, and execution instructions |
| [Frontier LLM implementation prompt](47-improvement-llm-instructions.md) | Instructions for GPT-6 Astra or Fable 5.1, Claude Code orchestration, and independent Codex adversarial review |
| [Standalone Codex review prompt](47-improvement-codex-review.md) | Direct stdin input for a scoped independent protocol, code, or evidence review |

**Status:** this delivery is an evidence-backed plan and implementation instruction package. The proposed `temporal_gait` package, notebooks, and `temporal-gait` Slurm launchers are not implemented by these documents. Their example commands are contracts for the implementation step, not commands claimed to work today. No new HAIC jobs, video searches, downloads, model training, or real-data evaluations were performed. Existing notebooks, manuscript revisions, results, and unrelated working-tree edits are preserved.

The requested numbered note belongs here, alongside `46-update-fmts.md`, which contains the task, and the earlier laterality research notes. Code paths below are relative to the `sjepa` checkout unless a document says otherwise.

## 2. What the existing evidence supports

Use the latest [FMTS manuscript](../neurips-laterality/docs/fmts_revisions/paper_v9.md) and its [retained numerical evidence](../neurips-laterality/docs/fmts_revisions/review/numerical_evidence.json), rather than combining numbers from different notebook generations.

| Finding | Retained evidence | Interpretation |
| --- | --- | --- |
| The current cohort is limited | 625 accepted sequences, 93 source videos; five outer folds and five seeds | A source-held-out study within GAVD; not a person-held-out or foundation-scale study |
| Pretraining hurts the expanded ridge readout | Initialized encoder mean source-balanced R² = 0.222544; trained teacher R² = 0.100777–0.114209 | A robust negative result for this representation/readout/target combination |
| The effect is larger than the mask differences | Teacher minus initialization ΔR² = −0.108334 to −0.121766; every condition is worse in all five seeds | Investigate common preprocessing, objective, optimization, and readout choices before finer mask policies |
| Structured masks have no demonstrated advantage | MAMP-style minus uniform −0.000834; mixture +0.000485; connected region −0.008669; all corresponding retained intervals cross zero | Absence of convincing advantage, not proof of equivalence |
| Feature correspondence improves | 375/375 trained diagnostic rows favor the matching clip; 33/75 unique initialization rows do | These are reused model/video comparisons, not 375 independent successes or evidence of forecasting |
| Input and target paths disagree | Recomputed score on resized input: R² ≈ 0.218, sign agreement 70.4%, overlap 623 sequences / 92 sources | A diagnostic of path mismatch, not an information-theoretic ceiling |
| Readout capacity is a concern | 2,890 features; maximum ridge alpha selected in 49/125 teacher fits and 96/125 online fits | Study regularization and representation/readout interaction; do not assume extending alpha will help |
| Real future prediction remains untested | Notebooks 10 and 14 contain forecasting machinery and synthetic demonstrations | A concrete implementation starting point, not real-gait forecast evidence |

The means and teacher differences above were recomputed from the retained per-seed arrays for this review. Raw held-out predictions and checkpoints are unavailable in the inspected local evidence package, so inference and the source bootstrap could not be independently rerun. The retained confidence intervals are conditional on the previously fitted models and splits. They do not include retraining, new splits, or the cumulative effect of developing many hypotheses on these sources.

Several existing controls are strong and should survive the revision: source grouping, matched initialization, paired source draws, per-source weights, fold-local readout preprocessing, count-matched mask controls, distinct constructed versus native reflection behavior, and explicit synthetic-result labeling. The [audit](47-improvement-audit.md) distinguishes the original 00–06 symmetry study, the 07–14 extensions, and the real 15–18 motion study. They are not one interchangeable experiment.

## 3. The main shortcomings and how to resolve them

### 3.1 The endpoint is a useful probe, but a weak flagship task

The signed target averages normalized differences between left and right median pose-coordinate speeds. It is computable from the same detector output that supplies model input. A direct implementation of the target formula on the original observations can recover it by definition. That is a measurement reference, not a learned competitor or a clinical outcome.

The target is unchanged under reversal of a trajectory and its intervals. A uniform time scaling multiplies both sides' speeds by the same factor and cancels in the ratio, apart from the small epsilon. Thus **missing duration alone cannot explain the laterality deficit**. Nonuniform frame gaps, interpolation, aliasing, normalization, and changed paired-valid support can alter it. The same argument applies to a uniform spatial scale.

Retain this score as a representation diagnostic and historical bridge. Also report the five pairwise signed contrasts and their support; their average can cancel opposing effects. Do not relabel pose speed as stride asymmetry, affected side, foot contact, loading, joint force, or disease severity.

The new primary task should predict **future observed image-plane joint trajectories from a past prefix in seconds**. Forecast error at a prespecified horizon tests information beyond a statistic already present in the current clip. Add displacement and velocity errors as secondary outcomes; measure 3D pose estimates separately because inferred depth is not calibrated ground truth. Independently annotated gait events or severity would strengthen clinical significance, but require a separate annotation/external-data study and are not assumed available.

### 3.2 The representation sees a different temporal measurement

In `laterality/geometry.py`, short-gap interpolation and resizing use sample indices; the entire clip becomes 64 positions regardless of length. The target instead uses original timestamps and observed paired support. This is reasonable for some action-recognition tasks but cannot be assumed suitable for quantitative motion.

Run a controlled reconstruction ladder on the **same sequences and same endpoints**:

1. Original observed positions and timestamps: verify the target implementation and support.
2. Change time handling alone while holding observed transitions fixed.
3. Add interpolation alone, with an explicit gap limit in seconds.
4. Change normalization/fallback alone.
5. Apply the historical 64-position resize; separately test physical-time resampling with antialiasing when downsampling.
6. Combine the proposed changes and report interaction effects and coverage changes.

Report both common-support results and all-eligible results; otherwise an apparent gain may just exclude difficult observations. Retain a unit check for uniform time dilation, which should leave the normalized laterality score almost unchanged. This prevents a plausible but incorrect causal explanation from becoming the paper's story.

Coordinate geometry also needs an isolated check: the current augmentation rotates x–z, mixing image position and inferred depth. Convert x/y into a common pixel scale using original frame/crop metadata before the new 2D Euclidean measurement; independently compare the historical pseudo-3D path. Do not carry its augmentation into a new 2D forecast without testing what it preserves. Retain the exact historical coordinates and augmentation only in the reproduction arm.

### 3.3 The SSL cohort should not depend on a downstream target

`laterality/data.py` excludes sequences whose target cannot be computed. That is a defensible endpoint-evaluation cohort, but needlessly restricts pretraining to well-observed examples satisfying all five bilateral pairs.

Maintain two manifests: a broad **label-blind SSL eligibility** manifest based on usable observations, and **task-specific evaluation eligibility** tables. Full-video training uses every eligible walking bout from permitted training sources, including bouts that cannot support the laterality target. Visibility thresholds and quality criteria must be frozen from training/development data. Missing target labels do not exclude an otherwise useful SSL sequence.

### 3.4 The current objective can reward identity, pose, or missingness cues

The local objective uses centered, sharpened feature cross-entropy and a pooled VICReg term. Same-clip matching can improve through static appearance in the pose, joint identity, viewpoint, or observation patterns. Noncollapsed pooled features do not guarantee preserved local motion or high within-source temporal variation.

Measure teacher entropy, channel occupancy, token variance, effective rank, within-source temporal variance, static-frame versus moving-clip responses, and validity-only/source-nuisance probes. Track encoder and projector separately. Include position-only, repeated-frame, shuffled-time, and temporally mismatched same-source controls. A cross-source mismatch alone can be solved by source cues.

Do not diagnose collapse merely because downstream R² falls. The current loss and correspondence scores do not identify the mechanism. Compare objective components, scales, and gradients before attributing the result to any one regularizer.

### 3.5 Mask difficulty and optimization are not settled

The motion experiments actually hide about 17% of valid all-body tokens; region experiments about 9.9%. A configured fraction of 0.5 acts on a smaller eligibility set and is not a 50% whole-skeleton mask. Batch-minimum eligibility also makes mask count depend on the least observed example. Neighbor visibility makes some masked tasks easy.

For the revised masked baseline, define the denominator explicitly and use per-example ragged masks with per-example loss normalization. Begin with realized mask ratios 0.50 and 0.75 of valid eligible tokens on development sources, a minimum visible-context requirement, and contiguous temporal/body blocks. Report actual ratio, valid targets, context tokens, and immediate-neighbor visibility. Pair structured policies with random controls matching both target count and training exposure. High ratios used in other datasets are hypotheses, not defaults established for noisy GAVD poses.

The historical 1,200 updates are an exposure budget, not convergence evidence. With fixed EMA 0.999, the algebraic coefficient of the initial teacher after 1,200 updates is approximately 0.301, with a 693-update half-life. This motivates inspecting teacher lag; it does not explain the result by itself, since the online encoder also performs worse. Preserve the exact old schedule for reproduction, then test a separate development-selected schedule with learning curves.

## 4. What the literature changes

The [full review](47-improvement-literature.md) records titles, dates, primary URLs, and limitations. These are the most actionable conclusions:

| Research | Design consequence |
| --- | --- |
| [Original S-JEPA, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) | The local small model and budget differ substantially from the published action-recognition system. Describe it as a local S-JEPA variant and reproduce its actual recipe. |
| [MAMP, ICCV 2023](https://arxiv.org/abs/2308.07092) | Motion-aware masking is established; merely adding it is not a novelty claim. Compare its mechanism against count-matched uniform targets. |
| [SLiM, August 2026 version](https://arxiv.org/abs/2603.10648v3) | Compact feature prediction and semantic tube masks provide a recent efficiency/shortcut-control reference. Do not conflate compact action tokens with preservation of quantitative gait timing. |
| [V-JEPA](https://arxiv.org/abs/2404.08471) and [V-JEPA 2](https://arxiv.org/abs/2506.09985) | Separate masked latent learning, observable future prediction, and action-conditioned planning. GAVD observations support the first two, not intervention planning. |
| [V-JEPA 2.1](https://arxiv.org/html/2603.14482v3) | Test visible-token loss and intermediate-layer supervision separately and together. In Table 1, context loss improves ADE20K mIoU 22.2→33.8 but reduces SSv2 accuracy 72.8→62.5; multi-level prediction restores it to 72.1. Dense learning has a tradeoff to investigate. |
| [GaitForeMer](https://arxiv.org/abs/2207.00106) | Forecasting-based self-supervision for gait impairment already exists. A contribution needs a better-controlled mechanism, endpoint, or transfer result. |
| Clinical and identity-oriented gait SSL in the review | Identity invariance can discard the amplitude, timing, and asymmetry that clinical measurement needs. Transfer only augmentations whose effects on the intended endpoint are justified. |

The proposed contribution is therefore **a reproducible account of when temporal JEPA learning preserves, ignores, or improves observable gait dynamics under noisy and incomplete measurements**, with a controlled repair if one succeeds. No cited paper establishes that our proposed recipe will improve GAVD results.

## 5. Full GAVD data and temporal protocol

### 5.1 Use the full sequences while preserving independent evaluation

The full videos and walking sequences are already on HAIC. The implementation must consume explicit user-configured video, walking-bout, pose, provenance, and reservation manifests. Do not search remote directories, infer dataset roots, download GAVD, or quietly substitute the local 625-sequence cache. Reading code that explains the adapter is allowed; locating the private assets is not part of this task.

“Full sequences” means index and make available **all eligible time ranges of every permitted walking bout**, preserving the original full-video timeline. It does not mean putting an arbitrarily long video into one transformer, using test videos for pretraining, including unrelated nonwalking footage, or forcing every short bout into a long-window evaluation. Training windows sample across whole bouts; evaluation uses a deterministic complete window index with explicit edge/short-bout exclusions and coverage reports.

Require source-video IDs, parent/duplicate groups, known subject IDs only when genuinely supplied, sequence/bout boundaries, original frame IDs, presentation timestamps where available, decoding/extraction version, and coordinate-system metadata. Frame index divided by nominal FPS is only acceptable when constant frame rate is established; variable-frame-rate videos require presentation timestamps. Repeated or nonmonotone times and pose/frame mismatches stop the affected record with a logged reason.

Split sources/duplicate groups **before** window generation, feature extraction that learns parameters, or fitting normalization. Mirrors, crops, extraction variants, overlapping bouts, and nearby windows inherit the group split. Resolve duplicates only within explicitly supplied assets; do not treat perceptual similarity as verified person identity. Preserve existing held-out reservations from other gavd6 studies via an explicit manifest; a missing reservation must never be interpreted as permission to reuse those sources.

The historical 93 sources are development-exposed. Any new reserved test must exclude those groups and all other declared development/confirmation reservations. If the full cohort contains no untouched eligible groups, report development-only cross-validation rather than claiming new confirmation. All permitted data can contribute across source-held-out folds, but each evaluated model must exclude its own test sources. A single model pretrained on all GAVD is a separate transductive experiment.

### 5.2 Model input and forecast boundary

Recommended starting point, selected for feasibility rather than presumed optimality:

| Setting | Proposed first configuration | Required check |
| --- | --- | --- |
| Pose stream | 33 joints retained in storage; positions, observed validity, elapsed times/observation age; 2D primary, depth secondary | Detector/schema/version consistent; no silent 22↔33-joint conversion |
| Physical-time grid | 25 Hz where observations support it; retain original timestamps and sampling ages | Do not invent higher-rate measurements; separately inspect 12.5/25 Hz sensitivity |
| Prefix duration | 2.56 s, 64 samples; 1.28 s short-bout secondary; 5.12 s long-context extension | Same supported source/window set for paired comparisons; report duration coverage |
| Forecast horizons | 0.25, 0.50, 1.00 s; 0.50 s primary | Fixed permissible timestamp tolerance and common evaluation support |
| Patch duration | Two samples at 25 Hz = 80 ms, with a four-sample control | Expose actual patch times; compare token/compute costs |
| Window indexing | Training: source→bout→time, with coverage-balanced schedules; evaluation: fixed stride such as 0.50 s | Duplicate source windows are correlated, not additional independent subjects |
| Observed/interpolated masks | Keep distinct; short gaps limited in seconds | No score on invented coordinates; report coverage and gap distribution |

Define the sample grid exactly. At forecast issue time `b`, the raw prefix is `[b−2.56, b)`. Its 64 bin-left queries are `b−2.56 + i/25`, `i=0,…,63`; the first-to-last query span is 2.52 s and the last query is 40 ms before issue. Causal sampling uses observations at or before each query and records their ages. Supply identical sampled observations/times to learned and simple baselines. The named 2.56 s duration refers to the window, not a 64-point endpoint span.

Forecast queries are `b+h` in original time, independent of the 25-Hz input grid. Score the nearest original observed pose within 20 ms of each query, break ties toward the earlier observation, and record its actual time offset; require it to remain strictly after `b`. Do not round a 0.50 s horizon to a different grid horizon. Future latent supervision may use an 80 ms interval ending at `b+h`, with two query bins ending at `b+h−0.04` and `b+h`, but this is separate from the single-endpoint primary coordinate score. Targets without the declared support are unscored, not imputed.

No global preprocessing may read beyond the forecast boundary: origin, body scale, interpolation endpoints, crop box, pose smoothing, detector tracking state, normalization statistics, and quality-based context selection are all included. A pose estimator or offline smoother that used future video frames invalidates a strictly past-only claim even if the subsequent tensor slicing is correct. Re-extract causally where necessary, or explicitly describe the offline-observation limitation.

Use a fixed prefix-derived origin and scale to express future positions. Preserve root-relative joint motion and root displacement as separately reported channels; per-frame recentering can remove locomotion. Do not use future pelvis positions to normalize predicted trajectories. Prefix-wide processing is allowed when making one forecast at its end; it is not automatically causal for predictions made at every earlier time.

Future observations may enter the stop-gradient target encoder and scoring during training. They must never enter the context encoder, predictor inputs, online feature normalization, or inference-time support indicators. Mutating observations that remain after the boundary must leave past tensors and predicted outputs unchanged with fixed model state and randomness. Moving an observation's timestamp into the past changes available information and is not this invariance test. For a masked-observation claim, also mask before preprocessing that could spread hidden coordinate values into visible context; separately identify ordinary feature masking on already prepared inputs.

## 6. Prioritized experimental program

### E0 — Recover evidence and identify the measurement bottleneck

**Question:** how much of the apparent representation deficit comes from observation/preprocessing/readout choices?

Reproduce retained arithmetic and, when the user supplies original run paths, regenerate predictions from exact checkpoints or explicitly retrain a historical bridge. Save the complete per-source prediction table. Run the reconstruction ladder in Section 3.2 and a readout decomposition: support-only; paired means; variation; changes; means+variation+changes; and each with/without support. Fit direct position/velocity/bone-angle summaries and matched initialization controls on identical folds.

Use a ridge grid extending to 10⁸ only inside development/inner training. If its boundary is selected, assess train/validation curves and scaling rather than automatically extending again. A low-rank train-fitted PCA ridge and a small train-selected nonlinear readout test the linear-accessibility hypothesis. Fit all preprocessing within the relevant inner partition. Keep capacity, fit budget, and feature dimensions visible.

**Decision:** if preprocessing/readout repairs most of the deficit, emphasize the measurement mechanism. If initialized encoders remain better after these repairs, pursue objective changes. Neither branch invalidates the recorded negative result.

### E1 — Time-faithful masked S-JEPA baseline

**Question:** does a physically timed, complete-bout sampling pipeline improve retained motion under an otherwise controlled objective?

First compare historical versus time-faithful preparation on a common legacy-source subset, with the original fixed training recipe. Next compare both preparations on the same full-cohort development subset, still at matched source draws and exposure. Only then expand training duration. This separates preprocessing, cohort expansion, and optimization effects instead of attributing their combination to a single change.

Distinguish exact historical replay from the mechanism experiment. Replay uses the original complete selected clips and original endpoint; the mechanism experiment applies both preparation paths to identical newly indexed raw windows and recomputes window endpoints, so its scores are not expected to reproduce the paper's clip-level R². In the first mechanism contrast, keep four-sample patches, coordinate channels, capacity, augmentations, masks, and optimization fixed. Add explicit clock channels and the proposed two-sample patches as separately named comparisons after time/support handling is checked. The revised configuration is an eventual baseline, not permission to change several factors while attributing the gain to timestamps alone.

Use one random/block mask baseline selected on development data. Preserve exact initialized/online/EMA checkpoints. Evaluate the historical laterality diagnostic plus future-coordinate probes from frozen prefix features. A masked model remains bidirectional within the available prefix; do not label its pretraining task future-only.

**Falsifier:** no improvement on a common evaluation population, or gains fully reproduced by a validity/timing-only probe, weakens the claimed motion-preservation mechanism.

### E2 — Causal future-feature JEPA with observable evaluation

**Primary new direction.** Predict stop-gradient features of future target windows from the observed prefix and horizon embedding. Future target encoding must be independent of the prefix encoder's execution; no full-sequence contextualization followed by slicing. Predict all selected future horizons in a shared model where feasible, normalize losses per window/horizon, and record absent targets.

Distinguish two evaluations:

1. A decoder from frozen **context** features tests whether future pretraining improved the representation.
2. A decoder from **predicted future features** tests the actual latent predictor's usefulness. Specify whether decoder training uses predicted or observed-future features; compare both on training/development data because their distributions differ.

An observed-future encoder is a clearly labeled diagnostic with privileged information, not a deployable method or a guaranteed upper bound. Observable coordinate/velocity errors can be compared across models; raw latent losses from independently trained teachers cannot.

Required baselines: last position; robust constant velocity fitted to a fixed recent prefix interval; prefix-fitted periodic/harmonic extrapolation with a declared short-prefix fallback; direct past-pose ridge; a small direct forecasting TCN/transformer trained with the same data; initialized encoder plus the same decoder; time-faithful masked JEPA; and future JEPA with temporally mismatched targets. Use within-source wrong-time targets when available, with guards against accidentally equivalent gait phases, as well as cross-source controls.

**Falsifier:** beating initialization but not direct forecasting is evidence about representation learning, not superior forecasting. Beating only persistence at short horizons is insufficient. Gains that disappear under source-balanced evaluation or matched observation support are not a temporal-model success.

### E3 — Dense and deep latent prediction, with a mechanism control

Adapt the idea from V-JEPA 2.1 to the skeleton baseline using a small factorial comparison:

| Arm | Masked target loss | Visible/context target loss | Intermediate-layer losses |
| --- | --- | --- | --- |
| M | Yes | No | No |
| M+D | Yes | Yes | No |
| M+L | Yes | No | Yes |
| M+D+L | Yes | Yes | Yes |

Use matched mask schedules, initialization, observations, update count, and report the additional compute. Begin with layers 2 and 4 for a four-layer model; use separate adapters where needed. Normalize each layer loss and each visible/masked contribution so adding tokens does not silently multiply the effective learning rate. Warm up a context-loss coefficient from zero to a development-selected value, initially testing 0.05 and 0.20.

This is an adaptation, not an implementation of the full published V-JEPA 2.1 system. Its success criterion is simultaneous usefulness for local movement and the declared downstream task, not merely reduced dense loss. Confidence weighting is a separate local proposal; pair it with uniform weighting and identical observation support.

An additional motion-prediction auxiliary loss is a secondary diagnostic, not part of the initial winner. Predict displacement/velocity over hidden or future observations with robust masked error. Compare JEPA+auxiliary, JEPA alone, auxiliary-only, and matched-shuffled-motion targets. If auxiliary-only reproduces the gain, report that the JEPA component has not demonstrated incremental value. Do not train directly on the signed laterality score and then advertise its recovery as label-blind discovery.

### E4 — Frozen video foundation-model transfer

Use explicitly configured released V-JEPA 2 or V-JEPA 2.1 weights as a **frozen comparator** on the same GAVD prefix windows. Start with a released smaller/distilled variant whose inference fits one GPU; verify the exact checkpoint and its required resolution/frame sampling at implementation time. Preserve local spatial/temporal tokens rather than immediately averaging them all.

Compare pose-only, RGB-only, and pose+RGB late fusion with matched readout capacity and identical test windows. Add still-frame, temporally shuffled RGB, background/crop, and within-source mismatch controls. Person-centered crops must use only prefix evidence. Record author checkpoint hash, code revision, preprocessing, and known pretraining data provenance. Web-video overlap may be unknown, so pretrained-model performance cannot automatically be called contamination-free generalization.

If RGB improves measurements beyond pose, a later study can distill its training-source features into a skeleton student. Distillation requires a frozen independent comparator and pose+auxiliary controls; it should not precede evidence that the video features are useful.

### E5 — Optional extension after a real forecasting result

Only after E2 succeeds, test longer horizons, uncertainty, missing observations, and transfer. Quantile or ensemble predictions need a distinct source-level calibration partition, coverage and interval width, and no claim of conditional coverage guaranteed by ordinary conformal methods. Recursive rollout must be evaluated separately from direct multi-horizon prediction and must not consume true intermediate frames.

Reflection equivariance remains a secondary scientific question. Preserve signed channels under anatomical reflection; do not impose invariance on the entire representation. Compare native behavior, an explicitly constructed odd readout, and any learned channel action fitted on training sources. Existing [gavd6 latent-laterality work](../../gavd6/slurm/latent-laterality/README.md) stopped confirmation after a uniform control reproduced the SG-JEPA gain. Do not revive that stopped workflow or present its validation as positive support.

## 7. Optimization and budget selection

Use the following as a **bounded development search**, not a prescription to run every combination:

| Component | Reproduction | Revised development starting point |
| --- | --- | --- |
| Model | Width 96, encoder 4 layers, predictor 2, four heads | Keep this capacity for attribution; test width 192 / six heads only after a mechanism passes |
| Updates | 1,200, exact historical constant-rate recipe for 15–18 | Checkpoints at 0, 300, 1,200, 3,600, 10,800; select a budget using development curves |
| Optimizer | Historical AdamW parameters | LR 3×10⁻⁴ initially, with 10⁻⁴ and 10⁻³ sensitivity; 5% warmup then cosine decay; weight decay 0.05; clipping 1 |
| Teacher | Fixed EMA 0.999 in historical motion experiment | Compare 0.99→0.9999 against 0.999→0.9999 on development data; log cumulative decay product and teacher/online divergence |
| Batch | 20 | Aim for 32–64 distinct source draws if available; source-aware sampler and dynamic padding |
| Objective | Centered CE + 0.05×VICReg | Keep as controlled baseline; normalized feature L1/Huber is a separate V-JEPA-inspired ablation, not a drop-in equivalent |
| VICReg | Weights 25/25/1 inside external multiplier | Test multiplier 0, 0.01, 0.05 with collapse monitoring; monitor gradients and covariance sample size |
| Precision | BF16 training; FP32 evaluation | Keep, with FP32 sensitive reductions; verify one short BF16/FP32 parity case; no FP8/INT8 accuracy claims |
| Seeds | 42–46 for historical comparison | One-seed engineering smoke, then three development seeds, five final seeds if feasible |

Gradient accumulation can increase optimizer batch exposure but does not reproduce large-batch covariance statistics if VICReg is computed separately on each microbatch. State the actual statistical batch size or implement a justified cross-batch statistic. Do not silently call the two equivalent.

Use successive decisions: E0 without encoder training; a common-subset E1 pilot; three development seeds for promising E1/E2 settings; E3 only after reliable baseline behavior; E4 once frozen inference is feasible. Do not launch the full Cartesian product of masks, losses, rates, horizons, seeds, and folds.

Budget matching has two meanings: same optimizer updates/valid-target exposure for causal attribution, and equal GPU time or FLOPs for efficiency. Report both where relevant. The full GAVD sequence/source count and token count are not known locally; choose actual step budgets after an explicit HAIC inventory and measured pilot. Scheduler time limits in the runbook are not measured runtime estimates.

## 8. Evaluation, statistics, and decision gates

Predeclare the E2 primary outcome as **source-balanced mean Euclidean 2D forecast error at 0.50 s** for the eight knee/ankle/heel/foot-tip landmarks `[25,26,27,28,29,30,31,32]`. Require at least three of these four bilateral pairs to be observed at the scored target time; score only paired valid landmarks. Use aspect-correct full-frame pixels, then divide error by a prefix-derived projected body-length scale. Depth estimates and the other landmarks remain secondary. These are detector-coordinate forecasts, not motion-capture ground truth.

Define that scale as the median over valid prefix sides/times of `distance(shoulder,hip) + distance(hip,knee) + distance(knee,ankle)`, using image-plane pixels and at least eight fully observed chain samples at distinct original timestamps on each side; held/repeated grid samples cannot manufacture that support. Require scale ≥ `max(20 pixels, 0.02 × frame_diagonal_pixels)`. This avoids dividing by a nearly zero shoulder/hip width in side views. The proposed support/scale thresholds are engineering defaults to audit on development data and freeze before test use; they are not clinically validated. Save exclusions and view-specific coverage. Report the historical body-width-normalized laterality unchanged, and report root-relative limb forecast error and root displacement separately to reveal camera/root-motion dependence.

Use the 20 ms timestamp tolerance and query convention in Section 5.2. Horizon, joint, and seed summaries are secondary unless explicitly registered. No errors may be calculated on interpolated target coordinates as though observed.

Compute window errors on valid targets, average within each bout, then within each source video, then equally across source videos. Duplicate/known-person connected groups are split and bootstrap clusters; equal-group weighting is a separately named sensitivity analysis, not a silent change to this equal-video estimand. When resampling a group, retain all its videos and their within-video reductions. If instead using a window-weighted within-source estimand, name it and keep it fixed. Pair methods on identical scored windows/joints and publish additional all-eligible coverage. Aggregate source predictions before statistics; thousands of overlapping windows do not create thousands of independent observations.

On development data, choose the strongest non-JEPA comparator and freeze its specification. Report final error ratios and differences against all required baselines; do not choose the comparator retrospectively on the final test. A suggested practical improvement threshold is 5% relative error reduction, to be finalized from development repeatability before test access. It is an engineering target, not an established clinically meaningful difference.

Use paired source/duplicate-group bootstrap intervals with, for example, 2,000 replicates on a single sealed test; repeated-seed variability is reported separately. Cross-validation intervals with overlapping training sets remain conditional/descriptive unless the entire split/training procedure is rerun. Use a prespecified hierarchical testing order or Holm correction if making multiple confirmatory claims; exploratory panels remain labeled as such.

Readout/model/hyperparameter selection belongs to development or inner training. If frozen encoders saw inner-validation sources without labels, disclose that semi-supervised selection setting; for a fully inductive nested comparison, retrain representations within each inner split or use a separate development partition. Do not silently compare settings with different information access.

Fit readouts and direct forecasting baselines on the same training-source pool; use development sources for selection. A separate calibration partition is for optional uncertainty/threshold calibration, not the only data on which JEPA's readout may fit. Freeze whether a final model refits on train+development, and apply that policy equally to all methods. For pose-derived future targets, varying readout-fit sources is an annotation-free fit-budget study, not automatically evidence of reduced manual labeling; clinical label efficiency requires independent labels.

The go/no-go gates are:

| Gate | Pass requirement | Consequence of failure |
| --- | --- | --- |
| G0: evidence | Claims linked to actual artifacts; partial/synthetic outputs identified | Preserve historical limits; do not invent missing predictions |
| G1: information boundary | Timestamp, split, future-mutation, target-support, and provenance tests pass | Fix before training |
| G2: measurement | Common-support baselines and adequate coverage established | Narrow endpoints or report data limitation |
| G3: development utility | Candidate improves over initialization and the frozen non-JEPA comparator across development seeds; no nuisance-only explanation | Stop scale-up or report null; do not open sealed test |
| G4: confirmation | Frozen protocol, complete predicted outputs, all declared seeds/arms, paired analysis | Report incomplete if missing; never average only successful jobs |

A worthwhile negative outcome is that direct forecasting or auxiliary-only learning outperforms JEPA. A worthwhile mechanistic result is that time-faithful preparation fixes the gap without a new objective. Neither requires manufacturing a foundation-model claim.

## 9. Notebook and code changes to request

Preserve original 00–06 as historical protocol evidence. Keep 07–18 educational sources under `tutorials/` as the source of truth; regenerate selected notebooks rather than editing JSON cells independently. Update their explanations and link them to saved new-study artifacts. Do not replace historical outputs with proposed or synthetic values.

The implementation should add a separate study in the gavd6 layout:

```text
gavd6/
  src/gavd6_sjepa/research_directions/temporal_gait/
  scripts/research_directions/temporal_gait/
  notebooks/temporal_gait/
  slurm/temporal-gait/
  tests/temporal_gait/
  docs/studies/temporal-gait/{protocol,execution,results}/
```

Copy the architectural conventions of [motion-preservation](../../gavd6/src/gavd6_sjepa/research_directions/motion_preservation/workflow.py), its dataclass configuration, thin notebook runner, per-stage artifacts, and [Slurm launchers](../../gavd6/slurm/motion-preservation/README.md). Do not copy its scientific assumptions: its GAVD stress stage uses a short window and an eight-sequence cap and can discover a reservation path. Those behaviors do not satisfy this full-sequence study.

The [execution specification](47-improvement-experiments.md) supplies the proposed stage DAG, `TG_*` settings, full-bout manifests, required outputs, resource bounds, dry-run commands, and resume behavior. A pilot must end at development evaluation. Opening a final test requires the saved frozen selection/decision artifacts, not just a successful training exit code.

Every completed experiment needs resolved configuration, schema version, cohort/split/implementation digests, exact source lists, selected window index, input/checkpoint hashes, initialized/online/teacher weights, optimizer/EMA/RNG state, learning curves, per-window and per-source predictions, exclusions, coverage, bootstrap draws or deterministic regeneration inputs, executed notebook, and machine-readable completion status. Log enough to reproduce the result without reconstructing it from notebook screenshots.

## 10. Timing and FMTS positioning

The [official FMTS call](https://fmts-workshop.github.io/cfp.html), checked on 15 September 2026, lists a deadline of 16 September 2026 at 11:59 UTC, a four-page main-paper limit, and explicitly welcomes negative and exploratory results. Do not promise a validated full-cohort study before that deadline.

For the immediate submission, retain the documented negative result, explain its limited scope, preserve the uncertainty qualifications, and describe future prediction as planned work. For the subsequent research cycle, prioritize E0→E1→E2, then the E3 mechanism test and E4 foundation-model comparator. Only include new numbers once their real artifacts and full evaluation have passed the stated checks.

The eventual paper should answer one question: **when does learning to predict hidden/future features help measure and forecast observable gait movement?** The answer must be supported by the paired controlled comparisons, whether positive, mixed, or negative.

## 11. Review and verification of this planning package

Three parallel research roles audited the implementation/results, reviewed primary literature, and designed the execution contract. Independent cross-checks identified and resolved contradictory pilot settings, ambiguous input-grid/horizon semantics, underspecified scoring joints/body scale, unequal readout-versus-forecaster fitting access, and ambiguity between video weighting and duplicate-group bootstrap units. A final code/evidence cross-check found no blocking contradiction in the revised plan and prompts. This is a review of the proposed study, not validation of future empirical outcomes.

Local document checks passed for all six files: 52 relative links resolved, the embedded example JSON parsed and matched the declared key settings, and all five Bash snippets passed syntax checking. Bash syntax checking does not establish that future launchers exist. Retained seed-score means/differences were recomputed; inference and historical bootstrap replay remain unavailable. No original notebook, training implementation, manuscript, or existing result was edited.
