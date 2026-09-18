# Method evidence for the September 13 progress report and tutorial slides

This review checks the current executable method against the retained results. It reads the seven top-level notebooks, the `sjepa` package, the two evaluation runners, upstream angle-feature code, the locked fold registry, and the earlier review ledger. It does not retrain a model or alter notebooks, caches, checkpoints, or evaluation predictions. Small read-only checks recompute group separation, window counts, the fixed readout mask, and a feature-constructor example.

The relevant guidance in the plain-research-writing skill and its paper, tutorial, and figure references was read in full. The descriptions below distinguish current implementation, retained measurements, mathematical properties, and experiments still needed.

## 1. The most consequential distinctions

1. **The frozen five-fold S-JEPA score of 0.438 uses the earlier centering objective.** The AR-5 entry in `docs/03-0802-PHASE_LEDGER.md` explicitly states that the frozen R1 run applied the running-center update once per example and was not rerun after the once-per-batch correction. Current `train_v2.py` scores all examples against the same center snapshot and updates the center once per optimizer batch. The five-fold artifact is a retained result for the earlier objective; current corrected-objective notebook results are separate fold-0 demonstrations.
2. **Current fold-local training excludes held-out source videos, including during self-supervision.** Notebook 03 now uses only fold-0 training clips, with an explicit disjoint-source assertion. Notebook 04 continues only those training clips. Notebook 06 creates a fresh model for its fold-0 demonstration. The five-fold runner likewise creates each model inside `run_fold` and constructs its SSL dataset from that fold's `train_recs`.
3. **No current path preserves every aspect of the original physical gait.** Translation plus uniform scale preserves angles and length ratios within each detected 2-D pose. Per-frame centering removes root travel; frame-varying scaling changes cross-time coordinate differences. Missing observations are interpolated, clips are windowed, and short clips receive repeated frames. The data have no measured 3-D depth channel.
4. **The current Random Forest uses a small core angle feature set.** The broad feature schema and the notebook-02 relevance table do not mean speed, cadence, stride timing, shoulder symmetry, or all named symmetry indices were extracted for this experiment. Moreover, upstream `from_joint_angles` currently assigns the right ankle range to both ankle-range fields; this is directly reproducible without training.
5. **Source grouping supports a specific separation claim, not participant independence or confirmatory inference.** There are 35 cached source IDs and 47 cached clips. The same person may appear in multiple source videos; a source can contain more than one person. Pooled results count test clips, not participants or equally weighted sources. No completed paired source-bootstrap inference, nested selection procedure, multiple-seed sweep, or external clinical test was found in the retained results.

## 2. From video to a geometric observation

### What is retained

`sjepa/data.py:load_video_sequence` returns arrays with shape `(T, 33, 3)`. The three channels are pixel `x`, pixel `y`, and MediaPipe visibility, a detector confidence-related quantity. The cache stores cleaned pixel coordinates and normalized coordinates. It does not store physical depth in its third channel. An old sentence in `sjepa/augment.py` mentions weak `z`; that sentence disagrees with the arrays actually used here.

Frame selection uses integer stride `round(source_fps / target_fps)` rather than timestamp resampling (`data.py:73–74`). Therefore nominal 15 fps is not always the true retained sample rate. A 24 fps source sampled every two frames yields 12 fps, for example. A diagram may label 32 frames as about 2.1 seconds at the nominal 15 fps, but cannot imply identical physical observation duration for all clips.

### Cleaning changes the observations

`clean_sequence` (`data.py:130–161`) checks whether at least 30% of frames have finite `x` coordinates for all joints, trims leading/trailing frames with no finite `x` detections, then interpolates every missing joint/channel value over time. There is no maximum allowed gap length. A channel with no finite values becomes zero. The docstring's phrase “short gaps” is not an implementation constraint. Visibility is interpolated along with coordinates.

Neither a frame-validity mask nor a missingness mask is saved or carried through training. The report should not repeat old documentation that describes validity/padding masking as a completed repair.

### Precisely what normalization preserves

For each frame, let the pelvis midpoint be `p(t)` and the shoulder midpoint be `s(t)`. The normalized point is

`q_j(t) = [x_j(t) − p(t)] / max(||s(t) − p(t)||, 0.001)`.

This is the operation in `data.py:164–182`. It uses the same positive scalar for every joint and both coordinate axes in a frame, so ordinary nondegenerate 2-D joint angles, relative length ratios, and the left/right labeling of that frame are preserved up to floating-point error. Normalization does not independently stretch the limbs or stretch `x` differently from `y`.

However, all absolute root progression is removed, absolute lengths are scaled away, and frame-dependent scale changes motion amplitudes across time. The transform can magnify pose errors when the measured torso is very short; the current cache maximum absolute normalized coordinate is **38.2183**. The lower scale bound avoids division by zero and does not bound normalized coordinate magnitude. This transform is independent for each clip/frame and fits no population statistics using test data, which is a useful leakage barrier but does not repair physical information lost by the transform.

### Windows preserve the recorded order within their retained span

`sliding_windows` (`data.py:258–276`) takes 32-frame windows at stride 16 in the laptop profile. It does not time-warp or phase-align the sequence. Consecutive windows overlap, and the trailing remainder is omitted unless it completes a stride-aligned window. Clips shorter than 32 frames repeat the last frame. Repeated frames receive no padding-validity mask in attention or pooling.

The current cache yields 481 laptop-profile windows. Three clips require padding: `3FXUw98rrUY` has 15 frames and receives 17; `gp4H7Z2Vvn0_clip-01` has 29 and receives 3; `pFLC9C-xH8E_clip-01` has 26 and receives 6. These are three windows from three clips, not three added independent observations. Under the shorter smoke profile the counts change.

## 3. Geometry, angles, and the actual symmetry measurements

### What the Random Forest receives

`sjepa/classical.py:sequence_to_feature_vector` calls `get_joint_angles` on cleaned raw pixel coordinates, then `GaitFeatureVector.from_joint_angles`. The angle calculator is in `../../ambient/pose/joint_angles.py`; its angle is the arccosine of the normalized dot product of two vectors meeting at a joint, expressed in degrees:

- hip: shoulder → hip → knee;
- knee: hip → knee → ankle;
- ankle: knee → ankle → foot index.

These are projected 2-D included angles, not calibrated 3-D joint rotations or automatically equivalent to clinical flexion angles. A straight three-point chain has an included angle near 180°. Degenerate vectors are rejected. The code accepts an angle when the geometric mean of the three landmark confidences meets the configured threshold, so pose confidence can also affect the angle summaries.

The constructor at `../../ambient/classification/features.py:842–918` fills six angle means, three absolute left/right differences of those means, and six range fields. The asymmetry formula is `abs(mean_left − mean_right)`, not the mean framewise absolute difference and not a gait-cycle-aligned comparison. Equal mean angles do not imply matched left/right trajectories. The range is the maximum minus minimum of valid angles over the clip, so outliers can affect it.

The full array schema contains 82 entries, but most are default zero; a default constant fps entry also exists. `train_rf_and_predict` removes columns that are constant in the training fold. It then fits a `StandardScaler` on training rows and uses the fitted transformation on test rows. No test variance or test standardization is used. The forest uses 100 trees, depth 5, square-root feature subsampling, balanced class weights, and seed 42.

**Newly verified implementation limitation:** line 912 sets `left_ankle_range` from `right_ankle_stats`, and line 915 sets `right_ankle_range` from that same source. A synthetic constructor input with left ankle range 12 and right range 15 returns 15 in both fields. Thus the six intended range fields are not six independently correct summaries in the current code. This finding should be recorded with a correction-and-rerun task; retained RF results must not be silently relabeled as a corrected baseline. The old run manifest identifies a dirty tree rather than a complete code snapshot, so the scope of the evidence is “current dependency contains this defect,” unless additional historical source verification establishes when it was introduced.

### What the study has not isolated

The notebooks do not contain a completed comparison attributing predictive performance separately to hip, knee, ankle, shoulder, or gait-phase asymmetry. The notebook-02 mapping lists possible biomechanical relevance, not extracted measurements or measured per-feature effects. It is appropriate to teach those relationships as motivation and propose controlled angle-group removals, while avoiding a conclusion that a particular angle explains MS/PD discrimination.

The current network is a general Transformer supplied with geometry and joint/time identities. There is no explicit proof or constraint that its representation is exactly invariant or equivariant to reflection, rotation, or scale. No reflection-equivariance error is computed in these top-level notebooks. A coordinate transformation's mathematical property must be kept distinct from a learned model's measured response to it.

## 4. Teaching the repaired JEPA objective accurately

### Tokens carry joint and time identity

`sjepa/tokenizer.py` reshapes four consecutive frames of one joint into a 12-number vector (`4 × [x,y,visibility]`) and linearly projects it to width 96. A 32-frame window becomes eight time blocks × 33 joints = **264 tokens**. This initial grouping preserves every entry and its order through reshape/permutation; the learned projection, attention, and subsequent pooling are representations, not lossless reconstruction guarantees. Learned spatial and temporal embeddings identify the joint and time block.

### Two branches of the same training observation

`random_view` applies a small per-window in-plane rotation (±15°), uniform scale jitter (±10%), translation (±0.1 normalized units), and horizontal reflection with probability 0.5. The rotation, scale, translation, and reflection decision are shared across all frames of a window. Reflection negates `x` and exchanges left/right landmark indices. Visibility values follow those relabeled landmarks.

These transformations preserve within-frame shape up to a similarity/reflection transform and preserve frame order. They change coordinates and may change signed laterality. Their clinical acceptability has not been established by a mirror/no-mirror ablation. Do not claim that the model preserves every disease-relevant asymmetry merely because a mirror transform preserves unsigned geometric shape.

The trainable view encoder sees only each example's visible context tokens. The predictor receives the encoded context, a learned placeholder at every hidden location, and explicit joint/time positions. The slowly updated target encoder receives the complete original window and supplies target features. Gradients update the view encoder, predictor, and mask token; target-encoder parameters are updated by an exponential moving average of view-encoder parameters. This moving average smooths the target over updates. Its existence does not guarantee freedom from collapse or a clinically useful representation.

### Why the old predictor failed mechanically

The legacy predictor supplied identical placeholders at all hidden locations without adding their joint/time identities. With dropout zero, its permutation-equivariant attention could give identical target-position outputs. `PredictorV2` adds separate learned spatial and temporal positions to every predictor token. Existing correctness tests check that hidden outputs differ and respond to position changes. These tests establish an implementation property, not a downstream accuracy gain.

### Stochastic masking replaces permanent hidden joints

The legacy mask hides the same twelve shoulder/lower-body joints at every time block. In the repaired sampler, different examples receive different connected joint regions over contiguous time spans. The nominal target fraction is 0.6; whole regions can overshoot that budget, so it is not an exact 60% quota. Clinical-region weighting uses a default factor of 1.5 adjusted for region overlap, not an exact per-joint 1.5× target probability. Every joint can be visible on some draws and hidden on others.

The fallback preserves at least one token from the designated clinical joint set when all of that set would otherwise be hidden. Because that set includes shoulders, describe this as a “clinical-set context token,” not a guaranteed lower-leg or contralateral cue. Masking uses attention's `key_padding_mask` to exclude hidden JEPA targets; that mechanism is unrelated to the missing physical-frame padding mask described above.

### The latent loss does not predict diagnostic probabilities

`CenteringSharpeningCE` subtracts a running feature center from target features, scales predicted and target features by temperatures 0.1 and 0.06, and applies softmax over feature dimensions. It computes cross-entropy between these distributions only at selected target positions. These feature-dimension probabilities are not probabilities of normal/MS/PD. The normal/MS/PD label is discarded by current `train_sjepa_v2`; labels enter later in the supervised classification probe.

Current training scores all examples against one center snapshot and applies a single center update from all masked targets in the batch. The retained 1000-update R1 used the earlier per-example update, according to the AR-5 ledger. This is a substantive objective distinction.

### Source-uniform training changes weights, not the cached skeletons

Current `train_v2.py` assigns each window probability proportional to the inverse number of windows from its source ID, then samples with replacement. Each source has equal expected exposure, regardless of clip length. Actual draw counts need not match exactly. This protects optimization from domination by long source videos. It reweights the training distribution; it does not preserve the original empirical frequencies, invent new participants, or change test-set weights.

The optimizer runs for a specified update budget, with 10% warmup, learning-rate decay and EMA scheduling tied to the update horizon. Full model/optimizer/center/random state can be checkpointed. Effective rank and embedding standard deviation are diagnostic measures of representation variation across a batch; nonzero values show variation, not necessarily gait information. The approximate EMA half-life displayed by the code uses the midpoint momentum, not an empirical fitted teacher time constant.

### The VICReg branch is historical

Notebook 04 retains `vicreg` in its filename but current code calls label-free `train_sjepa_v2`, which does not add VICReg. The legacy `train.py` can combine prediction matching with VICReg. Its variance term discourages low spread; covariance discourages redundant dimensions; invariance compares two embeddings. Its class-aware option first subtracts class means, so the variance term acts on **within-class residual spread**. It does not directly push class centers apart or establish compact separated diagnosis clusters. It uses class labels, so it is supervised adaptation if enabled. Several old docstrings state otherwise and should not be copied into new prose.

## 5. Split boundaries and current notebook budgets

The registry `artifacts/eval/g1/fold_registry.json` is a saved five-fold `StratifiedGroupKFold` partition with seed 42. The code aims for class balance while keeping source IDs together. Source-to-class association and uneven groups mean “stratified” does not imply identical class counts per fold.

Read-only recomputation from current cached records gives:

| Fold | Training clips | Test clips | Training sources | Test sources | Training 32-frame windows | Test windows | Shared source IDs |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | 37 | 10 | 29 | 6 | 395 | 86 | 0 |
| 1 | 37 | 10 | 28 | 7 | 376 | 105 | 0 |
| 2 | 38 | 9 | 28 | 7 | 426 | 55 | 0 |
| 3 | 38 | 9 | 28 | 7 | 354 | 127 | 0 |
| 4 | 38 | 9 | 27 | 8 | 373 | 108 | 0 |

Window extraction occurs from records already assigned to training or testing. It does not randomly assign overlapping windows to opposite sides. Basic clip-local cleaning/normalization can precede the group split because they fit no cross-record statistics; population scalers, encoder weights and probe coefficients must be learned only from training records.

| Current notebook/path | Full laptop budget | Smoke budget | Training exposure | What the evaluation means |
|---|---:|---:|---|---|
| 03 | 800 updates | 60 | Fold-0 training clips, all conditions; no label use in loss | Saves the initial label-free checkpoint |
| 04 | 400 additional updates | 40 | Same fold-0 training clips | Compares frozen probes on the same ten test clips |
| 06 live demonstration | 500 updates | 60 | Fresh model; fold-0 training clips | Single-fold demonstration, not the five-fold result |
| Five-fold R1 runner/artifact | 1,000 per fold | Not the notebook smoke run | Fresh model inside each source-grouped fold | Pooled historical out-of-fold result using earlier centering |

Notebook 04 loads the checkpoint's weights but does not pass `resume_state` into `train_sjepa_v2`. It therefore restarts optimizer state, running center and schedule for those additional updates. “Continue training from the weights” or “warm-started continuation” is accurate; “exact continuation of the original schedule” is not.

The former notebook-03 training-on-all-sources defect was corrected before the current version; the AR-3 ledger records it. A current notebook-04 result must still be tied to the saved checkpoint actually loaded, since the checkpoint filename alone is not a cryptographic assertion of split membership. Notebook 03's source code and retained stage metadata provide supporting provenance but do not record a full source list in checkpoint extras.

## 6. How the clip representation becomes a prediction

Inference supplies the complete normalized window to the frozen target encoder. It averages token features at a fixed, seeded graph-time readout mask, then averages window embeddings to obtain one vector per clip. The fixed mask comes from `sample_target_mask(33, 8, default_rng(0), target_ratio=0.6)`. Direct recomputation selects **161 of 264 tokens**, or **61.0%**, and includes every one of the 33 joints at some time blocks. It is not the legacy twelve-joint anatomical mask.

This readout is fixed before observing test labels. The `StandardScaler` and balanced logistic-regression probe are fitted on training clip embeddings, then applied to test embeddings. The final output is a normal/MS/PD class prediction. Double averaging can obscure when or where a brief asymmetry appeared; no current result identifies that as the cause of a failure.

Notebook 05 embeds all cached clips and fits t-SNE/UMAP displays to the whole collection, including clips used for encoder training. The silhouette score evaluates class-label grouping in the original embedding space. Both are descriptive analyses. They are not a second held-out evaluation and should not choose a model using labels from the nominal test fold. Separate nonlinear maps fitted to two checkpoints also need not align pointwise, so apparent movement between plot coordinates is not itself a measured representation trajectory.

## 7. What the controls can establish

The control implementations are in `scripts/scripts_phase0_provenance.py`:

- Mean pose: normalized coordinate means, 66 numbers. It removes frame order and time variation.
- Pose mean plus standard deviation: 132 numbers. It removes frame order but retains amplitude and pose dispersion, which can include genuine gait signal as well as camera/pose effects.
- Visibility-only: global mean, global standard deviation, and per-joint means of detector visibility. Strong performance establishes a potential route through detector/acquisition correlates; it does not prove every classifier used that route.
- Duration/acquisition: two identical copies of frame count. It is a clip-length proxy, not a direct frame-rate, resolution, or camera classifier.
- Body proportion: nine unnormalized pixel distances from median landmark positions. It may encode image scale and pose; it is not calibrated anthropometry.

Each is tested with both logistic regression and Random Forest using the same source-grouped folds and training-only preprocessing. Showing the strongest classifier per control is a descriptive best-of-two comparison, not a preselected hypothesis test. The report can preserve both scores in a table or explicitly identify the selection rule.

Controls support the statement that strong label prediction is possible without detailed temporal ordering. They do not quantify what fraction of S-JEPA/RF accuracy is caused by nuisance, prove disease labels are invalid, or prove temporal information has no value. Those causal conclusions need controlled ablations or independent evaluation.

## 8. Statistical and interpretive limits

Macro-F1 averages the F1 score of the three condition labels equally. F1 is the harmonic mean of precision and recall for one class. Its numerator and denominator depend on predictions; pooled macro-F1 is not generally equal to the mean of fold macro-F1 values. Use pooled values together, and label fold standard deviation as dispersion, not a confidence interval.

The retained scores are pooled over 47 clip predictions, with each clip predicted when its source was held out. Sources with more clips contribute more test predictions. Training-source balancing does not alter this evaluation weighting. Five folds share much of their training data and are not five independent cohorts; a single initialization seed is not repeated model evidence. The data have already been inspected through multiple development iterations, so these are development estimates.

No retained confidence interval or significance test justifies “statistically significant,” “equivalent,” or a clinical discrimination claim. A proposed null can be phrased as “the learned representation does not add reproducible prediction value beyond angle summaries and controls on genuinely independent sources/participants,” but it must be identified as an organizing research question rather than a preregistered test. The present experiments provide observed differences, not a formal rejection with a controlled false-positive rate.

The one-third line in existing scoreboards is an idealized three-balanced-class reference, not an evaluated random or majority macro-F1 baseline for this imbalanced finite sample. Omit it or label it explicitly. Do not identify the old-to-new score difference as a causal estimate of removing label use, changing masks, or fixing positions: several interventions and evaluation summaries changed together.

The physical-AI motivation is reasonable because geometric relations and temporal coordination constrain the observations being modeled. The current task predicts missing **latent features** from same-window context and evaluates frozen representations. It does not implement a validated future-state simulator, a clinical diagnosis system, or a general physical world model. Those are possible directions with separate tests.

## 9. Suggested sequence for the new tutorial and report

Begin with the repeated geometry of walking: left and right limbs alternate, joint angles relate segments even as a person moves in the image, and a symmetric appearance at a single instant is different from bilateral coordination through a stride. Use health-condition examples as externally sourced motivation, with variation within each condition and no universal normal/asymmetric diagnostic threshold.

Then walk through observations and their limitations, the angle baseline, the missing-feature prediction question, the failed fixed-position design, the repaired training split, and the prediction results. Place the no-time-order and visibility controls alongside accuracy so the reader can see why class discrimination alone leaves the learned mechanism unresolved. End with the next tests required by the evidence: correct the angle-feature dependency, rerun the corrected centering objective, improve sampling/validity, verify participant identity, and isolate angle/temporal/mirror effects using training-only selection and a fresh evaluation set.

This sequence allows both successes and failures to contribute. The predictor-position tests and disjoint-source registry verify concrete implementation properties; the frozen R1 result records a weak representation relative to stronger baselines; current fold-0 continuation probes describe a limited newer experiment; the unresolved measurement and identity questions determine what should be tested next.
