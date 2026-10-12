# Notebook and implementation evidence audit

Audit date: 8 October 2026, America/Los_Angeles. Cell indices below are zero-based positions in the notebook JSON, including Markdown cells. This audit reads all seven root notebooks, their code, Markdown, and saved text outputs; it also inspects the saved training, t-SNE, and UMAP images and the active implementation. It does not rerun model training or modify notebooks or model code. Small deterministic mask calculations and cache inventories were recomputed on the current implementation and are explicitly distinguished from saved experimental results.

## Evidence hierarchy and experimental identity

The source-grouped full-data experiment is different from the older `g1`/E0/R1 experiments and from the standalone Fold-0 teaching execution. Its registry SHA-256 is `9496e61b050fc8b62ccbc59cb737376396205a3356b323733b9b8915d56600e1`; its laptop configuration directory is `laptop-1d09c8e1eea5`. Results from different inventories, masks, code versions, or checkpoints must not be merged.

| Notebook | Total cells | Code cells | Code cells with execution counts | Status of saved evidence |
|---|---:|---:|---:|---|
| 00 overview | 12 | 6 | 6 | Executed inventory and gallery |
| 01 extraction | 16 | 9 | 9 | Executed sample extraction, cache inventory, exclusion log |
| 02 masking | 18 | 10 | 10 | Executed split, mask-bank diagnostics, and mask illustration |
| 03 training | 20 | 8 | 8 | Executed standalone Fold-0 training and diagnostic summaries |
| 04 continuation | 17 | 7 | 7 | Executed standalone Fold-0 continuation and validation selection |
| 05 visualization | 21 | 8 | 8 | Executed training-only representations, plots, and silhouette |
| 06 capstone | 20 | 7 | **0** | No saved code-cell results; numerical claims are Markdown summaries of an external saved run |

The user's source-of-truth instruction makes the top-level notebooks authoritative for the experiment being described. Notebook 06 is a readable description and entry point, **not a file with saved executed capstone outputs**. Its Markdown reports the historical run: 216 seconds, S-JEPA source-weighted macro-F1 0.457, RF 0.411, and mean pose 0.452, with an explicit left-ankle-range defect. There is no `latest_run` lookup or literal capstone result directory in its code. Cell 10 would create a new `OUTPUT_DIR` if executed; it does not select an existing newest directory. Identify the retained historical `results.json`/`oof.json` matching the notebook, then independently verify those values. Do not silently replace the notebook's RF with a later number.

The parallel results audit also identifies a newer complete run at `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20261009T060955236580Z`, whose corrected RF differs. That is **additional external-to-notebook audit evidence**, not a replacement for the notebook record under the user's instruction. The original and corrected runs may be reported separately with explicit provenance if appropriate. UTC 9 October corresponds to the local evening of 8 October; the date is not evidence of a future experiment.

The historical artifact matching notebook 06 is `artifacts/runs/full-v1/9496e61b050f/laptop-1d09c8e1eea5/capstone-20260921T011305448893Z`: its recorded wall time is 215.981 seconds, RF source-weighted macro-F1 is 0.4112629104, and S-JEPA is 0.4566241591. These match the notebook's rounded prose and must be the primary pair if the paper follows that experiment. The detailed numerical audit independently recomputes OOF metrics and confidence intervals; neither its later recomputation nor the 5,000-mask calculation below should be described as a saved notebook code-cell output.

The current notebooks 00–05 have filesystem timestamps on 9 October UTC, but their Markdown includes historical September descriptions. A file timestamp alone does not establish execution provenance. The numeric output should take precedence over stale narrative, and the matching checkpoint/run must be identified for any manuscript claim. The standalone Fold-0 checkpoint directory is separate from each `capstone-*` directory; matching seeds and scores do not turn these into independent seeds or identical checkpoint bytes.

## Notebook 00: collection and source grouping

Cell 8 reports 91 raw clips from 41 source IDs: Normal 26/16, MS 30/13, PD 35/12 (clips/sources). Cell 10 embeds one example per label, which is an illustration rather than a systematic assessment of pose quality or diagnosis. Cell 9 notes concentration in one MS source and one PD source. The model labels are collection labels, with no demonstrated clinical adjudication.

`source_id_from_name` strips the documented `_P...` or `_clip-...` suffix while protecting genuine 11-character IDs (`sjepa/data.py:189`). The full inventory validates the filename pattern and enforces one label per source (`sjepa/splits.py:33,50`). This prevents known same-source clips crossing a partition. It does not establish that different source IDs show different people, exclude reposted footage, or confirm diagnoses. “41 participants,” “patient-independent,” and “clinically verified MS/PD” would be unsupported descriptions.

## Notebook 01: extraction, cleaning, coordinates, and cache

Cell 6 extracts a sample Normal clip, `-0wbleNgAwg_P1_01.mp4`, producing `(68,33,3)` at approximately 15 fps and reporting 54% fully detected frames. Cell 9 reports 48 frames after trimming/cleaning and normalized horizontal coordinates from −0.71 to 1.39. Cell 13 reports retained frames of 1,427 Normal, 3,450 MS, and 2,766 PD; 88 usable clips remain, with all 41 source IDs still represented. The three rejected clips are `JD1AGVpftps_P1_02`, `diCVwltkV5M_P1_01`, and `tsOMPBS277Q_P1`. The common “too few valid frames” message does not identify whether the initial 30% detection gate or the later eight-frame minimum failed. The reviewed exclusion file explicitly retains that uncertainty.

Cell 15 prints that all 88 sequences are shaped correctly, but its local loop checks only `recs[:5]`. Later full-registry inventory validation does check all cached arrays for shape, finite values, frame count, and fps metadata. Cite the registry validator for an all-record check, not the cell's stronger printed wording.

### Correct causal order and the three coordinate conventions

The draft's Section 3 wrongly describes cleaning/normalization before detection and conflates image-normalized coordinates with body-normalized model coordinates. The actual path is:

1. OpenCV decodes the whole frame; its BGR image is converted to RGB. Frame indices are retained at integer stride `max(1,round(source_fps/15))` (`sjepa/data.py:45`). There is no exact interpolation/resampling onto a common 15-Hz timeline.
2. `SequenceKeypointExtractor.extract_from_image` invokes MediaPipe on each retained image. The wrapper takes `pose_landmarks[0]` (`ambient/pose/keypoint_extractor.py:194`). The landmarker factory uses IMAGE mode and defaults to one pose; this is not a tracked participant identity across frames.
3. MediaPipe's image-normalized landmark coordinates `(u,v)` are converted to pixel coordinates `(W u,H v)` by `KeypointSet.from_mediapipe` (`ambient/pose/keypoint_data.py:523`). Separate clamped `x_normalized`/`y_normalized` fields also exist, but the S-JEPA extractor does not read them. The pixel fields use the original, unclamped landmark values. `z` is available in the ambient object but is not placed into this experiment's arrays.
4. `_keypointset_to_row` saves pixel `x`, pixel `y`, and MediaPipe `visibility` into `(T,33,3)`, with NaNs for missing detections (`sjepa/data.py:106`). Visibility is not anatomical depth, disease confidence, or a measured physical gait quantity.
5. `clean_sequence` first rejects a sequence if fewer than 30% of sampled frames have finite x coordinates for all 33 joints. It trims ends where no x coordinate is finite, then interpolates each joint/channel over time using `numpy.interp`; all missing channels are set to zero (`sjepa/data.py:130`). There is **no maximum interior gap length** despite the docstring and notebook phrase “short gaps.” For a channel missing at one end of the retained interval, interpolation also uses the nearest finite endpoint. Visibility itself is interpolated. No low-visibility coordinate threshold is applied at this cleaning stage.
6. Let pixel joint position be `x_tj`, pelvis midpoint `p_t=(x_t,23+x_t,24)/2`, and shoulder midpoint `s_t=(x_t,11+x_t,12)/2`. Model coordinates are `(x_tj-p_t)/max(||s_t-p_t||_2,10^-3)` for every frame (`sjepa/data.py:164`). Visibility passes through this normalization unchanged after the earlier cleaning. The scale is a per-frame torso length, not image width or height.
7. The NPZ stores both **cleaned pixel coordinates** under `keypoints` and **torso-normalized coordinates** under `keypoints_norm`. `load_raw()` means the first of these, not uncleaned detector output or raw video. The encoder and mean-pose control read the second; RF and the visibility control read the first.

This removes absolute pelvis translation and apparent torso scale and preserves defined within-frame angles under a common positive scale. It does not recover calibrated 3D position, physical speed, stride length in meters, or viewpoint invariance. A scale that changes between frames also changes apparent inter-frame motion. Unbounded interpolation can attenuate irregular movement. The manuscript should describe these as limitations of the observation process, not as established explanations for any particular prediction error.

The target fps is only nominal. For example, an original 24-Hz stream uses stride two and becomes about 12 Hz. Cached `fps=15` records the requested rate, not each clip's realized cadence. A 32-frame window is nominally about 2.1 seconds; do not assert identical physical duration across clips. The saved cache does not preserve original timestamps or a gap mask, so it cannot support a retrospective estimate of interpolated fractions from normalized arrays alone.

## Notebook 02: masks and tokenization

Cell 6 uses the full shared registry. Cell 9 prints laptop dimensions: 32 frames, 33 joints, three channels, four frames per token, eight time positions, and 264 tokens. The `pretrain 40ep, finetune 30ep` strings are legacy configuration fields; the active loop uses update counts. Cell 11 lists the twelve clinically motivated indices `{11,12,23,24,25,26,27,28,29,30,31,32}`. They guide region-sampling weights but do not restrict the input to twelve landmarks or define a permanent target set.

Cell 13 samples six distinct masks, each containing context and targets. Its saved 512-mask seed-0 bank has mean target fraction 0.63; minimum fraction of windows in which a joint is ever visible is 0.74, and minimum ever-targeted fraction is 0.81. Recalculation gives 0.6327607126, 0.740234375, and 0.80859375, respectively. These are sampler diagnostics, not measurements of realized token exposure during every capstone training run. Token visibility differs substantially from visibility anywhere in a window: shoulders and hips are visible in only about 20–23% of tokens, even though they are visible somewhere in roughly 74–79% of windows. At middle block 5, left/right hip context frequencies are only 7.0%/7.6% in the saved bank, versus 51.6%/45.5% at block 1. Passing the coverage gate does not establish uniform, balanced, or clinically beneficial masks.

Cell 17 replays eight masks on one training window. All joints occur in both roles across that small bank, yet masks 1 and 6 hide both hips for all eight blocks. The animation is a fixed seeded replay, not a new random draw on every playback.

### Exact sampler, including the omitted multiplicity correction

`sjepa/masking_v2.py:37` defines seven anatomical groups: head `(0–8)`, mouth `(9,10)`, left arm `(11,13,15,17,19,21)`, right arm `(12,14,16,18,20,22)`, trunk `(11,12,23,24)`, left leg `(23,25,27,29,31)`, and right leg `(24,26,28,30,32)`. All 33 landmarks can be targeted. For group G, its pre-normalization sampling weight is

`w_G = b_G / mean_{j in G}(m_j)`,

where `b_G=1.5` if G contains any of the twelve selected joints and `1` otherwise, and `m_j` is the number of groups containing joint j. Weights are then normalized across groups (`_region_weights`, line 84). Shoulders belong to arm and trunk; hips belong to leg and trunk. The correction is approximate, so duplicated membership still affects final marginal exposure. Arms receive the 1.5 factor because each contains a shoulder; this also increases targeting of their nonselected elbows, wrists, and hand landmarks. “The twelve joints are masked 1.5 times more often” is false.

The resulting group-selection probabilities are head 0.127854, mouth 0.127854, each arm 0.164384, trunk 0.095890, and each leg 0.159817. A region draw chooses a contiguous time span uniformly from lengths one through six and chooses a valid start uniformly. Its joint–time rectangle is added to the union until reaching the rounded target budget `round(.6*264)=158`, subject to a 36-draw guard. A whole last region can overshoot the budget. The code clears a random selected-set token if every such token is hidden, guaranteeing selected-set context somewhere in the window; it separately guarantees nonempty target and context. It does not guarantee a visible hip, a visible leg, context in every block, or that every joint is visible in an individual window. Overlapping spans can cover a joint for the entire window even though an individual span is at most six blocks. Middle blocks occur in more admissible intervals and are masked more frequently than boundary blocks.

A new, deterministic seed-0 bank of 5,000 masks on the current code gives target fraction 0.6331606061. Target marginals are 77.84%/78.31% for shoulders, 77.26%/76.87% for hips, about 62.06–62.54% for other leg joints, 63.61%/64.30% for corresponding arm joints, 59.65% for the nine head landmarks and 52.18% for the two mouth landmarks. Averaging all eleven facial landmarks gives about 58.3%; do not use 58% as if each facial landmark had that marginal. These are fresh audit diagnostics, not a new classification experiment. Their downstream benefit remains unmeasured because there is no mask ablation.

The notebook's suggestion that motion-based masking is “contraindicated” because reduced motion may be informative is a rationale, not an established result. No experiment here tests that alternative or shows that the 1.5 factor preserves a disease-specific signal. The paper should explicitly identify this as a chosen inductive bias with unknown classification effect.

## Notebook 03: active representation learner and saved training evidence

Cell 7 prints the registry and Fold-0 partition: 51 training clips/24 sources, 19 validation clips/8 sources, and 18 test clips/9 sources. Training contains 15 Normal, 23 MS, and 13 PD clips from 9, 8, and 7 sources. Cell 10 builds `repaired=True`, the active predictor with spatial and temporal positional embeddings; it chooses MPS. The historical filename `pretrain_normal` does not describe the active training population.

Cell 14 runs 800 label-free updates and prints final effective rank `12.134888648986816`. Cell 16 shows loss falling from about six toward one, increasing effective rank, and a small late teacher-drift statistic. Cell 18 is stronger evidence than approximate graph readings because it prints exact summaries:

| Diagnostic | First-50 mean | Last-50 mean | Final point |
|---|---:|---:|---:|
| Loss | 3.858368 | 1.022220 | 1.097049 |
| Effective rank | 8.132109 | 10.939291 | 12.134889 |
| Embedding standard deviation | 0.153868 | 0.525584 | 0.549869 |
| Teacher drift | 0.022306 | 0.019918 | 0.019944 |

Cell 18 also reports 25,600 draws across 24 training sources, with observed counts from 1,025 to 1,129. Source-uniform probability does not require equal finite-batch exposure. Cell 17's older exact rank `12.134796142578125` disagrees slightly with the current code-cell output. Its dated narrative and assertion that only rank was printed numerically are stale after the addition/execution of cell 18.

The effective-rank code centers a batch of all-token teacher window embeddings and takes entropy of normalized singular values (`train_v2.py:83`). With batch size 32, the centered rank is bounded by 31, not 96. This argues against completely constant outputs in these batches, not against every kind of collapse or against representation of nuisance factors. Teacher drift averages `1-cosine` equally over parameter tensors before the EMA update; zero-initialized tensors can produce a misleading initial spike near 0.31. It is not teacher movement since initialization or a percentage error.

### Model and optimization actually used

`SkeletonTokenizer` takes each joint's four successive `(x,y,visibility)` observations, linearly projects 12 values to 96, and adds learned joint and time-position embeddings. Token index is `33*t+j`. The encoder uses three pre-norm transformer blocks, four heads, MLP ratio two, GELU, zero dropout, and final layer normalization. The predictor uses two blocks of width 96 with its own factorized joint/time position embeddings. The teacher begins as a copy of the student and is updated by EMA, with no loss gradient.

For each example, the view encoder tokenizes the complete augmented array and uses an attention key-padding mask that excludes target slots as keys/values. Hidden outputs are then replaced by a learned mask token before the predictor. Thus hidden values do not influence visible context through attention; the code does not physically remove hidden token computations in its per-example implementation. The teacher sees the complete unaugmented training window. The loss predicts teacher features at target positions, rather than reconstructing coordinates or predicting diagnoses (`models.py:108,161,283`).

The student probabilities are `softmax(u/.10)`, teacher probabilities are `softmax((z-c)/.06)`, and detached-teacher cross-entropy is averaged over target tokens within each example and then equally over examples. The center updates once per batch using the token-weighted mean of masked teacher features and momentum .9 (`losses.py:71`). AdamW uses lr .001, decay .05, betas (.9,.95), norm clipping 1.0, a 10%-of-stage warmup, and cosine lr decay. Teacher EMA momentum follows a cosine from .996 toward 1.0. The loop uses all training conditions without their labels, draws windows with replacement using inverse window count per source, and has one mask per example. No class-aware VICReg term or progressive addition of conditions is used (`train_v2.py:112`; `full_experiment.py:91`).

Student augmentation applies a single geometric transform per window: rotation in ±15 degrees, scale [0.9,1.1], translation ±0.1 torso units per axis, and horizontal reflection with probability .5. Reflection negates x and swaps bilateral joint/visibility slots. The third channel is visibility throughout; a stale `augment.py` docstring incorrectly mentions weak depth. Time order is not shuffled, but retention of order alone does not prove preservation of all diagnostic motion information.

### Mask/reflection correspondence defect

The target mask is sampled in original joint slots, then the input is reflected and left/right slots permuted. The mask is **not permuted to follow physical joints**, while the teacher stays untransformed (`train_v2.py:204–219`, `augment.py:83–90`). Let pi swap left/right indices. If a teacher target joint j has `m_j=1` and `m_pi(j)=0`, reflected student context at pi(j) contains transformed observations of physical joint j. Intended withheld-joint information can therefore be present on the opposite side. This is a within-training target/context semantic defect, not access to held-out sources or a condition-label leak.

The fresh 5,000-mask audit bank has at least one such cross-side target/context mismatch in 99.86% of masks. Conditional on reflection, it exposes an average 35.3856 target tokens; the pooled exposed share of target tokens is 21.1694%, or 10.5847% in expectation when multiplied by reflection probability .5. The latter is an analytic expectation over the mask bank, **not measured exposure from the saved training stream**. This diagnostic does not estimate the effect on classification. A corrected or no-reflection experiment would be needed; its benefit must not be assumed. A valid repair could align the student context mask with the landmark permutation or remove reflection, but either changes the trained procedure and requires clearly labeled new results. No model code was changed in this audit.

## Notebook 04: continuation, linear probe, and selection

Cell 8 loads `ssl.pt`; cell 10 trains 400 further label-free updates on the same training records. Encoder, predictor, mask token, and teacher weights carry over, while optimizer, center, lr/EMA schedules, and seeded sampling streams restart. This is a two-stage 800+400 recipe, not one uninterrupted 1,200-update run. Reexecuting the training cell without reloading the starting checkpoint adds more work to the current weights and overwrites the named checkpoint; filenames alone do not certify update history.

Cell 13 fits a separate probe for each candidate, scores source-weighted validation macro-F1, and prints `ssl: 0.29397943186565934`, `continued: 0.27610529941022`, choosing `ssl`. The difference is −0.01787413246. This is Fold-0 validation evidence, not test performance. The class-F1 breakdown in cell 14 is described as a separate later CPU check, not output from cell 13. Do not cite it as if printed by that cell without retaining its independent derivation. After cell 13, local variables `m` and `probe` remain at the loop's final `continued` iteration even though `selected='ssl'`; downstream notebook 06 correctly reloads the selected checkpoint and its matching probe.

The probe is `StandardScaler` plus `LogisticRegression(C=1,max_iter=2000,class_weight='balanced',random_state=42)`, fit only on training clips (`full_experiment.py:56`). `max_iter` is a cap, not 2,000 observed optimization iterations. Class weighting balances label contributions, not source contributions within a label. In Fold 0, 13/23 MS training clips share one source, so that source supplies 13/23 of the MS class's loss weight. Source-uniform SSL and source-weighted scoring do not correct this supervised-head weighting.

### Correct encoder readout

`embed_records` reads the **EMA target encoder**, not the context/student encoder (`full_experiment.py:17`, `models.py:300`). Each full unmasked window is encoded. A fixed seed-0 sample of the mask sampler selects **161 of 264 output token vectors** for an arithmetic mean, then all window vectors are averaged equally into a 96-dimensional clip vector. This readout selection does not hide input coordinates. All 33 landmarks appear at least once in the selection; all ten leg landmarks 23–32 occur in all eight blocks and both shoulders in seven blocks. The readout differs from the all-token embeddings used for training diagnostics. Calling this “pooling twelve clinical joints” or “encoding masked test inputs” would be wrong.

## Notebook 05: descriptive training representations

Cell 9 loads both standalone Fold-0 checkpoints and extracts vectors only for its 51 training clips/24 sources, saving `training_embeddings.npz`. It also computes 66 visibility features, concatenating each of 33 landmarks' mean and population standard deviation over the entire cleaned clip. Cell 11 standardizes each representation independently on those training clips and uses t-SNE with seed 42, PCA initialization, and perplexity 15. Cell 14 uses UMAP with seed 42, 15 neighbors, and minimum distance .3. Both visualizations were inspected; their groups contain mixed labels. Axes, absolute map positions, and gaps have no clinical units, and the two plots are different views of the same training examples.

Cell 17 scores silhouette in **standardized original feature space**, not 2D projected coordinates, and reports `ssl −0.004`, `continued −0.012`, `visibility −0.004`. It averages clips without equal-source weighting. The near-zero values support weak average Euclidean grouping by collection label on these training vectors. They do not prove absence of useful classifier information, equality of representations, or statistically established deterioration. The visibility and SSL values match only after rounding. The current plots do not color by source; repeated-source clustering cannot be separated from cross-source label clustering by visual inspection. UMAP warnings concern threading/widgets, not failed projection. Historical detailed spatial descriptions should be checked against the retained images before reuse.

## Notebook 06 and the full evaluation implementation

The code is unexecuted within the file. Its standalone Markdown metrics, runtime, RF implementation status, and biological interpretations therefore need reconciliation against the chosen saved capstone artifact. The active `run_fold` trains a new model for each outer fold and independently evaluates 800-update and continued candidates on that fold's validation sources. Strictly higher validation source-weighted macro-F1 selects continuation; ties retain the initial stage. It does not refit the selected probe on validation clips. Test embedding extraction follows selection. Nonlearned per-clip features can be precomputed/cached for all clips because this computation is stateless; their scaler and classifier are still trained on the training partition only. Cache reuse is not extra independent experimental replication.

The outer split is five-fold `StratifiedKFold` on one row per source, shuffle seed 42. Each outer development set has a four-fold splitter with seed 43+k, but **only its first split** is used as the inner validation holdout. This is not full nested inner cross-validation. Every retained clip receives exactly one OOF prediction, and source IDs are disjoint among train, validation, and test within each fold. The code validates these facts (`splits.py:86,125`; `full_experiment.py:208`). These controls address known source overlap; they do not retrospectively prove that all exploratory decisions preceded test inspection.

Equal-source evaluation assigns each clip weight `1/n_s` for its source s, forms weighted confusion counts, computes classwise F1, then averages three classes. It does not merge clips into a source label, average per-source F1, or measure performance on 41 verified people. Fold mean and pooled OOF macro-F1 differ because F1 is nonlinear; population SD over five fold scores is not a confidence interval. Folds have overlapping training sets, and a seed-42 run is not a multi-seed study. If adding a source-cluster bootstrap to frozen OOF predictions, label it post hoc and conditional on these trained models; it does not include retraining variability or fix unverified source independence.

### Baselines and feature differences

The mean-pose control averages all 33 torso-normalized x/y positions over every retained frame, yielding 66 numbers. It discards frame order but can still encode average posture, body proportions, occupancy of gait phases, viewpoint, and preprocessing. The visibility control uses 66 mean/std features from cleaned visibility. Both use the same fixed scaler/logistic-regression recipe as the learned readout, with their own training-only fitted parameters. They have different dimensions and do not receive S-JEPA's two-stage validation choice. Full-clip controls include tail frames that the encoder's full-window enumeration can omit; the fixed readout also weights joints/time positions unequally. These are comparisons of complete procedures, not pure causal interventions on temporal information.

RF reads cleaned pixel coordinates and computes six 2D hip/knee/ankle angle means, three absolute bilateral mean differences, and six ranges through the legacy `GaitFeatureVector.from_joint_angles` path. Angle confidence is the geometric mean of the three landmark visibility values, with threshold .3; degenerate geometry is rejected. All-NaN joint statistics become zero. The feature container has 82 named slots, but this path does not compute gait speed, cadence, stride length, balance, or full clinical kinematics. Constants are removed using training data, leaving the varying angle summary columns. RF uses 100 trees, depth 5, square-root feature sampling, class balancing, seed 42, and training-only standardization. Shared pose extraction/splits alone do not make it a complete state-of-the-art or budget-matched classical gait baseline.

**The ankle-range bug belongs to the notebook's historical result and has been corrected in the current code.** The original line assigned `left_ankle_range` from `right_ankle_stats`, duplicating the right range. Current `ambient/classification/features.py:912` assigns `left_ankle_stats`, and the working-tree diff shows exactly that correction. Notebook 06 states the defect is present and quotes the historical RF score; retain that pairing as the notebook-authoritative result. The parallel numerical audit reports a changed RF score in later capstone artifacts and is responsible for independently verifying it. The current source code describes what a new execution would do, not the feature implementation behind the older number. Preserve both run identities if discussing the correction, without implying a performance improvement from fixing a feature bug. This audit made no change to that user-modified file.

The majority control selects the most common **training-clip** label, with label-list order resolving ties; it does not balance by source or use test labels. Three classes do not imply a universal macro-F1 chance threshold of 1/3.

## Newly recomputed cache quantities

These read-only counts use the retained cache and `sliding_windows(window=32,stride=16)`:

| Label | Retained clips | Retained frames | Padded clips | Windows |
|---|---:|---:|---:|---:|
| Normal | 24 | 1,427 | 5 | 59 |
| MS | 29 | 3,450 | 0 | 171 |
| PD | 35 | 2,766 | 10 | 128 |
| Total | 88 | 7,643 | 15 | 358 |

Retained lengths range from 14 to 450 frames. Fold train/validation/test window totals are 207/56/95, 226/55/77, 238/83/37, 224/79/55, and 190/74/94. Counts of windows are not independent observations. Short clips repeat the final frame; long clips use full windows at stride 16 without forcing a last window at the clip end. Padding is label-associated here, providing another possible cue whose impact was not isolated.

## Section-ready replacement language

**Preprocessing and coordinates.** “MediaPipe Pose Landmarker was applied to retained full video frames. Its image-normalized landmark positions were converted to pixel coordinates using frame width and height, and the first returned pose supplied 33 `(x,y,visibility)` landmarks; depth was discarded. We rejected clips with complete finite x-coordinate detections in fewer than 30% of sampled frames, trimmed empty ends, and interpolated missing values independently for each landmark and channel without a maximum gap length. Clips with fewer than eight cleaned frames were excluded. We stored both these cleaned pixel coordinates and a second representation centered at the hip midpoint and divided by the per-frame distance between hip and shoulder midpoints, clamped below at `10^-3`. The encoder and mean-pose control used the torso-normalized representation; RF angles used cleaned pixel coordinates. Visibility was interpolated during cleaning but unchanged by spatial normalization. Integer-stride frame subsampling targeted approximately 15 fps and did not standardize exact elapsed time.”

**Masking.** “All 33 landmarks entered the encoder. Each training example received an independently drawn mask formed by unions of seven anatomical groups over contiguous spans of one to six of the eight time blocks. Group weights were multiplied by 1.5 when the group contained a shoulder or lower-limb landmark, then divided by the mean group-membership multiplicity of its joints and normalized. This made the twelve selected landmarks an input to the sampling rule, rather than a fixed hidden set or a restriction on the input. Regions were added until approximately 60% of tokens were targets; the final region could overshoot the target. The sampler retained selected-set context somewhere in the window, but a joint could be hidden throughout a window and individual time blocks could lack context. A separate seed-0 bank of 5,000 masks targeted 63.3% of tokens on average and revealed uneven joint and time coverage. No ablation establishes that this sampling bias improves classification.”

**Readout.** “For classification, the frozen EMA target encoder processed each complete window. We averaged 161 output token features selected by one fixed seed-0 mask and then averaged the resulting 96-dimensional window vectors across the clip. The readout mask selected output positions; it did not hide any input landmarks. A separately fitted training-only scaler and class-balanced logistic-regression probe predicted the three collection labels.”

**Correspondence limitation.** “Horizontal reflection permuted bilateral landmark slots in the student view without permuting its context mask, while teacher targets stayed in the original slots. Consequently, transformed observations of a target joint could appear in the student's opposite-side context. This limits interpretation of the learned objective; its effect on classification has not been isolated in a corrected or no-reflection experiment.”

## Root causes and required manuscript corrections

| Root cause | Observable consequence | Required correction |
|---|---|---|
| Image, pixel, and torso coordinate conventions merged in prose | Sections 3 and 4.2 appear contradictory and causal order is reversed | Describe the ordered conversion/cleaning/normalization chain once and name the cache used by each branch |
| Names retained from legacy notebook/code stages | Apparent normal-only pretraining, progressive diagnoses, VICReg, epoch schedules | Report the active all-condition label-free update-based loop |
| Source comments treated as experimental evidence | Claims that motion masking is unsuitable, that collapse is prevented, or that geometric transformations preserve all useful gait signal | Mark motivations as hypotheses and restrict claims to measured diagnostics/held-out comparisons |
| Clinical landmark list mistaken for mask or input restriction | Twelve-joint encoder and exact 1.5× per-joint targeting claims | All 33 inputs; seven weighted groups; multiplicity correction; realized token marginals and overshoot |
| Mask/augmentation coordinate systems not aligned | Reflected context can contain intended target physical joints | Disclose the defect, keep its effects unquantified, require a labeled corrective experiment for causal conclusions |
| Student and teacher/readout roles conflated | Wrong encoder named; output selection described as input masking | Specify full-input EMA teacher and 161-token output pooling |
| Markdown and current implementation describe different run versions | Historical RF score/defect and old tutorial diagnostics coexist with later code/executions | Honor notebook-authoritative historical results and identify their matching OOF artifacts; report newer corrected runs separately |
| Pooled source and clip/fold statistics conflated | Wrong headline F1, apparent universal winner, false independent repetitions | Define weighting, pooled calculation, fold dispersion, bootstrap scope, and one seed |
| Baseline container names mistaken for computed gait measurements | “82 gait features,” physical speed, stride, or biomarker claims | State the actually populated angle summaries and observational limitations |
| Small web-video pilot given clinical semantics | Participant-level independence, diagnosis, causal temporal benefit | Collection-label prediction on known source groups; no clinical validation or isolated motion benefit |

The strongest defensible manuscript is a controlled empirical pilot asking what source-grouped evaluation and order-free controls permit one to infer about this particular skeleton-prediction procedure. It cannot establish neural decoding, disease mechanisms, validated clinical biomarkers, general S-JEPA superiority, or equivalence of temporal and static representations.
