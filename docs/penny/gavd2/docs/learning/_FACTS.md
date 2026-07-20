# Ground-truth facts for the gavd -> gavd2 learning paper

Internal shared spec. NOT user-facing. Every number is verified from the notebooks,
the gavd/_RESULTS_GROUND_TRUTH.md file, the gavd2 docs/paper.md, and the project
memory. Writing rules for ALL sections that draw on this file:

- Audience: a serious high school student. Plain words, short sentences, complete
  thoughts. Explain every term the first time it appears. Do NOT assume ML background.
- NO em-dashes and NO en-dashes anywhere. Use commas, periods, or the word "to" for ranges.
- Do not use fancy words. Prefer "used" over "leveraged", "shows" over "demonstrates",
  "small" over "diminutive". Keep it natural and readable start to finish.
- IEEE citation style, numeric [1], [2] in text, full list at the end.

## The one-sentence story

The first attempt (folder `gavd`) built the whole pipeline but three quiet bugs made its
headline number meaningless. The second attempt (folder `gavd2`) fixed the bugs and, more
importantly, turned the comparison against the old baseline into a fair, controlled one.
The honest result is smaller than the first flashy number, but it is real.

## The task and the dataset

- Goal: read a walking video and name the gait condition (5 classes:
  normal, parkinsons, stroke, cerebral palsy, myopathic).
- The bottleneck is labels. Unlabeled walking video is cheap; clinician-graded clips
  are scarce and expensive. So we pretrain on lots of unlabeled video and spend the
  few labels only at the very end.
- Dataset: GAVD (Gait Abnormality in Video Dataset) [4]. 374 sequence CSVs across 11
  condition folders. 91,624 frame-rows, mean 245 frames per sequence.
  Per-condition CSV counts: abnormal 190, antalgic 4, cerebral palsy 16, exercise 24,
  inebriated 2, myopathic 47, normal 12, parkinsons 9, prosthetic 3, stroke 12, style 55.
- Each CSV is one sequence = a run of frames from ONE YouTube video. A single video can
  back more than one sequence.
- The "exp5 68": the prior study curated exactly 68 sequences over 5 classes:
  normal 12, parkinsons 9, stroke 12, cerebralpalsy 15, myopathic 20 = 68.
  Those 68 come from only 12 unique YouTube videos.
- The prior baseline [4]: a Random Forest, 100 trees, on 82 hand-engineered gait
  features (joint angles, step timing, symmetry indices). Evaluated per sequence on a
  single seed-42 70/30 split (47 train / 21 test). Best test accuracy = 0.762.
  Chance on 5 classes = 0.20. So 0.762 is a real, hard result and the reference point.

## What a JEPA is (Joint-Embedding Predictive Architecture)

Four pieces:
1. Context encoder (the "online" network): sees the VISIBLE joints/frames of a masked
   clip, produces a context embedding. A small transformer.
2. Target encoder: an EMA (exponential moving average) copy of the context encoder,
   updated as target = m*target + (1-m)*context with m = 0.996, stop-gradient (no
   gradients flow to it). It sees the full clip and gives the "answer key" embeddings.
3. Predictor: a shallow MLP that maps the context embedding at the hidden spots to a
   guess of the target embedding there.
4. Masking: decides which tokens are hidden. Block masking, two styles (see below).

Key idea: predict the hidden content in a LEARNED latent space, not in raw pixels/coords.
JEPA family papers: I-JEPA [1] (images), V-JEPA [2] (video, contributed the LayerNorm-on-
target trick), VICReg [3] (anti-collapse without negatives), LeCun world-model framing [6].

## Encoder architecture (exact, gavd2 baseline nb04/nb05)

- A pose clip is a tensor of shape (T=32 frames, J=33 joints, 3 coords) = (32, 33, 3).
- Flattened into T*J = 1,056 tokens in ROW-MAJOR (t, j) order: token n = t*33 + j.
- ContextEncoder: Linear(3 -> 64) input projection; learned time_embed (32, 64) and
  joint_embed (33, 64), both init std 0.1; position added as pos[t,j] =
  time_embed[t] + joint_embed[j]; then a 2-layer nn.TransformerEncoder (4 heads,
  feed-forward 128, GELU, dropout 0, batch_first). Total encoder params = 71,360.
  EMBED_DIM D = 64.
- Target encoder: EMA copy, m = 0.996, stop-gradient.
- Predictor: shallow MLP (Linear-GELU-Linear, hidden 2D).
- Training: 400 real steps, batch 16, LR 1e-3, Adam, CPU, seed 42.

## The 33 BlazePose joints and the limb GROUPS (exact, from nb04 cell 9)

BlazePose [5] gives 33 landmarks per frame. The code groups them:
  face:      [0,1,2,3,4,5,6,7,8,9,10]
  left_arm:  [11,13,15,17,19,21]
  right_arm: [12,14,16,18,20,22]
  torso:     [11,12,23,24]
  left_leg:  [23,25,27,29,31]
  right_leg: [24,26,28,30,32]
Landmark 27 = left ankle, 28 = right ankle, 25 = left knee, 26 = right knee,
23 = left hip, 24 = right hip. (Left = odd-ish indices on one side; the code treats
27/28 as left/right ankle for the clinical scalars.)

## Masking (exact, nb04 make_block_mask + make_batch)

Two block-masking styles, chosen 50/50 per clip, MASK_RATIO = 0.4:
- Style A "limb over time": hide ONE whole limb (left_arm, right_arm, left_leg, or
  right_leg) across a window of consecutive frames. The model must reconstruct that
  limb's motion from the OTHER limbs and the frames around it.
- Style B "time window": hide ALL 33 joints across a short window of frames. The model
  must fill a gap in time.
- Contrast (too easy, not used): scattering single random joints. A lone hidden joint
  is guessable from its own neighbors, so no coordination is learned. Blocks force the
  model to reason about coordinated whole-body motion, which is what walking is.

## The loss (exact, the CORRECTED version)

L = VICREG_SIM * sim_loss + VICREG_VAR * var_loss + VICREG_COV * cov_loss
- sim_loss = MSE between the predictor output and the LayerNorm-normalized EMA target.
  LayerNorm makes the loss measure DIRECTION, not size, so a drifting target scale
  cannot inflate it (V-JEPA trick [2]).
- var_loss, cov_loss = VICReg variance and covariance guards, applied to the ONLINE
  context embedding ONLY (never the stop-gradient target).
- Weights: VICREG_SIM 25.0, VICREG_VAR 0.5, VICREG_COV 0.04, VAR_TARGET (gamma) 0.5,
  EPS 1e-4.

## THE THREE BUGS in the first attempt (gavd), all real, all fixed

### Bug 1: the pipeline never ran on real data (silent synthetic cache)
- Notebooks 00 to 03 ran their SMOKE synthetic path while 04/05 had SMOKE_TEST=False.
  The synthetic caches passed silently: 05's only guard was `clips.shape[0] < 10`, and
  the synthetic holdout had 26 clips (26 >= 10, so it passed).
- Evidence it was synthetic: corpus.npz was (16, 32, 33, 3); labeled_holdout.npz was
  (26, ...); ALL seq_ids contained the string "synthetic"; class counts were
  {6,5,5,5,5}, NOT the real {12,9,12,15,20}=68.
- Result: the "real" run scored about 0.25, which is just noise from an 8-item test set.
- Fix direction (finished in gavd2): mode-stamped provenance, a canonical_id_hash on
  every artifact, and a hard fail-stop so a synthetic/stale cache can never masquerade
  as a real run.

### Bug 2: the encoder was permutation-invariant (no positional embeddings)
- The first ContextEncoder.forward was just `transformer(input_proj(x))` with NO time or
  joint or positional embedding. A plain transformer treats its 1,056 tokens as an
  UNORDERED set. The probe then mean-pools all tokens, so the clip embedding was a fully
  order-agnostic BAG of points. It erased "which joint moves when", which is exactly the
  definition of a gait class.
- Fix: add a learned time_embed (T, D) and joint_embed (33, D), std 0.1, added as
  pos[t,j] = time_embed[t] + joint_embed[j] over the row-major (t,j) tokens. Now the
  encoder knows frame 0 from frame 31 and a left knee from a right knee. This is the
  standard I-JEPA / V-JEPA fix. Verified that permutation-invariance is now broken.

### Bug 3: the training loss drifted UP (VICReg on the target + un-normalized L2)
- The first loss applied VICReg variance+covariance to BOTH the prediction AND the EMA
  target, with heavy weights (25/25/1), gamma 1, and NO target normalization.
- Over a long run (400 steps) the variance term kept inflating the target encoder's
  embedding scale (nothing pulled it back), and the un-normalized L2 turned that
  inflation into a RISING loss. Total loss fell for ~50 steps then climbed; MSE climbed
  with it. This is NOT collapse: collapse drives embedding std toward 0, but here the
  std kept RISING. The pairing (MSE up + std up) is the tell.
- Old loss numbers (400 steps): total rises ~46 to 62; MSE rises ~0.36 to 0.47; std inflating.
- Fix (3 parts, all needed): (1) LayerNorm the EMA target before the L2 (scale-invariant,
  V-JEPA); (2) apply variance/covariance to the ONLINE context embedding only; (3) keep
  VICReg light (SIM 25, VAR 0.5, COV 0.04, gamma 0.5) so it is a guard rail, not a push.
- Fixed loss numbers (README before/after table): total falls ~12.8 to 6.0; MSE falls
  ~0.24 to 0.23; embedding std steady near 0.37 (short-toy) / rises to healthy on the
  full run.

### Bug 4 (the big honesty bug): WINDOW LEAKAGE inflated the headline
- Not a crash, a measurement bug. The first eval classified per WINDOW with a stratified
  split. The windows are overlapping crops of only a few sequences (mean 7 windows/seq).
  A per-window split puts windows from the SAME sequence in both train and test, so the
  classifier can match a test window to a near-duplicate training window. That is window
  leakage and it inflates accuracy.
- Measured directly on gavd iteration-1 real run (42 surviving sequences, 296 clips):
  - per-CLIP stratified split (LEAKY): linear 0.880 +/- 0.026, MLP 0.915, RF 0.881.
  - per-SEQUENCE GroupShuffleSplit (honest): linear 0.494 +/- 0.172.
  - Inflation: about 39 accuracy points.
- The baseline [4] is per SEQUENCE, so per sequence is the ONLY comparable unit.

## THE FOUR gavd2 CORRECTIONS (the controlled comparison)

1. Exact-68 lock. Resolve the labelled set to the EXACT 68 exp5 sequences by sequence
   id, with a three-tier resolver (unpickle the 82-feature file and read each object's
   id/label/ORDER; else glob the 5 curated class folders; else a checked-in constant).
   A REAL locked run FAIL-STOPS if all three fail (no silent heuristic fallback).
   Spelling detail: the full tree names the folder "cerebral palsy" (with a space) but
   the curated tree uses "cerebralpalsy"; the label is canonicalized to "cerebralpalsy"
   everywhere and a separate map recovers the on-disk spelling for file reads, so the
   class is never silently dropped.
2. Per-sequence evaluation. Mean-pool a sequence's window embeddings into ONE vector per
   sequence (the baseline's unit), then split BY sequence. Also reproduce exp5's exact
   seed-42 47/21 split (from the pickle's native feature-list ORDER) for a like-for-like
   point. The leaky per-clip number is kept ONLY as a labelled diagnostic.
3. Video-level leakage control. Exclude from the unlabeled pretraining bank every window
   whose source video also backs a held-out labelled sequence. So the encoder cannot
   have pretrained on a near-identical frame from a held-out clip's own video.
4. Matched probe + artifact fingerprinting. A Random Forest matched to the exp5 family
   (100 trees, max_depth 5, class_weight balanced, seed 42), plus a canonical_id_hash
   stamped on every artifact so a stale mix is caught.

## gavd2 REAL run funnel and numbers (verified, 68/68 coverage)

- canonical_id_hash = 06adde2b13f8. Coverage chased to 68 of 68.
- Two coverage fixes to reach 68/68 (both root-caused, not fudged):
  (a) One cerebral-palsy sequence (id cljas5esv00fn3n6lewd5xqdl): whole-frame MediaPipe
      found 0 of 40 poses because the video decoded at 640x360 while the recorded GAVD
      bounding box was in 1280x720, so the walker was a tiny far-left region. A
      resolution-scaled bbox crop (scale bbox by decoded/original) found 40 of 40.
  (b) One stroke sequence (id cljr5hwxc000f3n6lof5w9tyt): only 15 usable frames. nb03's
      MIN_LEN was 16, which dropped it. Lowered MIN_LEN 16 -> 12 (exp5 has no such floor
      and pads short clips), which admits the 15-frame sequence padded to T=32.
- Corpus (unlabeled pretraining bank, nb03 windowing, T=32, stride 16): about 1,974
  clips (pipeline.md says "pretrained on 1,974 unlabelled clips"). Video-leakage
  exclusion dropped 0 windows on this download set.
- Labeled holdout: 864 windowed clips from all 68 sequences (0 missing).

### RQ1: the controlled comparison (HEADLINE = per-sequence, 20x 70/30 splits on 68 seqs)
- Linear (logistic) probe:      0.486 +/- 0.102 (macro-F1 0.433)
- MLP probe:                    0.626 +/- 0.083 (macro-F1 0.576)
- Random Forest (exp5-family):  0.579 +/- 0.114 (macro-F1 0.544)
- exp5 EXACT split (seed-42 47/21, 21/21 test seqs available): RF = 0.619
- vs baseline 0.762, chance 0.20.
- Per-CLIP DIAGNOSTIC (LEAKY, labelled as such): linear 0.866, MLP 0.920, RF 0.883.
  The ~30-40 point gap between per-clip and per-sequence IS the window-leak inflation.
- Reading: the frozen encoder learns real, transferable gait structure (more than
  doubles chance on unseen sequences, MLP reaches 0.63) but the tuned 82-feature Random
  Forest stays ahead on this small labelled set. The binding constraint is sample size,
  not representation quality. High variance because only ~20 sequences per test fold.

### RQ2: label efficiency (per-sequence linear probe)
- 25% of training sequences: 0.393
- 50%: 0.417
- 75%: 0.457
- 100%: 0.486
- Reading: degrades gracefully, stays about double chance even at a quarter of labels.
  The SHAPE is the payoff of pretraining, not the absolute heights.

### RQ3: clinical structure (Ridge probe from frozen latent to a scalar, R-squared)
- step_amplitude:  R-squared = 0.682 (strongly encoded, with NO labels at all)
- asymmetry_index: R-squared = 0.081 (weakly encoded)
- These two scalars are computed straight from a (T,33,3) clip (see clip_scalars below).
  Because they are LINEARLY decodable, a positive R-squared is meaningful.

### RQ4: VICReg ablation (toggle var/cov in a faithful mini training loop)
- Final embedding std WITH VICReg:    0.904
- Final embedding std WITHOUT VICReg:  0.743
- Reading: variance+covariance do real anti-collapse work ON TOP of the EMA target.
  Do NOT claim OFF collapses to zero; it does not on this data.

### Per-class confusion (aggregated over per-clip splits, rows=true; SHARES the leak caveat)
                 normal park  stroke CP    myop
  normal          0.91  0.01  0.05  0.01  0.03
  parkinsons      0.00  0.91  0.03  0.00  0.05
  stroke          0.07  0.03  0.86  0.01  0.03
  cerebral palsy  0.02  0.00  0.01  0.78  0.19
  myopathic       0.03  0.08  0.02  0.00  0.87
- normal/parkinsons/stroke/myopathic clean (0.86-0.91 recall); cerebral palsy weakest
  (0.78), most often confused with myopathic (0.19), which makes clinical sense (both
  alter load-bearing, can look hypotonic). CP has fewest sequences (15 labelled, and
  only 4 survived in iteration 1), so its estimates are the least reliable.

## PENNY'S NEUROSCIENCE MAPPING (gait/neuroscience/) and WHERE it enters the pipeline

Penny Inouye leads the neuroscience grounding. She graded a big feature list per
condition: which features matter (Priority H/M/L/NA), the neurological reason, a
"what is significant" numeric threshold, and a citation. Files in gait/neuroscience/:
- pd-features.csv (Parkinson's) FILLED IN.
- stroke-features.csv FILLED IN.
- cerebral-palsy-features.csv TEMPLATE (blank, gradings arrive early August 2026).
- myopathic-features.csv TEMPLATE (blank, arrives early August 2026).
- gait-analysis-neuroscience.pdf (the source reference).

Her key graded rationales (verbatim gist, with her thresholds and sources):
- Parkinson's, hip_asymmetry (Priority M): "Parkinson's often begins affecting one side
  first, leading to postural asymmetry. In this study, 16/20 participants (80%) had
  asymmetry on hip balance control." Source PMC4102504.
- Parkinson's, ankle_asymmetry (M): 15/20 (75%) had ankle asymmetry. PMC4102504.
- Parkinson's, stride_length_m (Priority H): "A key symptom of PD is the shuffling gait,
  which would cause a greatly reduced stride length." (< ~0.9 m suspicious.)
- Parkinson's, left/right_hip_range, knee_range, ankle_range (M): reduced range of
  motion in PD. Source PMC8699192.
- Stroke, knee_asymmetry (Priority H): "One symptom of a stroke is a stiff-knee gait ...
  one side of the body has been weakened ... a difference of 17 degrees or greater in
  knee flexion can signal a stroke. Stiff-knee gait affects 25%-75% of individuals with
  post-stroke gait impairment." Source sciencedirect S0268003324001839.
- Stroke, hip_asymmetry / ankle_asymmetry (M/H): a stroke hits ONE side of the brain, so
  one side of the body is impaired, producing asymmetry. Source pubmed 32521470.
- Stroke, walking_speed_ms (H): drops to ~0.50 m/s or slower.

THE PIPELINE LINK (this is the crux of Penny's section, write it explicitly):
Her two dominant clinical axes are (1) LEFT-vs-RIGHT ASYMMETRY (stroke and PD begin
one-sided) and (2) REDUCED RANGE / SHORT STRIDE (PD shuffling; also myopathic). These
two axes appear at THREE concrete places in our code:

1. Masking (nb04, make_block_mask Style A "limb over time"). The mask hides one WHOLE
   limb (left_leg = [23,25,27,29,31] or right_leg = [24,26,28,30,32]) across a window of
   frames. To fill it in, the encoder must reconstruct one leg's motion from the OTHER
   leg and the torso. That is exactly the left-vs-right comparison Penny says defines
   stroke asymmetry. We are not telling the model "compare the legs"; the masking makes
   comparing the legs the only way to solve the task. So Penny's asymmetry rationale is
   the REASON Style A masking (whole-limb, not scattered joints) is the right choice.

2. The clinical probes (nb05, clip_scalars, RQ3). Two scalars computed straight from the
   (T,33,3) clip:
     asymmetry_index = |left_leg_swing - right_leg_swing| / (sum + eps), using
                       left ankle (joint 27) x-range vs right ankle (joint 28) x-range.
                       This is a direct, transparent stand-in for Penny's stroke
                       knee_asymmetry / ankle_asymmetry and PD hip/ankle_asymmetry.
     step_amplitude  = 0.5 * (left ankle swing range + right ankle swing range).
                       This is the reduced-range / short-stride axis: PD shuffling and
                       myopathic short steps vs a full stride. It stands in for Penny's
                       PD stride_length_m and the range-of-motion features.
   RQ3 asks whether the frozen latent LINEARLY encodes these two axes with no labels.
   Result: step_amplitude R-squared 0.682 (the model captured the stride/range axis on
   its own), asymmetry_index R-squared 0.081 (the asymmetry axis is present but faint).
   On the real 82-feature path you would read Penny's documented H-priority scalars
   directly; these two are transparent code stand-ins that a linear probe can hit.

3. The confusion matrix reading (nb05). Penny's clinical reasoning explains WHY the
   errors land where they do: cerebral palsy is confused with myopathic (0.19) because
   both alter load-bearing and can look hypotonic. The neuroscience is how we sanity-
   check that the model's mistakes are clinically sensible, not random.

Also note the STATUS honestly: PD and stroke gradings are done; CP and myopathic are
templates that arrive early August 2026, so their per-feature clinical probes are future
work. The current RQ3 uses the two transparent stand-in scalars above, chosen to line up
with the axes Penny already graded as high priority for the conditions we do have.

## Authorship and titles
Alex Mui, Penny Inouye, Theodore Mui (equal co-authors). Penny leads neuroscience
grounding. Alex and Theodore lead the ML / pose pipeline. Phil Mui is Research Advisor.
Paper flavor name: Gait-JEPA (a skeleton-JEPA over pose sequences, not pixels).

## IEEE references (use these exact entries)
[1] M. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
    Predictive Architecture," in Proc. IEEE/CVF CVPR, 2023.
[2] A. Bardes et al., "V-JEPA: Latent Video Prediction for Self-Supervised Video
    Representation Learning," Meta AI, 2024.
[3] A. Bardes, J. Ponce, and Y. LeCun, "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning," in Proc. ICLR, 2022.
[4] Ranjan et al., "Gait Abnormality in Video Dataset (GAVD)," 2025.
[5] V. Bazarevsky et al., "BlazePose: On-device Real-time Body Pose Tracking," 2020.
[6] Y. LeCun, "A Path Towards Autonomous Machine Intelligence," 2022.
Optional supporting (Penny's neuroscience grounding, cite where used):
[7] PD hip/ankle asymmetry 16/20 and 15/20, PMC4102504.
[8] PD reduced hip ROM, PMC8699192.
[9] Post-stroke stiff-knee gait, ScienceDirect S0268003324001839.
[10] Post-stroke hip asymmetry, PubMed 32521470.

## Existing gavd2 SVGs we can REUSE (in ../images/, reference as ../../images/NAME.svg
   from a doc in docs/learning/, OR copy into docs/learning/images/):
pipeline-overview, four-pieces, exact-68-lock, per-sequence-pooling, window-leakage,
cooccurring-video-exclusion, controlled-comparison, label-efficiency, masking-styles,
confusion-matrix, collapse-vicreg, loss-fix-curves, pos-embed-fix, dataset-funnel,
neuroscience-axes, real-training-curve, clip-vs-sequence, results-scorecard,
probe-vs-baseline, walk-skeleton.gif.

## SVG house style (match exactly)
- White background rect first. viewBox based, sans-serif.
- Palette: title #0f172a bold; subtitle/body #475569; arrows/lines #334155;
  green accent #15803d on #f0fdf4; blue accent #1d4ed8 on #eff6ff; red/wrong #b91c1c on
  #fee2e2; amber note #d97706 on #fff7ed; muted grey #cbd5e1 / #64748b.
- Rounded rect boxes (rx 6-8). text-anchor="middle" for centered labels. Arrow marker
  id="arrow" with #334155 fill. NO overlapping text, NO lines crossing labels. Keep it
  uncluttered: few words per box, generous spacing. Verify by rendering with rsvg-convert.
