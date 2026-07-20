# Real-data results ground truth (Gait-JEPA on GAVD)

Internal spec for the paper, tutorial, and slides. NOT user-facing. Every number
below was produced by executing the notebooks in REAL mode (SMOKE_TEST=False) against
the cache in `gavd_cache/`, verified 2026-07-13. NO em-dashes / en-dashes anywhere.

## What actually ran (provenance)

- The `gavd_cache/` skeleton `.npz` files are REAL MediaPipe BLAZEPOSE_33 extractions
  (structured non-zero z channel, std ~0.13-0.68 per condition; coordinates are real
  pose, not the synthetic fallback). They were produced by a run on the full alexpose
  stack (cv2 + mediapipe present).
- The `extraction_report.csv` currently in cache reads ok=False for all 339 rows with
  note "ambient path failed (No module named 'loguru'); ... inline failed (No module
  named 'cv2')". That report is from a LATER re-scan on a machine WITHOUT cv2/mediapipe;
  it did not overwrite the real cached skeletons. So: the skeletons are real; that one
  CSV is a stale artifact. Do NOT quote the extraction_report ok-count as a result.
- corpus.npz and labeled_holdout.npz were rebuilt (nb03) from the real skeletons.
- jepa_encoder_gavd.pt was trained by nb04 on the real corpus (400 steps, seed 42, CPU).
- nb04 and nb05 both re-execute top-to-bottom in REAL mode with ZERO errors.

## Dataset funnel (exact, real)

- GAVD total: 374 sequence CSVs across 11 condition folders, 91,624 frame-rows,
  mean 245 frames/sequence.
- Per-condition CSV counts: abnormal 190, antalgic 4, cerebral palsy 16, exercise 24,
  inebriated 2, myopathic 47, normal 12, parkinsons 9, prosthetic 3, stroke 12, style 55.
- Unique YouTube videos referenced: 69 (many sequences share a video).
- Videos downloaded OK: 67 of 69 (2 dead / blocked links).
- Sequences with a real skeleton successfully extracted: 227 (per condition:
  abnormal 118, antalgic 2, cerebral palsy 4, exercise 18, inebriated 1, myopathic 33,
  normal 10, parkinsons 9, prosthetic 3, stroke 8, style 21).
- Manifest marks 68 sequences as the labeled 5-class subset (target: normal 12,
  parkinsons 9, stroke 12, cerebral palsy 15, myopathic 20).
- Of those 68, only 42 survived download + extraction and reached the labeled holdout:
  normal 9, parkinsons 9, stroke 8, cerebral palsy 4, myopathic 12 = 42 sequences.
- Unlabeled pretraining corpus (nb03 windowing, T=32, overlap): 1,571 clips of shape
  (32, 33, 3). This is the bank nb04 pretrains on (no labels).
- Labeled holdout: 296 windowed clips from the 42 labeled sequences (per class WINDOWS:
  normal 86, parkinsons 47, stroke 70, cerebral palsy 24, myopathic 69). Mean 7.0
  windows per sequence (range 1 to 60).

## Encoder / method (exact, as trained)

- ContextEncoder: input_proj Linear(3->64), learned time_embed (32,64) + joint_embed
  (33,64) init std 0.1, added as pos[t,j]=time_embed[t]+joint_embed[j] over row-major
  (t,j) tokens (n=t*33+j); 2-layer nn.TransformerEncoder, 4 heads, dim_feedforward=128,
  gelu, dropout 0, batch_first. Total encoder params: 71,360. EMBED_DIM D=64.
- Target encoder: EMA copy, m=0.996, stop-gradient.
- Predictor: shallow MLP (Linear-GELU-Linear, hidden 2*D).
- Loss (corrected): L2 between prediction and LayerNorm-normalized EMA target, plus
  LIGHT VICReg variance+covariance on the ONLINE context embedding only. Weights:
  VICREG_SIM 25.0, VICREG_VAR 0.5, VICREG_COV 0.04, VAR_TARGET(gamma) 0.5, EPS 1e-4.
- Masking: spatiotemporal block, MASK_RATIO 0.4 (batch reported hidden fraction 0.25),
  limb-over-time (Style A) + time-window (Style B).
- Training: 400 steps, BATCH 16, LR 1e-3, Adam, CPU, seed 42.

## nb04 real training trajectory (400 steps, verified)

step   total    sim(MSE)  var     cov     emb_std
0      32.021   1.2776    0.1334  0.3418  0.377
50      9.118   0.3603    0.1477  0.8891  0.367
100     7.774   0.3072    0.0413  1.8407  0.539
150     6.589   0.2595    0.0124  2.3978  0.625
200     6.293   0.2470    0.0043  2.8811  0.659
250     6.360   0.2520    0.0007  1.4945  0.658
300     6.997   0.2762    0.0004  2.2875  0.726
350     7.424   0.2931    0.0002  2.4437  0.751
399     5.508   0.2169    0.0000  2.1652  0.763

- Total loss falls 32.0 -> 5.5 overall (a fast drop to ~9 by step 50, then a gentle
  decline with minor fluctuations to 5.5; NOT the runaway upward drift the old loss
  showed). Prediction MSE (sim) falls 1.28 -> 0.22. Embedding std RISES 0.38 -> 0.76,
  so no collapse (collapse would drive std toward 0). Final embedding std 0.763; mean
  per-dimension std 0.818, min/max per-dim 0.363 / 1.977 (many dimensions used).
- Interpretation for prose: the corrected loss (LayerNorm target + online-only VICReg +
  light weights) is stable over the full 400-step real run. Contrast with the OLD loss
  which drifted UP after ~50 steps. Do not claim perfectly monotone; say "falls sharply
  then settles, with the embedding spread growing healthily and no collapse."

## nb05 real evaluation results (frozen probe, mean over N_SPLITS=20)

THE HEADLINE NUMBERS AS NOTEBOOK 05 REPORTS THEM (per-CLIP stratified 70/30 splits,
StandardScaler refit per fold, on 296 windowed clips):
- Linear (logistic) probe:      acc 0.880 +/- 0.026, macro-F1 0.874
- MLP probe:                    acc 0.915 +/- 0.022, macro-F1 0.910
- Random Forest on embeddings:  acc 0.881 +/- 0.027, macro-F1 0.879
- Chance = 0.20. Prior baseline (RF on 82 hand features) = 0.76.
- All three probes beat the 0.76 baseline on this per-clip metric.

CRITICAL HONESTY CAVEAT (must be foregrounded in ALL THREE deliverables):
The 296 clips are OVERLAPPING WINDOWS from only 42 sequences (mean 7 windows/seq).
A per-clip stratified split puts windows from the SAME sequence in both train and test,
so the encoder can match a test window to a near-duplicate training window. This is
WINDOW LEAKAGE and it inflates accuracy. Measured directly:
- per-CLIP stratified split (leaky, what nb05 prints): linear acc 0.880 +/- 0.026
- per-SEQUENCE GroupShuffleSplit (leakage-free, groups=seq_ids): linear acc 0.494 +/- 0.172
- Leakage inflation: about 39 accuracy points.
So the rigorous, leakage-free, sequence-level number is ~0.49 +/- 0.17 on 42 sequences,
which is well above the 0.20 chance level but BELOW the 0.76 per-sequence baseline, and
extremely high-variance because there are only 42 sequences (about 13 in each test fold).
FRAMING RULE: report BOTH. Present 0.88-0.92 as "per-clip, with window leakage, the
metric the notebook currently prints"; present 0.49 as "the honest sequence-level result".
The paper's central claim is NOT "we beat 76 percent"; it is "a frozen pose JEPA learns a
gait representation that is strongly above chance at the sequence level and beats the
baseline at the clip level, and the gap between the two exposes window leakage and the
small-sample ceiling as the real obstacles." Never headline the leaky number alone.

RQ2 label efficiency (per-clip splits, linear probe, mean over 20 splits):
- 25% of training labels: 0.746 +/- 0.047
- 50%: 0.820 +/- 0.051
- 75%: 0.864 +/- 0.024
- 100%: 0.880 +/- 0.026
(rises quickly then flattens; same per-clip leakage caveat applies, so describe the
SHAPE as the takeaway, not the absolute heights.)

RQ3 clinical structure (Ridge linear probe from frozen latent to scalar, mean R^2 over 20 splits):
- asymmetry_index: R^2 = 0.154 +/- 0.079  (weakly encoded)
- step_amplitude:  R^2 = 0.719 +/- 0.113  (strongly encoded)
Interpretation: the frozen embedding linearly preserves step amplitude well and
asymmetry only weakly. Honest: step_amplitude is clearly recoverable, asymmetry is
present but faint. (Verified linear-probe ceilings from raw coords, prior smoke work:
asymmetry ~0.70, step_amplitude ~0.84 - so the encoder captures most of the step
amplitude ceiling and little of the asymmetry ceiling.)

RQ4 VICReg ablation (faithful mini nb04 loop on the labeled clips, toggle var/cov):
- Final embedding std WITH VICReg:    0.889
- Final embedding std WITHOUT VICReg: 0.766
- The ON run's spread sits above OFF: variance+covariance do real anti-collapse work on
  top of the EMA target. Do NOT claim OFF collapses to zero (it does not on this data).

Per-class confusion (aggregated over 20 per-clip splits, rows=true, cols=pred, normalized):
                 normal park  stroke CP    myop
  normal          0.91  0.01  0.05  0.01  0.03
  parkinsons      0.00  0.91  0.03  0.00  0.05
  stroke          0.07  0.03  0.86  0.01  0.03
  cerebral palsy  0.02  0.00  0.01  0.78  0.19
  myopathic       0.03  0.08  0.02  0.00  0.87
Reading: normal, parkinsons, stroke, myopathic are cleanly separated (0.86-0.91 recall);
cerebral palsy is the weakest (0.78) and is most often confused with myopathic (0.19),
which makes clinical sense (both alter load-bearing and can look hypotonic). CP also has
the FEWEST sequences (4), so its estimates are the least reliable. This is a per-clip
confusion, so it shares the leakage caveat.

## Citations available (IEEE style)

[1] M. Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
    Predictive Architecture" (I-JEPA), CVPR 2023.
[2] A. Bardes et al., "V-JEPA: Latent Video Prediction for Self-Supervised Video
    Representation Learning," Meta AI, 2024. (LayerNorm-on-target design.)
[3] A. Bardes, J. Ponce, Y. LeCun, "VICReg: Variance-Invariance-Covariance
    Regularization for Self-Supervised Learning," ICLR 2022.
[4] Ranjan et al., "Gait Abnormality in Video Dataset (GAVD)," 2025. (Baseline: RF, 100
    trees, 82 hand features, 70/30 split, best 76 percent test acc, 5 classes.)
[5] V. Bazarevsky et al., "BlazePose: On-device Real-time Body Pose Tracking," 2020.
[6] Y. LeCun, "A Path Towards Autonomous Machine Intelligence," 2022. (JEPA world-model framing.)
Optional supporting (used in RQ3 neuroscience grounding, Penny Inouye's mapping):
    stiff-knee post-stroke gait (sciencedirect S0268003324001839), stroke hip asymmetry
    (pubmed 32521470), PD postural asymmetry 16/20 hip & 15/20 ankle (PMC4102504), PD
    reduced hip ROM (PMC8699192). CP and myopathic gradings arrive early August 2026.

## Authorship
Alex Mui, Penny Inouye, Theodore Mui (equal co-authors). Penny leads neuroscience
grounding (delivered early August 2026); Alex and Theodore lead the ML / pose pipeline.
Phil Mui is Research Advisor. Paper title: "Gait-JEPA". Method flavor: skeleton-JEPA
(pose-sequence JEPA, not pixels).
