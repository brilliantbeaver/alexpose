# Pipeline reference: the six gavd2 notebooks

One page per notebook: what it reads, what it writes, the key config, and the coverage it
reports. All artifacts live under `CONFIG["CACHE_DIR"]` (default `gavd2/cache`) and carry
a `{ns}` suffix (`""` for a locked exp5-exact run, `_firstN` for the exploratory run) plus
a `canonical_id_hash` stamp so a stale-artifact mix is caught. Every notebook has
`SMOKE_TEST` as its first CONFIG key, and runs top to bottom on CPU in seconds with
synthetic data and no network when `SMOKE_TEST=True`. The real numbers quoted below are
from the full real run (all six notebooks with `SMOKE_TEST=False`) on the exact exp5 68
sequences, chased to full 68-of-68 coverage.

## 00 - scan all GAVD CSVs (the exact-68 lock)

- **Reads:** the GAVD CSV tree (`GAVD_DATA_DIR`), the curated exp5 tree (`EXP4_DATA_DIR`)
  and/or the exp5 pickle (`EXP5_FEATURES_PKL`) via the three-tier resolver.
- **Writes:** `manifest{ns}.csv` (one row per sequence, canonical `condition`,
  `is_labeled`, `in_exp5_curated`, `video_id`), `labeled_manifest{ns}.csv` (the 68 rows),
  `exp5_split{ns}.csv` (exp5's exact seed-42 47/21 partition), `canonical_id_hash{ns}.txt`.
- **Key config:** `SMOKE_TEST`, `EXPLORATORY_FIRST_N` (default False), `LABELED_CLASSES`
  (canonical spellings), `BASELINE_ACC=0.76`.
- **Coverage:** prints the labelled-id source tier and, in a real locked run, asserts the
  68 resolve with the exact per-class counts (normal 12, parkinsons 9, stroke 12,
  cerebralpalsy 15, myopathic 20).

## 01 - bulk download YouTube

- **Reads:** `manifest{ns}.csv`.
- **Writes:** `download_report{ns}.csv` (per unique video: cached, ok, valid, size,
  condition, is_labeled).
- **Key config:** `SMOKE_TEST`, `MAX_VIDEOS`, `QUALITY`, ffprobe validation kept verbatim.
- **Coverage:** dedups to unique videos, sorts labelled-backing videos first so a capped
  run fetches the 68-subset first, and prints "labelled videos ready: K of target".
  Reuses the shared external video cache, so it re-downloads only genuinely missing files.

## 02 - batch extract skeletons

- **Reads:** `manifest{ns}.csv`, `download_report{ns}.csv`, the cached videos, the GAVD
  CSVs. Ambient `SequenceKeypointExtractor` (MediaPipe BlazePose, whole frame,
  `min_keypoints=25`, `filter_empty=True`) is the primary path; a raw yt-dlp/OpenCV/
  MediaPipe inline fallback runs whole-frame too when `INLINE_USE_BBOX_CROP=False`.
- **Writes:** `skeletons_<canon>.npz` per condition (canonical condition field),
  `extraction_report{ns}.csv` (per sequence: n_frames, ok, note, is_labeled, fail_reason).
- **Key config:** `SMOKE_TEST`, `MIN_KEYPOINTS=25`, `INLINE_USE_BBOX_CROP=False`.
- **Coverage:** maps the canonical label back to the on-disk folder (`cerebralpalsy` ->
  `cerebral palsy`) so cerebral-palsy extractions do not fail; for labelled sequences it
  records a per-sequence failure reason (no video / empty extraction / <8 frames) and
  prints coverage of the 68 with the missing ids. On the real run the whole-frame pass
  extracted 67 of the 68; the one cerebral-palsy miss was recovered with a resolution-
  scaled bbox crop (its video downloaded at 640x360 while the recorded bbox was in
  1280x720, so the walker was a tiny far-left region whole-frame detection missed),
  bringing extraction to 68 of 68.

## 03 - build pretraining corpus

- **Reads:** `skeletons_*.npz`, `manifest{ns}.csv`, `labeled_manifest{ns}.csv`.
- **Writes:** `corpus{ns}.npz` (unlabelled window bank, video-leakage-cleaned) and
  `labeled_holdout{ns}.npz` (windowed clips + `seq_ids`, `labels`, `classes`,
  `expected_seq_ids`, `missing_seq_ids`, `canonical_id_hash`).
- **Key config:** `SMOKE_TEST`, `T=32`, `STRIDE=16`, `MIN_LEN=12`, `LABELED_CLASSES`
  (canonical order).
- **Coverage:** canonicalizes the cache-derived condition (so cerebral palsy survives),
  intersects the labelled set with the exact 68 in a real run, and EXCLUDES from the
  unlabelled bank every window whose source video also backs a held-out labelled sequence
  (video-level leakage control), printing the excluded-window count and the realized
  per-class holdout counts. `MIN_LEN` is 12 (not 16) so a genuine 15-frame stroke sequence
  is padded into a single window rather than dropped, which keeps the probe set at the full
  68. On the real run the holdout is 864 clips from all 68 sequences (0 missing), and the
  video-leakage exclusion dropped 0 windows on this download set.

## 04 - pretrain the JEPA at scale

- **Reads:** `corpus{ns}.npz`; verifies its `canonical_id_hash`. Loads the holdout's
  `expected`/`missing` for a provenance print.
- **Writes:** `jepa_encoder_gavd{ns}.pt` (state_dict + config, config stamped with the
  hash).
- **Key config:** `SMOKE_TEST`, `T=32`, `N_JOINTS=33`, `EMBED_DIM=64`, `EMA_M=0.996`,
  VICReg `SIM=25.0 VAR=0.5 COV=0.04 VAR_TARGET=0.5`, `MASK_RATIO=0.4`, 400 real steps.
- **Preserved:** ContextEncoder with time+joint positional embeddings; the corrected loss
  (L2 on the LayerNorm-normalized EMA target plus light online-only VICReg). On the real
  run it pretrained on 1,974 unlabelled clips for 400 steps with no collapse (final
  embedding standard deviation 0.82).

## 05 - frozen probe, full eval (the controlled comparison)

- **Reads:** `jepa_encoder_gavd{ns}.pt`, `labeled_holdout{ns}.npz`, `exp5_split{ns}.csv`.
- **Writes:** figures and printed metrics (no cache artifact).
- **Key config:** `SMOKE_TEST`, `CLASSES`/`CLASS_COUNTS` (canonical), `BASELINE_ACC=0.76`,
  `TEST_FRAC=0.30`, `N_SPLITS=20`, `SEED=42`.
- **Evaluation:** mean-pools window embeddings to one vector per sequence, then reports
  the honest per-sequence band (linear, MLP, and a Random Forest matched to exp5:
  100 trees, max_depth 5, class_weight balanced) as the headline, plus a like-for-like
  point on exp5's exact seed-42 split (restricted to available ids), beside 0.76 and the
  0.20 chance line. Per sequence is the honest, comparable unit, because the baseline was
  also scored per sequence; the per-clip number is kept only as a labelled leaky diagnostic
  that measures the window-leak, never as a result.
  On the real run (68 of 68 sequences, 20 splits): per-sequence linear 0.486 +/- 0.102,
  MLP 0.626 +/- 0.083, Random Forest 0.579 +/- 0.114; the exp5 exact-split matched Random
  Forest scores 0.619 (21 of 21 test sequences) beside the 0.762 baseline. The leaky
  per-clip diagnostic reads about 0.87 to 0.92 (linear 0.866, MLP 0.920, RF 0.883); the
  30-to-40 point gap above the per-sequence band is the window-leak, not learning, so it is
  reported only to size the leak. RQ3 recovers step amplitude at R-squared 0.682 and asymmetry at 0.081; RQ4
  shows the embedding spread higher with VICReg (0.904) than without (0.743).

## Related pages

- [[glossary]] - the ubiquitous language.
- [learning/learning-journey.md](learning/learning-journey.md) - the plain-language learning story of the whole journey.
- ADRs under `adr/` - the decisions behind this pipeline.
