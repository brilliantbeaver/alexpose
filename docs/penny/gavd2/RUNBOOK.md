# Iteration-2 REAL run runbook

The build ships smoke-verified notebooks; this runbook is the real (SMOKE_TEST=False)
run that produces the iteration-2 numbers. It needs the full pose stack (OpenCV,
MediaPipe, yt-dlp) and the alexpose checkout, so run it in an environment that has them,
for example the alexpose venv at `/Users/pmui/dev/alexpose/.venv`.

## 0. Prerequisites

- `.env` in this folder points `ALEXPOSE_REPO`, `GAVD_DATA_DIR`, `YOUTUBE_CACHE_DIR`,
  `EXP4_DATA_DIR`, and `EXP5_FEATURES_PKL` at your machine (a filled `.env` is provided).
  Leave `GAVD_CACHE_DIR` blank so artifacts land in `gavd2/cache`.
- The real deps: `uv sync --extra real` (or use a venv that already has cv2, mediapipe,
  yt-dlp, pandas, torch, scikit-learn).

## 1. Flip each notebook to REAL

In each of `00` through `05`, set the first CONFIG key `SMOKE_TEST` to `False`. Leave
`EXPLORATORY_FIRST_N` at `False` so the run is locked to the exact exp5 68 and writes the
un-suffixed (locked) artifacts.

## 2. Run in order

```bash
cd gait/skeleton-jepa/gavd2
PY=/Users/pmui/dev/alexpose/.venv/bin/python
for nb in 00-scan-all-gavd-csvs 01-bulk-download-youtube 02-batch-extract-skeletons \
          03-build-pretraining-corpus 04-pretrain-jepa-at-scale 05-frozen-probe-full-eval; do
  $PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
      --output "$nb.ipynb" "$nb.ipynb"
done
```

Expected checkpoints:

- **00** prints the labelled-id source tier (expect `tier2:exp4-glob` locally), asserts
  the 68 resolve with per-class counts normal 12, parkinsons 9, stroke 12, cerebralpalsy
  15, myopathic 20, and writes `manifest.csv`, `labeled_manifest.csv` (68 rows),
  `exp5_split.csv` (47 train / 21 test), `canonical_id_hash.txt`.
- **01** fetches the 12 unique labelled-backing videos first and prints "labelled videos
  ready: K of 12". Most are already cached in the shared `YOUTUBE_CACHE_DIR`.
- **02** prints "Labelled-68 coverage: N of 68 extracted OK" and lists any missing ids
  with a reason (no video / empty extraction / <8 frames).
- **03** prints the excluded-window count (video-level leakage) and the realized
  per-class holdout counts.
- **04** prints "expected 68 vs survived K vs missing 68-K" and stamps the encoder.
- **05** prints "evaluated on N of 68 sequences", the per-sequence headline band, the
  matched Random Forest, and the exp5 exact-split like-for-like point beside 0.76.

## 3. Chase 68/68 coverage (the locked decision)

After the first pass, read `cache/extraction_report.csv` and filter to
`is_labeled == True and ok == False`. For each missing labelled sequence:

- `fail_reason == "no video"`: the video did not download. Re-attempt with
  `nb01` (its `YT_DLP_COOKIES_FROM_BROWSER` option can help age or region gated clips), or
  fetch it directly with yt-dlp into `YOUTUBE_CACHE_DIR`. Some links may be permanently
  dead or region-blocked; record those as unrecoverable.
- `fail_reason == "empty extraction"`: MediaPipe found nothing. Try
  `INLINE_USE_BBOX_CROP = True` in `nb02` for that class, or inspect the video.
- `fail_reason == "<8 frames"`: too few usable frames; usually unrecoverable, record it.

Re-run `02 -> 03 -> 04 -> 05` after recovering any videos. Report the final realized
"N of 68"; if N is below 68, `nb05` reports the exp5 exact-split point restricted to the
available ids and the headline band on the N present.

## 4. Refresh the numbers in the docs and slides

Once `05` has run, update `docs/paper.md`, `docs/pipeline.md`, and the two decks under
`slides/`, replacing every "pending iteration-2 refresh" placeholder with the realized
numbers, and re-render `docs/paper.html`:

```bash
pandoc docs/paper.md -f gfm -t html5 --standalone --embed-resources \
  --metadata title="Gait-JEPA iteration 2" -o docs/paper.html
```

## 5. Commit with outputs cleared

Clear notebook outputs before committing:

```bash
$PY -m jupyter nbconvert --clear-output --inplace gait/skeleton-jepa/gavd2/*.ipynb
```
