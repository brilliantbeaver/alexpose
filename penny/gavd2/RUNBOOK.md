# Iteration-2 REAL run runbook: controlled and enhanced lanes

The build ships smoke-verified notebooks; this runbook is the real (SMOKE_TEST=False)
run that produces the iteration-2 numbers. It needs the full pose stack (OpenCV,
MediaPipe, yt-dlp) and the Alexpose checkout. Notebooks 00 through 05 produce the
controlled baseline. After 00 through 03, Notebooks 06/07 and 08/09 are two independent
enhanced experiment lanes.

## 0. Prerequisites

- `.env` in this folder points `ALEXPOSE_REPO`, `GAVD_DATA_DIR`, `YOUTUBE_CACHE_DIR`,
  `EXP4_DATA_DIR`, and `EXP5_FEATURES_PKL` at your machine (a filled `.env` is provided).
  Leave `GAVD_CACHE_DIR` blank so artifacts land in `gavd2/cache`.
- The real deps: `uv sync --extra real` (or use a venv that already has cv2, mediapipe,
  yt-dlp, pandas, torch, scikit-learn).

## 1. Flip each notebook to REAL

In each notebook you intend to run, set the first CONFIG key `SMOKE_TEST` to `False`. Leave
`EXPLORATORY_FIRST_N` at `False` so the run is locked to the exact exp5 68 and writes the
un-suffixed (locked) artifacts.

## 2. Run in order

```bash
cd gait/skeleton-jepa/gavd2
GAVD2_PYTHON=/absolute/path/to/python
for nb in 00-scan-all-gavd-csvs 01-bulk-download-youtube 02-batch-extract-skeletons \
          03-build-pretraining-corpus 04-pretrain-jepa-at-scale 05-frozen-probe-full-eval; do
  "$GAVD2_PYTHON" -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=7200 \
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

## 2A. Run the enhanced MLP-predictor lane

Notebooks 06 and 07 reuse the locked data from 00 through 03. Use a dedicated cache so
Notebook 08 cannot overwrite the checkpoint later:

```bash
export GAVD_CACHE_DIR=/absolute/path/to/gavd2-cache-enhanced-mlp
```

Run 00 through 03 with that cache, or copy the complete locked upstream artifacts into
it. Then execute:

```bash
for nb in 06-pretrain-enhanced-jepa 07-enhanced-probe-full-eval; do
  "$GAVD2_PYTHON" -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=7200 --output "$nb.ipynb" "$nb.ipynb"
done
```

Expect Notebook 06 to report 801,920 encoder parameters, 65,920 predictor parameters,
and 4,000 updates. Expect Notebook 07 to load the checkpoint strictly and report 68
sequence embeddings.

## 2B. Run the enhanced transformer-predictor lane

Use another cache directory:

```bash
export GAVD_CACHE_DIR=/absolute/path/to/gavd2-cache-enhanced-transformer
```

Run 00 through 03 with that cache, or copy the same complete locked upstream artifacts
into it. Then execute:

```bash
for nb in 08-pretrain-enhanced-predictor 09-enhanced-predictor-full-eval; do
  "$GAVD2_PYTHON" -m jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.timeout=7200 --output "$nb.ipynb" "$nb.ipynb"
done
```

Before accepting the run, verify:

- Notebook 08 prints `Loaded real corpus`, not a synthetic fallback.
- The corpus has shape `(1974, 32, 33, 3)` and hash `06adde2b13f8`.
- The encoder has 801,920 parameters and the predictor has 297,984.
- Notebook 09 prints `Loaded enhanced encoder` and 68 unique sequences.
- The exact EXP5 split includes 21 of 21 test sequences.
- Per-clip scores remain labelled as leakage diagnostics.

Notebooks 06 and 08 currently share a checkpoint model ID and filename. A strict state
dictionary load cannot identify which discarded training predictor produced those
encoder weights. Cache isolation is therefore required for a reproducible comparison.

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

Once an evaluation notebook has run, update the documentation for its own lane. Do not
replace historical 04/05 or 06/07 numbers with 08/09 numbers. Update the comparison table
in `docs/enhanced-predictor.md`, then re-render `docs/paper.html` if `docs/paper.md`
changed:

```bash
pandoc docs/paper.md -f gfm -t html5 --standalone --embed-resources \
  --resource-path=docs \
  --metadata title="Gait-JEPA iteration 2" -o docs/paper.html

pandoc docs/learning/learning-journey.md -f gfm -t html5 --standalone \
  --toc --embed-resources --resource-path=docs/learning \
  --css=paper.css -o docs/learning/learning-journey.html
```

The resource paths are required because both Markdown files use image paths relative to
their own directories. A successful render prints no missing-resource warnings.

## 5. Commit with outputs cleared

Clear notebook outputs before committing:

```bash
"$GAVD2_PYTHON" -m jupyter nbconvert --clear-output --inplace ./*.ipynb
```
