# Shared facts for the Skeleton-JEPA FULL-DATASET (gavd/) tutorial series

This is an internal spec that every notebook and every SVG in
`gait/skeleton-jepa/gavd/` must stay consistent with. It is NOT a
user-facing page. Do not link to it from the tutorials.

## What this series is (and how it differs from tutorials/)

`gait/skeleton-jepa/tutorials/` is the CONCEPT series: it walks one
demo clip (`B5hrxKe2nP8`) through the whole idea, step by step, to
teach what a Skeleton-JEPA is.

`gait/skeleton-jepa/gavd/` is the FULL-DATASET series: it scales the
exact same pipeline to EVERY GAVD sequence. It scans all 374 CSVs
across 11 conditions, bulk-downloads the unique YouTube videos,
batch-extracts skeletons over every sequence (mirroring
`alexpose/experiments/exp5/01_extract_features.ipynb`), builds a large
UNLABELED pretraining corpus, pretrains the JEPA on that corpus, and
finally spends the scarce 68 clinical labels on a frozen probe that we
compare against the prior 76 percent Random Forest baseline.

The through-line of the whole series: unlabeled walking video is
cheap and plentiful (all 374 sequences), clinical labels are scarce
and precious (only 68). The full-dataset series is about turning that
big cheap pile into a strong encoder, then spending the 68 labels only
at the very end.

The six notebooks (each `NN-title.ipynb`):

- 00-scan-all-gavd-csvs.ipynb   : walk all 11 condition folders and all 374 CSVs, build a manifest DataFrame (one row per sequence: condition, seq id, video id, url, frame span, num frames, bbox present), visualize the dataset at scale (class balance, sequence-length distribution, unique-video count, the 68-clip labeled subset vs the full unlabeled pool). Writes manifest.csv.
- 01-bulk-download-youtube.ipynb : from the manifest, dedup to the unique YouTube video ids, download each once into the cache (resumable: skip already-cached), record success/failure per video, write download_report.csv. Mirrors alexpose YouTubeHandler with inline yt-dlp fallback.
- 02-batch-extract-skeletons.ipynb : for every sequence whose video downloaded, run MediaPipe BLAZEPOSE_33 over its frames (exp5 pattern: GAVDDataLoader -> SequenceKeypointExtractor.extract_from_sequence), quality-filter, and cache each sequence to a per-condition skeletons_<condition>.npz. Writes an extraction_report.csv.
- 03-build-pretraining-corpus.ipynb : load every cached sequence, pelvis-center + torso-normalize, slice into overlapping fixed-length windows (T frames), stack into one big unlabeled clip bank (N, T, 33, 3). Hold out the 68 labeled clips (5-class subset) for the probe only. Writes corpus.npz and labeled_holdout.npz.
- 04-pretrain-jepa-at-scale.ipynb : train the Skeleton-JEPA (context encoder, EMA target encoder, predictor, VICReg loss) on the full unlabeled corpus with block masking. Monitor loss terms + embedding std to confirm no collapse. Writes jepa_encoder_gavd.pt.
- 05-frozen-probe-full-eval.ipynb : freeze the encoder, embed the 68 labeled clips, train linear + MLP + RF probes on the 70/30 split, plot the label-efficiency curve, fit neuroscience linear probes to H-priority scalars, run the VICReg on/off ablation, and compare everything to the 76 percent RF baseline.

## Cache artifacts passed between notebooks (all under GAVD_CACHE_DIR, default ./cache)

- 00 writes `manifest.csv`            (one row per sequence). 01, 02, 03 read it.
- 01 writes `download_report.csv`     (one row per unique video id: cached?, ok?, size). 02 reads it.
- 02 writes `skeletons_<condition>.npz` per condition + `extraction_report.csv`. 03 reads them.
- 03 writes `corpus.npz` (unlabeled windowed bank) + `labeled_holdout.npz` (68-clip 5-class set). 04 and 05 read them.
- 04 writes `jepa_encoder_gavd.pt`    (encoder weights + config). 05 reads it.

Every notebook that reads an upstream artifact must gracefully fall
back to synthetic data (with a clear printed message) if the artifact
is missing, so any notebook opens and runs on its own.

## Non-negotiable house-style rules (shared with the ViT/VICReg/gait/tutorials series)

- NO em-dashes and NO en-dashes ANYWHERE (prose, code comments, SVG text). ASCII hyphen `-` and commas only.
- Natural, connected prose that flows across paragraphs. Explain, then show. No bullet-dump lessons.
- Every notebook is SELF-CONTAINED for dependencies and runnable in Google Colab.
  - Cell 0 = markdown "Open In Colab" badge:
    `[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/philmui/worldmodels/blob/main/gait/skeleton-jepa/gavd/NN-title.ipynb)`
  - Cell 1 = markdown intro (what this notebook does, how it builds on the previous one).
  - Cell 2 (after a short "Run locally or in Colab" markdown) = code "Colab setup": use `importlib.util.find_spec` to pip-install ONLY missing deps, then load .env with python-dotenv, then add ALEXPOSE_REPO to sys.path. Map package->import name (scikit-learn->sklearn, opencv-python->cv2, mediapipe->mediapipe, yt-dlp->yt_dlp, python-dotenv->dotenv). Core deps (numpy, pandas, matplotlib, scikit-learn, torch, python-dotenv, tqdm) install always; REAL-mode deps (opencv-python, mediapipe, yt-dlp) install only when CONFIG has SMOKE_TEST False.
  - Cell 3 = code CONFIG dict with `SMOKE_TEST` as the FIRST key (default True).
- CONFIG dict pattern: `SMOKE_TEST` first, default True. SMOKE_TEST=True => tiny synthetic data, no network, no MediaPipe, runs top-to-bottom on CPU in seconds. SMOKE_TEST=False (REAL) => scan/download/extract the real full dataset. In REAL mode the default caps are MAX_VIDEOS=None and MAX_SEQ_PER_CONDITION=None (process EVERYTHING), but the CONFIG documents that you can set small integer caps for a quick real trial.
- Use the EXACT Colab-setup `_ensure`/`_import_name` idiom from tutorials/ (see below). Do NOT invent a different installer name.
- CREDENTIALS / KEYS / LOCAL PATHS: always load with `load_dotenv(find_dotenv())` from `python-dotenv`. No other .env-loading idiom (no hardcoded paths, no manual file parsing). Any secret or key a notebook ever needs is read from the environment AFTER this call via `os.getenv(...)`.
- Commit notebooks with OUTPUTS CLEARED.
- Each notebook must EMBED at least one SVG from images/ with `![alt](images/name.svg)` plus a one-line italic caption, and must ANIMATE a walking skeleton inline (synthetic in SMOKE, real extracted skeleton in REAL) using the shared animation helper so the motion behind the numbers is always visible.
- SVGs are hand-authored, saved in `gavd/images/`, white bg, rounded rects, slate/blue palette (#dbeafe fill / #1d4ed8 stroke for boxes, #0f172a titles, #475569 body, #334155 arrows), sans-serif, viewBox around 900-980 wide. Match the existing `gait/skeleton-jepa/tutorials/images/*.svg` style. No text overlaps, no clutter.

## The canonical Colab-setup cell (copy this idiom verbatim, adjust the REAL dep list)

```python
# Colab setup and local .env loading.
# Installs only missing packages, loads .env for local paths, and makes the
# local alexpose checkout importable so we can use its ambient package.
import importlib.util, subprocess, sys

_import_name = {
    "scikit-learn": "sklearn",
    "opencv-python": "cv2",
    "yt-dlp": "yt_dlp",
    "python-dotenv": "dotenv",
}

def _ensure(pkgs):
    """pip install any packages whose import is not already available."""
    missing = [p for p in pkgs if importlib.util.find_spec(_import_name.get(p, p)) is None]
    if missing:
        print("Installing:", " ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + missing)
    return missing

# Core packages, always needed (smoke mode uses only these).
_ensure(["numpy", "pandas", "matplotlib", "scikit-learn", "torch", "python-dotenv", "tqdm"])

# REAL-path packages, installed only when SMOKE_TEST is False.
try:
    _want_real = not CONFIG["SMOKE_TEST"]
except NameError:
    _want_real = False
if _want_real:
    _ensure(["opencv-python", "mediapipe", "yt-dlp"])   # trim to what THIS notebook needs

# Credentials, keys, and local paths are ALWAYS loaded with
# load_dotenv(find_dotenv()) - this is the required idiom for this series.
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv(find_dotenv())          # searches upward for .env; no-op if absent
print("Loaded environment via load_dotenv(find_dotenv()).")

ALEXPOSE_REPO = os.getenv("ALEXPOSE_REPO")
GAVD_DATA_DIR = os.getenv("GAVD_DATA_DIR")
YOUTUBE_CACHE_DIR = os.getenv("YOUTUBE_CACHE_DIR")
GAVD_CACHE_DIR = os.getenv("GAVD_CACHE_DIR")
if ALEXPOSE_REPO and os.path.isdir(ALEXPOSE_REPO) and ALEXPOSE_REPO not in sys.path:
    sys.path.insert(0, ALEXPOSE_REPO)
print("Setup complete.")
```

Note: the Colab-setup cell runs BEFORE the CONFIG cell, so it guards
`CONFIG["SMOKE_TEST"]` with `try/except NameError` defaulting to smoke
(install nothing heavy). This matches tutorials/ exactly.

## The alexpose repo and API surface (the "use the repo" requirement)

Repo path (local dev): `/Users/pmui/dev/alexpose/`. GitHub is private, so
the REAL path relies on ALEXPOSE_REPO being on sys.path, and every notebook
also has a minimal INLINE FALLBACK (raw yt-dlp + cv2 + mediapipe) so it still
runs in a fresh Colab where the private repo is unavailable. If even the
inline fallback deps are missing, fall back to synthetic so nothing crashes.

Exact calls (confirmed against the repo):

- `from ambient.gavd import GAVDDataLoader`
  - `df = GAVDDataLoader().load_gavd_data(str(csv_path))` -> pandas DataFrame of one sequence. Columns include seq, frame_num, bbox (parsed to dict), url, id, gait_pat.
- `from ambient.video.youtube_handler import YouTubeHandler`
  - `YouTubeHandler(download_dir=Path(...)).get_or_download_video(url)` -> Path to `{id}.mp4` (or None on failure). Wraps yt_dlp; default quality "best[height<=720]".
  - `.extract_video_id(url)` -> the 11-char id.
- `from ambient.pose.keypoint_extractor import SequenceKeypointExtractor`
  - `ex = SequenceKeypointExtractor()`
  - `kps = ex.extract_from_sequence(sequence_data=df, video_base_path=Path(YOUTUBE_CACHE_DIR), filter_empty=True, min_keypoints=25, verbose=False)` -> list of KeypointSet.
  - A KeypointSet has `.keypoints` (list of 33), each with `.x .y .z .confidence .visibility .presence .name`.
  - COORDINATE UNITS: the alexpose KeypointSet stores PIXELS in `.x`/`.y` (landmark.x * frame_width, landmark.y * frame_height) and the raw MediaPipe depth in `.z`; the [0,1] values live in the separate `.x_normalized`/`.y_normalized` fields. KeypointSet also carries `.frame_width`/`.frame_height`. Raw MediaPipe (inline fallback) returns `.x`/`.y` normalized in [0,1] and raw `.z`. To keep both real paths consistent, notebook 02 emits x,y NORMALIZED to [0,1] on BOTH paths (read `.x_normalized`/`.y_normalized`, or divide `.x`/`.y` by frame size on the ambient path) and keeps z raw. The bbox from the CSV is used on the inline path to crop to the person before pose, then joints are mapped back into whole-frame [0,1].
- `from ambient.pose.joint_angles import get_joint_angles` (the classic hand-feature path; mention as the baseline route JEPA replaces, only used to compute neuroscience scalars in 05 if desired).

The exp5 batch pattern to mirror (notebook 02):
```
condition_paths = [p for p in Path(GAVD_DATA_DIR).iterdir() if p.is_dir() and p.name[0] in string.ascii_letters]
for cond in condition_paths:
    for csv_path in cond.glob("*.csv"):
        df = GAVDDataLoader().load_gavd_data(str(csv_path))
        kps = extractor.extract_from_sequence(sequence_data=df, video_base_path=Path(YOUTUBE_CACHE_DIR), filter_empty=True, min_keypoints=25)
        # -> stack to (T, 33, 3), cache
```

## GAVD data facts (exact)

- CSV location: `<GAVD_DATA_DIR>/<condition>/<seq>.csv`. One CSV = one sequence = frames from ONE YouTube video.
- Columns (10): `seq, frame_num, cam_view, gait_event, dataset, gait_pat, bbox, vid_info, id, url`.
  - `bbox` is a python-dict-as-string `{'top':..,'left':..,'height':..,'width':..}` in PIXELS. Parse with `ast.literal_eval`.
  - `vid_info` is `{'height':..,'width':..,'mime_type':'video/mp4'}`.
  - `url` is the YouTube watch URL; `id` is the 11-char id; `{id}.mp4` is the cached filename.
  - `gait_pat` equals the parent folder name; `frame_num` is 1-based within the source video.
- Condition folders and CSV counts (exact, as of this build): abnormal 190, antalgic 4, cerebral palsy 16, exercise 24, inebriated 2, myopathic 47, normal 12, parkinsons 9, prosthetic 3, stroke 12, style 55. TOTAL = 374 sequences, 91624 rows.
- Multiple sequences can share ONE video id (different people / camera sides in the same clip), so the number of UNIQUE videos to download is smaller than 374. The bulk-download notebook must DEDUP by video id.
- The clinically labeled 5-class subset used by the prior work / baseline: normal (12), parkinsons (9), stroke (12), cerebral palsy (15), myopathic (20) = 68 sequences. Baseline: Random Forest (100 trees) on 82 hand features, 70/30 split, best 76 percent test accuracy across 5 classes. Chance = 20 percent. NOTE: cerebral palsy has 16 CSVs but the labeled subset uses 15, and myopathic has 47 CSVs but the labeled subset uses 20; the manifest keeps ALL sequences for pretraining and marks which 68 are the labeled subset.
- CANONICAL DEMO CLIP for a single worked example: YouTube id `B5hrxKe2nP8` (`https://www.youtube.com/watch?v=B5hrxKe2nP8`), a PARKINSONS clip, cached at `<YOUTUBE_CACHE_DIR>/B5hrxKe2nP8.mp4`, referenced by two parkinsons CSVs: `cljanb45y00083n6lmh1qhydd.csv` and `cljan9b4p00043n6ligceanyp.csv`.

## BLAZEPOSE_33 spec (exact)

33 landmark names, index order:
```
0 NOSE, 1 LEFT_EYE_INNER, 2 LEFT_EYE, 3 LEFT_EYE_OUTER, 4 RIGHT_EYE_INNER,
5 RIGHT_EYE, 6 RIGHT_EYE_OUTER, 7 LEFT_EAR, 8 RIGHT_EAR, 9 MOUTH_LEFT,
10 MOUTH_RIGHT, 11 LEFT_SHOULDER, 12 RIGHT_SHOULDER, 13 LEFT_ELBOW,
14 RIGHT_ELBOW, 15 LEFT_WRIST, 16 RIGHT_WRIST, 17 LEFT_PINKY, 18 RIGHT_PINKY,
19 LEFT_INDEX, 20 RIGHT_INDEX, 21 LEFT_THUMB, 22 RIGHT_THUMB, 23 LEFT_HIP,
24 RIGHT_HIP, 25 LEFT_KNEE, 26 RIGHT_KNEE, 27 LEFT_ANKLE, 28 RIGHT_ANKLE,
29 LEFT_HEEL, 30 RIGHT_HEEL, 31 LEFT_FOOT_INDEX, 32 RIGHT_FOOT_INDEX
```

35 skeleton edges, as (i, j) index pairs:
```
Face:  (0,1)(1,2)(2,3)(0,4)(4,5)(5,6)(0,9)(0,10)(9,10)
Torso: (11,12)(11,23)(12,24)(23,24)
LArm:  (11,13)(13,15)(15,17)(15,19)(15,21)(17,19)
RArm:  (12,14)(14,16)(16,18)(16,20)(16,22)(18,20)
LLeg:  (23,25)(25,27)(27,29)(27,31)(29,31)
RLeg:  (24,26)(26,28)(28,30)(28,32)(30,32)
```

6 semantic groups (for limb-based block masking):
```
face:     [0,1,2,3,4,5,6,7,8,9,10]
left_arm: [11,13,15,17,19,21]
right_arm:[12,14,16,18,20,22]
torso:    [11,12,23,24]
left_leg: [23,25,27,29,31]
right_leg:[24,26,28,30,32]
```
Each landmark has (x, y, z) plus a visibility/confidence score. This series uses C=3 (x, y, z) throughout; be explicit in each notebook.

## Skeleton-JEPA method facts (must match gait/skeleton-jepa/README.md and tutorials/)

- Flavor: pose-sequence JEPA on the 33 joints over time, NOT pixels.
- Masking: spatiotemporal BLOCK masking, two styles. Style A limb-over-time (hide one whole limb across a window of frames). Style B time-window (hide all joints across a short window). Scattered single-joint masking is the too-easy baseline to contrast against.
- Four pieces:
  1. context encoder: reads VISIBLE joints/frames -> context embedding. Small transformer (nn.TransformerEncoder, 2 layers, gelu, batch_first, dropout 0, dim_feedforward = 2*D) PLUS learned positional embeddings. Tokens are the clip flattened to (B, T*33, C) in row-major (t, j) order (token n = t*33 + j). Before the transformer, add a learned time embedding `time_embed` of shape (T, D) and a learned joint embedding `joint_embed` of shape (33, D), broadcast as `pos[t,j] = time_embed[t] + joint_embed[j]` and reshaped to (T*33, D). Init both at std 0.1 so they are on the scale of the projected coordinates (smaller and the transformer's LayerNorm washes them out). WHY (gavd/ divergence from tutorials/03): without positional embeddings a transformer over flattened tokens is permutation-invariant, so a mean-pooled clip embedding is a bag of coordinates that discards frame order and left/right joint identity - it cannot represent gait dynamics at all. This was the root cause of notebook 05 giving negative R-squared for any temporal scalar. The concept tutorials/03-05 still omit pos-embeds (they only teach the pieces); the gavd/ series ADDS them because it actually evaluates the encoder. This is the standard I-JEPA / V-JEPA fix.
  2. target encoder: EMA copy of the context encoder, reads the FULL sequence -> target embeddings (answer key). Update `target = m*target + (1-m)*context`, m near 1 (0.996), stop-gradient (requires_grad False, EMA only).
  3. predictor: takes context + positions of hidden tokens -> predicts their target embeddings. Shallow MLP (Linear-GELU-Linear, hidden = 2*D) is fine for these notebooks.
  4. loss (gavd/ notebook 04, corrected): L2 (invariance) between the prediction and the LAYER-NORMALIZED EMA target, plus LIGHT VICReg variance + covariance applied to the ONLINE context embedding ONLY (never the stop-gradient target). NO negatives, NO decoder.
- Collapse trap: model can cheat with a constant embedding; the EMA target plus the light VICReg variance guard prevents it.
- WHY the loss differs from the concept tutorials/03: applying the VICReg variance/covariance to BOTH pred AND the EMA target (as tutorials/03 does) makes the target encoder's embedding scale inflate over long training, so the raw (un-normalized) L2 climbs and the TOTAL LOSS RISES after ~50 steps. The gavd/ series runs 400 real steps, long enough to expose this, so it (1) LayerNorms the target before the L2 (scale-invariant, V-JEPA style), (2) regularizes the online context only, and (3) uses light weights with a modest variance target. Verified over 400 steps: total falls 12.8 -> 6.0, MSE falls 0.24 -> 0.23, context std stays ~0.37 (no collapse).
- VICReg weights (gavd/ notebook 04): VICREG_SIM=25.0, VICREG_VAR=0.5, VICREG_COV=0.04, VAR_TARGET=0.5, EPS=1e-4, EMA_M=0.996, EMBED_DIM=64. (The concept tutorials/03 still uses the older 25/25/1, gamma=1 weights on the short toy run where the rise never shows.)
- The `make_target_encoder`, `ema_update`, `Predictor` definitions match tutorials/03 verbatim. `ContextEncoder` in the gavd/ series takes extra `T` and `n_joints` args and adds the time+joint positional embeddings described in piece 1 (this is the one intentional divergence from tutorials/03). `vicreg_loss` in gavd/ notebook 04 has the corrected signature `vicreg_loss(pred, target, cfg, context=None)` described above. Notebooks 04 and 05 MUST define the SAME `ContextEncoder` (identical pos-embed layout) and 04 MUST save `T` and `N_JOINTS` in the checkpoint config so 05 can rebuild it identically; 05 rebuilds from the saved config and, on any state_dict shape mismatch (for example a stale checkpoint from an older nb04), prints a WARNING to re-run nb04 and falls back to a fresh encoder rather than crashing.
- Clip embedding for the frozen probe (notebook 05): flatten to (T*33, C) in the SAME (t, j) order, run the frozen encoder, `mean(dim=1)` over tokens -> (D,). With pos-embeds this pooled vector is NOT permutation-invariant, so it carries temporal and left/right structure. (Plain mean beat mean+temporal-std on the small labeled set in testing.)
- Evaluation story: pretrain encoder on UNLABELED skeletons (here: the whole 374-sequence pool), freeze it, spend the labeled clips (the prior work used 68; a cached labeled_holdout.npz may hold fewer) only on a small probe. North star: match or beat 76 percent RF with a frozen probe. Because the labeled set is small, notebook 05 reports RQ1/RQ2/RQ3 as the MEAN (+/- std) over N_SPLITS=20 stratified 70/30 splits, not a single noisy split.
- RQ3 clinical scalars (notebook 05): use LINEARLY DECODABLE proxies so a positive R-squared is meaningful - `asymmetry_index` (|L-R leg swing range| ratio) and `step_amplitude` (mean ankle swing range). Do NOT probe a cycle-to-cycle timing coefficient of variation: it is nonlinear in the coordinates (a ratio of statistics of frame-to-frame diffs), so NO linear probe recovers it from ANY embedding - a negative R-squared there is a property of the target, not the encoder. Verified linear-probe ceilings from raw coords: asymmetry ~0.70, step_amplitude ~0.84, stride_time_cv ~0.02.
- RQ4 ablation (notebook 05): run a faithful miniature of the nb04 loop on the labeled clips (block masking + EMA target + LayerNorm-target L2) and toggle var/cov. Use a faster EMA (0.99) and slightly stronger DEMO-ONLY weights (var 1.0, gamma 1.0), 200 steps, so the effect shows on a fresh encoder. On easy smoke data the EMA target does most anti-collapse work and VICReg ADDS a margin (ON std climbs above OFF); do NOT claim the OFF run crashes to zero.

## The shared inline animation helper (use this exact shape in every notebook)

Every notebook animates a walking skeleton inline. Use a self-contained helper
(no imports beyond matplotlib + numpy) that returns an HTML5 video / JS animation
that renders in both Jupyter and Colab. It takes a (T, 33, 3) array + the edge
list, draws joints as dots and edges as lines per frame, and uses only x, y
(z ignored for the 2D view). In SMOKE mode feed a synthetic walking skeleton;
in REAL mode feed the extracted skeleton. Keep T small for the animation (<= 24
frames) so it renders fast. Match the tutorials/ animation-helper behavior.

## Execution / verification

- SMOKE_TEST verification env (torch only, no cv2/mediapipe): `/Users/pmui/vaults/worldmodels/tutorials/vicreg/.venv/bin/python` (Python 3.14, torch 2.12.1, numpy 2.5.0, sklearn, matplotlib present; pandas, tqdm, dotenv ABSENT). The Colab-setup cell (cell 2) `_ensure(["numpy","pandas","matplotlib","scikit-learn","torch","python-dotenv","tqdm"])` will pip-install pandas/tqdm/dotenv into this venv when the verifier runs it, so later cells CAN import pandas/tqdm/dotenv - but ONLY after the setup cell has run. Do NOT import pandas, tqdm, or dotenv at module scope before cell 2. All notebooks MUST run 0-error top-to-bottom here with SMOKE_TEST=True.
- REAL-path spot-check env (full stack): `/Users/pmui/dev/alexpose/.venv/bin/python` (cv2 4.12.0, mediapipe 0.10.31, yt_dlp, pandas 2.3.3). Use only to sanity-check the REAL path against the cached `B5hrxKe2nP8.mp4`.
- rsvg-convert at /opt/homebrew/bin/rsvg-convert renders SVG->PNG for the declutter/adversarial-review pass.
