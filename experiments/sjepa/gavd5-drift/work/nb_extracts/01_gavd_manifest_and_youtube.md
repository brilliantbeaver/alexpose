# NOTEBOOK 01_gavd_manifest_and_youtube.ipynb
cells=15


=== CELL 0 [markdown] ===
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brilliantbeaver/alexpose/blob/main/penny/gavd3/01_gavd_manifest_and_youtube.ipynb)

# 01. GAVD manifests and YouTube source videos

Turn condition CSV folders into a sequence-level manifest, download each source once, and keep the original frame timeline intact.

**Research use only.** This tutorial does not diagnose a person or validate a clinical device.

**Run it:** locally, use `uv sync` then `uv run jupyter lab` from this folder. In Colab, use the badge and run the setup cell. Restart the kernel after changing `penny/gavd3/.env`.

**Keep the walk visible:** notebook 01 opens the source video, and notebook 02 shows frame, bbox, and skeleton alignment. Revisit those views whenever a latent or classifier result looks surprising.


=== CELL 1 [code] ===
```python
from pathlib import Path
import os
import sys
import subprocess

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/brilliantbeaver/alexpose.git"

if IN_COLAB:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "numpy", "pandas", "scipy", "scikit-learn", "matplotlib",
        "seaborn", "torch", "tqdm", "python-dotenv", "yt-dlp[default]",
        "opencv-python-headless", "mediapipe<1", "joblib", "pyarrow",
    ])
    clone_dir = Path("/content/alexpose")
    if not (clone_dir / ".git").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", REPO_URL, str(clone_dir)])
    os.chdir(clone_dir)


def find_project_root(start=None):
    env_root = os.getenv("ALEXPOSE_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / ".git").exists() and (candidate / "data" / "gavd").exists():
            return candidate
        print(f"Ignoring invalid ALEXPOSE_ROOT: {candidate}")
    start = Path(start or Path.cwd()).resolve()
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists() and (candidate / "data" / "gavd").exists():
            return candidate
    return start


PROJECT_ROOT = find_project_root()
TUTORIAL_DIR = PROJECT_ROOT / "experiments" / "sjepa" / "gavd5"

try:
    from dotenv import load_dotenv
    load_dotenv(TUTORIAL_DIR / ".env", override=False)
    load_dotenv(PROJECT_ROOT / ".env", override=False)
except Exception:
    pass

MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")
if MODE == "smoke":
    print(
        "SMOKE MODE: hand-authored motions test code paths only. "
        "They have no pathophysiological or clinical validity."
    )

PREFERRED_ROOT = Path(
    os.getenv(
        "GAVD4_ROOT",
        "/Users/pmui/vaults/worldmodels/gait/skeleton-jepa/gavd4",
    )
).expanduser()

requested_data = os.getenv("GAVD4_DATA_DIR") or os.getenv("GAVD_DATA_GAVD_DIR")
if requested_data and Path(requested_data).expanduser().exists():
    DATA_GAVD_DIR = Path(requested_data).expanduser()
elif requested_data:
    print(f"Ignoring missing GAVD CSV path: {Path(requested_data).expanduser()}")
    if (PREFERRED_ROOT / "data-gavd").exists():
        DATA_GAVD_DIR = PREFERRED_ROOT / "data-gavd"
    else:
        DATA_GAVD_DIR = PROJECT_ROOT / "data" / "gavd"
elif (PREFERRED_ROOT / "data-gavd").exists():
    DATA_GAVD_DIR = PREFERRED_ROOT / "data-gavd"
else:
    DATA_GAVD_DIR = PROJECT_ROOT / "data" / "gavd"

requested_youtube = os.getenv("GAVD4_YOUTUBE_DIR") or os.getenv("GAVD_YOUTUBE_DIR")
if requested_youtube:
    YOUTUBE_DIR = Path(requested_youtube).expanduser()
elif PREFERRED_ROOT.exists():
    YOUTUBE_DIR = PREFERRED_ROOT / "youtube"
else:
    YOUTUBE_DIR = PROJECT_ROOT / "experiments" / "sjepa" / "gavd5" / "work" / "youtube"

CACHE_DIR = Path(
    os.getenv("GAVD_CACHE_DIR", TUTORIAL_DIR / "work" / "cache")
).expanduser()
ARTIFACT_ROOT = Path(
    os.getenv("GAVD_ARTIFACT_DIR", TUTORIAL_DIR / "work" / "artifacts")
).expanduser()
ARTIFACT_DIR = ARTIFACT_ROOT / MODE
POSE_DIR = ARTIFACT_DIR / "poses"

for folder in [CACHE_DIR, ARTIFACT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg-cache"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import pandas as pd
from IPython.display import SVG, display


def show_tutorial_svg(filename):
    '''Render a repository SVG reliably in local Jupyter and Colab.'''
    path = TUTORIAL_DIR / "images" / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing tutorial figure {path}. Clone the full alexpose repository."
        )
    display(SVG(filename=str(path)))

print(f"mode: {MODE}")
print(f"project: {PROJECT_ROOT}")
print(f"GAVD CSVs: {DATA_GAVD_DIR}")
print(f"YouTube cache: {YOUTUBE_DIR}")
print(f"artifacts: {ARTIFACT_DIR}")

```
--- outputs (1) ---
[stream] mode: real
project: /Users/theodoremui/dev/alexpose
GAVD CSVs: /Users/theodoremui/dev/alexpose/data/gavd
YouTube cache: /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd5/work/youtube
artifacts: /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd5/work/artifacts/real


=== CELL 2 [code] ===
```python
show_tutorial_svg("04_gavd_pipeline.svg")

```
--- outputs (1) ---
[out] <IPython.core.display.SVG object>

=== CELL 3 [code] ===
```python
import ast
from urllib.parse import parse_qs, urlparse

CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
CONDITION_ALIASES = {
    "normal": ["normal"],
    "parkinsons": ["parkinsons", "parkinson", "parkinsons disease"],
    "stroke": ["stroke"],
    "cerebralpalsy": ["cerebralpalsy", "cerebral palsy", "cerebral_palsy"],
    "myopathic": ["myopathic", "myopathy"],
}
COHORT_ROOT = PROJECT_ROOT / "data" / "gavd"
EXPECTED_SEQUENCE_COUNTS = {
    "normal": 12,
    "parkinsons": 9,
    "stroke": 12,
    "cerebralpalsy": 16,
    "myopathic": 47,
}
EXPECTED_VIDEO_COUNTS = {
    "normal": 1,
    "parkinsons": 2,
    "stroke": 3,
    "cerebralpalsy": 2,
    "myopathic": 10,
}


def canonical_condition(value):
    compact = str(value).strip().lower().replace("_", " ")
    for canonical, aliases in CONDITION_ALIASES.items():
        if compact in [a.replace("_", " ") for a in aliases]:
            return canonical
    return compact.replace(" ", "")


def resolve_condition_folder(root, condition):
    root = Path(root)
    for alias in CONDITION_ALIASES[condition]:
        candidate = root / alias
        if candidate.exists():
            return candidate
    return root / condition


def youtube_id_from_url(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    parsed = urlparse(text)
    if parsed.netloc.endswith("youtu.be"):
        value = parsed.path.strip("/").split("/")[0]
        return value if len(value) == 11 else None
    if "youtube.com" in parsed.netloc:
        value = parse_qs(parsed.query).get("v", [None])[0]
        return value if value and len(value) == 11 else None
    return text if len(text) == 11 else None


def safe_literal(value):
    if value is None or pd.isna(value):
        return {}
    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def selected_csv_paths(root=DATA_GAVD_DIR, cohort_root=COHORT_ROOT):
    """Resolve the fixed 96-sequence tutorial cohort against a data source."""
    paths = {}
    for condition in CONDITIONS:
        source_folder = resolve_condition_folder(root, condition)
        cohort_folder = resolve_condition_folder(cohort_root, condition)
        selected_ids = sorted(path.stem for path in cohort_folder.glob("*.csv"))
        expected = EXPECTED_SEQUENCE_COUNTS[condition]
        if len(selected_ids) != expected:
            raise ValueError(
                f"Tutorial cohort definition for {condition} has "
                f"{len(selected_ids)} sequences; expected {expected}: {cohort_folder}"
            )
        missing = [
            sequence_id for sequence_id in selected_ids
            if not (source_folder / f"{sequence_id}.csv").is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Data source {source_folder} is missing {len(missing)} selected "
                f"{condition} sequences; first missing IDs: {missing[:5]}"
            )
        paths[condition] = [
            source_folder / f"{sequence_id}.csv"
            for sequence_id in selected_ids
        ]
    return paths


def scan_gavd_manifest(root=DATA_GAVD_DIR, cohort_root=COHORT_ROOT):
    rows = []
    selected_paths = selected_csv_paths(root, cohort_root)
    for condition in CONDITIONS:
        for csv_path in selected_paths[condition]:
            frame_table = pd.read_csv(csv_path)
            required = {"seq", "frame_num", "bbox", "id", "url"}
            missing = required.difference(frame_table.columns)
            if missing:
                raise ValueError(f"{csv_path} is missing {sorted(missing)}")
            if frame_table.empty:
                raise ValueError(f"Selected GAVD file is empty: {csv_path}")
            sequence_ids = frame_table["seq"].dropna().astype(str).unique()
            if len(sequence_ids) != 1:
                raise ValueError(f"{csv_path} contains {len(sequence_ids)} sequence ids")
            if sequence_ids[0] != csv_path.stem:
                raise ValueError(
                    f"Sequence ID {sequence_ids[0]} does not match filename {csv_path.stem}"
                )
            first = frame_table.iloc[0]
            video_id = youtube_id_from_url(first.get("id")) or youtube_id_from_url(first.get("url"))
            if video_id is None:
                raise ValueError(f"Could not parse a YouTube ID from {csv_path}")
            frames = pd.to_numeric(frame_table["frame_num"], errors="raise").astype(int)
            rows.append({
                "condition": condition,
                "dataset_annotation": first.get("dataset"),
                "gait_pattern_annotation": first.get("gait_pat"),
                "sequence_id": sequence_ids[0],
                "csv_path": str(csv_path),
                "video_id": video_id,
                "url": first.get("url"),
                "first_frame": int(frames.min()),
                "last_frame": int(frames.max()),
                "n_annotated_frames": int(frames.nunique()),
                "cam_view": first.get("cam_view"),
                "source_height": safe_literal(first.get("vid_info")).get("height"),
                "source_width": safe_literal(first.get("vid_info")).get("width"),
            })
    manifest = pd.DataFrame(rows)
    if manifest["sequence_id"].duplicated().any():
        raise ValueError("Sequence IDs must be unique across the selected conditions")
    actual_counts = manifest["condition"].value_counts().to_dict()
    if actual_counts != EXPECTED_SEQUENCE_COUNTS:
        raise ValueError(
            f"Selected cohort does not match the 96-sequence contract: {actual_counts}"
        )
    return manifest.sort_values(["condition", "sequence_id"]).reset_index(drop=True)


manifest = scan_gavd_manifest()

```

=== CELL 4 [markdown] ===
## One CSV is one sequence

The ten GAVD columns are:

seq, frame_num, cam_view, gait_event, dataset, gait_pat, bbox, vid_info, id, url

frame_num is a 1-based absolute frame number in the original source video. It is not relative to the sequence. The CSV has no FPS field, so notebook 02 probes the downloaded MP4.

The `condition` field comes from the folder chosen for this project. The original `dataset` and `gait_pat` values are retained as `dataset_annotation` and `gait_pattern_annotation`. These are dataset annotations, not diagnoses independently verified by this tutorial.

This notebook uses pandas directly. It does not call the legacy GAVD loader because that loader can download videos as a hidden side effect.


=== CELL 5 [code] ===
```python
display(manifest.head())
manifest_path = ARTIFACT_DIR / "manifest.csv"
manifest.to_csv(manifest_path, index=False)
print(f"wrote {manifest_path}")

```
--- outputs (2) ---
[out]        condition dataset_annotation gait_pattern_annotation  \
0  cerebralpalsy      Abnormal Gait          cerebral palsy   
1  cerebralpalsy      Abnormal Gait          cerebral palsy   
2  cerebralpalsy      Abnormal Gait          cerebral palsy   
3  cerebralpalsy      Abnormal Gait          cerebral palsy   
4  cerebralpalsy      Abnormal Gait          cerebral palsy   

                 sequence_id  \
0  cljar878f00c03n6ly2v2ay88   
1  cljar9bqo00c43n6l2u5zmlru   
2  cljar9t8o00c83n6ltculhoct   
3  cljarar9t00cc3n6lqhi9udoc   
4  cljarbn1y00cg3n6l1u4i0d5l   

                                            csv_path     video_id  \
0  /Users/theodoremui/dev/alexpose/data/gavd/cere...  wRntYsztIEY   
1  /Users/theodoremui/dev/alexpose/data/gavd/cere...  wRntYsztIEY   
2  /Users/theodoremui/dev/alexpose/data/gavd/cere...  wRntYsztIEY   
3  /Users/theodoremui/dev/alexpose/data/gavd/cere...  wRntYsztIEY   
4  /Users/theodoremui/dev/alexpose/data/gavd/cere...  wRntYsztIEY   

                                           url  first_frame  last_frame  \
0  https://www.youtube.com/watch?v=wRntYsztIEY            1         194   
1  https://www.youtube.com/watch?v=wRntYsztIEY          234         428   
2  https://www.youtube.com/watch?v=wRntYsztIEY          460        1100   
3  https://www.youtube.com/watch?v=wRntYsztIEY         1246        2009   
4  https://www.youtube.com/watch?v=wRntYsztIEY         2026        2177   

   n_annotated_frames    cam_view  source_height  source_width
[stream] wrote /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd5/work/artifacts/real/manifest.csv


=== CELL 6 [markdown] ===
## Count sequences and independent source videos

Sequence count is not the same as independent-video count. This distinction matters because several annotated sequences can come from the same YouTube upload.


=== CELL 7 [code] ===
```python
census = (
    manifest.groupby("condition")
    .agg(
        sequences=("sequence_id", "nunique"),
        annotated_frames=("n_annotated_frames", "sum"),
        unique_videos=("video_id", "nunique"),
    )
    .reindex(CONDITIONS)
)
actual_sequence_counts = census["sequences"].astype(int).to_dict()
actual_video_counts = census["unique_videos"].astype(int).to_dict()
if actual_sequence_counts != EXPECTED_SEQUENCE_COUNTS:
    raise ValueError(
        f"Unexpected tutorial sequence census: {actual_sequence_counts}"
    )
if actual_video_counts != EXPECTED_VIDEO_COUNTS:
    raise ValueError(
        f"Unexpected tutorial source-video census: {actual_video_counts}"
    )
display(census)
census.to_csv(ARTIFACT_DIR / "source_video_census.csv")
print("all selected sequences:", manifest["sequence_id"].nunique())
print("all selected videos:", manifest["video_id"].nunique())

```
--- outputs (2) ---
[out]                sequences  annotated_frames  unique_videos
condition                                                
normal                12              2598              1
parkinsons             9              1609              2
stroke                12              3609              3
myopathic             47              8871             10
cerebralpalsy         16              5546              2
[stream] all selected sequences: 96
all selected videos: 18


=== CELL 8 [markdown] ===
The experiment is locked to the 96 sequence IDs stored in the repository dataset, even when `DATA_GAVD_DIR` points to a larger mounted corpus. The cohort has only 18 source videos, and all 12 canonical normal sequences come from one video.

Preparing manifests or downloading videos does not expose a sample to the optimizer. Notebook 04 enforces the training boundary separately:

1. finish `data/gavd/normal` training;
2. resume with `parkinsons` while replaying normal;
3. resume with `stroke` while replaying the earlier groups;
4. resume with `myopathic`;
5. resume with `cerebralpalsy`.

The ordering is an executable curriculum, not a display preference. No later stage reinitializes the view encoder or predictor.


=== CELL 9 [code] ===
```python
video_concentration = (
    manifest.groupby(["condition", "video_id"])
    .size()
    .rename("sequences")
    .reset_index()
    .sort_values(["condition", "sequences"], ascending=[True, False])
)
display(video_concentration)
video_concentration.to_csv(
    ARTIFACT_DIR / "source_video_concentration.csv", index=False
)
normal_videos = manifest.loc[
    manifest["condition"].eq("normal"), "video_id"
].unique().tolist()
if len(normal_videos) != EXPECTED_VIDEO_COUNTS["normal"]:
    raise ValueError(
        f"Selected normal cohort has unexpected source videos: {normal_videos}"
    )

```
--- outputs (1) ---
[out]         condition     video_id  sequences
0   cerebralpalsy  DlPDuHBAP7A          8
1   cerebralpalsy  wRntYsztIEY          8
10      myopathic  R8LRCiTvUz8         12
4       myopathic  5qM6wxZ_dNs          8
8       myopathic  9qg-3smwLcs          6
5       myopathic  7Ft1bUTzxkM          5
7       myopathic  8PPLTf0fZsY          4
11      myopathic  jzJIpY6vRLo          4
9       myopathic  HDkWDe6FZDg          3
2       myopathic  05oyBOE_0UE          2
6       myopathic  8NlJgIySMG0          2
3       myopathic  4Yiqk-ud8rI          1
12         normal  3KnFt8bH3tE         12
14     parkinsons  _Wn9oYGpRdM          7
13     parkinsons  B5hrxKe2nP8          2
15         stroke  5gpoegYv1hs          7
16         stroke  8mTHlAIdea0          4
17         stroke  9VzOTO0nV8U          1

=== CELL 10 [markdown] ===
## Download each unique source once

Set GAVD_MODE=real and GAVD_DOWNLOAD=1 in the root .env, or change DOWNLOAD_VIDEOS below. The full source is cached as youtube/condition/video_id.ext. Keeping the full video protects the CSV frame numbers.

The pose pipeline does not use audio. The downloader therefore prefers an adaptive H.264/MP4 video-only stream instead of YouTube's combined format 18, whose CDN URL can return HTTP 403. It first uses YouTube's embedded-web player client because the default Android VR URLs are currently PO-token-limited in this environment, then falls back to the TV client for videos whose owners disabled embedding, and finally tries HLS. It caps each stream at the annotation height and 30 FPS so absolute frame numbers retain their intended timing. No FFmpeg merge is required on the primary paths.

yt-dlp also needs a supported JavaScript runtime for reliable YouTube extraction. This cell enables Deno or Node automatically when either is installed, retries freshly extracted media URLs, audits every source instead of stopping at the first failure, and saves per-video diagnostics.

YouTube availability and terms can change. Download only material you are allowed to use.


=== CELL 11 [code] ===
```python
import cv2
import json
import shutil


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
YTDLP_EMBEDDED_ATTEMPTS = 2
YTDLP_TV_ATTEMPTS = 3
YTDLP_HLS_ATTEMPTS = 6


def cached_video_path(condition, video_id):
    folder = Path(YOUTUBE_DIR) / condition
    exact = folder / f"{video_id}.mp4"
    if exact.exists() and exact.suffix.lower() in VIDEO_SUFFIXES:
        return exact
    matches = sorted(
        path for path in folder.glob(f"{video_id}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES
        and not path.name.endswith(".part")
    )
    return matches[0] if matches else exact


def validate_video(path, required_last_frame=None):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "reason": "missing or empty"}
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, _ = cap.read()
    cap.release()
    reaches_annotation = (
        required_last_frame is None or frames >= int(required_last_frame)
    )
    return {
        "ok": bool(ok and fps > 0 and frames > 0 and reaches_annotation),
        "fps": fps,
        "frames": frames,
        "required_last_frame": required_last_frame,
        "reaches_annotation": reaches_annotation,
        "size_mb": path.stat().st_size / 1_000_000,
    }


def saved_format_metadata(condition, video_id):
    info_path = Path(YOUTUBE_DIR) / condition / f"{video_id}.info.json"
    if not info_path.exists():
        return {}
    info = json.loads(info_path.read_text(encoding="utf-8"))
    return {
        "format_id": info.get("format_id"),
        "format_note": info.get("format_note"),
        "download_fps": info.get("fps"),
        "download_width": info.get("width"),
        "download_height": info.get("height"),
    }


def retry_delay(attempt):
    """Exponential retry delay capped at 30 seconds."""
    return min(2 ** max(attempt - 1, 0), 30)


def unique_video_table(table):
    return (
        table.groupby(["condition", "video_id", "url"], as_index=False)
        .agg(
            required_last_frame=("last_frame", "max"),
            source_height=("source_height", "max"),
        )
    )


def video_format_selector(source_height, hls_only=False):
    try:
        max_height = int(source_height)
    except (TypeError, ValueError):
        max_height = 720
    if max_height <= 0:
        max_height = 720
    if hls_only:
        return f"best[protocol^=m3u8][height<={max_height}][fps<=30]"
    constraints = f"[height<={max_height}][fps<=30]"
    return (
        f"bestvideo[ext=mp4][vcodec^=avc1]{constraints}/"
        f"bestvideo[ext=mp4]{constraints}/"
        f"bestvideo{constraints}/"
        "best[ext=mp4][vcodec!=none][fps<=30]/"
        "best[vcodec!=none][fps<=30]"
    )


def javascript_runtime_options():
    # Deno is enabled by default in yt-dlp; Node must be explicit.
    if shutil.which("deno"):
        return {"js_runtimes": {"deno": {}}}
    if node_path := shutil.which("node"):
        return {"js_runtimes": {"node": {"path": node_path}}}
    raise RuntimeError(
        "yt-dlp needs Deno >=2.3 or Node >=22 for YouTube challenge solving. "
        "Install one and restart the kernel."
    )


def downloader_options(folder, source_height, strategy):
    if strategy not in {"embedded", "tv", "hls"}:
        raise ValueError(f"Unknown YouTube download strategy: {strategy}")
    options = {
        "outtmpl": str(folder / "%(id)s.%(ext)s"),
        "format": video_format_selector(
            source_height, hls_only=strategy == "hls"
        ),
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 3,
        "file_access_retries": 3,
        "retry_sleep_functions": {
            "http": retry_delay,
            "fragment": retry_delay,
        },
        "continuedl": strategy == "embedded",
        "noplaylist": True,
        "writeinfojson": True,
        "overwrites": True,
        "quiet": False,
        **javascript_runtime_options(),
    }
    if strategy == "embedded":
        options["extractor_args"] = {
            "youtube": {"player_client": ["web_embedded"]},
        }
    elif strategy == "tv":
        options["extractor_args"] = {
            "youtube": {"player_client": ["tv_downgraded"]},
        }
    return options


def download_unique_videos(table, max_videos=None):
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    unique = unique_video_table(table)
    if max_videos is not None:
        unique = unique.head(max_videos)
    results = []
    for row in unique.itertuples(index=False):
        folder = Path(YOUTUBE_DIR) / row.condition
        folder.mkdir(parents=True, exist_ok=True)
        destination = cached_video_path(row.condition, row.video_id)
        status = validate_video(destination, row.required_last_frame)
        if status.get("ok"):
            results.append({
                **row._asdict(),
                "status": "cached",
                "path": str(destination),
                "error": None,
                **saved_format_metadata(row.condition, row.video_id),
                **status,
            })
            continue
        info = None
        error = None
        strategies = (
            ["embedded"] * YTDLP_EMBEDDED_ATTEMPTS
            + ["tv"] * YTDLP_TV_ATTEMPTS
            + ["hls"] * YTDLP_HLS_ATTEMPTS
        )
        for attempt, strategy in enumerate(strategies, start=1):
            try:
                with YoutubeDL(
                    downloader_options(folder, row.source_height, strategy)
                ) as downloader:
                    info = downloader.extract_info(row.url, download=True)
                break
            except DownloadError as exc:
                error = exc
                if attempt < len(strategies):
                    print(
                        f"Retrying {row.video_id} via {strategies[attempt]} "
                        f"({attempt + 1}/{len(strategies)})"
                    )
        if info is None:
            destination = cached_video_path(row.condition, row.video_id)
            results.append({
                **row._asdict(),
                "status": "failed",
                "path": str(destination),
                "error": str(error),
                **validate_video(destination, row.required_last_frame),
            })
            continue

        if info.get("id") != row.video_id:
            raise ValueError(
                f"Downloaded YouTube ID {info.get('id')} instead of {row.video_id}"
            )

        destination = cached_video_path(row.condition, row.video_id)
        status = validate_video(destination, row.required_last_frame)
        results.append({
            **row._asdict(),
            "status": "downloaded" if status.get("ok") else "failed",
            "path": str(destination),
            "error": None if status.get("ok") else f"Video validation failed: {status}",
            "format_id": info.get("format_id"),
            "format_note": info.get("format_note"),
            "download_fps": info.get("fps"),
            "download_width": info.get("width"),
            "download_height": info.get("height"),
            **status,
        })
    return pd.DataFrame(results)


def audit_cached_videos(table):
    unique = unique_video_table(table)
    rows = []
    for row in unique.itertuples(index=False):
        path = cached_video_path(row.condition, row.video_id)
        rows.append({
            **row._asdict(),
            "path": str(path),
            **saved_format_metadata(row.condition, row.video_id),
            **validate_video(path, row.required_last_frame),
        })
    return pd.DataFrame(rows)


DOWNLOAD_VIDEOS = MODE == "real" and os.getenv("GAVD_DOWNLOAD", "0") == "1"
MAX_VIDEOS = None
if DOWNLOAD_VIDEOS:
    download_report = download_unique_videos(manifest, MAX_VIDEOS)
else:
    print("D
... [truncated, 8628 chars total]
```
--- outputs (1) ---
[out]         condition     video_id                                          url  \
0   cerebralpalsy  DlPDuHBAP7A  https://www.youtube.com/watch?v=DlPDuHBAP7A   
1   cerebralpalsy  wRntYsztIEY  https://www.youtube.com/watch?v=wRntYsztIEY   
2       myopathic  05oyBOE_0UE  https://www.youtube.com/watch?v=05oyBOE_0UE   
3       myopathic  4Yiqk-ud8rI  https://www.youtube.com/watch?v=4Yiqk-ud8rI   
4       myopathic  5qM6wxZ_dNs  https://www.youtube.com/watch?v=5qM6wxZ_dNs   
5       myopathic  7Ft1bUTzxkM  https://www.youtube.com/watch?v=7Ft1bUTzxkM   
6       myopathic  8NlJgIySMG0  https://www.youtube.com/watch?v=8NlJgIySMG0   
7       myopathic  8PPLTf0fZsY  https://www.youtube.com/watch?v=8PPLTf0fZsY   
8       myopathic  9qg-3smwLcs  https://www.youtube.com/watch?v=9qg-3smwLcs   
9       myopathic  HDkWDe6FZDg  https://www.youtube.com/watch?v=HDkWDe6FZDg   
10      myopathic  R8LRCiTvUz8  https://www.youtube.com/watch?v=R8LRCiTvUz8   
11      myopathic  jzJIpY6vRLo  https://www.youtube.com/watch?v=jzJIpY6vRLo   
12         normal  3KnFt8bH3tE  https://www.youtube.com/watch?v=3KnFt8bH3tE   
13     parkinsons  B5hrxKe2nP8  https://www.youtube.com/watch?v=B5hrxKe2nP8   
14     parkinsons  _Wn9oYGpRdM  https://www.youtube.com/watch?v=_Wn9oYGpRdM   
15         stroke  5gpoegYv1hs  https://www.youtube.com/watch?v=5gpoegYv1hs   
16         stroke  8mTHlAIdea0  https://www.youtube.com/watch?v=8mTHlAIdea0   
17         stroke  9VzOTO0nV8U  https://www.youtube.com/watch?v=9VzOTO0nV8U   

=== CELL 12 [markdown] ===
## Preview a walk in context

When the source MP4 is cached, the notebook embeds that exact file in an HTML5 player:the same bytes later used for validation and pose extraction. Before download, it uses a privacy-enhanced YouTube iframe and keeps a direct link under the player for browsers or notebook hosts that block third-party frames.


=== CELL 13 [code] ===
```python
import re
from html import escape
from IPython.display import HTML, Video, display


def youtube_player(video_id, width=720):
    """Return an explicit YouTube iframe plus a usable fallback link."""
    video_id = str(video_id).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError(f"Invalid YouTube video ID: {video_id!r}")
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    height = round(width * 9 / 16)
    safe_id = escape(video_id, quote=True)
    watch_url = f"https://www.youtube.com/watch?v={safe_id}"
    embed_url = f"https://www.youtube-nocookie.com/embed/{safe_id}?rel=0"
    return HTML(f"""
    <div style="max-width: {width}px">
      <div style="position: relative; width: 100%; padding-top: 56.25%">
        <iframe
          src="{embed_url}" title="YouTube source video"
          width="{width}" height="{height}"
          style="position: absolute; inset: 0; width: 100%; height: 100%; border: 0"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
          referrerpolicy="strict-origin-when-cross-origin" allowfullscreen>
        </iframe>
      </div>
      <p style="margin: .5rem 0 0">
        If the player is blocked, <a href="{watch_url}" target="_blank" rel="noopener noreferrer">open this video on YouTube</a>.
      </p>
    </div>
    """)


def show_source_video(row, width=720):
    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive")
    path = cached_video_path(row["condition"], row["video_id"])
    if path.exists():
        print("Cached source:", path)
        status = validate_video(path, row.get("last_frame"))
        if not status["ok"]:
            raise ValueError(f"Cached video is not usable: {status}")
        display(Video(
            filename=str(path), embed=True, width=int(width),
            html_attributes="controls preload='metadata' playsinline",
        ))
        return
    print("Cached MP4 is absent; downstream extraction needs a download.")
    display(youtube_player(row["video_id"], width=width))

example = manifest.loc[manifest["condition"].eq("normal")].iloc[0]
show_source_video(example)
```
--- outputs (2) ---
[stream] Cached source: /Users/theodoremui/dev/alexpose/experiments/sjepa/gavd5/work/youtube/normal/3KnFt8bH3tE.mp4

[out] <IPython.core.display.Video object>

=== CELL 14 [markdown] ===
## Checkpoint before moving on

- manifest.csv has one row per sequence.
- Conditions use the canonical spelling cerebralpalsy.
- Cached files are separated by condition and named by YouTube ID.
- Each cached video opens, has a positive FPS, and contains frames.
- The source-video census is saved beside the scientific results.

Notebook 02 uses each CSV's annotated frame span and bbox. It does not process an entire YouTube video as one sample.
