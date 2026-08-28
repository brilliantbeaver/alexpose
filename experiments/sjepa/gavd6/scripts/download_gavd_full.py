#!/usr/bin/env -S uv run --no-sync python
"""Download and validate the unique YouTube sources in a full GAVD manifest.

The input is the one-row-per-sequence ``gavd_full_sequences.csv`` produced in
step 3 of the full-GAVD setup.  It deliberately downloads a complete source
video for each unique YouTube ID instead of cutting per-sequence clips: GAVD's
``frame_num`` values are absolute, one-based source-video frame numbers.

The downloader is a standalone version of the policy in foundation notebook
``01_gavd_manifest_and_youtube.ipynb``.  It is resumable: a cached video is
accepted only after OpenCV can decode it and it reaches every sequence's last
annotated frame.  Failed videos remain in the audit report and are never
silently substituted.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable

import cv2
import pandas as pd


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
YTDLP_EMBEDDED_ATTEMPTS = 2
YTDLP_TV_ATTEMPTS = 3
YTDLP_HLS_ATTEMPTS = 6
REQUIRED_MANIFEST_COLUMNS = {"video_id", "url", "last_frame", "source_height"}


def default_root() -> Path | None:
    value = os.getenv("GAVD_FULL_ROOT")
    return Path(value).expanduser() if value else None


def parse_args() -> argparse.Namespace:
    root = default_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=root / "manifests" / "gavd_full_sequences.csv" if root else None,
        help="One-row-per-sequence CSV from the manifest-building step.",
    )
    parser.add_argument(
        "--youtube-dir",
        type=Path,
        default=root / "youtube" if root else None,
        help="Flat cache root; files are stored as all/<youtube-id>.<ext>.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "reports" / "video_download_audit.csv" if root else None,
        help="CSV audit path. It is atomically updated after every source.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        help="Process only the first N deterministic video IDs (for a pilot).",
    )
    parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        help="Process one YouTube ID; may be supplied more than once.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of deterministic shards for a Slurm job array (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based current shard index (default: 0).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and enumerate the selected videos without contacting YouTube.",
    )
    args = parser.parse_args()
    for name in ("manifest", "youtube_dir", "report"):
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required (or set GAVD_FULL_ROOT)")
    if args.max_videos is not None and args.max_videos < 1:
        parser.error("--max-videos must be positive")
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")
    return args


def cached_video_path(youtube_dir: Path, video_id: str) -> Path:
    """Return an existing supported cache entry, or the expected MP4 path."""
    folder = youtube_dir / "all"
    exact = folder / f"{video_id}.mp4"
    if exact.exists():
        return exact
    matches = sorted(
        path
        for path in folder.glob(f"{video_id}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES and not path.name.endswith(".part")
    )
    return matches[0] if matches else exact


def validate_video(path: Path, required_last_frame: int | None = None) -> dict[str, Any]:
    """Check that an OpenCV-readable file reaches the last required frame."""
    if not path.exists() or path.stat().st_size == 0:
        return {"ok": False, "reason": "missing or empty"}
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        readable, _ = capture.read()
    finally:
        capture.release()
    reaches_annotation = (
        required_last_frame is None or frames >= int(required_last_frame)
    )
    return {
        "ok": bool(readable and fps > 0 and frames > 0 and reaches_annotation),
        "fps": fps,
        "frames": frames,
        "required_last_frame": required_last_frame,
        "reaches_annotation": reaches_annotation,
        "size_mb": path.stat().st_size / 1_000_000,
    }


def saved_format_metadata(youtube_dir: Path, video_id: str) -> dict[str, Any]:
    path = youtube_dir / "all" / f"{video_id}.info.json"
    if not path.exists():
        return {}
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"info_json_error": str(error)}
    return {
        "format_id": info.get("format_id"),
        "format_note": info.get("format_note"),
        "download_fps": info.get("fps"),
        "download_width": info.get("width"),
        "download_height": info.get("height"),
    }


def load_unique_videos(manifest_path: Path) -> pd.DataFrame:
    """Validate the sequence manifest and reduce it to one row per source."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    manifest = pd.read_csv(manifest_path, low_memory=False)
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(
            f"{manifest_path} is missing required columns: {sorted(missing)}"
        )
    selected = manifest.loc[:, sorted(REQUIRED_MANIFEST_COLUMNS)].copy()
    selected["video_id"] = selected["video_id"].astype("string").str.strip()
    selected["url"] = selected["url"].astype("string").str.strip()
    if selected[["video_id", "url"]].isna().any().any() or (
        selected[["video_id", "url"]] == ""
    ).any().any():
        raise ValueError("Manifest contains an empty video_id or url")
    invalid_ids = selected.loc[
        ~selected["video_id"].str.fullmatch(r"[A-Za-z0-9_-]{11}", na=False),
        "video_id",
    ].unique()
    if len(invalid_ids):
        raise ValueError(
            "Manifest contains invalid YouTube IDs: "
            + ", ".join(map(str, invalid_ids[:20]))
        )
    selected["last_frame"] = pd.to_numeric(
        selected["last_frame"], errors="raise"
    ).astype(int)
    if (selected["last_frame"] < 1).any():
        raise ValueError("Manifest contains a non-positive last_frame")
    selected["source_height"] = pd.to_numeric(
        selected["source_height"], errors="coerce"
    )

    # Different query-string spellings of a watch URL are harmless; different
    # YouTube IDs are not.  The step-3 manifest already derives video_id from
    # url/id, so video_id is the stable global cache key.
    return (
        selected.groupby("video_id", as_index=False, dropna=False)
        .agg(
            url=("url", "first"),
            required_last_frame=("last_frame", "max"),
            source_height=("source_height", "max"),
            sequences=("last_frame", "size"),
        )
        .sort_values("video_id", kind="stable")
        .reset_index(drop=True)
    )


def select_videos(
    videos: pd.DataFrame,
    *,
    video_ids: Iterable[str] | None,
    shard_count: int,
    shard_index: int,
    max_videos: int | None,
) -> pd.DataFrame:
    selected = videos
    if video_ids:
        requested = {str(video_id).strip() for video_id in video_ids}
        selected = selected.loc[selected["video_id"].isin(requested)]
        absent = sorted(requested - set(selected["video_id"].astype(str)))
        if absent:
            raise ValueError(f"Requested YouTube IDs are not in the manifest: {absent}")
    selected = selected.iloc[shard_index::shard_count]
    if max_videos is not None:
        selected = selected.head(max_videos)
    return selected.reset_index(drop=True)


def video_format_selector(source_height: Any, hls_only: bool = False) -> str:
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


def retry_delay(attempt: int) -> int:
    return min(2 ** max(attempt - 1, 0), 30)


def javascript_runtime_options() -> dict[str, Any]:
    """Select the JavaScript runtime yt-dlp needs for modern YouTube pages."""
    if shutil.which("deno"):
        return {"js_runtimes": {"deno": {}}}
    if node_path := shutil.which("node"):
        return {"js_runtimes": {"node": {"path": node_path}}}
    raise RuntimeError(
        "yt-dlp needs Deno >=2.3 or Node >=22 for YouTube challenge solving. "
        "Install one, then rerun this job."
    )


def downloader_options(folder: Path, source_height: Any, strategy: str) -> dict[str, Any]:
    if strategy not in {"embedded", "tv", "hls"}:
        raise ValueError(f"Unknown YouTube download strategy: {strategy}")
    options: dict[str, Any] = {
        "outtmpl": str(folder / "%(id)s.%(ext)s"),
        "format": video_format_selector(source_height, hls_only=strategy == "hls"),
        "socket_timeout": 30,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 3,
        "file_access_retries": 3,
        "retry_sleep_functions": {"http": retry_delay, "fragment": retry_delay},
        "continuedl": strategy == "embedded",
        "noplaylist": True,
        "writeinfojson": True,
        "overwrites": True,
        "quiet": False,
        **javascript_runtime_options(),
    }
    if strategy == "embedded":
        options["extractor_args"] = {"youtube": {"player_client": ["web_embedded"]}}
    elif strategy == "tv":
        options["extractor_args"] = {"youtube": {"player_client": ["tv_downgraded"]}}
    return options


def attempt_strategies() -> list[str]:
    return (
        ["embedded"] * YTDLP_EMBEDDED_ATTEMPTS
        + ["tv"] * YTDLP_TV_ATTEMPTS
        + ["hls"] * YTDLP_HLS_ATTEMPTS
    )


def download_video(row: Any, youtube_dir: Path) -> dict[str, Any]:
    """Download one unique source, returning a report row instead of raising."""
    from yt_dlp import YoutubeDL

    video_id = str(row.video_id)
    required_last_frame = int(row.required_last_frame)
    folder = youtube_dir / "all"
    folder.mkdir(parents=True, exist_ok=True)
    destination = cached_video_path(youtube_dir, video_id)
    cached_status = validate_video(destination, required_last_frame)
    common = row._asdict()
    if cached_status.get("ok"):
        return {
            **common,
            "status": "cached",
            "strategy": None,
            "path": str(destination),
            "error": None,
            **saved_format_metadata(youtube_dir, video_id),
            **cached_status,
        }

    info: dict[str, Any] | None = None
    error: Exception | None = None
    successful_strategy: str | None = None
    strategies = attempt_strategies()
    for attempt, strategy in enumerate(strategies, start=1):
        try:
            with YoutubeDL(downloader_options(folder, row.source_height, strategy)) as downloader:
                info = downloader.extract_info(str(row.url), download=True)
            if info.get("id") != video_id:
                raise ValueError(
                    f"Downloaded YouTube ID {info.get('id')} instead of {video_id}"
                )
            successful_strategy = strategy
            break
        except Exception as caught:  # Audit each source even when yt-dlp changes errors.
            error = caught
            if attempt < len(strategies):
                next_strategy = strategies[attempt]
                print(
                    f"Retrying {video_id} via {next_strategy} "
                    f"({attempt + 1}/{len(strategies)}): {caught}",
                    file=sys.stderr,
                )

    destination = cached_video_path(youtube_dir, video_id)
    status = validate_video(destination, required_last_frame)
    if info is None:
        return {
            **common,
            "status": "failed",
            "strategy": None,
            "path": str(destination),
            "error": str(error) if error else "No downloader result",
            **saved_format_metadata(youtube_dir, video_id),
            **status,
        }
    return {
        **common,
        "status": "downloaded" if status.get("ok") else "failed",
        "strategy": successful_strategy,
        "path": str(destination),
        "error": None if status.get("ok") else f"Video validation failed: {status}",
        "format_id": info.get("format_id"),
        "format_note": info.get("format_note"),
        "download_fps": info.get("fps"),
        "download_width": info.get("width"),
        "download_height": info.get("height"),
        **status,
    }


def write_report(rows: list[dict[str, Any]], report_path: Path) -> None:
    """Replace the report atomically so interruption never leaves a partial CSV."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(report_path)


def main() -> int:
    args = parse_args()
    videos = load_unique_videos(args.manifest)
    selected = select_videos(
        videos,
        video_ids=args.video_ids,
        shard_count=args.shard_count,
        shard_index=args.shard_index,
        max_videos=args.max_videos,
    )
    print(
        f"Manifest: {args.manifest}\n"
        f"Unique videos: {len(videos)}; selected: {len(selected)} "
        f"(shard {args.shard_index}/{args.shard_count})\n"
        f"Cache: {args.youtube_dir}\nReport: {args.report}"
    )
    if args.dry_run:
        selected.to_csv(args.report, index=False)
        print("Dry run complete; no YouTube requests made.")
        return 0

    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected.itertuples(index=False), start=1):
        print(f"[{index}/{len(selected)}] {row.video_id}")
        result = download_video(row, args.youtube_dir)
        rows.append(result)
        write_report(rows, args.report)
        print(f"  {result['status']}: {result['path']}")

    report = pd.DataFrame(rows)
    failed = report.loc[~report["ok"].fillna(False)] if not report.empty else report
    print(
        f"Completed {len(report)} sources: "
        f"{len(report) - len(failed)} valid, {len(failed)} failed."
    )
    if not failed.empty:
        print(f"Inspect failures in {args.report}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
