"""Bounded, auditable YouTube acquisition for the GAVD notebooks.

The module distinguishes acquisition failures that may improve on retry from
valid media that cannot cover its annotations.  The latter is dataset attrition,
not a network error, and must not be downloaded repeatedly.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Callable

import cv2
import numpy as np
import pandas as pd


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
DOWNLOAD_STRATEGIES = ("embedded", "tv", "hls", "fallback")
TERMINAL_FAILURES = {
    "annotation_span_too_short",
    "private",
    "unavailable",
    "copyright",
}


class DownloadBatchError(RuntimeError):
    """Raised only when the caller explicitly requests strict acquisition."""


def cached_video_path(youtube_dir: Path | str, condition: str, video_id: str) -> Path:
    folder = Path(youtube_dir) / condition
    exact = folder / f"{video_id}.mp4"
    if exact.exists() and exact.suffix.lower() in VIDEO_SUFFIXES:
        return exact
    matches = sorted(
        path
        for path in folder.glob(f"{video_id}.*")
        if path.suffix.lower() in VIDEO_SUFFIXES
        and not path.name.endswith(".part")
    )
    return matches[0] if matches else exact


def validate_video(path: Path | str, required_last_frame: int | None = None) -> dict[str, object]:
    """Open the media and decode both its first and last required frames."""
    path = Path(path)
    base = {
        "required_last_frame": required_last_frame,
        "fps": np.nan,
        "frames": 0,
        "reaches_annotation": False,
        "decode_opened": False,
        "first_frame_decoded": False,
        "last_annotated_frame_decoded": False,
        "size_mb": 0.0,
    }
    if not path.exists() or path.stat().st_size == 0:
        return {
            **base,
            "ok": False,
            "reason": "missing or empty",
            "failure_kind": "missing_media",
            "retry_recommended": True,
        }

    base["size_mb"] = path.stat().st_size / 1_000_000
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    base["decode_opened"] = opened
    if not opened:
        capture.release()
        return {
            **base,
            "ok": False,
            "reason": "container did not open",
            "failure_kind": "container_open_failed",
            "retry_recommended": True,
        }

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    first_ok, _ = capture.read()
    last_ok = first_ok
    if required_last_frame is not None and required_last_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(required_last_frame) - 1)
        last_ok, _ = capture.read()
    capture.release()
    reaches = required_last_frame is None or frames >= int(required_last_frame)
    base.update(
        {
            "fps": fps,
            "frames": frames,
            "reaches_annotation": reaches,
            "first_frame_decoded": bool(first_ok),
            "last_annotated_frame_decoded": bool(last_ok),
        }
    )

    if not first_ok or fps <= 0 or frames <= 0:
        return {
            **base,
            "ok": False,
            "reason": "invalid FPS/frame count or first frame did not decode",
            "failure_kind": "decode_failed",
            "retry_recommended": True,
        }
    if not reaches:
        return {
            **base,
            "ok": False,
            "reason": f"media has {frames} frames but annotations require frame {required_last_frame}",
            "failure_kind": "annotation_span_too_short",
            "retry_recommended": False,
        }
    if not last_ok:
        return {
            **base,
            "ok": False,
            "reason": f"annotated frame {required_last_frame} did not decode",
            "failure_kind": "last_annotated_frame_decode_failed",
            "retry_recommended": True,
        }
    return {
        **base,
        "ok": True,
        "reason": "ok",
        "failure_kind": None,
        "retry_recommended": False,
    }


def classify_download_error(error: BaseException | str) -> tuple[str, bool]:
    message = str(error).lower()
    if "sign in to confirm" in message or "not a bot" in message:
        return "anti_bot", True
    if "429" in message or "too many requests" in message:
        return "rate_limited", True
    if "requested format is not available" in message:
        return "format_unavailable", True
    if "timed out" in message or "timeout" in message:
        return "timeout", True
    if any(token in message for token in ("http error 500", "http error 502", "http error 503", "http error 504")):
        return "server_error", True
    if any(token in message for token in ("connection reset", "connection aborted", "temporary failure")):
        return "network_error", True
    if "private video" in message:
        return "private", False
    if "this video is unavailable" in message:
        return "unavailable", False
    if "video unavailable" in message:
        # Some YouTube clients return this generic response while another
        # client still exposes formats; exhaust the bounded client fallbacks.
        return "client_unavailable", True
    if "copyright" in message:
        return "copyright", False
    return "download_error", True


def saved_format_metadata(youtube_dir: Path | str, condition: str, video_id: str) -> dict[str, object]:
    info_path = Path(youtube_dir) / condition / f"{video_id}.info.json"
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


def unique_video_table(table: pd.DataFrame) -> pd.DataFrame:
    return (
        table.groupby(["condition", "video_id", "url"], as_index=False)
        .agg(
            required_last_frame=("last_frame", "max"),
            source_height=("source_height", "max"),
        )
    )


def _max_height(source_height: object) -> int:
    try:
        value = int(float(source_height))
    except (TypeError, ValueError):
        value = 720
    return value if value > 0 else 720


def video_format_selector(source_height: object, strategy: str) -> str:
    max_height = _max_height(source_height)
    constraints = f"[height<={max_height}][fps<=30]"
    if strategy == "hls":
        return (
            f"best[protocol^=m3u8]{constraints}/"
            "best[protocol^=m3u8][fps<=30]"
        )
    if strategy == "fallback":
        # A muxed stream is preferable here.  This final selector fixes cases
        # where the source has ordinary formats but none match a strict client.
        return (
            f"best[vcodec!=none]{constraints}/"
            f"bestvideo{constraints}/"
            "best[vcodec!=none][fps<=30]"
        )
    return (
        f"bestvideo[ext=mp4][vcodec^=avc1]{constraints}/"
        f"bestvideo[ext=mp4]{constraints}/"
        f"bestvideo{constraints}/"
        "best[ext=mp4][vcodec!=none][fps<=30]/"
        "best[vcodec!=none][fps<=30]"
    )


def javascript_runtime_options() -> dict[str, object]:
    if shutil.which("deno"):
        return {"js_runtimes": {"deno": {}}}
    if node_path := shutil.which("node"):
        return {"js_runtimes": {"node": {"path": node_path}}}
    raise RuntimeError(
        "yt-dlp needs Deno >=2.3 or Node >=22 for YouTube challenge solving. "
        "Install one and restart the kernel."
    )


def downloader_options(folder: Path, source_height: object, strategy: str) -> dict[str, object]:
    if strategy not in DOWNLOAD_STRATEGIES:
        raise ValueError(f"Unknown YouTube download strategy: {strategy}")
    options: dict[str, object] = {
        "outtmpl": str(folder / "%(id)s.%(ext)s"),
        "format": video_format_selector(source_height, strategy),
        "socket_timeout": 30,
        # Small internal retries handle a broken fragment; client-level retries
        # below handle extraction/format failures and remain fully auditable.
        "retries": 2,
        "fragment_retries": 2,
        "extractor_retries": 1,
        "file_access_retries": 1,
        "continuedl": True,
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
    elif strategy == "fallback":
        options["extractor_args"] = {
            "youtube": {"player_client": ["web_embedded", "tv_downgraded"]}
        }
    return options


def _result_row(
    row: object,
    youtube_dir: Path | str,
    status: str,
    validation: dict[str, object],
    attempts: list[dict[str, object]],
    error: str | None = None,
    info: dict[str, object] | None = None,
) -> dict[str, object]:
    path = cached_video_path(youtube_dir, str(row.condition), str(row.video_id))
    metadata = saved_format_metadata(youtube_dir, str(row.condition), str(row.video_id))
    if info:
        metadata.update(
            {
                "format_id": info.get("format_id"),
                "format_note": info.get("format_note"),
                "download_fps": info.get("fps"),
                "download_width": info.get("width"),
                "download_height": info.get("height"),
            }
        )
    return {
        **row._asdict(),
        "status": status,
        "path": str(path),
        "error": error,
        "attempt_count": len(attempts),
        "strategies_attempted": ",".join(str(item["strategy"]) for item in attempts),
        "attempt_log": json.dumps(attempts, ensure_ascii=False),
        **metadata,
        **validation,
    }


def download_unique_videos(
    table: pd.DataFrame,
    youtube_dir: Path | str,
    max_videos: int | None = None,
    retry_cooldown_seconds: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
) -> pd.DataFrame:
    """Download each source with at most one attempt per named strategy."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    unique = unique_video_table(table)
    if max_videos is not None:
        unique = unique.head(max_videos)
    results: list[dict[str, object]] = []
    for row in unique.itertuples(index=False):
        folder = Path(youtube_dir) / str(row.condition)
        folder.mkdir(parents=True, exist_ok=True)
        destination = cached_video_path(youtube_dir, str(row.condition), str(row.video_id))
        validation = validate_video(destination, int(row.required_last_frame))
        if validation["ok"]:
            results.append(_result_row(row, youtube_dir, "cached", validation, []))
            continue
        if validation["failure_kind"] == "annotation_span_too_short":
            # More network requests cannot add frames to the public source.
            results.append(
                _result_row(row, youtube_dir, "excluded_terminal", validation, [])
            )
            continue

        attempts: list[dict[str, object]] = []
        info: dict[str, object] | None = None
        error_message: str | None = None
        error_kind = str(validation["failure_kind"])
        retryable = bool(validation["retry_recommended"])
        for attempt_index, strategy in enumerate(DOWNLOAD_STRATEGIES):
            if attempt_index and retry_cooldown_seconds > 0:
                delay = min(retry_cooldown_seconds * (2 ** (attempt_index - 1)), 30.0)
                print(f"Cooling down {delay:.0f}s before {row.video_id} strategy={strategy}")
                sleep(delay)
            try:
                with YoutubeDL(downloader_options(folder, row.source_height, strategy)) as downloader:
                    extracted = downloader.extract_info(str(row.url), download=True)
                info = extracted if isinstance(extracted, dict) else None
                if info and info.get("id") != row.video_id:
                    raise ValueError(
                        f"Downloaded YouTube ID {info.get('id')} instead of {row.video_id}"
                    )
                destination = cached_video_path(
                    youtube_dir, str(row.condition), str(row.video_id)
                )
                validation = validate_video(destination, int(row.required_last_frame))
                attempts.append(
                    {
                        "strategy": strategy,
                        "outcome": "validated" if validation["ok"] else "validation_failed",
                        "failure_kind": validation["failure_kind"],
                        "reason": validation["reason"],
                    }
                )
                error_kind = str(validation["failure_kind"])
                retryable = bool(validation["retry_recommended"])
                if validation["ok"] or not retryable:
                    break
            except DownloadError as exc:
                error_message = str(exc)
                error_kind, retryable = classify_download_error(exc)
                attempts.append(
                    {
                        "strategy": strategy,
                        "outcome": "download_error",
                        "failure_kind": error_kind,
                        "reason": error_message,
                    }
                )
                if not retryable:
                    break
            # Local filesystem errors, an unexpected video ID, and programming
            # errors are not source attrition.  Let them fail loudly instead of
            # burning retries or producing a misleading exclusion row.

        if not validation["ok"] and error_message:
            validation = {
                **validation,
                "reason": error_message,
                "failure_kind": error_kind,
                "retry_recommended": retryable,
            }
        if validation["ok"]:
            status = "downloaded"
            error_message = None
        elif not retryable or error_kind in TERMINAL_FAILURES:
            status = "excluded_terminal" if destination.exists() else "failed_terminal"
        else:
            status = "failed_retryable"
        results.append(
            _result_row(
                row,
                youtube_dir,
                status,
                validation,
                attempts,
                error_message,
                info,
            )
        )
    return pd.DataFrame(results)


def audit_cached_videos(table: pd.DataFrame, youtube_dir: Path | str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in unique_video_table(table).itertuples(index=False):
        path = cached_video_path(youtube_dir, str(row.condition), str(row.video_id))
        validation = validate_video(path, int(row.required_last_frame))
        if validation["ok"]:
            status = "cached"
        elif validation["failure_kind"] == "annotation_span_too_short":
            status = "excluded_terminal"
        else:
            status = "missing"
        rows.append(_result_row(row, youtube_dir, status, validation, []))
    return pd.DataFrame(rows)


def failure_summary(report: pd.DataFrame) -> dict[str, object]:
    failed = report.loc[~report["ok"].fillna(False).astype(bool)].copy()
    retryable = failed.loc[failed["retry_recommended"].fillna(False).astype(bool)]
    terminal = failed.loc[~failed["retry_recommended"].fillna(False).astype(bool)]
    return {
        "failed_sources": int(len(failed)),
        "retryable_sources": retryable["video_id"].astype(str).tolist(),
        "terminal_exclusions": terminal["video_id"].astype(str).tolist(),
        "failure_kinds": failed["failure_kind"].fillna("unknown").value_counts().to_dict(),
    }
