"""Retry the seven transient/anti-bot GAVD YouTube sources exactly once.

This script intentionally does not use browser cookies or an authenticated account.
It records both download success and whether OpenCV can decode the last annotated
frame, because metadata availability alone is not dataset eligibility.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import pandas as pd
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError


RETRY_VIDEO_IDS = (
    "WWS-iOlLsoo",
    "OoCDFmCm1DE",
    "yFBy0X0D-w8",
    "dxRMtNtjwCc",
    "JUMhhwFANKE",
    "n93bgWhLZk4",
    "_-Ubl8iD2B0",
)
VIDEO_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov", ".m4v")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_media(folder: Path, video_id: str) -> Path | None:
    matches = [
        path
        for suffix in VIDEO_SUFFIXES
        for path in folder.glob(f"{video_id}{suffix}")
        if path.is_file() and path.stat().st_size > 0
    ]
    return sorted(matches)[0] if matches else None


def decode_audit(path: Path | None, required_last_frame: int) -> dict[str, object]:
    if path is None:
        return {
            "downloaded": False,
            "decode_opened": False,
            "fps": None,
            "frames": 0,
            "required_last_frame": required_last_frame,
            "last_annotated_frame_decoded": False,
            "decoded_frame_eligible": False,
            "size_mb": 0.0,
            "sha256": None,
        }
    capture = cv2.VideoCapture(str(path))
    opened = bool(capture.isOpened())
    fps = float(capture.get(cv2.CAP_PROP_FPS)) if opened else 0.0
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if opened else 0
    first_ok = False
    last_ok = False
    if opened:
        first_ok, _ = capture.read()
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(required_last_frame - 1, 0))
        last_ok, _ = capture.read()
    capture.release()
    eligible = bool(opened and first_ok and last_ok and fps > 0 and frames >= required_last_frame)
    return {
        "downloaded": True,
        "decode_opened": opened,
        "fps": fps,
        "frames": frames,
        "required_last_frame": required_last_frame,
        "last_annotated_frame_decoded": bool(last_ok),
        "decoded_frame_eligible": eligible,
        "size_mb": round(path.stat().st_size / 1_000_000, 3),
        "sha256": sha256_file(path),
    }


def classify_error(message: str | None) -> str | None:
    if not message:
        return None
    lowered = message.lower()
    if "sign in to confirm" in lowered or "not a bot" in lowered:
        return "anti_bot"
    if "too many requests" in lowered or "http error 429" in lowered:
        return "rate_limited"
    if "private video" in lowered:
        return "private"
    if "video is unavailable" in lowered:
        return "unavailable"
    return "download_error"


def ytdlp_options(folder: Path) -> dict[str, object]:
    if node_path := shutil.which("node"):
        js_runtimes = {"node": {"path": node_path}}
    elif shutil.which("deno"):
        js_runtimes = {"deno": {}}
    else:
        raise RuntimeError("yt-dlp YouTube challenge solving requires Node or Deno.")
    return {
        "outtmpl": str(folder / "%(id)s.%(ext)s"),
        "format": (
            "bestvideo[ext=mp4][vcodec^=avc1][height<=720][fps<=30]/"
            "bestvideo[ext=mp4][height<=720][fps<=30]/"
            "bestvideo[height<=720][fps<=30]/"
            "best[ext=mp4][vcodec!=none][height<=720][fps<=30]/"
            "best[vcodec!=none][height<=720][fps<=30]"
        ),
        "noplaylist": True,
        "writeinfojson": True,
        "continuedl": True,
        "overwrites": False,
        "socket_timeout": 30,
        # The requested retry is one extractor/download attempt per source.
        "retries": 0,
        "fragment_retries": 0,
        "extractor_retries": 0,
        "file_access_retries": 0,
        "js_runtimes": js_runtimes,
        "extractor_args": {
            "youtube": {"player_client": ["web_embedded", "tv_downgraded"]}
        },
        "quiet": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-seconds", type=float, default=5.0)
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("work/artifacts/real/evaluation_protocol/video_download_audit.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurips-brain-body/docs/youtube_antibot_retry_2026-09-04.csv"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    audit_path = (root / args.audit).resolve()
    output_path = (root / args.output).resolve()
    media_root = root / "work" / "youtube"
    audit = pd.read_csv(audit_path)
    targets = audit.loc[audit["video_id"].astype(str).isin(RETRY_VIDEO_IDS)].copy()
    missing_ids = set(RETRY_VIDEO_IDS) - set(targets["video_id"].astype(str))
    if missing_ids:
        raise ValueError(f"Retry IDs absent from download audit: {sorted(missing_ids)}")
    targets = targets.set_index("video_id").loc[list(RETRY_VIDEO_IDS)].reset_index()

    rows: list[dict[str, object]] = []
    for index, row in enumerate(targets.itertuples(index=False)):
        folder = media_root / str(row.condition)
        folder.mkdir(parents=True, exist_ok=True)
        before = find_media(folder, str(row.video_id))
        attempted = before is None
        error: str | None = None
        info: dict[str, object] | None = None
        started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        if attempted:
            try:
                with YoutubeDL(ytdlp_options(folder)) as downloader:
                    extracted = downloader.extract_info(str(row.url), download=True)
                info = extracted if isinstance(extracted, dict) else None
            except DownloadError as exc:
                error = str(exc)
        path = find_media(folder, str(row.video_id))
        decoded = decode_audit(path, int(row.required_last_frame))
        rows.append(
            {
                "checked_at_utc": started,
                "condition": str(row.condition),
                "video_id": str(row.video_id),
                "url": str(row.url),
                "retry_attempted": attempted,
                "status": (
                    "cached"
                    if not attempted
                    else "downloaded"
                    if decoded["downloaded"]
                    else "failed"
                ),
                "error_class": classify_error(error),
                "error": error,
                "path": str(path) if path else str(folder / f"{row.video_id}.mp4"),
                "format_id": info.get("format_id") if info else None,
                "format_note": info.get("format_note") if info else None,
                **decoded,
            }
        )
        print(
            f"{row.video_id}: {rows[-1]['status']}; "
            f"frames={decoded['frames']}; required={row.required_last_frame}; "
            f"eligible={decoded['decoded_frame_eligible']}"
        )
        if index + 1 < len(targets) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    result = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    summary = {
        "checked_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "retry_scope": list(RETRY_VIDEO_IDS),
        "attempted": int(result["retry_attempted"].sum()),
        "downloaded": int(result["downloaded"].sum()),
        "decoded_frame_eligible": int(result["decoded_frame_eligible"].sum()),
        "failed": int((result["status"] == "failed").sum()),
        "error_classes": result["error_class"].fillna("none").value_counts().to_dict(),
        "authentication": "none; no browser cookies",
        "attempt_policy": "one yt-dlp extractor/download attempt per missing source",
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
