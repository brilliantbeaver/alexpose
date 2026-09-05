from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile

import cv2
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "neurips-brain-body" / "youtube_download.py"
SPEC = importlib.util.spec_from_file_location("youtube_download", MODULE_PATH)
youtube_download = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(youtube_download)


def test_error_classification_distinguishes_retryable_and_terminal():
    assert youtube_download.classify_download_error(
        "Sign in to confirm you're not a bot"
    ) == ("anti_bot", True)
    assert youtube_download.classify_download_error(
        "Requested format is not available"
    ) == ("format_unavailable", True)
    assert youtube_download.classify_download_error(
        "Private video"
    ) == ("private", False)
    assert youtube_download.classify_download_error(
        "Video unavailable"
    ) == ("client_unavailable", True)
    assert youtube_download.classify_download_error(
        "This video is unavailable"
    ) == ("unavailable", False)


def test_short_valid_media_is_terminal_attrition():
    with tempfile.TemporaryDirectory(dir=ROOT / "work" / "cache") as folder:
        path = Path(folder) / "short.avi"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48)
        )
        assert writer.isOpened()
        for value in range(8):
            writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
        writer.release()
        valid = youtube_download.validate_video(path, required_last_frame=6)
        short = youtube_download.validate_video(path, required_last_frame=12)
        assert valid["ok"] is True
        assert short["ok"] is False
        assert short["failure_kind"] == "annotation_span_too_short"
        assert short["retry_recommended"] is False


def test_failure_summary_keeps_retryable_sources_separate():
    report = pd.DataFrame(
        [
            {"video_id": "short", "ok": False, "retry_recommended": False, "failure_kind": "annotation_span_too_short"},
            {"video_id": "network", "ok": False, "retry_recommended": True, "failure_kind": "timeout"},
            {"video_id": "good", "ok": True, "retry_recommended": False, "failure_kind": None},
        ]
    )
    summary = youtube_download.failure_summary(report)
    assert summary["terminal_exclusions"] == ["short"]
    assert summary["retryable_sources"] == ["network"]


def test_valid_cached_source_needs_zero_download_attempts():
    cache_root = ROOT / "work" / "cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cache_root) as folder:
        youtube_dir = Path(folder)
        condition_dir = youtube_dir / "normal"
        condition_dir.mkdir()
        path = condition_dir / "abcdefghijk.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
        )
        assert writer.isOpened()
        for value in range(8):
            writer.write(np.full((48, 64, 3), value * 20, dtype=np.uint8))
        writer.release()
        table = pd.DataFrame(
            [{
                "condition": "normal",
                "video_id": "abcdefghijk",
                "url": "https://www.youtube.com/watch?v=abcdefghijk",
                "last_frame": 6,
                "source_height": 480,
            }]
        )
        report = youtube_download.download_unique_videos(
            table, youtube_dir, retry_cooldown_seconds=0
        )
        assert report.loc[0, "status"] == "cached"
        assert report.loc[0, "attempt_count"] == 0
