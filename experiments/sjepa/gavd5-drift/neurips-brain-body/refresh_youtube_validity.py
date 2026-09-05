"""Create a dated, resumable metadata-availability snapshot for GAVD URLs.

This performs no video download.  A public result means that yt-dlp could
resolve metadata at the stated time and network location; it does not prove
that the annotated frame span decodes or that pose extraction will pass QC.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import pandas as pd
import yt_dlp
from yt_dlp.utils import DownloadError

from evaluation_protocol import CONDITIONS, scan_sequence_manifest, source_table


def classify_error(message: str) -> str:
    lower = message.lower()
    if "private video" in lower:
        return "private"
    if "video is unavailable" in lower or "removed" in lower:
        return "unavailable"
    return "probe_error"


def downloader_options() -> dict[str, object]:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "simulate": True,
        "noplaylist": True,
        "socket_timeout": 20,
        "retries": 1,
        "extractor_retries": 1,
        "js_runtimes": {"node": {}},
    }


def probe(downloader: yt_dlp.YoutubeDL, url: str) -> dict[str, object]:
    try:
        info = downloader.extract_info(url, download=False)
        return {
            "url_status": "public",
            "availability": info.get("availability"),
            "duration_seconds": info.get("duration"),
            "live_status": info.get("live_status"),
            "title": info.get("title"),
            "metadata_fps": info.get("fps"),
            "error": None,
        }
    except DownloadError as exc:
        message = str(exc)
        return {
            "url_status": classify_error(message),
            "availability": None,
            "duration_seconds": None,
            "live_status": None,
            "title": None,
            "metadata_fps": None,
            "error": message,
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data-gavd"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neurips-brain-body/docs/youtube_validity_2026-09-04.csv"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Polite delay between fresh probes (default: 1 second).",
    )
    args = parser.parse_args()

    manifest = scan_sequence_manifest(args.data_root)
    sources = source_table(manifest)
    urls = manifest.groupby("video_id", as_index=False)["url"].first()
    sources = sources.merge(urls, on="video_id", validate="one_to_one")
    done: dict[str, dict[str, object]] = {}
    if args.resume and args.output.exists():
        previous = pd.read_csv(args.output)
        # Rate limits and transient network failures are deliberately retried.
        previous = previous.loc[
            previous["url_status"].isin({"public", "private", "unavailable"})
        ]
        done = previous.set_index("video_id").to_dict("index")

    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    records: list[dict[str, object]] = []
    total = len(sources)
    with yt_dlp.YoutubeDL(downloader_options()) as downloader:
        for number, row in enumerate(sources.itertuples(index=False), start=1):
            if row.video_id in done:
                result = done[row.video_id]
            else:
                print(f"[{number:03d}/{total}] {row.condition} {row.video_id}", flush=True)
                result = probe(downloader, row.url)
                result["checked_at_utc"] = checked_at
                result["yt_dlp_version"] = yt_dlp.version.__version__
                time.sleep(max(args.delay_seconds, 0.0))
            records.append(
                {
                    "condition": row.condition,
                    "video_id": row.video_id,
                    "url": row.url,
                    "sequences": row.sequences,
                    "annotated_frames": row.annotated_frames,
                    "required_last_frame": row.last_frame,
                    **result,
                }
            )
            pd.DataFrame(records).to_csv(args.output, index=False)

    snapshot = pd.DataFrame(records).sort_values(["condition", "video_id"])
    snapshot.to_csv(args.output, index=False)
    summary = (
        snapshot.groupby(["condition", "url_status"], as_index=False)
        .agg(source_videos=("video_id", "nunique"), sequences=("sequences", "sum"))
        .sort_values(["condition", "url_status"])
    )
    summary_path = args.output.with_name(args.output.stem + "_summary.csv")
    summary.to_csv(summary_path, index=False)
    payload = {
        "checked_at_utc": checked_at,
        "yt_dlp_version": yt_dlp.version.__version__,
        "definition": (
            "public means metadata-resolvable only; decoded frame-span and pose QC "
            "are separate downstream gates"
        ),
        "raw_sequences": int(manifest["sequence_id"].nunique()),
        "raw_source_videos": int(manifest["video_id"].nunique()),
        "public_sequences": int(
            snapshot.loc[snapshot["url_status"].eq("public"), "sequences"].sum()
        ),
        "public_source_videos": int(snapshot["url_status"].eq("public").sum()),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
