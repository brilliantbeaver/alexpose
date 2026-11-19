#!/usr/bin/env python3
"""Acquire five requested GAVD categories and make exact inclusive video clips.

Run from the repository root with an environment providing yt-dlp and loguru.
Uses Ambient's shared yt-dlp configuration; no study packages are imported.
Full sources are retained. Outputs are lossless relative to decoded yuv420p
source pixels, permitting independent, exhaustive frame-hash verification.
"""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("cerebral palsy", "myopathic", "normal", "parkinsons", "stroke")
HEADER = ["seq", "frame_num", "cam_view", "gait_event", "dataset", "gait_pat",
          "bbox", "vid_info", "id", "url"]
MEDIA_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".m4v")
PRINT_LOCK = threading.Lock()


def now():
    return datetime.now(timezone.utc).isoformat()


def say(message):
    with PRINT_LOCK:
        print(message, flush=True)


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def inventory(annotation_root, output_root):
    sequences, sources = [], {}
    names = set()
    for category in CATEGORIES:
        paths = sorted((annotation_root / category).glob("*.csv"))
        if not paths:
            raise ValueError(f"No annotation CSVs: {category}")
        for path in paths:
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != HEADER:
                    raise ValueError(f"Unexpected header: {path}")
                rows = list(reader)
            if not rows:
                raise ValueError(f"Empty annotation CSV: {path}")
            first = rows[0]
            video_info = ast.literal_eval(first["vid_info"])
            for field in ("seq", "id", "url", "gait_pat", "vid_info"):
                if len({row[field] for row in rows}) != 1:
                    raise ValueError(f"Inconsistent {field}: {path}")
            if first["seq"] != path.stem or first["gait_pat"] != category:
                raise ValueError(f"CSV identity mismatch: {path}")
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", first["id"]):
                raise ValueError(f"Invalid YouTube ID: {path}")
            frames = [int(row["frame_num"]) for row in rows]
            start, end = frames[0], frames[-1]
            if start < 1 or frames != list(range(start, end + 1)):
                raise ValueError(f"Frames must be positive, ordered, contiguous: {path}")
            name = f"{first['id']}-{start}-{end}.mp4"
            key = (category, name)
            if key in names:
                raise ValueError(f"Output name collision: {key}")
            names.add(key)
            seq = dict(sequence_id=path.stem, gait_pat=category, video_id=first["id"],
                       annotation_csv=relative(path), annotation_sha256=sha256(path),
                       start_frame=start, end_frame=end, expected_frames=len(frames),
                       annotation_video_info=video_info,
                       output_path=relative(output_root / category / name), status="pending")
            sequences.append(seq)
            source_key = f"{category}/{first['id']}"
            src = sources.setdefault(source_key, dict(
                source_key=source_key, gait_pat=category, video_id=first["id"],
                url=first["url"], annotation_video_info=video_info, annotation_video_infos=[video_info],
                required_last_frame=0, sequence_ids=[], status="pending"))
            if src["url"] != first["url"]:
                raise ValueError(f"Inconsistent source metadata: {source_key}")
            if video_info not in src["annotation_video_infos"]:
                src["annotation_video_infos"].append(video_info)
                src["annotation_video_info"] = max(src["annotation_video_infos"], key=lambda info: int(info["height"]))
            src["required_last_frame"] = max(src["required_last_frame"], end)
            src["sequence_ids"].append(path.stem)
    return sequences, list(sources.values())


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames,duration,time_base:format=duration",
        "-of", "json", str(path)], capture_output=True, text=True, timeout=1200)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(f"Source/output decode failed: {result.stderr[-3000:]}")
    data = json.loads(result.stdout)
    if len(data.get("streams", [])) != 1:
        raise RuntimeError("No video stream")
    stream = data["streams"][0]
    stream["decoded_frames"] = int(stream.pop("nb_read_frames"))
    if stream["decoded_frames"] <= 0:
        raise RuntimeError("No decoded frames")
    return stream


class DownloadLog:
    def __init__(self):
        self.messages = []

    def debug(self, message):
        pass

    def warning(self, message):
        self.messages.append(str(message))

    def error(self, message):
        self.messages.append(str(message))


def acquire(source, source_root):
    """Reuse validated canonical files or download into a staging directory."""
    from ambient.utils.youtube_cache import YtDlpConfig, _build_ydl_options, _build_output_template
    from yt_dlp import YoutubeDL

    folder = source_root / source["gait_pat"]
    folder.mkdir(parents=True, exist_ok=True)
    candidates = [folder / (source["video_id"] + ext) for ext in MEDIA_EXTENSIONS]
    existing = [path for path in candidates if path.is_file() and path.stat().st_size]
    if len(existing) > 1:
        raise RuntimeError(f"Ambiguous source media: {existing}")
    recovery_attempts = []
    if existing:
        media = existing[0]
        # Existing canonical files must decode; arbitrary files elsewhere are never reused.
        try:
            metadata = probe(media)
            return dict(status="available", acquisition="reused_canonical", path=relative(media),
                        bytes=media.stat().st_size, sha256=sha256(media), probe=metadata, attempts=[])
        except Exception as exc:
            quarantine = folder / ".invalid" / (now().replace(":", "-") + "-" + media.name)
            quarantine.parent.mkdir(parents=True, exist_ok=True)
            media.replace(quarantine)
            recovery_attempts.append(dict(strategy="cached_validation", status="quarantined",
                                          error=str(exc), preserved_path=relative(quarantine)))

    stage = folder / ".acquiring" / source["video_id"]
    stage.mkdir(parents=True, exist_ok=True)
    attempts = recovery_attempts
    # Explicit compatibility policy; the CSV does not establish source FPS.
    # Avoid 50/60 fps alternatives; record the actual selected and decoded rates.
    height = int(source["annotation_video_info"]["height"])
    selector = (f"bestvideo[ext=mp4][height<={height}][fps<=30]/"
                f"bestvideo[height<={height}][fps<=30]/"
                f"best[ext=mp4][height<={height}][fps<=30]/best[height<={height}][fps<=30]")
    for strategy, args in [("default", {}), ("embedded", {"youtube": {"player_client": ["web_embedded"]}}),
                           ("android", {"youtube": {"player_client": ["android"]}})]:
        log = DownloadLog()
        cfg = YtDlpConfig(output_dir=stage, format_selector=selector, max_retries=2,
                          max_concurrent_fragments=2, extractor_args=args)
        options = _build_ydl_options(_build_output_template(stage, "id"), cfg)
        options.update(dict(quiet=True, noprogress=True, noplaylist=True, logger=log,
                            ignoreerrors=False, socket_timeout=30, extractor_retries=1,
                            skip_unavailable_fragments=False, postprocessors=[],
                            remote_components=["ejs:github"]))
        if shutil.which("node"):
            options["js_runtimes"] = {"node": {"path": shutil.which("node")}}
        attempt = dict(strategy=strategy, started_at=now())
        try:
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(source["url"], download=True)
                if not info or info.get("id") != source["video_id"]:
                    raise RuntimeError("Downloaded metadata ID mismatch")
                downloaded = [stage / (source["video_id"] + ext) for ext in MEDIA_EXTENSIONS]
                downloaded = [path for path in downloaded if path.is_file() and path.stat().st_size]
                if len(downloaded) != 1:
                    raise RuntimeError("Expected exactly one complete downloaded source")
                media = downloaded[0]
                metadata = probe(media)
                final = folder / media.name
                media.replace(final)
                attempt.update(status="downloaded", finished_at=now(), messages=log.messages)
                attempts.append(attempt)
                return dict(status="available", acquisition="downloaded", path=relative(final),
                            downloaded_at=now(), bytes=final.stat().st_size, sha256=sha256(final),
                            probe=metadata, selected_format={k: info.get(k) for k in
                                ("id", "title", "format_id", "ext", "width", "height", "fps", "duration", "vcodec", "acodec")},
                            attempts=attempts)
        except Exception as exc:
            attempt.update(status="failed", finished_at=now(), error=str(exc), messages=log.messages)
            attempts.append(attempt)
            # A complete-looking but invalid staged file must not poison retries.
            for suffix in MEDIA_EXTENSIONS:
                invalid = stage / (source["video_id"] + suffix)
                if invalid.exists():
                    quarantine = stage / ".invalid" / (strategy + "-" + invalid.name)
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    invalid.replace(quarantine)
    return dict(status="unavailable", acquisition="failed", attempts=attempts,
                error=attempts[-1]["error"])


def extract(sequence, source, old_sequences):
    output = ROOT / sequence["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = output.parent / ".provenance" / (sequence["sequence_id"] + ".json")
    old = old_sequences.get(sequence["sequence_id"], {})
    if provenance.exists() and old.get("status") != "completed":
        old = json.loads(provenance.read_text())
    if output.exists():
        if (old.get("source_sha256") == source["sha256"] and
                old.get("annotation_sha256") == sequence["annotation_sha256"] and
                old.get("sha256") == sha256(output) and old.get("status") == "completed"):
            return {**sequence, **{k: old[k] for k in ("sha256", "bytes", "probe", "source_sha256")},
                    "status": "completed", "processing": "reused_verified_hash"}
        raise RuntimeError(f"Existing output lacks matching provenance; preserved: {output}")
    if sequence["end_frame"] > source["probe"]["decoded_frames"]:
        raise RuntimeError(f"Source has {source['probe']['decoded_frames']} frames; requires {sequence['end_frame']}")
    temp = output.with_name(output.stem + ".partial.mp4")
    video_filter = (f"trim=start_frame={sequence['start_frame'] - 1}:end_frame={sequence['end_frame']},"
                    "setpts=PTS-STARTPTS")
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-xerror", "-y", "-threads", "2",
               "-i", str(ROOT / source["path"]), "-map", "0:v:0", "-an", "-sn", "-dn",
               "-vf", video_filter, "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "0",
               "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-map_metadata", "-1", "-map_chapters", "-1",
               "-movflags", "+faststart", str(temp)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1200)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    metadata = probe(temp)
    if metadata["decoded_frames"] != sequence["expected_frames"]:
        raise RuntimeError(f"Wrong frame count: {metadata['decoded_frames']} vs {sequence['expected_frames']}")
    record = {**sequence, "status": "completed", "processing": "extracted",
              "source_sha256": source["sha256"], "sha256": sha256(temp),
              "bytes": temp.stat().st_size, "probe": metadata}
    # Persist provenance before publication so an interrupted batch is resumable.
    save_json(provenance, record)
    temp.replace(output)
    return record


def process_source(source, sequences, source_root, old_sources, old_sequences):
    say(f"SOURCE {source['source_key']}: {len(sequences)} sequences")
    try:
        acquired = acquire(source, source_root)
        record = {**source, **acquired}
        if record["status"] == "available":
            record["covers_all_annotations"] = record["probe"]["decoded_frames"] >= source["required_last_frame"]
            record["frame_coverage_shortfall"] = max(0, source["required_last_frame"] - record["probe"]["decoded_frames"])
        old = old_sources.get(source["source_key"], {})
        if record.get("sha256") == old.get("sha256") and old.get("status") == "available":
            for field in ("downloaded_at", "selected_format", "attempts"):
                if field in old:
                    record[field] = old[field]
        save_json(source_root / source["gait_pat"] / ".provenance" / (source["video_id"] + ".json"), record)
        results = []
        for seq in sequences:
            if record["status"] != "available":
                results.append({**seq, "status": "blocked_source", "error": record.get("error")})
                continue
            try:
                results.append(extract(seq, record, old_sequences))
            except Exception as exc:
                results.append({**seq, "status": "extraction_failed", "error": str(exc)})
        say(f"FINISHED {source['source_key']}: {record['status']}; "
            f"{sum(row['status'] == 'completed' for row in results)}/{len(results)} clips")
        return record, results
    except Exception as exc:
        record = {**source, "status": "failed", "error": str(exc)}
        say(f"FAILED {source['source_key']}: {exc}")
        return record, [{**seq, "status": "blocked_source", "error": str(exc)} for seq in sequences]


def summary(manifest):
    result = {}
    for category in (*CATEGORIES, "TOTAL"):
        sequences = [row for row in manifest["sequences"] if category == "TOTAL" or row["gait_pat"] == category]
        sources = [row for row in manifest["sources"] if category == "TOTAL" or row["gait_pat"] == category]
        completed = [row for row in sequences if row["status"] == "completed"]
        result[category] = dict(expected_sources=len(sources), available_sources=sum(row["status"] == "available" for row in sources),
                                expected_sequences=len(sequences), completed_sequences=len(completed),
                                expected_frames=sum(row["expected_frames"] for row in sequences),
                                completed_frames=sum(row["expected_frames"] for row in completed),
                                completed_ml_exclusions=sum(bool(row.get("exclude_from_ml")) for row in completed),
                                source_bytes=sum(row.get("bytes", 0) for row in sources),
                                clip_bytes=sum(row.get("bytes", 0) for row in completed))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=ROOT / "data/gavd")
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/videos-gavd")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/videos-ms/gavd-sequences")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    sequences, sources = inventory(args.annotation_root, args.output_root)
    manifest_path = args.output_root / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    old_sources = {row["source_key"]: row for row in old.get("sources", [])}
    old_sequences = {row["sequence_id"]: row for row in old.get("sequences", [])}
    for source in sources:
        sidecar = args.source_root / source["gait_pat"] / ".provenance" / (source["video_id"] + ".json")
        if old_sources.get(source["source_key"], {}).get("status") != "available" and sidecar.exists():
            old_sources[source["source_key"]] = json.loads(sidecar.read_text())
    manifest = dict(schema_version=1, started_at=now(), completed_at=None,
                    categories=list(CATEGORIES), annotation_root=relative(args.annotation_root),
                    source_root=relative(args.source_root), output_root=relative(args.output_root),
                    frame_convention="GAVD one-based inclusive [start,end]; ffmpeg trim [start-1,end)",
                    encoding="libx264 crf=0 preset=fast yuv420p; full image; no audio; fps_mode=passthrough",
                    format_policy="Best available video at or below annotation height and <=30 nominal fps; no FPS resampling",
                    sources=sources, sequences=sequences)
    if args.inventory_only:
        text = summary(manifest)
        print(json.dumps(text, indent=2))
        return 0
    # Retain completed checkpoint records until their replacements are ready.
    # Never erase provenance merely because a rerun has started.
    for i, sequence in enumerate(sequences):
        previous = old_sequences.get(sequence["sequence_id"], {})
        if previous.get("annotation_sha256") == sequence["annotation_sha256"] and previous.get("status") == "completed":
            sequences[i] = {**previous, **{key: value for key, value in sequence.items() if key != "status"}}
    for i, source in enumerate(sources):
        previous = old_sources.get(source["source_key"], {})
        if previous.get("status") == "available":
            sources[i] = {**previous, **{key: value for key, value in source.items() if key != "status"}}
    if args.workers < 1:
        parser.error("--workers must be positive")
    sys.path.insert(0, str(ROOT))
    import yt_dlp.version
    manifest["tools"] = dict(python=sys.version.split()[0], yt_dlp=yt_dlp.version.__version__,
                             ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[0])
    if shutil.which("node"):
        manifest["tools"]["node"] = subprocess.check_output(["node", "--version"], text=True).strip()
    manifest["summary"] = summary(manifest)
    save_json(manifest_path, manifest)
    source_indexes = {row["source_key"]: i for i, row in enumerate(sources)}
    sequence_indexes = {row["sequence_id"]: i for i, row in enumerate(sequences)}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for source in sources:
            selected = [row for row in sequences if row["sequence_id"] in source["sequence_ids"]]
            futures.append(pool.submit(process_source, source, selected, args.source_root, old_sources, old_sequences))
        for future in as_completed(futures):
            source, finished = future.result()
            manifest["sources"][source_indexes[source["source_key"]]] = source
            for seq in finished:
                manifest["sequences"][sequence_indexes[seq["sequence_id"]]] = seq
            manifest["summary"] = summary(manifest)
            save_json(manifest_path, manifest)
    manifest["completed_at"] = now()
    manifest["summary"] = summary(manifest)
    save_json(manifest_path, manifest)
    save_json(args.source_root / "manifest.json", {key: value for key, value in manifest.items() if key != "sequences"})
    print(json.dumps(manifest["summary"], indent=2))
    return 0 if all(row["status"] == "completed" for row in manifest["sequences"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
