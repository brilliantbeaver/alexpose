#!/usr/bin/env python3
"""Independently verify GAVD clips against CSV ranges and decoded source pixels.

Requires ffmpeg and ffprobe, but no Python packages. This verifier deliberately
does not trust the preparation manifest or its recorded frame counts. It reads
the category CSVs anew and decodes source media and clips independently. Pixel
checks require losslessly encoded clips in the canonical yuv420p pixel format.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("cerebral palsy", "myopathic", "normal", "parkinsons", "stroke")
HEADER = ["seq", "frame_num", "cam_view", "gait_event", "dataset", "gait_pat", "bbox", "vid_info", "id", "url"]
MEDIA_SUFFIXES = (".mp4", ".mkv", ".webm", ".mov")


@dataclass(frozen=True)
class Sequence:
    csv_path: Path
    category: str
    sequence_id: str
    video_id: str
    start: int
    end: int
    annotation_rows: int
    annotation_width: int
    annotation_height: int
    csv_sha256: str

    @property
    def filename(self) -> str:
        return f"{self.video_id}-{self.start}-{self.end}.mp4"

    @property
    def count(self) -> int:
        return self.end - self.start + 1


@dataclass
class Decoded:
    hashes: list[str]
    times: list[Fraction]
    dimensions: tuple[int, int]


def discover_sequences(annotation_root: Path, categories: tuple[str, ...]) -> list[Sequence]:
    """Validate every CSV independently, including contiguity and identities."""
    sequences = []
    output_keys: set[tuple[str, str]] = set()
    for category in categories:
        directory = annotation_root / category
        paths = sorted(directory.glob("*.csv"))
        if not paths:
            raise ValueError(f"No sequence CSVs in {directory}")
        for path in paths:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                if reader.fieldnames != HEADER:
                    raise ValueError(f"Unexpected CSV header: {path}")
                rows = list(reader)
            if not rows:
                raise ValueError(f"Empty sequence CSV: {path}")
            if any(None in row or any(value is None for value in row.values()) for row in rows):
                raise ValueError(f"Malformed CSV row: {path}")
            for key in ("seq", "id", "url", "gait_pat", "vid_info"):
                if len({row[key] for row in rows}) != 1:
                    raise ValueError(f"Inconsistent {key}: {path}")
            first = rows[0]
            if first["seq"] != path.stem or first["gait_pat"] != category:
                raise ValueError(f"CSV filename or gait_pat disagrees with rows: {path}")
            if not re.fullmatch(r"[A-Za-z0-9_-]{11}", first["id"]):
                raise ValueError(f"Invalid YouTube ID: {path}")
            frames = [int(row["frame_num"]) for row in rows]
            if frames[0] < 1 or frames != list(range(frames[0], frames[-1] + 1)):
                raise ValueError(f"Sequence frames are not positive, ordered and contiguous: {path}")
            info = ast.literal_eval(first["vid_info"])
            sequence = Sequence(path, category, path.stem, first["id"], frames[0], frames[-1],
                                len(rows), int(info["width"]), int(info["height"]),
                                hashlib.sha256(path.read_bytes()).hexdigest())
            key = (category, sequence.filename)
            if key in output_keys:
                raise ValueError(f"Two CSVs map to the same output: {category}/{sequence.filename}")
            output_keys.add(key)
            sequences.append(sequence)
    return sequences


def decode_hashes(path: Path, timeout: int = 3600) -> Decoded:
    """Fully decode every video frame, preserving order and original PTS ticks."""
    command = ["ffmpeg", "-v", "error", "-xerror", "-threads", "1", "-i", str(path),
               "-map", "0:v:0", "-an", "-sn", "-dn", "-pix_fmt", "yuv420p", "-c:v", "rawvideo",
               "-threads", "1", "-fps_mode", "passthrough", "-enc_time_base:v", "demux",
               "-f", "framemd5", "-"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if result.returncode or result.stderr.strip():
        raise ValueError(f"Decode error ({result.returncode}) for {path}: {result.stderr[-2000:]}")
    hashes, times = [], []
    time_base = None
    dimensions = None
    for line in result.stdout.splitlines():
        if line.startswith("#tb 0:"):
            time_base = Fraction(line.split(":", 1)[1].strip())
        elif line.startswith("#dimensions 0:"):
            dimensions = tuple(int(value) for value in line.split(":", 1)[1].strip().split("x"))
        elif line and not line.startswith("#"):
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 6 or fields[0] != "0" or time_base is None:
                raise ValueError(f"Unexpected framemd5 record for {path}: {line}")
            times.append(int(fields[2]) * time_base)
            hashes.append(fields[5])
    if not hashes or dimensions is None:
        raise ValueError(f"No decoded video frames: {path}")
    return Decoded(hashes, times, dimensions)


def probe(path: Path) -> dict[str, Any]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "stream=index,codec_type,codec_name,width,height,pix_fmt,avg_frame_rate,time_base,start_time",
                             "-of", "json", str(path)], capture_output=True, text=True, timeout=60, check=True)
    if result.stderr.strip():
        raise ValueError(f"ffprobe error for {path}: {result.stderr[-1000:]}")
    return json.loads(result.stdout)


def compare_decoded(source: Decoded, output: Decoded, start: int, end: int) -> list[str]:
    """Return failures, including equal-length clips shifted by a single frame."""
    issues = []
    expected_count = end - start + 1
    if len(output.hashes) != expected_count:
        issues.append(f"Decoded frame count {len(output.hashes)} != expected {expected_count}")
    if end > len(source.hashes):
        issues.append(f"End frame {end} exceeds source decoded frame count {len(source.hashes)}")
    expected_hashes = source.hashes[start - 1:end]
    if output.hashes != expected_hashes:
        mismatches = sum(left != right for left, right in zip(output.hashes, expected_hashes))
        mismatches += abs(len(output.hashes) - len(expected_hashes))
        issues.append(f"Pixel hashes differ in {mismatches} frames from source slice [{start - 1}:{end}]")
    if output.dimensions != source.dimensions:
        issues.append(f"Output dimensions {output.dimensions} != source {source.dimensions}")
    if any(right <= left for left, right in zip(output.times, output.times[1:])):
        issues.append("Output presentation timestamps are not strictly increasing")
    if output.times and abs(output.times[0]) > Fraction(1, 1000):
        issues.append(f"Output first frame does not start near zero: {float(output.times[0]):.9f}s")
    if end <= len(source.times) and len(output.times) == expected_count:
        selected = source.times[start - 1:end]
        # MP4 may quantize timestamps to its own timescale. One millisecond
        # bounds muxer rounding while catching changes to gait playback speed.
        relative = [value - selected[0] for value in selected]
        if any(abs(actual - expected) > Fraction(1, 1000)
               for actual, expected in zip(output.times, relative)):
            issues.append("Relative output frame times differ from source by more than 1 ms")
    return issues


def verify_source_group(key: tuple[str, str], sequences: list[Sequence], source_root: Path,
                        output_root: Path, timeout: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    category, video_id = key
    candidates = [source_root / category / (video_id + suffix) for suffix in MEDIA_SUFFIXES]
    candidates = [path for path in candidates if path.is_file()]
    source_record: dict[str, Any] = {"gait_pat": category, "video_id": video_id,
                                   "expected_sequences": len(sequences)}
    source = None
    if len(candidates) == 1:
        source_path = candidates[0]
        source_record["path"] = str(source_path)
        try:
            source = decode_hashes(source_path, timeout)
            source_record.update(status="verified_decode", decoded_frames=len(source.hashes),
                                 width=source.dimensions[0], height=source.dimensions[1],
                                 bytes=source_path.stat().st_size, media=probe(source_path))
        except (ValueError, OSError, subprocess.SubprocessError) as error:
            source_record.update(status="invalid_source", error=str(error))
    elif not candidates:
        source_record.update(status="missing_source")
    else:
        source_record.update(status="ambiguous_source", candidates=[str(path) for path in candidates])
    records = []
    for sequence in sequences:
        output_path = output_root / category / sequence.filename
        record: dict[str, Any] = {"sequence_id": sequence.sequence_id, "gait_pat": category,
                                  "video_id": video_id, "csv": str(sequence.csv_path),
                                  "csv_sha256": sequence.csv_sha256, "output": str(output_path),
                                  "start_frame": sequence.start, "end_frame": sequence.end,
                                  "expected_frames": sequence.count,
                                  "annotation_dimensions": [sequence.annotation_width, sequence.annotation_height]}
        if not output_path.is_file():
            record["status"] = "missing_output"
        else:
            try:
                output = decode_hashes(output_path, timeout)
                metadata = probe(output_path)
                issues = []
                if source is None:
                    issues.append("Cannot establish frame alignment without exactly one decodable source")
                else:
                    issues.extend(compare_decoded(source, output, sequence.start, sequence.end))
                if len(metadata.get("streams", [])) != 1:
                    issues.append("Clip must contain exactly one video stream and no other streams")
                if metadata.get("streams", [{}])[0].get("codec_type") != "video":
                    issues.append("Clip first stream is not video")
                record.update(status="verified" if not issues else "invalid_output", issues=issues,
                              decoded_frames=len(output.hashes), width=output.dimensions[0],
                              height=output.dimensions[1], bytes=output_path.stat().st_size,
                              first_frame_md5=output.hashes[0], last_frame_md5=output.hashes[-1],
                              first_pts_seconds=float(output.times[0]),
                              last_pts_seconds=float(output.times[-1]))
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                record.update(status="invalid_output", issues=[str(error)])
        records.append(record)
    return source_record, records


def verify(annotation_root: Path, source_root: Path, output_root: Path,
           categories: tuple[str, ...] = CATEGORIES, jobs: int = 2, timeout: int = 3600) -> dict[str, Any]:
    sequences = discover_sequences(annotation_root, categories)
    grouped: dict[tuple[str, str], list[Sequence]] = defaultdict(list)
    expected_paths = set()
    for sequence in sequences:
        grouped[(sequence.category, sequence.video_id)].append(sequence)
        expected_paths.add(output_root / sequence.category / sequence.filename)
    extras = sorted(str(path) for category in categories for path in (output_root / category).glob("*.mp4")
                    if path not in expected_paths)
    report: dict[str, Any] = {"schema_version": 1, "started_at": datetime.now(timezone.utc).isoformat(),
                              "method": "Independent CSV inventory; complete ffmpeg yuv420p framemd5 decode; every output hash compared to source[start-1:end]; dimensions and presentation timing checked",
                              "categories": list(categories), "sources": [], "sequences": [],
                              "unexpected_outputs": extras}
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(verify_source_group, key, group, source_root, output_root, timeout): key
                   for key, group in grouped.items()}
        for future in as_completed(futures):
            source_record, records = future.result()
            report["sources"].append(source_record)
            report["sequences"].extend(records)
            print(f"{source_record['gait_pat']}/{source_record['video_id']}: "
                  f"{dict(Counter(row['status'] for row in records))}", flush=True)
    report["sources"].sort(key=lambda row: (row["gait_pat"], row["video_id"]))
    report["sequences"].sort(key=lambda row: (row["gait_pat"], row["sequence_id"]))
    statuses = Counter(row["status"] for row in report["sequences"])
    report["summary"] = {"expected_sources": len(grouped), "expected_sequences": len(sequences),
                          "expected_frames": sum(sequence.count for sequence in sequences),
                          "sequence_statuses": dict(statuses),
                          "source_statuses": dict(Counter(row["status"] for row in report["sources"])),
                          "verified_frames": sum(row["expected_frames"] for row in report["sequences"]
                                                 if row["status"] == "verified"),
                          "unexpected_outputs": len(extras),
                          "by_category": {category: {
                              "expected_sequences": sum(seq.category == category for seq in sequences),
                              "expected_frames": sum(seq.count for seq in sequences if seq.category == category),
                              "statuses": dict(Counter(row["status"] for row in report["sequences"]
                                                       if row["gait_pat"] == category))}
                              for category in categories}}
    report["complete"] = statuses.get("verified", 0) == len(sequences) and not extras
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", type=Path, default=PROJECT_ROOT / "data/gavd")
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data/videos-gavd")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data/videos-ms/gavd-sequences")
    parser.add_argument("--report", type=Path, help="Default: <output-root>/verification.json")
    parser.add_argument("--category", action="append", choices=CATEGORIES)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--decode-timeout", type=int, default=3600, help="Seconds allowed per complete decode")
    args = parser.parse_args()
    if args.jobs < 1 or args.decode_timeout < 1:
        parser.error("jobs and decode-timeout must be positive")
    try:
        report = verify(args.annotation_root.resolve(), args.source_root.resolve(), args.output_root.resolve(),
                        tuple(args.category) if args.category else CATEGORIES, args.jobs, args.decode_timeout)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        return 2
    target = args.report or args.output_root / "verification.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(target)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {target}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
