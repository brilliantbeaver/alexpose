#!/usr/bin/env python3
"""Verify the complete GAVD CSV derivative against its downloaded source snapshot.

Every parsed field (including empty strings), row multiplicity, header, sequence
filename, and literal gait-pattern directory is checked. A temporary on-disk
SQLite database keeps exact row comparisons from retaining the dataset in RAM.
This verifies annotation references, not the availability of YouTube videos.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

HEADER = "seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url".split(",")
SOURCE_NAMES = {f"GAVD_Clinical_Annotations_{number}.csv" for number in range(1, 6)}
DEFAULT_OUTPUT = Path(__file__).resolve().parent


def read_rows(path, issues):
    """Yield correctly shaped records; record every schema failure."""
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != HEADER:
            issues.append(f"Header mismatch: {path}")
        for line, row in enumerate(reader, 2):
            if len(row) != len(HEADER):
                issues.append(f"Expected 10 fields at {path}:{line}; got {len(row)}")
                continue
            yield row


def check_manifest(source_dir, source_paths, issues):
    """Check independently computed byte hashes against recorded provenance."""
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.is_file():
        issues.append(f"Source manifest missing: {manifest_path}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Source manifest must be a JSON object")
    commit = manifest.get("commit", manifest.get("upstream_commit", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        issues.append("Source manifest must record a 40-character upstream commit")
    files = manifest.get("files", [])
    if isinstance(files, dict):
        entries = {name: entry for name, entry in files.items()}
        if not all(isinstance(entry, dict) for entry in entries.values()):
            raise ValueError("Source manifest file entries must be JSON objects")
    else:
        if not isinstance(files, list) or not all(isinstance(entry, dict) for entry in files):
            raise ValueError("Source manifest files must be a list of JSON objects")
        entries = {entry.get("name", entry.get("path", "")): entry for entry in files}
        if len(entries) != len(files):
            issues.append("Duplicate file names in source manifest")
    if set(entries) != SOURCE_NAMES:
        issues.append("Source manifest does not list exactly the five upstream CSV parts")
    for path in source_paths:
        entry = entries.get(path.name)
        if entry is None:
            continue
        sha256 = hashlib.sha256()
        git_blob = hashlib.sha1(f"blob {path.stat().st_size}\0".encode("ascii"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                sha256.update(block)
                git_blob.update(block)
        if entry.get("sha256") != sha256.hexdigest():
            issues.append(f"Source SHA256 mismatch: {path.name}")
        expected_blob = entry.get("git_blob_sha", entry.get("git_blob_sha1", entry.get("sha")))
        if expected_blob is not None and expected_blob != git_blob.hexdigest():
            issues.append(f"Source Git blob mismatch: {path.name}")
        expected_size = entry.get("bytes", entry.get("size"))
        if expected_size is not None and expected_size != path.stat().st_size:
            issues.append(f"Source byte-size mismatch: {path.name}")
    return manifest


def verify_processing(source_dir=None, output_dir=None):
    """Return a JSON-compatible, exhaustive annotation comparison report."""
    output_dir = Path(output_dir or DEFAULT_OUTPUT).resolve()
    source_dir = Path(source_dir or output_dir / "source").resolve()
    issues = []
    source_paths = sorted(source_dir.glob("*.csv"))
    source_inventory = {path.relative_to(source_dir).as_posix() for path in source_dir.rglob("*.csv")}
    if source_inventory != SOURCE_NAMES:
        issues.append("Source directory must contain exactly the five numbered upstream CSVs")
    manifest = check_manifest(source_dir, source_paths, issues)
    # A recursive inventory catches stale duplicate directories and misplaced CSVs.
    output_paths = sorted(path for path in output_dir.rglob("*.csv")
                          if not path.is_relative_to(source_dir))
    if not output_paths:
        issues.append("No transformed CSV files found")
    source_stats = []
    categories = defaultdict(lambda: {"sequences": 0, "rows": 0})
    sequences = {}
    videos = set()
    source_total = local_total = 0
    with tempfile.TemporaryDirectory(prefix="gavd-verify-") as temp_dir:
        database = sqlite3.connect(str(Path(temp_dir) / "rows.sqlite3"))
        database.execute("PRAGMA journal_mode=OFF")
        database.execute("PRAGMA synchronous=OFF")
        database.execute("CREATE TABLE records (record TEXT PRIMARY KEY, seq TEXT, frame TEXT, "
                         "source_count INTEGER NOT NULL, local_count INTEGER NOT NULL)")
        insert_source = ("INSERT INTO records VALUES (?, ?, ?, 1, 0) "
                         "ON CONFLICT(record) DO UPDATE SET source_count=source_count+1")
        insert_local = ("INSERT INTO records VALUES (?, ?, ?, 0, 1) "
                        "ON CONFLICT(record) DO UPDATE SET local_count=local_count+1")
        for path in source_paths:
            count = 0
            part_seqs = set()
            with database:
                for row in read_rows(path, issues):
                    database.execute(insert_source, (json.dumps(row, ensure_ascii=False, separators=(",", ":")), row[0], row[1]))
                    count += 1
                    part_seqs.add(row[0])
                    if row[0] in sequences and sequences[row[0]] != row[5]:
                        issues.append(f"Source sequence has conflicting gait patterns: {row[0]}")
                    sequences[row[0]] = row[5]
                    videos.add(row[8])
            source_stats.append({"name": path.name, "rows": count, "sequences": len(part_seqs)})
            if not count:
                issues.append(f"Empty source CSV: {path.name}")
            source_total += count
        if manifest:
            entries = manifest.get("files", [])
            if isinstance(entries, list):
                entries = {entry.get("name", entry.get("path", "")): entry for entry in entries}
            for stat in source_stats:
                expected = entries.get(stat["name"], {}).get("rows")
                if expected is not None and expected != stat["rows"]:
                    issues.append(f"Source manifest row-count mismatch: {stat['name']}")
        seen_seqs = Counter()
        with database:
            for path in output_paths:
                relative = path.relative_to(output_dir)
                if len(relative.parts) != 2:
                    issues.append(f"Transformed CSV is outside category/<seq>.csv: {relative}")
                seen_seqs[path.stem] += 1
                count = 0
                for row in read_rows(path, issues):
                    database.execute(insert_local, (json.dumps(row, ensure_ascii=False, separators=(",", ":")), row[0], row[1]))
                    count += 1
                    if row[0] != path.stem:
                        issues.append(f"Sequence does not match filename: {relative}, seq={row[0]}")
                    if row[5] != path.parent.name:
                        issues.append(f"Gait label does not match directory: {relative}, gait_pat={row[5]!r}")
                if not count:
                    issues.append(f"Empty transformed CSV: {relative}")
                categories[path.parent.name]["sequences"] += 1
                categories[path.parent.name]["rows"] += count
                local_total += count
        for seq, count in seen_seqs.items():
            if count != 1:
                issues.append(f"Sequence appears in {count} transformed CSV files: {seq}")
        if set(seen_seqs) != set(sequences):
            issues.append(f"Sequence inventory mismatch: missing={len(set(sequences)-set(seen_seqs))}, "
                          f"extra={len(set(seen_seqs)-set(sequences))}")
        missing, extra = database.execute(
            "SELECT COALESCE(SUM(MAX(source_count-local_count,0)),0), "
            "COALESCE(SUM(MAX(local_count-source_count,0)),0) FROM records").fetchone()
        duplicate_records = {}
        duplicate_keys = {}
        for label, column in (("source", "source_count"), ("local", "local_count")):
            duplicate_records[label] = database.execute(
                f"SELECT COALESCE(SUM(MAX({column}-1,0)),0) FROM records").fetchone()[0]
            duplicate_keys[label] = database.execute(
                f"SELECT COUNT(*) FROM (SELECT seq,frame FROM records GROUP BY seq,frame "
                f"HAVING SUM({column})>1)").fetchone()[0]
            if duplicate_records[label] or duplicate_keys[label]:
                issues.append(f"Duplicate {label} records={duplicate_records[label]}, "
                              f"duplicate (seq, frame_num) keys={duplicate_keys[label]}")
        database.close()
    if missing or extra:
        issues.append(f"Exact row multiset mismatch: missing={missing}, extra={extra}")
    return {
        "success": not issues,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "upstream_commit": manifest.get("commit", manifest.get("upstream_commit")) if manifest else None,
        "comparison": "all ten parsed string fields, including empty values and row multiplicities",
        "source_parts": source_stats,
        "source_rows": source_total,
        "local_rows": local_total,
        "local_csvs": len(output_paths),
        "source_sequences": len(sequences),
        "source_video_ids": len(videos),
        "categories": dict(sorted(categories.items())),
        "missing_rows": missing,
        "extra_rows": extra,
        "duplicate_records": duplicate_records,
        "duplicate_seq_frame_keys": duplicate_keys,
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_OUTPUT / "source")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, help="Write the complete JSON verification report")
    args = parser.parse_args()
    try:
        report = verify_processing(args.source_dir, args.output_dir)
    except (OSError, ValueError, csv.Error, sqlite3.Error, TypeError, KeyError) as exc:
        report = {"success": False, "issues": [f"Verification could not complete: {exc}"]}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
