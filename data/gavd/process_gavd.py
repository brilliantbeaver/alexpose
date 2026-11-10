#!/usr/bin/env python3
"""Losslessly split the five pinned GAVD CSV parts into gait_pat/seq.csv.

All fields remain strings. Parts are read in numeric order, including sequences
that cross part boundaries. Outputs are staged before publication; existing
CSV rows must be a subset of the source, and unexpected CSVs cause refusal.
Publication is atomic per file, not a transaction over the entire directory.
"""

import argparse
from collections import Counter, OrderedDict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
HEADER = 'seq,frame_num,cam_view,gait_event,dataset,gait_pat,bbox,vid_info,id,url'.split(',')
SOURCE_NAMES = [f'GAVD_Clinical_Annotations_{i}.csv' for i in range(1, 6)]
PATTERNS = {'abnormal', 'antalgic', 'cerebral palsy', 'exercise', 'inebriated',
            'myopathic', 'normal', 'parkinsons', 'pregnant', 'prosthetic', 'stroke', 'style'}


def source_files(source_dir):
    """Require the complete source inventory and verify manifest byte digests."""
    manifest = json.loads((source_dir / 'manifest.json').read_text(encoding='utf-8'))
    entries = manifest['files']
    if len(entries) != 5 or {entry['name'] for entry in entries} != set(SOURCE_NAMES):
        raise ValueError('Manifest must contain exactly the five numbered CSV parts')
    actual = {p.relative_to(source_dir).as_posix() for p in source_dir.rglob('*.csv')}
    if actual != set(SOURCE_NAMES):
        raise ValueError('Source CSV inventory differs from the five numbered parts')
    by_name = {entry['name']: entry for entry in entries}
    for name in SOURCE_NAMES:
        path = source_dir / name
        entry = by_name[name]
        content = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(content)).encode('ascii') + b'\0' + content).hexdigest()
        if (len(content) != entry['bytes'] or
                hashlib.sha256(content).hexdigest() != entry['sha256'] or
                blob != entry['git_blob_sha']):
            raise ValueError(f'Source checksum mismatch: {path}')
        yield path, entry


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.reader(stream, strict=True)
        if next(reader, None) != HEADER:
            raise ValueError(f'Incorrect CSV header: {path}')
        for line, row in enumerate(reader, 2):
            if len(row) != len(HEADER):
                raise ValueError(f'{path}:{line}: expected ten fields')
            yield tuple(row)


def process(source_dir, output_dir):
    source_dir, output_dir = source_dir.resolve(), output_dir.resolve()
    if source_dir == output_dir or output_dir.is_relative_to(source_dir):
        raise ValueError('Output must not be inside the source directory')
    output_dir.mkdir(parents=True, exist_ok=True)
    # The staging directory is outside the output tree to keep discovery unambiguous.
    with tempfile.TemporaryDirectory(prefix='.gavd-stage-', dir=output_dir.parent) as temp:
        stage = Path(temp)
        handles = OrderedDict()
        paths = set()
        seq_patterns = {}
        keys = set()
        total_rows = 0
        try:
            for path, entry in source_files(source_dir):
                count = 0
                for row in rows(path):
                    seq, frame, pattern = row[0], row[1], row[5]
                    if not re.fullmatch(r'[A-Za-z0-9_-]+', seq) or pattern not in PATTERNS:
                        raise ValueError(f'Unsafe/unknown sequence or gait pattern in {path}: {seq!r}, {pattern!r}')
                    if not frame.isdecimal():
                        raise ValueError(f'Invalid frame number in {path}: {frame!r}')
                    if seq_patterns.setdefault(seq, pattern) != pattern:
                        raise ValueError(f'Sequence {seq} has multiple gait labels')
                    key = (seq, int(frame))
                    if key in keys:
                        raise ValueError(f'Duplicate sequence/frame key: {key}')
                    keys.add(key)
                    relative = Path(pattern) / f'{seq}.csv'
                    if relative in handles:
                        stream, writer = handles.pop(relative)
                    else:
                        staged_path = stage / relative
                        staged_path.parent.mkdir(parents=True, exist_ok=True)
                        stream = staged_path.open('a', encoding='utf-8', newline='')
                        writer = csv.writer(stream, lineterminator="\n")
                        if relative not in paths:
                            writer.writerow(HEADER)
                            paths.add(relative)
                    handles[relative] = (stream, writer)
                    writer.writerow(row)
                    if len(handles) > 32:
                        _, (old_stream, _) = handles.popitem(last=False)
                        old_stream.close()
                    count += 1
                if count != entry['rows']:
                    raise ValueError(f'Source row count mismatch: {path}')
                total_rows += count
                print(f'Read {path.name}: {count:,} rows')
        finally:
            for stream, _ in handles.values():
                stream.close()
        if not total_rows:
            raise ValueError('No source rows')
        # Validate every existing CSV before changing any annotation files.
        existing = set()
        for path in output_dir.rglob('*.csv'):
            if path.is_relative_to(source_dir):
                continue
            relative = path.relative_to(output_dir)
            if relative not in paths:
                raise ValueError(f'Unexpected CSV (move it outside the output tree): {path}')
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError(f'Refusing symlink output: {path}')
            if Counter(rows(path)) - Counter(rows(stage / relative)):
                raise ValueError(f'Existing CSV has rows absent from source; preserve/reconcile it first: {path}')
            existing.add(relative)
        for relative in paths:
            if (output_dir / relative.parent).is_symlink():
                raise ValueError(f'Refusing symlink category: {relative.parent}')
        changed = 0
        for relative in sorted(paths):
            destination = output_dir / relative
            staged_path = stage / relative
            if destination.exists() and destination.read_bytes() == staged_path.read_bytes():
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staged_path, destination)
            changed += 1
        print(f'Complete: {len(paths):,} sequences, {len(set(seq_patterns.values()))} gait patterns, '
              f'{total_rows:,} rows; {len(paths - existing):,} new CSVs, {changed:,} written')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'source')
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        process(args.source_dir, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as exc:
        print(f'Processing failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
