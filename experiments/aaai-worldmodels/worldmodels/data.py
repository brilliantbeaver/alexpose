"""Fresh GAVD annotation audit and fixed source-disjoint partitions.

No cached inventory is used to decide which CSVs exist. Media manifests supply
only provenance and previously established exclusions, never clinical truth.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

LABELS = ('normal', 'parkinsons', 'stroke', 'cerebral palsy', 'myopathic')
JOINT_IDS = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
JOINTS = JOINT_IDS
CSV_FIELDS = ('seq', 'frame_num', 'cam_view', 'gait_event', 'dataset', 'gait_pat', 'bbox', 'vid_info', 'id', 'url')


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def find_video(repo_root, source_id, label=None):
    """Reuse existing category layout; new downloads use sources/, once per ID."""
    root = Path(repo_root) / 'data/videos-gavd'
    directories = [root / 'sources', root]
    if label:
        directories.append(root / label)
    directories.extend(root / c for c in LABELS if c != label)
    candidates = [p for d in directories for ext in ('.mp4', '.mkv', '.webm', '.mov')
                  if (p := d / (source_id + ext)).is_file()]
    if len(candidates) > 1:
        fingerprints = {sha256_file(p) for p in candidates}
        if len(fingerprints) > 1:
            raise ValueError(f'Ambiguous different source files for {source_id}: {candidates}')
    return candidates[0] if candidates else None


def build_inventory(repo_root):
    """Return one row per selected sequence plus a fresh all-label audit.

    `eligible` means media range eligible, not pose-quality approved. The known
    content exclusions remain excluded even if a new download is present.
    """
    repo = Path(repo_root).resolve()
    root = repo / 'data/gavd'
    source_manifest = root / 'source/manifest.json'
    media_manifest = repo / 'data/videos-gavd/manifest.json'
    media = json.loads(media_manifest.read_text()) if media_manifest.exists() else {}
    media_sources = {s['video_id']: s for s in media.get('sources', [])}
    excluded = set(media.get('ml_exclusion_sequence_ids', []))
    # Preserve the documented reference mismatch if the media manifest is absent.
    excluded.update(('cljxlvf5h003a3n6lqssm40es', 'cljxlvp0c003e3n6lhbqb87qc',
                     'cljxlw9dn003i3n6l27b3p6y1', 'cljxlwvyd003n3n6ljo58d6s9'))
    records, by_label, seen_sequences, signatures = [], {}, set(), []
    all_sources = set()
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or directory.name == 'source' or directory.name.startswith('.'):
            continue
        for path in sorted(directory.glob('*.csv')):
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                    raise ValueError(f'Unexpected schema: {path}')
                rows = list(reader)
            if not rows:
                raise ValueError(f'Empty sequence {path}')
            first = rows[0]
            for column in ('seq', 'id', 'url', 'gait_pat'):
                if len({r[column] for r in rows}) != 1:
                    raise ValueError(f'Inconsistent {column}: {path}')
            seq, label, source = first['seq'], first['gait_pat'], first['id']
            if path.stem != seq or label != directory.name or seq in seen_sequences:
                raise ValueError(f'Sequence/label mismatch or duplicate sequence: {path}')
            seen_sequences.add(seq)
            frames = [int(r['frame_num']) for r in rows]
            if min(frames) < 1 or len(frames) != len(set(frames)):
                raise ValueError(f'Invalid or duplicate one-based frame: {path}')
            digest = sha256_file(path)
            signatures.append((str(path.relative_to(root)), digest))
            entry = by_label.setdefault(label, {'sequences': 0, 'rows': 0, 'sources': set()})
            entry['sequences'] += 1
            entry['rows'] += len(rows)
            entry['sources'].add(source)
            all_sources.add(source)
            if label not in LABELS:
                continue
            video = find_video(repo, source, label)
            source_meta = media_sources.get(source, {})
            frame_count = source_meta.get('probe', {}).get('decoded_frames')
            reason = ''
            if seq in excluded:
                reason = 'video_reference_mismatch'
            elif video is None:
                reason = 'missing_video'
            elif frame_count is not None and max(frames) > int(frame_count):
                reason = 'source_too_short'
            records.append(dict(sequence_id=seq, source_id=source, label=label, url=first['url'],
                                csv_path=str(path.relative_to(repo)), csv_sha256=digest,
                                n_frames=len(rows), first_frame=min(frames), last_frame=max(frames),
                                contiguous=sorted(frames) == list(range(min(frames), max(frames)+1)),
                                views='|'.join(sorted({r['cam_view'] for r in rows})),
                                video_path=str(video.relative_to(repo)) if video else '',
                                video_sha256_recorded=source_meta.get('sha256', ''),
                                eligible=not reason, exclusion_reason=reason))
    table = pd.DataFrame(records).sort_values(['label', 'source_id', 'sequence_id']).reset_index(drop=True)
    audit = dict(schema_version=1,
                 annotation_identity=hashlib.sha256(json.dumps(signatures).encode()).hexdigest(),
                 source_manifest_sha256=sha256_file(source_manifest) if source_manifest.exists() else None,
                 media_manifest_sha256=sha256_file(media_manifest) if media_manifest.exists() else None,
                 all_sequences=len(seen_sequences), all_rows=sum(v['rows'] for v in by_label.values()),
                 all_sources=len(all_sources),
                 by_label={k: dict(sequences=v['sequences'], rows=v['rows'], sources=len(v['sources'])) for k,v in by_label.items()},
                 selected_sequences=len(table), selected_rows=int(table.n_frames.sum()),
                 selected_sources=int(table.source_id.nunique()),
                 media_eligible_sequences=int(table.eligible.sum()),
                 media_eligible_sources=int(table.loc[table.eligible, 'source_id'].nunique()),
                 exclusions={str(k):int(v) for k,v in table.loc[~table.eligible, 'exclusion_reason'].value_counts().items()},
                 eligibility_note='Media eligibility only. Frame alignment, person identity, and pose quality need review; recorded video hashes are not reverified by this inventory.')
    return table, audit


def assign_splits(table, seed=27, validation_fraction=0.2, test_fraction=0.2):
    """Deterministically stratify source IDs; no window-level random splitting.

    Partition all annotated sources before quality filtering. Labels with fewer
    than three sources fail. A source with multiple labels fails for explicit
    reconciliation instead of being silently split into multiple groups.
    """
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError('Invalid partition fractions')
    if table.groupby('source_id').label.nunique().max() != 1:
        raise ValueError('Mixed-label source requires an explicit grouping protocol')
    rng, mapping = np.random.default_rng(seed), {}
    for label in LABELS:
        sources = np.array(sorted(table.loc[table.label == label, 'source_id'].unique()))
        if len(sources) < 3:
            raise ValueError(f'Need at least three independent sources for {label}')
        sources = rng.permutation(sources)
        ntest = max(1, int(round(len(sources) * test_fraction)))
        nval = max(1, int(round(len(sources) * validation_fraction)))
        if ntest + nval >= len(sources):
            raise ValueError(f'No training sources remain for {label}')
        for i, source in enumerate(sources):
            mapping[source] = 'test' if i < ntest else 'validation' if i < ntest+nval else 'train'
    out = table.copy()
    out['split'] = out.source_id.map(mapping)
    assert out.groupby('source_id').split.nunique().max() == 1
    out.attrs['split_seed'] = seed
    return out


def save_inventory(table, audit, output_dir):
    """Save human-readable inventory and split identity, refusing changed overwrite."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_text = table.to_csv(index=False)
    report = dict(audit, partition_sha256=hashlib.sha256(csv_text.encode()).hexdigest(),
                  split_seed=table.attrs.get('split_seed'),
                  partitions=table.groupby(['split','label']).agg(sequences=('sequence_id','count'), sources=('source_id','nunique')).reset_index().to_dict('records') if 'split' in table else [])
    for name, text in [('inventory.csv', csv_text), ('inventory.json', json.dumps(report, indent=2)+'\n')]:
        path = output / name
        if path.exists() and path.read_text() != text:
            raise FileExistsError(f'Inventory changed; use a new run directory: {path}')
        path.write_text(text)
    return report
