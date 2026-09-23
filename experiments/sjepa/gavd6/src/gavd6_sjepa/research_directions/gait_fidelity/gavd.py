"""Recording-grouped real-video transfer, with no invented pose references.

GAVD labels supervise a fixed downstream probe only. The pose restorers are
trained on AMASS; GAVD labels, camera view and diagnosis never enter them.
Raw annotations are indexed in SQLite so workers read one sequence at a time.
"""
from __future__ import annotations

import argparse
import ast
import html
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile
from types import SimpleNamespace
import warnings

import numpy as np
import pandas as pd

from .common import atomic_json, code_identity, digest, locked, read_json, sha256, verify_code

SCHEMA = 'gait-fidelity-gavd-v1'
LABEL = 'gait_pattern_annotation'
SAFE = re.compile(r'^[A-Za-z0-9_-]+$')


def _safe(value):
    if not SAFE.fullmatch(str(value)):
        raise ValueError(f'Unsafe sequence/video identifier: {value!r}')
    return str(value)


def recording_groups(videos, identities=None, file_hashes=None):
    """Union known people and exact duplicate media; absence of IDs is explicit."""
    videos = sorted(set(map(str, videos)))
    parent = {v: v for v in videos}
    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v
    def join(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    if identities is not None:
        if not {'video_id', 'person_id'} <= set(identities):
            raise ValueError('Identity CSV requires video_id,person_id (one row per known person/video)')
        if identities[['video_id', 'person_id']].isna().any().any() or identities.person_id.astype(str).str.strip().eq('').any():
            raise ValueError('Identity crosswalk contains missing person IDs')
        for _, group in identities.groupby('person_id'):
            members = sorted(set(group.video_id.astype(str)) & set(videos))
            for v in members[1:]:
                join(members[0], v)
    hashes = {}
    for v, h in (file_hashes or {}).items():
        if h:
            if h in hashes:
                join(v, hashes[h])
            hashes[h] = v
    components = {}
    for v in videos:
        components.setdefault(find(v), []).append(v)
    return {v: digest(members)[:20] for members in components.values() for v in members}


def assign_splits(sequences, groups, *, seed=731, reservations=None):
    """Metadata-only stratification of whole recording/person components.

    Explicit historical reservations stay protected. Unknown prior exposure
    permits exploratory train/development, never independent confirmation.
    """
    labels = sequences.assign(group=sequences.video_id.map(groups)).groupby('group')[LABEL].agg(lambda x: tuple(sorted(set(x))))
    protected, exposed = set(), set()
    if reservations is not None:
        if 'video_id' not in reservations or reservations.video_id.duplicated().any():
            raise ValueError('Reservation CSV requires unique video_id rows')
        supported={'role','split','reserved','protected','previously_exposed','exposure_group','exposed','exposure','exposure_status'}
        if not supported & set(reservations):
            raise ValueError('Reservation CSV has no recognized role, reservation or exposure fields')
        def flag(value):
            word=str(value).strip().lower()
            if word in {'true','1','yes'}:return True
            if word in {'false','0','no','','unknown'}:return False
            raise ValueError(f'Unrecognized reservation boolean: {value!r}')
        for r in reservations.to_dict('records'):
            group = groups.get(str(r['video_id']))
            if group is None:
                continue
            # Historical exports can retain both role and split columns.
            # A blank or less restrictive role must not erase a protected split.
            roles = {str(r.get(k, '')).strip().lower() for k in ('role', 'split')} - {''}
            recognized = {'confirmation','test','reserved','protected','final','holdout','train','training','development','validation','source','calibration','excluded','unknown'}
            if roles - recognized:
                raise ValueError(f'Unrecognized historical GAVD roles: {sorted(roles-recognized)!r}')
            if roles & {'confirmation', 'test', 'reserved', 'protected', 'final','holdout','excluded'} or any(flag(r.get(k,'')) for k in ('reserved','protected')):
                protected.add(group)
            histories=[str(r[k]).strip().lower() for k in ('exposure','exposure_status') if k in r]
            if (roles & {'train','training','development','validation','source','calibration'}
                or any(flag(r.get(k,'')) for k in ('previously_exposed','exposure_group','exposed'))
                or any(value not in {'','unknown','unexposed','unexposed_verified','none','false','0'} for value in histories)):
                exposed.add(group)
    output = {g: 'protected' for g in protected}
    for label_set in sorted(set(labels)):
        members = sorted([g for g, label in labels.items() if label == label_set and g not in protected],
                         key=lambda g: digest([seed, g]))
        # Small categories stay visible in the inventory; an absent test class
        # is never described as validated because other classes have support.
        n = len(members)
        n_test = max(1, int(n * .2)) if n >= 5 else 0
        n_dev = max(1, int(n * .2)) if n >= 3 else 0
        eligible = [g for g in members if g not in exposed]
        tests = set(eligible[:n_test])
        remaining = [g for g in members if g not in tests]
        dev = set(remaining[:n_dev])
        output.update({g: 'confirmation' if g in tests else 'development' if g in dev else 'train' for g in members})
    return output


def _resolve_video(root, video):
    candidates = [p for folder in (root, root / 'all') for ext in ('.mp4', '.mkv', '.webm', '.mov', '.m4v')
                  if (p := folder / (video + ext)).is_file()]
    if len(candidates) > 1:
        hashes = {sha256(p) for p in candidates}
        if len(hashes) > 1:
            raise ValueError(f'Multiple different media versions for {video}; choose an unambiguous video root')
    return candidates[0].resolve() if candidates else None


def plan_gavd(cfg, *, video_root=None, annotation_root=None, manifest_dir=None,
              identity_csv=None, reservation_csv=None, shards=8, frame_origin=1):
    if shards < 1 or shards > 1024 or frame_origin != 1:
        raise ValueError('Positive shard count <=1024 and GAVD one-based annotation frames required')
    work = Path(cfg['work']); root = Path(cfg['asset_root'])
    destination = work / 'gavd'
    if destination.exists():
        raise FileExistsError('GAVD plan is immutable; use the saved plan or a new study directory')
    manifest_dir = Path(manifest_dir or root / 'manifests/gavd').resolve()
    full = Path(os.environ.get('GAVD_FULL_ROOT', root / 'data/gavd_full'))
    video_root = Path(video_root or full / 'youtube/all').resolve()
    annotation_root = Path(annotation_root or full / 'annotations/GAVD/data').resolve()
    if reservation_csv is None:
        known_reservation = os.environ.get('ST_GAVD_RESERVATION')
        if known_reservation:
            reservation_csv = Path(known_reservation)
            if not reservation_csv.is_file():
                raise FileNotFoundError(f'Declared historical GAVD reservation is missing: {reservation_csv}')
        else:
            historical = root / 'outputs/future-innovation/learning-curve/config/source-reservation.csv'
            if historical.is_file(): reservation_csv = historical
    sequences = pd.read_csv(manifest_dir / 'gavd_full_sequences.csv', keep_default_na=False, dtype={'video_id': str, 'sequence_id': str})
    videos = pd.read_csv(manifest_dir / 'gavd_full_videos.csv', keep_default_na=False, dtype={'video_id': str})
    needed = {'sequence_id', 'video_id', LABEL, 'dataset_annotation', 'cam_view', 'first_frame', 'last_frame', 'n_annotated_frames', 'source_height'}
    if not needed <= set(sequences) or sequences.sequence_id.duplicated().any() or videos.video_id.duplicated().any():
        raise ValueError('Incomplete/duplicate GAVD sequence/video manifest')
    if not set(sequences.video_id) <= set(videos.video_id):
        raise ValueError('Sequence video absent from video manifest')
    for column in ('sequence_id', 'video_id'):
        sequences[column].map(_safe)
    if sequences[LABEL].str.strip().eq('').any() or (sequences.first_frame < 1).any() or (sequences.last_frame < sequences.first_frame).any():
        raise ValueError('Empty gait labels or invalid annotation spans')
    annotations = sorted(annotation_root.glob('GAVD_Clinical_Annotations_*.csv'))
    if not annotations:
        raise FileNotFoundError(f'No GAVD_Clinical_Annotations_*.csv in {annotation_root}')
    paths = {v: _resolve_video(video_root, v) for v in sorted(set(sequences.video_id))}
    hashes = {v: sha256(p) if p else None for v, p in paths.items()}
    identities = pd.read_csv(identity_csv, dtype=str, keep_default_na=False) if identity_csv else None
    reservations = pd.read_csv(reservation_csv, dtype=str, keep_default_na=False) if reservation_csv else None
    links=[]
    if identities is not None:
        if not {'video_id','person_id'}<=set(identities):raise ValueError('Identity CSV requires video_id,person_id')
        if identities[['video_id','person_id']].isna().any().any() or identities.person_id.str.strip().eq('').any():
            raise ValueError('Identity crosswalk contains missing person IDs')
        links.append(identities.assign(person_id='person:'+identities.person_id))
    if reservations is not None:
        if 'video_id' not in reservations:raise ValueError('Reservation CSV requires video_id')
        for column in ('source_group','person_id','canonical_person_id'):
            if column in reservations:
                linked=reservations[reservations[column].str.strip().ne('')][['video_id',column]].rename(columns={column:'person_id'})
                linked['person_id']='reservation-'+column+':'+linked.person_id
                links.append(linked)
    groups = recording_groups(paths, pd.concat(links,ignore_index=True) if links else None, hashes)
    roles = assign_splits(sequences, groups, reservations=reservations)
    sequences['source_group'] = sequences.video_id.map(groups)
    sequences['split'] = sequences.source_group.map(roles)
    sequences['video_path'] = sequences.video_id.map(lambda v: str(paths[v] or ''))
    sequences['video_sha256'] = sequences.video_id.map(lambda v: hashes[v] or '')
    sequences['available'] = sequences.video_path.ne('')
    sequences['shard_index'] = sequences.source_group.map(lambda g: int(digest(g)[:12], 16) % shards)
    sequences['label_disagreement'] = sequences[LABEL].eq('normal') & sequences.dataset_annotation.ne('Normal Gait')
    sources = [manifest_dir / 'gavd_full_sequences.csv', manifest_dir / 'gavd_full_videos.csv', *annotations]
    sources += [Path(x).resolve() for x in (identity_csv, reservation_csv) if x]
    work.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.gavd-plan-', dir=work))
    database = sqlite3.connect(staging / 'annotations.sqlite')
    try:
        database.execute('CREATE TABLE frames (sequence TEXT, frame INTEGER, video TEXT, bbox TEXT, info TEXT, PRIMARY KEY(sequence,frame))')
        known = dict(zip(sequences.sequence_id, sequences.video_id))
        for path in annotations:
            for chunk in pd.read_csv(path, usecols=['seq', 'frame_num', 'id', 'bbox', 'vid_info'], dtype={'seq': str, 'id': str}, keep_default_na=False, chunksize=20000):
                chunk = chunk[chunk.seq.isin(known)]
                rows = []
                for r in chunk.itertuples(index=False):
                    values = dict(zip(chunk.columns, r))
                    if str(values['id']) != known[str(values['seq'])]:
                        raise ValueError('Frame annotation video differs from sequence manifest')
                    f = float(values['frame_num'])
                    if not np.isfinite(f) or not f.is_integer() or f < 1:
                        raise ValueError('Frame identifiers must be positive integers')
                    rows.append((str(values['seq']), int(f), str(values['id']), str(values['bbox']), str(values['vid_info'])))
                database.executemany('INSERT INTO frames VALUES (?,?,?,?,?)', rows)
        database.commit()
        bounds = {seq:(n,first,last) for seq,n,first,last in database.execute('SELECT sequence,COUNT(*),MIN(frame),MAX(frame) FROM frames GROUP BY sequence')}
        sequences['indexed_annotation_frames'] = sequences.sequence_id.map(lambda s:bounds.get(s,(0,None,None))[0]).astype(int)
        if not sequences.indexed_annotation_frames.eq(sequences.n_annotated_frames).all():
            raise ValueError('Raw annotation frame counts disagree with the sequence manifest')
        if any(bounds[row.sequence_id][1:] != (int(row.first_frame),int(row.last_frame)) for row in sequences.itertuples()):
            raise ValueError('Raw annotation frame bounds disagree with the sequence manifest')
    finally:
        database.close()
    sequences.to_csv(staging / 'sequences.csv', index=False)
    estimator_assets = {}
    from ..synthetic_training_v2.preparation import _config_files
    for spec in cfg['preparation'].get('estimators', []):
        estimator_assets.update(_config_files(spec['config']))
        for k in ('checkpoint', 'head_checkpoint'):
            if spec.get(k):
                estimator_assets[str(Path(spec[k]).resolve())] = sha256(spec[k])
    plan = dict(schema=SCHEMA, num_shards=shards, frame_origin=1, seed=731,
                samples=int(cfg['data']['samples']), hz=float(cfg['data']['hz']),
                estimators=cfg['preparation'].get('estimators', []), estimator_assets=estimator_assets,
                source_hashes={str(p.resolve()): sha256(p) for p in sources},
                sequences_sha256=sha256(staging / 'sequences.csv'), annotations_sha256=sha256(staging / 'annotations.sqlite'),
                code=code_identity(cfg['code_root']), code_root=cfg['code_root'],
                groups='recording_components_with_supplied_person_links_and_exact_media_duplicates',
                person_identity_verified=False,
                identity_crosswalk_supplied=identity_csv is not None,
                videos_with_supplied_person_links=len(set(identities.video_id)&set(paths)) if identities is not None else 0,
                prior_reservations_supplied=reservation_csv is not None,
                exposure_status='unknown_until_reviewed_ledger',
                checkpoint_labels_policy='AMASS restorers fixed; GAVD trains only downstream linear probes',
                probe=dict(ridge=1., bootstrap_draws=2000, bootstrap_seed=734,
                           endpoint='recording_group_macro_recall',
                           primary_comparator='unchanged', feature_version='normalized_pose_quantiles_velocity_v1'),
                counts=dict(sequences=len(sequences), videos=sequences.video_id.nunique(), groups=sequences.source_group.nunique(),
                            available_sequences=int(sequences.available.sum())),
                split_counts=sequences.groupby('split').agg(sequences=('sequence_id','size'), groups=('source_group','nunique')).to_dict('index'))
    plan['identity'] = digest(plan)
    atomic_json(staging / 'plan.json', plan)
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    _write_inventory_page(destination, sequences)
    return dict(status='GAVD_PLAN_COMPLETE', plan=str(destination / 'plan.json'), **plan['counts'], split_counts=plan['split_counts'])


def _load_plan(work):
    folder = Path(work) / 'gavd'; plan = read_json(folder / 'plan.json')
    identity = plan.pop('identity')
    if plan.get('schema') != SCHEMA or digest(plan) != identity:
        raise ValueError('GAVD plan identity changed')
    plan['identity'] = identity
    for name, key in (('sequences.csv', 'sequences_sha256'), ('annotations.sqlite', 'annotations_sha256')):
        if sha256(folder / name) != plan[key]:
            raise ValueError(f'Frozen GAVD input changed: {name}')
    verify_code(plan['code_root'], plan['code'])
    return folder, plan


def _write_inventory_page(folder, rows):
    rows.groupby([LABEL, 'cam_view', 'split'], dropna=False).agg(sequences=('sequence_id','size'),
        groups=('source_group','nunique'), available=('available','sum')).to_csv(folder / 'inventory.csv')
    text = '<!doctype html><meta charset="utf-8"><title>GAVD data inventory</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style>'
    text += '<h1>GAVD data inventory</h1><p>Camera side is not the affected limb. Labels describe sequences; source groups are not verified participant counts. Missing files remain in the inventory.</p>'
    text += rows[['sequence_id','video_id',LABEL,'cam_view','split','available','label_disagreement']].to_html(index=False, escape=True)
    (folder / 'inventory.html').write_text(text)


def annotation_box(bbox, info, frame_shape):
    """Strict metadata scaling; missing boxes never become whole-frame crops."""
    b, s = ast.literal_eval(bbox), ast.literal_eval(info)
    if not isinstance(b, dict) or not isinstance(s, dict):
        raise ValueError('Annotation box/video geometry must be dictionaries')
    values = [float(b[k]) for k in ('left', 'top', 'width', 'height')]
    source = [float(s[k]) for k in ('width', 'height')]
    if not np.isfinite(values + source).all() or min(values[2:] + source) <= 0:
        raise ValueError('Invalid annotated bounding box or source dimensions')
    h, w = frame_shape[:2]; left, top, width, height = values
    box = np.array([left, top, left + width, top + height]) * np.array([w/source[0], h/source[1]] * 2)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
    if np.any(box[2:] <= box[:2]):
        raise ValueError('Annotated person box lies outside decoded frame')
    return box


def sample_windows(first, last, fps, samples, hz):
    """Select nonoverlapping frame indices on the nominal video-rate grid.

    Videos slower than the requested rate cannot supply distinct observations.
    No repeated frames, interpolation, stretching or padded final windows.
    Extraction separately checks and retains decoder-reported timestamps.
    """
    if not np.isfinite(fps) or fps < hz - 1e-6:
        return []
    duration = (last - first) / fps
    count = max(0, int(math.floor((duration - (samples-1)/hz) / (samples/hz) + 1 + 1e-8)))
    result = []
    for k in range(count):
        indices = np.rint(first + (k*samples + np.arange(samples))*fps/hz).astype(int)
        if len(set(indices)) != samples or indices[-1] > last:
            continue
        result.append(indices)
    return result


def _check_artifacts(receipt):
    for path, expected in receipt.get('artifacts', {}).items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError(f'GAVD artifact changed: {path}')


def _check_confirmation(folder, plan, checkpoints=None):
    lock = read_json(folder / 'confirmation-lock.json')
    if lock['plan_identity'] != plan['identity']:
        raise ValueError('Confirmation population/configuration changed')
    if checkpoints is not None and lock['checkpoints'] != checkpoints:
        raise ValueError('Confirmation methods/checkpoints differ from frozen declaration')
    _check_artifacts(lock)
    return lock


def extract_gavd(cfg, shard_index, *, split='development'):
    folder, plan = _load_plan(cfg['work'])
    if split not in {'development', 'confirmation'} or not 0 <= shard_index < plan['num_shards']:
        raise ValueError('Invalid GAVD split/shard')
    if split == 'confirmation':
        _check_confirmation(folder, plan)
    for p, h in plan['estimator_assets'].items():
        if not Path(p).is_file() or sha256(p) != h:
            raise ValueError(f'Pose-estimator asset changed: {p}')
    if not plan['estimators']:
        raise ValueError('At least one declared pose estimator is required')
    destination = folder / 'shards' / split / f'{shard_index:04d}'
    with locked(destination.with_suffix('.lock')):
        if (destination / 'complete.json').exists():
            result = read_json(destination / 'complete.json'); _check_artifacts(result)
            if result['plan_identity'] != plan['identity']:
                raise ValueError('Extraction was made from a different plan')
            return result
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f'.{shard_index:04d}-', dir=destination.parent))
        from ..synthetic_training.estimators import StudentSpec, load_estimator
        from ..synthetic_training_v2.extraction import extract_tracks
        from ..synthetic_training_v2.runtime import require_haic_runtime
        import cv2
        require_haic_runtime()
        estimators = {s['student_id']: load_estimator(StudentSpec(**s), device='cuda') for s in plan['estimators']}
        rows = pd.read_csv(folder / 'sequences.csv', keep_default_na=False)
        roles = ['confirmation'] if split == 'confirmation' else ['train', 'development']
        rows = rows[rows.shard_index.eq(shard_index) & rows.split.isin(roles)]
        database = sqlite3.connect(f'file:{folder / "annotations.sqlite"}?mode=ro', uri=True)
        records, outcomes = [], []
        try:
            for video, group in rows.groupby('video_id', sort=True):
                path = Path(group.iloc[0].video_path)
                if not path.is_file() or sha256(path) != group.iloc[0].video_sha256:
                    outcomes.extend(dict(sequence_id=r.sequence_id, video_id=video, status='missing_or_changed_video', windows=0) for r in group.itertuples())
                    continue
                cap = cv2.VideoCapture(str(path))
                try:
                    fps = float(cap.get(cv2.CAP_PROP_FPS))
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for row in group.to_dict('records'):
                        seq = row['sequence_id']; successful = 0
                        try:
                            if not cap.isOpened() or not np.isfinite(fps) or fps <= 0 or total < row['last_frame']:
                                raise ValueError('Unreadable video, invalid FPS, or truncated annotation span')
                            windows = sample_windows(int(row['first_frame'])-1, int(row['last_frame'])-1, fps, plan['samples'], plan['hz'])
                            if not windows:
                                reason = 'source_fps_below_requested_rate' if fps < plan['hz'] else 'sequence_too_short_for_window'
                                outcomes.append(dict(sequence_id=seq, video_id=video, status=reason, windows=0)); continue
                            for frames in windows:
                                annotations = {int(f)-1: (bbox, info) for f, bbox, info in database.execute(
                                    'SELECT frame,bbox,info FROM frames WHERE sequence=? AND frame BETWEEN ? AND ? ORDER BY frame',
                                    (seq, int(frames[0])+1, int(frames[-1])+1))}
                                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
                                images, boxes, actual_times = [], [], []
                                wanted = set(frames.tolist())
                                for f in range(int(frames[0]), int(frames[-1])+1):
                                    ok, frame = cap.read()
                                    if not ok:
                                        raise ValueError('Failed frame decode within selected interval')
                                    if f not in wanted:
                                        continue
                                    if f not in annotations:
                                        raise ValueError('Missing annotated box on selected frame')
                                    stamp = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
                                    if not np.isfinite(stamp) or abs(stamp - f/fps) > max(.002, .55/fps):
                                        raise ValueError('Decoded timestamps disagree with constant-rate frame clock; review variable-rate source')
                                    images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    boxes.append(annotation_box(*annotations[f], frame.shape))
                                    actual_times.append(stamp)
                                times = np.asarray(actual_times)
                                if len(times) != plan['samples'] or np.any(np.diff(times) <= 0):
                                    raise ValueError('Repeated/nonmonotonic source timestamps')
                                for name, estimator in estimators.items():
                                    track = extract_tracks(estimator, images, boxes, times, box_source='gavd_annotated_person_box')
                                    item = f'{seq}-{int(frames[0]):09d}-{name}'
                                    _safe(item)
                                    rel = Path('tracks') / (item + '.npz'); (staging / rel).parent.mkdir(exist_ok=True)
                                    np.savez_compressed(staging / rel, **track.inputs(), source_frames=frames, boxes=np.asarray(boxes))
                                    records.append({**row, 'window_id': item, 'extractor': name, 'fps': fps,
                                                    'start_frame': int(frames[0]), 'end_frame': int(frames[-1]),
                                                    'track': str(destination / rel), 'track_sha256': sha256(staging / rel),
                                                    'observed_fraction': float(track.observed.mean()),
                                                    'extraction_status': json.dumps(track.status_counts, sort_keys=True),
                                                    'source_time_grid': 'nearest_native_frames_actual_decoded_seconds',
                                                    'max_grid_deviation_s': float(np.max(abs(times-(times[0]+np.arange(len(times))/plan['hz']))))})
                                    if successful == 0 and name == next(iter(estimators)):
                                        _review_sheet(images, track.xy, staging / 'review' / f'{seq}.jpg')
                                successful += 1
                            outcomes.append(dict(sequence_id=seq, video_id=video, status='complete', windows=successful))
                        except (ValueError, OSError, KeyError, SyntaxError, TypeError) as exc:
                            # Retain failures in the declared denominator; never
                            # compare methods on separately successful cohorts.
                            outcomes.append(dict(sequence_id=seq, video_id=video, status='failed', windows=successful, reason=str(exc)))
                finally:
                    cap.release()
        finally:
            database.close()
        pd.DataFrame(records).to_csv(staging / 'tracks.csv', index=False)
        pd.DataFrame(outcomes, columns=['sequence_id','video_id','status','windows','reason']).to_csv(staging / 'coverage.csv', index=False)
        review_files = sorted((staging / 'review').glob('*.jpg'))
        page = '<!doctype html><meta charset="utf-8"><title>GAVD review</title><h1>GAVD pose review</h1><p>Estimated left is blue; right is orange. These are model assignments, not verified anatomical labels.</p>'
        page += ''.join(f'<figure><img style="max-width:95%" src="review/{html.escape(p.name)}"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in review_files)
        (staging / 'review.html').write_text(page)
        result = dict(status='GAVD_EXTRACTION_COMPLETE', plan_identity=plan['identity'], output=str(destination),
                      split=split, shard_index=shard_index, sequences=len(rows), tracks=len(records),
                      manifest=str(destination / 'tracks.csv'), coverage=str(destination / 'coverage.csv'),
                      artifacts={str(destination / p.relative_to(staging)): sha256(p) for p in staging.rglob('*') if p.is_file()})
        atomic_json(staging / 'complete.json', result)
        if destination.exists():
            raise FileExistsError(destination)
        staging.rename(destination)
        return result


def _review_sheet(images, xy, path):
    from PIL import Image, ImageDraw
    from .visualization import EDGES
    selected = np.linspace(0, len(images)-1, 6).astype(int)
    canvas = Image.new('RGB', (960, 480), 'white')
    for k, i in enumerate(selected):
        im = Image.fromarray(images[i]); draw = ImageDraw.Draw(im)
        for a, b in EDGES:
            if np.isfinite(xy[i, [a,b]]).all():
                draw.line([tuple(xy[i,a]),tuple(xy[i,b])], fill=(255,220,70), width=3)
        for j, (x,y) in enumerate(xy[i]):
            if np.isfinite([x,y]).all():
                draw.ellipse((x-4,y-4,x+4,y+4), fill=(30,140,255) if j%2==0 else (255,100,40))
        im.thumbnail((320,240)); canvas.paste(im, (k%3*320,k//3*240))
    path.parent.mkdir(exist_ok=True); canvas.save(path)


def pose_features(xy, observed, timestamps, *, input_origin=None, input_scale=None):
    """Fixed interpretable descriptors; missing values are imputed on train only.

    Per-joint coordinate quantiles, velocity RMS and coverage use a single
    input-derived scale, followed by projected knee excursion and support.
    These are predictive descriptors, not reference-validated clinical metrics.
    """
    from .measurements import knee_angles
    xy = np.asarray(xy, float); observed = np.asarray(observed, bool)
    if xy.ndim != 3 or xy.shape[1:] != (12,2) or observed.shape != xy.shape[:-1]:
        raise ValueError('Expected one body12 sequence and observation mask')
    t = np.asarray(timestamps, float)
    if t.shape != (len(xy),) or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Features require actual monotonically increasing timestamps')
    valid = observed & np.isfinite(xy).all(-1)
    points = xy[valid]
    scale = np.linalg.norm(np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0)) if len(points)>=2 else 0.
    origin = np.median(points,axis=0) if len(points) else np.zeros(2)
    if input_origin is not None:
        origin=np.asarray(input_origin,float)
    if input_scale is not None:
        scale=float(input_scale)
    q = np.where(valid[...,None], (xy-origin)/max(scale,1e-6), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        quantiles = np.nanquantile(q,[.1,.5,.9],axis=0).ravel()
        velocity = np.diff(q,axis=0)/np.diff(t)[:,None,None]
        rms = np.sqrt(np.nanmean(velocity**2,axis=0)).ravel()
        angles, support = knee_angles(xy, valid)
        excursion = np.diff(np.nanquantile(angles,[.05,.95],axis=0),axis=0).ravel()
    return np.concatenate([quantiles,rms,valid.mean(0),excursion,support.mean(0)])


def fit_probe(x, labels, groups, *, ridge=1.):
    """Group-weighted ridge classification with training-only preprocessing."""
    x=np.asarray(x,float); labels=np.asarray(labels,str); groups=np.asarray(groups,str)
    if len(x)!=len(labels) or len(x)!=len(groups) or len(set(labels))<2:
        raise ValueError('Probe needs aligned training rows from at least two gait labels')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        median=np.nanmedian(x,axis=0)
    median=np.nan_to_num(median); x=np.where(np.isfinite(x),x,median)
    weights=np.array([1./np.sum(groups==g) for g in groups]); weights/=weights.sum()
    mean=np.sum(x*weights[:,None],axis=0); scale=np.sqrt(np.sum((x-mean)**2*weights[:,None],axis=0));scale=np.maximum(scale,1e-6)
    design=np.column_stack([(x-mean)/scale,np.ones(len(x))]);classes=np.unique(labels)
    y=(labels[:,None]==classes).astype(float); penalty=np.eye(design.shape[1])*ridge;penalty[-1,-1]=0
    coef=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights[:,None]*y))
    return dict(median=median,mean=mean,scale=scale,coef=coef,classes=classes)


def probe_predict(model,x):
    x=np.asarray(x,float);x=np.where(np.isfinite(x),x,model['median'])
    scores=np.column_stack([(x-model['mean'])/model['scale'],np.ones(len(x))])@model['coef']
    return model['classes'][np.argmax(scores,axis=1)]


def _checkpoint_map(values):
    result={}
    for item in values or []:
        if '=' not in item:
            raise ValueError('--checkpoint must be NAME=/absolute/path/checkpoint.pt')
        name,path=item.split('=',1);_safe(name);path=Path(path).expanduser().resolve()
        if name in {'unchanged','filter1','view_only','availability_only'} or name in result or not path.is_file():
            raise ValueError('Duplicate/reserved method name or missing checkpoint')
        result[name]=dict(path=str(path),sha256=sha256(path))
    return result


def _validate_restorer_signature(payload,plan):
    signature=payload['signature']
    if signature['model']['window_size']!=plan['samples']:
        raise ValueError('Checkpoint window differs from frozen GAVD sampling')
    if not np.isclose(float(signature.get('sampling_hz',np.nan)),plan['hz'],rtol=0,atol=1e-8):
        raise ValueError('Checkpoint sampling rate differs from the GAVD protocol')
    if signature.get('evidence_status') not in {'technical-source-screen','automated-source-screen','audited-source'}:
        raise ValueError('GAVD scientific evaluation requires a source-trained checkpoint with recorded evidence provenance')


def ledger_checkpoints(cfg):
    """Resolve every declared seed of the primary comparison from receipts."""
    from .scheduler import verify_completed
    work=Path(cfg['work']);state=read_json(work/'ledger.json');plan=read_json(work/'plan.json')
    primary={cfg['evaluation']['primary_candidate'],cfg['evaluation']['primary_comparator']}
    values=[]
    for phase in plan['phases']:
        if phase['phase']=='pretrain' or phase['recipe']['recipe_id'] not in primary:
            continue
        receipt=state['completed'].get(phase['phase_id'])
        if receipt is None:
            raise ValueError('Complete all primary comparison seeds before GAVD evaluation, or explicitly declare --checkpoint NAME=PATH')
        verify_completed(receipt)
        values.append(f"{phase['recipe']['recipe_id']}-seed-{phase['seed']}={receipt['result']['checkpoint']}")
    if len(values)!=len(primary)*len(cfg['seeds']):
        raise ValueError('Saved plan does not contain the complete declared primary comparison')
    return values


def lock_gavd(work, checkpoints, exposure_csv):
    folder,plan=_load_plan(work); methods=_checkpoint_map(checkpoints)
    if not methods:
        raise ValueError('Freeze at least one completed AMASS restorer')
    rows=pd.read_csv(folder/'sequences.csv',keep_default_na=False)
    # All videos linked to a test component must have verified exposure status.
    selected=rows[rows.split.eq('confirmation')]
    exposure=pd.read_csv(exposure_csv,dtype=str,keep_default_na=False)
    required={'video_id','exposure','reviewed_by','evidence'}
    if not required<=set(exposure) or exposure.video_id.duplicated().any():
        raise ValueError('Exposure ledger requires unique video_id,exposure,reviewed_by,evidence')
    audit=exposure.set_index('video_id').reindex(sorted(set(selected.video_id)))
    if audit.empty or audit.isna().any().any() or not audit.exposure.eq('unexposed_verified').all() or audit[['reviewed_by','evidence']].eq('').any().any():
        raise ValueError('Every confirmation video needs documented unexposed_verified status')
    from .training import load_model
    for method in methods.values():
        _,payload=load_model(method['path'],device='cpu')
        _validate_restorer_signature(payload,plan)
    path=folder/'confirmation-lock.json'
    value=dict(schema='gait-fidelity-gavd-confirmation-v1',plan_identity=plan['identity'],checkpoints=methods,
               exposure_sha256=sha256(exposure_csv), groups=sorted(selected.source_group.unique()),
               probe=plan['probe'],independence_unit='recording_component_not_verified_person',
               artifacts={str(Path(exposure_csv).resolve()):sha256(exposure_csv),**{v['path']:v['sha256'] for v in methods.values()}})
    if path.exists() and read_json(path)!=value:
        raise ValueError('Confirmation declaration is immutable')
    atomic_json(path,value)
    return dict(status='GAVD_CONFIRMATION_LOCKED',lock=str(path),groups=len(value['groups']))


def _track_tables(folder, plan, split):
    tables=[];coverage=[]
    needed=['development'] if split=='development' else ['development','confirmation']
    for partition in needed:
        for i in range(plan['num_shards']):
            receipt=read_json(folder/'shards'/partition/f'{i:04d}'/'complete.json')
            if receipt['plan_identity']!=plan['identity']:
                raise ValueError('Extraction plan differs')
            _check_artifacts(receipt)
            if receipt['tracks']:
                tables.append(pd.read_csv(receipt['manifest'],keep_default_na=False))
            coverage.append(pd.read_csv(receipt['coverage'],keep_default_na=False))
    if not tables:
        raise ValueError('No extracted GAVD windows; inspect coverage before changing the declared protocol')
    rows=pd.concat(tables,ignore_index=True)
    if rows.window_id.duplicated().any() or rows.groupby('source_group').split.nunique().gt(1).any():
        raise ValueError('Duplicate windows or source-group leakage')
    return rows,pd.concat(coverage,ignore_index=True)


def grouped_scores(table, *, draws=2000, seed=734):
    """Macro recall from group means; bootstrap resamples whole source groups."""
    table=table.copy();table['correct']=table.label.eq(table.predicted).astype(float)
    cells=table.groupby(['source_group','label']).correct.mean().unstack('label')
    values=cells.to_numpy();rng=np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        recalls=np.nanmean(values,axis=0)
        samples=[]
        for _ in range(draws):
            take=values[rng.integers(len(values),size=len(values))]
            # A replicate missing an entire class is unsupported, not zero.
            if np.isfinite(take).any(0).all():
                samples.append(float(np.nanmean(take,axis=0).mean()))
    adequate=bool(len(values)>=2 and cells.notna().sum().ge(2).all())
    return dict(macro_recall=float(recalls.mean()),groups=len(values),labels=list(cells.columns),
                per_label_recall=dict(zip(cells.columns,recalls.tolist())),
                per_label_groups={k:int(cells[k].notna().sum()) for k in cells},
                bootstrap_ci95=np.quantile(samples,[.025,.975]).tolist() if adequate and len(samples)>=100 else None,
                interval_status='descriptive' if adequate else 'insufficient_independent_groups_per_label',
                supported_bootstrap_draws=len(samples),requested_bootstrap_draws=draws)


def paired_group_interval(table, candidate, comparator='unchanged', *, draws=2000, seed=735):
    """Paired macro-recall difference; all methods retain the same sequences."""
    keys=['sequence_id','source_group','label']
    part=table[table.method.isin([candidate,comparator])].copy()
    part['correct']=part.label.eq(part.predicted).astype(float)
    pivot=part.pivot(index=keys,columns='method',values='correct')
    if candidate not in pivot or comparator not in pivot or pivot.isna().any().any():
        return dict(status='unmatched_evaluation_population')
    delta=(pivot[candidate]-pivot[comparator]).rename('delta').reset_index()
    cells=delta.groupby(['source_group','label']).delta.mean().unstack('label')
    x=cells.to_numpy();rng=np.random.default_rng(seed);boot=[]
    for _ in range(draws):
        sample=x[rng.integers(len(x),size=len(x))]
        if np.isfinite(sample).any(0).all():
            boot.append(float(np.nanmean(sample,axis=0).mean()))
    adequate=bool(len(x)>=2 and cells.notna().sum().ge(2).all())
    return dict(status='descriptive_paired_recording_group_estimate' if adequate else 'insufficient_independent_groups_per_label',candidate=candidate,comparator=comparator,
                improvement=float(np.nanmean(x,axis=0).mean()),groups=len(x),
                ci95=np.quantile(boot,[.025,.975]).tolist() if adequate and len(boot)>=100 else None,
                supported_bootstrap_draws=len(boot),requested_bootstrap_draws=draws)


def seeded_primary_comparison(table,candidate,comparator,*,draws=2000,seed=736):
    """Crossed recording/seed uncertainty for the declared neural comparison."""
    rows=table.copy()
    parsed=rows.method.str.extract(r'^(.*)-seed-(\d+)$')
    rows['family']=parsed[0];rows['seed']=parsed[1]
    rows=rows[rows.family.isin([candidate,comparator])].copy()
    if rows.empty:
        return dict(status='no_complete_named_seed_comparison')
    rows['correct']=rows.label.eq(rows.predicted).astype(float)
    cells=rows.groupby(['source_group','label','seed','family']).correct.mean().unstack('family')
    if candidate not in cells or comparator not in cells or cells.isna().any().any():
        return dict(status='unmatched_seed_or_recording_support')
    delta=(cells[candidate]-cells[comparator]).unstack('seed')
    if delta.isna().any().any():
        return dict(status='incomplete_seed_coverage')
    groups=sorted(set(delta.index.get_level_values('source_group')));labels=sorted(set(delta.index.get_level_values('label')))
    seeds=delta.columns.tolist();values=np.full((len(groups),len(labels),len(seeds)),np.nan)
    for (group,label),row in delta.iterrows(): values[groups.index(group),labels.index(label)]=row.to_numpy()
    rng=np.random.default_rng(seed);boot=[]
    for _ in range(draws):
        sample=values[rng.integers(len(groups),size=len(groups))][:,:,rng.integers(len(seeds),size=len(seeds))]
        if np.isfinite(sample).any(0).all():boot.append(float(np.nanmean(sample,axis=0).mean()))
    adequate=bool(len(groups)>=2 and (np.isfinite(values).any(2).sum(0)>=2).all())
    return dict(status='descriptive_secondary_transfer_comparison' if adequate else 'insufficient_independent_groups_per_label',candidate=candidate,comparator=comparator,
                improvement=float(np.nanmean(values,axis=0).mean()),groups=len(groups),seeds=len(seeds),
                crossed_group_seed_ci95=np.quantile(boot,[.025,.975]).tolist() if adequate and len(boot)>=100 else None,
                supported_bootstrap_draws=len(boot),
                interpretation='Positive favors candidate. Seeds and clips do not increase the independent recording count; unknown person overlap remains.')


def evaluate_gavd(cfg, checkpoints, *, split='development', device='cpu'):
    folder,plan=_load_plan(cfg['work']);methods=_checkpoint_map(checkpoints)
    if split not in {'development','confirmation'}:
        raise ValueError('Use development or explicitly locked confirmation')
    if split=='confirmation':
        _check_confirmation(folder,plan,methods)
    rows,coverage=_track_tables(folder,plan,split)
    rows=rows[rows.split.isin(['train',split])].copy()
    from .training import load_model,normalize_batch,_tensors
    from ..synthetic_training_v2.data import filter_tracks
    import torch
    models={}
    for name,entry in methods.items():
        model,payload=load_model(entry['path'],device=device)
        _validate_restorer_signature(payload,plan)
        models[name]=model
    signature=digest(dict(plan=plan['identity'],methods=methods,split=split,device=device))
    destination=folder/'evaluation'/signature[:16]
    if (destination/'summary.json').exists():
        result=read_json(destination/'summary.json');_check_artifacts(result);return result
    destination.parent.mkdir(exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.evaluation-',dir=destination.parent))
    records=[];features=[]
    for r in rows.to_dict('records'):
        with np.load(r['track'],allow_pickle=False) as raw:
            inputs={k:raw[k][None] for k in ('xy','confidence','observed','timestamps')}
        normalized,origin,scale,_=normalize_batch(inputs)
        predictions={'unchanged':inputs['xy'][0], 'filter1':filter_tracks(inputs,1)[0]}
        with torch.inference_mode():
            tensors=_tensors(normalized,device)
            for name,model in models.items():
                predictions[name]=(model(tensors).float().cpu().numpy()*scale[:,None,None,None]+origin[:,None,None])[0]
        for name,xy in predictions.items():
            # Keep the same input-supported joint population for all methods.
            # Imputed model joints have no reference and do not manufacture support.
            feature=pose_features(xy,inputs['observed'][0],inputs['timestamps'][0],input_origin=origin[0],input_scale=scale[0])
            features.append(feature)
            records.append({k:r[k] for k in ('sequence_id','source_group','video_id','split','extractor','cam_view','source_height',LABEL)}|dict(method=name))
    frame=pd.DataFrame(records); feature_names=[f'f{i:03d}' for i in range(len(features[0]))]
    frame=pd.concat([frame,pd.DataFrame(np.stack(features),columns=feature_names)],axis=1)
    # Windows -> sequences -> groups. A long sequence or many clips from a
    # single video cannot increase its weight in the training objective.
    keys=['method','extractor','sequence_id','source_group','video_id','split',LABEL,'cam_view','source_height']
    frame=frame.groupby(keys,dropna=False)[feature_names].mean().reset_index()
    predictions=[];summaries={}
    for (method,extractor),part in frame.groupby(['method','extractor']):
        train=part[part.split.eq('train')];test=part[part.split.eq(split)]
        if train.empty or test.empty or train[LABEL].nunique()<2:
            summaries[f'{method}/{extractor}']=dict(status='insufficient_label_or_partition_support');continue
        model=fit_probe(train[feature_names],train[LABEL],train.source_group,ridge=plan['probe']['ridge'])
        predicted=probe_predict(model,test[feature_names])
        table=test[['sequence_id','source_group','video_id','cam_view',LABEL]].rename(columns={LABEL:'label'}).copy()
        table['predicted']=predicted;table['method']=method;table['extractor']=extractor
        predictions.append(table)
        np.savez_compressed(staging/f'probe-{method}-{extractor}.npz',**model)
        summaries[f'{method}/{extractor}']=dict(status='descriptive_recording_group_estimate',
            **grouped_scores(table,draws=plan['probe']['bootstrap_draws'],seed=plan['probe']['bootstrap_seed']),
            labels_absent_from_training=sorted(set(test[LABEL])-set(train[LABEL])),
            train_groups=train.source_group.nunique(),evaluation_groups=test.source_group.nunique())
    # A camera/height-only probe exposes predictable acquisition confounds.
    # Its fixed encoding uses no inferred pose or diagnostic-label information.
    views=['left side','right side','front','back','other','']
    for extractor,part in frame[frame.method.eq('unchanged')].groupby('extractor'):
        def nuisance_features(value):
            return np.column_stack([*(value.cam_view.eq(v).astype(float) for v in views),
                                    pd.to_numeric(value.source_height,errors='raise').to_numpy(float)/1000.])
        train=part[part.split.eq('train')];test=part[part.split.eq(split)]
        if train.empty or test.empty or train[LABEL].nunique()<2:
            continue
        model=fit_probe(nuisance_features(train),train[LABEL],train.source_group,ridge=plan['probe']['ridge'])
        table=test[['sequence_id','source_group','video_id','cam_view',LABEL]].rename(columns={LABEL:'label'}).copy()
        table['predicted']=probe_predict(model,nuisance_features(test));table['method']='view_only';table['extractor']=extractor
        predictions.append(table)
        np.savez_compressed(staging/f'probe-view_only-{extractor}.npz',**model)
        summaries[f'view_only/{extractor}']=dict(status='acquisition_confound_control',
            **grouped_scores(table,draws=plan['probe']['bootstrap_draws'],seed=plan['probe']['bootstrap_seed']))
        # Observation failure can itself correlate with category. Give it its
        # own control so label predictability is not mistaken for gait quality.
        support_columns=feature_names[96:108]
        support_model=fit_probe(train[support_columns],train[LABEL],train.source_group,ridge=plan['probe']['ridge'])
        support_table=table.copy();support_table['method']='availability_only'
        support_table['predicted']=probe_predict(support_model,test[support_columns])
        predictions.append(support_table)
        np.savez_compressed(staging/f'probe-availability_only-{extractor}.npz',**support_model)
        summaries[f'availability_only/{extractor}']=dict(status='observation_failure_confound_control',
            **grouped_scores(support_table,draws=plan['probe']['bootstrap_draws'],seed=plan['probe']['bootstrap_seed']))
    if not predictions:
        raise ValueError('No label probe has both training and evaluation support')
    predictions=pd.concat(predictions,ignore_index=True)
    comparisons={}
    for extractor,part in predictions.groupby('extractor'):
        for method in sorted(set(part.method)-{'unchanged','view_only','availability_only'}):
            comparisons[f'{method}/{extractor}']=paired_group_interval(part,method,draws=plan['probe']['bootstrap_draws'])
        evaluation=cfg.get('evaluation',{})
        if 'primary_candidate' in evaluation and 'primary_comparator' in evaluation:
            comparisons[f'primary/{extractor}']=seeded_primary_comparison(part,evaluation['primary_candidate'],evaluation['primary_comparator'],
                                                                          draws=plan['probe']['bootstrap_draws'])
    atomic_json(staging/'comparisons.json',comparisons)
    view_rows=[]
    for (method,extractor,view),part in predictions.groupby(['method','extractor','cam_view'],dropna=False):
        stats=grouped_scores(part,draws=0)
        view_rows.append(dict(method=method,extractor=extractor,cam_view=view,macro_recall=stats['macro_recall'],
                              groups=stats['groups'],labels=';'.join(stats['labels'])))
    pd.DataFrame(view_rows).to_csv(staging/'by-view.csv',index=False)
    predictions.to_csv(staging/'predictions.csv',index=False);frame.to_csv(staging/'sequence-features.csv',index=False)
    coverage.to_csv(staging/'extraction-coverage.csv',index=False)
    atomic_json(staging/'scores.json',summaries)
    (staging/'report.md').write_text('# GAVD transfer evaluation\n\nThis evaluates preservation of sequence-label information using fixed AMASS restorers and training-only linear probes. '
        'It does not measure pose accuracy, affected-side accuracy or clinical validity. '
        'Intervals resample recording components, which may contain unidentified repeated people. '
        'Read extraction-coverage.csv alongside scores.json: short, low-frame-rate, missing and failed clips remain explicit exclusions. '
        'Classes without evaluation support are not validated.\n')
    result=dict(status='GAVD_TRANSFER_EVALUATED',plan_identity=plan['identity'],split=split,checkpoints=methods,
                output=str(destination),scores=summaries,clinical_validation=False,
                independent_person_confirmation=False,confirmation_exposure_reviewed=split=='confirmation',
                artifacts={str(destination/p.relative_to(staging)):sha256(p) for p in staging.rglob('*') if p.is_file()})
    atomic_json(staging/'summary.json',result)
    if destination.exists():
        raise FileExistsError(destination)
    staging.rename(destination)
    return result


def add_commands(commands):
    for name in ('gavd-plan','gavd-extract','gavd-evaluate','gavd-lock'):
        sub=commands.add_parser(name);sub.add_argument('--work',type=Path,required=True)
        if name=='gavd-plan':
            for arg in ('video-root','annotation-root','manifest-dir','identity-csv','reservation-csv'):
                sub.add_argument('--'+arg,type=Path)
            sub.add_argument('--shards',type=int,default=8);sub.add_argument('--frame-origin',type=int,default=1)
        if name in {'gavd-extract','gavd-evaluate'}:
            sub.add_argument('--split',choices=('development','confirmation'),default='development')
        if name=='gavd-extract':
            sub.add_argument('--shard-index',type=int,required=True)
        if name in {'gavd-evaluate','gavd-lock'}:
            sub.add_argument('--checkpoint',action='append',default=[])
        if name=='gavd-evaluate':
            sub.add_argument('--device',choices=('cpu','cuda'),default='cpu')
        if name=='gavd-lock':
            sub.add_argument('--exposure-csv',type=Path,required=True)


def run_command(args):
    from .config import load_config
    cfg=load_config(args.work)
    if args.command=='gavd-plan':
        return plan_gavd(cfg,**{k:getattr(args,k,None) for k in ('video_root','annotation_root','manifest_dir','identity_csv','reservation_csv','shards','frame_origin')})
    if args.command=='gavd-extract':
        if not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('GAVD pose extraction belongs in a Slurm GPU allocation')
        return extract_gavd(cfg,args.shard_index,split=getattr(args,'split','development'))
    if args.command=='gavd-lock':
        return lock_gavd(args.work,args.checkpoint or ledger_checkpoints(cfg),args.exposure_csv)
    return evaluate_gavd(cfg,args.checkpoint or ledger_checkpoints(cfg),split=args.split,device=args.device)
