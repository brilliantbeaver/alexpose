"""Auditable cohort reuse for the second reference-aware gait study.

The module creates no poses and fits no model. It keeps the immutable prepared
arrays in place, records their identities, and selects rows by prepared_index.
Only exact footage hashes and recorded source IDs establish grouping edges.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .data import CSV_FIELDS, JOINT_IDS, LABELS, build_inventory, sha256_file
from .pose import normalize_window
from .settings import PROJECT, REPOSITORY

C, H = 16, 8
LEGS = tuple(tuple(JOINT_IDS.index(j) for j in ids)
             for ids in ((25, 27, 29, 31), (26, 28, 30, 32)))
DEFAULT = PROJECT / 'artifacts/runs/reference-v2-cohort-v1'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def common_support(valid):
    """The original common observed distal-target gate, without fitting."""
    v = np.asarray(valid)
    if v.ndim != 3 or v.shape[1:] != (24, 16) or v.dtype != np.bool_:
        raise ValueError('Expected Boolean support [N,24,16]')
    support = v[:, C:] & v[:, :C].any(1)[:, None]
    keep = np.ones(len(v), bool)
    for leg in LEGS:
        keep &= (support[:, :4, leg].sum((1, 2)) >= 4)
        keep &= (support[:, :, leg].sum((1, 2)) >= 8)
    return keep


def relation_components(source_ids, media):
    """Union only source IDs with byte-identical original footage."""
    sources = sorted(set(map(str, source_ids)))
    parent = {s: s for s in sources}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges = []
    for _, group in media.groupby('sha256', sort=True):
        ids = sorted(set(group.source_id) & set(sources))
        for b in ids[1:]:
            a = ids[0]
            ra, rb = find(a), find(b)
            parent[max(ra, rb)] = min(ra, rb)
            edges.append(dict(source_a=a, source_b=b, relation='byte_identical_original_footage',
                              evidence_sha256=group.sha256.iloc[0]))
    return {s: find(s) for s in sources}, edges


def component_folds(meta, n_splits=5, seed=2701):
    """Stratify source components; require a reconciled single label per group."""
    needed = {'source_id', 'component_id', 'label'}
    if not needed.issubset(meta):
        raise ValueError('Missing source/component/label contract')
    if meta.groupby('source_id').component_id.nunique().max() != 1:
        raise ValueError('A source cannot belong to multiple components')
    groups = meta[['component_id', 'label']].drop_duplicates().sort_values('component_id')
    if groups.groupby('component_id').label.nunique().max() != 1:
        raise ValueError('Mixed-label component requires explicit endpoint reconciliation')
    if groups.label.value_counts().min() < n_splits:
        raise ValueError('Insufficient components per label for declared folds')
    mapping = {}
    splitter = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(groups, groups.label)):
        mapping.update({s: fold for s in groups.iloc[test].component_id})
    return meta.component_id.map(mapping).to_numpy(int)


def load_cohort(path=None, verify=True):
    """Return x, valid, metadata, actual decoder times for the frozen new cohort.

    The metadata's prepared_index maps exactly to the immutable original rows.
    Future support is an offline evaluation gate, never an inference input.
    """
    root = Path(path or DEFAULT)
    identity = json.loads((root/'identity.json').read_text())
    prepared = REPOSITORY / identity['prepared_directory']
    if verify:
        for name, expected in identity['prepared_inputs'].items():
            if sha256_file(prepared/name) != expected:
                raise ValueError(f'Prepared artifact hash mismatch: {name}')
        if sha256_file(root/'cohort.csv') != identity['cohort_sha256']:
            raise ValueError('Cohort membership hash mismatch')
    meta = pd.read_csv(root/'cohort.csv', keep_default_na=False)
    index = meta.prepared_index.to_numpy(int)
    with np.load(prepared/'windows.npz', allow_pickle=False) as z:
        x, valid = z['x'][index], z['valid'][index]
    if not common_support(valid).all():
        raise ValueError('Declared common target support failed')
    times = np.asarray([json.loads(s) for s in meta.timestamp_seconds], dtype=np.float64)
    if times.shape != (len(meta), C+H) or not np.isfinite(times).all() or not (np.diff(times, axis=1)>0).all():
        raise ValueError('Invalid measured decoder-time matrix')
    if meta.window_id.duplicated().any() or meta.groupby('component_id').outer_fold.nunique().max()!=1:
        raise ValueError('Duplicated window identity or nonisolated grouping')
    return x, valid, meta, times


def all_annotations(repo):
    records = []
    for directory in sorted((repo/'data/gavd').iterdir()):
        if not directory.is_dir() or directory.name=='source' or directory.name.startswith('.'):
            continue
        for path in sorted(directory.glob('*.csv')):
            with path.open(newline='') as stream:
                reader = csv.DictReader(stream)
                if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                    raise ValueError(f'Invalid annotation schema: {path}')
                rows = list(reader)
            if not rows:
                raise ValueError('Empty annotation')
            first = rows[0]
            for key in ('seq','id','gait_pat','url'):
                if len({r[key] for r in rows}) != 1:
                    raise ValueError(f'Conflicting {key}: {path}')
            frames = np.array([int(r['frame_num']) for r in rows])
            if first['seq']!=path.stem or first['gait_pat']!=directory.name or np.any(frames<1) or len(set(frames))!=len(frames):
                raise ValueError(f'Invalid annotation identity/frame convention: {path}')
            records.append(dict(sequence_id=first['seq'],source_id=first['id'],label=first['gait_pat'],
                annotation_rows=len(rows),first_frame=int(frames.min()),last_frame=int(frames.max()),
                views='|'.join(sorted({r['cam_view'] for r in rows})),
                csv_path=str(path.relative_to(repo)),csv_sha256=sha256_file(path)))
    table = pd.DataFrame(records)
    if table.sequence_id.duplicated().any():
        raise ValueError('Repeated sequence across transformed annotations')
    return table


def media_inventory(repo, source_ids):
    records = []
    for base in ('data/videos-gavd','data/videos-ms','data/videos-more'):
        root = repo/base
        if not root.exists():
            continue
        for path in sorted(root.rglob('*')):
            if path.suffix.lower() not in ('.mp4','.mov','.mkv','.webm') or not path.is_file():
                continue
            # Raw sources are exact 11-character IDs. Frame-trimmed derivatives
            # are audited separately through their explicit source manifest.
            if path.stem not in source_ids:
                continue
            records.append(dict(source_id=path.stem,path=str(path.relative_to(repo)),
                                bytes=path.stat().st_size,sha256=sha256_file(path)))
    return pd.DataFrame(records, columns=['source_id','path','bytes','sha256'])


def reconstruct(prepared, x, valid, meta):
    """Replay all prepared windows from native sequence caches, without inference."""
    max_error = 0.
    for seq, rows in meta.groupby('sequence_id', sort=True):
        with np.load(prepared/'sequence_poses'/f'{seq}.npz', allow_pickle=False) as z:
            pose = {k:z[k].copy() for k in z.files}
        if not np.array_equal(pose['decoded_indexes'],pose['frame_numbers']-1):
            raise ValueError('Native source frame origin mismatch')
        lookup = {int(n):i for i,n in enumerate(pose['frame_numbers'])}
        for i,row in rows.iterrows():
            native = np.array(json.loads(row.frame_numbers),int)
            indexes = [lookup[int(n)] for n in native]
            expected, mask = normalize_window(pose['xy'][indexes],pose['confidence'][indexes],context_frames=C)
            mask[C:] &= mask[:C].any(0)[None]
            expected = np.where(mask[...,None],expected,0)
            max_error = max(max_error,float(np.max(np.abs(expected-x[i]))))
            np.testing.assert_array_equal(expected,x[i])
            np.testing.assert_array_equal(mask,valid[i])
            np.testing.assert_array_equal(np.asarray(json.loads(row.timestamp_seconds)),pose['timestamp_seconds'][indexes])
            fingerprint = hashlib.sha256(json.dumps([row.source_id,native.tolist()],separators=(',',':')).encode()).hexdigest()
            if row.window_id!=fingerprint:
                raise ValueError('Window frame identity mismatch')
    return dict(windows_reconstructed=len(meta),sequences_reconstructed=int(meta.sequence_id.nunique()),
                maximum_absolute_error=max_error,frame_origin_and_times_verified=True)


def prepare(output=None):
    """Audit once and publish a new cohort manifest after all invariants pass."""
    started = time.monotonic()
    output = Path(output or DEFAULT)
    if output.exists():
        raise FileExistsError('A new cohort identity needs an unused output directory')
    repo = REPOSITORY
    prepared = PROJECT/'artifacts/runs/restart-full10hz-v1'
    prep = json.loads((prepared/'preparation.json').read_text())
    if any(prep.get(k)!=v for k,v in [('frames',24),('context_frames',16),('sample_fps',10.)]):
        raise ValueError('Prepared time/shape contract mismatch')
    for name,key in [('windows.npz','data_sha256'),('windows.csv','metadata_sha256')]:
        if sha256_file(prepared/name)!=prep[key]:
            raise ValueError(f'Frozen preparation mismatch: {name}')
    meta = pd.read_csv(prepared/'windows.csv',keep_default_na=False)
    with np.load(prepared/'windows.npz',allow_pickle=False) as z:
        x,valid = z['x'],z['valid']
    if x.shape!=(len(meta),24,16,2) or x.dtype!=np.float32 or valid.dtype!=np.bool_:
        raise ValueError('Unexpected prepared tensor contract')
    if not np.isfinite(x).all() or np.any(x[~valid]!=0) or meta.window_id.duplicated().any():
        raise ValueError('Invalid normalized/missing observations or duplicated windows')
    replay = reconstruct(prepared,x,valid,meta)
    selected, fresh = build_inventory(repo)
    catalog = all_annotations(repo)
    media = media_inventory(repo,set(catalog.source_id))
    components,edges = relation_components(catalog.source_id,media)
    progress = json.loads((prepared/'source_progress.json').read_text())
    pose_records=[]
    for item in progress['sources']:
        source=item['source_id'];path=prepared/'source_poses'/f'{source}.npz'
        cache_hash=sha256_file(path)
        if cache_hash!=item['source_cache_sha256']:
            raise ValueError(f'Raw pose cache hash mismatch: {source}')
        with np.load(path,allow_pickle=False) as z:
            provenance=json.loads(str(z['provenance_json']))
            if not np.array_equal(z['decoded_indexes'],z['frame_numbers']-1):
                raise ValueError(f'Source cache frame mismatch: {source}')
        video=repo/provenance['video_path']
        video_hash=sha256_file(video)
        if video_hash!=provenance['video_sha256'] or video_hash!=item['video_sha256']:
            raise ValueError(f'Actual source footage changed: {source}')
        pose_records.append(dict(source_id=source,source_pose_path=str(path.relative_to(repo)),
            source_pose_sha256=cache_hash,video_path=str(video.relative_to(repo)),video_sha256=video_hash,
            pose_model_sha256=provenance['pose_model_sha256'],requested_frames=item['requested_frames']))
    keep=common_support(valid)
    meta['prepared_index']=np.arange(len(meta))
    cohort=meta.loc[keep].copy().reset_index(drop=True)
    cohort['component_id']=cohort.source_id.map(components)
    cohort['outer_fold']=component_folds(cohort)
    original=pd.read_csv(PROJECT/'artifacts/runs/restart-dynamics-v1/cohort.csv',keep_default_na=False)
    if cohort.window_id.tolist()!=original.window_id.tolist():
        raise ValueError('Common-support reconstruction changed the historical model row set')
    same_folds=bool(np.array_equal(cohort.outer_fold,original.outer_fold))
    times=np.asarray([json.loads(s) for s in cohort.timestamp_seconds],float)
    if not np.isfinite(times).all() or not (np.diff(times,axis=1)>0).all():
        raise ValueError('Nonmonotonic actual decoder timestamps')
    annotation_now=catalog.set_index('sequence_id').csv_sha256
    selected_before=pd.read_csv(prepared/'selected_sequences.csv',keep_default_na=False)
    if not (selected_before.set_index('sequence_id').csv_sha256==annotation_now.reindex(selected_before.sequence_id)).all():
        raise ValueError('Prepared selected annotations changed')
    # Store complete catalogs and small reference tables, never duplicate tensors.
    output.mkdir(parents=True)
    catalog.to_csv(output/'all_annotations.csv',index=False)
    selected.to_csv(output/'selected_inventory.csv',index=False)
    cohort.to_csv(output/'cohort.csv',index=False)
    meta.loc[~keep].to_csv(output/'support_exclusions.csv',index=False)
    media.to_csv(output/'media_relations.csv',index=False)
    pd.DataFrame(pose_records).to_csv(output/'source_cache_manifest.csv',index=False)
    pd.DataFrame([dict(source_id=s,component_id=c) for s,c in sorted(components.items())]).to_csv(output/'source_components.csv',index=False)
    write_json(output/'relation_graph.json',dict(edges=edges,component_policy='Source ID or byte-identical original footage only; no inferred participant identities.',
        verified_cross_source_edges=len(edges),unknown_participant_overlap=True))
    mixed=catalog.groupby('source_id').label.agg(lambda s:sorted(set(s)))
    mixed={k:v for k,v in mixed.items() if len(v)>1}
    summary=cohort.groupby('label').agg(windows=('window_id','size'),sequences=('sequence_id','nunique'),sources=('source_id','nunique'),components=('component_id','nunique')).reset_index()
    summary.to_csv(output/'coverage_by_label.csv',index=False)
    cohort.groupby(['label','view']).agg(windows=('window_id','size'),sequences=('sequence_id','nunique'),sources=('source_id','nunique')).reset_index().to_csv(output/'coverage_by_view.csv',index=False)
    fold_counts=cohort[['source_id','component_id','label','outer_fold']].drop_duplicates().groupby(['outer_fold','label']).size().unstack(fill_value=0)
    fold_counts.to_csv(output/'fold_counts.csv')
    attrition=pd.read_csv(prepared/'attrition.csv',keep_default_na=False)
    coverage=[]
    for label in LABELS:
        cat=catalog[catalog.label==label];eligible=selected[(selected.label==label)&selected.eligible]
        candidate=selected_before[selected_before.label==label];final=cohort[cohort.label==label]
        row=dict(label=label,catalog_sequences=len(cat),catalog_sources=int(cat.source_id.nunique()),media_sequences=len(eligible),media_sources=int(eligible.source_id.nunique()),
            candidate_sequences=len(candidate),candidate_sources=int(candidate.source_id.nunique()),candidate_windows=int(attrition.loc[attrition.label==label,'candidate_windows'].sum()),
            prepared_windows=int((meta.label==label).sum()),model_windows=len(final),model_sequences=int(final.sequence_id.nunique()),model_sources=int(final.source_id.nunique()))
        coverage.append(row)
    pd.DataFrame(coverage).to_csv(output/'attrition_by_label.csv',index=False)
    key_quantiles=['prefix_valid_fraction','future_valid_fraction','prefix_mean_confidence','future_mean_confidence','final_horizon_seconds','maximum_grid_error_seconds']
    quantiles={key:{str(q):float(cohort[key].quantile(q)) for q in (0,.25,.5,.75,1)} for key in key_quantiles}
    write_json(output/'audit.json',dict(identity='reference-v2-cohort-audit-v1',status='complete',fresh_inventory=fresh,mixed_label_sources=mixed,
        prepared_windows=len(meta),model_windows=len(cohort),model_sequences=int(cohort.sequence_id.nunique()),sources=int(cohort.source_id.nunique()),
        normal_sources=int(cohort.loc[cohort.label=='normal','source_id'].nunique()),source_components=int(cohort.component_id.nunique()),
        exact_footage_cross_source_edges=edges,folds_equal_historical=same_folds,source_caches_verified=len(pose_records),reconstruction=replay,
        timing=dict(actual_decoder_intervals_min_seconds=float(np.diff(times,axis=1).min()),actual_decoder_intervals_max_seconds=float(np.diff(times,axis=1).max()),all_windows_strictly_increasing=True),
        quality_and_timing_quantiles=quantiles,coverage_by_label=summary.to_dict('records'),seconds=time.monotonic()-started,
        limits=['Source IDs are not participant IDs.','Byte identity cannot detect mirrored or edited reuploads.','Participant overlap across distinct recordings remains unknown.',
            'Annotation folders do not independently verify clinical diagnosis.','Future observation gates define offline outcome availability, not a deployable decision rule.','Historical GAVD outcome exposure makes this an exploratory study.']))
    identity=dict(identity='reference-v2-cohort-v1',status='complete',prepared_directory=str(prepared.relative_to(repo)),
        prepared_inputs={name:sha256_file(prepared/name) for name in ['windows.npz','windows.csv','preparation.json','request.json','source_progress.json']},
        cohort_sha256=sha256_file(output/'cohort.csv'),code_sha256=sha256_file(__file__),
        code_revision=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        grouping='source_id plus verified byte-identical original-footage components',outer_folds=5,outer_seed=2701,
        annotation_identity=fresh['annotation_identity'],source_manifest_sha256=fresh['source_manifest_sha256'],media_manifest_sha256=fresh['media_manifest_sha256'],
        pose_model_sha256=sorted(set(r['pose_model_sha256'] for r in pose_records)),
        note='Prepared arrays reused by immutable hash and row index, with no new extraction or learned-model fitting.')
    write_json(output/'identity.json',identity)
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256_file(p) for p in sorted(output.glob('*')) if p.is_file()})
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    print(prepare(args.output))
