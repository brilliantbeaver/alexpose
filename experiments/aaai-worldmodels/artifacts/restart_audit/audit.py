"""Read-only audit of annotations, videos, and existing bounded pose cohorts."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from worldmodels.data import build_inventory, assign_splits, sha256_file
from worldmodels.workflow import load_prepared


def grouped(frame):
    return frame.groupby('label').agg(sequences=('sequence_id', 'nunique'), sources=('source_id', 'nunique')).reset_index().to_dict('records')


table, inventory = build_inventory(REPO)
table = assign_splits(table)
exclusions = json.loads((PROJECT/'docs/source_exclusions.json').read_text())
candidate = table[table.eligible & ~table.source_id.isin(exclusions['excluded_sources'])]
media = json.loads((REPO/'data/videos-gavd/manifest.json').read_text())
media_lookup = {row['video_id']: row for row in media['sources']}
hash_errors, hash_groups, media_stats = [], defaultdict(list), []
for row in table[table.video_path != ''].drop_duplicates('source_id').itertuples():
    path = REPO/row.video_path
    digest = sha256_file(path)
    hash_groups[digest].append(row.source_id)
    if row.video_sha256_recorded and digest != row.video_sha256_recorded:
        hash_errors.append(row.source_id)
    info = media_lookup.get(row.source_id, {})
    media_stats.append({'source_id': row.source_id, 'label': row.label, 'path': row.video_path, 'sha256': digest, 'bytes': path.stat().st_size, 'probe': info.get('probe', {})})

overlap_pairs = []
source_sampled = defaultdict(set)
source_frames = defaultdict(set)
all_intervals = defaultdict(list)
for row in table.itertuples():
    if not row.eligible or row.source_id in exclusions['excluded_sources']:
        continue
    frames = pd.read_csv(REPO/row.csv_path, usecols=['frame_num']).frame_num.to_numpy(int)
    source_sampled[row.source_id].update(map(int, frames[::2]))
    source_frames[row.source_id].update(map(int, frames))
    for old_id, old_frames in all_intervals[row.source_id]:
        shared = set(map(int, frames)).intersection(old_frames)
        if shared:
            overlap_pairs.append({'source_id': row.source_id, 'sequence_a': old_id, 'sequence_b': row.sequence_id, 'shared_annotated_frames': len(shared)})
    all_intervals[row.source_id].append((row.sequence_id, set(map(int, frames))))

cohorts = {}
for name in ['pilot-input-v2','pilot-input-clean-v1','extended-input-v1','extended-input-clean-v1']:
    base = PROJECT/'artifacts/runs'/name
    x, valid, meta = load_prepared(base)
    cohorts[name] = {'windows': len(meta), 'sequences': int(meta.sequence_id.nunique()), 'sources': int(meta.source_id.nunique()), 'x_shape': list(x.shape), 'valid_shape': list(valid.shape), 'metadata_columns': list(meta.columns), 'by_label': grouped(meta), 'by_partition': meta.groupby(['split','label']).agg(windows=('sequence_id','count'),sources=('source_id','nunique')).reset_index().to_dict('records'), 'source_split_conflicts': int((meta.groupby('source_id').split.nunique() > 1).sum()), 'source_annotation_split_mismatches': int((meta.merge(table[['sequence_id','split']],on='sequence_id',suffixes=('','_annotation')).eval('split != split_annotation')).sum()), 'nonfinite_x': int((~np.isfinite(x)).sum()), 'valid_fraction': float(valid.mean()), 'metadata_sha256': sha256_file(base/'windows.csv'), 'data_sha256': sha256_file(base/'windows.npz')}
    if 'horizon_seconds' in meta:
        cohorts[name]['forecast_horizon_seconds'] = {str(k): float(v) for k,v in meta.horizon_seconds.quantile([0,.25,.5,.75,1]).items()}
        cohorts[name]['unresolved_person_windows'] = int((meta.unresolved_person_fraction > 0).sum())

raw = PROJECT/'artifacts/runs/extended-input-v1/sequence_poses'
cache_schema = Counter()
cache_frames = Counter()
cached_unique_frames = defaultdict(set)
all_cached_unique_frames = defaultdict(set)
invalid_cache_records = []
cache_identities = []
for path in sorted(raw.glob('*.npz')):
    with np.load(path, allow_pickle=False) as z:
        identity = json.loads(str(z['provenance_json']))
        sequence, source = str(z['sequence_id']), str(z['source_id'])
        cache_schema[str(identity.get('schema'))] += 1
        cache_frames[len(z['frame_numbers'])] += 1
        row = table[table.sequence_id==sequence].iloc[0]
        errors = []
        if identity['csv_sha256'] != row.csv_sha256: errors.append('csv_hash')
        if identity['video_sha256'] != next(m['sha256'] for m in media_stats if m['source_id']==source): errors.append('video_hash')
        expected = pd.read_csv(REPO/row.csv_path,usecols=['frame_num']).sort_values('frame_num').iloc[::2].iloc[:32].frame_num.to_numpy(int)
        if not np.array_equal(expected,z['frame_numbers']): errors.append('frame_alignment')
        if errors: invalid_cache_records.append({'sequence_id': sequence, 'errors': errors})
        all_cached_unique_frames[source].update(map(int,z['frame_numbers']))
        if source not in exclusions['excluded_sources']:
            cached_unique_frames[source].update(map(int,z['frame_numbers']))
        cache_identities.append({'sequence_id':sequence,'source_id':source,'path':str(path.relative_to(REPO)),'sha256':sha256_file(path)})

gate = pd.read_csv(PROJECT/'artifacts/runs/extended-input-v1/person_gate_frames.csv')
gate = gate[~gate.source_id.isin(exclusions['excluded_sources'])]
cache_stats = {'files':len(cache_identities),'schemas':dict(cache_schema),'lengths':dict(cache_frames),'invalid_records':invalid_cache_records,'unique_sampled_frames_before_manual_exclusions':sum(map(len,all_cached_unique_frames.values())),'unique_sampled_frames_after_manual_exclusions':sum(map(len,cached_unique_frames.values())),'person_gate_frame_counts_after_manual_exclusions':{str(k):int(v) for k,v in gate.reason.value_counts().items()}}
pose_summary = Counter()
for path in (PROJECT/'artifacts/poses').glob('*.npz'):
    with np.load(path, allow_pickle=False) as z:
        identity=json.loads(str(z['provenance_json']))
        pose_summary[(str(identity.get('schema')),len(z['frame_numbers']))]+=1

result = {'audit_identity':'medwm-restart-data-audit-v1','inventory':inventory,'manual_source_exclusions':exclusions['excluded_sources'],'candidate_after_all_existing_exclusions':{'sequences':len(candidate),'sources':int(candidate.source_id.nunique()),'annotation_rows':int(candidate.n_frames.sum()),'by_label':grouped(candidate),'unique_annotated_source_frames':sum(map(len,source_frames.values())),'unique_stride_two_source_frames':sum(map(len,source_sampled.values())),'potential_new_stride_two_source_frames_beyond_32_cache':sum(len(frames-cached_unique_frames[source]) for source,frames in source_sampled.items())},'media':{'source_files':len(media_stats),'bytes':sum(m['bytes'] for m in media_stats),'manifest_hash_mismatches':hash_errors,'exact_duplicate_source_groups':[sources for sources in hash_groups.values() if len(sources)>1],'sources':media_stats},'overlapping_annotation_sequences':overlap_pairs,'cohorts':cohorts,'extended_raw_cache':cache_stats,'other_worldmodels_pose_cache_counts':[{'schema':schema,'frames':frames,'files':count} for (schema,frames),count in sorted(pose_summary.items())],'raw_cache_identities':cache_identities,'input_identities':{'source_manifest_sha256':sha256_file(REPO/'data/gavd/source/manifest.json'),'media_manifest_sha256':sha256_file(REPO/'data/videos-gavd/manifest.json'),'source_exclusions_sha256':sha256_file(PROJECT/'docs/source_exclusions.json'),'audit_script_sha256':sha256_file(__file__)},'limitations':['Exact file hashes do not detect recoded reposts or shared participants.','Patient identity and verified diagnosis are not established by source IDs.','Previous test outcomes have already been inspected; a new cross-validation result on these sources is exploratory.','Full-sequence pose eligibility has not been measured.','Raw-cache identity checks verify CSV/video bytes and frame alignment; complete extractor/model source hash verification is recorded separately.']}
(PROJECT/'docs/restart/data_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['media','raw_cache_identities','cohorts']},indent=2))
print(json.dumps(result['cohorts'],indent=2))
