"""Fresh twelve-folder inventory and source folds for the predictive study."""
from __future__ import annotations
import hashlib,json,re
from pathlib import Path
import numpy as np
import pandas as pd
from .data import CSV_COLUMNS,file_sha256,find_video,load_windows,manifest_digest
STUDY_LABELS=('abnormal','antalgic','cerebral palsy','exercise','inebriated','myopathic','normal','parkinsons','pregnant','prosthetic','stroke','style')
BINARY_RULE='normal_folder_0_all_other_condition_folders_1_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def source_table(metadata):
    """One row per recording, with each category membership represented once."""
    if not {'source_id','label'}.issubset(metadata.columns):
        raise ValueError('Metadata needs source_id and label')
    records=[]
    for source,group in metadata.groupby('source_id',sort=True):
        labels=sorted(set(group.label.astype(str)))
        if not set(labels).issubset(STUDY_LABELS):raise ValueError('Unknown condition label')
        binary={int(label!='normal') for label in labels}
        if len(binary)!=1:raise ValueError(f'Source mixes normal and abnormal folders: {source}')
        value=next(iter(binary))
        if 'binary_label' in group and not group.binary_label.eq(value).all():
            raise ValueError('binary_label disagrees with the folder rule')
        records.append(dict(source_id=str(source),binary_label=value,categories='|'.join(labels),
                            n_sequences=int(group.sequence_id.nunique()) if 'sequence_id' in group else len(group)))
    return pd.DataFrame(records,columns=['source_id','binary_label','categories','n_sequences'])


def build_study_manifest(repo_root,pose_dir=None,prior_sources=None,inspect_media=False):
    """Read fresh condition CSVs. Historical indices supply prior-use flags only."""
    root=Path(repo_root).resolve();annotations=root/'data/gavd'
    folders=sorted(p for p in annotations.iterdir() if p.is_dir() and p.name!='source' and any(p.glob('*.csv')))
    if {p.name for p in folders}!=set(STUDY_LABELS):
        raise ValueError('Study requires exactly the twelve declared condition folders')
    prior_path=root/'experiments/aaai/artifacts/full_window_index.csv'
    prior_basis='explicit_source_list' if prior_sources is not None else 'unknown'
    if prior_sources is None and prior_path.exists():
        prior_sources=set(pd.read_csv(prior_path,dtype=str).source_id)
        prior_basis='earlier_window_index:'+file_sha256(prior_path)
    prior=set(map(str,prior_sources or []));rows=[]
    for directory in folders:
        for path in sorted(directory.glob('*.csv')):
            table=pd.read_csv(path,dtype=str,keep_default_na=False)
            if tuple(table.columns)!=CSV_COLUMNS or table.empty:raise ValueError(f'Unexpected schema/empty CSV: {path}')
            if not table.seq.eq(path.stem).all() or not table.gait_pat.eq(directory.name).all():
                raise ValueError(f'Sequence/folder mismatch: {path}')
            if table.id.nunique()!=1 or table.url.nunique()!=1:raise ValueError(f'Multiple sources: {path}')
            source=str(table.iloc[0].id)
            if not re.fullmatch(r'[A-Za-z0-9_-]{11}',source):raise ValueError('Invalid source ID')
            frames=pd.to_numeric(table.frame_num,errors='raise').to_numpy(float)
            if not np.isfinite(frames).all() or np.any(frames<1) or not np.equal(frames,np.floor(frames)).all() or len(np.unique(frames))!=len(frames):
                raise ValueError(f'Frames must be unique positive integers: {path}')
            pose=Path(pose_dir)/f'{path.stem}.npz' if pose_dir is not None else None
            rows.append(dict(sequence_id=path.stem,label=directory.name,binary_label=int(directory.name!='normal'),
                source_id=source,csv_path=str(path),csv_sha256=file_sha256(path),n_frames=len(table),
                first_frame=int(frames.min()),last_frame=int(frames.max()),url=table.iloc[0].url,
                camera_views=','.join(sorted(set(table.cam_view))),dataset_groups='|'.join(sorted(set(table.dataset))),
                prior_use=('previous_study' if source in prior else 'not_in_previous_study') if prior_basis!='unknown' else 'unknown',
                media_status='not_checked',pose_status=('present_unverified' if pose.exists() else 'missing') if pose is not None else 'not_checked',
                exclusion_reason='eligibility_not_evaluated'))
    result=pd.DataFrame(rows)
    if result.sequence_id.duplicated().any():raise ValueError('Sequence occurs in multiple folders')
    source_table(result)
    if inspect_media:
        paths={s:find_video(root,s) for s in result.source_id.unique()}
        result['video_path']=result.source_id.map(lambda s:str(paths[s]) if paths[s] else '')
        result['media_status']=result.source_id.map(lambda s:'decodable_first_frame' if paths[s] else 'missing_or_undecodable')
    result.attrs.update(manifest_sha256=_digest({'annotations':manifest_digest(result),'binary_rule':BINARY_RULE}),binary_rule=BINARY_RULE,prior_use_basis=prior_basis)
    return result


def load_study_windows(manifest,cache_dir,**window_kwargs):
    """Apply unchanged QC before folds. Return X, metadata, full eligibility ledger.

    The old loader sees a temporary single split and no five-class assertion.
    Its old split assignment is never reused. All existing pose bytes enter the
    study identity, including currently excluded poses.
    """
    if 'require_coverage' in window_kwargs:raise ValueError('Study eligibility precedes fold assignment')
    temporary=manifest.copy();temporary['split']='train'
    x,meta=load_windows(temporary,cache_dir,require_coverage=False,**window_kwargs)
    attrs=dict(meta.attrs)
    meta=meta.drop(columns=['split']).merge(manifest[['sequence_id','binary_label','camera_views','prior_use']],on='sequence_id',how='left',validate='many_to_one',sort=False)
    meta.insert(0,'window_index',np.arange(len(meta),dtype=np.int64))
    ledger=manifest.copy();counts=meta.groupby('sequence_id').size()
    ledger['n_windows']=ledger.sequence_id.map(counts).fillna(0).astype(int);ledger['eligible']=ledger.n_windows.gt(0)
    reasons={r['sequence_id']:r['reason'] for r in attrs['exclusions']}
    ledger['exclusion_reason']=ledger.sequence_id.map(reasons).fillna('none')
    ledger['pose_status']=ledger.sequence_id.map(lambda s:'present_checked' if (Path(cache_dir)/f'{s}.npz').exists() else 'missing')
    poses={s:file_sha256(Path(cache_dir)/f'{s}.npz') for s in sorted(manifest.sequence_id) if (Path(cache_dir)/f'{s}.npz').exists()}
    identity=_digest(dict(annotations=manifest_digest(manifest),binary_rule=BINARY_RULE,pose_sha256=poses,preprocessing_sha256=attrs['corpus_sha256'],eligible_sequences=sorted(meta.sequence_id.unique())))
    attrs.update(corpus_sha256=identity,binary_rule=BINARY_RULE,study_manifest_sha256=manifest.attrs.get('manifest_sha256',manifest_digest(manifest)))
    meta.attrs.update(attrs);ledger.attrs.update(attrs)
    return x,meta,ledger


def make_outer_folds(eligible_metadata,seed=42,n_folds=4,inner_normal_count=5):
    """Freeze source folds and inner roles before fitting any model.

    Normal and abnormal test counts each differ by at most one. A deterministic
    greedy multilabel assignment spreads rare abnormal category memberships.
    Exactly one assignment is made for a source with several abnormal labels.
    """
    if n_folds!=4 or inner_normal_count<4:raise ValueError('Four folds and at least four inner normals are required')
    if 'eligible' in eligible_metadata and not eligible_metadata.eligible.all():
        raise ValueError('Filter to eligible observations before fold assignment')
    table=source_table(eligible_metadata)
    normals=table.loc[table.binary_label.eq(0),'source_id'].to_numpy()
    abnormal=table.loc[table.binary_label.eq(1)].copy()
    if len(normals)<n_folds or len(abnormal)<n_folds:raise ValueError('Every fold needs binary source support')
    rng=np.random.default_rng(seed);normal_parts=np.array_split(rng.permutation(normals),n_folds)
    if min(len(normals)-len(part) for part in normal_parts)<=inner_normal_count:
        raise ValueError('Insufficient normal sources for separate fit and inner reference roles')
    assignment={source:fold for fold,part in enumerate(normal_parts) for source in part}
    categories=[label for label in STUDY_LABELS if label!='normal']
    memberships={row.source_id:tuple(row.categories.split('|')) for row in abnormal.itertuples()}
    totals=np.array([sum(label in values for values in memberships.values()) for label in categories],float)
    weights=1/np.maximum(totals,1)
    ties={source:float(rng.random()) for source in sorted(memberships)}
    order=sorted(memberships,key=lambda s:(-sum(weights[categories.index(c)] for c in memberships[s]),ties[s],s))
    quotas=np.full(n_folds,len(abnormal)//n_folds,int)
    quotas[rng.permutation(n_folds)[:len(abnormal)%n_folds]]+=1
    counts=np.zeros(n_folds,int);category_counts=np.zeros((n_folds,len(categories)),int)
    for source in order:
        vector=np.array([label in memberships[source] for label in categories],int)
        candidates=np.flatnonzero(counts<quotas)
        costs={int(f):float(np.sum(weights*((category_counts[f]+vector-totals/n_folds)**2-(category_counts[f]-totals/n_folds)**2)))
               +float((counts[f]+1)**2-counts[f]**2)/max(len(abnormal),1) for f in candidates}
        chosen=min(candidates,key=lambda f:(costs[int(f)],counts[f]/quotas[f],int(f)))
        assignment[source]=int(chosen);counts[chosen]+=1;category_counts[chosen]+=vector
    table['outer_fold']=table.source_id.map(assignment).astype(int)
    for fold in range(n_folds):
        choices=sorted(table.loc[table.binary_label.eq(0)&table.outer_fold.ne(fold),'source_id'],
                       key=lambda s:_digest([int(seed),int(fold),'inner_normal_reference',s]))
        validation=set(choices[:inner_normal_count])
        table[f'role_fold_{fold}']=[('test' if row.outer_fold==fold else 'val' if row.source_id in validation
                                   else 'train' if row.binary_label==0 else 'auxiliary') for row in table.itertuples()]
    table.attrs.update(seed=seed,n_folds=n_folds,inner_normal_count=inner_normal_count,
                       registry_sha256=_digest(table.sort_values('source_id').to_dict('records')),
                       eligibility_corpus_sha256=eligible_metadata.attrs.get('corpus_sha256','not_supplied'))
    return table


def fold_metadata(metadata,registry,outer_fold,seed=None,inner_normal_count=None):
    """Apply frozen roles. Non-test abnormal sources are auxiliary, never fitted."""
    role=f'role_fold_{int(outer_fold)}'
    if role not in registry or registry.source_id.duplicated().any():raise ValueError('Malformed source registry or unknown fold')
    if not set(metadata.source_id).issubset(set(registry.source_id)):raise ValueError('Unregistered source')
    if seed is not None and 'seed' in registry.attrs and seed!=registry.attrs['seed']:raise ValueError('Registry seed is already frozen')
    if inner_normal_count is not None and 'inner_normal_count' in registry.attrs and inner_normal_count!=registry.attrs['inner_normal_count']:
        raise ValueError('Inner reference size is already frozen')
    if not {'binary_label','outer_fold'}.issubset(registry):raise ValueError('Registry lacks source labels or outer folds')
    assigned=registry[role]
    if not set(assigned).issubset({'train','val','test','auxiliary'}):raise ValueError('Unknown scientific role')
    if not assigned.eq('test').equals(registry.outer_fold.eq(outer_fold)):
        raise ValueError('Outer-test role disagrees with the frozen source fold')
    outside=registry.outer_fold.ne(outer_fold)
    if (outside & registry.binary_label.eq(1) & assigned.ne('auxiliary')).any():
        raise ValueError('Non-test abnormal sources must remain auxiliary')
    if (outside & registry.binary_label.eq(0) & ~assigned.isin(['train','val'])).any():
        raise ValueError('Non-test normal sources need distinct train or reference roles')
    if (assigned.eq('val') & registry.binary_label.eq(0)).sum()<4:
        raise ValueError('At least four inner normal validation sources are required')
    observed=source_table(metadata).set_index('source_id').binary_label
    declared=registry.set_index('source_id').binary_label.reindex(observed.index)
    if not observed.eq(declared).all():raise ValueError('Registry labels differ from fresh metadata')
    out=metadata.drop(columns=['split','outer_fold'],errors='ignore').copy();lookup=registry.set_index('source_id')
    out['split']=out.source_id.map(lookup[role]);out['outer_fold']=int(outer_fold)
    out['fit_eligible']=out.split.eq('train')&out.binary_label.eq(0)
    out['reference_eligible']=out.split.eq('val')&out.binary_label.eq(0)
    if out.groupby('source_id').split.nunique().max()>1 or out.loc[out.split.isin(['train','val']),'binary_label'].ne(0).any():
        raise ValueError('Source overlap or abnormal fitting/reference source')
    out.attrs.update(metadata.attrs)
    out.attrs['registry_sha256']=registry.attrs.get('registry_sha256',_digest(registry.to_dict('records')))
    return out
