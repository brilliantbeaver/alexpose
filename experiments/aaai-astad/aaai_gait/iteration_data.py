"""Fresh outcome-independent cohort, recording groups, and roles for iteration 02.

Source IDs identify recordings, not verified participants. Historical uniform
16-mask/seed17 support is retained with lower-limb readouts 25 through 32.
The separate 48-frame timing grid never changes the primary 32-frame cohort.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .data import JOINTS, POSE_VERSION, _resample, file_sha256, find_video, load_pose, normalize_window
from .model import uniform_mask
from .study_data import build_study_manifest, load_study_windows, make_outer_folds, source_table

READOUT_JOINTS = tuple(range(25,33))
FOCAL_LABELS = ('normal','parkinsons','stroke','cerebral palsy','myopathic')
VERSION = 'iteration02_lowerlimb_u16_seed17_separate_parent48_v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def historical_mask_bank():
    return uniform_mask(16,128,.5,generator=torch.Generator().manual_seed(17)).numpy().reshape(16,8,16)


@dataclass
class IterationData:
    x: np.ndarray
    metadata: pd.DataFrame
    manifest: pd.DataFrame
    sequence_ledger: pd.DataFrame
    source_registry: pd.DataFrame
    parent_raw: np.ndarray | None
    parent_times: np.ndarray | None
    parent_x: np.ndarray | None
    parent_pelvis: np.ndarray | None
    parent_metadata: pd.DataFrame
    provenance: dict
    media_audit: pd.DataFrame
    near_duplicate_candidates: pd.DataFrame
    source_eligibility: pd.DataFrame

    def fold_metadata(self,fold):
        return iteration_fold_metadata(self.metadata,self.source_registry,fold)


def audit_media(root,manifest,workers=4):
    """Hash actual video bytes; thumbnails screen footage, never person identity."""
    import cv2
    def one(source):
        path=find_video(root,source)
        row=dict(source_id=source,video_path='',video_sha256='',media_status='missing_or_undecodable',
                 video_bytes=0,source_fps=np.nan,video_frames=0,duration_seconds=np.nan,
                 thumbnail_hashes='',thumbnail_dhashes='',thumbnail_unique=0)
        if path is None:return row
        row.update(video_path=str(path),video_sha256=file_sha256(path),video_bytes=path.stat().st_size,media_status='decodable_first_frame')
        cap=cv2.VideoCapture(str(path))
        try:
            fps=float(cap.get(cv2.CAP_PROP_FPS));n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            row.update(source_fps=fps,video_frames=n,duration_seconds=n/fps if fps>0 else np.nan)
            hashes,dh=[],[]
            for fraction in (.1,.3,.5,.7,.9):
                cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,min(n-1,int(n*fraction))))
                ok,frame=cap.read()
                if not ok:hashes.append('');dh.append('');continue
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                hashes.append(hashlib.sha256(cv2.resize(gray,(64,64),interpolation=cv2.INTER_AREA).tobytes()).hexdigest())
                small=cv2.resize(gray,(9,8),interpolation=cv2.INTER_AREA)
                number=0
                for bit in (small[:,1:]>small[:,:-1]).reshape(-1):number=(number<<1)|int(bit)
                dh.append(f'{number:016x}')
            row.update(thumbnail_hashes='|'.join(hashes),thumbnail_dhashes='|'.join(dh),thumbnail_unique=len(set(h for h in hashes if h)))
        finally:cap.release()
        return row
    with ThreadPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(one,sorted(manifest.source_id.unique())))
    audit=pd.DataFrame(rows);candidates=[];usable=audit[audit.thumbnail_unique.ge(3)].to_dict('records')
    for i,a in enumerate(usable):
        for b in usable[i+1:]:
            distances=[(int(x,16)^int(y,16)).bit_count() for x,y in zip(a['thumbnail_dhashes'].split('|'),b['thumbnail_dhashes'].split('|')) if x and y]
            if len(distances)>=3 and sum(d<=3 for d in distances)>=3:
                candidates.append(dict(source_a=a['source_id'],source_b=b['source_id'],median_dhash_distance=float(np.median(distances)),
                    close_sample_pairs=sum(d<=3 for d in distances),exact_media_bytes=a['video_sha256']==b['video_sha256'],review_status='unreviewed_footage_candidate'))
    return audit,pd.DataFrame(candidates,columns=['source_a','source_b','median_dhash_distance','close_sample_pairs','exact_media_bytes','review_status'])


def lower_limb_eligibility(x,metadata):
    """Historical finite-target/two-own-context predicate; deduplicate target times."""
    if x.shape!=(len(metadata),32,16,2):raise ValueError('Expected aligned [N,32,16,2]')
    valid=np.isfinite(x.reshape(len(x),8,4,16,2)).all(axis=(2,4));bank=historical_mask_bank()
    exposure=np.zeros_like(valid,dtype=np.int16)
    for mask in bank:
        count=(valid & ~mask).sum(axis=1)
        exposure+=mask & valid & (count[:,None,:]>=2)
    starts=np.array([int(str(v).rsplit(':',1)[1]) for v in metadata.window_id]);sequences=metadata.sequence_id.to_numpy()
    records=[]
    for source,indexes in metadata.groupby('source_id',sort=True).indices.items():
        group=metadata.iloc[indexes]
        row=dict(source_id=str(source),binary_label=int(group.binary_label.iloc[0]),categories='|'.join(sorted(set(group.label))),
            n_sequences=int(group.sequence_id.nunique()),n_windows=len(group),requested_target_exposures=0,valid_target_exposures=0)
        for joint in READOUT_JOINTS:
            slot=JOINTS.index(joint);unique={(str(sequences[ix]),int(starts[ix]+patch*4)) for ix in indexes for patch in np.flatnonzero(exposure[ix,:,slot]>0)}
            row[f'joint_{joint}_distinct_patches']=len(unique)
            row['requested_target_exposures']+=int(bank[:,:,slot].sum())*len(group)
            row['valid_target_exposures']+=int(exposure[indexes,:,slot].sum())
        row['profile_eligible']=all(row[f'joint_{j}_distinct_patches']>=4 for j in READOUT_JOINTS)
        row['exclusion_reason']='none' if row['profile_eligible'] else 'insufficient_lower_limb_target_support'
        records.append(row)
    ledger=pd.DataFrame(records)
    return metadata.source_id.isin(ledger.loc[ledger.profile_eligible,'source_id']).to_numpy(),ledger


def make_source_registry(metadata,media_audit,seed=42,confirmed_duplicate_groups=None):
    """Group exact-byte and explicitly confirmed footage duplicates in every role."""
    sources=source_table(metadata);media=media_audit.set_index('source_id');parents={s:s for s in sources.source_id}
    def find(s):
        while parents[s]!=s:parents[s]=parents[parents[s]];s=parents[s]
        return s
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parents[max(a,b)]=min(a,b)
    hashes={}
    for s in parents:
        h=str(media.loc[s,'video_sha256'])
        if not h:raise ValueError(f'Unverified media for retained source {s}')
        if h in hashes:union(s,hashes[h])
        else:hashes[h]=s
    for members in confirmed_duplicate_groups or []:
        members=list(map(str,members))
        if not set(members).issubset(parents):raise ValueError('Unknown duplicate source')
        for s in members[1:]:union(members[0],s)
    sources['cluster_id']=sources.source_id.map(lambda s:'recording:'+find(s))
    grouped=metadata.copy();grouped['source_id']=grouped.source_id.map(sources.set_index('source_id').cluster_id)
    # This rejects normal/abnormal label conflicts inside confirmed duplicate groups.
    roles=make_outer_folds(grouped,seed=seed,n_folds=4,inner_normal_count=5).rename(columns={'source_id':'cluster_id'})
    result=sources.merge(roles[['cluster_id','outer_fold']+[f'role_fold_{f}' for f in range(4)]],on='cluster_id',validate='many_to_one')
    for fold in range(4):
        role=f'role_fold_{fold}';result[role]=result[role].replace({'val':'reference'})
        groups=sorted(set(result.loc[result[role].eq('train'),'cluster_id']))
        if len(groups)<3:raise ValueError('Insufficient normal groups for three internal folds')
        parts=np.array_split(np.random.default_rng(seed+10000+fold).permutation(groups),3)
        assignments={g:k for k,part in enumerate(parts) for g in part}
        result[f'inner_fold_{fold}']=result.cluster_id.map(assignments).fillna(-1).astype(int)
    result['video_sha256']=result.source_id.map(media.video_sha256)
    result['duplicate_group_size']=result.groupby('cluster_id').source_id.transform('size')
    recording_categories=result.groupby('cluster_id').categories.agg(lambda values:'|'.join(sorted({label for value in values for label in value.split('|')})))
    result['recording_categories']=result.cluster_id.map(recording_categories)
    result['is_focal']=result.recording_categories.map(lambda v:bool(set(v.split('|')) & set(FOCAL_LABELS)))
    result['five_class_eligible']=result.recording_categories.isin(FOCAL_LABELS)
    result['five_class_label']=result.recording_categories.where(result.five_class_eligible,'')
    result['label_conflict']=result.recording_categories.str.contains('|',regex=False)
    result['five_class_exclusion_reason']=np.where(result.five_class_eligible,'none',np.where(result.label_conflict,'multiple_source_category_memberships','outside_five_focal_labels'))
    if 'prior_use' in metadata:result['prior_use']=result.source_id.map(metadata.groupby('source_id').prior_use.first())
    result.attrs.update(seed=seed,n_folds=4,reference_normal_count=5,internal_folds=3,
        grouping='source_id_plus_confirmed_recording_duplicates_not_participants',registry_sha256=_digest(result.sort_values('source_id').to_dict('records')))
    return result


def iteration_fold_metadata(metadata,registry,fold):
    if fold not in range(4):raise ValueError('Outer fold must be 0 through 3')
    lookup=registry.set_index('source_id')
    if not set(metadata.source_id).issubset(lookup.index):raise ValueError('Unregistered window')
    out=metadata.copy()
    for key in ['cluster_id',f'role_fold_{fold}',f'inner_fold_{fold}','is_focal','five_class_eligible','five_class_label']:out[key]=out.source_id.map(lookup[key])
    out['split']=out[f'role_fold_{fold}'];out['inner_fold']=out[f'inner_fold_{fold}'];out['outer_fold']=fold
    if out.groupby('cluster_id').split.nunique().max()!=1:raise ValueError('Duplicate group crosses roles')
    if out.loc[out.split.isin(['train','reference']),'binary_label'].ne(0).any():raise ValueError('Abnormal fitting/reference source')
    out.attrs.update(metadata.attrs,registry_sha256=registry.attrs.get('registry_sha256'))
    return out


def normalization_parameters(raw32):
    """Expose exactly normalize_window's fixed first-half scale and frame anchors."""
    raw=np.asarray(raw32,np.float32)
    if raw.shape!=(32,16,2):raise ValueError('Expected raw central [32,16,2]')
    pelvis=(raw[:,6]+raw[:,7])/2;shoulder=(raw[:,0]+raw[:,1])/2
    torso=np.linalg.norm(shoulder[:16]-pelvis[:16],axis=-1);good=torso[np.isfinite(torso)&(torso>1e-6)]
    return (float(np.median(good)) if len(good) else np.nan),pelvis


def _view_info(annotations,start_seconds,end_seconds,source_fps):
    frames=annotations.frame_num.to_numpy(int)
    low=int(np.floor(start_seconds*source_fps+1+1e-7));high=int(np.ceil(end_seconds*source_fps+1-1e-7))
    selected=annotations[(frames>=low)&(frames<=high)];views=sorted(set(selected.cam_view.astype(str)))
    events=selected[selected.gait_event.astype(str).str.strip().ne('')]
    return dict(camera_views=','.join(views),camera_view_mixed=len(views)>1,source_frame_first=low,source_frame_last=high,
        annotation_event_count=len(events),annotation_event_types='|'.join(sorted(set(events.gait_event.astype(str)))))


def _quality(x):
    v=np.isfinite(x.reshape(len(x)//4,4,16,2)).all(axis=(1,3));half=len(v)//2
    return bool(v.mean()>=.75 and min(v[:half].mean(),v[half:].mean())>=.5),float(v.mean())


def enrich_and_build_parents(metadata,manifest,pose_dir,include_parents):
    """Build separate stride16 parent48 grid; never shift primary window starts."""
    meta=metadata.copy();lookup=manifest.set_index('sequence_id');extra=[]
    raw_parents=[];time_parents=[];central=[];pelvis_values=[];parent_rows=[];rejects={}
    # Include all verified sequences from retained sources, independently of
    # whether a sequence contributed a historical stride16 primary window.
    selected=manifest[manifest.source_id.isin(set(meta.source_id))]
    grouped=meta.groupby('sequence_id',sort=False).indices
    for r in selected.itertuples(index=False):
        pose=load_pose(Path(pose_dir)/f'{r.sequence_id}.npz');fps=float(pose['fps'])
        if not np.isfinite(fps) or fps<15:continue
        raw=pose['raw_xy'].copy();raw[(~np.isfinite(pose['confidence']))|(pose['confidence']<.5)]=np.nan
        xy,times=_resample(raw[:,JOINTS],pose['frame_numbers'],fps,15,1)
        annotations=pd.read_csv(r.csv_path,keep_default_na=False).sort_values('frame_num');source_frames=annotations.frame_num.to_numpy(int)
        for index in grouped.get(r.sequence_id,[]):
            w=meta.iloc[index];start=int(str(w.window_id).rsplit(':',1)[1]);scale,_=normalization_parameters(xy[start:start+32])
            extra.append(dict(window_index=int(index),sequence_camera_views=str(w.camera_views),normalization_scale=scale,
                              **_view_info(annotations,w.start_seconds,w.end_seconds,fps)))
        if not include_parents:continue
        for start in range(0,len(xy)-48+1,16):
            p=xy[start:start+48];pt=times[start:start+48];c=p[8:40];scale,pelvis=normalization_parameters(c)
            normalized=normalize_window(c)
            pn=(p-(p[:,6]+p[:,7])[:,None]/2)/scale
            qp,coverage=_quality(pn);qc,ccoverage=_quality(normalized)
            lo=int(np.floor(pt[0]*fps+1+1e-7));hi=int(np.ceil(pt[-1]*fps+1-1e-7))
            covering=source_frames[(source_frames>=lo)&(source_frames<=hi)]
            continuous=len(covering)>0 and np.all(np.diff(covering)==1)
            reason='none' if qp and qc and continuous else 'annotation_gap' if not continuous else 'parent_quality' if not qp else 'central_quality'
            if reason!='none':rejects[reason]=rejects.get(reason,0)+1;continue
            index=len(raw_parents);raw_parents.append(p);time_parents.append(pt);central.append(normalized);pelvis_values.append(pelvis)
            parent_rows.append(dict(window_index=index,timing_window_index=index,window_id=f'{r.sequence_id}:parent48:{start}:central32',
                parent_window_id=f'{r.sequence_id}:{start}',sequence_id=r.sequence_id,source_id=r.source_id,label=r.label,binary_label=r.binary_label,
                prior_use=r.prior_use,sequence_camera_views=r.camera_views,start_seconds=float(pt[8]),end_seconds=float(pt[39]),
                parent_start_seconds=float(pt[0]),parent_end_seconds=float(pt[-1]),normalization_scale=scale,fps=15.,source_fps=fps,
                valid_token_fraction=ccoverage,parent_valid_token_fraction=coverage,
                **_view_info(annotations,pt[8],pt[39],fps)))
    columns=[c for c in pd.DataFrame(extra) if c!='window_index' and c in meta]
    attrs=dict(meta.attrs);meta=meta.drop(columns=columns).merge(pd.DataFrame(extra),on='window_index',validate='one_to_one',sort=False);meta.attrs.update(attrs)
    pm=pd.DataFrame(parent_rows);pm.attrs['exclusions_by_reason']=rejects
    if not include_parents:return meta,None,None,None,None,pm
    def array(values,shape,dtype):return np.stack(values).astype(dtype) if values else np.empty(shape,dtype)
    return meta,array(raw_parents,(0,48,16,2),np.float32),array(time_parents,(0,48),np.float64),array(central,(0,32,16,2),np.float32),array(pelvis_values,(0,32,2),np.float32),pm


def build_iteration_data(repo_root,*,seed=42,output_dir=None,include_parents=True,confirmed_duplicate_groups=None,media_workers=4):
    root=Path(repo_root).resolve();base=root/'experiments/aaai';pose_dir=base/'artifacts/poses'
    manifest=build_study_manifest(root,pose_dir=pose_dir);media,candidates=audit_media(root,manifest,media_workers);mi=media.set_index('source_id')
    cache_rows=[]
    for r in manifest.itertuples(index=False):
        path=pose_dir/f'{r.sequence_id}.npz';reason='none';ph=''
        if not path.exists():reason='pose_missing'
        else:
            ph=file_sha256(path);p=json.loads(str(load_pose(path)['provenance']));actual=mi.loc[r.source_id,'video_sha256']
            if not actual:reason='pose_media_unverified'
            elif p.get('video_sha256')!=actual:reason='pose_media_hash_mismatch'
            elif p.get('csv_sha256')!=r.csv_sha256 or p.get('source_id')!=r.source_id or p.get('sequence_id')!=r.sequence_id:reason='pose_annotation_identity_mismatch'
            elif p.get('version')!=POSE_VERSION or p.get('frame_number_base')!=1 or p.get('min_visibility')!=.5 or p.get('bbox_margin')!=.05:reason='pose_configuration_mismatch'
        cache_rows.append(dict(sequence_id=r.sequence_id,pose_sha256=ph,cache_verified=reason=='none',cache_exclusion_reason=reason))
    cache=pd.DataFrame(cache_rows);safe=manifest[manifest.sequence_id.isin(cache.loc[cache.cache_verified,'sequence_id'])].copy();safe.attrs.update(manifest.attrs)
    x,meta,load_ledger=load_study_windows(safe,pose_dir);pre=meta.copy();keep,eligibility=lower_limb_eligibility(x,meta)
    meta=meta.loc[keep].copy().reset_index(drop=True);meta['pre_profile_window_index']=meta.window_index;meta['window_index']=np.arange(len(meta),dtype=np.int64);x=x[keep]
    meta,pr,pt,px,pp,pm=enrich_and_build_parents(meta,safe,pose_dir,include_parents)
    reviewed_groups=confirmed_duplicate_groups or []
    if not {s for group in reviewed_groups for s in group}.issubset(set(manifest.source_id)):
        raise ValueError('Reviewed duplicate source is absent from the fresh inventory')
    retained_ids=set(meta.source_id)
    retained_groups=[[s for s in group if s in retained_ids] for group in reviewed_groups]
    retained_groups=[group for group in retained_groups if len(group)>1]
    registry=make_source_registry(meta,media,seed,retained_groups)
    ledger=manifest.merge(cache,on='sequence_id',validate='one_to_one');pc=pre.groupby('sequence_id').size();ac=meta.groupby('sequence_id').size()
    ledger['pre_profile_windows']=ledger.sequence_id.map(pc).fillna(0).astype(int);ledger['retained_windows']=ledger.sequence_id.map(ac).fillna(0).astype(int);ledger['eligible']=ledger.retained_windows.gt(0)
    reasons=load_ledger.set_index('sequence_id').exclusion_reason
    ledger['exclusion_reason']=['none' if r.eligible else r.cache_exclusion_reason if not r.cache_verified else 'insufficient_source_lower_limb_support' if r.pre_profile_windows else reasons.get(r.sequence_id,'no_valid_windows') for r in ledger.itertuples()]
    fold_counts=[]
    for f in range(4):
        for (role,binary),g in registry.groupby([f'role_fold_{f}','binary_label']):fold_counts.append(dict(outer_fold=f,role=role,binary_label=int(binary),sources=len(g),recording_clusters=int(g.cluster_id.nunique())))
    prov=dict(version=VERSION,created_at_utc=datetime.now(timezone.utc).isoformat(),seed=seed,manifest_sha256=manifest.attrs['manifest_sha256'],
        window_corpus_sha256=pre.attrs['corpus_sha256'],mask_bank_sha256=hashlib.sha256(historical_mask_bank().tobytes()).hexdigest(),source_registry_sha256=registry.attrs['registry_sha256'],
        code_sha256={str(p.relative_to(root)):file_sha256(p) for p in [Path(__file__),base/'aaai_gait/data.py',base/'aaai_gait/study_data.py',base/'aaai_gait/model.py']},
        pose_corpus_sha256=_digest(cache[['sequence_id','pose_sha256']].to_dict('records')),video_corpus_sha256=_digest(media[['source_id','video_sha256']].to_dict('records')),
        fresh_inventory=dict(sequences=len(manifest),rows=int(manifest.n_frames.sum()),sources=int(manifest.source_id.nunique()),normal_sources=int(manifest.loc[manifest.label.eq('normal'),'source_id'].nunique())),
        pre_profile=dict(windows=len(pre),sequences=int(pre.sequence_id.nunique()),sources=int(pre.source_id.nunique())),
        retained=dict(windows=len(meta),sequences=int(meta.sequence_id.nunique()),sources=len(registry),recording_clusters=int(registry.cluster_id.nunique()),normal_sources=int(registry.binary_label.eq(0).sum()),focal_sources=int(registry.is_focal.sum())),
        parent48=dict(windows=len(pm),sources=int(pm.source_id.nunique()) if len(pm) else 0,normal_sources=int(pm.loc[pm.label.eq('normal'),'source_id'].nunique()) if len(pm) else 0,exclusions_by_reason=pm.attrs.get('exclusions_by_reason',{})),
        exact_duplicate_clusters=int(registry.video_sha256.value_counts().gt(1).sum()),
        confirmed_recording_duplicate_clusters=int(registry.loc[registry.duplicate_group_size.gt(1),'cluster_id'].nunique()),
        confirmed_duplicate_groups_inventory=reviewed_groups,
        confirmed_duplicate_groups_retained=retained_groups,
        near_duplicate_screen=dict(candidates=len(candidates),status='thumbnail_footage_screen_not_person_identity_not_exhaustive'),
        by_label=meta.groupby('label').agg(sources=('source_id','nunique'),sequences=('sequence_id','nunique'),windows=('window_id','size')).reset_index().to_dict('records'),fold_counts=fold_counts,
        readout_joints=list(READOUT_JOINTS),input_mask_joints=list(JOINTS),mask_seed=17,mask_repeats=16,source_not_participant=True,
        all_normal_sources_previously_explored=bool(meta.loc[meta.label.eq('normal'),'prior_use'].eq('previous_study').all()),pose_exclusions=cache.groupby('cache_exclusion_reason').size().to_dict(),
        raw_parent_contract='Separate stride16 grid; original32=parent[8:40]; selected raw XY confidence>=0.5; fixed central first-half torso scale and central per-frame pelvis anchors',
        five_class_conflicts='exclude_sources_with_multiple_category_memberships')
    prov['cohort_sha256']=_digest({k:v for k,v in prov.items() if k!='created_at_utc'})
    data=IterationData(x,meta,manifest,ledger,registry,pr,pt,px,pp,pm,prov,media,candidates,eligibility)
    if output_dir is not None:save_iteration_data(data,output_dir,pre_metadata=pre,cache_audit=cache)
    return data


def save_iteration_data(data,output_dir,*,pre_metadata=None,cache_audit=None):
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);identity=out/'provenance.json'
    if identity.exists() and json.loads(identity.read_text()).get('cohort_sha256')!=data.provenance['cohort_sha256']:raise ValueError('Refusing to overwrite a different cohort identity')
    for name,table in [('manifest',data.manifest),('window_index',data.metadata),('sequence_ledger',data.sequence_ledger),('source_registry',data.source_registry),('media_audit',data.media_audit),('near_duplicate_candidates',data.near_duplicate_candidates),('source_eligibility',data.source_eligibility),('timing_window_index',data.parent_metadata),('window_index_before_profile_qc',pre_metadata),('pose_cache_audit',cache_audit)]:
        if table is not None:table.to_csv(out/f'{name}.csv',index=False)
    np.save(out/'windows.npy',data.x);np.save(out/'evaluation_mask_bank.npy',historical_mask_bank())
    for name,value in [('parent_raw48',data.parent_raw),('parent_times48',data.parent_times),('parent_central32',data.parent_x),('parent_central_pelvis',data.parent_pelvis)]:
        if value is not None:np.save(out/f'{name}.npy',value)
    for name,value in [('provenance',data.provenance),('registry_attributes',data.source_registry.attrs)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')
    (out/'artifact_hashes.json').write_text(json.dumps({p.name:file_sha256(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='artifact_hashes.json'},indent=2)+'\n')
    return out


def load_iteration_data(output_dir):
    """Verify a frozen cohort; this does not replace the fresh build's inventory."""
    out=Path(output_dir);hashes=json.loads((out/'artifact_hashes.json').read_text())
    for name,digest in hashes.items():
        if file_sha256(out/name)!=digest:raise ValueError(f'Changed cohort artifact {name}')
    def table(name):
        path=out/f'{name}.csv'
        return pd.read_csv(path,keep_default_na=False) if path.stat().st_size>1 else pd.DataFrame()
    def array(name):return np.load(out/f'{name}.npy') if (out/f'{name}.npy').exists() else None
    registry=table('source_registry');registry.attrs.update(json.loads((out/'registry_attributes.json').read_text()))
    return IterationData(array('windows'),table('window_index'),table('manifest'),table('sequence_ledger'),registry,
        array('parent_raw48'),array('parent_times48'),array('parent_central32'),array('parent_central_pelvis'),table('timing_window_index'),
        json.loads((out/'provenance.json').read_text()),table('media_audit'),table('near_duplicate_candidates'),table('source_eligibility'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args();data=build_iteration_data(args.root,seed=args.seed,output_dir=args.output)
    print(json.dumps({k:data.provenance[k] for k in ['cohort_sha256','fresh_inventory','pre_profile','retained','parent48','exact_duplicate_clusters','near_duplicate_screen','pose_exclusions','fold_counts']},indent=2))

if __name__=='__main__':main()
