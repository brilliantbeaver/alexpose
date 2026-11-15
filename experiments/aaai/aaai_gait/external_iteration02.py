"""Independent participant method replication on six synchronized PiG angles.

No BlazePose mapping, pooled GAVD participants, weight transfer or JEPA claim.
Selections and protocol precede all external fitted outcomes.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib,json,time,zipfile,zlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .external_data import FILES, _RemoteZip
from .data import _resample, file_sha256
from .retained_prediction import RidgeStats, select_correction_recipe
from .iteration_scoring import NormalDistance, BilateralReference, ranking_metrics, paired_bootstrap
ANGLES=('LHipAngles','RHipAngles','LKneeAngles','RKneeAngles','LAnkleAngles','RAnkleAngles')
ROOT=Path(__file__).resolve().parents[3]
PREP=ROOT/'experiments/aaai/artifacts/iteration02/preparation/external_replication'
DATA=ROOT/'data/external_gait/van_criekinge_2023_figshare_v1'

def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=str)+'\n')
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def angle_mask_bank(repeats=16,seed=17):
    rng=np.random.default_rng(seed);hidden=np.zeros((repeats,48),bool)
    for row in hidden:row[rng.choice(48,24,replace=False)]=True
    return hidden

def angle_patches(x):
    x=np.asarray(x,np.float32)
    if x.ndim!=3 or x.shape[1:]!=(32,6) or np.isinf(x).any():raise ValueError('Expected finite-or-NaN [windows,32,6] angles')
    return x.reshape(-1,8,4,6).transpose(0,1,3,2).reshape(-1,48,4)

def permitted_features(p,hidden,targets):
    """Only visible values enter the bases; targets are separately returned."""
    p=np.asarray(p);hidden=np.asarray(hidden,bool);targets=np.asarray(targets,int);n=len(p)
    if p.shape!=(n,48,4) or hidden.shape!=(n,48) or targets.shape!=(n,):raise ValueError('Bad angle feature shapes')
    if not hidden[np.arange(n),targets].all():raise ValueError('Requested targets must be hidden')
    valid=np.isfinite(p).all(2);visible=~hidden&valid;slots=targets%6;qt=targets//6;side=slots%2;kind=slots//2
    visibility=visible.reshape(n,8,6)
    count=np.stack([visibility[:,:,s::2].sum((1,2)) for s in (0,1)],1)
    times=np.stack([visibility[:,:,s::2].any(2).sum(1) for s in (0,1)],1)
    good=valid[np.arange(n),targets]&(count[np.arange(n),side]>=2)&(count[np.arange(n),1-side]>=2)&(times[np.arange(n),side]>=2)&(times[np.arange(n),1-side]>=2)
    def basis(opposite=False):
        order=np.asarray([[k]+[j for j in range(3) if j!=k] for k in range(3)])[kind];joints=2*order+(side^int(opposite))[:,None]
        grid=p.reshape(n,8,6,4);leg=grid[np.arange(n)[:,None,None],np.arange(8)[None,:,None],joints[:,None,:]]
        allowed=visibility[np.arange(n)[:,None,None],np.arange(8)[None,:,None],joints[:,None,:]];values=np.where(allowed[...,None],leg,0.).astype(float)
        t=np.arange(8)[None,:,None];prev=np.where(allowed&(t<qt[:,None,None]),t,-1).max(1);nxt=np.where(allowed&(t>qt[:,None,None]),t,8).min(1)
        hp,hn=prev>=0,nxt<8;row=np.arange(n)[:,None];role=np.arange(3)[None,:]
        vp=values[row,np.maximum(prev,0),role,3];vn=values[row,np.minimum(nxt,7),role,0]
        frac=(4*qt[:,None,None]+np.arange(4)[None,None,:]-(4*prev+3)[...,None])/np.maximum((4*nxt-(4*prev+3))[...,None],1)
        estimate=vp[...,None]+frac*(vn-vp)[...,None]
        estimate=np.where((hp&hn)[...,None],estimate,np.where(hp[...,None],vp[...,None],np.where(hn[...,None],vn[...,None],0.)))
        same=allowed[np.arange(n),qt];estimate=np.where(same[...,None],values[np.arange(n),qt],estimate)
        support=np.stack([same,hp,hn,allowed.mean(1),np.where(hp,qt[:,None]-prev,0)/8.,np.where(hn,nxt-qt[:,None],0)/8.],2)
        local=np.concatenate([estimate,support],2).reshape(n,30)
        delta=(t-qt[:,None,None])/8.;means=values.mean(3);den=np.maximum(allowed.sum(1),1)
        moments=np.stack([(means*delta**power).sum(1)/den for power in range(4)],2);extra=np.concatenate([moments,support],2).reshape(n,30)
        identity=np.concatenate([np.eye(3)[kind],(2*side-1)[:,None],(qt/7.)[:,None],((qt/7.)**2)[:,None]],1)
        return np.concatenate([local,identity],1),np.concatenate([extra,identity],1)
    own,extra=basis();other,_=basis(True)
    return dict(own=own,own_residual=np.concatenate([own,extra],1),both_residual=np.concatenate([own,other],1),target=p[np.arange(n),targets].astype(float),eligible=good)

def extract_angles(path):
    import c3d
    with Path(path).open('rb') as stream:
        reader=c3d.Reader(stream);labels=[str(s).strip().split(':')[-1] for s in reader.point_labels]
        units=reader.get('POINT:ANGLE_UNITS');membership=reader.get('POINT:ANGLES')
        if units is None or units.string_value.strip().lower()!='deg':raise ValueError('Unverified angle units')
        declared=[str(s).strip().split(':')[-1] for s in membership.string_array] if membership is not None else []
        if not set(ANGLES).issubset(declared) or not set(ANGLES).issubset(labels):raise ValueError('Missing declared angle channels')
        indexes=[labels.index(name) for name in ANGLES];values=[];frames=[]
        for frame,points,_ in reader.read_frames(copy=True):
            selected=points[indexes];a=selected[:,0].copy();a[(selected[:,3]<0)|~np.isfinite(selected[:,:3]).all(1)]=np.nan;values.append(a);frames.append(int(frame))
        rate=float(reader.point_rate)
    if rate<15 or not np.isfinite(rate) or len(frames)<2 or not np.all(np.diff(frames)==1):raise ValueError('Invalid raw angle clock')
    raw=np.asarray(values,np.float32);numbers=np.asarray(frames);sampled,times=_resample(raw[:,:,None],numbers-numbers[0]+1,rate,15,1)
    return sampled[:,:,0],times,dict(raw_frames=len(raw),source_rate_hz=rate,first_raw_frame=int(numbers[0]),last_raw_frame=int(numbers[-1]),raw_sample_span_seconds=float((numbers[-1]-numbers[0])/rate),units='deg',component_index=0)

def participant_roles(eligible):
    table=eligible[['participant_id','group','binary_label']].copy().sort_values('participant_id');table['source_id']=table.participant_id;assignment={}
    for label in [0,1]:
        ids=table.loc[table.binary_label.eq(label),'participant_id'].to_numpy()
        if len(ids)<4:raise ValueError('Need all four participant test folds')
        for fold,part in enumerate(np.array_split(np.random.default_rng(42+label).permutation(ids),4)):
            for source in part:assignment[source]=fold
    table['outer_fold']=table.participant_id.map(assignment)
    for fold in range(4):
        remaining=table[table.binary_label.eq(0)&table.outer_fold.ne(fold)].participant_id.tolist()
        if len(remaining)<26:raise ValueError('Too few normals for20 references and internal folds')
        refs=set(sorted(remaining,key=lambda p:digest([42,fold,'reference',p]))[:20])
        table[f'role_fold_{fold}']=['test' if r.outer_fold==fold else 'reference' if r.participant_id in refs else 'train' if r.binary_label==0 else 'auxiliary' for r in table.itertuples()]
        fits=table.loc[table[f'role_fold_{fold}'].eq('train'),'participant_id'].to_numpy();mapping={s:k for k,part in enumerate(np.array_split(np.random.default_rng(10042+fold).permutation(fits),3)) for s in part}
        table[f'inner_fold_{fold}']=table.participant_id.map(mapping).fillna(-1).astype(int)
    return table.reset_index(drop=True)

def download_and_prepare(prep=PREP,data=DATA,workers=4):
    prep=Path(prep);data=Path(data);proposal=json.loads((prep/'protocol_proposal.json').read_text());selection=pd.read_csv(prep/'selected_trials.csv',keep_default_na=False)
    if file_sha256(prep/'selected_trials.csv')!=proposal['selected_trials_sha256']:raise ValueError('Frozen selected trial list changed')
    def one(row):
        spec=next(f for f in FILES if f['group']==row['group']);path=data/'replication_c3d'/row['group']/row['participant_directory']/Path(row['member']).name
        try:
            if path.exists():
                payload=path.read_bytes()
                if len(payload)!=row['bytes'] or f'{zlib.crc32(payload)&0xffffffff:08x}'!=row['crc32']:raise ValueError('Cached selected member CRC mismatch')
            else:
                remote=_RemoteZip(spec,data/'range_cache')
                with zipfile.ZipFile(remote) as archive:payload=archive.read(row['member'])
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
            sampled,times,audit=extract_angles(path)
            return dict(row,selected_file=str(path),sha256=file_sha256(path),status='parsed',**audit),sampled,times
        except Exception as exc:return dict(row,status='failed',error=f'{type(exc).__name__}: {str(exc).split("https://")[0]}'),None,None
    records=[];arrays={};start=time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,row) for row in selection.to_dict('records')]
        for i,future in enumerate(as_completed(futures),1):
            record,x,t=future.result();records.append(record)
            if x is not None:arrays[record['participant_id']]=(x,t)
            if i%10==0:print(f'External selected trials: {i}/{len(selection)} processed, {time.perf_counter()-start:.1f}s',flush=True)
    records=pd.DataFrame(records).sort_values('participant_id');records.to_csv(prep/'trial_download_ledger.csv',index=False)
    if records.loc[records.status.eq('parsed'),'sha256'].duplicated().any():raise ValueError('Same C3D bytes assigned to multiple participants')
    xs=[];meta=[];ledger=[];bank=angle_mask_bank()
    for row in records.to_dict('records'):
        pid=row['participant_id'];windows=[];starts=[]
        if pid in arrays:
            raw,times=arrays[pid]
            for k in range(0,len(raw)-32+1,16):
                win=raw[k:k+32];v=np.isfinite(win.reshape(8,4,6)).all(1)
                if v.mean()>=.75 and min(v[:4].mean(),v[4:].mean())>=.5:windows.append(win);starts.append(k)
        supported=[set() for _ in range(6)]
        for win,k in zip(windows,starts):
            p=angle_patches(win[None])[0]
            for mask in bank:
                target=np.flatnonzero(mask);f=permitted_features(np.broadcast_to(p,(len(target),48,4)),np.broadcast_to(mask,(len(target),48)),target)
                for q in target[f['eligible']]:supported[q%6].add(k+(q//6)*4)
        eligible=bool(windows and all(len(v)>=4 for v in supported))
        ledger.append(dict(participant_id=pid,group=row['group'],binary_label=int(row['group']=='stroke'),windows_before_support=len(windows),eligible=eligible,
            reason='none' if eligible else row.get('error','insufficient_valid_windows_or_target_support'),**{f'{ANGLES[j]}_distinct_patches':len(supported[j]) for j in range(6)}))
        if not eligible:continue
        for win,k in zip(windows,starts):
            index=len(xs);xs.append(win);meta.append(dict(window_index=index,source_id=pid,participant_id=pid,sequence_id=pid,window_id=f'{pid}:{k}',start_grid_index=k,start_seconds=float(arrays[pid][1][k]),end_seconds=float(arrays[pid][1][k+31]),label='normal' if row['group']=='healthy' else 'stroke',binary_label=int(row['group']=='stroke')))
    ledger=pd.DataFrame(ledger);metadata=pd.DataFrame(meta);x=np.stack(xs);registry=participant_roles(ledger[ledger.eligible])
    np.save(prep/'windows.npy',x);np.save(prep/'mask_bank.npy',bank);metadata.to_csv(prep/'window_index.csv',index=False);ledger.to_csv(prep/'participant_eligibility.csv',index=False);registry.to_csv(prep/'participant_registry.csv',index=False)
    summary=dict(selected=len(selection),parsed=int(records.status.eq('parsed').sum()),eligible=len(registry),by_group=registry.groupby('group').size().to_dict(),windows=len(x),source_unit='dataset participant directory, one frozen walking trial each',trial_selection_sha256=proposal['selected_trials_sha256'],registry_sha256=file_sha256(prep/'participant_registry.csv'),outcomes='none: preparation only',download_failures=records[records.status.ne('parsed')][['participant_id','error']].to_dict('records') if records.status.ne('parsed').any() else [])
    dump(prep/'cohort_preparation.json',summary);dump(prep/'data_hashes.json',{n:file_sha256(prep/n) for n in ['selected_trials.csv','protocol_proposal.json','trial_download_ledger.csv','windows.npy','mask_bank.npy','window_index.csv','participant_eligibility.csv','participant_registry.csv','cohort_preparation.json']})
    print(json.dumps(summary,indent=2),flush=True);return x,metadata,registry

def feature_rows(x,metadata,bank,batch_windows=32):
    p=angle_patches(x)
    for start in range(0,len(x),batch_windows):
        for mid,mask in enumerate(bank):
            targets=np.flatnonzero(mask);wi=np.repeat(np.arange(start,min(start+batch_windows,len(x))),len(targets));ti=np.tile(targets,min(start+batch_windows,len(x))-start)
            feat=permitted_features(p[wi],np.broadcast_to(mask,(len(wi),48)),ti);use=feat['eligible']
            yield pd.DataFrame(dict(window_index=wi[use],angle=ti[use]%6,patch_index=ti[use]//6,mask_id=mid)),{k:v[use] for k,v in feat.items() if k!='eligible'}

def training_features(x,metadata,seed,draws=512):
    if set(metadata.label)!={'normal'}:raise ValueError('External fitting is normal only')
    ids=metadata.source_id.to_numpy();unique=np.unique(ids);rng=np.random.default_rng(seed)
    schedule=np.concatenate([rng.permutation(unique) for _ in range((draws+len(unique)-1)//len(unique))])[:draws]
    wi=np.asarray([rng.choice(np.flatnonzero(ids==s)) for s in schedule]);masks=angle_mask_bank(draws,seed+100000);presentation,target=np.nonzero(masks)
    p=angle_patches(x);f=permitted_features(p[wi[presentation]],masks[presentation],target);use=f.pop('eligible');f={k:v[use] for k,v in f.items()}
    sources=ids[wi[presentation[use]]];joints=target[use]%6;counter=pd.DataFrame(dict(source=sources,angle=joints)).groupby(['source','angle']).size()
    if set(counter.index.get_level_values(0))!=set(unique) or not counter.groupby(level=0).size().eq(6).all():raise ValueError('Insufficient training support; no favorable resampling')
    weights=np.asarray([len(sources)/(len(unique)*6*counter.loc[(s,j)]) for s,j in zip(sources,joints)])
    return f,weights,dict(participants=unique.tolist(),presentations=draws,rows=len(sources),seed=seed,feature_dimension=36,residual_dimension=72)

def fit_models(x,metadata,seed):
    f,w,audit=training_features(x,metadata,seed);base_stats=RidgeStats(36,outputs=4);base_stats.add(f['own'],f['target'],w);base=base_stats.fit(1.)
    residual=f['target']-base.predict(f['own']);stats={};grid={}
    for branch in ['own','both']:
        s=RidgeStats(72,outputs=4);s.add(f[branch+'_residual'],residual,w);stats[branch]=s;grid[branch]={a:s.fit(a) for a in [.1,1.,10.]}
    audit.update(base_rank=base.feature_rank,residual_ranks={b:{a:m.feature_rank for a,m in ms.items()} for b,ms in grid.items()},training_risks={b:{str(a):stats[b].risk(m) for a,m in ms.items()} for b,ms in grid.items()},zero_training_risk=stats['own'].risk())
    return dict(base=base,grid=grid,audit=audit)

def internal_report(models,x,metadata,internal):
    sums={('base',0.,1.):np.zeros((len(x),6))};counts=np.zeros((len(x),6))
    for b in ['own','both']:
        for a in [.1,1.,10.]:
            for gate in [0.,.5,1.]:sums[b,a,gate]=np.zeros((len(x),6))
    for rows,f in feature_rows(x,metadata,angle_mask_bank()):
        wi=rows.window_index.to_numpy();j=rows.angle.to_numpy();np.add.at(counts,(wi,j),1);base=models['base'].predict(f['own'])
        np.add.at(sums['base',0.,1.],(wi,j),np.square(base-f['target']).mean(1))
        for b in ['own','both']:
            for a,model in models['grid'][b].items():
                correction=model.predict(f[b+'_residual'])
                for gate in [0.,.5,1.]:np.add.at(sums[b,a,gate],(wi,j),np.square(base+gate*correction-f['target']).mean(1))
    out=[]
    for (branch,alpha,gate),total in sums.items():
        values=np.divide(total,counts,out=np.full_like(total,np.nan),where=counts>0);table=pd.DataFrame(values).assign(source_id=metadata.source_id.to_numpy()).groupby('source_id').mean()
        for source,row in table.iterrows():out.append(dict(source_id=source,policy='U',branch=branch,regularization=alpha,multiplier=gate,loss=float(row.mean()),inner_fold=internal,readout_count=int(row.notna().sum())))
    return pd.DataFrame(out)

def selected_predictions(models,recipe,f):
    base=models['base'].predict(f['own']);out={'base':base}
    for branch in ['own','both']:
        gate=recipe[branch+'_multiplier'];model=models['grid'][branch][recipe[branch+'_alpha']]
        out[branch]=base.copy() if gate==0 else base+gate*model.predict(f[branch+'_residual'])
    return out

def score_profiles(models,recipe,x,metadata,out):
    allrows=[];out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for batch,(rows,f) in enumerate(feature_rows(x,metadata,angle_mask_bank())):
        pred=selected_predictions(models,recipe,f);a=rows.copy()
        for branch in ['base','own','both']:a['loss_'+branch]=np.square(pred[branch]-f['target']).mean(1)
        allrows.append(a);np.savez_compressed(out/f'predictions_{batch:04}.npz',**{c:rows[c].to_numpy() for c in rows},target=f['target'],**pred)
    raw=pd.concat(allrows,ignore_index=True);raw['benefit']=raw.loss_own-raw.loss_both;raw['sham_benefit']=raw.loss_base-raw.loss_own
    cols=['loss_base','loss_own','loss_both','benefit','sham_benefit'];compact=raw.groupby(['window_index','angle','patch_index'])[cols].mean().reset_index()
    compact.to_csv(out/'target_losses.csv.gz',index=False);table=compact.merge(metadata[['window_index','source_id','start_grid_index']],on='window_index',validate='many_to_one')
    table['physical_patch']=table.start_grid_index+4*table.patch_index;coverage=table[['source_id','angle','physical_patch']].drop_duplicates().groupby(['source_id','angle']).size().rename('distinct_patches');coverage.to_csv(out/'target_coverage.csv')
    if not coverage.ge(4).all() or not coverage.groupby(level=0).size().eq(6).all():raise ValueError('Scoring lost frozen angle support')
    window=table.groupby(['source_id','window_index','angle'])[cols].mean();profiles=window.groupby(['source_id','angle']).mean();profiles.to_csv(out/'participant_angle_profiles.csv')
    return profiles

def motion_descriptors(x,metadata):
    with np.errstate(all='ignore'):values=np.concatenate([np.nanmedian(x,axis=1),np.nanpercentile(x,75,axis=1)-np.nanpercentile(x,25,axis=1)],1)
    names=[f'{name}_{s}' for s in ['median','iqr'] for name in ANGLES];table=pd.DataFrame(values,columns=names).assign(source_id=metadata.source_id.to_numpy()).groupby('source_id').mean()
    if not np.isfinite(table.to_numpy()).all():raise ValueError('Undefined descriptor despite frozen angle support')
    return table

def subset(x,meta,ids):
    use=meta.source_id.isin(ids).to_numpy();m=meta.loc[use].copy().reset_index(drop=True);m['window_index']=np.arange(len(m));return x[use],m

def dependency_fixtures():
    """Paired conditional-innovation and independent controls, five fresh splits."""
    from threadpoolctl import threadpool_limits
    records=[];mask=angle_mask_bank()[0];target=int(np.flatnonzero(mask&~mask[np.arange(48)^1])[0]);opposite=target^1
    with threadpool_limits(limits=1):
        for seed in range(5):
            rng=np.random.default_rng(seed+991);p=rng.normal(size=(1024,48,4));noise=rng.normal(size=(1024,4))*.15
            for positive in [False,True]:
                a=p.copy();innovation=p[:,opposite] if positive else p[rng.permutation(len(p)),opposite];a[:,target]=2*innovation+noise
                f=permitted_features(a,np.broadcast_to(mask,(len(a),48)),np.full(len(a),target));assert f['eligible'].all();fit=slice(0,512);test=slice(512,None)
                s=RidgeStats(36,outputs=4);s.add(f['own'][fit],f['target'][fit]);base=s.fit(1.);y=f['target']-base.predict(f['own']);losses={}
                for branch in ['own','both']:
                    s=RidgeStats(72,outputs=4);s.add(f[branch+'_residual'][fit],y[fit]);model=s.fit(1.)
                    losses[branch]=float(np.square(base.predict(f['own'][test])+model.predict(f[branch+'_residual'][test])-f['target'][test]).mean())
                base_loss=float(np.square(base.predict(f['own'][test])-f['target'][test]).mean());records.append(dict(seed=seed,positive_dependency=positive,base_loss=base_loss,**losses,relative_benefit=(losses['own']-losses['both'])/base_loss))
    table=pd.DataFrame(records);positive=float(table[table.positive_dependency].relative_benefit.mean());null=float(table[~table.positive_dependency].relative_benefit.mean())
    return dict(status='pass' if positive>.25 and abs(null)<.15 else 'fail',positive_mean_relative_benefit=positive,negative_mean_relative_benefit=null,records=records,interpretation='Software fixture only, not evidence of a neurological disease mechanism')

def freeze_demographic_sensitivity(prep=PREP):
    from scipy.optimize import linear_sum_assignment
    import re
    prep=Path(prep);path=prep.parent/'demographics_review/participant_demographics.csv';dem=pd.read_csv(path);registry=pd.read_csv(prep/'participant_registry.csv')
    def key(value):
        match=re.fullmatch(r'(?:healthy:|stroke:)?(SUBJ|TVC)0*(\d+)',str(value),re.I)
        if not match:raise ValueError('Unrecognized participant ID for verified metadata join')
        return match.group(1).upper()+str(int(match.group(2)))
    dem['join_key']=dem.participant_id.map(key);registry['join_key']=registry.participant_id.map(key)
    if dem.join_key.duplicated().any():raise ValueError('Duplicate clinical metadata identity')
    joined=registry.merge(dem.drop(columns='participant_id'),on='join_key',how='left',validate='one_to_one')
    if joined[['age_years','sex_reported']].isna().any().any():raise ValueError('Missing verified age or sex')
    if not joined.sex_reported.isin(['M','F']).all():raise ValueError('Unknown author-reported sex code')
    pairs=[]
    for fold,g in joined.groupby('outer_fold'):
        controls=g[g.binary_label.eq(0)].sort_values('source_id');cases=g[g.binary_label.eq(1)].sort_values('source_id')
        age=np.abs(cases.age_years.to_numpy()[:,None]-controls.age_years.to_numpy()[None,:]);same=cases.sex_reported.to_numpy()[:,None]==controls.sex_reported.to_numpy()[None,:];valid=same&(age<=5)
        tie=np.array([[int(digest([a,b])[:8],16)/(2**32)*1e-6 for b in controls.source_id] for a in cases.source_id]);cost=np.where(valid,age+tie,1e6+tie)
        ii,jj=linear_sum_assignment(cost)
        for i,j in zip(ii,jj):
            if valid[i,j]:pairs.append(dict(outer_fold=int(fold),pair_id=f'{fold}:{cases.iloc[i].source_id}:{controls.iloc[j].source_id}',stroke_id=cases.iloc[i].source_id,control_id=controls.iloc[j].source_id,stroke_age=int(cases.iloc[i].age_years),control_age=int(controls.iloc[j].age_years),absolute_age_difference=int(age[i,j]),sex_reported=cases.iloc[i].sex_reported))
    pairs=pd.DataFrame(pairs);joined.to_csv(prep/'participant_demographics_joined.csv',index=False);pairs.to_csv(prep/'age_sex_matches.csv',index=False)
    proposal=json.loads((prep/'protocol_proposal.json').read_text());proposal['demographics_sensitivity']=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),source_sha256=file_sha256(path),caliper_years=5,same_sex=True,matching='within outer fold maximum cardinality then minimum absolute age, no replacement, SHA256 pair tie break',model_refit=False,bootstrap='1000 whole matched-pair draws within fold, fixed original fold pair weights',if_empty_fold='four-fold sensitivity unavailable',speed_matching=False)
    dump(prep/'protocol_proposal.json',proposal);hashes=json.loads((prep/'data_hashes.json').read_text())
    for name in ['protocol_proposal.json','participant_demographics_joined.csv','age_sex_matches.csv']:hashes[name]=file_sha256(prep/name)
    dump(prep/'data_hashes.json',hashes);return joined,pairs

def age_sex_sensitivity(scores,prep,out):
    from sklearn.metrics import roc_auc_score
    pairs=pd.read_csv(Path(prep)/'age_sex_matches.csv');lookup=scores.set_index('source_id');records=[]
    for r in pairs.itertuples():
        for source in [r.stroke_id,r.control_id]:
            row=lookup.loc[source].to_dict();row.update(source_id=source,pair_id=r.pair_id);records.append(row)
    table=pd.DataFrame(records);table.to_csv(Path(out)/'age_sex_matched_scores.csv',index=False)
    if pairs.outer_fold.nunique()!=4:return dict(status='unavailable',reason='A planned outer fold has no age/sex pair',pairs=len(pairs))
    metrics={m:ranking_metrics(table,m) for m in ['M','R','S']};counts=pairs.groupby('outer_fold').size();weights=(counts**2)/(counts**2).sum();rng=np.random.default_rng(42);effects=[]
    for _ in range(1000):
        delta=[]
        for fold in range(4):
            ids=pairs[pairs.outer_fold.eq(fold)].pair_id.to_numpy();draw=rng.choice(ids,size=len(ids),replace=True);sample=pd.concat([table[table.pair_id.eq(p)] for p in draw],ignore_index=True)
            delta.append(roc_auc_score(sample.binary_label,sample.S)-roc_auc_score(sample.binary_label,sample.M))
        effects.append(float(weights.to_numpy()@delta))
    return dict(status='complete',pairs=len(pairs),participants=len(table),by_fold=counts.to_dict(),by_sex=pairs.groupby('sex_reported').size().to_dict(),mean_absolute_age_difference=float(pairs.absolute_age_difference.mean()),maximum_absolute_age_difference=int(pairs.absolute_age_difference.max()),metrics=metrics,S_minus_M=metrics['S']['weighted_auroc']-metrics['M']['weighted_auroc'],conditional_95_interval=np.quantile(effects,[.025,.975]).tolist(),resampling='whole matched pairs within fold; fixed original fold weights; models held fixed',interpretation='Prespecified age/sex-balanced sensitivity, not causal adjustment')

def run_replication(prep=PREP):
    from threadpoolctl import threadpool_limits
    prep=Path(prep);hashes=json.loads((prep/'data_hashes.json').read_text())
    if not {'participant_demographics_joined.csv','age_sex_matches.csv'}.issubset(hashes):
        raise ValueError('Freeze the demographic sensitivity before external fitting')
    for name,value in hashes.items():
        if file_sha256(prep/name)!=value:raise ValueError('Changed prepared external data: '+name)
    proposal=json.loads((prep/'protocol_proposal.json').read_text());code={p.name:file_sha256(p) for p in [Path(__file__),Path(__file__).with_name('external_data.py'),Path(__file__).with_name('retained_prediction.py'),Path(__file__).with_name('iteration_scoring.py')]}
    identity=dict(proposal=proposal,data=hashes,code=code);protocol_hash=digest(identity)[:20];out=ROOT/'experiments/aaai/artifacts/iteration02'/('external_'+protocol_hash);out.mkdir(exist_ok=True)
    if (out/'scores.csv').exists():raise ValueError('External scored run already exists; preserve it')
    dump(out/'protocol.json',dict(protocol_hash=protocol_hash,identity=identity,frozen_at_utc=datetime.now(timezone.utc).isoformat(),status='frozen_before_external_fitting'))
    snapshot=out/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in code:(snapshot/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    fixture=dependency_fixtures();dump(out/'dependency_fixtures.json',fixture)
    if fixture['status']!='pass':raise ValueError('External dependency fixtures failed')
    x=np.load(prep/'windows.npy');meta=pd.read_csv(prep/'window_index.csv');registry=pd.read_csv(prep/'participant_registry.csv');motion=motion_descriptors(x,meta);motion.to_csv(out/'descriptors_M12.csv');allscores=[];adequacy=[]
    with threadpool_limits(limits=1):
        for fold in range(4):
            folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True);role=f'role_fold_{fold}';inner=f'inner_fold_{fold}';train=registry[registry[role].eq('train')];reports=[];start=time.perf_counter()
            for internal in range(3):
                fitids=train.loc[train[inner].ne(internal),'source_id'];vids=train.loc[train[inner].eq(internal),'source_id'];fx,fm=subset(x,meta,fitids);vx,vm=subset(x,meta,vids)
                models=fit_models(fx,fm,42+fold*100+internal);reports.append(internal_report(models,vx,vm,internal));joblib.dump(models,folder/f'internal_{internal}_models.joblib')
            report=pd.concat(reports,ignore_index=True);report.to_csv(folder/'internal_candidates.csv',index=False);recipe=select_correction_recipe(report,'U');dump(folder/'selection.json',recipe);adequacy.append(dict(outer_fold=fold,**recipe))
            fx,fm=subset(x,meta,train.source_id);models=fit_models(fx,fm,42+fold*100+99);joblib.dump(models,folder/'models.joblib');dump(folder/'fit_audit.json',models['audit'])
            refs=registry.loc[registry[role].eq('reference'),'source_id'].tolist();tests=registry.loc[registry[role].eq('test'),'source_id'].tolist();sx,sm=subset(x,meta,refs+tests);sm.to_csv(folder/'scoring_window_index.csv',index=False);profiles=score_profiles(models,recipe,sx,sm,folder/'predictions')
            normal_distance=NormalDistance.fit(motion,train.source_id);scores=registry.set_index('source_id').loc[tests,['participant_id','group','binary_label']].copy();scores['M']=normal_distance.score(motion).reindex(tests);scores['outer_fold']=fold;scores['cluster_id']=scores.index;scores['focal_eligible']=True;scores['relational_supported']=recipe['relational_supported'];reference_audit={}
            for field in ['benefit','sham_benefit','loss_base','loss_own','loss_both']:
                p=profiles[field].unstack('angle');bilateral=pd.DataFrame(dict(left=p[[0,2,4]].mean(1),right=p[[1,3,5]].mean(1)));reference=BilateralReference.fit(bilateral,refs);r=reference.score(bilateral).reindex(tests);reference_audit[field]=reference.audit()
                if field=='benefit':scores['R_raw']=r;scores['R']=r if recipe['relational_supported'] else 0.
                else:scores[field+'_distance']=r;scores[field+'_fusion']=.5*(scores.M+r)
            scores['S']=.5*(scores.M+scores.R);scores['source_id']=scores.index;scores.to_csv(folder/'scores.csv',index=False);allscores.append(scores.reset_index(drop=True));dump(folder/'reference_audit.json',reference_audit);dump(folder/'M_audit.json',normal_distance.audit())
            print(f'External fold{fold}: completed in {time.perf_counter()-start:.1f}s; normal gate {recipe["relational_supported"]}',flush=True)
    scores=pd.concat(allscores,ignore_index=True);scores.to_csv(out/'scores.csv',index=False);dump(out/'adequacy.json',adequacy)
    methods=['M','R','S','R_raw','sham_benefit_fusion','loss_base_fusion','loss_own_fusion','loss_both_fusion'];metrics={m:ranking_metrics(scores,m) for m in methods};dump(out/'metrics.json',metrics)
    effects={m:paired_bootstrap(scores,'S',m,1000,42) for m in ['M','sham_benefit_fusion','loss_base_fusion','loss_own_fusion','loss_both_fusion']};dump(out/'paired_effects.json',effects)
    demo=age_sex_sensitivity(scores,prep,out);dump(out/'age_sex_sensitivity.json',demo)
    dump(out/'completion.json',dict(status='complete',protocol_hash=protocol_hash,participants=len(scores),normal=int(scores.binary_label.eq(0).sum()),stroke=int(scores.binary_label.eq(1).sum()),normal_adequacy_folds=sum(r['relational_supported'] for r in adequacy),finished_at_utc=datetime.now(timezone.utc).isoformat(),no_transfer_or_pooling=True))
    print(json.dumps(dict(output=str(out),metrics=metrics,effects=effects,age_sex_sensitivity=demo),indent=2),flush=True);return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    if args.stage=='prepare':
        download_and_prepare();freeze_demographic_sensitivity()
    else:run_replication()
if __name__=='__main__':main()
