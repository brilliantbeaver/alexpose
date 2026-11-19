"""Fixed observation stresses and training-pool diagnostics for reference v2."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from .data import sha256_file
from .settings import PROJECT
from .v2_data import load_cohort
from .v2_masks import MaskConfig,make_mask
from .v2_models import ResidualMLP,causal_features,predict_forecaster,TARGET_JOINTS
from .v2_runner import window_errors
from . import v2_evaluation as ev

def write_json(path,value):
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n')

def window_seed(identity,salt):
    return int(hashlib.sha256(f'{salt}:{identity}'.encode()).hexdigest()[:16],16)%(2**32)

def stress_inputs(prefix,valid,times,window_ids):
    """Fixed40% removal and0.02 torso-unit noise preserve references/anchors."""
    x,v,t=np.asarray(prefix),np.asarray(valid),np.asarray(times)
    if x.shape[1:]!=(16,16,2) or v.shape!=x.shape[:-1] or t.shape!=(len(x),16) or len(window_ids)!=len(x):
        raise ValueError('Need exact prefixes and window identities')
    hidden=np.zeros_like(v);noisy=x.copy()
    for i,identity in enumerate(window_ids):
        h,protected,_=make_mask(x[i],v[i],t[i],MaskConfig('uniform'),seed=window_seed(identity,7301))
        hidden[i]=h
        noise=np.random.default_rng(window_seed(identity,7401)).normal(0.,.02,size=x[i].shape)
        noisy[i]=np.where((v[i]&~protected)[...,None],x[i]+noise,x[i])
    return {'uniform_dropout':(x,v,hidden),'coordinate_noise':(noisy,v,None)}

def shifted_opposite(prefix,valid,window_ids,target_leg):
    """Rotate opposite-leg coordinate/validity pairs before protected anchors."""
    x,v=np.asarray(prefix).copy(),np.asarray(valid).copy()
    if target_leg not in (0,1) or x.shape[1:]!=(16,16,2) or v.shape!=x.shape[:-1]:
        raise ValueError('Need a target leg and exact prefix arrays')
    opposite=TARGET_JOINTS[4:8] if target_leg==0 else TARGET_JOINTS[:4]
    for i,identity in enumerate(window_ids):
        seed=window_seed(identity,7501)
        for j in opposite:
            measured=np.flatnonzero(v[i,:,j])
            if not len(measured):continue
            latest=int(measured[-1])
            if latest<2:continue
            offset=1+seed%(latest-1)
            x[i,:latest,j]=np.roll(x[i,:latest,j],offset,axis=0)
            v[i,:latest,j]=np.roll(v[i,:latest,j],offset,axis=0)
    return x,v

def checkpoint_for_role(core,ledger,role):
    matches=[r for r in ledger.values() if role in r['roles']]
    if len(matches)!=1:raise ValueError('Expected one checkpoint for role '+role)
    row=matches[0]
    if sha256_file(core/row['checkpoint'])!=row['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    return joblib.load(core/row['checkpoint'])

def predict_condition(checkpoint,x,v,t,ids,condition,stresses):
    if condition=='observed':return predict_forecaster(checkpoint,x,v,t)
    if condition=='opposite_time_shift':
        parts=[]
        for leg in (0,1):
            changed,support=shifted_opposite(x,v,ids,leg)
            p=predict_forecaster(checkpoint,changed,support,t)
            parts.append(p[:,:,leg*4:(leg+1)*4])
        return np.concatenate(parts,axis=2)
    a,b,h=stresses[condition]
    return predict_forecaster(checkpoint,a,b,t,hidden=h)

def nuisance_sensitivity(core,meta,output):
    views=[]
    for source,part in meta.groupby('source_id',sort=True):
        values=sorted(set(part.view.astype(str)))
        views.append({'source_id':source,'view':values[0] if len(values)==1 else 'mixed'})
    scores=pd.read_csv(core/'source_scores.csv').merge(pd.DataFrame(views),on='source_id',validate='one_to_one')
    strata=[]
    for view,part in scores.groupby('view',sort=True):
        n=int(part.label.eq('normal').sum());a=len(part)-n
        row={'view':view,'normal':n,'abnormal':a,'status':'estimable' if min(n,a)>=3 else 'insufficient_three_per_class'}
        if min(n,a)>=3:
            row.update({name:float(roc_auc_score(part.label.ne('normal'),part[name])) for name in
                        ('calibrated_two_sided','posture_two_sided','low_motion')})
        strata.append(row)
    pairs=[]
    for fold in sorted(scores.outer_fold.unique()):
        train=scores[scores.outer_fold.ne(fold)&scores.label.eq('normal')]
        z=np.column_stack([np.log(train.prefix_motion+ev.FLOOR),train.prefix_missingness])
        mu,sd=z.mean(0),z.std(0);sd[sd<1e-6]=1.
        for view,part in scores[scores.outer_fold.eq(fold)].groupby('view',sort=True):
            normal=part[part.label.eq('normal')].sort_values('source_id')
            abnormal=part[part.label.ne('normal')].sort_values('source_id')
            if not len(normal) or not len(abnormal):continue
            def standard(frame):return (np.column_stack([np.log(frame.prefix_motion+ev.FLOOR),frame.prefix_missingness])-mu)/sd
            cost=((standard(normal)[:,None]-standard(abnormal)[None])**2).sum(2)
            ni,ai=linear_sum_assignment(cost)
            for n,a in zip(ni,ai):
                pairs.append({'outer_fold':int(fold),'view':view,'normal_source':normal.iloc[n].source_id,
                    'abnormal_source':abnormal.iloc[a].source_id,'squared_standardized_distance':float(cost[n,a])})
    ids={r[k] for r in pairs for k in ('normal_source','abnormal_source')}
    matched=scores[scores.source_id.isin(ids)].copy()
    matched.to_csv(output/'matched_source_scores.csv',index=False)
    pd.DataFrame(pairs).to_csv(output/'nuisance_matched_pairs.csv',index=False)
    eligible=[int(f) for f,g in matched.groupby('outer_fold') if min(g.label.eq('normal').sum(),g.label.ne('normal').sum())>=2]
    subset=matched[matched.outer_fold.isin(eligible)]
    summary={'definition':'One-to-one nearest matching within outer fold and literal view, without replacement; log motion and missingness standardized on permitted normal training sources.',
        'pairs':len(pairs),'retained_sources':len(matched),'original_sources':len(scores),
        'view_strata':strata,'evaluable_matched_folds':eligible,'evaluable_matched_sources':len(subset),
        'limits':'No caliper guarantees similarity. Inspect pair distances. The population changes and unmeasured confounding remains.'}
    if len(subset):summary['paired_within_fold']=ev.paired_within_fold_bootstrap(subset,'calibrated_two_sided','low_motion',reps=5000,seed=5706)
    else:summary['paired_within_fold_status']='insufficient class support in matched folds'
    write_json(output/'nuisance_sensitivity.json',summary)
    return summary

def source_identity_probe(core,meta,x,v,t,ledger):
    """Decode recording labels inside the NORMAL model training pool only."""
    data=meta[meta.outer_fold.ne(0)&meta.label.eq('normal')].copy()
    counts=data.groupby('source_id').sequence_id.nunique();sources=sorted(counts[counts>=2].index)
    data=data[data.source_id.isin(sources)]
    seq={s:sorted(g.sequence_id.unique())[0] for s,g in data.groupby('source_id')}
    fit=data.sequence_id.eq(data.source_id.map(seq)).to_numpy();ii=data.index.to_numpy()
    if len(sources)<3:return {'status':'insufficient source classes with two sequences'}
    f,_,_=causal_features(x[ii,:16],v[ii,:16],t[ii,:16]);rows=[]
    for label,policy in [('M0','none'),('M3','anatomical_tubes'),('M4','motion_balanced'),('untrained','none')]:
        ck=checkpoint_for_role(core,ledger,f'outer0:mask_{policy}:seed27')
        torch.manual_seed(27);network=ResidualMLP(832,32)
        if label!='untrained':network.load_state_dict(ck['state_dict'])
        network.eval()
        with torch.no_grad():h=network.encoder(torch.as_tensor(((f-ck['mean'])/ck['scale']).astype(np.float32))).numpy()
        scaler=StandardScaler().fit(h[fit])
        classifier=LogisticRegression(C=1.,class_weight='balanced',max_iter=2000,random_state=821)
        classifier.fit(scaler.transform(h[fit]),data.source_id.to_numpy()[fit])
        pred=classifier.predict(scaler.transform(h[~fit]))
        rows.append({'representation':label,'source_classes':len(sources),'training_windows':int(fit.sum()),
            'sequence_heldout_windows':int((~fit).sum()),'balanced_accuracy':float(balanced_accuracy_score(data.source_id.to_numpy()[~fit],pred)),
            'uniform_class_baseline':1/len(sources),'iterations':classifier.n_iter_.tolist(),
            'source_ids':sources,'fit_sequence_per_source':seq})
    return {'status':'completed_training_pool_diagnostic','outer_pool':0,'rows':rows,
        'interpretation':'Recording-source information in features, not new-person, disease, or biometric validation.',
        'limits':['The forecaster saw all these normal training sequences and their training futures.',
            'Different annotation sequences may overlap in physical frames.','Source differences can reflect view, morphology, recording, or quality.']}

def run(core,cohort,output):
    core,output=Path(core),Path(output)
    if output.exists():raise FileExistsError('Use a fresh robustness analysis identity')
    for name,expected in json.loads((core/'artifact_hashes.json').read_text()).items():
        if sha256_file(core/name)!=expected:raise ValueError('Core artifact changed: '+name)
    x,v,meta,t=load_cohort(cohort);meta['group_id']=meta.component_id
    ledger=json.loads((core/'model_fit_ledger.json').read_text());output.mkdir(parents=True)
    records=[];conditions=('observed','uniform_dropout','coordinate_noise','opposite_time_shift')
    definitions=[('ridge_posture','ridge_posture'),('ridge_posture_velocity','ridge_posture_velocity')]
    definitions+=[(f'mlp_{p}',p) for p in ('none','uniform','time_blocks','anatomical_tubes','motion_balanced')]
    for fold in sorted(meta.outer_fold.unique()):
        ii=np.flatnonzero(meta.outer_fold.eq(fold));a,b,tt=x[ii,:16],v[ii,:16],t[ii,:16]
        ids=meta.iloc[ii].window_id.tolist();stresses=stress_inputs(a,b,tt,ids)
        for arm,mode in definitions:
            if arm.startswith('mlp_'):
                checkpoints=[checkpoint_for_role(core,ledger,f'outer{fold}:mask_{mode}:seed{seed}') for seed in (27,41,73)]
            else:checkpoints=[checkpoint_for_role(core,ledger,f'outer{fold}:ref_{mode}:refit')]
            for condition in conditions:
                pred=np.mean([predict_condition(ck,a,b,tt,ids,condition,stresses) for ck in checkpoints],axis=0)
                values=window_errors(x[ii],v[ii],pred)
                source=ev.aggregate_windows(meta.iloc[ii],values,name='mse')
                source['arm'],source['condition']=arm,condition;records.append(source)
    source=pd.concat(records,ignore_index=True);source.to_csv(output/'source_stress_scores.csv',index=False)
    effects=[]
    for condition in conditions:
        wide=source[source.condition.eq(condition)].pivot(index=['source_id','label','component_id','outer_fold'],columns='arm',values='mse').reset_index()
        for method,reference in [('mlp_anatomical_tubes','mlp_uniform'),('mlp_motion_balanced','mlp_uniform'),
                                 ('ridge_posture_velocity','ridge_posture')]:
            effects.append({'condition':condition,**ev.normal_forecast_contrast(wide,method,reference,reps=5000,seed=5702)})
    nuisance=nuisance_sensitivity(core,meta,output)
    probe=source_identity_probe(core,meta,x,v,t,ledger);write_json(output/'source_identity_probe.json',probe)
    summary={'identity':'reference-v2-observation-sensitivity-v1','status':'complete','conditions':list(conditions),
        'methods':[a for a,_ in definitions],'normal_forecast_contrasts':effects,
        'nuisance_sensitivity':nuisance,'source_identity_probe':probe,
        'inputs':{'core_results_sha256':sha256_file(core/'results.json'),
            'cohort_identity_sha256':sha256_file(Path(cohort)/'identity.json'),'code_sha256':sha256_file(__file__)},
        'limits':['Observation stresses are not neurological interventions.','Intervals condition on fixed source-trained predictors.',
            'Source identity is not verified participant identity; nuisance matching is not causal control.']}
    write_json(output/'results.json',summary)
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256_file(p)
        for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    return summary

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core',type=Path,default=PROJECT/'artifacts/runs/reference-v2-core-v1')
    p.add_argument('--cohort',type=Path,default=PROJECT/'artifacts/runs/reference-v2-cohort-v1')
    p.add_argument('--output',type=Path,default=PROJECT/'artifacts/runs/reference-v2-robustness-v1')
    args=p.parse_args()
    with threadpool_limits(limits=1):result=run(args.core,args.cohort,args.output)
    print(json.dumps({'status':result['status'],'conditions':result['conditions']}))

if __name__=='__main__':main()
