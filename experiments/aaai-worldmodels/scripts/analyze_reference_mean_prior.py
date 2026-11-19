"""Separately declared post hoc, normal-training mean forecast controls.

The two untuned priors minimize the same masked coordinate objective as the
compact models. They preserve all completed core results and score directions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from worldmodels import v2_data,v2_models as vm,v2_evaluation as ev
from worldmodels.v2_runner import window_errors

PROTOCOL=PROJECT/'docs/reference_v2/mean_prior_protocol.json'
REVIEW=PROJECT/'docs/reference_v2/mean_prior_review.json'
DEFAULT=PROJECT/'artifacts/runs/reference-v2-mean-prior-posthoc-v1'
CORE=PROJECT/'artifacts/runs/reference-v2-core-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    def serial(x):
        if isinstance(x,np.ndarray):return x.tolist()
        if isinstance(x,np.generic):return x.item()
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=serial)+'\n')


def fit_outer_priors(x,valid,metadata,times,outer_fold):
    """Fit both intercept-only controls from permitted normal sources only.

    Each output's mean uses source/sequence/window hierarchy weights, multiplied
    by 64 / observed-coordinate-count for its leg and by its target validity.
    Thus missingness cannot change a window's total loss mass within a leg.
    The factor 64 matches the ridge solver and cancels from the fitted mean.
    """
    if len(x)!=len(metadata) or np.shape(x)[1:]!=(24,16,2) or np.shape(valid)!=np.shape(x)[:-1]:
        raise ValueError('Expected aligned fixed-time cohort arrays')
    if metadata.groupby('component_id').source_id.nunique().max()!=1:
        raise ValueError('This diagnostic requires the current singleton source components')
    use=np.flatnonzero(metadata.label.eq('normal') & metadata.outer_fold.ne(outer_fold))
    if not len(use):raise ValueError('No permitted normal training sources')
    meta=metadata.iloc[use].copy().reset_index(drop=True)
    meta['group_id']=meta.component_id
    _,anchor,known=vm.causal_features(x[use,:16],valid[use,:16],times[use,:16])
    target=np.asarray(x[use][:,16:,vm.TARGET_JOINTS],float)
    support=valid[use][:,16:,vm.TARGET_JOINTS]&known[:,None]
    observed=np.repeat(support[...,None],2,axis=-1)
    if not np.isfinite(target[observed]).all():raise ValueError('Observed training targets must be finite')
    hierarchy=vm.hierarchical_weights(meta)
    effective=np.zeros(target.shape,float)
    for start in (0,4):
        count=observed[:,:,start:start+4].sum((1,2,3))
        if np.any(count==0):raise ValueError('Each training window needs observed coordinates for both legs')
        effective[:,:,start:start+4]=observed[:,:,start:start+4]*(hierarchy*64/count)[:,None,None,None]
    denominator=effective.sum(0)
    if np.any(denominator<=0):raise ValueError('A future output has no permitted normal training support')
    safe=np.where(observed,target,0.)
    residual=np.where(observed,target-anchor[:,None],0.)
    absolute=(effective*safe).sum(0)/denominator
    difference=(effective*residual).sum(0)/denominator
    source_rows=meta.assign(hierarchy_weight=hierarchy).groupby('source_id',sort=True).agg(
        training_windows=('window_id','size'),training_sequences=('sequence_id','nunique'),hierarchy_weight=('hierarchy_weight','sum')).reset_index()
    return {'identity':'reference-v2-posthoc-normal-mean-priors-v1','outer_fold':int(outer_fold),
            'training_source_ids':sorted(meta.source_id.unique().tolist()),'training_window_ids':meta.window_id.tolist(),
            'mean_pose':absolute,'mean_residual':difference,'effective_output_weight':denominator,
            'source_rows':source_rows.to_dict('records'),
            'weighting':'Equal source, sequence, window hierarchy; within-leg observed-coordinate normalization; per-output weighted mean',
            'interpretation':'Intercept-only optimum of the declared normal-training coordinate loss; no model or direction selection'}


def predict_priors(model,prefix,observed,times):
    _,anchor,_=vm.causal_features(prefix,observed,times)
    return {'mean_pose':np.broadcast_to(np.asarray(model['mean_pose']),(len(prefix),8,8,2)).astype(np.float32),
            'mean_residual':(anchor[:,None]+np.asarray(model['mean_residual'])[None]).astype(np.float32)}


def isolation_check(x,valid,metadata,times,outer_fold):
    """Poison every held-out future value and support bit before refitting."""
    original=fit_outer_priors(x,valid,metadata,times,outer_fold)
    hold=metadata.outer_fold.eq(outer_fold).to_numpy()
    changed=np.array(x,copy=True);changed_valid=np.array(valid,copy=True)
    changed[hold,16:]=1e20;changed_valid[hold,16:]=~changed_valid[hold,16:]
    other=fit_outer_priors(changed,changed_valid,metadata,times,outer_fold)
    for key in ('mean_pose','mean_residual','effective_output_weight'):
        np.testing.assert_array_equal(original[key],other[key])
    assert original['training_source_ids']==other['training_source_ids']
    assert not set(original['training_source_ids'])&set(metadata.loc[hold,'source_id'])
    before=predict_priors(original,x[hold,:16],valid[hold,:16],times[hold,:16])
    after=predict_priors(other,changed[hold,:16],changed_valid[hold,:16],times[hold,:16])
    for key in before:np.testing.assert_array_equal(before[key],after[key])
    return {'outer_fold':int(outer_fold),'training_sources':len(original['training_source_ids']),
            'heldout_sources':int(metadata.loc[hold,'source_id'].nunique()),
            'poisoned_heldout_future_fit_invariance':True,'poisoned_heldout_future_prediction_invariance':True}


def self_test():
    rows=[]
    for source,seq,value in [('a','a1',0),('a','a1',0),('a','a1',0),('a','a2',10),('b','b1',2),('b','b1',2),('c','c1',50),('d','d1',100)]:
        rows.append({'source_id':source,'sequence_id':seq,'component_id':source,'label':'stroke' if source=='d' else 'normal',
                     'outer_fold':0 if source=='c' else 1,'window_id':f'w{len(rows)}','value':value})
    meta=pd.DataFrame(rows);x=np.ones((len(meta),24,16,2),np.float32)
    for i,value in enumerate(meta.value):x[i,16:]=value
    valid=np.ones(x.shape[:-1],bool);times=np.broadcast_to(np.arange(24)*.1,(len(meta),24)).copy()
    model=fit_outer_priors(x,valid,meta,times,0)
    np.testing.assert_allclose(model['mean_pose'],3.5);np.testing.assert_allclose(model['mean_residual'],2.5)
    assert model['training_source_ids']==['a','b']
    check=isolation_check(x,valid,meta,times,0)
    valid[0,16,vm.TARGET_JOINTS[0]]=False
    baseline=fit_outer_priors(x,valid,meta,times,0);x[0,16,vm.TARGET_JOINTS[0]]=np.nan
    other=fit_outer_priors(x,valid,meta,times,0)
    for key in ['mean_pose','mean_residual']:np.testing.assert_array_equal(baseline[key],other[key])
    # An independent intercept fit verifies the objective under unequal target
    # visibility rather than merely reproducing the mean formula above.
    from sklearn.linear_model import Ridge
    use=np.flatnonzero(meta.source_id.isin(['a','b']))
    hierarchy=np.array([1/12,1/12,1/12,1/4,1/4,1/4])
    target=x[use][:,16:,vm.TARGET_JOINTS]
    support=np.repeat(valid[use][:,16:,vm.TARGET_JOINTS][...,None],2,axis=-1)
    maximum=0.
    for h in range(8):
        for j in range(8):
            for coordinate in range(2):
                seen=support[:,h,j,coordinate]
                leg=slice(0,4) if j<4 else slice(4,8)
                weights=hierarchy[seen]/support[:,:,leg].sum((1,2,3))[seen]
                fitted=Ridge(alpha=1.).fit(np.zeros((seen.sum(),1)),target[seen,h,j,coordinate],sample_weight=weights)
                maximum=max(maximum,abs(float(fitted.intercept_)-baseline['mean_pose'][h,j,coordinate]))
    assert maximum<1e-12
    return {'status':'passed_synthetic_software_checks_only','fully_observed_hierarchical_mean':3.5,
            'fully_observed_residual_mean':2.5,'heldout_future_poison_test':check,
            'missing_target_value_invariance':True,'all128_intercepts_maximum_sklearn_discrepancy':maximum}


def verify_declaration(protocol_path,review_path):
    protocol=json.loads(Path(protocol_path).read_text());review=json.loads(Path(review_path).read_text())
    if protocol['script_sha256']!=sha(__file__):raise ValueError('Diagnostic script changed after declaration')
    if review.get('status')!='accepted_before_posthoc_fitting':raise ValueError('Independent pre-fit review is required')
    if review.get('protocol_sha256')!=sha(protocol_path) or review.get('script_sha256')!=sha(__file__):
        raise ValueError('Independent review does not match the declaration and implementation')
    for path,expected in protocol['input_sha256'].items():
        if sha(PROJECT/path)!=expected:raise ValueError('Declared input changed: '+path)
    return protocol


def run(output=DEFAULT,protocol_path=PROTOCOL,review_path=REVIEW):
    protocol=verify_declaration(protocol_path,review_path);output=Path(output)
    if output.exists():raise FileExistsError('A separate unused diagnostic identity is required')
    manifest=json.loads((CORE/'artifact_hashes.json').read_text())
    for path in ['results.json','source_scores.csv']:
        if sha(CORE/path)!=manifest[path]:raise ValueError('Frozen core input differs from its manifest')
    x,valid,meta,times=v2_data.load_cohort();started=time.monotonic();software=self_test()
    predictions={name:np.zeros((len(meta),8,8,2),np.float32) for name in ('mean_pose','mean_residual')}
    checks=[];models=[]
    for fold in sorted(meta.outer_fold.unique()):
        checks.append(isolation_check(x,valid,meta,times,int(fold)))
        model=fit_outer_priors(x,valid,meta,times,int(fold));models.append(model)
        ids=np.flatnonzero(meta.outer_fold.eq(fold))
        for name,value in predict_priors(model,x[ids,:16],valid[ids,:16],times[ids,:16]).items():predictions[name][ids]=value
    window_rows=[];source_rows=[]
    for name,prediction in predictions.items():
        for horizon in (4,8):
            error=window_errors(x,valid,prediction,horizon=horizon)
            frame=meta[['window_id','source_id','sequence_id','component_id','label','outer_fold']].copy()
            frame['arm'],frame['horizon'],frame['mse']=name,horizon,error;window_rows.append(frame)
            source=ev.aggregate_windows(meta,error,name='mse');source['arm'],source['horizon']=name,horizon;source_rows.append(source)
    source_long=pd.concat(source_rows,ignore_index=True)
    wide=source_long[source_long.horizon.eq(8)].pivot(index=['source_id','label','component_id','outer_fold'],columns='arm',values='mse').reset_index()
    wide=wide.merge(pd.read_csv(CORE/'source_scores.csv')[['source_id','persistence','ridge_posture']],on='source_id',validate='one_to_one')
    comparisons=[ev.normal_forecast_contrast(wide,name,reference,reps=protocol['bootstrap']['draws'],seed=protocol['bootstrap']['seed'])
                 for name in ('mean_pose','mean_residual') for reference in ('persistence','ridge_posture')]
    output.mkdir(parents=True)
    for model in models:write_json(output/f"fold_{model['outer_fold']}_normal_means.json",model)
    np.savez_compressed(output/'predictions.npz',**predictions)
    pd.concat(window_rows,ignore_index=True).to_csv(output/'window_scores.csv',index=False)
    source_long.to_csv(output/'source_scores_long.csv',index=False);wide.to_csv(output/'source_scores.csv',index=False)
    result={'identity':protocol['identity'],'status':'complete','analysis_status':'Post hoc, outcome-exposed exploratory diagnostic; not a new primary endpoint',
            'windows':len(meta),'sources':meta.source_id.nunique(),'normal_sources':meta.loc[meta.label.eq('normal'),'source_id'].nunique(),
            'models':['mean_pose','mean_residual'],'hyperparameter_selection':'none','comparisons':comparisons,
            'normal_mse':{name:float(wide.loc[wide.label.eq('normal'),name].mean()) for name in ['mean_pose','mean_residual','persistence','ridge_posture']},
            'software_test':software,'isolation_checks':checks,'seconds':time.monotonic()-started,
            'protocol_sha256':sha(protocol_path),'review_sha256':sha(review_path),'script_sha256':sha(__file__),'inputs':protocol['input_sha256'],
            'limitations':['Post hoc addition after core and external outcomes were known.',
                'Constant priors remain conditional on the existing body-reference normalization.',
                'Intervals condition on the fitted means and original posture predictions.',
                'The primary core results, candidate selection, and anomaly scores are unchanged.']}
    write_json(output/'results.json',result)
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in output.glob('*') if p.is_file()})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true');parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--protocol',type=Path,default=PROTOCOL);parser.add_argument('--review',type=Path,default=REVIEW)
    args=parser.parse_args()
    if args.self_test:print(json.dumps(self_test(),indent=2))
    else:
        result=run(args.output,args.protocol,args.review)
        print(json.dumps({key:result[key] for key in ['identity','status','normal_mse','seconds']},indent=2))
