from pathlib import Path
import sys,json,hashlib,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from worldmodels.v2_data import load_cohort
from worldmodels import v2_evaluation as ev
from worldmodels import v2_models as vm
from worldmodels.v2_masks import POLICIES,make_mask,MaskConfig
from worldmodels.v2_transfer import participant_errors,average_components
start=time.perf_counter()
x,v,m,t=load_cohort()
assert len(x)==2815 and m.source_id.nunique()==91
assert m[m.label.eq('normal')].source_id.nunique()==29
assert m.groupby('source_id').outer_fold.nunique().eq(1).all()
for leg in [[8,10,12,14],[9,11,13,15]]:
    s=v[:,16:,leg]&v[:,:16,leg].any(1)[:,None]
    assert (s[:,:4].sum((1,2))>=4).all() and (s.sum((1,2))>=8).all()
# Independent scalar-loop observed speed calculation, then the declared hierarchy.
manual=[]
for i in range(len(x)):
    legs=[]
    for joints in [[8,10,12,14],[9,11,13,15]]:
        rates=[]
        for q in range(1,16):
            for j in joints:
                if v[i,q,j] and v[i,q-1,j]:
                    rates.append(np.linalg.norm(x[i,q,j]-x[i,q-1,j])/(t[i,q]-t[i,q-1]))
        assert rates
        legs.append(np.mean(rates))
    manual.append(np.mean(legs))
expected=ev.window_covariates(x,v,t).prefix_motion.to_numpy()
np.testing.assert_allclose(manual,expected,rtol=2e-7,atol=1e-8)
# Every calibration source is absent even from intermediate nuisance predictors.
rotations=[]
for f in sorted(m.outer_fold.unique()):
    for r in ev.calibration_schedule(m,int(f)):
        R,C,T=map(set,[r['predictor_fit_sources'],r['calibration_sources'],r['heldout_sources']])
        assert not(R&C or R&T or C&T)
        for inner in r['nuisance_oof']:
            assert not set(inner['train_sources'])&C
            assert not set(inner['train_sources'])&set(inner['validation_sources'])
        rotations.append((len(R),len(C),len(T)))
# Masks use exact equal budgets and protect refs/latest anchors on real examples.
count=0
for i in m.sort_values(['source_id','window_id']).drop_duplicates('source_id').index:
    budgets=[]
    for policy in POLICIES:
        hidden,protected,info=make_mask(x[i,:16],v[i,:16],t[i,:16],MaskConfig(policy),seed=932)
        assert not np.any(hidden & (~v[i,:16]|protected))
        if policy!='none':budgets.append(int(hidden.sum()))
        for j in range(16):
            q=np.flatnonzero(v[i,:16,j])
            if len(q):assert not hidden[q[-1],j]
        count+=1
    assert len(set(budgets))==1
# Test strict predictor information boundaries on synthetic observations.
rng=np.random.default_rng(903)
xx=rng.normal(size=(24,16,16,2)).astype(np.float32)
vv=rng.random((24,16,16))>.06;vv[:,-1]=True
ff=rng.normal(size=(24,8,16,2)).astype(np.float32)
fv=np.ones((24,8,16),bool);tt=np.tile(np.arange(16)*.1,(24,1))
mm=pd.DataFrame({'source_id':[f'n{i//2}' for i in range(24)],'sequence_id':[f's{i//2}' for i in range(24)],'label':'normal'})
for mode in ['ridge_posture','ridge_posture_velocity','mlp']:
    model=vm.fit_forecaster(xx,vv,tt,ff,fv,mm,vm.ForecastConfig(mode=mode,steps=5))
    hidden,_,_=make_mask(xx[0],vv[0],tt[0],MaskConfig('uniform'),seed=43)
    h=np.repeat(hidden[None],len(xx),axis=0)&vv
    safe=xx.copy();safe[h]=np.nan;safe[~vv]=1e20
    a=vm.predict_forecaster(model,xx,vv,tt,hidden=h)
    b=vm.predict_forecaster(model,safe,vv,tt,hidden=h)
    np.testing.assert_array_equal(a,b)
    try:vm.predict_forecaster(model,np.concatenate([xx,ff],1),np.concatenate([vv,fv],1),np.tile(np.arange(24)*.1,(24,1)))
    except ValueError:pass
    else:raise AssertionError('Full window was accepted at prefix boundary')
# Independently replay multinomial bootstrap via explicit resampled sklearn AUCs.
df=pd.DataFrame({'source_id':[f'u{i}' for i in range(15)],'component_id':[f'u{i}' for i in range(15)],
    'label':(['normal']*3+['stroke']*2)*3,'outer_fold':np.repeat(np.arange(3),5),
    'candidate':rng.integers(0,5,15)/5,'reference':rng.normal(size=15)})
result=ev.paired_within_fold_bootstrap(df,'candidate','reference',reps=200,seed=239)
boot=np.zeros(200);point=0;total=0;rr=np.random.default_rng(239)
for f,g in df.groupby('outer_fold'):
    n=g[g.label.eq('normal')];p=g[g.label.ne('normal')]
    nw=rr.multinomial(len(n),np.full(len(n),1/len(n)),size=200)
    pw=rr.multinomial(len(p),np.full(len(p),1/len(p)),size=200)
    pairs=len(n)*len(p);total+=pairs
    point+=pairs*(roc_auc_score(g.label.ne('normal'),g.candidate)-roc_auc_score(g.label.ne('normal'),g.reference))
    for k in range(200):
        a=n.iloc[np.repeat(np.arange(len(n)),nw[k])]
        b=p.iloc[np.repeat(np.arange(len(p)),pw[k])]
        z=pd.concat([a,b])
        boot[k]+=pairs*(roc_auc_score(z.label.ne('normal'),z.candidate)-roc_auc_score(z.label.ne('normal'),z.reference))
point/=total;boot/=total
np.testing.assert_allclose(result['estimate'],point,atol=1e-15)
np.testing.assert_allclose(result['ci95'],np.quantile(boot,[.025,.975]),atol=1e-15)
# External hierarchy has equal recording, not equal window, mass.
ex=np.zeros((4,24,16,2));valid=np.ones((4,24,16),bool)
meta=pd.DataFrame({'source_id':['a','a','a','b'],'participant_id':['a','a','a','b'],
 'component_id':['a','a','a','b'],'sequence_id':['r1','r1','r2','r3'],'label':['normal']*3+['stroke']})
pred=np.zeros((4,8,8,2));pred[:2]=1;pred[2]=3;pred[3]=2
errors=participant_errors(meta,ex,valid,pred).set_index('source_id')
assert errors.loc['a','mse']==5 and errors.loc['b','mse']==4
report={'status':'passed','real_windows':len(x),'real_sources':m.source_id.nunique(),'normal_sources':29,
 'manual_window_motion_max_abs':float(np.max(np.abs(np.asarray(manual)-expected))),
 'calibration_rotations':len(rotations),'rotation_counts':rotations,'actual_time_mask_checks':count,
 'synthetic_predictor_modes_checked':3,'hidden_nan_and_missing_extreme_invariance':True,
 'future_window_rejected':True,'independent_bootstrap_draws':200,'bootstrap_max_ci_discrepancy':float(np.max(np.abs(np.array(result['ci95'])-np.quantile(boot,[.025,.975])))),
 'participant_recording_weighting_checked':True,'seconds':time.perf_counter()-start,
 'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(ROOT/'artifacts/reference-v2-independent-review/prefit_checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
