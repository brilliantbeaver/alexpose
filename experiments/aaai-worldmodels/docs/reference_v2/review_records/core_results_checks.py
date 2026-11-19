"""Independently replay completed core evidence without fitting a model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,sys,time
import joblib,numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[4];P=ROOT/'experiments/aaai-worldmodels';sys.path.insert(0,str(P))
from worldmodels.v2_data import load_cohort
from worldmodels.v2_models import predict_forecaster
CORE=P/'artifacts/runs/reference-v2-core-v1';OUT=P/'artifacts/reference-v2-independent-review'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
J=[8,10,12,14,9,11,13,15]
def window_error(x,v,p,h=8):
    target=x[:,16:16+h][:,:,J].astype(float);support=v[:,16:16+h][:,:,J]&v[:,:16][:,:,J].any(1)[:,None]
    square=(target-np.asarray(p[:,:h],float))**2;errors=[]
    for sl in (slice(0,4),slice(4,8)):
        mask=np.broadcast_to(support[:,:,sl,None],square[:,:,sl].shape)
        assert np.all(mask.sum((1,2,3))>0)
        errors.append(np.where(mask,square[:,:,sl],0).sum((1,2,3))/mask.sum((1,2,3)))
    return .5*(errors[0]+errors[1])
def source_error(meta,values):
    frame=meta[['source_id','sequence_id']].copy();frame['error']=values
    return frame.groupby(['source_id','sequence_id'],sort=True).error.mean().groupby('source_id',sort=True).mean()
def paired_rank(frame,candidate,reference,reps=5000,seed=5701):
    frame=frame.sort_values('component_id');rng=np.random.default_rng(seed);boot=np.zeros((reps,2));point=np.zeros(2);pairs=0
    for fold,g in frame.groupby('outer_fold',sort=True):
        n,a=g[g.label.eq('normal')],g[g.label.ne('normal')];nn,na=len(n),len(a);weight=nn*na;pairs+=weight
        wn=rng.multinomial(nn,np.full(nn,1/nn),size=reps);wa=rng.multinomial(na,np.full(na,1/na),size=reps)
        for j,key in enumerate((candidate,reference)):
            point[j]+=weight*roc_auc_score(g.label.ne('normal'),g[key])
            av=a[key].to_numpy();nv=n[key].to_numpy();pair=(av[:,None]>nv[None]).astype(float)+.5*(av[:,None]==nv[None])
            # Each row is an explicit paired weighted source draw.
            boot[:,j]+=np.sum((wa@pair)*wn,axis=1)
    return point/pairs,np.quantile((boot[:,0]-boot[:,1])/pairs,[.025,.975])
def nuisance_prediction(model,frame):
    z=np.column_stack([np.log(frame.prefix_motion.to_numpy()+1e-8),frame.prefix_missingness])
    return ((z-np.array(model['mean']))/np.array(model['scale']))@np.array(model['coefficient'])+model['intercept']
def independent_nuisance(frame,alphas):
    z=np.column_stack([np.log(frame.prefix_motion.to_numpy()+1e-8),frame.prefix_missingness]);y=np.log(frame.mse.to_numpy()+1e-8)
    def fit(ii,alpha):
        mean=z[ii].mean(0);scale=z[ii].std(0);scale[scale<1e-6]=1.
        ridge=Ridge(alpha=alpha).fit((z[ii]-mean)/scale,y[ii])
        return mean,scale,ridge
    errors={}
    for alpha in alphas:
        losses=[]
        for i in range(len(y)):
            ii=np.arange(len(y))!=i;mean,scale,model=fit(ii,alpha)
            losses.append(float((model.predict(((z[i]-mean)/scale)[None])[0]-y[i])**2))
        errors[alpha]=np.mean(losses)
    chosen=min(errors,key=lambda a:(errors[a],-a));mean,scale,model=fit(np.ones(len(y),bool),chosen)
    return chosen,lambda data:model.predict((np.column_stack([np.log(data.prefix_motion.to_numpy()+1e-8),data.prefix_missingness])-mean)/scale)
def main():
    start=time.monotonic();result=read(CORE/'results.json');assert result['status']=='complete' and not result['test_only']
    hashes=read(CORE/'artifact_hashes.json')
    for name,h in hashes.items():assert sha(CORE/name)==h,name
    freeze=read(P/'docs/reference_v2/protocol_freeze.json');request=read(CORE/'request.json')
    for name,h in freeze['implementation_sha256'].items():assert sha(P/name)==h,name
    assert sha(P/'docs/reference_v2/protocol.md')==freeze['protocol_sha256']
    x,v,meta,t=load_cohort();assert request['cohort_identity_sha256']==sha(P/'artifacts/runs/reference-v2-cohort-v1/identity.json')
    assert len(meta)==result['windows'];assert meta.source_id.nunique()==result['sources']
    sources=meta[['source_id','label','component_id','outer_fold']].drop_duplicates().set_index('source_id')
    normal=set(sources[sources.label.eq('normal')].index);checkpoint_count=0;outer_replays=0;max_mse=0.;max_window_mse=0.
    ledger=read(CORE/'model_fit_ledger.json');allpred={}
    for stage in ('refs','masks'):
        saved=pd.read_csv(CORE/f'{stage}_source_scores.csv');windows=pd.read_csv(CORE/f'{stage}_window_scores.csv')
        with np.load(CORE/f'{stage}_predictions.npz',allow_pickle=False) as z:
            for arm in z.files:
                p=z[arm].copy();allpred[arm]=p
                for horizon in (4,8):
                    losses=window_error(x,v,p,horizon);actual=saved[saved.arm.eq(arm)&saved.horizon.eq(horizon)].set_index('source_id').mse
                    expected=source_error(meta,losses);err=float(np.max(np.abs(expected-actual)))
                    max_mse=max(max_mse,err);assert err<1e-12,(arm,horizon,err)
                    win=windows[windows.arm.eq(arm)&windows.horizon.eq(horizon)].set_index('window_id').mse.reindex(meta.window_id).to_numpy()
                    err=float(np.max(np.abs(losses-win)));max_window_mse=max(max_window_mse,err);assert err<1e-12
    for row in ledger.values():
        ck=joblib.load(CORE/row['checkpoint']);assert sha(CORE/row['checkpoint'])==row['checkpoint_sha256']
        train=set(ck['training_source_ids']);assert train==set(row['training_source_ids']) and train<=normal
        assert train==set(row['identity']['sources'])
        for role in row['roles']:
            found=re.match(r'outer(\d+):',role)
            if found:assert not train&set(sources[sources.outer_fold.eq(int(found.group(1)))].index)
            found=re.fullmatch(r'outer(\d+):mask_([a-z_]+):seed(\d+)',role)
            if found:
                fold,policy,seed=int(found.group(1)),found.group(2),int(found.group(3));ii=np.flatnonzero(meta.outer_fold.eq(fold))
                p=predict_forecaster(ck,x[ii,:16],v[ii,:16],t[ii,:16]);np.testing.assert_array_equal(p,allpred[f'mlp_{policy}_seed{seed}'][ii]);outer_replays+=1
            found=re.fullmatch(r'outer(\d+):ref_(ridge_posture(?:_velocity)?):refit',role)
            if found:
                fold,mode=int(found.group(1)),found.group(2);ii=np.flatnonzero(meta.outer_fold.eq(fold))
                p=predict_forecaster(ck,x[ii,:16],v[ii,:16],t[ii,:16]);np.testing.assert_array_equal(p,allpred[mode][ii]);outer_replays+=1
        checkpoint_count+=1
    calibrators=0;trajectory_replays=0;oof_replays=0;max_cdf=0.;max_nuisance=0.;primary_sources={}
    for filename,key in [('calibration_models.json','calibrated_two_sided'),('posture_calibration_models.json','posture_two_sided')]:
        component_tables=[]
        for row in read(CORE/filename):
            cal=read(CORE/row['calibrator_path']);schedule=row['schedule'];R=set(schedule['predictor_fit_sources']);C=set(schedule['calibration_sources']);T=set(schedule['heldout_sources'])
            assert not R&C and not R&T and not C&T
            ck=joblib.load(CORE/row['forecast_checkpoint']);assert set(ck['training_source_ids'])==R
            for split in schedule['nuisance_oof']:
                assert not set(split['train_sources'])&(C|T|set(split['validation_sources']))
                role=f"outer{row['outer_fold']}:{row['family']}:cal{row['rotation']}:gOOF{split['nuisance_fold']}:refit"
                matched=[item for item in ledger.values() if role in item['roles']];assert len(matched)==1
                oofck=joblib.load(CORE/matched[0]['checkpoint']);assert set(oofck['training_source_ids'])==set(split['train_sources'])
                ii=np.flatnonzero(meta.source_id.isin(split['validation_sources']));p=predict_forecaster(oofck,x[ii,:16],v[ii,:16],t[ii,:16])
                expected=source_error(meta.iloc[ii],window_error(x[ii],v[ii],p));nuisance_rows=pd.DataFrame(cal['nuisance_reference_rows']).set_index('source_id')
                np.testing.assert_allclose(expected,nuisance_rows.mse.reindex(expected.index),rtol=0,atol=1e-12);oof_replays+=1
            reference_rows=pd.DataFrame(cal['nuisance_reference_rows']);chosen,g=independent_nuisance(reference_rows,request['config']['alphas'])
            assert chosen==cal['nuisance_model']['alpha']
            discrepancy=float(np.max(np.abs(g(reference_rows)-nuisance_prediction(cal['nuisance_model'],reference_rows))))
            max_nuisance=max(max_nuisance,discrepancy);assert discrepancy<1e-10
            with np.load(CORE/row['predictions_path'],allow_pickle=False) as data:
                for prefix in ('calibration','heldout'):
                    ii=data[prefix+'_indexes'];p=predict_forecaster(ck,x[ii,:16],v[ii,:16],t[ii,:16]);np.testing.assert_array_equal(p,data[prefix+'_prediction']);trajectory_replays+=1
                    expected=source_error(meta.iloc[ii],window_error(x[ii],v[ii],p))
                    table=pd.DataFrame(cal['calibration_source_rows']) if prefix=='calibration' else pd.read_csv(CORE/row['scores_path'])
                    np.testing.assert_allclose(expected,table.set_index('source_id').mse.reindex(expected.index),rtol=0,atol=1e-12)
            calibration=pd.DataFrame(cal['calibration_source_rows']).set_index('source_id').sort_index().reset_index()
            reference=np.log(calibration.mse+1e-8)-nuisance_prediction(cal['nuisance_model'],calibration)
            np.testing.assert_allclose(reference,cal['calibration_residuals'],rtol=0,atol=1e-11)
            test=pd.read_csv(CORE/row['scores_path']);res=np.log(test.mse.to_numpy()+1e-8)-nuisance_prediction(cal['nuisance_model'],test)
            ref=np.asarray(cal['calibration_residuals']);cdf=np.array([(.5+np.sum(ref<r)+.5*np.sum(ref==r))/(len(ref)+1) for r in res])
            err=float(np.max(np.abs(np.abs(2*cdf-1)-test.two_sided)));max_cdf=max(max_cdf,err);assert err<1e-12
            component_tables.append(test[['source_id','two_sided']]);calibrators+=1
        expected=pd.concat(component_tables).groupby('source_id').two_sided.mean();primary_sources[key]=expected
    scores=pd.read_csv(CORE/'source_scores.csv').set_index('source_id')
    for key,expected in primary_sources.items():np.testing.assert_allclose(expected,scores[key].reindex(expected.index),rtol=0,atol=1e-12)
    np.testing.assert_allclose(scores.low_motion,-scores.persistence,rtol=0,atol=1e-15)
    point,ci=paired_rank(scores.reset_index(),'calibrated_two_sided','low_motion')
    np.testing.assert_allclose(point,[result['primary_anomaly']['candidate_auroc'],result['primary_anomaly']['reference_auroc']],rtol=0,atol=1e-12)
    np.testing.assert_allclose(ci,result['primary_anomaly']['ci95'],rtol=0,atol=1e-12)
    point2,ci2=paired_rank(scores.reset_index(),'calibrated_two_sided','posture_two_sided')
    np.testing.assert_allclose(ci2,result['posture_calibration_contrast']['ci95'],rtol=0,atol=1e-12)
    for record in read(CORE/'selection.json').values():
        assert set(record['training_sources'])<=normal
        for split in record['membership']:assert not set(split['train_sources'])&set(split['validation_sources'])
        errors=record['validation_source_errors']
        for pool in (errors if isinstance(errors,list) else [errors]):
            for alpha,rows in pool.items():assert len(rows)==len(record['training_sources']) and len({r['source_id'] for r in rows})==len(rows)
    record={'identity':'reference-v2-independent-core-result-reconstruction-v1','status':'passed','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'core_results_sha256':sha(CORE/'results.json'),'core_manifest_sha256':sha(CORE/'artifact_hashes.json'),'artifact_hashes_verified':len(hashes),'normal_only_checkpoint_memberships':checkpoint_count,'outer_neural_and_compact_trajectory_replays':outer_replays,'calibrators_reconstructed':calibrators,'nuisance_oof_trajectory_replays':oof_replays,'maximum_independent_nuisance_prediction_discrepancy':max_nuisance,'calibration_and_heldout_trajectory_replays':trajectory_replays,'forecast_arms_reconstructed':len(allpred),'maximum_source_mse_discrepancy':max_mse,'maximum_window_mse_discrepancy':max_window_mse,'maximum_tail_score_discrepancy':max_cdf,'primary_anomaly':result['primary_anomaly'],'posture_comparison':result['posture_calibration_contrast'],'seconds':time.monotonic()-start,'limits':'Saved-fit prediction and aggregation replay; no model was retrained. Identity/clinical and full-pipeline uncertainty limitations remain.'}
    (OUT/'core_results_checks.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:record[k] for k in ['status','artifact_hashes_verified','normal_only_checkpoint_memberships','outer_neural_and_compact_trajectory_replays','calibration_and_heldout_trajectory_replays','maximum_source_mse_discrepancy','seconds']},indent=2))
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
