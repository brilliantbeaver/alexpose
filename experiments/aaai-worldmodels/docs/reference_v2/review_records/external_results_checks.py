"""Independent frozen-transfer and pre-score exclusion result audit; no fitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import joblib,numpy as np,pandas as pd,torch
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P));sys.path.insert(0,str(Path(__file__).resolve().parent))
from worldmodels.v2_external import load_external
from worldmodels.v2_models import predict_forecaster
from core_results_checks import window_error,source_error,paired_rank,nuisance_prediction
O=P/'artifacts/reference-v2-independent-review';CORE=P/'artifacts/runs/reference-v2-core-v1';TRANSFER=P/'artifacts/runs/reference-v2-transfer-v1';SENS=P/'artifacts/runs/reference-v2-identity-sensitivity-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def native_window_error(x,v,p):
    # External producer uses native float32 squares/sums. Preserve that arithmetic.
    js=[8,10,12,14,9,11,13,15];target=x[:,16:][:,:,js];support=v[:,16:][:,:,js]&v[:,:16][:,:,js].any(1)[:,None]
    errors=[]
    for sl in (slice(0,4),slice(4,8)):
        mask=np.broadcast_to(support[:,:,sl,None],target[:,:,sl].shape)
        square=np.where(mask,(target[:,:,sl]-p[:,:,sl])**2,0.)
        errors.append(square.sum((1,2,3))/mask.sum((1,2,3)))
    return np.mean(errors,axis=0)
def main():
    started=time.monotonic();result=read(TRANSFER/'results.json');assert result['status']=='complete';hash_count=0
    for base in (TRANSFER,SENS):
        for name,h in read(base/'artifact_hashes.json').items():assert sha(base/name)==h;hash_count+=1
    x,v,meta,t=load_external();scores=pd.read_csv(TRANSFER/'participant_scores.csv').set_index('source_id');cov=pd.read_csv(TRANSFER/'participant_covariates.csv').set_index('source_id')
    assert len(scores)==90 and len(x)==773 and meta.recording_id.nunique()==168
    assert set(scores.index)==set(meta.participant_id)
    manual_motion=[]
    for i in range(len(x)):
        legmeans=[]
        for js in ([8,10,12,14],[9,11,13,15]):
            values=[]
            for q in range(1,16):
                for j in js:
                    if v[i,q,j] and v[i,q-1,j]:values.append(np.linalg.norm(x[i,q,j]-x[i,q-1,j])/(t[i,q]-t[i,q-1]))
            assert values;legmeans.append(np.mean(values))
        manual_motion.append(np.mean(legmeans))
    motion=source_error(meta,manual_motion);missing=source_error(meta,1-v[:,:16].mean((1,2)))
    np.testing.assert_allclose(motion,cov.prefix_motion.reindex(motion.index),rtol=0,atol=1e-12)
    np.testing.assert_allclose(missing,cov.prefix_missingness.reindex(missing.index),rtol=0,atol=1e-12)
    components=pd.read_csv(TRANSFER/'participant_component_scores.csv');families={'primary':[],'posture':[]};max_mse=0.;max_tail=0.;max_dtype=0.;max_dtype_tail=0.;replays=0
    with np.load(TRANSFER/'component_predictions.npz',allow_pickle=False) as saved:
        for family,ledger in [('primary','calibration_models.json'),('posture','posture_calibration_models.json')]:
            for row in read(CORE/ledger):
                name=f"{family}_fold{row['outer_fold']}_rotation{row['rotation']}";ck=joblib.load(CORE/row['forecast_checkpoint']);cal=read(CORE/row['calibrator_path'])
                assert set(ck['training_source_ids'])==set(cal['schedule']['predictor_fit_sources'])
                assert not set(meta.source_id)&(set(ck['training_source_ids'])|set(cal['calibration_sources']))
                prediction=predict_forecaster(ck,x[:,:16],v[:,:16],t[:,:16]);np.testing.assert_array_equal(prediction,saved[name]);replays+=1
                mse=source_error(meta,native_window_error(x,v,prediction));double_mse=source_error(meta,window_error(x,v,prediction));max_dtype=max(max_dtype,float(np.max(np.abs(mse-double_mse))));table=components[components.component.eq(name)].set_index('source_id')
                err=float(np.max(np.abs(mse-table.mse.reindex(mse.index))));max_mse=max(max_mse,err);assert err<1e-12
                frame=cov.reindex(mse.index).copy();frame['mse']=mse
                residual=np.log(mse.to_numpy()+1e-8)-nuisance_prediction(cal['nuisance_model'],frame)
                ref=np.asarray(cal['calibration_residuals']);cdf=np.array([(.5+np.sum(ref<r)+.5*np.sum(ref==r))/(len(ref)+1) for r in residual]);tail=np.abs(2*cdf-1)
                double_residual=np.log(double_mse.to_numpy()+1e-8)-nuisance_prediction(cal['nuisance_model'],frame)
                double_tail=np.array([abs(2*((.5+np.sum(ref<r)+.5*np.sum(ref==r))/(len(ref)+1))-1) for r in double_residual]);max_dtype_tail=max(max_dtype_tail,float(np.max(np.abs(double_tail-tail))))
                err=float(np.max(np.abs(tail-table.two_sided.reindex(mse.index))));max_tail=max(max_tail,err);assert err<1e-12
                families[family].append(pd.DataFrame({'source_id':mse.index,'tail':tail,'mse':mse.to_numpy()}))
    for family,key in [('primary','two_sided'),('posture','posture_two_sided')]:
        assert len(families[family])==15
        expected=pd.concat(families[family]).groupby('source_id').mean(numeric_only=True)
        np.testing.assert_allclose(expected['tail'],scores[key].reindex(expected.index),rtol=0,atol=1e-12)
        np.testing.assert_allclose(expected.mse,scores['mse' if family=='primary' else 'posture_mse'].reindex(expected.index),rtol=0,atol=1e-12)
    anchors=np.zeros((len(x),8,2),dtype=float)
    for i in range(len(x)):
        for out,j in enumerate([8,10,12,14,9,11,13,15]):
            known=np.flatnonzero(v[i,:16,j]);anchors[i,out]=x[i,known[-1],j] if len(known) else 0.
    persistence=np.repeat(anchors[:,None],8,axis=1);expected=source_error(meta,native_window_error(x,v,persistence))
    np.testing.assert_allclose(expected,scores.persistence_mse.reindex(expected.index),rtol=0,atol=1e-12)
    np.testing.assert_allclose(scores.low_motion,-scores.persistence_mse,rtol=0,atol=1e-15)
    for reference,key in [('low_motion','primary_anomaly'),('posture_two_sided','posture_calibration_contrast')]:
        point,ci=paired_rank(scores.reset_index(),'two_sided',reference,seed=5703)
        np.testing.assert_allclose(point,[result[key]['candidate_auroc'],result[key]['reference_auroc']],rtol=0,atol=1e-12);np.testing.assert_allclose(ci,result[key]['ci95'],rtol=0,atol=1e-12)
    rule=read(P/'docs/reference_v2/external_identity_sensitivity_protocol.json');sensitivity=read(SENS/'results.json');kept=pd.read_csv(SENS/'participant_scores.csv').set_index('source_id')
    assert rule['excluded_participant_ids']==['PD_SV_001'] and sensitivity['excluded_participant_ids']==rule['excluded_participant_ids']
    assert len(kept)==89 and set(kept.index)==set(scores.index)-{'PD_SV_001'}
    pd.testing.assert_frame_equal(kept,scores.loc[kept.index],check_exact=False,rtol=0,atol=1e-14)
    assert sensitivity['inputs']['declaration_sha256']==sha(P/'docs/reference_v2/external_identity_sensitivity_protocol.json')
    for reference,key in [('low_motion','primary_anomaly'),('posture_two_sided','posture_calibration_contrast')]:
        point,ci=paired_rank(kept.reset_index(),'two_sided',reference,seed=5703)
        np.testing.assert_allclose(point,[sensitivity[key]['candidate_auroc'],sensitivity[key]['reference_auroc']],rtol=0,atol=1e-12);np.testing.assert_allclose(ci,sensitivity[key]['ci95'],rtol=0,atol=1e-12)
    record={'identity':'reference-v2-independent-external-result-reconstruction-v1','status':'passed','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'transfer_results_sha256':sha(TRANSFER/'results.json'),'sensitivity_results_sha256':sha(SENS/'results.json'),'artifact_hashes_verified':hash_count,'component_checkpoint_trajectory_replays':replays,'component_participant_aggregations_verified':replays,'maximum_participant_mse_discrepancy':max_mse,'maximum_tail_score_discrepancy':max_tail,'maximum_native_float32_vs_double_mse_discrepancy':max_dtype,'maximum_float32_vs_double_tail_discrepancy':max_dtype_tail,'participant_prefix_covariates_rebuilt':len(motion),'primary_anomaly':result['primary_anomaly'],'posture_comparison':result['posture_calibration_contrast'],'secondary_identity_exclusion':sensitivity['primary_anomaly'],'seconds':time.monotonic()-started,'limits':['No model was fitted in this audit.','External units are dataset-coded participants; the generic component wording inherited by the sensitivity function must be read at this unit.','Participant aggregation differs from GAVD source-video aggregation; equal recording weighting does not eliminate acquisition/domain differences.','Intervals condition on fixed source-trained predictors and calibration distributions.']}
    (O/'external_results_checks.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:record[k] for k in ['status','component_checkpoint_trajectory_replays','maximum_participant_mse_discrepancy','maximum_tail_score_discrepancy','seconds']},indent=2))
if __name__=='__main__':
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):main()
