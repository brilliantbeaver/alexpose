"""Independent nuisance-regression and transport replay of the three floors."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,sys,time
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P));sys.path.insert(0,str(Path(__file__).resolve().parent))
from core_results_checks import paired_rank
from worldmodels import v2_evaluation as ev
O=P/'artifacts/reference-v2-independent-review';C=P/'artifacts/runs/reference-v2-core-v1';B=P/'artifacts/runs/reference-v2-floor-posthoc-v2'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fit_reference(frame,floor):
    z=np.column_stack([np.log(frame.prefix_motion.to_numpy()+floor),frame.prefix_missingness]);y=np.log(frame.mse.to_numpy()+floor)
    def fit(ii,alpha):
        mean,scale=z[ii].mean(0),z[ii].std(0);scale[scale<1e-6]=1.
        return mean,scale,Ridge(alpha=alpha).fit((z[ii]-mean)/scale,y[ii])
    scores={}
    for alpha in [.1,1.,10.,100.]:
        errors=[]
        for i in range(len(y)):
            mean,scale,model=fit(np.arange(len(y))!=i,alpha);errors.append(float((model.predict(((z[i]-mean)/scale)[None])[0]-y[i])**2))
        scores[alpha]=np.mean(errors)
    alpha=min(scores,key=lambda a:(scores[a],-a));mean,scale,model=fit(np.ones(len(y),bool),alpha)
    return alpha,lambda df:model.predict((np.column_stack([np.log(df.prefix_motion.to_numpy()+floor),df.prefix_missingness])-mean)/scale)
def prediction(model,df,floor):
    z=np.column_stack([np.log(df.prefix_motion.to_numpy()+floor),df.prefix_missingness]);return ((z-np.asarray(model['mean']))/np.asarray(model['scale']))@np.asarray(model['coefficient'])+model['intercept']
def main():
    start=time.monotonic();r=read(B/'results.json');assert r['status']=='complete' and r['baseline_replay']
    hashes=read(B/'artifact_hashes.json')
    for name,h in hashes.items():assert sha(B/name)==h,name
    original=pd.read_csv(C/'source_scores.csv').sort_values('source_id').reset_index(drop=True)
    original_hash=sha(C/'artifact_hashes.json');all_scores={};maxg=0.;replays=0
    for floor in (1e-10,1e-8,1e-6):
        fitted=read(B/f'calibrators_floor_{floor:.0e}.json');byfold={}
        for item in fitted:
            cal=item['calibrator'];reference=pd.DataFrame(cal['nuisance_reference_rows']);calibration=pd.DataFrame(cal['calibration_source_rows'])
            alpha,independent=fit_reference(reference,floor);assert alpha==cal['nuisance_model']['alpha']
            discrepancy=float(np.max(np.abs(independent(reference)-prediction(cal['nuisance_model'],reference,floor))));maxg=max(maxg,discrepancy);assert discrepancy<1e-10
            ref=np.log(calibration.mse.to_numpy()+floor)-prediction(cal['nuisance_model'],calibration,floor)
            source_order=np.argsort(calibration.component_id.to_numpy());np.testing.assert_allclose(ref[source_order],cal['calibration_residuals'],rtol=0,atol=1e-11)
            row=next(row for row in read(C/'calibration_models.json') if row['outer_fold']==item['outer_fold'] and row['rotation']==item['rotation'])
            held=pd.read_csv(C/row['scores_path']).sort_values('source_id');res=np.log(held.mse.to_numpy()+floor)-prediction(cal['nuisance_model'],held,floor)
            ref=np.asarray(cal['calibration_residuals']);tail=np.array([abs(2*(.5+np.sum(ref<q)+.5*np.sum(ref==q))/(len(ref)+1)-1) for q in res])
            byfold.setdefault(item['outer_fold'],[]).append((item['rotation'],held,tail));replays+=1
        rows=[]
        for fold,components in sorted(byfold.items()):
            components.sort(key=lambda a:a[0]);metadata=components[0][1][['source_id','label','component_id','outer_fold']].copy()
            values=np.column_stack([a[2] for a in components]);metadata['candidate']=values.mean(1);rows.append(metadata)
        raw=pd.concat(rows).sort_values('source_id').reset_index(drop=True)
        # This is the frozen producer's transport, independently reconstructed.
        expected=pd.read_csv(io.StringIO(raw.to_csv(index=False)));expected['low_motion']=original.low_motion
        actual=pd.read_csv(B/f'source_scores_floor_{floor:.0e}.csv').sort_values('source_id').reset_index(drop=True)
        np.testing.assert_array_equal(expected.candidate,actual.candidate)
        np.testing.assert_array_equal(actual.candidate,original.calibrated_two_sided)
        point,ci=paired_rank(expected,'candidate','low_motion');record=next(a for a in r['results'] if a['floor']==floor)
        np.testing.assert_allclose(point,[record['primary_anomaly']['candidate_auroc'],record['primary_anomaly']['reference_auroc']],rtol=0,atol=1e-12)
        np.testing.assert_allclose(ci,record['primary_anomaly']['ci95'],rtol=0,atol=1e-12)
        all_scores[str(floor)]=actual.candidate.tolist()
    assert ev.FLOOR==1e-8 and sha(C/'artifact_hashes.json')==original_hash
    record={'identity':'reference-v2-independent-floor-result-reconstruction-v1','status':'passed','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'results_sha256':sha(B/'results.json'),'artifact_hashes_verified':len(hashes),'independent_normal_nuisance_refits':replays,'maximum_independent_nuisance_prediction_discrepancy':maxg,'source_vectors_reconstructed_bitwise':len(all_scores),'all_three_vectors_bitwise_equal_to_original_primary':True,'paired_endpoints_and_intervals_recomputed':3,'original_evaluator_global_unchanged':True,'seconds':time.monotonic()-start,'limits':['Explicitly post hoc numerical sensitivity, retaining the failed v1 identity.','The producer CSV path is required for exact ties; numerical vector tolerance alone is insufficient.','No forecast refit, new score direction, or primary result replacement.']}
    (O/'floor_results_checks.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
