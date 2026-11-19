"""Independent reconstruction of the post hoc intercept-prior controls."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np,pandas as pd
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P));sys.path.insert(0,str(Path(__file__).resolve().parent))
from worldmodels.v2_data import load_cohort
from core_results_checks import window_error,source_error
O=P/'artifacts/reference-v2-independent-review';B=P/'artifacts/runs/reference-v2-mean-prior-posthoc-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def main():
    start=time.monotonic();r=read(B/'results.json');assert r['status']=='complete' and 'Post hoc' in r['analysis_status']
    hashes=read(B/'artifact_hashes.json')
    for name,h in hashes.items():assert sha(B/name)==h
    declaration=read(P/'docs/reference_v2/mean_prior_protocol.json');review=read(P/'docs/reference_v2/mean_prior_review.json')
    assert sha(P/'scripts/analyze_reference_mean_prior.py')==declaration['script_sha256']==review['script_sha256']
    assert r['protocol_sha256']==sha(P/'docs/reference_v2/mean_prior_protocol.json')==review['protocol_sha256']
    x,v,m,t=load_cohort();J=[8,10,12,14,9,11,13,15];pred={n:np.zeros((len(x),8,8,2),np.float32) for n in ('mean_pose','mean_residual')};maxmean=0.
    anchors=np.zeros((len(x),8,2),float)
    for i in range(len(x)):
        for out,j in enumerate(J):
            known=np.flatnonzero(v[i,:16,j]);anchors[i,out]=x[i,known[-1],j] if len(known) else 0.
    for fold in sorted(m.outer_fold.unique()):
        ck=read(B/f'fold_{fold}_normal_means.json');ii=np.flatnonzero(m.label.eq('normal')&m.outer_fold.ne(fold));held=np.flatnonzero(m.outer_fold.eq(fold));train=m.iloc[ii]
        assert set(ck['training_source_ids'])==set(train.source_id) and set(ck['training_window_ids'])==set(train.window_id)
        sequence_counts={s:g.sequence_id.nunique() for s,g in train.groupby('source_id')};window_counts=train.groupby(['source_id','sequence_id']).size().to_dict()
        w=np.array([1/sequence_counts[row.source_id]/window_counts[(row.source_id,row.sequence_id)] for row in train.itertuples()]);w/=w.mean()
        yy=x[ii,16:][:,:,J].astype(float);support=v[ii,16:][:,:,J]&v[ii,:16][:,:,J].any(1)[:,None];weights=np.zeros_like(yy)
        for sl in (slice(0,4),slice(4,8)):
            mask=np.broadcast_to(support[:,:,sl,None],yy[:,:,sl].shape);weights[:,:,sl]=mask*(w*64/mask.sum((1,2,3)))[:,None,None,None]
        den=weights.sum(0);absolute=(weights*np.where(support[...,None],yy,0.)).sum(0)/den
        residual=(weights*np.where(support[...,None],yy-anchors[ii,None],0.)).sum(0)/den
        for name,mean in [('mean_pose',absolute),('mean_residual',residual)]:
            discrepancy=float(np.max(np.abs(mean-ck[name])));maxmean=max(maxmean,discrepancy);assert discrepancy<1e-12
        np.testing.assert_allclose(den,ck['effective_output_weight'],rtol=0,atol=1e-10)
        pred['mean_pose'][held]=np.broadcast_to(absolute,(len(held),8,8,2)).astype(np.float32)
        pred['mean_residual'][held]=(anchors[held,None]+residual[None]).astype(np.float32)
    source=pd.read_csv(B/'source_scores_long.csv');window=pd.read_csv(B/'window_scores.csv');maxmse=0.
    with np.load(B/'predictions.npz') as saved:
        for name,values in pred.items():
            np.testing.assert_array_equal(values,saved[name])
            for h in (4,8):
                errors=window_error(x,v,values,h);want=source_error(m,errors);actual=source[source.arm.eq(name)&source.horizon.eq(h)].set_index('source_id').mse
                error=float(np.max(np.abs(want-actual)));maxmse=max(maxmse,error);assert error<1e-12
                actual=window[window.arm.eq(name)&window.horizon.eq(h)].set_index('window_id').mse.reindex(m.window_id)
                np.testing.assert_allclose(errors,actual,rtol=0,atol=1e-12)
    wide=pd.read_csv(B/'source_scores.csv');normal=wide[wide.label.eq('normal')].sort_values('component_id');rng=np.random.default_rng(5801);draw=rng.integers(len(normal),size=(5000,len(normal)))
    for row in r['comparisons']:
        candidate=normal[row['method']].to_numpy();reference=normal[row['reference']].to_numpy();difference=reference-candidate
        np.testing.assert_allclose(difference.mean(),row['estimate'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(np.quantile(difference[draw].mean(1),[.025,.975]),row['ci95'],rtol=0,atol=1e-12)
    record={'identity':'reference-v2-independent-posthoc-mean-prior-reconstruction-v1','status':'passed','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'results_sha256':sha(B/'results.json'),'artifact_hashes_verified':len(hashes),'outer_prior_states_reconstructed':5,'forecast_predictions_replayed_bitwise':2,'maximum_fitted_mean_discrepancy':maxmean,'maximum_source_mse_discrepancy':maxmse,'paired_contrasts_recomputed':len(r['comparisons']),'normal_mse':r['normal_mse'],'seconds':time.monotonic()-start,'limits':['The diagnostic is explicitly post hoc after internal and external outcome exposure.','It uses the existing body reference and normal training sources; no primary anomaly candidate is changed.','Intervals condition on fitted priors and original posture predictions.']}
    (O/'mean_prior_results_checks.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
