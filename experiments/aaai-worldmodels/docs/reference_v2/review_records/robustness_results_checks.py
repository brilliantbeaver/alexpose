"""Independent structural and result checks for the fixed robustness analysis."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
P=Path(__file__).resolve().parents[2];sys.path.insert(0,str(P));sys.path.insert(0,str(Path(__file__).resolve().parent))
from worldmodels.v2_data import load_cohort
from worldmodels.v2_robustness import stress_inputs,shifted_opposite
from core_results_checks import paired_rank
O=P/'artifacts/reference-v2-independent-review';C=P/'artifacts/runs/reference-v2-core-v1';B=P/'artifacts/runs/reference-v2-robustness-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def main():
    started=time.monotonic();r=read(B/'results.json');assert r['status']=='complete'
    hashes=read(B/'artifact_hashes.json')
    for name,h in hashes.items():assert sha(B/name)==h
    accepted=read(P/'docs/reference_v2/prefit_acceptance.json')
    assert sha(P/'worldmodels/v2_robustness.py')==accepted['separate_dependency_sha256']['worldmodels/v2_robustness.py']
    x,v,m,t=load_cohort();ids=m.window_id.tolist();stresses=stress_inputs(x[:,:16],v[:,:16],t[:,:16],ids)
    for i in range(len(x)):
        protected=np.zeros((16,16),bool);protected[:,[0,1,6,7]]=v[i,:16,[0,1,6,7]].T
        for j in range(16):
            known=np.flatnonzero(v[i,:16,j])
            if len(known):protected[known[-1],j]=True
        hidden=stresses['uniform_dropout'][2][i];eligible=v[i,:16]&~protected
        assert not np.any(hidden&~eligible) and hidden.sum()==int(np.floor(.4*eligible.sum()+.5))
        noisy=stresses['coordinate_noise'][0][i]
        np.testing.assert_array_equal(noisy[protected|~v[i,:16]],x[i,:16][protected|~v[i,:16]])
    for leg in (0,1):
        xx,vv=shifted_opposite(x[:,:16],v[:,:16],ids,leg);own=[8,10,12,14] if leg==0 else [9,11,13,15];opposite=[9,11,13,15] if leg==0 else [8,10,12,14]
        np.testing.assert_array_equal(xx[:,:,own],x[:,:16][:,:,own]);np.testing.assert_array_equal(vv[:,:,own],v[:,:16][:,:,own])
        np.testing.assert_array_equal(xx[:,:,:8],x[:,:16,:8]);np.testing.assert_array_equal(vv[:,:,:8],v[:,:16,:8])
        for i in range(len(x)):
            for j in opposite:
                ix=np.flatnonzero(v[i,:16,j])
                if len(ix):
                    assert np.flatnonzero(vv[i,:,j])[-1]==ix[-1];np.testing.assert_array_equal(xx[i,ix[-1],j],x[i,ix[-1],j])
                original=sorted(tuple(row) for row in np.column_stack([x[i,:16,j],v[i,:16,j]]))
                shifted=sorted(tuple(row) for row in np.column_stack([xx[i,:,j],vv[i,:,j]]));assert original==shifted
    core=pd.read_csv(C/'source_scores.csv').set_index('source_id');stress=pd.read_csv(B/'source_stress_scores.csv');max_observed=0.
    for arm,g in stress[stress.condition.eq('observed')].groupby('arm'):
        values=g.set_index('source_id').mse;delta=float(np.max(np.abs(values-core[arm].reindex(values.index))));max_observed=max(max_observed,delta);assert delta<2e-6
    assert len(stress)==91*7*4
    matching=pd.read_csv(B/'nuisance_matched_pairs.csv');expected=[]
    view={s:(g.view.astype(str).iloc[0] if g.view.astype(str).nunique()==1 else 'mixed') for s,g in m.groupby('source_id')}
    core['view']=[view[s] for s in core.index]
    for fold in sorted(core.outer_fold.unique()):
        train=core[core.outer_fold.ne(fold)&core.label.eq('normal')];z=np.column_stack([np.log(train.prefix_motion+1e-8),train.prefix_missingness]);mu,sd=z.mean(0),z.std(0);sd[sd<1e-6]=1.
        for literal,g in core[core.outer_fold.eq(fold)].groupby('view'):
            n,a=g[g.label.eq('normal')].sort_index(),g[g.label.ne('normal')].sort_index()
            if not len(n) or not len(a):continue
            transform=lambda df:(np.column_stack([np.log(df.prefix_motion+1e-8),df.prefix_missingness])-mu)/sd
            distance=np.sum((transform(n)[:,None]-transform(a)[None])**2,axis=2);ni,ai=linear_sum_assignment(distance)
            for nn,aa in zip(ni,ai):expected.append((int(fold),literal,n.index[nn],a.index[aa],float(distance[nn,aa])))
    actual=[tuple(row) for row in matching.itertuples(index=False,name=None)];assert len(expected)==len(actual)
    for a,b in zip(sorted(actual),sorted(expected)):
        assert a[:4]==b[:4];np.testing.assert_allclose(a[4],b[4],rtol=0,atol=1e-10)
    assert not matching.normal_source.duplicated().any() and not matching.abnormal_source.duplicated().any()
    matched=pd.read_csv(B/'matched_source_scores.csv');want=set(matching.normal_source)|set(matching.abnormal_source);assert set(matched.source_id)==want
    summary=read(B/'nuisance_sensitivity.json');estimable=matched[matched.outer_fold.isin(summary['evaluable_matched_folds'])]
    point,ci=paired_rank(estimable,'calibrated_two_sided','low_motion',seed=5706)
    np.testing.assert_allclose(ci,summary['paired_within_fold']['ci95'],rtol=0,atol=1e-12)
    np.testing.assert_allclose(point,[summary['paired_within_fold']['candidate_auroc'],summary['paired_within_fold']['reference_auroc']],rtol=0,atol=1e-12)
    probe=read(B/'source_identity_probe.json')
    for row in probe['rows']:
        assert row['source_classes']==len(row['source_ids']) and max(row['iterations'])<2000
        for source,seq in row['fit_sequence_per_source'].items():
            subset=m[m.source_id.eq(source)];assert subset.label.eq('normal').all() and subset.outer_fold.ne(0).all() and subset.sequence_id.nunique()>=2
            assert seq==sorted(subset.sequence_id.unique())[0]
    record={'identity':'reference-v2-independent-robustness-checks-v1','status':'passed','reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),'result_sha256':sha(B/'results.json'),'artifact_hashes_verified':len(hashes),'real_windows_with_mask_and_noise_protection_checked':len(x),'real_target_leg_shift_checks':len(x)*2,'maximum_observed_condition_mse_difference_from_core':max_observed,'matched_pairs_recomputed':len(expected),'matched_sources':len(matched),'estimable_matched_sources':len(estimable),'paired_matching_endpoint':summary['paired_within_fold'],'source_probe_classes':probe['rows'][0]['source_classes'],'seconds':time.monotonic()-started,'limits':['Standalone neural inference can differ from core one-thread arithmetic at float32 rounding.','The source-ID probe is an optimization/training-pool diagnostic and annotation sequences can overlap.','View/motion matching changes the population, gives no caliper similarity guarantee, and does not establish causal control.','No movement stress identifies a neurological intervention or mechanism.']}
    (O/'robustness_results_checks.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');print(json.dumps({k:record[k] for k in ['status','real_windows_with_mask_and_noise_protection_checked','maximum_observed_condition_mse_difference_from_core','matched_pairs_recomputed','seconds']},indent=2))
if __name__=='__main__':main()
