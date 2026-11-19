"""Post hoc equal-sequence weighting with recording-cluster uncertainty.

Frozen forecasts are reused. No eligibility, fitted model, or primary endpoint
changes. Every bootstrap draw carries all sequences of a selected recording.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[1]
CORE=PROJECT/'artifacts/runs/reference-v2-core-v1'
OUT=PROJECT/'artifacts/runs/reference-v2-sequence-scale-posthoc-v2'
PROTOCOL=PROJECT/'docs/draft03/reviewer_support/sequence_sensitivity_protocol.json'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weighted_draws(sums,counts,indices):
    return sums[indices].sum(axis=1)/counts[indices].sum(axis=1)


def main():
    declaration=json.loads(PROTOCOL.read_text())
    assert declaration['status']=='declared_before_sensitivity_computation'
    assert declaration['script_sha256']==sha(__file__)
    for name,value in declaration['inputs'].items():assert sha(PROJECT/name)==value,name
    recorded=json.loads((CORE/'artifact_hashes.json').read_text())
    for name in ['refs_window_scores.csv','cohort.csv','results.json']:
        assert sha(CORE/name)==recorded[name],name
    if OUT.exists():raise FileExistsError('Analysis identity cannot be overwritten')
    draws=np.array([[0,0],[0,1],[1,1]])
    np.testing.assert_allclose(weighted_draws(np.array([8.,-3.]),np.array([1,3]),draws),[8.,1.25,-1.])
    windows=pd.read_csv(CORE/'refs_window_scores.csv')
    windows=windows[windows.horizon.eq(8)&windows.arm.isin(['persistence','ridge_posture'])]
    cohort=pd.read_csv(CORE/'cohort.csv')
    keys=['window_id','source_id','sequence_id','label','component_id','outer_fold']
    assert not windows.duplicated(['window_id','arm']).any()
    assert len(windows)==2*len(cohort) and set(windows.window_id)==set(cohort.window_id)
    assert windows.mse.ge(0).all() and np.isfinite(windows.mse).all()
    wide=windows.pivot(index=keys,columns='arm',values='mse').reset_index()
    wide.columns.name=None
    pd.testing.assert_frame_equal(wide[keys].sort_values('window_id').reset_index(drop=True),
        cohort[keys].sort_values('window_id').reset_index(drop=True))
    assert not wide[['persistence','ridge_posture']].isna().any().any()
    sequences=wide.groupby(keys[1:],sort=True)[['persistence','ridge_posture']].mean().reset_index()
    assert len(sequences)==456 and sequences.sequence_id.nunique()==456
    normal=sequences[sequences.label.eq('normal')].copy()
    assert len(normal)==141 and normal.component_id.nunique()==29
    assert normal.groupby('component_id').outer_fold.nunique().eq(1).all()
    normal['difference']=normal.persistence-normal.ridge_posture
    groups=normal.groupby(['component_id','outer_fold'],sort=True).agg(
        sequences=('sequence_id','size'),persistence=('persistence','sum'),
        ridge_posture=('ridge_posture','sum'),difference=('difference','sum')).reset_index()
    rng=np.random.default_rng(6827);indices=[]
    for fold,part in groups.groupby('outer_fold',sort=True):
        positions=part.index.to_numpy()
        indices.append(positions[rng.integers(len(positions),size=(5000,len(positions)))])
    sample=np.concatenate(indices,axis=1);counts=groups.sequences.to_numpy()
    bootstrap={arm:weighted_draws(groups[arm].to_numpy(),counts,sample)
               for arm in ['persistence','ridge_posture','difference']}
    point={arm:float(normal[arm].mean()) for arm in bootstrap}
    assert np.all(bootstrap['persistence']>0)
    relative=bootstrap['difference']/bootstrap['persistence']
    frozen=json.loads((CORE/'results.json').read_text())
    primary=next(row for row in frozen['forecast_contrasts']
        if row['method']=='ridge_posture' and row['reference']=='persistence')
    recovered=normal.groupby('component_id')[['persistence','ridge_posture']].mean().mean()
    np.testing.assert_allclose(recovered.persistence,primary['reference_mse'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(recovered.ridge_posture,primary['method_mse'],atol=1e-12,rtol=0)
    result={'identity':declaration['identity'],'status':'complete',
        'analysis_status':'Post hoc weighting sensitivity; original primary results unchanged.',
        'forecast_refit':False,'sequences':456,'normal_sequences':141,'normal_recording_components':29,
        'sequence_weighted_mse':{arm:point[arm] for arm in ['persistence','ridge_posture']},
        'paired_improvement':point['difference'],
        'ci95':np.quantile(bootstrap['difference'],[.025,.975]).tolist(),
        'relative_reduction':point['difference']/point['persistence'],
        'relative_ci95':np.quantile(relative,[.025,.975]).tolist(),
        'recording_weighted_primary':primary,'resamples':5000,'seed':6827,
        'resampling':'Whole recording components within original folds; all sequences and multiplicities retained.',
        'weights':'Windows average within sequence; each sequence has equal point-estimate weight.',
        'uncertainty':'Conditional on frozen overlapping fits; unknown participant overlap and retraining variation excluded.',
        'software_checks':['Unequal group-size fixture recovers expected effects8,1.25,-1.','Recording-weighted path recovers original MSEs.'],
        'protocol_sha256':sha(PROTOCOL),'script_sha256':sha(__file__),'inputs':declaration['inputs']}
    OUT.mkdir(parents=True)
    sequences.to_csv(OUT/'per_sequence_errors.csv',index=False)
    groups.to_csv(OUT/'normal_recording_sums.csv',index=False)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    hashes={f.name:sha(f) for f in OUT.iterdir() if f.is_file()}
    (OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','normal_sequences','normal_recording_components',
        'sequence_weighted_mse','paired_improvement','ci95','relative_reduction']},indent=2))


if __name__=='__main__':main()
