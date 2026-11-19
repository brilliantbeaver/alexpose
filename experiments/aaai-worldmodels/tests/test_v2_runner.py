"""Frozen-entry, source isolation, scoring and exact-resume checks for v2."""
import json
from dataclasses import replace
import hashlib
import numpy as np
import pandas as pd
import pytest
import torch
from worldmodels import v2_models as vm
from worldmodels import v2_runner as runner


def fixture(normal=20,abnormal=5):
    rng=np.random.default_rng(77);n=normal+abnormal
    x=rng.normal(scale=.2,size=(n,24,16,2)).astype(np.float32)
    x[:,16:]=x[:,15:16]+rng.normal(scale=.06,size=(n,8,16,2))
    valid=np.ones((n,24,16),bool);times=np.tile(np.arange(24)*.1,(n,1))
    source=[f'n{i:02}' for i in range(normal)]+[f'a{i:02}' for i in range(abnormal)]
    fold=list(np.arange(normal)%5)+list(np.arange(abnormal)%5)
    meta=pd.DataFrame(dict(window_id=[f'w{i}' for i in range(n)],source_id=source,component_id=source,
        sequence_id=[f'q{i}' for i in range(n)],label=['normal']*normal+['stroke']*abnormal,outer_fold=fold))
    return x,valid,meta,times


def small_config(**kwargs):
    values=dict(identity='synthetic-software-check-v1',alphas=(1.,),mask_steps=(1,),seeds=(27,),
        policies=('none','uniform','motion_balanced'),forecast_reps=8,primary_reps=8,pooled_reps=8,
        untrained_probe=False,partition_seeds=(),core_seconds=300.)
    values.update(kwargs);return runner.RunConfig(**values)


def test_window_errors_match_equal_leg_torch_and_ignore_missing_target():
    x,v,meta,t=fixture();p=runner.basic_predictions(x,v,t)['persistence']
    v[:,17,vm.TARGET_JOINTS[0]]=False;x[:,17,vm.TARGET_JOINTS[0]]=np.nan
    errors=runner.window_errors(x,v,p)
    _,a,k=vm.causal_features(x[:,:16],v[:,:16],t[:,:16])
    y,m=vm.target_values(x[:,16:],v[:,16:],a,k)
    expected=vm.masked_window_loss(torch.as_tensor(p-a[:,None]),torch.as_tensor(y),torch.as_tensor(m)).numpy()
    np.testing.assert_allclose(errors,expected,rtol=1e-6)
    changed=x.copy();changed[:,17,vm.TARGET_JOINTS[0]]=1e30
    np.testing.assert_array_equal(errors,runner.window_errors(changed,v,p))


def test_reference_predictions_have_future_invariance_and_actual_time_velocity():
    x,v,meta,t=fixture()
    for k in range(24):x[:,k,:,0]=k*.2
    predictions=runner.basic_predictions(x,v,t)
    np.testing.assert_allclose(predictions['damped_velocity'][:,0,:,0],3.1,atol=1e-6)
    changed=x.copy();changed[:,16:]=1e10;cv=v.copy();cv[:,16:]=False
    for name,p in predictions.items():np.testing.assert_array_equal(p,runner.basic_predictions(changed,cv,t)[name])


def test_inner_source_splits_keep_groups_separate_and_require_normal():
    _,_,meta,_=fixture();meta['group_id']=meta.component_id
    normal=meta[meta.label.eq('normal')]
    splits=runner.inner_source_splits(normal)
    assert set(s for f in splits for s in f['validation_sources'])==set(normal.source_id)
    for split in splits:assert not set(split['train_sources'])&set(split['validation_sources'])
    with pytest.raises(ValueError,match='normal'):runner.inner_source_splits(meta)


def test_freeze_rejects_absence_status_protocol_and_code_change(tmp_path):
    with pytest.raises(ValueError,match='requires'):runner.verify_freeze(tmp_path/'missing.json',project=tmp_path)
    docs=tmp_path/'docs/reference_v2';docs.mkdir(parents=True);(docs/'protocol.md').write_text('protocol')
    freeze=dict(status='accepted_before_outer_fitting',independent_review='independent.md',
        protocol_sha256=runner.sha256_file(docs/'protocol.md'),implementation_sha256={})
    for name in runner.IMPLEMENTATIONS:
        p=tmp_path/name;p.parent.mkdir(exist_ok=True);p.write_text(name);freeze['implementation_sha256'][name]=runner.sha256_file(p)
    path=docs/'protocol_freeze.json';path.write_text(json.dumps(freeze))
    runner.verify_freeze(path,project=tmp_path)
    (tmp_path/runner.IMPLEMENTATIONS[0]).write_text('changed')
    with pytest.raises(ValueError,match='Implementation'):runner.verify_freeze(path,project=tmp_path)
    freeze['status']='draft';path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError,match='not accepted'):runner.verify_freeze(path,project=tmp_path)


def test_cache_resume_corruption_and_completed_immutability(tmp_path):
    arrays=fixture();cfg=small_config();study=runner.Study(tmp_path,*arrays,cfg,test_only=True)
    sources=study._training_sources(0)
    ck=study.fit_new(sources,vm.ForecastConfig(mode='ridge_posture',ridge_alpha=1.),'synthetic')
    original=study.predict_new(ck,np.array([0,1]));count=len(study.ledger)
    resumed=runner.Study(tmp_path,*arrays,cfg,test_only=True)
    other=resumed.fit_new(sources,vm.ForecastConfig(mode='ridge_posture',ridge_alpha=1.),'synthetic')
    np.testing.assert_array_equal(original,resumed.predict_new(other,np.array([0,1])))
    assert count==len(resumed.ledger)
    with pytest.raises(ValueError,match='Resume request'):
        runner.Study(tmp_path,*arrays,replace(cfg,identity='changed'),test_only=True)
    path=next((tmp_path/'models').glob('*.joblib'));path.write_bytes(path.read_bytes()+b'corrupt')
    resumed=runner.Study(tmp_path,*arrays,cfg,test_only=True)
    with pytest.raises(ValueError,match='identity mismatch'):
        resumed.fit_new(sources,vm.ForecastConfig(mode='ridge_posture',ridge_alpha=1.),'synthetic')
    (tmp_path/'results.json').write_text('{}')
    with pytest.raises(FileExistsError,match='immutable'):runner.Study(tmp_path,*arrays,cfg,test_only=True)


def test_runner_refuses_component_merge_and_abnormal_training(tmp_path):
    x,v,m,t=fixture();bad=m.copy();bad.loc[1,'component_id']=bad.loc[0,'component_id']
    with pytest.raises(ValueError,match='Multi-source'):runner.Study(tmp_path,x,v,bad,t,small_config(),test_only=True)
    study=runner.Study(tmp_path,x,v,m,t,small_config(),test_only=True)
    with pytest.raises(ValueError,match='abnormal'):study.fit_new(['n00','a00'],vm.ForecastConfig(mode='ridge_posture'),'bad')


def test_synthetic_complete_staged_replay_has_normal_memberships_and_same_c_test_model(tmp_path):
    arrays=fixture();study=runner.Study(tmp_path,*arrays,small_config(),test_only=True)
    study.run('all')
    results=json.loads((tmp_path/'results.json').read_text())
    assert results['status']=='complete' and results['test_only']
    assert results['sources']==25
    for entry in study.ledger.values():assert all(s.startswith('n') for s in entry['training_source_ids'])
    for family in ['calibration_models.json','posture_calibration_models.json']:
        records=json.loads((tmp_path/family).read_text());assert len(records)==15
        for row in records:
            train=set(row['schedule']['predictor_fit_sources'])
            assert not train&set(row['schedule']['calibration_sources'])
            assert not train&set(row['schedule']['heldout_sources'])
            assert runner.sha256_file(tmp_path/row['forecast_checkpoint'])==row['forecast_checkpoint_sha256']
    source=pd.read_csv(tmp_path/'source_scores.csv')
    assert {'calibrated_two_sided','posture_two_sided','low_motion','ridge_posture_velocity'}.issubset(source)
    for p,h in json.loads((tmp_path/'artifact_hashes.json').read_text()).items():assert runner.sha256_file(tmp_path/p)==h


def test_inner_alpha_selection_weights_each_source_equally(tmp_path,monkeypatch):
    x,v,meta,t=fixture();study=runner.Study(tmp_path,x,v,meta,t,small_config(alphas=(1.,10.)),test_only=True)
    normal=meta[meta.label.eq('normal')].copy();normal['group_id']=normal.component_id
    splits=runner.inner_source_splits(normal,seed=study.config.inner_seed)
    # Validation folds have7,7,6sources. A'sequal-foldmean(.607) looksbetter
    # than B(.62), but A'sequal-sourceerror(.637) isworse andmustlose.
    err_a={s:(.91 if len(split['validation_sources'])==7 else 0.)
           for split in splits for s in split['validation_sources']}
    monkeypatch.setattr(study,'fit_new',lambda sources,config,role:config)
    def prediction(config,indexes):
        out=np.zeros((len(indexes),8,8,2))
        out[:,0,0,0]=[err_a[study.meta.iloc[i].source_id] if config.ridge_alpha==1 else .62 for i in indexes]
        return out
    monkeypatch.setattr(study,'predict_new',prediction)
    monkeypatch.setattr(runner,'window_errors',lambda x,v,p,**kwargs:p[:,0,0,0])
    selected=study.choose_alpha(sorted(normal.source_id),'ridge_posture','synthetic_weight_check')
    assert selected['selected_alpha']==10.
    assert np.isclose(selected['inner_source_mse']['1.0'],.637)
    assert len(selected['validation_source_errors']['1.0'])==20


def test_alternate_partition_refits_keep_each_source_in_one_heldout_fold(tmp_path):
    study=runner.Study(tmp_path,*fixture(),small_config(partition_seeds=(2702,)),test_only=True)
    result=study.run_retraining()
    assert len(result)==1 and result[0]['partition_seed']==2702
    frame=pd.read_csv(tmp_path/'retraining_source_scores.csv')
    assert len(frame)==25 and frame.source_id.nunique()==25
    for entry in study.ledger.values():assert all(s.startswith('n') for s in entry['training_source_ids'])


def test_real_entry_rejects_changed_config_before_loading_data(monkeypatch):
    from dataclasses import asdict
    frozen=json.loads(json.dumps(asdict(runner.RunConfig())))
    monkeypatch.setattr(runner,'verify_freeze',lambda path=None:{'run_config':frozen})
    with pytest.raises(ValueError,match='configuration differs'):
        runner.run(config=runner.RunConfig(seeds=(27,)))


def test_real_entry_rejects_changed_cohort_before_loading_or_fitting(tmp_path,monkeypatch):
    from dataclasses import asdict
    (tmp_path/'identity.json').write_text('approved identity')
    (tmp_path/'cohort.csv').write_text('approved membership')
    frozen={'run_config':json.loads(json.dumps(asdict(runner.RunConfig()))),
            'cohort_identity_sha256':runner.sha256_file(tmp_path/'identity.json'),
            'cohort_membership_sha256':runner.sha256_file(tmp_path/'cohort.csv')}
    monkeypatch.setattr(runner,'verify_freeze',lambda path=None:frozen)
    # A different membership file must be refused before any tensor is loaded.
    (tmp_path/'cohort.csv').write_text('changed membership')
    monkeypatch.setattr(runner.v2_data,'load_cohort',lambda *a,**k:pytest.fail('Changed cohort was loaded'))
    with pytest.raises(ValueError,match='Cohort identity or membership differs'):
        runner.run(cohort=tmp_path)
