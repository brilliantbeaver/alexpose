"""Independent invariants and tiny fixture execution, not clinical evidence."""
from dataclasses import replace
import json
import numpy as np
import pandas as pd
import pytest
import torch
from worldmodels.demo import make_demo
from worldmodels.model import ModelConfig,SkeletonJEPA
from worldmodels.masking import forecast_mask
from worldmodels.data import sha256_file
from worldmodels.revision import (RevisionProtocol,RevisionJEPA,last_observed,common_support,clone_forecaster,
    source_schedule,predict,energy_score,baselines,develop,freeze,evaluate_once,_code_identity,uncertainty_summary)

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

def tiny():
    return RevisionProtocol(frames=4,context_frames=2,horizons=(1,2),seeds=(7,),dim=16,encoder_layers=1,
                            pretrain_steps=1,forecast_steps=1,batch_size=2,prior_samples=2,scoring_batch_size=2)

def test_persistence_skip_and_future_leakage():
    p=tiny();d=make_demo(n=3,frames=4)
    base=SkeletonJEPA(ModelConfig(frames=4,dim=16,encoder_layers=1))
    model=clone_forecaster(base,p,'residual_det',7).eval()
    x,v=torch.from_numpy(d['x']),torch.from_numpy(d['valid'])
    out=model(x,v,forecast_mask(v,2))['coordinates']
    anchor,_,_=last_observed(x,v,2)
    torch.testing.assert_close(out,anchor[:,None].expand_as(out))
    torch.nn.init.normal_(model.coordinate_head.weight,std=.1)
    out=model(x,v,forecast_mask(v,2))['coordinates']
    changed=x.clone();changed[:,2:]+=100
    torch.testing.assert_close(out,model(changed,v,forecast_mask(v,2))['coordinates'])
    with pytest.raises(ValueError,match='future'):
        model(x,v,common_support(v,2)&False)

def test_unseen_prefix_joint_never_enters_common_loss_and_control_anchor_fixed():
    p=tiny();d=make_demo(n=2,frames=4);v=d['valid'];v[:,:2,0]=False
    support=common_support(torch.from_numpy(v),2)
    assert not support[:,:,0].any()
    guard=RevisionJEPA(ModelConfig(frames=4,dim=16,encoder_layers=1),2,residual=True)
    with pytest.raises(ValueError,match='future'):
        guard(torch.from_numpy(d['x']),torch.from_numpy(v),support)
    base=SkeletonJEPA(ModelConfig(frames=4,dim=16,encoder_layers=1))
    model=clone_forecaster(base,p,'residual_det',7)
    actual=predict(model,d['x'],v,p,seed=3)[0]
    shuffled=predict(model,d['x'],v,p,seed=3,control='shuffled')[0]
    static=predict(model,d['x'],v,p,seed=3,control='static')[0]
    np.testing.assert_array_equal(actual,shuffled);np.testing.assert_array_equal(actual,static)

def test_energy_estimator_and_equal_source_schedule():
    y=np.zeros((1,1,1,2));mask=np.ones((1,1,1),bool)
    draws=np.array([[[[[1.,1.]]]],[[[[-1.,-1.]]]]])
    np.testing.assert_allclose(energy_score(draws,y,mask),0.)
    np.testing.assert_allclose(energy_score(draws[:1],y,mask),1.)
    meta=pd.DataFrame(dict(source_id=['a','a','b','c'],sequence_id=['a1','a2','b1','c1'],label='normal',split='train'))
    a=source_schedule(meta,10,3,7);b=source_schedule(meta,10,3,7)
    assert a==b
    assert all(meta.iloc[rows].source_id.nunique()==3 for rows in a)
    bad=meta.copy();bad.loc[0,'split']='test'
    with pytest.raises(ValueError): source_schedule(bad,1,2,7)

def test_constant_velocity_uses_observation_times():
    x=np.zeros((1,4,16,2),np.float32);x[:,0]=1;x[:,1]=3
    valid=np.ones(x.shape[:-1],bool)
    output=baselines(x,valid,2,np.zeros((16,2)),np.array([15.]))
    np.testing.assert_allclose(output['velocity'][0,2:,0,0],[5,7])

def test_saturated_partial_correlation_is_not_reported():
    s=pd.DataFrame(dict(mse_h2=[1,2,3],variance_h2=[3,2,1],context_valid_fraction=[.8,.9,1.],
        future_valid_fraction=[1.,.9,.8],sample_fps=[12,15,30],prefix_mean_confidence=[.7,.8,.9],
        future_mean_confidence=[.6,.8,1.]))
    result=uncertainty_summary(s,2)
    assert result['partial_rank_correlation'] is None
    assert result['partial_rank_residual_dof']<3
    assert result['aurc_prefix_confidence'] is not None
    assert result['retained_risk']['uncertainty']['0.5']['actual_coverage']==pytest.approx(2/3)

def test_development_excludes_test_and_all_complete_arms_can_freeze(tmp_path,monkeypatch):
    import worldmodels.revision as module
    p=tiny();data=make_demo(n=9,frames=4);data['x'][6:]+=1000
    input_dir=tmp_path/'prepared';input_dir.mkdir()
    np.savez_compressed(input_dir/'windows.npz',x=data['x'],valid=data['valid'])
    meta=pd.DataFrame(dict(source_id=data['source_ids'],sequence_id=[f'q{i}' for i in range(9)],
        label='normal',split=['train']*3+['validation']*3+['test']*3,sample_fps=15.))
    meta.to_csv(input_dir/'windows.csv',index=False)
    (input_dir/'preparation.json').write_text(json.dumps(dict(context_frames=2,
        data_sha256=sha256_file(input_dir/'windows.npz'),metadata_sha256=sha256_file(input_dir/'windows.csv'))))
    original=module._score_arm
    def assert_validation(model,x,valid,meta,*args,**kwargs):
        assert set(meta.split)=={'validation'}
        assert np.abs(x).max()<100
        return original(model,x,valid,meta,*args,**kwargs)
    monkeypatch.setattr(module,'_score_arm',assert_validation)
    result=develop(input_dir,tmp_path/'development',p)
    assert len(result['records'])==3
    assert not (tmp_path/'development/test_access.json').exists()
    assert freeze(tmp_path/'development')['status']=='accepted'
    rejected=tmp_path/'rejected';rejected.mkdir()
    (rejected/'development.json').write_text(json.dumps(dict(code_sha256=_code_identity(),artifact_sha256={},
        protocol={},validation_gate={'passed':False})))
    (rejected/'freeze.json').write_text(json.dumps({'status':'rejected_execution'}))
    with pytest.raises(ValueError,match='sealed'): evaluate_once(rejected,tmp_path/'forbidden-test')
    assert not (tmp_path/'forbidden-test').exists()
