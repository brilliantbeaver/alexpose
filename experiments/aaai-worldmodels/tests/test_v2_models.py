"""Scientific target weighting, causal-feature, and fitting checks for v2."""
import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import Ridge

from worldmodels.v2_masks import MaskConfig,batch_masks
from worldmodels.v2_models import (ForecastConfig,TARGET_JOINTS,causal_features,
    target_values,fit_forecaster,predict_forecaster,masked_window_loss,
    hierarchical_weights,_fit_ridge)


def fixture(n=18):
    rng=np.random.default_rng(27)
    x=rng.normal(scale=.2,size=(n,16,16,2)).astype('float32')
    v=rng.random((n,16,16))>.1;v[:,-1]=True
    t=np.broadcast_to(np.arange(16)*.1,(n,16)).copy()
    future=np.repeat(x[:,-1:],8,axis=1)+rng.normal(scale=.04,size=(n,8,16,2))
    fv=rng.random((n,8,16))>.15
    x[~v]=np.nan;future[~fv]=np.nan
    meta=pd.DataFrame({'source_id':[f's{i//3}' for i in range(n)],
        'sequence_id':[f'q{i//2}' for i in range(n)],'label':['normal']*n})
    return x,v,t,future,fv,meta


@pytest.mark.parametrize('mode',['mlp','ridge_posture','ridge_posture_velocity'])
def test_hidden_and_missing_values_cannot_change_features(mode):
    x,v,t,_,_,_=fixture()
    h,_=batch_masks(x,v,t,MaskConfig('uniform'))
    f,a,k=causal_features(x,v,t,mode=mode,hidden=h)
    changed=x.copy();changed[h|~v]=1e20
    ff,aa,kk=causal_features(changed,v,t,mode=mode,hidden=h)
    np.testing.assert_array_equal(f,ff);np.testing.assert_array_equal(a,aa);np.testing.assert_array_equal(k,kk)
    changed[h|~v]=np.nan
    np.testing.assert_array_equal(f,causal_features(changed,v,t,mode=mode,hidden=h)[0])
    assert f.shape[1]=={'mlp':832,'ridge_posture':24,'ridge_posture_velocity':64}[mode]


def test_earlier_fill_cannot_see_later_prefix():
    x=np.zeros((1,16,16,2));v=np.zeros((1,16,16),bool);v[:,10:]=True;x[:,10:]=2
    t=np.arange(16)[None]*.1
    f,a,_=causal_features(x,v,t)
    rel=f[:,:512].reshape(1,16,16,2)
    np.testing.assert_array_equal(rel[:,:10],-2)
    np.testing.assert_array_equal(rel[:,10:],0)
    np.testing.assert_array_equal(a,2)


def test_equal_leg_loss_ignores_missing_targets_and_has_finite_gradient():
    p=torch.zeros((2,8,8,2),requires_grad=True)
    y=torch.ones_like(p);m=torch.ones_like(p,dtype=torch.bool)
    y[:,:,4:]=3;m[0,:,4:]=False;m[0,0,4]=True
    y=torch.where(m,y,torch.full_like(y,float('nan')))
    losses=masked_window_loss(p,y,m)
    torch.testing.assert_close(losses,torch.tensor([5.,5.]))
    losses.mean().backward();assert torch.isfinite(p.grad).all()
    assert torch.all(p.grad[~m]==0)


def test_hierarchy_balances_group_then_sequence():
    m=pd.DataFrame({'source_id':['a']*4+['b']*2,'sequence_id':['q']*3+['r','s','s']})
    w=hierarchical_weights(m)
    assert np.isclose(w[:4].sum(),w[4:].sum())
    assert np.isclose(w[:3].sum(),w[3])


def test_ridge_matches_independent_solver_with_equal_leg_loss():
    rng=np.random.default_rng(5);n=28
    f=rng.normal(size=(n,5));y=rng.normal(size=(n,8,8,2));m=rng.random(y.shape)>.25
    weights=rng.uniform(.5,2,n);alpha=4.
    y[~m]=np.nan
    safe=np.where(m,y,0.)
    model=_fit_ridge(f,safe,m,weights,alpha)
    z=(f-model['mean'])/model['scale'];flat=safe.reshape(n,-1);mm=m.reshape(n,-1)
    count=np.stack([m[:,:,:4].reshape(n,-1).sum(1),m[:,:,4:].reshape(n,-1).sum(1)],axis=1)
    legs=np.tile(np.repeat([0,1],8),8)
    for k in [0,8,63,127]:
        use=mm[:,k];w=weights[use]*64/count[use,legs[k]]
        ref=Ridge(alpha=alpha).fit(z[use],flat[use,k],sample_weight=w)
        np.testing.assert_allclose(z@model['coef'][:,k]+model['intercept'][k],ref.predict(z),atol=1e-10)


@pytest.mark.parametrize('mode',['mlp','ridge_posture','ridge_posture_velocity'])
def test_real_fit_api_save_replay_and_training_only_identity(mode,tmp_path):
    x,v,t,y,m,meta=fixture()
    config=ForecastConfig(mode=mode,steps=4,mask_policy='uniform' if mode=='mlp' else 'none')
    ck=fit_forecaster(x,v,t,y,m,meta,config)
    p=predict_forecaster(ck,x[:3],v[:3],t[:3]);assert p.shape==(3,8,8,2)
    path=tmp_path/'checkpoint.joblib';joblib.dump(ck,path)
    np.testing.assert_array_equal(p,predict_forecaster(joblib.load(path),x[:3],v[:3],t[:3]))
    assert ck['training_source_ids']==sorted(meta.source_id.unique())
    assert ck['seconds']>0
    changed=meta.copy();changed.loc[0,'label']='stroke'
    with pytest.raises(ValueError,match='normal'):
        fit_forecaster(x,v,t,y,m,changed,config)
    with pytest.raises(ValueError,match='exact'):
        predict_forecaster(ck,np.concatenate([x[:3],x[:3,:8]],axis=1),
                           np.concatenate([v[:3],v[:3,:8]],axis=1),np.arange(24)[None].repeat(3,axis=0))


def test_training_seed_is_reproducible_and_pattern_features_have_no_coordinates():
    x,v,t,y,m,meta=fixture()
    cfg=ForecastConfig(steps=5,seed=41,mask_policy='motion_balanced')
    a=fit_forecaster(x,v,t,y,m,meta,cfg);b=fit_forecaster(x,v,t,y,m,meta,cfg)
    np.testing.assert_array_equal(predict_forecaster(a,x,v,t),predict_forecaster(b,x,v,t))
    f,_,_=causal_features(x,v,t,pattern_only=True)
    changed=x*17
    ff,_,_=causal_features(changed,v,t,pattern_only=True)
    np.testing.assert_array_equal(f,ff)


def test_ridge_velocity_uses_actual_times_and_future_is_separate():
    x=np.zeros((1,16,16,2));v=np.ones((1,16,16),bool);t=np.arange(16)[None]*.1
    x[:,:,:,0]=np.arange(16)[None,:,None]*.2
    f,_,_=causal_features(x,v,t,mode='ridge_posture_velocity')
    vel=f[:,24:40].reshape(1,8,2)
    np.testing.assert_allclose(vel[:,:,0],2)
    np.testing.assert_allclose(vel[:,:,1],0)
