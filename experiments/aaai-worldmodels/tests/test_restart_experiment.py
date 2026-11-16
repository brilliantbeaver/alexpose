"""Leakage, masking, weighting and solver checks for the fresh dynamics audit."""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from worldmodels.restart_experiment import (C,H,ARMS,LEGS,features,targets,fill_prefix,
    fit_grid,predict,predict_checkpoint,hierarchical_weights,source_mean,source_folds,error_by_window,shift_offsets)


def fixture(n=30):
    rng=np.random.default_rng(44)
    x=rng.normal(size=(n,C+H,16,2)).astype(np.float32)
    valid=rng.random((n,C+H,16))>.1
    return x,valid


def test_future_and_missing_values_cannot_change_inputs_or_predictions():
    x,v=fixture();ids=[f'window{i}' for i in range(len(x))];rng=np.random.default_rng(5)
    changed=x.copy();cv=v.copy()
    changed[:,C:]=rng.normal(size=changed[:,C:].shape)*100
    cv[:,C:]=~cv[:,C:]
    changed[:,:C][~v[:,:C]]=99999
    for arm in ARMS:
        for leg in (0,1):
            f,_,_=features(x,v,leg,arm,ids);ff,_,_=features(changed,cv,leg,arm,ids)
            np.testing.assert_array_equal(f,ff)
            y,m=targets(x,v,leg)
            model=fit_grid(f,y,m,np.ones(len(x)),[1])[1]
            ck=dict(model=model,arm=arm,leg=leg,beta=.5)
            np.testing.assert_allclose(predict_checkpoint(ck,x,v,ids),predict_checkpoint(ck,changed,cv,ids))
            np.testing.assert_allclose(predict_checkpoint(ck,x,v,ids),predict_checkpoint(ck,x[:,:C],v[:,:C],ids))


def test_masked_ridge_matches_independent_weighted_solver():
    rng=np.random.default_rng(7)
    x=rng.normal(size=(70,12));y=rng.normal(size=(70,4));mask=rng.random(y.shape)>.15
    weights=rng.uniform(.1,3,len(x));alpha=3.
    m=fit_grid(x,y,mask,weights,[alpha])[alpha]
    z=(x-m['mean'])/m['scale']
    for k in range(y.shape[1]):
        keep=mask[:,k]
        adjusted=weights*y.shape[1]/mask.sum(1)
        ref=Ridge(alpha=alpha).fit(z[keep],y[keep,k],sample_weight=adjusted[keep])
        np.testing.assert_allclose(predict(m,x)[:,k],ref.predict(z),atol=1e-10)
    altered=y.copy();altered[~mask]=np.nan
    other=fit_grid(x,altered,mask,weights,[alpha])[alpha]
    np.testing.assert_allclose(m['coef'],other['coef'])


def test_source_sequence_window_weighting_and_aggregation():
    meta=pd.DataFrame(dict(source_id=['a']*4+['b']*2,sequence_id=['a1','a1','a1','a2','b1','b1'],label=['normal']*6))
    w=hierarchical_weights(meta)
    assert np.isclose(w[:4].sum(),w[4:].sum())
    assert np.isclose(w[:3].sum(),w[3])
    values=source_mean(meta,[0,0,0,10,3,3]).set_index('source_id').value
    assert values['a']==5 and values['b']==3


def test_shift_keeps_own_history_target_anchor_and_capacity():
    x,v=fixture(3)
    for leg in (0,1):
        actual,anchor,known=features(x,v,leg,'bilateral')
        shifted,sanchor,sknown=features(x,v,leg,'shifted',['a','b','c'])
        assert shifted.shape==actual.shape
        np.testing.assert_array_equal(anchor,sanchor)
        np.testing.assert_array_equal(known,sknown)
        # First 128 entries are eight times eight joints times two coordinates.
        aa=actual[:,:128].reshape(3,8,8,2);ss=shifted[:,:128].reshape(3,8,8,2)
        np.testing.assert_array_equal(aa[:,:,:4],ss[:,:,:4])
        np.testing.assert_array_equal(aa[:,-1],ss[:,-1])
        assert not np.allclose(aa[:,:-1,4:],ss[:,:-1,4:])
        for i,offset in enumerate(shift_offsets(['a','b','c'])):
            np.testing.assert_array_equal(np.roll(aa[i,:-1,4:],offset,axis=0),ss[i,:-1,4:])
        va=actual[:,128:192].reshape(3,8,8);vs=shifted[:,128:192].reshape(3,8,8)
        for i,offset in enumerate(shift_offsets(['a','b','c'])):
            np.testing.assert_array_equal(np.roll(va[i,:-1,4:],offset,axis=0),vs[i,:-1,4:])


def test_fill_never_uses_later_prefix_observation_for_earlier_value():
    x=np.ones((1,24,16,2));v=np.zeros((1,24,16),bool)
    x[:,8]=12;v[:,8]=True
    a,_,anchor,known=fill_prefix(x,v)
    np.testing.assert_array_equal(a[:,:8],0)
    np.testing.assert_array_equal(a[:,8:],12)
    assert known.all()


def test_source_folds_do_not_split_repeated_windows():
    rows=[]
    for label in ('normal','stroke'):
        for s in range(7):
            rows.extend([dict(source_id=f'{label}{s}',label=label)]*(s+1))
    meta=pd.DataFrame(rows);folds=source_folds(meta)
    assert set(folds)==set(meta.source_id)
    assert set(folds.values())==set(range(5))
    assert folds==source_folds(meta.sample(frac=1,random_state=4))


def test_missing_target_never_affects_error():
    target=np.ones((2,H*4*2));pred=np.zeros_like(target);mask=np.ones_like(target,bool)
    mask[:,::3]=False;target[~mask]=np.nan
    np.testing.assert_allclose(error_by_window(target,pred,mask),1)


def test_shift_offsets_are_stable_varied_and_nonidentity():
    ids=[f'window-{i}' for i in range(40)]
    offsets=shift_offsets(ids)
    assert set(offsets)==set(range(1,7))
    np.testing.assert_array_equal(offsets,shift_offsets(ids))
    np.testing.assert_array_equal(offsets[::-1],shift_offsets(ids[::-1]))


def test_coordinate_support_reweighting_preserves_window_mass():
    x=np.array([[0.],[1.],[2.]])
    y=np.array([[1.,5.],[2.,4.],[3.,3.]])
    mask=np.array([[True,False],[True,True],[False,True]])
    w=np.array([2.,1.,3.])
    effective=w*y.shape[1]/mask.sum(1)
    np.testing.assert_allclose((effective[:,None]*mask).sum(1),w*y.shape[1])
    m=fit_grid(x,y,mask,w,[1])[1]
    assert np.isfinite(predict(m,x)).all()
