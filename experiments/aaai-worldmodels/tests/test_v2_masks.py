"""Canonical mask information boundaries and fair-budget invariants."""
import numpy as np
import pytest

from worldmodels.data import JOINT_IDS
from worldmodels.v2_masks import (MaskConfig, POLICIES, REFERENCES, make_mask,
                                prefix_motion, protected_tokens, batch_masks)


def fixture(n=1):
    rng=np.random.default_rng(91)
    x=rng.normal(size=(n,16,16,2))
    v=rng.random((n,16,16))>.15
    v[:,-1]=True
    x[~v]=np.nan
    t=np.broadcast_to(np.cumsum(rng.uniform(.09,.11,16)),(n,16)).copy()
    return x,v,t


@pytest.mark.parametrize('policy',POLICIES)
def test_budget_support_replay_and_no_natural_missing_leak(policy):
    x,v,t=fixture()
    h,p,a=make_mask(x[0],v[0],t[0],MaskConfig(policy))
    assert not np.any(h & (~v[0] | p))
    assert p[-1].all()
    for joint in REFERENCES:
        j=JOINT_IDS.index(joint)
        np.testing.assert_array_equal(p[:,j],v[0,:,j])
    n=(v[0]&~p).sum()
    expected=0 if policy=='none' else int(np.floor(.4*n+.5))
    assert int(h.sum())==a['hidden_tokens']==expected
    changed=x[0].copy();changed[~v[0]]=1e30
    np.testing.assert_array_equal(h,make_mask(changed,v[0],t[0],MaskConfig(policy))[0])
    np.testing.assert_array_equal(h,make_mask(x[0],v[0],t[0],MaskConfig(policy))[0])


def test_all_corruptions_have_exact_common_budget_on_varied_missingness():
    x,v,t=fixture(19)
    for i in range(len(x)):
        counts=[make_mask(x[i],v[i],t[i],MaskConfig(p),seed=i)[0].sum()
                for p in POLICIES if p!='none']
        assert len(set(counts))==1


def test_latest_observed_anchor_is_protected_even_before_last_frame():
    _,v,_=fixture();v[0,7:,8]=False
    p=protected_tokens(v[0]);assert p[6,8] and not p[7:,8].any()


def test_motion_uses_only_observed_actual_time_pairs_and_unknown_is_selectable():
    x=np.zeros((16,16,2));v=np.ones((16,16),bool);t=np.arange(16)*.1
    x[1:,8,0]=np.arange(1,16)*.2
    m=prefix_motion(x,v,t)
    np.testing.assert_allclose(m[1:,8],2.)
    assert np.isnan(m[0]).all()
    v[5,8]=False;x[5,8]=np.nan
    m=prefix_motion(x,v,t)
    assert np.isnan(m[5:7,8]).all()
    for policy in ('motion_balanced','motion_focused'):
        assert any(make_mask(x,v,t,MaskConfig(policy),seed=s)[0][0,8] for s in range(100))


def test_future_suffix_not_accepted_and_bad_time_rejected():
    x,v,t=fixture()
    with pytest.raises(ValueError,match='exact'):
        make_mask(np.concatenate([x[0],x[0][:8]]),np.concatenate([v[0],v[0][:8]]),np.arange(24)*.1)
    with pytest.raises(ValueError,match='increasing'):
        make_mask(x[0],v[0],t[0][::-1])


def test_batch_masks_reproduce_and_do_not_mutate_inputs():
    x,v,t=fixture(4);vc=v.copy()
    masks,audit=batch_masks(x,v,t,MaskConfig('anatomical_tubes'),seed=41)
    np.testing.assert_array_equal(masks,batch_masks(x,v,t,MaskConfig('anatomical_tubes'),seed=41)[0])
    np.testing.assert_array_equal(v,vc)
    assert len(audit)==4
