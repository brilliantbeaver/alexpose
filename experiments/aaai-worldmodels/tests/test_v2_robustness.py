"""Observation stresses preserve the declared information access and support."""
import numpy as np
from worldmodels.v2_robustness import stress_inputs,shifted_opposite
from worldmodels.v2_masks import protected_tokens
from worldmodels.v2_models import causal_features,TARGET_JOINTS

def test_stresses_protect_references_anchors_and_replay_by_identity():
    rng=np.random.default_rng(17);x=rng.normal(size=(3,16,16,2));v=rng.random((3,16,16))>.1
    t=np.tile(np.arange(16)*.1,(3,1));ids=np.array(['a','b','c'])
    stresses=stress_inputs(x,v,t,ids)
    for i in range(3):
        protected=protected_tokens(v[i]);hidden=stresses['uniform_dropout'][2][i]
        assert not np.any(hidden&(~v[i]|protected))
        np.testing.assert_array_equal(x[i][protected],stresses['coordinate_noise'][0][i][protected])
    perm=np.array([2,0,1]);other=stress_inputs(x[perm],v[perm],t[perm],ids[perm])
    np.testing.assert_array_equal(stresses['uniform_dropout'][2][perm],other['uniform_dropout'][2])
    np.testing.assert_array_equal(stresses['coordinate_noise'][0][perm],other['coordinate_noise'][0])

def test_opposite_time_shift_preserves_own_leg_anchor_and_coordinate_support_pairs():
    rng=np.random.default_rng(18);x=rng.normal(size=(2,16,16,2));v=np.ones((2,16,16),bool)
    v[0,14:,TARGET_JOINTS[4]]=False;t=np.tile(np.arange(16)*.1,(2,1))
    _,before,known=causal_features(x,v,t)
    changed,cv=shifted_opposite(x,v,['first','second'],0)
    _,after,ck=causal_features(changed,cv,t)
    np.testing.assert_array_equal(before,after);np.testing.assert_array_equal(known,ck)
    np.testing.assert_array_equal(x[:,:,TARGET_JOINTS[:4]],changed[:,:,TARGET_JOINTS[:4]])
    np.testing.assert_array_equal(x[:,:,:8],changed[:,:,:8])
    for i in range(2):
        for j in TARGET_JOINTS[4:]:
            assert sorted(map(tuple,np.column_stack([x[i,:,j],v[i,:,j]])))==sorted(map(tuple,np.column_stack([changed[i,:,j],cv[i,:,j]])))
