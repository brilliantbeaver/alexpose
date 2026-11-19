"""Access, grouping, weighting and scientific fixtures for angle replication."""
import numpy as np
import pandas as pd
import pytest
from aaai_gait.external_iteration02 import (angle_mask_bank,angle_patches,permitted_features,
    participant_roles,training_features,selected_predictions,dependency_fixtures)


def test_six_angle_mask_budget_and_hidden_input_boundaries():
    masks=angle_mask_bank(128,17);assert masks.shape==(128,48) and (masks.sum(1)==24).all()
    np.testing.assert_array_equal(masks,angle_mask_bank(128,17))
    p=np.random.default_rng(5).normal(size=(128,48,4));target=np.array([np.flatnonzero(m)[0] for m in masks]);f=permitted_features(p,masks,target)
    assert f['own'].shape==(128,36) and f['own_residual'].shape==(128,72) and f['both_residual'].shape==(128,72)
    altered=p.copy();altered[masks]+=10000.;g=permitted_features(altered,masks,target)
    for name in ['own','own_residual','both_residual']:np.testing.assert_array_equal(f[name],g[name])
    changed=p.copy();opposite=(np.arange(48)[None,:]%2)!=(target[:,None]%2);changed[opposite]+=500.
    g=permitted_features(changed,masks,target)
    np.testing.assert_array_equal(f['own'],g['own']);np.testing.assert_array_equal(f['own_residual'],g['own_residual'])
    assert np.abs(f['both_residual']-g['both_residual']).max()>0


def test_participant_roles_keep_reference_and_test_out_of_selection():
    rows=[dict(participant_id=f'healthy:SUBJ{i:03}',group='healthy',binary_label=0) for i in range(128)]
    rows += [dict(participant_id=f'stroke:TVC{i:03}',group='stroke',binary_label=1) for i in range(50)]
    registry=participant_roles(pd.DataFrame(rows))
    for f in range(4):
        normal=registry[registry.binary_label.eq(0)];assert normal.groupby(f'role_fold_{f}').size().to_dict()=={'reference':20,'test':32,'train':76}
        assert registry.loc[registry[f'role_fold_{f}'].ne('train'),f'inner_fold_{f}'].eq(-1).all()
        assert set(normal.loc[normal[f'role_fold_{f}'].eq('train'),f'inner_fold_{f}'])=={0,1,2}
    assert (registry[[f'role_fold_{f}' for f in range(4)]].eq('test').sum(1)==1).all()


def test_zero_gate_exact_and_only_normal_training():
    x=np.random.default_rng(4).normal(size=(12,32,6));meta=pd.DataFrame(dict(source_id=['a']*2+['b']*10,label='normal'))
    f,w,audit=training_features(x,meta,42,draws=128);assert np.isfinite(w).all() and (w>0).all()
    class Zero:
        def predict(self,a):return np.full((len(a),4),2.)
    class Explode:
        def predict(self,a):raise AssertionError('Zero-gated model was called')
    models=dict(base=Zero(),grid={'own':{1.:Explode()},'both':{1.:Explode()}})
    recipe=dict(own_multiplier=0.,both_multiplier=0.,own_alpha=1.,both_alpha=1.)
    p=selected_predictions(models,recipe,f)
    np.testing.assert_array_equal(p['base'],p['own']);np.testing.assert_array_equal(p['base'],p['both'])
    with pytest.raises(ValueError,match='normal only'):training_features(x,meta.assign(label='stroke'),42)


def test_positive_and_independent_innovation_fixtures():
    result=dependency_fixtures();assert result['status']=='pass';assert len(result['records'])==10
