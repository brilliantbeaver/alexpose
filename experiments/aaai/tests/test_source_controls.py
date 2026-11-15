"""Analytical source weights and training-only nuisance/motion controls."""
import numpy as np
import pandas as pd
import pytest
from aaai_gait.source_controls import (TrainOnlyTransform,aggregate_source_features,
    fit_view_vocabulary,source_view_features,fit_source_controls,align_frozen_windows)
from aaai_gait.demo import synthetic_corpus


def test_all_missing_training_feature_uses_zero_without_reading_test():
    transform=TrainOnlyTransform.fit([[1.,np.nan],[3.,np.nan]])
    np.testing.assert_array_equal(transform.medians,[2.,0.])
    np.testing.assert_array_equal(transform.all_missing_columns,[False,True])
    before=transform.scaler.mean_.copy()
    np.testing.assert_allclose(transform.transform([[1000.,88.],[np.nan,np.nan]]),[[998.,88.],[0.,0.]])
    np.testing.assert_array_equal(transform.scaler.mean_,before)


def test_source_hierarchy_and_mixed_camera_views_do_not_count_windows_as_people():
    x=np.random.default_rng(42).normal(size=(5,32,16,2));x[[0,1,4],:,0,:]=np.nan
    meta=pd.DataFrame(dict(source_id=['source']*5,sequence_id=['long']*4+['short'],
        label=['stroke']*4+['style'],binary_label=1,fps=15.,camera_views=['front']*4+['right side']))
    result,names=aggregate_source_features(x,meta)
    assert len(result)==1 and result.iloc[0].categories=='stroke|style' and len(names)==36
    assert result.iloc[0].availability_11==.25
    values,diagnostics=source_view_features(meta,fit_view_vocabulary(meta,['source']))
    assert values.loc['source','view_front']==.5 and values.loc['source','view_right side']==.5
    assert not diagnostics.unknown_view_present.any()
    mixed=meta.iloc[:1].assign(camera_views='front,right side')
    values,diagnostics=source_view_features(mixed,('front',))
    assert values.loc['source','view_front']==.5
    assert diagnostics.loc['source','unknown_view_fraction']==.5
    assert diagnostics.loc['source','unknown_views']=='right side'


def cohort():
    x,meta=synthetic_corpus(seed=42,sources_per_class=6)
    keep=meta.label.eq('normal')|(meta.label.eq('stroke')&meta.source_id.isin([f'synthetic-2-{i}' for i in range(4)]))
    x=x[keep];meta=meta.loc[keep].reset_index(drop=True)
    meta['source_id']=[('n' if label=='normal' else 'a')+source.rsplit('-',1)[-1] for label,source in zip(meta.label,meta.source_id)]
    meta['binary_label']=meta.label.ne('normal').astype(int)
    meta['camera_views']=meta.source_id.map(lambda s:'front' if s.startswith('n') else 'right side')
    meta.loc[meta.source_id.eq('n4'),'camera_views']='reference_only_view'
    meta.loc[meta.source_id.isin(['n5','a0']),'camera_views']='unseen_test_view'
    meta['start_seconds']=0.;meta['window_id']=meta.sequence_id+':0'
    roles=dict(train=['n0','n1','n2','n3'],val=['n4'],test=['n5','a0'],auxiliary=['a1','a2','a3'])
    return x,meta,roles


def test_fits_do_not_use_reference_or_test_features_views_or_labels():
    x,meta,roles=cohort();features,names=aggregate_source_features(x,meta)
    features.loc[features.source_id.isin(roles['train']),names[0]]=np.nan
    first,models,audit=fit_source_controls(features,meta,roles,0,motion_columns=names)
    changed=features.copy();held=changed.source_id.isin(roles['val']+roles['test'])
    changed.loc[held,names]=1e7
    altered_meta=meta.copy();altered_meta.loc[altered_meta.source_id.isin(roles['val']+roles['test']),'camera_views']='other_unseen_view'
    _,models2,_=fit_source_controls(changed,altered_meta,roles,0,motion_columns=names)
    np.testing.assert_array_equal(models['motion_transform'].medians,models2['motion_transform'].medians)
    np.testing.assert_array_equal(models['motion_transform'].scaler.mean_,models2['motion_transform'].scaler.mean_)
    np.testing.assert_array_equal(models['motion_model'].covariance_,models2['motion_model'].covariance_)
    assert models['motion_transform'].medians[0]==0
    assert models['normal_view_vocabulary']==('front',)
    assert models['supervised_view_vocabulary']==('front','right side')
    assert models2['normal_view_vocabulary']==models['normal_view_vocabulary']
    assert first.normal_unknown_view_present.all() and first.supervised_unknown_view_present.all()
    assert audit['normal_fit_source_ids']==roles['train']
    assert set(audit['supervised_fit_source_ids'])==set(roles['train']+roles['auxiliary'])
    assert 'reference_only_view' not in audit['supervised_view_vocabulary']
    assert not audit['hyperparameters_selected_from_test']
    with pytest.raises(ValueError,match='overlap'):
        fit_source_controls(features,meta,{**roles,'val':['n0']},0,motion_columns=names)


def test_frozen_alignment_reorders_exact_windows_and_rejects_time_changes():
    x,meta,_=cohort();selected=meta.iloc[[5,0,9]].copy()
    order=np.random.default_rng(9).permutation(len(meta))
    restored,aligned=align_frozen_windows(x[order],meta.iloc[order].reset_index(drop=True),selected)
    np.testing.assert_array_equal(restored,x[[5,0,9]])
    assert aligned.source_id.tolist()==selected.source_id.tolist()
    changed=selected.copy();changed.loc[changed.index[0],'start_seconds']=.001
    with pytest.raises(ValueError,match='start time'):align_frozen_windows(x,meta,changed)
    with pytest.raises(ValueError,match='unique'):align_frozen_windows(x,meta,pd.concat([selected,selected]))
