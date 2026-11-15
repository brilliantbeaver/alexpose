"""Analytical profile examples test support, weighting and leakage boundaries."""
from dataclasses import FrozenInstanceError
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from aaai_gait.predictive_profile import (READOUT_JOINTS,GROUPS,aggregate_profiles,
    compress_loss_rows,ProfileReference,score_profiles,leave_one_reference_out,
    fold_ranking_metrics,paired_fold_bootstrap,common_target_intersection,freeze_profile_eligibility)


def example(windows=1,patches=4):
    meta=pd.DataFrame([dict(source_id='source',sequence_id='seq',label='normal',binary_label=0,
        split='val',outer_fold=0,start_seconds=w*16/15,fps=15,end_seconds=w*16/15+31/15,
        window_index=100+w) for w in range(windows)])
    rows=[]
    for w in range(windows):
        for joint in READOUT_JOINTS:
            for patch in range(patches):
                for mask in (0,1):
                    rows.append(dict(window_index=w,joint=joint,patch_index=patch,mask_id=mask,
                        loss_own=2.,loss_patch=1.5,loss_full=1.,target_type='latent',valid=True,
                        target_valid=True,own_visible_count=2))
    return pd.DataFrame(rows),meta


def reference_rows(n=5):
    rows=[]
    for i in range(n):
        row=dict(source_id=f'r{i}',binary_label=0,split='val',target_type='latent',scorable=True)
        for field in ('benefit','own','full'):
            for k,name in enumerate(GROUPS):row[f'{field}_{name}']=float(i*(k+1))
        rows.append(row)
    return pd.DataFrame(rows)


def test_duplicate_masks_and_overlapping_windows_do_not_create_support():
    rows,meta=example(windows=2,patches=3)
    meta.loc[1,['start_seconds','end_seconds']]=[0,31/15]
    p,s,_=aggregate_profiles(pd.concat([rows,rows]),meta)
    assert not p.iloc[0].scorable
    assert set(s.n_distinct_patches)=={3}
    # A fourth genuinely different time patch makes the source eligible.
    rows,meta=example(windows=2,patches=4);meta.loc[1,['start_seconds','end_seconds']]=[0,31/15]
    p,s,_=aggregate_profiles(rows,meta)
    assert p.iloc[0].scorable and set(s.n_distinct_patches)=={4}
    bad=rows.iloc[:1].copy();bad['loss_full']=99
    with pytest.raises(ValueError,match='Conflicting'):aggregate_profiles(pd.concat([rows,bad]),meta)


def test_hierarchical_source_mean_equal_joint_groups_and_matched_raw():
    rows,meta=example(windows=3)
    meta.loc[2,'sequence_id']='second'
    rows['loss_own']=rows.window_index.map({0:0.,1:10.,2:20.})
    rows['loss_full']=0.;rows['loss_patch']=0.
    p,s,_=aggregate_profiles(rows,meta)
    # First sequence mean(0,10)=5; second=20; source mean=12.5.
    assert p.iloc[0].benefit_left_leg==12.5 and p.iloc[0].raw_benefit==12.5
    rows,meta=example();rows['loss_full']=np.where(rows.joint.eq(25),12.,0.)
    rows['loss_own']=rows.loss_full+2
    hips=rows[rows.joint.eq(25)].copy();hips['joint']=23;hips[list(('loss_own','loss_patch','loss_full'))]=1e6
    p,_,_=aggregate_profiles(pd.concat([rows,hips]),meta)
    assert p.iloc[0].full_left_leg==3 and p.iloc[0].raw_full==1
    assert p.iloc[0].raw_benefit==2 and p.iloc[0].anomaly_contrast==-2


def test_missing_joint_is_unscorable_without_normal_imputation():
    rows,meta=example();rows=rows[rows.joint.ne(31)]
    p,_,_=aggregate_profiles(rows,meta)
    assert not p.iloc[0].scorable and p.iloc[0].missing_joints=='31'
    assert np.isnan(p.iloc[0].benefit_left_leg) and np.isnan(p.iloc[0].raw_full)


def test_compressed_rows_preserve_mask_weighted_means():
    rows,meta=example(windows=2)
    rows.loc[(rows.patch_index==0)&(rows.mask_id==1),'loss_own']=50
    a,sa,_=aggregate_profiles(rows,meta)
    b,sb,_=aggregate_profiles(compress_loss_rows(rows),meta)
    pd.testing.assert_frame_equal(a,b);pd.testing.assert_frame_equal(sa,sb)


def test_fixed_budget_selection_uses_no_scores_and_never_refills_after_edit():
    rows,meta=example(windows=3)
    p,s,selection=aggregate_profiles(rows,meta,fixed_budget=4,seed=73)
    changed=rows.copy();changed['loss_own']+=100
    p2,s2,selection2=aggregate_profiles(changed,meta,fixed_budget=4,seed=73)
    pd.testing.assert_frame_equal(selection,selection2)
    assert selection.groupby(['source_id','joint']).size().eq(4).all()
    assert s.n_distinct_patches.eq(4).all()
    # Remove a selected physical patch from every overlapping occurrence.
    chosen=selection.iloc[0]
    starts=meta.start_seconds.to_numpy()[rows.window_index]+rows.patch_index*4/15
    keep=~(rows.joint.eq(chosen.joint)&np.isclose(starts,chosen.patch_start_ns/1e9))
    after,_,_=aggregate_profiles(rows[keep],meta,selection=selection)
    assert not after.iloc[0].scorable


def test_valid_nan_is_model_failure_and_pair_support_is_common():
    rows,meta=example();bad=rows.copy();bad.loc[0,'loss_full']=np.nan
    with pytest.raises(FloatingPointError):aggregate_profiles(bad,meta)
    with pytest.raises(FloatingPointError):common_target_intersection(rows,bad)
    bad.loc[0,'valid']=False
    a,b=common_target_intersection(rows,bad)
    assert len(a)==len(b)==len(rows)-1
    bad=rows.copy();bad.loc[0,'own_visible_count']=1
    with pytest.raises(ValueError,match='two visible'):aggregate_profiles(bad,meta)


def test_reference_sample_variance_fixed_shrinkage_and_train_test_isolation():
    rows=reference_rows();ref=ProfileReference.fit(rows)
    matrix=rows[[f'benefit_{g}' for g in GROUPS]].to_numpy()
    variance=matrix.var(axis=0,ddof=1)
    np.testing.assert_allclose(ref.reference_variance,.5*variance+.5*variance.mean())
    query=rows.iloc[:1].assign(source_id='test',split='test').copy()
    for i,g in enumerate(GROUPS):query[f'benefit_{g}']=ref.mean[i]+np.sqrt(ref.reference_variance[i])
    np.testing.assert_allclose(ref.score(query),[1.])
    with pytest.raises(FrozenInstanceError):ref.field='own'
    with pytest.raises(ValueError,match='validation'):ProfileReference.fit(rows.assign(split='test'))
    with pytest.raises(ValueError,match='validation'):ProfileReference.fit(rows.assign(binary_label=1))
    with pytest.raises(ValueError,match='overlaps'):ProfileReference.fit(rows,training_source_ids=['r0'])
    with pytest.raises(ValueError,match='Insufficient'):ProfileReference.fit(rows.iloc[:3])
    assert ref.parameters_sha256==ProfileReference.fit(rows.sample(frac=1,random_state=3)).parameters_sha256
    altered=rows.copy();altered.loc[0,'benefit_arms']+=.1
    assert ref.parameters_sha256!=ProfileReference.fit(altered).parameters_sha256
    folded=rows.assign(outer_fold=0)
    with pytest.raises(ValueError,match='one outer fold'):
        ProfileReference.fit(folded.assign(outer_fold=[0,0,0,0,1]))
    with pytest.raises(ValueError,match='different outer fold'):
        ProfileReference.fit(folded).score(query.assign(outer_fold=1))


def test_profile_eligibility_is_fixed_before_labels_losses_or_folds():
    from aaai_gait.model import JOINT_INDICES
    _,meta=example(windows=2)
    meta.loc[1,'source_id']='missing_arm';meta.loc[1,'sequence_id']='other'
    x=np.ones((2,32,16,2),np.float32)
    # Only three observed patches of the left wrist: insufficient distinct
    # target support regardless of how many masks or overlapping windows exist.
    x[1,12:,JOINT_INDICES.index(15)]=np.nan
    keep,ledger=freeze_profile_eligibility(x,meta)
    np.testing.assert_array_equal(keep,[True,False])
    assert ledger.set_index('source_id').loc['missing_arm','joint_15_distinct_patches']<=3
    changed=meta.assign(label='stroke',binary_label=1,split='test',outer_fold=3)
    keep2,ledger2=freeze_profile_eligibility(x*100+73,changed)
    np.testing.assert_array_equal(keep,keep2)
    assert ledger.attrs['eligibility_sha256']==ledger2.attrs['eligibility_sha256']
    # Repeating a single three-patch window cannot manufacture four physical
    # observations, even with all 16 mask draws repeated.
    repeated=np.repeat(x[1:2],3,axis=0)
    repeated_meta=pd.concat([meta.iloc[1:2]]*3,ignore_index=True)
    keep3,ledger3=freeze_profile_eligibility(repeated,repeated_meta)
    assert not keep3.any() and not ledger3.profile_eligible.any()


def test_fixed_budget_is_identical_across_neural_and_linear_target_units():
    rows,meta=example(windows=3)
    latent=rows.copy();coordinate=rows.assign(target_type='coordinate')
    linear=coordinate.assign(loss_own=80.,loss_patch=50.,loss_full=25.)
    _,_,selection_latent=aggregate_profiles(latent,meta,fixed_budget=4,seed=42)
    _,_,selection_coordinate=aggregate_profiles(coordinate,meta,fixed_budget=4,seed=42)
    _,_,selection_linear=aggregate_profiles(linear,meta,fixed_budget=4,seed=42)
    pd.testing.assert_frame_equal(selection_latent,selection_coordinate)
    pd.testing.assert_frame_equal(selection_latent,selection_linear)


def test_zero_reference_variance_and_leave_one_out_are_explicit():
    rows=reference_rows(4)
    for g in GROUPS:rows[f'benefit_{g}']=1.
    ref=ProfileReference.fit(rows)
    assert ref.reference_variance==(1e-8,)*3
    result=leave_one_reference_out(rows,rows.iloc[:1].assign(source_id='test',split='test'),reference=ref)
    assert len(result)==4 and result.sensitivity.all()
    assert result.n_reference_sources.eq(3).all() and not result.main_reference_minimum_met.any()


def score_table():
    rows=[]
    for fold,offset in enumerate((100.,0.,10.,20.)):
        for label in (0,1):
            for i in range(3):
                rows.append(dict(source_id=f's{fold}_{label}_{i}',outer_fold=fold,binary_label=label,
                    split='test',target_type='latent',categories='normal' if not label else ('stroke|style' if i==0 else 'stroke'),
                    a=offset+label+.2*i,b=offset+label+.2*i))
    return pd.DataFrame(rows)


def test_primary_is_within_fold_not_pooled_and_categories_keep_support():
    scores=score_table();report=fold_ranking_metrics(scores,['a'])
    assert report['methods']['a']['primary_auroc']==1
    assert roc_auc_score(scores.binary_label,scores.a)<1
    assert report['methods']['a']['normal_weighted_ap']==1
    assert report['methods']['a']['categories']['style']['n_category_sources']==4
    assert report['methods']['a']['categories']['pregnant']['normal_weighted_auroc'] is None
    assert report['coverage']['n_common_scorable_sources']==24


def test_normal_count_weights_and_paired_bootstrap():
    scores=score_table()
    scores.loc[scores.outer_fold.eq(1),'a']=-scores.loc[scores.outer_fold.eq(1),'a']
    scores=scores[~(scores.outer_fold.eq(0)&scores.binary_label.eq(0)&scores.source_id.str.endswith('_2'))]
    report=fold_ranking_metrics(scores,['a'])
    assert np.isclose(report['methods']['a']['primary_auroc'],8/11)
    scores['b']=scores['a']
    interval=paired_fold_bootstrap(scores,'a','b',repeats=24)
    assert interval['point_difference']==0 and interval['interval_95']==[0.,0.]
    with pytest.raises(ValueError,match='one outer-test'):fold_ranking_metrics(pd.concat([scores,scores.iloc[:1]]),['a'])
    absent=scores[~(scores.outer_fold.eq(0)&scores.binary_label.eq(0))]
    assert fold_ranking_metrics(absent,['a'])['methods']['a']['primary_auroc'] is None
    with pytest.raises(ValueError,match='both binary'):paired_fold_bootstrap(absent,'a','b',repeats=2)
