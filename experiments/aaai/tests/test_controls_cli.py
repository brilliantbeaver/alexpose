"""Frozen-reference restoration and paired-control normalization boundaries."""
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from aaai_gait.demo import synthetic_corpus
from aaai_gait.predictive_profile import ProfileReference,GROUPS

path=Path(__file__).resolve().parents[1]/'scripts/run_study_controls.py'
spec=importlib.util.spec_from_file_location('controls_cli',path)
controls=importlib.util.module_from_spec(spec);spec.loader.exec_module(controls)


def test_control_normalization_uses_original_crop_prefix_and_cancels_old_scale():
    x,meta=synthetic_corpus(frames=40,sources_per_class=2);x=x[:1];meta=meta.iloc[:1].assign(start_seconds=0.)
    base,_,_=controls.normalize_control_windows(x,meta)
    changed=x.copy();changed[:,20:]*=100
    after,_,_=controls.normalize_control_windows(changed,meta)
    np.testing.assert_allclose(base[:,:20],after[:,:20])
    rescaled,_,_=controls.normalize_control_windows(x*7,meta)
    np.testing.assert_allclose(base,rescaled,atol=1e-6)
    missing=x.copy();missing[:,4:20,[0,1]]=np.nan
    absent,_,audit=controls.normalize_control_windows(missing,meta)
    assert len(absent)==0 and not audit.normalization_eligible.any()


def test_restore_reference_preserves_fold_and_parameter_hash():
    rows=pd.DataFrame([dict(source_id=f'normal{i}',binary_label=0,split='val',target_type='latent',
        scorable=True,outer_fold=2,**{f'benefit_{g}':i+j for j,g in enumerate(GROUPS)}) for i in range(5)])
    reference=ProfileReference.fit(rows);data=reference.to_dict()
    restored=controls.restore_reference(data,2,rows.source_id)
    assert restored==reference and restored.parameters_sha256==reference.parameters_sha256
    with pytest.raises(ValueError,match='fold'):controls.restore_reference(data,1,rows.source_id)
    corrupted={**data,'mean':[10,20,30]}
    with pytest.raises(ValueError,match='fingerprint'):controls.restore_reference(corrupted,2,rows.source_id)
    with pytest.raises(ValueError,match='sources'):controls.restore_reference(data,2,['different'])


def test_total_paired_dropout_is_unscorable_and_retains_requested_support():
    meta=pd.DataFrame([dict(source_id='normal',sequence_id='seq',label='normal',binary_label=0,
                           start_seconds=0.,fps=15,split='development',outer_fold=0)])
    rows=pd.DataFrame([dict(window_index=0,joint=j,patch_index=t,mask_id=0,target_type='coordinate',
                           valid=True,loss_own=2.,loss_patch=1.,loss_full=.5) for j in (13,14,15,16,25,26,27,28,29,30,31,32) for t in range(4)])
    profiles,support,_=controls._paired_profiles(rows.iloc[:0],rows,meta)
    assert not profiles.scorable.any() and profiles.raw_full.isna().all()
    assert profiles.requested_targets.sum()==len(rows)
    assert support.n_distinct_patches.eq(0).all()
