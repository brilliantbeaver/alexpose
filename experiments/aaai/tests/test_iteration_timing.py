from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest

from aaai_gait.data import normalize_window
from aaai_gait.iteration_data import normalization_parameters
from aaai_gait.iteration_timing import (
    parent_mask_bank,prepare_timing_pair,select_parent_budget,
    build_timing_census,summarize_source_pairs,_score_pair,
    run_timing_fold,
)
from aaai_gait.retained_prediction import fit_local_baseline,fit_correction_grid


def articulated_parent(seed=9):
    rng=np.random.default_rng(seed);t=np.arange(48)/15
    raw=np.zeros((48,16,2),np.float32)
    raw[:,:,0]=np.arange(16)[None,:]*.1+np.sin(t[:,None]*2)
    raw[:,:,1]=np.arange(16)[None,:]*.2+np.cos(t[:,None])*.2
    raw+=rng.normal(0,.03,raw.shape).astype(np.float32)
    central=normalize_window(raw[8:40]);scale,pelvis=normalization_parameters(raw[8:40])
    return raw,central,pelvis,scale


def test_parent_mask_and_zero_offset_preserve_every_permitted_value():
    masks=parent_mask_bank()
    assert masks.shape==(16,12,16) and np.all(masks.sum((1,2))==96)
    np.testing.assert_array_equal(masks,parent_mask_bank())
    args=articulated_parent()
    for side in (0,1):
        pair=prepare_timing_pair(*args,masks,0,side)
        np.testing.assert_array_equal(pair.aligned,pair.edited)
        assert np.all(pair.diagnostics["displacement_rms_torso"]==0)
        assert pair.eligible.any()
    with pytest.raises(ValueError,match="no-wrap"):
        prepare_timing_pair(*args,masks,8,0)


def test_hidden_donor_values_and_hidden_hip_cannot_reappear():
    raw,central,pelvis,scale=articulated_parent()
    masks=np.repeat(parent_mask_bank()[:1],16,axis=0)
    before=prepare_timing_pair(raw,central,pelvis,scale,masks,4,0)
    modified=raw.reshape(12,4,16,2).transpose(0,2,1,3).copy()
    modified[masks[0]]+=10000
    modified=modified.transpose(0,2,1,3).reshape(48,16,2)
    after=prepare_timing_pair(modified,central,pelvis,scale,masks,4,0)
    np.testing.assert_array_equal(before.hidden,after.hidden)
    permitted=np.repeat((~before.hidden).reshape(16,8,16),4,axis=1)
    np.testing.assert_array_equal(before.edited[permitted],after.edited[permitted])
    for patch in range(8):
        donor_hip_hidden=masks[:,patch+3,7]
        assert after.hidden.reshape(16,8,16)[donor_hip_hidden,patch,9::2].all()


def test_distal_transform_preserves_target_and_own_leg_with_fixed_reference():
    raw,central,pelvis,scale=articulated_parent()
    original=raw.copy();anchor=pelvis.copy()
    masks=parent_mask_bank();pair=prepare_timing_pair(raw,central,pelvis,scale,masks,-4,0)
    np.testing.assert_array_equal(pair.aligned[:,:,6::2],pair.edited[:,:,6::2])
    np.testing.assert_array_equal(raw,original);np.testing.assert_array_equal(pelvis,anchor)
    for m,patch,joint in np.argwhere(~pair.hidden.reshape(16,8,16)):
        if joint not in (9,11,13,15):continue
        original_frames=np.arange(8+4*patch,12+4*patch);donor=original_frames-4
        expected=((raw[donor,joint]-raw[donor,7])+raw[original_frames,7]-pelvis[4*patch:4*patch+4])/scale
        np.testing.assert_allclose(pair.edited[m,4*patch:4*patch+4,joint],expected,rtol=1e-6,atol=2e-7)
    noise=prepare_timing_pair(raw,central,pelvis,scale,masks,-4,0,variant="noise")
    drop=prepare_timing_pair(raw,central,pelvis,scale,masks,-4,0,variant="dropout")
    np.testing.assert_array_equal(noise.hidden,pair.hidden)
    assert np.all(drop.edited_hidden|~drop.hidden)
    assert drop.diagnostics["removed_distal_tokens"].sum()>0


def test_real_coordinate_prediction_invariance_and_zero_response():
    raw,central,pelvis,scale=articulated_parent()
    x=np.stack([articulated_parent(s)[1] for s in range(6)])
    ids=np.array([f"f{i//2}" for i in range(6)]);labels=np.full(6,"normal")
    base=fit_local_baseline(x,source_ids=ids,labels=labels,draws=12)
    bundle=fit_correction_grid(base,x,source_ids=ids,labels=labels,draws=12,alphas=(1.,))[1.]
    row=SimpleNamespace(timing_window_index=0,source_id="test",sequence_id="seq",normalization_scale=scale,start_seconds=8/15)
    pair=prepare_timing_pair(raw,central,pelvis,scale,parent_mask_bank(),4,0)
    scored=_score_pair(pair,{"U":bundle},row,4,"timing")
    assert len(scored) and scored.own_prediction_max_change.eq(0).all() and scored.target_max_change.eq(0).all()
    zero=prepare_timing_pair(raw,central,pelvis,scale,parent_mask_bank(),0,0)
    scored=_score_pair(zero,{"U":bundle},row,0,"zero")
    assert scored.effect.eq(0).all()
    crop=prepare_timing_pair(raw,central,pelvis,scale,parent_mask_bank(),4,0,variant="whole_body")
    cropped=_score_pair(crop,{"U":bundle},row,4,"whole_body")
    assert not cropped.fixed_target.any() and cropped.target_max_change.gt(0).any()


def test_budget_and_score_free_census_have_separate_population(tmp_path):
    raw,central,pelvis,scale=articulated_parent()
    metadata=pd.DataFrame([dict(source_id="normal-a",sequence_id="seq",timing_window_index=i,
        parent_start_seconds=i*16/15,start_seconds=(i*16+8)/15,parent_window_id=f"seq:{i*16}",
        source_fps=30.,normalization_scale=scale) for i in range(3)])
    registry=pd.DataFrame([dict(source_id="normal-a",binary_label=0,**{f"role_fold_{f}":"test" for f in range(4)}),
                           dict(source_id="no-parent",binary_label=0,**{f"role_fold_{f}":"test" for f in range(4)})])
    data=SimpleNamespace(parent_raw=np.repeat(raw[None],3,axis=0),parent_x=np.repeat(central[None],3,axis=0),
        parent_pelvis=np.repeat(pelvis[None],3,axis=0),parent_times=np.arange(48)[None]/15+np.arange(3)[:,None]*16/15,
        parent_metadata=metadata,source_registry=registry)
    assert select_parent_budget(metadata,2)=={0,2}
    full,budget,audit=build_timing_census(data,tmp_path,max_parents_per_source=1)
    assert audit["model_outcomes_used"] is False and audit["parent_count"]==3
    assert set(full.source_id)=={"normal-a","no-parent"}
    assert full.loc[full.source_id.eq("no-parent"),"eligible"].eq(False).all()
    assert np.all(full.distinct_target_patches>=budget.distinct_target_patches)
    assert (tmp_path/"parent_frame_provenance.csv.gz").exists()


def test_mask_repetitions_do_not_inflate_distinct_target_support():
    rows=[]
    for joint in range(25,33):
        for repeat in range(16):
            rows.append(dict(source_id="s",sequence_id="seq",parent_index=0,joint=joint,patch_index=0,patch_start_ns=1,
                policy="U",variant="timing",offset=4,mask_id=repeat,effect=1.,fixed_target=True))
    source,joints=summarize_source_pairs(pd.DataFrame(rows))
    assert not source.eligible.any() and joints.distinct_patches.eq(1).all()
    assert source.effect.isna().all()


def test_timing_runner_checks_source_roles_and_saves_auditable_calls(tmp_path):
    raw,central,pelvis,scale=articulated_parent()
    x=np.stack([articulated_parent(s)[1] for s in range(6)])
    ids=np.array([f"fit{i//2}" for i in range(6)]);labels=np.full(6,"normal")
    base=fit_local_baseline(x,source_ids=ids,labels=labels,draws=12)
    bundle=fit_correction_grid(base,x,source_ids=ids,labels=labels,draws=12,alphas=(1.,))[1.]
    metadata=pd.DataFrame([dict(source_id="test",sequence_id="seq",timing_window_index=0,
        parent_start_seconds=0.,start_seconds=8/15,parent_window_id="seq:0",source_fps=30.,normalization_scale=scale)])
    registry=pd.DataFrame([dict(source_id="test",binary_label=0,role_fold_0="test")])
    data=SimpleNamespace(parent_raw=raw[None],parent_x=central[None],parent_pelvis=pelvis[None],
        parent_times=np.arange(48)[None]/15,parent_metadata=metadata,source_registry=registry)
    table,audit=run_timing_fold(data,0,{"U":bundle},tmp_path,max_parents_per_source=1)
    assert len(table) and audit["model_calls_real"] and audit["zero_identity_assertion_rows"]>0
    assert audit["exact_zero_and_own_invariance_assertions_passed"]
    assert (tmp_path/"target_rows/test.csv.gz").exists()
    with pytest.raises(ValueError,match="forbidden role"):
        run_timing_fold(data,0,{"U":bundle},tmp_path/"forbidden",source_ids=["test"],scope="inner_validation")
