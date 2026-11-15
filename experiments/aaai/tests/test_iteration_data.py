"""Scientific cohort invariants for the fresh iteration, without model outcomes."""
import json
import numpy as np
import pandas as pd
import pytest
from aaai_gait.data import JOINTS, normalize_window
from aaai_gait.iteration_data import (
    historical_mask_bank, lower_limb_eligibility, make_source_registry,
    iteration_fold_metadata, normalization_parameters, enrich_and_build_parents,
)


def fixture_metadata():
    rows=[dict(source_id=f'n{i:02}',sequence_id=f'nseq{i}',label='normal',binary_label=0) for i in range(28)]
    rows += [dict(source_id=f'a{i:02}',sequence_id=f'aseq{i}',label='stroke',binary_label=1) for i in range(12)]
    return pd.DataFrame(rows)


def test_roles_disjoint_internal_folds_and_duplicate_grouping():
    meta=fixture_metadata();media=pd.DataFrame(dict(source_id=meta.source_id,video_sha256=['hash'+s for s in meta.source_id]))
    registry=make_source_registry(meta,media)
    for fold in range(4):
        assigned=iteration_fold_metadata(meta,registry,fold)
        normal=assigned[assigned.binary_label.eq(0)]
        assert normal.groupby('split').size().to_dict()=={'reference':5,'test':7,'train':16}
        fit=normal[normal.split.eq('train')]
        assert sorted(fit.inner_fold.unique())==[0,1,2]
        assert sorted(fit.groupby('inner_fold').size())==[5,5,6]
        assert assigned[assigned.split.ne('train')].inner_fold.eq(-1).all()
    # Duplicate byte identities cross no outer or inner role, even if source IDs differ.
    media.loc[media.source_id.eq('n01'),'video_sha256']='hashn00'
    registry=make_source_registry(meta,media)
    pair=registry[registry.source_id.isin(['n00','n01'])]
    assert pair.cluster_id.nunique()==1
    for f in range(4):
        assert pair[f'role_fold_{f}'].nunique()==1
        assert pair[f'inner_fold_{f}'].nunique()==1
    media.loc[media.source_id.eq('a00'),'video_sha256']='hashn00'
    with pytest.raises(ValueError,match='mixes normal'):
        make_source_registry(meta,media)


def test_uniform_bank_and_deduplicated_support_match_historical():
    from aaai_gait.context_prediction import ContextConfig, make_mask_bank
    historical=make_mask_bank(ContextConfig(),mask_seed=17,mask_repeats=16).numpy().reshape(16,8,16)
    np.testing.assert_array_equal(historical_mask_bank(),historical)
    x=np.ones((2,32,16,2),np.float32)
    meta=pd.DataFrame([dict(source_id='s',sequence_id='seq',label='normal',binary_label=0,window_id='seq:0')]*2)
    keep,ledger=lower_limb_eligibility(x,meta)
    assert keep.all()
    for j in range(25,33):assert ledger[f'joint_{j}_distinct_patches'].iloc[0]<=8
    # Missing upper limbs cannot exclude complete lower-limb readouts.
    x[:,:,2:6]=np.nan
    keep,_=lower_limb_eligibility(x,meta)
    assert keep.all()
    x[:,:,JOINTS.index(25)]=np.nan
    keep,_=lower_limb_eligibility(x,meta)
    assert not keep.any()


def test_parent48_grid_keeps_short_clips_and_original_normalization(tmp_path):
    raw=np.random.default_rng(4).normal(size=(48,33,2)).astype(np.float32)
    raw[:,23]=[-1.,0.];raw[:,24]=[1.,0.]
    raw[:,11]=[-1.,2.];raw[:,12]=[1.,2.]
    np.savez(tmp_path/'s.npz',raw_xy=raw,confidence=np.ones((48,33),np.float32),frame_numbers=np.arange(1,49),fps=15.,provenance=json.dumps({}))
    annotations=pd.DataFrame(dict(frame_num=np.arange(1,49),cam_view=['side']*20+['front']*28,gait_event=['']*48))
    annotations.loc[24,'gait_event']='initial_contact';csv=tmp_path/'s.csv';annotations.to_csv(csv,index=False)
    manifest=pd.DataFrame([dict(sequence_id='s',source_id='source',csv_path=str(csv),label='normal',binary_label=0,prior_use='previous_study',camera_views='front,side')])
    meta=pd.DataFrame([dict(window_index=0,window_id='s:0',sequence_id='s',source_id='source',start_seconds=0.,end_seconds=31/15,camera_views='front,side')])
    enriched,parents,times,central,pelvis,pm=enrich_and_build_parents(meta,manifest,tmp_path,True)
    assert len(parents)==1  # No unnecessary requirement for 56 raw samples.
    np.testing.assert_array_equal(parents[0],raw[:,JOINTS])
    np.testing.assert_allclose(central[0],normalize_window(raw[8:40,JOINTS]),equal_nan=True)
    scale,anchors=normalization_parameters(raw[8:40,JOINTS])
    np.testing.assert_allclose(pelvis[0],anchors)
    np.testing.assert_allclose(central[0],(parents[0,8:40]-anchors[:,None])/scale)
    assert pm.normalization_scale.iloc[0]==scale
    assert np.isclose(times[0,-1]-times[0,0],47/15)
    assert np.isclose(pm.end_seconds.iloc[0]-pm.start_seconds.iloc[0],31/15)
    assert bool(enriched.camera_view_mixed.iloc[0]) and bool(pm.camera_view_mixed.iloc[0])
    assert pm.annotation_event_count.iloc[0]==1
    # Per-frame view join must use the central interval, not all sequence annotations.
    annotations['cam_view']='side';annotations.loc[:6,'cam_view']='front';annotations.to_csv(csv,index=False)
    *_,pm=enrich_and_build_parents(meta,manifest,tmp_path,True)
    assert pm.camera_views.iloc[0]=='side' and not pm.camera_view_mixed.iloc[0]
