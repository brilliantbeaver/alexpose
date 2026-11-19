"""Scientific invariants with explicitly synthetic fixtures, never clinical evidence."""
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from aaai_gait.data import (LABELS, JOINTS, POSE_VERSION, CSV_COLUMNS,
    build_manifest, manifest_digest, grouped_split, assert_group_disjoint,
    normalize_pose, normalize_window, _resample, load_windows, extract_sequence, file_sha256)
from aaai_gait.features import build_features, fit_rf
from aaai_gait.evaluation import (aggregate_sequence_predictions, classification_metrics,
    cluster_bootstrap, calibrate_threshold, align_probabilities)


def write_csv(root, label, sequence, source, frames=range(1,65)):
    path = root/'data'/'gavd'/label/f'{sequence}.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    rows = [dict(seq=sequence,frame_num=f,cam_view='left side',gait_event='',dataset='synthetic',gait_pat=label,
                 bbox="{'left': 10, 'top': 10, 'width': 80, 'height': 80}",
                 vid_info="{'width': 100, 'height': 100, 'mime_type': 'video/mp4'}",id=source,
                 url=f'https://www.youtube.com/watch?v={source}') for f in frames]
    pd.DataFrame(rows,columns=CSV_COLUMNS).to_csv(path,index=False)
    return path


def fixture_manifest(tmp_path):
    for k,label in enumerate(LABELS):
        for i in range(5):
            write_csv(tmp_path,label,f'sequence{k}_{i}',f'{k:05d}{i:06d}')
    return build_manifest(tmp_path)


def test_manifest_refresh_and_duplicate_rejection(tmp_path):
    m=fixture_manifest(tmp_path)
    before=manifest_digest(m)
    path=tmp_path/'data/gavd/normal/sequence0_0.csv'
    table=pd.read_csv(path,keep_default_na=False,dtype=str)
    table.loc[0,'gait_event']='changed'
    table.to_csv(path,index=False)
    assert manifest_digest(build_manifest(tmp_path)) != before
    pd.concat([table,table.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Duplicate source frame'):
        build_manifest(tmp_path)


def test_group_split_reproducible_and_has_all_classes(tmp_path):
    manifest=fixture_manifest(tmp_path)
    split=grouped_split(manifest,attempts=100)
    pd.testing.assert_frame_equal(split,grouped_split(manifest,attempts=100))
    assert_group_disjoint(split)
    # A sequence duplicated across partitions is a true leakage trigger.
    bad=pd.concat([split,split.iloc[:1].assign(split='test' if split.iloc[0].split != 'test' else 'train')])
    with pytest.raises(ValueError,match='leaks'):
        assert_group_disjoint(bad)


def test_missing_group_coverage_fails(tmp_path):
    manifest=fixture_manifest(tmp_path)
    bad=manifest[~(manifest.label.eq('stroke') & ~manifest.sequence_id.eq('sequence2_0'))]
    with pytest.raises(ValueError,match='three source groups'):
        grouped_split(bad)


def test_normalization_and_missingness():
    rng=np.random.default_rng(1)
    xy=rng.normal(size=(64,33,2)).astype(np.float32)
    confidence=np.ones((64,33))
    a=normalize_pose(xy,confidence)
    b=normalize_pose(xy*3+np.array([100,200]),confidence)
    np.testing.assert_allclose(a,b,atol=1e-4)
    confidence[8,27]=0.1
    c=normalize_pose(xy,confidence)
    assert np.isnan(c[8,27]).all()
    # Unobserved source frames never become finite interpolated movement.
    out,_=_resample(xy[:3],np.array([1,2,5]),fps=10,target_fps=20)
    assert np.isnan(out[3:8]).all()


def test_windows_reject_stale_and_missing_inputs(tmp_path):
    m=fixture_manifest(tmp_path).iloc[:1].assign(split='train')
    r=m.iloc[0]
    cache=tmp_path/'cache'; cache.mkdir()
    xy=np.random.default_rng(2).normal(size=(64,33,2)).astype(np.float32)
    context=dict(version=POSE_VERSION,csv_sha256=r.csv_sha256,source_id=r.source_id,frame_number_base=1,min_visibility=.5,bbox_margin=.05,model_sha256="test")
    np.savez(cache/f'{r.sequence_id}.npz',xy=xy,raw_xy=xy,confidence=np.ones((64,33)),frame_numbers=np.arange(1,65),fps=15,provenance=json.dumps(context))
    x,meta=load_windows(m,cache,require_coverage=False)
    assert x.shape==(3,32,16,2)
    assert set(meta.source_id)=={r.source_id}
    assert set(meta.split)=={'train'}
    context['csv_sha256']='old'
    np.savez(cache/f'{r.sequence_id}.npz',xy=xy,raw_xy=xy,confidence=np.ones((64,33)),frame_numbers=np.arange(1,65),fps=15,provenance=json.dumps(context))
    x,meta=load_windows(m,cache,require_coverage=False)
    assert not len(x)
    assert meta.attrs['exclusions'][0]['reason']=='pose_provenance_mismatch'
    with pytest.raises(ValueError,match='lacks classes'):
        load_windows(m,cache)


def test_crop_frame_mapping_and_cache_invalidation(tmp_path):
    cv2=pytest.importorskip('cv2')
    path=write_csv(tmp_path,'normal','crop_test','00000000000',frames=[2,3])
    movie=tmp_path/'synthetic.avi'
    writer=cv2.VideoWriter(str(movie),cv2.VideoWriter_fourcc(*'MJPG'),10,(100,100))
    for color in (10,50,90):
        writer.write(np.full((100,100,3),color,np.uint8))
    writer.release()
    seen=[]
    class FakeExtractor:
        def extract_from_image(self,image,model_path=None):
            seen.append(float(image.mean()))
            return SimpleNamespace(keypoints=[SimpleNamespace(id=j,x=20+j,y=30+j,confidence=.9) for j in range(33)])
    row=dict(sequence_id='crop_test',source_id='00000000000',csv_path=path,csv_sha256=file_sha256(path))
    out=extract_sequence(row,movie,tmp_path/'poses',frame_number_base=1,extractor=FakeExtractor(),bbox_margin=0)
    with np.load(out,allow_pickle=False) as data:
        np.testing.assert_allclose(data['raw_xy'][0,0],[30,40])
        np.testing.assert_array_equal(data['frame_numbers'],[2,3])
    assert seen[0]>40 and seen[1]>80
    with pytest.raises(ValueError,match='Stale pose cache'):
        extract_sequence(row,movie,tmp_path/'poses',frame_number_base=0,extractor=FakeExtractor(),bbox_margin=0)


def test_features_kinematics_and_train_only_imputation():
    rng=np.random.default_rng(4)
    x=rng.normal(size=(10,32,16,2)).astype(np.float32)
    a,names=build_features(x,fps=15)
    b,_=build_features(x,fps=30)
    assert a.shape==(10,36) and len(names)==36
    j=names.index('left_wrist_speed_mean_torso_per_s')
    np.testing.assert_allclose(b[:,j],2*a[:,j])
    train=np.array([[1,np.nan],[3,1],[5,2],[7,3]],float)
    estimator=fit_rf(train,['normal','normal','stroke','stroke'],n_estimators=5)
    np.testing.assert_allclose(estimator.named_steps['impute'].statistics_,[4,2])
    estimator.predict([[1000,np.nan]])
    np.testing.assert_allclose(estimator.named_steps['impute'].statistics_,[4,2])


def test_mixed_label_recording_is_not_merged():
    metadata=pd.DataFrame(dict(sequence_id=['a','a','b'],source_id=['same']*3,
                              label=['normal','normal','stroke'],split=['test']*3))
    probabilities=np.array([[.8,.2],[.6,.4],[.1,.9]])
    meta,p=aggregate_sequence_predictions(metadata,probabilities,classes=['normal','stroke'])
    assert len(meta)==2
    np.testing.assert_allclose(p,[[.7,.3],[.1,.9]])
    report=classification_metrics(meta.label,p,classes=['normal','stroke'])
    assert report['macro_f1']==1 and report['abnormal_auroc']==1


def test_bootstrap_pairing_and_small_calibration():
    y=np.array(['normal','stroke']*10)
    groups=np.repeat(np.arange(10),2)
    p=np.tile([[.9,.1],[.1,.9]],(10,1))
    report=cluster_bootstrap(y,p,groups,classes=['normal','stroke'],repeats=50,other_probabilities=p)
    assert report['valid_repeats']==50
    assert report['intervals_95']['macro_f1']==[0,0]
    assert np.isinf(calibrate_threshold(np.arange(7),alpha=.05))
    assert calibrate_threshold(np.arange(19),alpha=.05)==18
    np.testing.assert_allclose(align_probabilities(p,['normal','stroke'],['stroke','normal']),p[:,::-1])


def test_future_coordinates_do_not_change_past_normalization():
    x=np.random.default_rng(8).normal(size=(32,16,2)).astype(np.float32)
    perturbed=x.copy()
    perturbed[16:] *= 100
    np.testing.assert_array_equal(normalize_window(x)[:16],normalize_window(perturbed)[:16])


def test_one_based_frame_timestamp_zero():
    xy=np.ones((4,33,2),np.float32)
    out,times=_resample(xy,np.arange(1,5),fps=30,target_fps=30)
    assert times[0]==0
    np.testing.assert_allclose(times,np.arange(4)/30)


def test_validity_gate_preserves_missing_coordinates(tmp_path):
    m=fixture_manifest(tmp_path).iloc[:1].assign(split='train')
    r=m.iloc[0]; cache=tmp_path/'validity_cache';cache.mkdir()
    raw=np.random.default_rng(12).normal(size=(32,33,2)).astype(np.float32)
    context=dict(version=POSE_VERSION,csv_sha256=r.csv_sha256,source_id=r.source_id,
                 frame_number_base=1,min_visibility=.5,bbox_margin=.05,model_sha256='test')
    def write(array):
        np.savez(cache/f'{r.sequence_id}.npz',raw_xy=array,xy=array,
                 confidence=np.ones((32,33)),frame_numbers=np.arange(1,33),fps=15,
                 provenance=json.dumps(context))
    missing=raw.copy();missing[:,[13,14,15,16]]=np.nan
    write(missing)
    x,meta=load_windows(m,cache,require_coverage=False)
    assert x.shape==(1,32,16,2)
    assert np.isnan(x).any()  # Missing values were not invented or zero-filled.
    assert meta.iloc[0].valid_token_fraction==.75
    missing[:,29]=np.nan;write(missing)
    x,meta=load_windows(m,cache,require_coverage=False)
    assert len(x)==0
    # Even when the overall configurable gate allows it, each half must have
    # at least half its tokens observed; the forecaster needs both sides.
    missing=raw.copy();missing[:16,[13,14,15,16,25,26,27,28,29]]=np.nan
    write(missing)
    x,_=load_windows(m,cache,require_coverage=False,min_valid_token_fraction=.7)
    assert len(x)==0
    write(np.concatenate([missing[16:],missing[:16]],axis=0))
    x,_=load_windows(m,cache,require_coverage=False,min_valid_token_fraction=.7)
    assert len(x)==0


def test_features_preserve_unknown_statistics_for_train_imputer():
    x=np.random.default_rng(9).normal(size=(4,32,16,2)).astype(np.float32)
    x[:,:,4]=np.nan  # Entire left wrist is unavailable in these synthetic clips.
    x[:,5:8,12]=np.nan
    features,names=build_features(x)
    assert np.isnan(features[:,names.index('left_wrist_speed_mean_torso_per_s')]).all()
    assert np.isfinite(features[:,names.index('right_wrist_speed_mean_torso_per_s')]).all()
    rf=fit_rf(features,['normal','normal','stroke','stroke'],n_estimators=5)
    assert np.isfinite(rf.predict_proba(features)).all()


def test_mixed_label_source_stays_together(tmp_path):
    manifest=fixture_manifest(tmp_path)
    manifest.loc[manifest.sequence_id.eq('sequence1_0'),'source_id']=manifest.iloc[0].source_id
    result=grouped_split(manifest,seed=12)
    assert_group_disjoint(result)
    assert result[result.source_id.eq(manifest.iloc[0].source_id)].split.nunique()==1


def test_low_source_fps_excluded_before_upsampling(tmp_path):
    m=fixture_manifest(tmp_path).iloc[:1].assign(split='train')
    r=m.iloc[0]; cache=tmp_path/'fps_cache';cache.mkdir()
    raw=np.random.default_rng(19).normal(size=(64,33,2)).astype(np.float32)
    context=dict(version=POSE_VERSION,csv_sha256=r.csv_sha256,source_id=r.source_id,
                 frame_number_base=1,min_visibility=.5,bbox_margin=.05,model_sha256='test')
    np.savez(cache/f'{r.sequence_id}.npz',raw_xy=raw,xy=raw,confidence=np.ones((64,33)),
             frame_numbers=np.arange(1,65),fps=10,provenance=json.dumps(context))
    x,meta=load_windows(m,cache,require_coverage=False,target_fps=15)
    assert len(x)==0
    assert meta.attrs['exclusions'][0]['reason']=='source_fps_below_target'
    x,meta=load_windows(m,cache,require_coverage=False,target_fps=10)
    assert len(x)>0 and set(meta.source_fps)=={10}


def test_raw_future_change_leaves_resampled_prefix_unchanged():
    raw=np.random.default_rng(29).normal(size=(64,16,2)).astype(np.float32)
    altered=raw.copy();altered[32:]*=100
    a,_=_resample(raw,np.arange(1,65),fps=30,target_fps=15)
    b,_=_resample(altered,np.arange(1,65),fps=30,target_fps=15)
    np.testing.assert_array_equal(normalize_window(a)[:16],normalize_window(b)[:16])
