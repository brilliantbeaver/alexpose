import numpy as np
import pytest

from worldmodels.restart_data import sampled_spans, supported_window, strict_gate, window_identity, _load_source_cache


def fixture_pose():
    xy=np.ones((24,33,2),dtype=float)
    xy[:,23:25]=[10,20]
    xy[:,11:13]=[10,10]
    confidence=np.ones((24,33),dtype=float)
    return xy,confidence


def test_fixed_nominal_time_sampling_and_gaps():
    spans=sampled_spans(np.r_[np.arange(1,100),np.arange(110,150)],29.97)
    assert len(spans)==2
    for span in spans:
        assert np.max(np.abs((span-span[0])/29.97-np.arange(len(span))/10))<.5/29.97+1e-8
        assert np.all(np.diff(span)>0)
    assert spans[0][-1]<100 and spans[1][0]==110
    with pytest.raises(ValueError):sampled_spans([1,2,3],8)
    with pytest.raises(ValueError):sampled_spans([1,1,2],30)


def test_no_future_scale_leakage_and_common_support():
    xy,confidence=fixture_pose()
    confidence[:16,15]=0
    first,valid,_=supported_window(xy,confidence)
    assert not valid[16:,4].any()
    assert not first[16:,4].any()
    changed=xy.copy();changed[16:]*=10
    second,_,_=supported_window(changed,confidence)
    np.testing.assert_array_equal(first[:16],second[:16])


def test_separate_prefix_and_future_quality():
    xy,confidence=fixture_pose()
    confidence[:6]=0
    assert supported_window(xy,confidence)[2]=='prefix_quality_below_0.7'
    xy,confidence=fixture_pose();confidence[16:19]=0
    assert supported_window(xy,confidence)[2]=='future_quality_below_0.7'


def test_unresolved_geometry_rejected_and_window_identity_stable():
    xy,confidence=fixture_pose()
    gated,reason=strict_gate(xy[0],confidence[0],{}, {},100,100)
    assert reason=='unresolved_bbox'
    assert not gated.any()
    assert window_identity('source',[1,4,7])==window_identity('source',np.array([1,4,7]))
    assert window_identity('source',[1,4,7])!=window_identity('source',[1,4,8])


def test_resume_rejects_changed_identity_and_frame_origin(tmp_path):
    import json
    expected={'requested_frame_numbers':[1,4],'video_sha256':'original'}
    path=tmp_path/'source.npz'
    fields=dict(xy=np.zeros((2,33,2)),confidence=np.ones((2,33)),frame_numbers=np.array([1,4]),
                decoded_indexes=np.array([0,3]),provenance_json=np.array(json.dumps(expected)))
    np.savez(path,**fields)
    assert _load_source_cache(path,expected)['xy'].shape==(2,33,2)
    with pytest.raises(ValueError,match='identity mismatch'):
        _load_source_cache(path,dict(expected,video_sha256='changed'))
    fields['decoded_indexes']=np.array([1,4])
    np.savez(path,**fields)
    with pytest.raises(ValueError,match='frame origin mismatch'):
        _load_source_cache(path,expected)
