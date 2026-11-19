"""External recording identity and exact shared observation contracts."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import pandas as pd

from worldmodels.pose import normalize_window
from worldmodels.v2_data import common_support
from worldmodels.data import sha256_file
from worldmodels.v2_external import load_external,load_recording_cache,windows_from_recording


def fixture():
    n=32
    xy=np.ones((n,33,2),np.float32)*100
    xy[:,23:25]=[100,200]
    xy[:,11:13]=[100,100]
    xy[:,25:]=[100,300]
    xy[:,25:,0]+=np.arange(n)[:,None]
    raw=dict(xy=xy,confidence=np.ones((n,33),np.float32),frame_numbers=np.arange(1,n+1),
        decoded_indexes=np.arange(n),timestamp_seconds=np.arange(n)*.1,nominal_seconds=np.arange(n)*.1)
    row=SimpleNamespace(video_id='001_PD_01_ML',participant_id='PD_ML_001',label='parkinsons',original_group='PD',severity='ML',
        direction='01',path='local.mov',sha256='fakefixture',fps=10.,width=640,height=480)
    return raw,row


def test_exact_shared_window_normalization_and_participant_sequence_contract():
    raw,row=fixture();x,v,meta,attr=windows_from_recording(raw,row)
    assert attr=={'candidate_windows':2,'retained_windows':2,'rejection_counts':{}}
    expected,observed=normalize_window(raw['xy'][:24],raw['confidence'][:24],context_frames=16)
    np.testing.assert_array_equal(x[0],expected)
    np.testing.assert_array_equal(v[0],observed)
    assert common_support(np.stack(v)).all()
    assert meta[0]['source_id']==meta[0]['component_id']==row.participant_id
    assert meta[0]['sequence_id']==meta[0]['recording_id']==row.video_id
    assert json.loads(meta[0]['frame_numbers'])==list(range(1,25))
    other=SimpleNamespace(**row.__dict__);other.video_id='001_PD_02_ML'
    second=windows_from_recording(raw,other)[2]
    assert meta[0]['source_id']==second[0]['source_id']
    assert meta[0]['sequence_id']!=second[0]['sequence_id']
    assert meta[0]['window_id']!=second[0]['window_id']


def test_future_changes_cannot_change_prefix_normalization_and_bad_times_are_explicit():
    raw,row=fixture();x,v,meta,attr=windows_from_recording(raw,row)
    changed={k:a.copy() for k,a in raw.items()};changed['xy'][16:]*=3
    altered=windows_from_recording(changed,row)[0]
    np.testing.assert_array_equal(x[0][:16],altered[0][:16])
    bad={k:a.copy() for k,a in raw.items()};bad['timestamp_seconds'][19]=bad['timestamp_seconds'][18]
    xx,vv,mm,aa=windows_from_recording(bad,row)
    assert not xx and aa['rejection_counts']=={'invalid_actual_decoder_times':2}
    missing={k:a.copy() for k,a in raw.items()};missing['confidence'][:16]=0
    xx,vv,mm,aa=windows_from_recording(missing,row)
    assert aa['retained_windows']==0 and sum(aa['rejection_counts'].values())==2


def test_cache_identity_native_origin_and_requested_frames(tmp_path):
    raw,row=fixture();n=len(raw['xy']);expected={'requested_frame_numbers':list(range(1,n+1)),'video_sha256':'exact'}
    path=tmp_path/'raw.npz';np.savez(path,**raw,provenance_json=np.array(json.dumps(expected)))
    result=load_recording_cache(path,expected)
    np.testing.assert_array_equal(result['decoded_indexes'],np.arange(n))
    with pytest.raises(ValueError,match='identity mismatch'):
        load_recording_cache(path,dict(expected,video_sha256='changed'))
    raw['decoded_indexes']+=1
    np.savez(path,**raw,provenance_json=np.array(json.dumps(expected)))
    with pytest.raises(ValueError,match='origin'):
        load_recording_cache(path,expected)


def test_external_loader_checks_identity_grouping_and_pilot_scope(tmp_path):
    raw,row=fixture();x,v,rows,attr=windows_from_recording(raw,row)
    np.savez(tmp_path/'windows.npz',x=np.stack(x),valid=np.stack(v))
    pd.DataFrame(rows).to_csv(tmp_path/'windows.csv',index=False)
    (tmp_path/'request.json').write_text('{}')
    identity=dict(data_sha256=sha256_file(tmp_path/'windows.npz'),metadata_sha256=sha256_file(tmp_path/'windows.csv'))
    (tmp_path/'identity.json').write_text(json.dumps(identity))
    prep=dict(status='complete',mode='bounded_extraction_pilot',frames=24,context_frames=16,sample_hz=10.,**identity,
        request_sha256=sha256_file(tmp_path/'request.json'),identity_sha256=sha256_file(tmp_path/'identity.json'))
    (tmp_path/'preparation.json').write_text(json.dumps(prep))
    names=['windows.npz','windows.csv','request.json','identity.json','preparation.json']
    (tmp_path/'artifact_hashes.json').write_text(json.dumps({n:sha256_file(tmp_path/n) for n in names}))
    with pytest.raises(ValueError,match='pilot'):
        load_external(tmp_path)
    xx,vv,mm,tt=load_external(tmp_path,allow_pilot=True)
    np.testing.assert_array_equal(xx,np.stack(x))
    assert len(mm)==2 and tt.shape==(2,24)
    (tmp_path/'request.json').write_text('{"changed":true}')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_external(tmp_path,allow_pilot=True)
