"""Replay every retained window from the recorded native-frame pose cache."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from worldmodels.pose import normalize_window
from worldmodels.data import JOINT_IDS

root=PROJECT/'artifacts/runs/restart-full10hz-v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report=json.loads((root/'preparation.json').read_text())
assert sha(root/'windows.csv')==report['metadata_sha256']
assert sha(root/'windows.npz')==report['data_sha256']
meta=pd.read_csv(root/'windows.csv')
with np.load(root/'windows.npz',allow_pickle=False) as archive:
    x=archive['x'];valid=archive['valid']
assert x.shape==(len(meta),24,16,2) and valid.shape==(len(meta),24,16)
assert x.dtype==np.float32 and valid.dtype==bool
assert np.isfinite(x).all() and np.all(x[~valid]==0)
assert meta.window_id.is_unique
assert meta.groupby('source_id').split.nunique().max()==1
assert (valid[:,:16].mean(axis=(1,2))>=.7).all()
assert (valid[:,16:].mean(axis=(1,2))>=.7).all()
assert np.all(~valid[:,16:] | valid[:,:16].any(axis=1)[:,None,:])
max_reconstruction_error=0.
for sequence,rows in meta.groupby('sequence_id'):
    with np.load(root/'sequence_poses'/f'{sequence}.npz',allow_pickle=False) as archive:
        pose={key:archive[key].copy() for key in archive.files}
    index={int(number):i for i,number in enumerate(pose['frame_numbers'])}
    assert str(pose['source_id'])==rows.source_id.iloc[0]
    for i,row in rows.iterrows():
        numbers=np.array(json.loads(row.frame_numbers))
        times=np.array(json.loads(row.timestamp_seconds))
        positions=[index[int(number)] for number in numbers]
        expected,mask=normalize_window(pose['xy'][positions],pose['confidence'][positions],context_frames=16)
        mask[16:]&=mask[:16].any(axis=0)
        expected=np.where(mask[...,None],expected,0)
        error=float(np.max(np.abs(expected-x[i])))
        max_reconstruction_error=max(error,max_reconstruction_error)
        np.testing.assert_array_equal(expected,x[i])
        np.testing.assert_array_equal(mask,valid[i])
        np.testing.assert_array_equal(times,pose['timestamp_seconds'][positions])
        np.testing.assert_allclose(np.array(json.loads(row.future_horizon_seconds)),times[16:]-times[15])
        identity=hashlib.sha256(json.dumps([row.source_id,numbers.tolist()],separators=(',',':')).encode()).hexdigest()
        assert identity==row.window_id
attrition=pd.read_csv(root/'attrition.csv')
assert attrition.retained_windows.sum()==len(meta)
assert len(attrition)==report['selected_sequences']
progress=json.loads((root/'source_progress.json').read_text())
assert progress['completed_sources']==progress['total_sources']==report['selected_sources']
assert sum(r['actual_model_calls'] for r in progress['sources'])==report['actual_model_calls_this_invocation']
for source in progress['sources']:
    assert sha(root/'source_poses'/f'{source["source_id"]}.npz')==source['source_cache_sha256']
assert sha(root/'source_snapshot/restart_data.py')==report['adapter_sha256']
checks={'identity':'restart-full10hz-reconstruction-check-v1','windows_reconstructed':len(meta),
    'sequences_reconstructed':int(meta.sequence_id.nunique()),'maximum_absolute_reconstruction_error':max_reconstruction_error,
    'all_array_shapes_dtypes_missingness_and_support_passed':True,'all_native_frames_timestamps_window_ids_matched':True,
    'source_partitions_disjoint':True,'attrition_and_calls_reconciled':True,
    'all_source_cache_payload_hashes_verified':True,'running_adapter_snapshot_hash_verified':True,
    'verified_at_utc':datetime.now(timezone.utc).isoformat(),
    'preparation_sha256':sha(root/'preparation.json'),'verifier_sha256':sha(Path(__file__)),
    'limitation':'This replay validates artifact construction and uses the same shared normalization API. It is not independent validation of source identity or diagnosis.'}
(PROJECT/'artifacts/restart_audit/full_cohort_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
