"""Independent external input audit; no forecasting or anomaly scores are read."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, time
import numpy as np
import pandas as pd
REPOSITORY=Path(__file__).resolve().parents[4]
PROJECT=REPOSITORY/'experiments/aaai-worldmodels';sys.path.insert(0,str(PROJECT))
from worldmodels.v2_external import load_external
OUT=PROJECT/'artifacts/reference-v2-independent-review'
DATA=PROJECT/'artifacts/runs/reference-v2-external-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
start=time.monotonic();x,v,meta,t=load_external(DATA)
manifest=json.loads((DATA/'artifact_hashes.json').read_text())
for name,h in manifest.items(): assert sha(DATA/name)==h,name
J=(11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32)
raws={};raw_count=0;actual_frames=0
for p in sorted((DATA/'recording_poses').glob('*.npz')):
    with np.load(p,allow_pickle=False) as z: raw={k:z[k].copy() for k in z.files}
    provenance=json.loads(str(raw['provenance_json']))
    assert sha(REPOSITORY/provenance['video_path'])==provenance['video_sha256']
    np.testing.assert_array_equal(raw['decoded_indexes'],raw['frame_numbers']-1)
    raws[p.stem]=raw;raw_count+=1;actual_frames+=len(raw['frame_numbers'])
maxerr=0.;rebuilt=0
for rec,rows in meta.groupby('recording_id'):
    raw=raws[rec];lookup={int(n):i for i,n in enumerate(raw['frame_numbers'])}
    for i,row in rows.iterrows():
        native=list(map(int,json.loads(row.frame_numbers)));ix=[lookup[n] for n in native]
        xy=np.asarray(raw['xy'][ix],dtype=float)[:,J];q=np.asarray(raw['confidence'][ix],dtype=float)[:,J]
        measured=np.isfinite(xy).all(-1)&np.isfinite(q)&(q>=.5)
        hips=measured[:,6]&measured[:,7]
        center=.5*(xy[:,6]+xy[:,7]);shoulders=.5*(xy[:,0]+xy[:,1])
        lengths=np.sqrt(np.sum((shoulders-center)**2,axis=1))
        scale_support=hips[:16]&measured[:16,0]&measured[:16,1]&(lengths[:16]>1e-6)
        scale=np.median(lengths[:16][scale_support]);assert np.isfinite(scale) and scale>0
        measured&=hips[:,None]
        values=np.where(measured[...,None],(xy-center[:,None])/scale,0).astype(np.float32)
        measured[16:]&=measured[:16].any(0)[None]
        values=np.where(measured[...,None],values,0.)
        maxerr=max(maxerr,float(np.max(np.abs(values-x[i]))))
        np.testing.assert_array_equal(values,x[i]);np.testing.assert_array_equal(measured,v[i])
        np.testing.assert_array_equal(t[i],raw['timestamp_seconds'][ix])
        assert hashlib.sha256(json.dumps([rec,native],separators=(',',':')).encode()).hexdigest()==row.window_id
        assert int(row.first_frame)==native[0] and int(row.last_frame)==native[-1]
        assert min(v[i,:16].mean(),v[i,16:].mean())>=.7
        for jj in ((8,10,12,14),(9,11,13,15)):
            support=measured[16:,jj]&measured[:16,jj].any(0)[None]
            assert support[:4].any() and support.any()
        rebuilt+=1
assert meta.source_id.eq(meta.participant_id).all() and meta.component_id.eq(meta.participant_id).all()
assert meta.groupby('participant_id').label.nunique().eq(1).all()
assert meta.groupby('recording_id').participant_id.nunique().eq(1).all()
qa=pd.read_csv(DATA/'visual_qa/selection_ledger.csv',keep_default_na=False)
assert len(qa)==282 and qa.participant_id.nunique()==94
pages={str(p.relative_to(PROJECT)):sha(p) for p in sorted((DATA/'visual_qa').glob('contact-*.png'))}
record={
 'identity':'reference-v2-independent-external-input-checks-v1','status':'passed_with_identity_and_acquisition_limits',
 'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),'script_sha256':sha(__file__),
 'external_identity_sha256':sha(DATA/'identity.json'),'manifest_files_verified':len(manifest),
 'raw_recording_caches':raw_count,'actual_video_hashes_verified':raw_count,'actual_requested_native_frames':actual_frames,
 'windows_rebuilt_with_independent_normalization':rebuilt,'maximum_coordinate_discrepancy':maxerr,
 'windows':len(meta),'recordings':meta.recording_id.nunique(),'coded_participants':meta.participant_id.nunique(),
 'by_label':meta.groupby('label').agg(windows=('window_id','size'),recordings=('recording_id','nunique'),participants=('participant_id','nunique')).reset_index().to_dict('records'),
 'visual_pages_independently_inspected':pages,'visual_selection_sha256':sha(DATA/'visual_qa/selection_ledger.csv'),
 'visual_participant_coverage':94,'visual_still_count':282,
 'visual_review':'All eight pages were viewed directly. Most midpoint stills show one apparent walker with broadly body-associated points. Boundaries frequently have no or cropped person; several foot or limb points are displaced from the visible outline. No anatomical accuracy or continuity guarantee follows from stills.',
 'identity_risk':{'participant_id':'PD_SV_001','panel':'094','retained_windows':int(meta.participant_id.eq('PD_SV_001').sum()),'observation':'A second person overlaps and follows the foreground walker in selected midpoint/end stills. First-pose identity through both recordings is unverified.','disposition':'Primary cohort must retain its declared identity; any participant-exclusion sensitivity must be frozen before external efficacy inspection and reported alongside the primary.'},
 'limits':['Reviewer knew the author-flagged panel before this inspection; this is independent inspection, not a fully blinded replicate.','Only the first selected recording and three frames per coded participant were visually inspected.','Different native FPS, framing, duration, support and clothing across labels remain potential recording confounds.','Coded participant IDs and diagnoses are provided by the dataset. Unknown overlap with GAVD is not excluded by hash disjointness.','No predictive-model score was read or computed in this check.'],
 'seconds':time.monotonic()-start}
(OUT/'external_input_checks.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:record[k] for k in ['status','windows','recordings','coded_participants','maximum_coordinate_discrepancy','seconds']},indent=2))
