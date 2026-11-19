"""Native-time Ambient pose preparation for frozen KOA-PD-NM transfer.

Original media stay in place. Recording caches are individually identity checked;
only an unfinished request with exactly the same identities can resume. The
source/component unit is the coded participant and sequence is the recording.
This module never fits or evaluates a gait predictor.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import JOINT_IDS,sha256_file
from .restart_data import sampled_spans,supported_window,window_identity
from .settings import PROJECT,REPOSITORY
from .v2_data import common_support

C,H,STRIDE,HZ=16,8,8,10.
CODE_SHA256=sha256_file(__file__)
DEFAULT=PROJECT/'artifacts/runs/reference-v2-external-v1'
INVENTORY=PROJECT/'docs/reference_v2/data_external_inventory.csv'


def write_json(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=str)+'\n');temp.replace(path)


def load_external(path=None,*,allow_pilot=False,verify=True):
    """Return x, valid, participant metadata, and actual times after identity QA."""
    output=Path(path or DEFAULT)
    prep=json.loads((output/'preparation.json').read_text())
    identity=json.loads((output/'identity.json').read_text())
    if prep.get('status')!='complete':
        raise ValueError('External preparation is incomplete')
    if not allow_pilot and prep.get('mode')!='all_available_recordings':
        raise ValueError('A bounded extraction pilot cannot be used as full external evaluation')
    if any(prep.get(k)!=v for k,v in [('frames',24),('context_frames',16),('sample_hz',10.)]):
        raise ValueError('External preparation shape/time contract differs')
    if verify:
        for name,expected in json.loads((output/'artifact_hashes.json').read_text()).items():
            if sha256_file(output/name)!=expected:
                raise ValueError(f'External artifact hash mismatch: {name}')
        for name,key in [('windows.npz','data_sha256'),('windows.csv','metadata_sha256'),('request.json','request_sha256'),('identity.json','identity_sha256')]:
            if sha256_file(output/name)!=prep.get(key):
                raise ValueError(f'External preparation hash mismatch: {name}')
        if identity.get('data_sha256')!=prep['data_sha256'] or identity.get('metadata_sha256')!=prep['metadata_sha256']:
            raise ValueError('External identity and preparation disagree')
    meta=pd.read_csv(output/'windows.csv',keep_default_na=False,dtype={'direction':str})
    with np.load(output/'windows.npz',allow_pickle=False) as z:
        x,valid=z['x'],z['valid']
    if x.shape!=(len(meta),24,16,2) or valid.shape!=x.shape[:-1] or not np.isfinite(x).all() or not common_support(valid).all():
        raise ValueError('External common-support tensor contract failed')
    if meta.window_id.duplicated().any() or not (meta.source_id.eq(meta.participant_id)&meta.source_id.eq(meta.component_id)).all():
        raise ValueError('External participant grouping or window identity failed')
    if meta.groupby('recording_id').participant_id.nunique().max()!=1 or meta.groupby('source_id').label.nunique().max()!=1:
        raise ValueError('External recording/participant identity conflict')
    times=np.asarray([json.loads(t) for t in meta.timestamp_seconds],float)
    if times.shape!=(len(meta),24) or not np.isfinite(times).all() or not (np.diff(times,axis=1)>0).all():
        raise ValueError('External measured timestamps invalid')
    return x,valid,meta,times


def load_recording_cache(path,expected):
    with np.load(path,allow_pickle=False) as z:
        fields={k:z[k].copy() for k in z.files}
    identity=json.loads(str(fields['provenance_json']))
    if identity!=expected:
        raise ValueError(f'External recording-cache identity mismatch: {path}')
    n=len(expected['requested_frame_numbers'])
    if fields['xy'].shape!=(n,33,2) or fields['confidence'].shape!=(n,33):
        raise ValueError('External raw pose shape mismatch')
    if not np.array_equal(fields['frame_numbers'],expected['requested_frame_numbers']):
        raise ValueError('External native-frame request mismatch')
    if not np.array_equal(fields['decoded_indexes'],fields['frame_numbers']-1):
        raise ValueError('External source frame-origin mismatch')
    return fields


def extract_recording(video,cache,expected,model):
    import cv2
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    if cache.exists():
        return load_recording_cache(cache,expected),0,True
    native=np.asarray(expected['requested_frame_numbers'],np.int64)
    xy=np.full((len(native),33,2),np.nan,np.float32)
    confidence=np.zeros((len(native),33),np.float32)
    times=np.full(len(native),np.nan,float)
    positions={int(n):i for i,n in enumerate(native)}
    cap=cv2.VideoCapture(str(video));estimator=SequenceKeypointExtractor();calls=0
    try:
        if not cap.isOpened():
            raise ValueError(f'External source no longer opens: {video}')
        for decoded_index in range(int(native[-1])):
            ok,bgr=cap.read()
            if not ok:
                break
            number=decoded_index+1
            if number not in positions:
                continue
            i=positions[number];times[i]=cap.get(cv2.CAP_PROP_POS_MSEC)/1000.
            pose=estimator.extract_from_image(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),str(model));calls+=1
            if pose is not None:
                for point in pose.keypoints:
                    if 0<=point.id<33:
                        xy[i,point.id]=[point.x,point.y];confidence[i,point.id]=point.confidence
    finally:
        cap.release();estimator.cleanup()
    fields=dict(xy=xy,confidence=confidence,frame_numbers=native,decoded_indexes=native-1,
        timestamp_seconds=times,nominal_seconds=(native-1)/expected['fps'],
        provenance_json=np.array(json.dumps(expected,sort_keys=True)))
    temp=cache.with_suffix('.tmp.npz');np.savez_compressed(temp,**fields);temp.replace(cache)
    return fields,calls,False


def windows_from_recording(raw,row):
    """Use fixed native windows and the exact GAVD quality/common-support gate."""
    values,masks,records=[],[],[]
    rejection=Counter();total=0
    for start in range(0,len(raw['frame_numbers'])-(C+H)+1,STRIDE):
        stop=start+C+H;total+=1
        numbers=raw['frame_numbers'][start:stop]
        times=raw['timestamp_seconds'][start:stop]
        if not np.isfinite(times).all() or not (np.diff(times)>0).all():
            rejection['invalid_actual_decoder_times']+=1;continue
        try:
            x,valid,reason=supported_window(raw['xy'][start:stop],raw['confidence'][start:stop],context=C,minimum=.7)
        except ValueError as exc:
            if 'torso scale' not in str(exc):
                raise
            rejection['missing_prefix_torso_scale']+=1;continue
        if reason:
            rejection[reason]+=1;continue
        if not common_support(valid[None])[0]:
            rejection['common_distal_target_support']+=1;continue
        # A recording ID, rather than participant ID, prevents the two native
        # videos' identical frame numbers from creating duplicate window IDs.
        record=dict(window_id=window_identity(row.video_id,numbers),source_id=row.participant_id,
            component_id=row.participant_id,participant_id=row.participant_id,
            sequence_id=row.video_id,recording_id=row.video_id,label=row.label,
            original_group=row.original_group,severity=row.severity,direction=str(row.direction),
            video_path=row.path,video_sha256=row.sha256,
            first_frame=int(numbers[0]),last_frame=int(numbers[-1]),
            frame_numbers=json.dumps(numbers.tolist()),timestamp_seconds=json.dumps(times.tolist()),
            nominal_seconds=json.dumps(raw['nominal_seconds'][start:stop].tolist()),
            future_horizon_seconds=json.dumps((times[C:]-times[C-1]).tolist()),
            final_horizon_seconds=float(times[-1]-times[C-1]),duration_seconds=float(times[-1]-times[0]),
            fps=float(row.fps),sample_fps=HZ,width=int(row.width),height=int(row.height),
            view='dataset_provided_sagittal_direction_'+str(row.direction),
            prefix_valid_fraction=float(valid[:C].mean()),future_valid_fraction=float(valid[C:].mean()),
            valid_fraction=float(valid.mean()),prefix_mean_confidence=float(np.mean(raw['confidence'][start:start+C])),
            future_mean_confidence=float(np.mean(raw['confidence'][start+C:stop])),
            person_selection='Ambient full-image first detection; no annotation bounding box')
        values.append(x);masks.append(valid);records.append(record)
    return values,masks,records,dict(candidate_windows=total,retained_windows=len(values),rejection_counts=dict(rejection))


def build_visual_qa(output,inventory):
    """Generate blinded stills for every decodable coded participant before scoring."""
    import cv2
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    qa=output/'visual_qa';qa.mkdir(exist_ok=True)
    # File choice is independent of pose quality, gait scores, or condition.
    representatives=inventory[inventory.first_and_last_decode].sort_values('video_id').drop_duplicates('participant_id').sort_values('participant_id')
    ledger=[];files=[]
    for page,start in enumerate(range(0,len(representatives),12),1):
        subset=representatives.iloc[start:start+12]
        fig,axes=plt.subplots(len(subset),3,figsize=(12,len(subset)*2.35),squeeze=False)
        for rindex,row in enumerate(subset.itertuples()):
            rawpath=output/'recording_poses'/f'{row.video_id}.npz'
            with np.load(rawpath,allow_pickle=False) as z:
                numbers=z['frame_numbers'];xy=z['xy'];confidence=z['confidence']
            cap=cv2.VideoCapture(str(REPOSITORY/row.path))
            try:
                for col,i in enumerate([0,len(numbers)//2,len(numbers)-1]):
                    number=int(numbers[i]);cap.set(cv2.CAP_PROP_POS_FRAMES,number-1)
                    ok,bgr=cap.read();ax=axes[rindex,col];ax.axis('off')
                    panel=f'{start+rindex+1:03d}.{col+1}'
                    if ok:
                        ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
                        good=np.isfinite(xy[i]).all(1)&(confidence[i]>=.5)
                        ax.scatter(xy[i,good,0],xy[i,good,1],s=4,c='#e66c30')
                    else:
                        ax.text(.5,.5,'Selected native frame did not decode',ha='center',va='center')
                    ax.set_title('Panel '+panel,fontsize=10)
                    ledger.append(dict(panel=panel,page=page,video_id=row.video_id,participant_id=row.participant_id,
                        label=row.label,frame_number=number,decoded=bool(ok),video_path=row.path))
            finally:
                cap.release()
        fig.suptitle('External observation QA: first recording per coded participant\nFirst, middle, and last sampled native frames; identifiers are in a separate ledger.',fontsize=12)
        fig.tight_layout(rect=[0,0,1,.975]);path=qa/f'contact-{page:02d}.png';fig.savefig(path,dpi=100);plt.close(fig);files.append(path.name)
    pd.DataFrame(ledger).to_csv(qa/'selection_ledger.csv',index=False)
    write_json(qa/'selection.json',dict(status='generated_before_model_scoring; visual_review_required',
        participants=len(representatives),frames=len(ledger),pages=files,
        selection='All coded participants with a metadata-decodable recording; first recording by video_id and first/middle/last sampled frame.',
        limitation='Still frames cannot verify person continuity, diagnosis, or every gait cycle; no GAVD person-box gate is available.'))


def prepare(output=None,*,max_recordings=None,video_ids=None,make_qa=True,max_seconds=3600):
    output=Path(output or DEFAULT)
    if (output/'preparation.json').exists():
        raise FileExistsError('Completed external preparation is immutable; use a new identity')
    started=time.monotonic()
    inventory=pd.read_csv(INVENTORY,keep_default_na=False,dtype={'direction':str})
    # A video can be present but unreadable. Keep its explicit attrition record.
    inventory=inventory.sort_values('video_id').reset_index(drop=True)
    if video_ids is not None:
        if set(video_ids)-set(inventory.video_id):
            raise ValueError('Unknown external recording requested')
        inventory=inventory[inventory.video_id.isin(video_ids)].copy()
    if max_recordings is not None:
        inventory=inventory.head(max_recordings).copy()
    from ambient.pose import keypoint_extractor,model_management,mediapipe_singleton
    model=REPOSITORY/'data/models/pose_landmarker_lite.task'
    gavd_request=PROJECT/'artifacts/runs/restart-full10hz-v1/request.json'
    expected_weight=json.loads(gavd_request.read_text())['pose_model_sha256']
    if sha256_file(model)!=expected_weight:
        raise ValueError('Actual external pose weights differ from frozen GAVD pose weights')
    request=dict(identity='reference-v2-external-v1',code_sha256=CODE_SHA256,inventory_sha256=sha256_file(INVENTORY),
        model_sha256=sha256_file(model),extractor_sha256=sha256_file(keypoint_extractor.__file__),
        frozen_gavd_request_sha256=sha256_file(gavd_request),
        support_sha256={str(Path(m.__file__).relative_to(REPOSITORY)):sha256_file(m.__file__) for m in [model_management,mediapipe_singleton]},
        normalization_sha256=sha256_file(Path(__file__).with_name('pose.py')),
        common_support_sha256=sha256_file(Path(__file__).with_name('v2_data.py')),
        supported_window_sha256=sha256_file(Path(__file__).with_name('restart_data.py')),
        recording_ids=inventory.video_id.tolist(),frames=24,context=16,stride=8,sample_hz=10.,confidence=.5,minimum_support=.7,
        resource_ceiling_seconds=max_seconds,
        participant_key='dataset group + severity + local subject number; normal group + number',
        person_selection='Ambient full-image first pose; no annotation box',
        coordinate_units='raw pixels; windows centered on per-frame pelvis and prefix-only torso scale',
        versions={n:importlib.metadata.version(n) for n in ['numpy','pandas','mediapipe','opencv-contrib-python']})
    output.mkdir(parents=True,exist_ok=True);(output/'recording_poses').mkdir(exist_ok=True)
    if (output/'request.json').exists() and json.loads((output/'request.json').read_text())!=request:
        raise ValueError('External resume identity changed')
    write_json(output/'request.json',request)
    inventory.to_csv(output/'recording_inventory.csv',index=False)
    prior_cache_hashes={}
    if (output/'progress.json').exists():
        prior_cache_hashes={r['recording_id']:r['cache_sha256'] for r in json.loads((output/'progress.json').read_text())['recordings']}
    values,masks,records=[],[],[]
    attrition=[];cache_log=[];calls_total=0
    for index,row in enumerate(inventory.itertuples()):
        if time.monotonic()-started>max_seconds:
            write_json(output/'resource_stop.json',dict(status='incomplete_resource_limit',seconds=time.monotonic()-started,
                completed_recordings=index,requested_recordings=len(inventory),resume='Identical request may reuse verified recording caches.'))
            raise TimeoutError('External extraction resource ceiling reached; partial caches retained')
        begin=time.monotonic();video=REPOSITORY/row.path
        if sha256_file(video)!=row.sha256:
            raise ValueError(f'External source-video hash changed: {row.video_id}')
        base=dict(recording_id=row.video_id,participant_id=row.participant_id,label=row.label,video_path=row.path)
        if not row.first_and_last_decode:
            attrition.append(dict(**base,status='metadata_decode_failure',candidate_windows=0,retained_windows=0,rejection_counts='{}'))
            continue
        numbers=sampled_spans(np.arange(1,int(row.nominal_frame_count)+1),float(row.fps))[0]
        expected=dict(schema='reference-v2-external-recording-v1',recording_id=row.video_id,participant_id=row.participant_id,
            video_path=row.path,video_sha256=row.sha256,requested_frame_numbers=numbers.tolist(),fps=float(row.fps),
            image_dimensions=[int(row.width),int(row.height)],frame_origin=1,decoded_index_origin=0,pose_format='BLAZEPOSE_33',
            code_sha256=CODE_SHA256,model_sha256=request['model_sha256'],extractor_sha256=request['extractor_sha256'],
            support_sha256=request['support_sha256'],sample_hz=HZ)
        cache=output/'recording_poses'/f'{row.video_id}.npz'
        if cache.exists() and row.video_id in prior_cache_hashes and sha256_file(cache)!=prior_cache_hashes[row.video_id]:
            raise ValueError(f'External resume payload hash changed: {row.video_id}')
        raw,calls,reused=extract_recording(video,cache,expected,model);calls_total+=calls
        xx,vv,mm,status=windows_from_recording(raw,row);values.extend(xx);masks.extend(vv);records.extend(mm)
        attrition.append(dict(**base,status='processed',**{k:v for k,v in status.items() if k!='rejection_counts'},
                              rejection_counts=json.dumps(status['rejection_counts'],sort_keys=True)))
        cache_log.append(dict(recording_id=row.video_id,participant_id=row.participant_id,
            cache_sha256=sha256_file(cache),actual_model_calls=calls,reused=reused,requested_frames=len(numbers),
            seconds=time.monotonic()-begin,retained_windows=status['retained_windows']))
        write_json(output/'progress.json',dict(completed_recordings=index+1,total_recordings=len(inventory),
            actual_model_calls=calls_total,recordings=cache_log,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(cache_log[-1]),flush=True)
    x=np.stack(values) if values else np.empty((0,24,16,2),np.float32)
    valid=np.stack(masks) if masks else np.empty((0,24,16),bool)
    meta=pd.DataFrame(records)
    if len(meta) and (meta.window_id.duplicated().any() or not common_support(valid).all()):
        raise ValueError('External duplicate windows or common-support failure')
    np.savez_compressed(output/'windows.npz',x=x,valid=valid)
    meta.to_csv(output/'windows.csv',index=False)
    pd.DataFrame(attrition).to_csv(output/'attrition.csv',index=False)
    if make_qa:
        build_visual_qa(output,inventory)
    full=max_recordings is None and video_ids is None
    identity=dict(identity='reference-v2-external-v1',run_identity=output.name,
        code_sha256=CODE_SHA256,request_sha256=sha256_file(output/'request.json'),
        data_sha256=sha256_file(output/'windows.npz'),metadata_sha256=sha256_file(output/'windows.csv'),
        recording_inventory_sha256=sha256_file(output/'recording_inventory.csv'),
        model_sha256=request['model_sha256'],frozen_gavd_request_sha256=request['frozen_gavd_request_sha256'],
        window_identity='SHA256(recording_id + native one-based frame tuple)',
        participant_identity='original group + severity + local number; NM + number',
        mode='all_available_recordings' if full else 'bounded_extraction_pilot')
    write_json(output/'identity.json',identity)
    report=dict(identity='reference-v2-external-v1',status='complete',mode='all_available_recordings' if full else 'bounded_extraction_pilot',
        recordings_cataloged=len(inventory),recordings_processed=len(cache_log),
        participants_cataloged=int(inventory.participant_id.nunique()),windows=len(meta),
        recordings_retained=int(meta.sequence_id.nunique()) if len(meta) else 0,
        participants_retained=int(meta.source_id.nunique()) if len(meta) else 0,
        actual_model_calls_this_invocation=calls_total,cache_reuses=sum(r['reused'] for r in cache_log),
        seconds=time.monotonic()-started,data_sha256=sha256_file(output/'windows.npz'),metadata_sha256=sha256_file(output/'windows.csv'),
        request_sha256=sha256_file(output/'request.json'),identity_sha256=sha256_file(output/'identity.json'),
        by_label=meta.groupby('label').agg(windows=('window_id','size'),recordings=('sequence_id','nunique'),participants=('source_id','nunique')).reset_index().to_dict('records') if len(meta) else [],
        source_unit='coded participant',sequence_unit='video recording',sample_hz=10.,frames=24,context_frames=16,
        annotation_box_gate='not available in external recordings',
        evaluation='No predictive-model scoring or target-domain fitting occurs in this preparation.',
        limitations=['Codes are dataset-provided participant identities; diagnoses are not independently reassessed.',
            'No bounding-box person gate; full-image first-pose detection can follow the wrong person.',
            'Metadata-decode failures and quality-dependent attrition change the observed participant population.',
            'Byte-hash disjointness from GAVD cannot detect a reencoded upload of the same footage or repeated people.'])
    write_json(output/'preparation.json',report)
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha256_file(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT)
    parser.add_argument('--max-recordings',type=int);parser.add_argument('--video',action='append');parser.add_argument('--no-qa',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=3600)
    args=parser.parse_args();prepare(args.output,max_recordings=args.max_recordings,video_ids=args.video,make_qa=not args.no_qa,max_seconds=args.max_seconds)
