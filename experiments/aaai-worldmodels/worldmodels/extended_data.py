"""Fresh 32-frame cohort with source-wise decoding and a declared person gate."""
from __future__ import annotations
import argparse,ast,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from .settings import configure
from .data import build_inventory,assign_splits,save_inventory,sha256_file,JOINT_IDS
from .pose import make_windows
from .workflow import json_write


def gate_person(xy,confidence,bbox,info,width,height):
    """Reject clear pelvis/box mismatch; unresolved geometry stays flagged."""
    conf=confidence.copy()
    try:
        b=ast.literal_eval(bbox) if isinstance(bbox,str) else bbox
        d=ast.literal_eval(info) if isinstance(info,str) else info
        aw,ah=float(d['width']),float(d['height'])
        if min(aw,ah,width,height)<=0 or abs((width/height)/(aw/ah)-1)>.05:
            return conf,'unresolved_aspect_ratio'
        left,top=float(b['left'])*width/aw,float(b['top'])*height/ah
        right,bottom=left+float(b['width'])*width/aw,top+float(b['height'])*height/ah
        if not np.isfinite([left,top,right,bottom]).all() or right<=left or bottom<=top:
            return conf,'unresolved_bbox'
    except (ValueError,TypeError,KeyError,SyntaxError):return conf,'unresolved_bbox'
    if min(conf[23],conf[24])<.5 or not np.isfinite(xy[[23,24]]).all():
        return np.zeros_like(conf),'missing_pelvis'
    pelvis=xy[[23,24]].mean(0)
    if not (left<=pelvis[0]<=right and top<=pelvis[1]<=bottom):
        return np.zeros_like(conf),'outside_annotated_person_box'
    return conf,'inside_box'


def prepare(output,frames=32,context_frames=16):
    import cv2
    from ambient.pose.keypoint_extractor import SequenceKeypointExtractor
    import ambient.pose.keypoint_extractor as ambient_extractor
    from .pose import _ADAPTER_SHA256
    project,repo=configure();output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    table,audit=build_inventory(repo);table=assign_splits(table,seed=27);save_inventory(table,audit,output)
    model=repo/'data/models/pose_landmarker_lite.task';model_hash=sha256_file(model)
    extractor_hash=sha256_file(ambient_extractor.__file__)
    started=time.monotonic();all_x=[];all_v=[];all_m=[];attrition=[];gates=[];reused=[]
    cache_root=output/'sequence_poses';cache_root.mkdir()
    for source,group in table[table.eligible].groupby('source_id',sort=True):
        path=Path(group.iloc[0].video_path)
        if not path.is_absolute():path=repo/path
        source_hash=sha256_file(path)
        cap=cv2.VideoCapture(str(path))
        fps=float(cap.get(cv2.CAP_PROP_FPS));width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not cap.isOpened() or not np.isfinite(fps) or fps<=0:raise ValueError(f'Cannot decode {source}')
        sequences={};point_cache={};requested=set()
        for row in group.sort_values('sequence_id').itertuples():
            rows=pd.read_csv(repo/row.csv_path,keep_default_na=False).sort_values('frame_num').iloc[::2].iloc[:frames]
            if len(rows)<frames:
                attrition.append(dict(sequence_id=row.sequence_id,source_id=source,label=row.label,split=row.split,windows=0,exclusion_reason='fewer_than_32_sampled_annotation_frames'))
                continue
            sequences[row.sequence_id]=(row,rows)
            requested.update(map(int,rows.frame_num))
            for cache in sorted((project/'artifacts/poses').glob(row.sequence_id+'-*.npz')):
                with np.load(cache,allow_pickle=False) as z:
                    identity=json.loads(str(z['provenance_json']))
                    expected={'csv_sha256':row.csv_sha256,'video_sha256':source_hash,'model_sha256':model_hash,'extractor_sha256':extractor_hash,'adapter_sha256':_ADAPTER_SHA256,'frame_step':2,'max_frames':16}
                    if any(identity.get(k)!=v for k,v in expected.items()):continue
                    for i,number in enumerate(z['frame_numbers']):
                        if int(number) not in point_cache:point_cache[int(number)]=(z['xy'][i].copy(),z['confidence'][i].copy(),float(z['timestamp_seconds'][i]))
                    reused.append({'sequence_id':row.sequence_id,'cache_sha256':sha256_file(cache),'pose_fingerprint':str(z['fingerprint'])})
                    break
        missing=requested-set(point_cache)
        extractor=SequenceKeypointExtractor()
        try:
            if missing:
                for index in range(max(missing)):
                    ok,frame=cap.read()
                    if not ok:break
                    number=index+1
                    if number not in missing:continue
                    pose=extractor.extract_from_image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB),str(model))
                    xy=np.full((33,2),np.nan,np.float32);conf=np.zeros(33,np.float32)
                    if pose is not None:
                        for kp in pose.keypoints:
                            if 0<=kp.id<33:xy[kp.id]=[kp.x,kp.y];conf[kp.id]=kp.confidence
                    point_cache[number]=(xy,conf,cap.get(cv2.CAP_PROP_POS_MSEC)/1000.)
        finally:cap.release();extractor.cleanup()
        for sequence,(row,rows) in sequences.items():
            raw=[];confidence=[];timestamps=[];reasons=[]
            for record in rows.itertuples():
                xy,conf,ts=point_cache.get(int(record.frame_num),(np.full((33,2),np.nan,np.float32),np.zeros(33,np.float32),np.nan))
                gated,reason=gate_person(xy,conf,record.bbox,record.vid_info,width,height)
                raw.append(xy);confidence.append(gated);timestamps.append(ts);reasons.append(reason)
                gates.append(dict(sequence_id=sequence,source_id=source,frame_number=int(record.frame_num),reason=reason))
            identity=dict(schema='extended32-person-gate-v1',csv_sha256=row.csv_sha256,video_sha256=source_hash,model_sha256=model_hash,
                extractor_sha256=extractor_hash,extended_adapter_sha256=sha256_file(__file__),frames=frames,context_frames=context_frames,
                original_adapter_sha256=_ADAPTER_SHA256,person_gate='pelvis_in_scaled_bbox_with_5pct_aspect_tolerance')
            import hashlib
            fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            pose=dict(xy=np.asarray(raw),confidence=np.asarray(confidence),frame_numbers=rows.frame_num.to_numpy(int),
                timestamp_seconds=np.asarray(timestamps),fps=np.array(fps),sequence_id=np.array(sequence),source_id=np.array(source),label=np.array(row.label),
                fingerprint=np.array(fingerprint),provenance_json=np.array(json.dumps(identity)),gate_reason=np.asarray(reasons))
            np.savez_compressed(cache_root/f'{sequence}.npz',**pose)
            x,valid,meta=make_windows(pose,frames=frames,stride=frames,context_frames=context_frames)
            attrition.append(dict(sequence_id=sequence,source_id=source,label=row.label,split=row.split,windows=len(x),exclusion_reason='' if len(x) else 'pose_quality_or_person_gate'))
            if len(x):
                meta['split']=row.split
                meta['prefix_valid_fraction']=valid[:,:context_frames].mean((1,2))
                meta['future_valid_fraction']=valid[:,context_frames:].mean((1,2))
                meta['prefix_mean_confidence']=pose['confidence'][:context_frames,JOINT_IDS].mean()
                meta['future_mean_confidence']=pose['confidence'][context_frames:,JOINT_IDS].mean()
                meta['horizon_seconds']=(frames-context_frames)/meta.sample_fps
                meta['unresolved_person_fraction']=sum(r.startswith('unresolved') for r in reasons)/len(reasons)
                meta['view']=row.views
                all_x.append(x);all_v.append(valid);all_m.append(meta)
        print(json.dumps({'source':source,'sequences':len(sequences),'retained_so_far':sum(map(len,all_x))}),flush=True)
    x=np.concatenate(all_x);valid=np.concatenate(all_v);meta=pd.concat(all_m,ignore_index=True)
    np.savez_compressed(output/'windows.npz',x=x,valid=valid);meta.to_csv(output/'windows.csv',index=False)
    pd.DataFrame(attrition).to_csv(output/'attrition.csv',index=False);pd.DataFrame(gates).to_csv(output/'person_gate_frames.csv',index=False)
    json_write(output/'reused_pose_provenance.json',reused)
    json_write(output/'preparation.json',dict(mode='all-sequences-extended32-person-gate',max_frames=frames,frames=frames,context_frames=context_frames,
        n_windows=len(x),n_sources=int(meta.source_id.nunique()),seconds=time.monotonic()-started,
        data_sha256=sha256_file(output/'windows.npz'),metadata_sha256=sha256_file(output/'windows.csv'),
        pose_model_sha256=model_hash,adapter_sha256=sha256_file(__file__),native_frame_step=2,
        limitation='Fixed frame horizon with variable seconds; bbox containment reduces obvious mismatches but does not verify identity or diagnosis.'))
    print('COMPLETE',len(x),meta.source_id.nunique(),time.monotonic()-started,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();prepare(a.output)
