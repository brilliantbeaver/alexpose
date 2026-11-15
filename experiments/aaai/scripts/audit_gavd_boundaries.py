"""Post-outcome image-boundary measurement audit. Never filters or fits data."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,json,time
import cv2
import numpy as np
import pandas as pd
BASE=Path(__file__).resolve().parents[1]
COHORT=BASE/'artifacts/iteration02/preparation/cohort_v2'
OUT=BASE/'artifacts/iteration02/d66e1b126c9d9a9adf1b/video_review'
JOINTS=np.arange(25,33)


def main():
    manifest=pd.read_csv(COHORT/'manifest.csv');meta=pd.read_csv(COHORT/'window_index.csv');media=pd.read_csv(COHORT/'media_audit.csv').set_index('source_id');registry=pd.read_csv(COHORT/'source_registry.csv').set_index('source_id')
    selected=manifest[manifest.sequence_id.isin(meta.sequence_id)];rows=[];sources=[];start=time.perf_counter()
    def one(source,group):
        path=media.loc[source,'video_path'];cap=cv2.VideoCapture(str(path));poses={};needed=[]
        for r in group.itertuples():
            with np.load(BASE/'artifacts/poses'/f'{r.sequence_id}.npz') as p:poses[r.sequence_id]={k:p[k].copy() for k in ['raw_xy','confidence','decoded','frame_numbers']}
            needed.extend(poses[r.sequence_id]['frame_numbers'].tolist())
        frames=np.unique(needed);dimensions={};previous=None;position_mismatches=0;decode_failures=0
        try:
            for frame in frames:
                index=int(frame)-1
                if previous is None or index!=previous+1:cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                ok,image=cap.read();previous=index if ok else None
                if not ok:decode_failures+=1;continue
                h,w=image.shape[:2];dimensions[int(frame)]=(w,h)
                position_mismatches+=int(abs(float(cap.get(cv2.CAP_PROP_POS_FRAMES))-(index+1))>.51)
        finally:cap.release()
        result=[];unique_all={};unique_retained={}
        for r in group.itertuples():
            p=poses[r.sequence_id];f=p['frame_numbers'].astype(int);xy=p['raw_xy'][:,JOINTS];confidence=p['confidence'][:,JOINTS]
            wh=np.array([dimensions.get(int(n),(np.nan,np.nan)) for n in f]);decoded=np.isfinite(wh).all(1)&p['decoded'].astype(bool)
            valid=(confidence>=.5)&np.isfinite(confidence)&np.isfinite(xy).all(2)&decoded[:,None]
            outside=valid&((xy[:,:,0]<0)|(xy[:,:,0]>=wh[:,0,None])|(xy[:,:,1]<0)|(xy[:,:,1]>=wh[:,1,None]))
            retained=np.zeros(len(f),bool);wm=meta[meta.sequence_id.eq(r.sequence_id)]
            for w in wm.itertuples():retained|=(f>=w.source_frame_first)&(f<=w.source_frame_last)
            rv=valid&retained[:,None];ro=outside&retained[:,None]
            for i,j in zip(*np.nonzero(valid)):
                key=(int(f[i]),int(JOINTS[j]));unique_all[key]=unique_all.get(key,False) or bool(outside[i,j])
                if retained[i]:unique_retained[key]=unique_retained.get(key,False) or bool(outside[i,j])
            rr=dict(sequence_id=r.sequence_id,source_id=source,label=r.label,binary_label=int(r.binary_label),annotated_frames=len(f),actual_frames_decoded=int(np.isfinite(wh).all(1).sum()),cache_and_current_decoded_frames=int(decoded.sum()),decoded_shapes='|'.join(f'{w}x{h}' for w,h in sorted(set(dimensions[int(n)] for n in f if int(n) in dimensions))),
                confident_joint_observations=int(valid.sum()),outside_joint_observations=int(outside.sum()),outside_fraction=float(outside.sum()/valid.sum()) if valid.any() else np.nan,
                frames_with_outside_joint=int(outside.any(1).sum()),retained_span_frames=int(retained.sum()),retained_span_confident_joint_observations=int(rv.sum()),retained_span_outside_joint_observations=int(ro.sum()),retained_span_outside_fraction=float(ro.sum()/rv.sum()) if rv.any() else np.nan,
                retained_span_frames_with_outside_joint=int(ro.any(1).sum()))
            for j in range(8):rr[f'joint_{25+j}_confident']=int(valid[:,j].sum());rr[f'joint_{25+j}_outside']=int(outside[:,j].sum())
            result.append(rr)
        summary=dict(source_id=source,binary_label=int(registry.loc[source,'binary_label']),categories=registry.loc[source,'categories'],retained_sequences=len(group),requested_unique_frames=len(frames),decoded_unique_frames=len(dimensions),decode_failures=decode_failures,decoder_position_mismatches=position_mismatches,
                     decoded_shapes='|'.join(f'{w}x{h}' for w,h in sorted(set(dimensions.values()))),
                     unique_confident_frame_joint_observations=len(unique_all),unique_outside_frame_joint_observations=sum(unique_all.values()),
                     retained_span_unique_confident_frame_joint_observations=len(unique_retained),retained_span_unique_outside_frame_joint_observations=sum(unique_retained.values()))
        return result,summary
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks=[pool.submit(one,source,group) for source,group in selected.groupby('source_id')]
        for k,future in enumerate(as_completed(tasks),1):
            r,s=future.result();rows.extend(r);sources.append(s)
            if k%20==0:print(f'Boundary audit {k}/{len(tasks)} sources, {time.perf_counter()-start:.1f}s',flush=True)
    table=pd.DataFrame(rows).sort_values(['source_id','sequence_id']);source=pd.DataFrame(sources).sort_values('source_id');OUT.mkdir(exist_ok=True);table.to_csv(OUT/'boundary_audit.csv',index=False);source.to_csv(OUT/'boundary_source_audit.csv',index=False)
    groups=[]
    for label,g in source.groupby('binary_label'):
        den=int(g.unique_confident_frame_joint_observations.sum());num=int(g.unique_outside_frame_joint_observations.sum());rden=int(g.retained_span_unique_confident_frame_joint_observations.sum());rnum=int(g.retained_span_unique_outside_frame_joint_observations.sum())
        groups.append(dict(binary_label=int(label),label='normal' if label==0 else 'all_non_normal',sources=len(g),sources_with_any_outside=int(g.unique_outside_frame_joint_observations.gt(0).sum()),sources_with_retained_span_outside=int(g.retained_span_unique_outside_frame_joint_observations.gt(0).sum()),unique_confident_frame_joint_observations=den,unique_outside_frame_joint_observations=num,outside_fraction=num/den if den else None,retained_span_unique_confident_frame_joint_observations=rden,retained_span_unique_outside_frame_joint_observations=rnum,retained_span_outside_fraction=rnum/rden if rden else None))
    record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),audit_type='post_outcome_measurement_limit_no_filtering_no_refitting',cohort_sha256=json.loads((COHORT/'provenance.json').read_text())['cohort_sha256'],joint_allowlist=JOINTS.tolist(),confidence_threshold=.5,boundary_rule='finite confidence>=0.5, cached/current decoded, x<0 or x>=decoded_width or y<0 or y>=decoded_height',coordinate_system='original raw_xy in decoded full-image pixels, before normalization/resampling',source_unit='recording ID, not verified participant',unique_observation_rule='One source/raw-frame/joint; any outside estimate among retained sequence annotations marks that physical observation outside',retained_span_rule='Raw source frames within the inclusive enclosing intervals of frozen retained32frame windows. This is raw interpolation support, not a count of scored targets.',dimensions='Every distinct annotated source frame from each retained sequence was requested and decoded; actual image.shape supplied dimensions.',sources=len(source),sequences=len(table),decoded_frames=int(source.decoded_unique_frames.sum()),decode_failures=int(source.decode_failures.sum()),decoder_position_mismatches=int(source.decoder_position_mismatches.sum()),groups=groups,sequence_observations=dict(confident=int(table.confident_joint_observations.sum()),outside=int(table.outside_joint_observations.sum()),retained_span_confident=int(table.retained_span_confident_joint_observations.sum()),retained_span_outside=int(table.retained_span_outside_joint_observations.sum())),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limitation='Detector confidence does not establish visibility, anatomical accuracy, crop correctness, or correct person selection. In-frame points can also be erroneous. Out-of-frame values may be inferred positions, not observed joints. These findings do not change study eligibility or fitted results.')
    for name in ['boundary_audit.csv','boundary_source_audit.csv']:record[name+'_sha256']=hashlib.sha256((OUT/name).read_bytes()).hexdigest()
    dump=OUT/'boundary_audit.json';dump.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)

if __name__=='__main__':main()
