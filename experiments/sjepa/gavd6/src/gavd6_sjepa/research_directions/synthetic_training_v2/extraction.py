"""Score-preserving adapter; historical MMPoseEstimator.predict stays untouched."""
from dataclasses import dataclass
import numpy as np
from .contracts import JOINTS

@dataclass
class TrackExtraction:
    xy: np.ndarray
    confidence: np.ndarray
    observed: np.ndarray
    timestamps: np.ndarray
    box_source: str
    status_counts: dict
    schema: tuple=JOINTS

    def inputs(self):return {k:getattr(self,k) for k in ('xy','confidence','observed','timestamps')}


def extract_tracks(estimator,images,boxes,timestamps,*,box_source,threshold=0.,batch_size=32):
    """No renderer labels or true-error arguments exist on this API.

    No-detection outputs retain a frame of missing joints. Missing scores are
    counted as unsupported and never replaced with guessed confidence.
    """
    import torch
    if not box_source or not 0<=threshold<=1 or batch_size<1:raise ValueError('Box provenance and valid source-set threshold required')
    images=list(images);boxes=np.asarray(boxes);times=np.asarray(timestamps,float)
    if boxes.shape!=(len(images),4) or times.shape!=(len(images),):raise ValueError('Frame/box/time alignment mismatch')
    if not np.isfinite(times).all() or np.any(np.diff(times)<=0):raise ValueError('Physical timestamps required')
    xy=np.full((len(images),12,2),np.nan,np.float32);scores=np.full((len(images),12),np.nan,np.float32)
    counts=dict(frames=len(images),missing_detections=0,unsupported_scores=0,nonfinite_joints=0)
    def array(v):return v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else np.asarray(v)
    estimator.model.eval()
    with torch.inference_mode():
        for start in range(0,len(images),batch_size):
            end=min(start+batch_size,len(images))
            samples=estimator.model.test_step(estimator._batch(images[start:end],boxes[start:end]))
            if len(samples)!=end-start:raise ValueError('Extractor dropped frame correspondence')
            for i,sample in enumerate(samples,start):
                inst=getattr(sample,'pred_instances',None)
                keypoints=None if inst is None else getattr(inst,'keypoints',None)
                if keypoints is None or len(keypoints)==0:counts['missing_detections']+=1;continue
                value=array(keypoints)
                if value.shape!=(1,17,2):raise ValueError('Expected one supplied-box COCO17 person')
                xy[i]=value[0,5:17]
                raw=getattr(inst,'keypoint_scores',None)
                if raw is None:counts['unsupported_scores']+=1;continue
                raw=array(raw)
                if raw.shape!=(1,17):raise ValueError('Unexpected score schema')
                scores[i]=raw[0,5:17]
    finite=np.isfinite(xy).all(-1);counts['nonfinite_joints']=int((~finite).sum())
    if np.any(np.isfinite(scores)&((scores<0)|(scores>1))):raise ValueError('Scores outside[0,1]; require source-only calibrated adapter')
    observed=finite & np.isfinite(scores) & (scores>=threshold)
    xy[~observed]=np.nan
    return TrackExtraction(xy,scores,observed,times,box_source,counts)


def perturb_boxes(boxes,*,shift_fraction=0.,scale=1.):
    boxes=np.asarray(boxes,float);size=boxes[...,2:]-boxes[...,:2]
    if np.any(size<=0) or scale<=0:raise ValueError('Positive box geometry required')
    center=(boxes[...,2:]+boxes[...,:2])/2+shift_fraction*size
    return np.concatenate((center-size*scale/2,center+size*scale/2),-1)
