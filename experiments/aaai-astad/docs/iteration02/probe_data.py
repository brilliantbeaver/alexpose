"""Read-only observation feasibility checks. Write only the adjacent JSON report."""
from pathlib import Path
import sys,json,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
B=Path(__file__).resolve().parents[2];ROOT=B.parents[1]
sys.path.insert(0,str(B))
from aaai_gait.data import load_pose,_resample,normalize_window,JOINTS,file_sha256
from aaai_gait.study_data import build_study_manifest
R=B/'artifacts/context_study/gavd/3044566091261819'
m=build_study_manifest(ROOT);e=pd.read_csv(R/'profile_eligibility.csv');pre=pd.read_csv(R/'window_index_before_profile_qc.csv')
legs=list(range(25,33));keep=e[[f'joint_{j}_distinct_patches' for j in legs]].ge(4).all(axis=1);ids=set(e.loc[keep,'source_id']);candidate=pre[pre.source_id.isin(ids)]
lower={'sources':len(ids),'normal_sources':int(e.loc[keep & e.binary_label.eq(0),'source_id'].nunique()),'sequences':int(candidate.sequence_id.nunique()),'windows':len(candidate),'by_label':candidate.groupby('label').agg(sources=('source_id','nunique'),windows=('window_id','size')).reset_index().to_dict('records'),'prior_use':candidate.drop_duplicates('source_id').groupby(['binary_label','prior_use']).size().rename('sources').reset_index().to_dict('records')}
sizes=[32,48,64,96,128];stats={n:{'windows':0,'sequences':set(),'sources':set(),'normal_sources':set(),'normal_windows':0} for n in sizes};complete={'windows':0,'sequences':set(),'sources':set(),'normal_sources':set(),'normal_windows':0};posehash={}
for row in m.itertuples():
 path=B/'artifacts/poses'/f'{row.sequence_id}.npz'
 if not path.exists():continue
 posehash[row.sequence_id]=file_sha256(path);pose=load_pose(path);fps=float(pose['fps'])
 if fps<15 or not np.isfinite(fps):continue
 raw=pose['raw_xy'].copy();raw[(~np.isfinite(pose['confidence']))|(pose['confidence']<.5)]=np.nan
 x,t=_resample(raw[:,JOINTS],pose['frame_numbers'],fps,15,1)
 for n in sizes:
  for start in range(0,len(x)-n+1,16):
   w=normalize_window(x[start:start+n]);v=np.isfinite(w.reshape(n//4,4,16,2)).all(axis=(1,3));h=len(v)//2
   if v.mean()<.75 or min(v[:h].mean(),v[h:].mean())<.5:continue
   groups=[stats[n]]+([complete] if n==32 and v.all() else [])
   for q in groups:
    q['windows']+=1;q['sequences'].add(row.sequence_id);q['sources'].add(row.source_id)
    if row.label=='normal':q['normal_sources'].add(row.source_id);q['normal_windows']+=1
for q in list(stats.values())+[complete]:
 for k,v in q.copy().items():
  if isinstance(v,set):q[k]=len(v)
events=[]
for row in m.itertuples():
 t=pd.read_csv(row.csv_path,keep_default_na=False,usecols=['id','seq','gait_pat','gait_event']);t=t[t.gait_event.str.strip().ne('')]
 if len(t):events.append(t)
events=pd.concat(events)
sourcefiles=[R/'profile_eligibility.csv',R/'window_index_before_profile_qc.csv',B/'aaai_gait/data.py',B/'aaai_gait/study_data.py',Path(__file__)]
record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'Read-only availability probes; no new model scores','fresh_inventory':{'sequences':len(m),'rows':int(m.n_frames.sum()),'sources':int(m.source_id.nunique()),'normal_sources':int(m.loc[m.label.eq('normal'),'source_id'].nunique()),'manifest_sha256':m.attrs['manifest_sha256']},'lower_limb_candidate':lower,'rewindowing':[{ 'frames':n,'sample_span_seconds':(n-1)/15,**q} for n,q in stats.items()],'complete_case_32frames_16joints':complete,'event_types':events.groupby('gait_event').agg(rows=('id','size'),sources=('id','nunique'),sequences=('seq','nunique')).reset_index().to_dict('records'),'event_categories':events.groupby('gait_pat').agg(rows=('id','size'),sources=('id','nunique')).reset_index().to_dict('records'),'parameters':{'source_frame_base':1,'min_source_fps':15,'target_fps':15,'confidence_threshold':.5,'stride':16,'patch_frames':4,'min_valid_tokens':.75,'min_half_valid_tokens':.5,'normalization':'existing normalize_window, prefix-half torso scale','candidate_readout_joints':legs,'candidate_min_distinct_patches':4,'input_and_mask_joints':list(JOINTS)},'files':[{'path':str(p.relative_to(ROOT)),'sha256':file_sha256(p)} for p in sourcefiles],'pose_files':posehash,'pose_corpus_sha256':hashlib.sha256(json.dumps(posehash,sort_keys=True).encode()).hexdigest()}
assert stats[32]['windows']==len(pre)
out=Path(__file__).with_name('data_feasibility_probe.json');out.write_text(json.dumps(record,indent=2));print(json.dumps({'output':str(out.relative_to(ROOT)),'lower_limb_candidate':lower,'rewindowing':record['rewindowing'],'complete_case':complete},indent=2))
