"""Render reproducible real-video person-identity review panels; inspect manually."""
from pathlib import Path
import argparse,json
import cv2,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from worldmodels.settings import configure
from worldmodels.data import JOINT_IDS,find_video,sha256_file
project,repo=configure()
p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/research_analysis/person_review_regenerated.png');a=p.parse_args()
output=project/a.output
if output.exists():raise FileExistsError(output)
sequences=['cljw6utb3002i3n6ll0pspyhl','cljw6vv80002m3n6l14b2y4jg','cljw6wq8o002q3n6lo4lmi8vd','cljw6xyo3002u3n6lyl23l27l','cljw7r04700753n6ljc4jrdyl','cljw7rexn00793n6l1q6fmh8e','cljw7tfqg007p3n6l0nwqb8w1']
fig,axes=plt.subplots(len(sequences),3,figsize=(12,18));records=[]
for row,seq in enumerate(sequences):
 posepath=project/'artifacts/runs/extended-input-v1/sequence_poses'/f'{seq}.npz'
 with np.load(posepath,allow_pickle=False) as z:pose={key:z[key] for key in z.files}
 source=str(pose['source_id']);video=find_video(repo,source,'stroke');cap=cv2.VideoCapture(str(video))
 indexes=[0,len(pose['frame_numbers'])//2,len(pose['frame_numbers'])-1]
 for column,i in enumerate(indexes):
  frame_number=int(pose['frame_numbers'][i]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame_number-1);ok,bgr=cap.read()
  if not ok:raise ValueError(f'Cannot decode {source}:{frame_number}')
  ax=axes[row,column];ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));ids=[j for j in JOINT_IDS if pose['confidence'][i,j]>=.5]
  ax.scatter(pose['xy'][i,ids,0],pose['xy'][i,ids,1],s=9,c='#ff9800');ax.set_title(f'{source} | frame {frame_number}',fontsize=9);ax.axis('off')
  records.append(dict(sequence_id=seq,source_id=source,frame_number=frame_number,pose_sha256=sha256_file(posepath),video_sha256=sha256_file(video)))
 cap.release()
fig.suptitle('Person continuity review: first, middle and last sampled frames',fontsize=14);fig.tight_layout(rect=(0,0,1,.98));output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(output,dpi=120);plt.close(fig)
output.with_suffix('.json').write_text(json.dumps(dict(code_sha256=sha256_file(__file__),image_sha256=sha256_file(output),records=records,note='Same source/frame selection as the original manual audit; rendering bytes can differ. Human review is required before an exclusion decision.'),indent=2)+'\n')
print(output)
