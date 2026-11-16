"""Render the frozen fifteen-frame qualitative audit selection."""
from pathlib import Path
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT=Path(__file__).resolve().parents[2]
REPO=PROJECT.parents[1]
ROOT=Path(__file__).resolve().parent
records=json.loads((ROOT/'pose_contact_sheet.json').read_text())['frames']
edges=[(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(27,29),(29,31),(24,26),(26,28),(28,30),(30,32)]
fig,axes=plt.subplots(5,3,figsize=(12,15))
for ax,record in zip(axes.flat,records):
    with np.load(PROJECT/record['source_pose_path'],allow_pickle=False) as z:
        identity=json.loads(str(z['provenance_json']))
        i=np.flatnonzero(z['frame_numbers']==record['frame_number']).item()
        xy=z['xy'][i].copy();confidence=z['confidence'][i].copy()
    video=REPO/identity['raw_source_provenance']['video_path']
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,record['frame_number']-1)
    ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError(f'Cannot decode audit frame: {record}')
    ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    for a,b in edges:
        if min(confidence[a],confidence[b])>=.5:
            ax.plot(xy[[a,b],0],xy[[a,b],1],color='#ff6c00',lw=1.8)
    ax.set_title(f'{record["label"]}: {record["source_id"]}\nframe {record["frame_number"]}; {record["gate_reason"]}',fontsize=9)
    ax.axis('off')
fig.suptitle('Full-duration sampling audit: first eligible sequence by ID for each label',fontsize=13)
fig.tight_layout();fig.savefig(ROOT/'pose_contact_sheet.png',dpi=120)
plt.close(fig)
