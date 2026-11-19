"""Deterministic high/low-error video review, separate from model selection."""
from pathlib import Path
import argparse,sys,subprocess
import cv2,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'experiments/aaai'))
from aaai_gait.data import JOINTS,find_video,file_sha256
from aaai_gait.media import EDGES
from aaai_gait.study_runner import dump_json
from aaai_gait.iteration_scoring import FOCAL
parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);args=parser.parse_args();out=args.run
folder=out/'video_review';folder.mkdir(exist_ok=True)
s=pd.read_csv(out/'source_scores.csv');s=s[s.policy.eq('U')&s.bank.eq('U')].copy()
s['prediction_error']=s[['left_loss_both','right_loss_both']].mean(axis=1)
meta=pd.read_csv(out/'window_metadata.csv');rows=[]
for label in FOCAL:
    candidates=s[s.categories.map(lambda v:label in v.split('|'))].sort_values(['prediction_error','source_id'])
    examples=[('low',candidates.iloc[0]),('high',candidates.iloc[-1])]
    fig,axes=plt.subplots(2,3,figsize=(12,6.8));fig.suptitle(f'Folder label: {label}. Lowest and highest prediction error.',fontsize=15)
    for line,(level,record) in enumerate(examples):
        windows=meta[meta.source_id.eq(record.source_id)].sort_values(['sequence_id','start_seconds'])
        window=windows.iloc[len(windows)//2];posepath=ROOT/f'experiments/aaai/artifacts/poses/{window.sequence_id}.npz';pose=np.load(posepath)
        video=find_video(ROOT,record.source_id);cap=cv2.VideoCapture(str(video));fps=float(pose['fps'])
        frames=pose['frame_numbers'];choices=[0,len(frames)//2,len(frames)-1]
        for column,k in enumerate(choices):
            frame=int(frames[k]);cap.set(cv2.CAP_PROP_POS_FRAMES,frame-1);ok,bgr=cap.read()
            if not ok:raise IOError(f'Cannot decode review frame{frame}')
            ax=axes[line,column];ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));xy=pose['raw_xy'][k].copy();xy[pose['confidence'][k]<.5]=np.nan
            for a,b in EDGES:
                if np.isfinite(xy[[a,b]]).all():ax.plot(xy[[a,b],0],xy[[a,b],1],color='#24d8ca',lw=1.4)
            ax.scatter(xy[list(JOINTS),0],xy[list(JOINTS),1],s=6,color='#ffa621')
            ax.set_title(f'{level} error | {record.source_id}\nframe{frame}, {(frame-1)/fps:.2f}s',fontsize=10);ax.axis('off')
        cap.release();start=float(window.start_seconds);end=float(window.end_seconds)+1/15
        preview=ROOT/'data/videos-gavd'/record.source_id/'previews'/f'iteration02_{window.sequence_id}_{start:.3f}_{end:.3f}.mp4'
        preview.parent.mkdir(parents=True,exist_ok=True)
        if not preview.exists():subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-ss',str(start),'-i',str(video),'-t',str(end-start),'-an','-vf','scale=640:-2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(preview)],check=True)
        rows.append(dict(label=label,selection=level,source_id=record.source_id,sequence_id=window.sequence_id,
            source_prediction_mse=float(record.prediction_error),source_M=float(record.M),source_S=float(record.S),
            view=str(window.camera_views),preview_path=str(preview.relative_to(ROOT)),preview_sha256=file_sha256(preview),
            preview_start_seconds=start,preview_end_seconds=end,playback_speed=1.,pose_sha256=file_sha256(posepath),
            reviewed_source_frames=[int(frames[k]) for k in choices],camera_motion='unreviewed',person_selection='unreviewed',visible_movement_note=''))
    fig.tight_layout(rect=[0,0,1,.95]);name=label.replace(' ','_');fig.savefig(folder/f'{name}.png',dpi=130);fig.savefig(folder/f'{name}.svg');plt.close(fig)
pd.DataFrame(rows).to_csv(folder/'review_ledger.csv',index=False)
dump_json(folder/'selection_protocol.json',dict(rule='Lowest and highest U source-mean both-leg MSE in each focal folder label; deterministic source-ID ties',
    window='middle source window after sequence/time sorting',sequence_frames='first, middle, last annotations',
    scope='ten purposively selected cases, not a representative camera census',labels='folder labels, not independent diagnoses',
    video_redistribution='not authorized by this analysis'))
print(folder)
