"""Small local video previews and selected-joint motion views."""
import ast
import subprocess
from .settings import ROOT
from .data import find_video, JOINTS
EDGES=((11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(25,27),(24,26),(26,28),(27,29),(29,31),(27,31),(28,30),(30,32),(28,32))

def show_video_context(manifest, *, frame_number_base=1, seconds=5, label='normal'):
    """Embed an annotated local clip or a source link. Never download media."""
    from IPython.display import display, Video, Markdown
    import cv2
    import imageio_ffmpeg
    candidates=manifest[manifest.label.eq(label)]
    if 'split' in candidates: candidates=candidates[candidates.split.eq('train')]
    candidates=candidates.drop_duplicates('source_id')
    if candidates.empty:
        display(Markdown(f'No sequence is available for the selected label: **{label}**.'))
        return None
    selected=None
    for row in candidates.to_dict('records'):
        video=find_video(ROOT,row['source_id'])
        if video: selected=(row,video);break
    if selected is None:
        row=candidates.iloc[0]
        display(Markdown(f"Video context: [open the {label} source]({row.url}). Notebook 01 prepares local downloads."))
        return None
    row,video=selected
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);cap.release()
    if fps<=0: raise ValueError('Unknown FPS')
    start=max(0,(row['first_frame']-frame_number_base)/fps)
    duration=min(seconds,(row['last_frame']-row['first_frame']+1)/fps)
    preview=ROOT/'data/videos-gavd'/row['source_id']/'previews'/f"{row['sequence_id']}_base{frame_number_base}_{seconds}s.mp4"
    preview.parent.mkdir(parents=True,exist_ok=True)
    if not preview.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-ss',str(start),'-i',str(video),'-t',str(duration),'-an','-vf','scale=640:-2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(preview)],check=True,capture_output=True)
    display(Markdown(f"**Observed video, separate from synthetic checks:** dataset label **{label}**, sequence `{row['sequence_id']}`, source `{row['source_id']}`, {start:.2f} s at {fps:.3f} frames/s. [Source]({row['url']})."))
    display(Video(str(preview),embed=True,width=560))
    return row

def plot_motion(window,*,title='Skeleton window',stride=8):
    import matplotlib.pyplot as plt
    index={joint:i for i,joint in enumerate(JOINTS)};times=list(range(0,len(window),stride))
    fig,axes=plt.subplots(1,len(times),figsize=(3*len(times),3.5),squeeze=False)
    for ax,t in zip(axes.ravel(),times):
        xy=window[t]
        for a,b in EDGES:
            pair=xy[[index[a],index[b]]];ax.plot(pair[:,0],pair[:,1],color='#007f82',lw=2)
        ax.scatter(xy[:,0],xy[:,1],s=12,color='#315cb5');ax.set_aspect('equal');ax.set_xlim(-.8,.8);ax.set_ylim(1.5,-1.3);ax.set_title(f'Frame {t}');ax.axis('off')
    fig.suptitle(title);fig.tight_layout();return fig

def annotated_frame(row,*,frame_number_base=1):
    import cv2
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    video=find_video(ROOT,row['source_id'])
    if video is None: raise FileNotFoundError('Download or restore this source first')
    ann=pd.read_csv(row['csv_path'],keep_default_na=False).sort_values('frame_num').iloc[0]
    cap=cv2.VideoCapture(str(video));cap.set(cv2.CAP_PROP_POS_FRAMES,int(ann.frame_num)-frame_number_base)
    ok,bgr=cap.read();cap.release()
    if not ok: raise IOError('Cannot decode annotated frame')
    box,info=ast.literal_eval(ann.bbox),ast.literal_eval(ann.vid_info);sy,sx=bgr.shape[0]/info['height'],bgr.shape[1]/info['width']
    fig,ax=plt.subplots(figsize=(9,5));ax.imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    ax.add_patch(Rectangle((box['left']*sx,box['top']*sy),box['width']*sx,box['height']*sy,fill=False,edgecolor='#ff9a35',lw=3))
    ax.set_title(f"Source frame {ann.frame_num}, decoded index {int(ann.frame_num)-frame_number_base}");ax.axis('off');return fig

def show_window_context(window_row,manifest):
    """Identify the exact sequence behind a prediction, without substituting another video."""
    from IPython.display import display,Markdown,Video
    import imageio_ffmpeg
    row=dict(window_row)
    if str(row['source_id']).startswith('synthetic-'):
        display(Markdown(f"**Synthetic example:** sequence `{row['sequence_id']}`, source `{row['source_id']}`. There is no corresponding observed video."));return
    start=float(row.get('start_seconds',0));end=float(row.get('end_seconds',start+2))
    display(Markdown(f"**This output's actual source:** sequence `{row['sequence_id']}`, video `{row['source_id']}`, {start:.3f} to {end:.3f} seconds. [Open source at this time](https://www.youtube.com/watch?v={row['source_id']}&t={int(start)}s)."))
    video=find_video(ROOT,row['source_id'])
    if video is None:return
    path=ROOT/'data/videos-gavd'/row['source_id']/'previews'/f"{row['sequence_id']}_{start:.3f}_{end:.3f}.mp4"
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-ss',str(start),'-i',str(video),'-t',str(max(.1,end-start+1/float(row.get('fps',15)))),'-an','-vf','scale=640:-2','-c:v','libx264','-pix_fmt','yuv420p','-y',str(path)],check=True,capture_output=True)
    display(Video(str(path),embed=True,width=560))
