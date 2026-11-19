"""Export a frozen sample of opaque gait clips and an unfilled review form.

Labels and source identifiers are kept in a separate withheld lookup. Text
already visible in video pixels can reveal a condition, so this prepares a
blinding aid rather than certifying blinded or completed human annotation.
"""
import argparse,hashlib,json,re,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
from aaai_gait.settings import ROOT,EXPERIMENT,Settings
from aaai_gait.study_data import STUDY_LABELS,build_study_manifest,load_study_windows
from aaai_gait.data import find_video,file_sha256
from aaai_gait.study_runner import dump_json

REVIEW_FIELDS=('reviewer_id','walking_visible','view','person_stable','left_right_confidence',
               'motion_asymmetry','arm_excursion_reduction','sustained_knee_flexion',
               'trunk_motion','walking_aid','uncertain_or_unscorable','visible_condition_clue','notes')


def select_sample(manifest,meta,*,seed=42,per_category=1,max_clips=12,frame_number_base=1):
    """Select source units from available media and QC windows without scores."""
    if not 1<=per_category<=2 or not 1<=max_clips<=12:
        raise ValueError('Use one or two sources per category and at most twelve clips')
    windows=meta.sort_values(['source_id','sequence_id','start_seconds']).drop_duplicates(['source_id','label'])
    info=manifest.set_index('sequence_id');candidates=[];media_failures=[]
    for row in windows.to_dict('records'):
        video=find_video(ROOT,row['source_id'])
        if video is None:
            media_failures.append(dict(source_id=row['source_id'],reason='missing_or_undecodable_media'));continue
        cap=cv2.VideoCapture(str(video));fps=float(cap.get(cv2.CAP_PROP_FPS))
        cap.set(cv2.CAP_PROP_POS_MSEC,float(row['start_seconds'])*1000);ok,_=cap.read();cap.release()
        if not ok or not np.isfinite(fps) or fps<=0:
            media_failures.append(dict(source_id=row['source_id'],reason='qc_window_start_not_decodable'));continue
        ann=info.loc[row['sequence_id']]
        row.update(video_path=str(video),source_fps=fps,url=ann.url,
                   annotation_end_seconds=(int(ann.last_frame)-frame_number_base+1)/fps)
        candidates.append(row)
    if not candidates:raise ValueError('No decodable source with an eligible pose window')
    choices={}
    for label,group in pd.DataFrame(candidates).groupby('label',sort=True):
        key=int(hashlib.sha256(f'{seed}:{label}'.encode()).hexdigest()[:16],16)
        rows=group.sort_values('source_id').to_dict('records')
        choices[label]=[rows[i] for i in np.random.default_rng(key).permutation(len(rows))]
    selected=[];used=set()
    for repetition in range(per_category):
        for label,rows in choices.items():
            if len(selected)>=max_clips:break
            row=next((r for r in rows if r['source_id'] not in used),None)
            if row is not None:selected.append(row);used.add(row['source_id'])
    table=pd.DataFrame(selected).sample(frac=1,random_state=seed).reset_index(drop=True)
    table.insert(0,'annotation_id',[f'clip-{i+1:03d}' for i in range(len(table))])
    return table,pd.DataFrame(media_failures)


def export_review_clips(sample_name,*,seed=42,per_category=1,max_clips=12,seconds=4.):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',sample_name):
        raise ValueError('Sample name must be a short plain identifier')
    if not np.isfinite(seconds) or not .5<=seconds<=10:raise ValueError('Clip duration must be 0.5 to 10 seconds')
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    clips=ROOT/'data/videos-gavd/review_clips'/sample_name
    withheld=EXPERIMENT/'artifacts/review_keys'/sample_name
    if clips.exists() or withheld.exists():
        raise FileExistsError('This frozen sample name already exists; choose a new name to preserve it')
    settings=Settings(demo=False)
    manifest=build_study_manifest(ROOT,settings.pose_dir)
    _,meta,ledger=load_study_windows(manifest,settings.pose_dir,
        expected_frame_number_base=settings.frame_number_base)
    selected,media_failures=select_sample(manifest,meta,seed=seed,per_category=per_category,
        max_clips=max_clips,frame_number_base=settings.frame_number_base)
    clips.mkdir(parents=True);withheld.mkdir(parents=True)
    # Freeze selection before any encoding. A failed export is never replaced
    # by a different source selected after viewing its content or model score.
    selected.to_csv(withheld/'selected_sources.csv',index=False)
    media_failures.to_csv(withheld/'media_failures.csv',index=False)
    ledger.to_csv(withheld/'qc_ledger.csv',index=False)
    records=[]
    for row in selected.to_dict('records'):
        target=clips/f"{row['annotation_id']}.mp4"
        duration=min(seconds,float(row['annotation_end_seconds'])-float(row['start_seconds']))
        record={**row,'requested_seconds':seconds,'exported_seconds':duration,'status':'failed','error':''}
        try:
            if duration<.5:raise ValueError('Too little annotated media remains after the QC window start')
            temporary=target.with_name(target.stem+'.tmp.mp4')
            subprocess.run([ffmpeg,'-hide_banner','-loglevel','error',
                '-ss',str(row['start_seconds']),'-i',row['video_path'],'-t',str(duration),
                '-an','-map_metadata','-1','-map_chapters','-1','-vf','scale=640:-2','-c:v','libx264',
                '-pix_fmt','yuv420p','-movflags','+faststart','-y',str(temporary)],
                check=True,capture_output=True,timeout=60)
            cap=cv2.VideoCapture(str(temporary));ok,frame=cap.read()
            record['decoded_frame_count']=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));cap.release()
            if not ok or frame is None:raise ValueError('Export did not decode')
            temporary.replace(target);record['clip_sha256']=file_sha256(target);record['status']='exported'
        except Exception as exc:
            record['error']=str(exc)
            if target.with_name(target.stem+'.tmp.mp4').exists():target.with_name(target.stem+'.tmp.mp4').unlink()
        records.append(record)
        print(row['annotation_id'],record['status'],flush=True)
    lookup=pd.DataFrame(records);lookup.to_csv(withheld/'lookup.csv',index=False)
    good=lookup[lookup.status.eq('exported')]
    form=pd.DataFrame({'annotation_id':good.annotation_id,'video_reference':good.annotation_id+'.mp4'})
    for field in REVIEW_FIELDS:form[field]=''
    form.to_csv(clips/'review_form.csv',index=False)
    (clips/'REVIEWER_README.md').write_text('''# Movement review package

This package contains opaque clip filenames and a blank review form. No human
review or agreement assessment has been completed by this export command.

View the entire clip before describing visible movement. Record uncertainty,
poor visibility, an unstable tracked person, and any visible condition clue.
Do not infer a diagnosis, neurological lesion, medication state, or muscle
force. Two reviewers should independently complete separate copies of the form.

File metadata and audio were removed. Titles, captions, faces, aids, or other
clues embedded in the video may still reveal its origin or suggest a condition.
This package therefore assists blinding but cannot guarantee it. The source and
condition lookup must remain with the study coordinator, outside this package.
''')
    dump_json(withheld/'protocol.json',dict(sample_name=sample_name,seed=seed,per_category=per_category,
        max_clips=max_clips,requested_seconds=seconds,selected_sources=len(selected),exported_clips=len(good),
        category_counts=selected.groupby('label').source_id.nunique().to_dict(),
        unavailable_categories=sorted(set(STUDY_LABELS)-set(selected.label)),
        partial_category_availability=len(set(selected.label))<len(STUDY_LABELS),
        manifest_sha256=manifest.attrs['manifest_sha256'],pose_corpus_sha256=meta.attrs['corpus_sha256'],
        code_sha256=file_sha256(Path(__file__)),frame_number_base=settings.frame_number_base,
        selection='Fixed category-keyed seed on source units with decodable media and at least one QC-passing pose window; first eligible sequence/time window',
        model_scores_used=False,human_annotations_completed=False,
        blinding='Opaque filenames, audio and file metadata removed; visible video content may reveal condition clues',
        reviewer_package=str(clips),withheld_lookup=str(withheld/'lookup.csv')))
    for path in withheld.iterdir():path.chmod(0o600)
    withheld.chmod(0o700)
    return clips,withheld


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-name',required=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--per-category',type=int,choices=(1,2),default=1)
    parser.add_argument('--max-clips',type=int,default=12)
    parser.add_argument('--seconds',type=float,default=4.)
    args=parser.parse_args()
    clips,withheld=export_review_clips(args.sample_name,seed=args.seed,per_category=args.per_category,
        max_clips=args.max_clips,seconds=args.seconds)
    print('REVIEWER_PACKAGE',clips);print('WITHHELD_COORDINATOR_KEY',withheld)

if __name__=='__main__':main()
