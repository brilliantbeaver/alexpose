"""True binary inference from sealed clean RF and mixed JEPA classifiers."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import joblib
import numpy as np
from .data import sha256_file
from .settings import PROJECT,configure
from .inference import _check_run,extract_video_range,FRAMES,STRIDE,CONTEXT,CONFIDENCE,MIN_VALID
from .pose import make_windows
from .binary_clean import matrix,PRIMARY_JEPA,CLASSES,IDENTITY
from .evaluation import align_probabilities

def verify_development(development):
    """Verify lineage before loading trusted local joblib artifacts."""
    development=Path(development);seal=json.loads((development/'seal.json').read_text())
    if sha256_file(development/'selection.json')!=seal['selection_sha256']: raise ValueError('Changed binary selection')
    if sha256_file(development/'refit_lineage.json')!=seal['lineage_sha256']: raise ValueError('Changed refit lineage')
    spec=json.loads((development/'selection.json').read_text())
    if spec['identity']!=IDENTITY or spec['primary_jepa']!=PRIMARY_JEPA: raise ValueError('Unsupported binary identity or primary')
    if sha256_file(Path(__file__).with_name('binary_clean.py'))!=spec['code_sha256']: raise ValueError('Changed binary method')
    if sha256_file(PROJECT/'docs/revision_protocol.md')!=spec['protocol_sha256']: raise ValueError('Changed binary protocol')
    for name,digest in spec['dependency_sha256'].items():
        if sha256_file(Path(__file__).with_name(name))!=digest: raise ValueError(f'Changed dependency: {name}')
    for name,version in spec['versions'].items():
        if importlib.metadata.version(name)!=version: raise ValueError(f'Changed runtime: {name}')
    prepared,baseline=Path(spec['prepared']),Path(spec['baseline'])
    for filename,key in [('windows.npz','data_sha256'),('windows.csv','metadata_sha256'),('preparation.json','preparation_sha256')]:
        if sha256_file(prepared/filename)!=spec[key]: raise ValueError(f'Changed prepared identity: {filename}')
    for name,digest in spec['baseline_artifact_sha256'].items():
        if sha256_file(baseline/name)!=digest: raise ValueError(f'Changed baseline artifact: {name}')
    for name in ('rf',PRIMARY_JEPA):
        if sha256_file(development/f'{name}.joblib')!=spec['choices'][name]['classifier_sha256']:
            raise ValueError(f'Changed binary classifier: {name}')
    return spec,_check_run(baseline)

def classify_pose(pose,development,*,max_windows=None):
    """Classify one unlabeled sequence without refitting or selecting a model."""
    development=Path(development);spec,baseline_record=verify_development(development)
    if pose.get('provenance',{}).get('pose_model_sha256')!=baseline_record.get('pose_model_sha256'):
        raise ValueError('Input pose model differs from training')
    x,valid,meta=make_windows(pose,FRAMES,STRIDE,CONTEXT,CONFIDENCE,MIN_VALID)
    if max_windows is not None:
        if max_windows<1: raise ValueError('max_windows must be positive')
        x,valid,meta=x[:max_windows],valid[:max_windows],meta.iloc[:max_windows]
    result=dict(schema=1,identity='sealed-clean-binary-video-inference-v1',training_identity=spec['identity'],
        primary_jepa=PRIMARY_JEPA,n_windows=len(x),input=pose.get('provenance',{}),person_review_status='required',
        decision_threshold=.5,interpretation='Research prediction of normal versus pooled GAVD abnormal labels, not a clinical diagnosis.',
        aggregation='Arithmetic mean of window probabilities within this sequence',
        selection_sha256=sha256_file(development/'selection.json'),wrapper_sha256=sha256_file(__file__),
        quality=dict(confidence_threshold=CONFIDENCE,minimum_valid_fraction=MIN_VALID,frames=FRAMES,context_frames=CONTEXT),classifiers={})
    if not len(x): return dict(result,status='quality_failure',reason='No quality-eligible window')
    for name in ('rf',PRIMARY_JEPA):
        features=matrix(name,x,valid,meta,spec['baseline'])
        estimator=joblib.load(development/f'{name}.joblib')
        probabilities=align_probabilities(estimator,features,classes=CLASSES).mean(0)
        abnormal=float(probabilities[CLASSES.index('abnormal')])
        result['classifiers'][name]=dict(probabilities=dict(zip(CLASSES,map(float,probabilities))),
            predicted_annotation='abnormal' if abnormal>=.5 else 'normal',threshold=.5,
            classifier_sha256=spec['choices'][name]['classifier_sha256'])
    result['quality'].update(mean_valid_fraction=float(meta.valid_fraction.mean()),sample_fps=sorted(map(float,meta.sample_fps.unique())),
        first_retained_frame=int(meta.first_frame.min()),last_retained_frame=int(meta.last_frame.max()))
    result['status']='predictions_require_person_review';return result

def classify_video(video,development,pose_model,*,start_frame=1,end_frame,max_windows=None):
    _,baseline_record=verify_development(development)
    if sha256_file(pose_model)!=baseline_record.get('pose_model_sha256'): raise ValueError('Supplied pose weights differ from training')
    pose=extract_video_range(video,pose_model,start_frame=start_frame,end_frame=end_frame,max_windows=max_windows)
    return classify_pose(pose,development,max_windows=max_windows)

def main(argv=None):
    configure();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--development',type=Path,default=PROJECT/'artifacts/runs/binary-development-clean-v1')
    parser.add_argument('--pose-model',type=Path,required=True)
    parser.add_argument('--start-frame',type=int,default=1);parser.add_argument('--end-frame',type=int,required=True)
    parser.add_argument('--max-windows',type=int);parser.add_argument('--output',type=Path)
    args=parser.parse_args(argv)
    try:
        result=classify_video(args.video,args.development,args.pose_model,start_frame=args.start_frame,end_frame=args.end_frame,max_windows=args.max_windows)
        status=2 if result['status']=='quality_failure' else 0
    except (ValueError,FileNotFoundError,RuntimeError,ImportError) as exc:
        result=dict(status='input_or_compatibility_failure',reason=str(exc),classifiers={});status=2
    encoded=json.dumps(result,indent=2,allow_nan=False)
    if args.output: args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(encoded+'\n')
    print(encoded);return status

if __name__=='__main__': sys.exit(main())
