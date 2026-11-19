"""Predict a GAVD annotated video sequence with a locally trained classifier bundle."""
from pathlib import Path
import sys,json,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import torch
from aaai_gait.settings import ROOT,Settings
from aaai_gait.data import build_manifest,find_video,extract_sequence,load_windows
from aaai_gait.model import load_checkpoint
from aaai_gait.training import embed_windows
from aaai_gait.features import build_features

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True,type=Path,help='Your own trusted real-GAVD run directory')
    p.add_argument('--sequence-id',required=True)
    a=p.parse_args();settings=Settings();torch.set_num_threads(2)
    # joblib/pickle files can execute code; only load bundles produced locally.
    bundle=joblib.load(a.run_dir/'classifiers.joblib')
    if bundle['synthetic']:raise ValueError('Synthetic models cannot be presented as GAVD classifiers')
    config=json.loads((a.run_dir/'provenance.json').read_text())['configuration']
    manifest=build_manifest(ROOT);row=manifest[manifest.sequence_id.eq(a.sequence_id)].copy()
    if len(row)!=1:raise ValueError('Expected one current selected GAVD sequence')
    source=row.iloc[0].source_id;video=find_video(ROOT,source)
    if video is None:raise FileNotFoundError('Run ingestion to retrieve this source under data/videos-gavd')
    extract_sequence(row.iloc[0].to_dict(),video,settings.pose_dir,frame_number_base=config['frame_number_base'])
    row['split']='test'
    x,meta=load_windows(row,settings.pose_dir,frames=config['frames'],stride=config['stride'],target_fps=config['target_fps'],require_coverage=False,expected_frame_number_base=config['frame_number_base'])
    if not len(x):raise ValueError(f"No usable window: {meta.attrs['exclusions']}")
    encoder,_=load_checkpoint(a.run_dir/'selected_encoder.pt',manifest_hash=bundle['manifest_hash'],split_hash=bundle['split_hash'])
    z=embed_windows(encoder,x);features,names=build_features(x,config['target_fps'])
    if names!=bundle['features']:raise ValueError('Feature schema changed')
    output=dict(sequence_id=a.sequence_id,source_id=source,n_windows=len(x),
                in_training=a.sequence_id in bundle['training_sequences'],interpretation='Dataset-condition scores, not a diagnosis')
    for name,values in [('rf',features),('probe',z)]:
        estimator=bundle[name];prob=estimator.predict_proba(values).mean(0)
        output[name]={str(c):float(v) for c,v in zip(estimator.classes_,prob)}
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
