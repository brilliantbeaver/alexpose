"""CLI for current annotations, media ingestion, audit and repeated experiments."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import os
from dataclasses import replace
import json
import importlib.metadata as im
import pandas as pd
import torch
from aaai_gait.settings import Settings, ROOT
from aaai_gait.data import build_manifest, grouped_split, download_sources, find_video, extract_sequence, load_windows, assert_group_disjoint
from aaai_gait.workflow import prepare_corpus,capstone,normal_anchor,anomaly_evaluation,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['manifest','ingest','audit','train','smoke'])
    parser.add_argument('--workers',type=int,default=int(os.getenv('AAAI_EXTRACT_WORKERS','4')),help='Independent MediaPipe extraction processes')
    parser.add_argument('--limit',type=int,default=None,help='Limit unique source downloads and sequence extractions for a pilot; never a paper result')
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    args=parser.parse_args();settings=Settings();torch.set_num_threads(2)
    manifest=grouped_split(build_manifest(ROOT),seed=settings.seed)
    settings.artifact_dir.mkdir(parents=True,exist_ok=True)
    if args.command=='manifest':
        manifest.to_csv(settings.artifact_dir/'fresh_manifest_and_split.csv',index=False)
        print(manifest.groupby('label').agg(sequences=('sequence_id','size'),rows=('n_frames','sum'),sources=('source_id','nunique')).to_string());return
    if args.command=='ingest':
        report=download_sources(manifest,ROOT,limit=args.limit)
        rows=manifest if args.limit is None else manifest[manifest.source_id.isin(report.source_id)].head(args.limit)
        from aaai_gait.batch import batch_extract
        batch_extract(rows,ROOT,settings.pose_dir,settings.frame_number_base,workers=args.workers)
        return
    if args.command=='audit':
        x,meta=load_windows(manifest,settings.pose_dir,frames=settings.frames,stride=settings.stride,target_fps=settings.target_fps,require_coverage=False,expected_frame_number_base=settings.frame_number_base)
        pd.DataFrame(meta.attrs['exclusions']).to_csv(settings.artifact_dir/'pose_exclusions.csv',index=False)
        meta.to_csv(settings.artifact_dir/'retained_window_index.csv',index=False)
        print(meta.groupby(['split','label']).agg(windows=('sequence_id','size'),sequences=('sequence_id','nunique'),sources=('source_id','nunique')).to_string())
        assert_group_disjoint(meta);print('All five labels remain in all partitions.');return
    if args.command=='train' and settings.demo:
        parser.error('Set AAAI_DEMO=0 in the root .env for train, or use smoke for a synthetic check.')
    records=[]
    for seed in args.seeds:
        run_settings=replace(settings,seed=seed,demo=args.command=='smoke',epochs=1 if args.command=='smoke' else settings.epochs)
        x,meta,provenance=prepare_corpus(run_settings)
        result=capstone(x,meta,provenance,run_settings,bootstrap_repeats=50 if run_settings.demo else 1000)
        anchor,_=normal_anchor(x,meta,provenance,run_settings)
        scores,anomaly=anomaly_evaluation(anchor,x,meta,batch_size=run_settings.batch_size)
        scores.to_csv(provenance['run_dir']/'anomaly_source_scores.csv',index=False)
        from aaai_gait.robustness import anomaly_bootstrap
        anomaly['source_bootstrap']=anomaly_bootstrap(scores,repeats=50 if run_settings.demo else 1000,seed=seed)
        write_json(provenance['run_dir']/'anomaly_metrics.json',anomaly)
        future,_=normal_anchor(x,meta,provenance,run_settings,mode='forecast')
        future_scores,future_metrics=anomaly_evaluation(future,x,meta,batch_size=run_settings.batch_size,mode='forecast')
        future_scores.to_csv(provenance['run_dir']/'forecast_anomaly_source_scores.csv',index=False)
        future_metrics['source_bootstrap']=anomaly_bootstrap(future_scores,repeats=50 if run_settings.demo else 1000,seed=seed)
        write_json(provenance['run_dir']/'forecast_anomaly_metrics.json',future_metrics)
        from aaai_gait.normal_distance import normal_distance
        distance_scores,distance_metrics=normal_distance(anchor,x,meta,batch_size=run_settings.batch_size)
        distance_scores.to_csv(provenance['run_dir']/'normal_distance_scores.csv',index=False)
        write_json(provenance['run_dir']/'normal_distance_metrics.json',distance_metrics)
        write_json(provenance['run_dir']/'software_versions.json',{name:im.version(name) for name in ['torch','numpy','pandas','scikit-learn','mediapipe','python-dotenv']})
        records.append(dict(seed=seed,synthetic=run_settings.demo,run_dir=str(provenance['run_dir']),selected_jepa=result['selected_jepa']))
        print(json.dumps(records[-1]),flush=True)
    write_json(settings.artifact_dir/('synthetic_run_index.json' if args.command=='smoke' else 'real_run_index.json'),records)
    write_json(settings.artifact_dir/'last_run_index.json',records)

if __name__=='__main__':main()
