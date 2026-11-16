"""Reproducible command-line preparation and matched-model comparisons."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .settings import configure
from .data import build_inventory, assign_splits, save_inventory, LABELS, sha256_file


def json_write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False, default=str)+'\n')


def prepare(output, full=False, max_frames=16):
    from .pose import prepare_windows
    project, repo = configure()
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'windows.npz').exists():
        raise FileExistsError('Use a new run directory; prepared cohorts are immutable')
    table,audit=build_inventory(repo)
    table=assign_splits(table,seed=int(os.getenv('WM_SEED','27')))
    save_inventory(table,audit,output)
    selected=table.loc[table.eligible].copy()
    if not full:
        selected=selected.sort_values('sequence_id').groupby('source_id',sort=True).head(1)
    selected.to_csv(output/'selected_sequences.csv',index=False)
    start=time.monotonic()
    model=os.getenv('WM_POSE_MODEL') or str(repo/'data/models/pose_landmarker_lite.task')
    x,valid,meta,attrition=prepare_windows(selected,repo,model,project/'artifacts/poses',frames=16,
        stride=8,context_frames=8,frame_step=2,max_frames=max_frames if max_frames else None)
    np.savez_compressed(output/'windows.npz',x=x,valid=valid)
    meta.to_csv(output/'windows.csv',index=False)
    attrition.to_csv(output/'attrition.csv',index=False)
    json_write(output/'preparation.json',dict(mode='all-sequences' if full else 'one-sequence-per-source-pilot',
        max_frames=max_frames,frame_step=2,frames=16,context_frames=8,seconds=time.monotonic()-start,
        n_windows=len(x),n_sources=int(meta.source_id.nunique()) if len(meta) else 0,
        data_sha256=sha256_file(output/'windows.npz'),metadata_sha256=sha256_file(output/'windows.csv'),
        note='Pilot timing and implementation evidence, not a full-cohort scientific result.'))
    print(json.dumps(dict(windows=len(x),sources=int(meta.source_id.nunique()) if len(meta) else 0,seconds=time.monotonic()-start)))
    return x,valid,meta


def load_prepared(directory):
    directory=Path(directory)
    identity=json.loads((directory/'preparation.json').read_text())
    for name,key in [('windows.npz','data_sha256'),('windows.csv','metadata_sha256')]:
        if sha256_file(directory/name)!=identity[key]: raise ValueError(f'Changed prepared artifact: {name}')
    with np.load(directory/'windows.npz',allow_pickle=False) as z:
        x,valid=z['x'],z['valid']
    meta=pd.read_csv(directory/'windows.csv')
    if len(x)!=len(meta) or meta.groupby('source_id').split.nunique().max()!=1:
        raise ValueError('Invalid prepared cohort alignment or partitions')
    return x,valid,meta


def main():
    configure()
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    prep=subs.add_parser('prepare');prep.add_argument('--output',required=True);prep.add_argument('--full',action='store_true');prep.add_argument('--max-frames',type=int,default=16)
    compare=subs.add_parser('compare');compare.add_argument('--input',required=True);compare.add_argument('--output',required=True);compare.add_argument('--epochs',type=int,default=int(os.getenv('WM_EPOCHS','5')));compare.add_argument('--seed',type=int,default=int(os.getenv('WM_SEED','27')))
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.output,args.full,args.max_frames)
    else:
        from .experiment import compare
        compare(args.input,args.output,args.epochs,args.seed)

if __name__=='__main__': main()
