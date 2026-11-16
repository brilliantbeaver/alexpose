"""Explicit real-data tutorial inputs and evidence, separate from study methods."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import numpy as np
from .data import LABELS, build_inventory, find_video, sha256_file
from .settings import PROJECT, REPOSITORY


def data_mode():
    mode=os.getenv('WM_DATA_MODE','synthetic').lower()
    if mode not in {'synthetic','real'}:
        raise ValueError('WM_DATA_MODE must be synthetic or real')
    return mode


def prepared_path():
    return Path(os.getenv('WM_PREPARED_DATA') or PROJECT/'artifacts/runs/pilot-input-clean-v1').resolve()


def comparison_path():
    return Path(os.getenv('WM_COMPARISON_RUN') or PROJECT/'artifacts/runs/pilot-comparison-clean-v1').resolve()


def real_inputs():
    from .workflow import load_prepared
    x,valid,metadata=load_prepared(prepared_path())
    if metadata.groupby('source_id').split.nunique().max()!=1:
        raise ValueError('Source leakage in prepared data')
    return x,valid,metadata


def lesson_data(n=16, frames=16, seed=27, conditions=False, split='train'):
    """Real mode returns only the declared partition, with source-balanced caps.

    In real mode n is an approximate maximum window count, with at least two
    sources per requested condition. No synthetic fallback is permitted.
    """
    if data_mode()=='synthetic':
        from .demo import make_demo
        return make_demo(n=n,frames=frames,seed=seed,conditions=conditions)
    x,valid,meta=real_inputs()
    if frames!=x.shape[1] or split not in {'train','validation','test'}:
        raise ValueError('Invalid real-data shape or requested partition')
    labels=LABELS if conditions else ('normal',)
    per_class=max(2,n//len(labels))
    indexes=[]
    for label in labels:
        selected=meta.loc[(meta.split==split)&(meta.label==label)].sort_values(['source_id','sequence_id','first_frame'])
        chosen=selected.groupby('source_id',sort=True).head(1).head(per_class)
        if len(chosen)<2:
            raise ValueError(f'Need two real {split} sources for {label}; no synthetic replacement')
        indexes.extend(chosen.index.tolist())
    selected_meta=meta.loc[indexes].reset_index(drop=True)
    return dict(x=x[indexes],valid=valid[indexes],labels=selected_meta.label.to_numpy(),
                source_ids=selected_meta.source_id.to_numpy(),metadata=selected_meta,
                evidence_type='real GAVD prepared poses',partition=split,
                provenance=dict(prepared_path=str(prepared_path()),
                    windows_sha256=sha256_file(prepared_path()/'windows.npz'),
                    metadata_sha256=sha256_file(prepared_path()/'windows.csv'),
                    selected_window_indexes=indexes,selection='one accepted window per source, deterministic source order'))


def real_example():
    _,_,meta=real_inputs()
    window=meta.loc[(meta.split=='train')&(meta.label=='normal')].sort_values(['source_id','sequence_id']).iloc[0]
    inventory,_=build_inventory(REPOSITORY)
    row=inventory.loc[inventory.sequence_id==window.sequence_id].iloc[0].copy()
    row['first_frame']=int(window.first_frame)
    row['last_frame']=int(window.last_frame)
    row['sample_fps']=float(window.sample_fps)
    return row


def bounded_preview(row=None):
    """Encode only the audited teaching window, under the required media root."""
    import imageio_ffmpeg
    row=real_example() if row is None else row
    source=find_video(REPOSITORY,row.source_id,row.label)
    if source is None:
        raise FileNotFoundError(f'Missing actual source {row.source_id}')
    identity=dict(source_sha256=sha256_file(source),first_frame=int(row.first_frame),last_frame=int(row.last_frame),
                  operation='inclusive one-based frame trim, no audio, video only')
    key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    directory=REPOSITORY/'data/videos-gavd/previews';directory.mkdir(parents=True,exist_ok=True)
    output=directory/f'worldmodels-{row.source_id}-{key}.mp4';sidecar=output.with_suffix('.json')
    if output.exists() and sidecar.exists():
        previous=json.loads(sidecar.read_text())
        if previous.get('output_sha256')==sha256_file(output) and previous.get('identity')==identity:
            return output
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    filter_string=f'trim=start_frame={identity["first_frame"]-1}:end_frame={identity["last_frame"]},setpts=PTS-STARTPTS'
    command=[ffmpeg,'-hide_banner','-loglevel','error','-y','-i',str(source),'-vf',filter_string,
             '-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(command,check=True,capture_output=True,timeout=90)
    sidecar.write_text(json.dumps(dict(identity=identity,output_sha256=sha256_file(output)),indent=2)+'\n')
    return output


def lesson_directory(name):
    run_id=os.getenv('WM_NOTEBOOK_RUN_ID') or 'interactive-real-v1'
    if Path(run_id).name!=run_id or Path(name).name!=name:
        raise ValueError('Use plain run and lesson names')
    directory=PROJECT/'artifacts/runs/notebook-real'/run_id/name
    directory.mkdir(parents=True,exist_ok=True)
    return directory


def _json(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,Path): return str(value)
    if hasattr(value,'to_dict'): return value.to_dict('records')
    raise TypeError(type(value).__name__)


def persist_lesson(name, namespace):
    """Save bounded named outputs only; never serialize arbitrary notebook globals."""
    if data_mode()!='real':
        return {'mode':'synthetic','persisted_real_evidence':False}
    from .training import save_checkpoint
    directory=lesson_directory(name)
    evidence=dict(mode='real',lesson=name,created_at=datetime.now(timezone.utc).isoformat(),
                  prepared_data_sha256=sha256_file(prepared_path()/'windows.npz'),
                  prepared_metadata_sha256=sha256_file(prepared_path()/'windows.csv'),
                  helper_sha256=sha256_file(__file__),artifacts={})
    fixture=namespace.get('fixture')
    if fixture is not None:
        if fixture.get('evidence_type')!='real GAVD prepared poses':
            raise ValueError('Synthetic fixture entered a real-data notebook')
        evidence['selection']=fixture['provenance']
        fixture['metadata'].to_csv(directory/'selected_windows.csv',index=False)
    for key in ('model','reference','adapted','mixed','normal','forecast','stochastic','positional'):
        candidate=namespace.get(key)
        if candidate is not None and hasattr(candidate,'run_metadata') and candidate.run_metadata:
            path=directory/f'{key}.pt'
            save_checkpoint(candidate,path,overwrite=False)
            evidence['artifacts'][path.name]=sha256_file(path)
    for key in ('history','stages','report','results','calibration_example','deterministic_history','positional_history','stochastic_history','positional_comparison','binary_report','research_report'):
        value=namespace.get(key)
        if value is not None:
            path=directory/f'{key}.json'
            path.write_text(json.dumps(value,indent=2,default=_json,allow_nan=False)+'\n')
            evidence['artifacts'][path.name]=sha256_file(path)
    arrays={}
    for path in sorted((directory/'poses').glob('*.npz')):
        evidence['artifacts'][str(path.relative_to(directory))]=sha256_file(path)
    for key in ('mask','future'):
        value=namespace.get(key)
        if hasattr(value,'detach'):
            arrays[key]=value.detach().cpu().numpy()
    completion=namespace.get('actual_completion')
    if isinstance(completion,dict):
        arrays.update({'actual_completion__'+k:v.detach().cpu().numpy() for k,v in completion.items() if hasattr(v,'detach')})
    for key in ('embeddings','feature_matrix','persistence_error','joint_residual'):
        value=namespace.get(key)
        if isinstance(value,np.ndarray): arrays[key]=value
    for key in ('scores','sampled','representations','positional_scores'):
        value=namespace.get(key)
        if isinstance(value,dict):
            arrays.update({key+'__'+k:v for k,v in value.items() if isinstance(v,np.ndarray)})
    if arrays:
        path=directory/'numerical_outputs.npz';np.savez_compressed(path,**arrays)
        evidence['artifacts'][path.name]=sha256_file(path)
    import matplotlib.pyplot as plt
    last_figure=namespace.get('fig')
    if last_figure is not None and hasattr(last_figure,'savefig'):
        path=directory/'last_figure.svg';last_figure.savefig(path,bbox_inches='tight')
        evidence['artifacts'][path.name]=sha256_file(path)
    for i in plt.get_fignums():
        path=directory/f'figure-{i}.svg';plt.figure(i).savefig(path,bbox_inches='tight')
        evidence['artifacts'][path.name]=sha256_file(path)
    evidence_path=directory/'evidence.json'
    evidence_path.write_text(json.dumps(evidence,indent=2,default=_json,allow_nan=False)+'\n')
    return evidence
