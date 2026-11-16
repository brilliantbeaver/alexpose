"""One coordinate-weight sensitivity; the original primary stays unchanged."""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from .data import sha256_file
from .model import ModelConfig,SkeletonJEPA
from .revision import (RevisionProtocol,clone_forecaster,train_forecaster,_load_inputs,_filtered_subset,
    _score_arm,_save_model,load_revision_checkpoint,_write,_code_identity,paired_gain_summary)
IDENTITY='coordinate-weight-ten-secondary-audit-v1'

def _audit_code():
    return dict(revision_dependencies=_code_identity(),objective_audit_sha256=sha256_file(__file__))

def _reference(directory):
    directory=Path(directory);record=json.loads((directory/'development.json').read_text())
    if record['code_sha256']!=_code_identity(): raise ValueError('Primary implementation changed')
    for package,version in record['versions'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('Primary runtime changed')
    for name,digest in record['artifact_sha256'].items():
        if sha256_file(directory/name)!=digest: raise ValueError(f'Primary artifact changed: {name}')
    p=RevisionProtocol(**record['protocol'])
    if p.coordinate_weight!=1: raise ValueError('Audit compares primary weight1 with weight10 only')
    input_dir=Path(record['inputs']['input_dir'])
    for name,key in [('windows.npz','windows_sha256'),('windows.csv','metadata_sha256'),('preparation.json','preparation_sha256')]:
        if sha256_file(input_dir/name)!=record['inputs'][key]: raise ValueError('Primary prepared data changed')
    return record,p,input_dir

def declared_change(protocol):
    revised=replace(protocol,coordinate_weight=10.)
    differences={key:[asdict(protocol)[key],value] for key,value in asdict(revised).items() if value!=asdict(protocol)[key]}
    if differences!={'coordinate_weight':[1.,10.]}: raise ValueError('More than coordinate weight changed')
    return revised

def develop(reference_dir,output_dir):
    reference_dir,output=Path(reference_dir),Path(output_dir)
    if output.exists(): raise FileExistsError('Use a new audit directory')
    reference,p,input_dir=_reference(reference_dir);q=declared_change(p)
    x,v,meta=_load_inputs(input_dir,p)
    tx,tv,tm,train_drops=_filtered_subset(x,v,meta,p,((meta.split=='train')&(meta.label=='normal')).to_numpy())
    vx,vv,vm,val_drops=_filtered_subset(x,v,meta,p,(meta.split=='validation').to_numpy())
    output.mkdir(parents=True);torch.set_num_threads(2);started=time.monotonic()
    gradient_path=Path(__file__).parents[1]/'docs/training_gradient_audit.json'
    declaration=dict(identity=IDENTITY,primary_reference=str(reference_dir.resolve()),
        primary_development_sha256=sha256_file(reference_dir/'development.json'),changed_field={'coordinate_weight':[1.,10.]},
        proposal='Training-only raw gradients motivate one sensitivity, not a causal optimizer claim.',
        no_further_search=True,primary_unchanged=True,all_seeds_retained=True,protocol=asdict(q),
        gradient_evidence_sha256=sha256_file(gradient_path),code_sha256=_audit_code(),versions=reference['versions'],
        final_step_only=True,common_initialization='Original uniform_pretrain.pt then same zero-head residual_det clone',
        common_batches='Original source_schedule.json; fresh optimizer and identical seed, no continued fitting')
    _write(output/'declaration.json',declaration)
    _write(output/'attrition.json',dict(train=train_drops,validation=val_drops))
    records=[];h=max(p.horizons)
    for seed in p.seeds:
        seed_dir=output/f'seed_{seed}';seed_dir.mkdir();prior_dir=reference_dir/f'seed_{seed}'
        checkpoint=torch.load(prior_dir/'uniform_pretrain.pt',map_location='cpu',weights_only=True)
        checkpoint['config']['joint_ids']=tuple(checkpoint['config']['joint_ids'])
        base=SkeletonJEPA(ModelConfig(**checkpoint['config']));base.load_state_dict(checkpoint['state'])
        model=clone_forecaster(base,p,'residual_det',seed)
        schedule=json.loads((prior_dir/'source_schedule.json').read_text())
        if len(schedule)!=p.forecast_steps: raise ValueError('Original schedule length changed')
        tick=time.monotonic();history=train_forecaster(model,tx,tv,tm,q,schedule,seed)
        training_seconds=time.monotonic()-tick
        _write(seed_dir/'history.json',history)
        _save_model(model,seed_dir/'residual_det_weight10.pt',dict(identity=IDENTITY,protocol=asdict(q),seed=seed,
            parent_pretrain_sha256=sha256_file(prior_dir/'uniform_pretrain.pt'),
            source_schedule_sha256=sha256_file(prior_dir/'source_schedule.json'),primary_unchanged=True))
        frames=_score_arm(model,vx,vv,vm,q,seed,'residual_det_weight10',seed_dir)
        normal=frames['actual'].query("label == 'normal'").set_index('source_id')
        primary=pd.read_csv(prior_dir/'residual_det_actual_sources.csv').query("label == 'normal'").set_index('source_id')
        persistence=pd.read_csv(reference_dir/'persistence_validation_sources.csv').query("label == 'normal'").set_index('source_id')
        if set(normal.index)!=set(primary.index) or set(normal.index)!=set(persistence.index): raise ValueError('Mismatched validation sources')
        row=dict(seed=seed,normal_mse=float(normal[f'mse_h{h}'].mean()),
            primary_weight1_mse=float(primary.loc[normal.index,f'mse_h{h}'].mean()),
            persistence_mse=float(persistence.loc[normal.index,f'mse_h{h}'].mean()),
            gain_over_weight1=float((primary.loc[normal.index,f'mse_h{h}']-normal[f'mse_h{h}']).mean()),
            gain_over_persistence=float((persistence.loc[normal.index,f'mse_h{h}']-normal[f'mse_h{h}']).mean()),
            normal_control_mse={c:float(f.query("label == 'normal'")[f'mse_h{h}'].mean()) for c,f in frames.items()},
            training_seconds=training_seconds,n_normal_sources=len(normal))
        records.append(row);_write(output/'progress.json',dict(completed=records))
    record=dict(status='secondary_development_complete_test_unexamined',declaration=declaration,records=records,
        seconds=time.monotonic()-started,artifact_sha256={str(f.relative_to(output)):sha256_file(f) for f in output.rglob('*') if f.is_file()})
    _write(output/'development.json',record);return record

def _verified_development(directory):
    directory=Path(directory);record=json.loads((directory/'development.json').read_text());d=record['declaration']
    if d['code_sha256']!=_audit_code(): raise ValueError('Audit code changed')
    if d['gradient_evidence_sha256']!=sha256_file(Path(__file__).parents[1]/'docs/training_gradient_audit.json'): raise ValueError('Gradient evidence changed')
    for package,version in d['versions'].items():
        if importlib.metadata.version(package)!=version: raise ValueError('Audit runtime changed')
    ref=Path(d['primary_reference'])
    if sha256_file(ref/'development.json')!=d['primary_development_sha256']: raise ValueError('Primary record changed')
    reference,p,input_dir=_reference(ref)
    for name,digest in record['artifact_sha256'].items():
        if sha256_file(directory/name)!=digest: raise ValueError(f'Audit artifact changed: {name}')
    if sorted(r['seed'] for r in record['records'])!=sorted(p.seeds): raise ValueError('Incomplete audit seeds')
    if not all(np.isfinite(r['normal_mse']) for r in record['records']): raise ValueError('Nonfinite audit outcome')
    return record,p,input_dir

def freeze(directory):
    directory=Path(directory);record,_,_=_verified_development(directory)
    if (directory/'freeze.json').exists(): raise FileExistsError('Audit already frozen')
    decision=dict(status='accepted',development_sha256=sha256_file(directory/'development.json'),
        all_declared_seeds_retained=True,primary_unchanged=True,code_sha256=_audit_code())
    _write(directory/'freeze.json',decision);return decision

def evaluate_once(directory,primary_test_dir,output_dir):
    directory,primary_test_dir,output=Path(directory),Path(primary_test_dir),Path(output_dir)
    sealed=json.loads((directory/'freeze.json').read_text())
    if sealed['status']!='accepted' or sealed['development_sha256']!=sha256_file(directory/'development.json'):
        raise ValueError('Audit freeze invalid')
    if (directory/'test_access.json').exists(): raise FileExistsError('Audit final test already accessed')
    if output.exists(): raise FileExistsError('Use a new final audit output')
    record,p,input_dir=_verified_development(directory);q=declared_change(p)
    primary=json.loads((primary_test_dir/'results.json').read_text())
    if primary['status']!='final_test_complete' or primary['frozen_development_sha256']!=record['declaration']['primary_development_sha256']:
        raise ValueError('Matching primary final evaluation must be released first')
    x,v,meta=_load_inputs(input_dir,p)
    sx,sv,sm,dropped=_filtered_subset(x,v,meta,p,(meta.split=='test').to_numpy())
    output.mkdir(parents=True);torch.set_num_threads(2)
    _write(directory/'test_access.json',dict(status='started',output=str(output.resolve())))
    _write(output/'attrition.json',dropped);records=[];paired_primary=[];paired_persistence=[];h=max(p.horizons)
    for seed in p.seeds:
        seed_dir=output/f'seed_{seed}';seed_dir.mkdir()
        model=load_revision_checkpoint(directory/f'seed_{seed}'/'residual_det_weight10.pt')
        frames=_score_arm(model,sx,sv,sm,q,seed,'residual_det_weight10',seed_dir)
        normal=frames['actual'].query("label == 'normal'").set_index('source_id')
        old=pd.read_csv(primary_test_dir/f'seed_{seed}'/'residual_det_actual_sources.csv').query("label == 'normal'").set_index('source_id')
        persistence=pd.read_csv(primary_test_dir/'persistence_sources.csv').query("label == 'normal'").set_index('source_id')
        if set(normal.index)!=set(old.index) or set(normal.index)!=set(persistence.index): raise ValueError('Final source mismatch')
        old=old.loc[normal.index];persistence=persistence.loc[normal.index]
        gain1=old[f'mse_h{h}']-normal[f'mse_h{h}'];gainp=persistence[f'mse_h{h}']-normal[f'mse_h{h}']
        paired_primary.append(pd.DataFrame(dict(source_id=normal.index,gain=gain1.to_numpy(),seed=seed)))
        paired_persistence.append(pd.DataFrame(dict(source_id=normal.index,gain=gainp.to_numpy(),seed=seed)))
        records.append(dict(seed=seed,normal_mse=float(normal[f'mse_h{h}'].mean()),primary_weight1_mse=float(old[f'mse_h{h}'].mean()),
            gain_over_weight1=float(gain1.mean()),gain_over_persistence=float(gainp.mean()),
            normal_control_mse={c:float(f.query("label == 'normal'")[f'mse_h{h}'].mean()) for c,f in frames.items()}))
    result=dict(status='secondary_final_complete',identity=IDENTITY,primary_unchanged=True,records=records,
        gain_over_primary=paired_gain_summary(pd.concat(paired_primary).groupby('source_id').gain.mean().to_numpy()),
        gain_over_persistence=paired_gain_summary(pd.concat(paired_persistence).groupby('source_id').gain.mean().to_numpy()),
        primary_final_results_sha256=sha256_file(primary_test_dir/'results.json'),frozen_development_sha256=sealed['development_sha256'])
    _write(output/'results.json',result)
    _write(directory/'test_access.json',dict(status='complete',output=str(output.resolve()),results_sha256=sha256_file(output/'results.json')))
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    dev=sub.add_parser('develop');dev.add_argument('--reference',required=True);dev.add_argument('--output',required=True)
    seal=sub.add_parser('freeze');seal.add_argument('--development',required=True)
    test=sub.add_parser('evaluate');test.add_argument('--development',required=True);test.add_argument('--primary-test',required=True);test.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='develop': result=develop(args.reference,args.output)
    elif args.command=='freeze': result=freeze(args.development)
    else: result=evaluate_once(args.development,args.primary_test,args.output)
    print(json.dumps(dict(status=result['status'],records=result.get('records'))))

if __name__=='__main__': main()
