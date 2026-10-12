"""One entry point for tutorial notebooks, fixture tests and Slurm allocations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .common import atomic_json, read_json, sha256, verify_code
from .config import initialize, load_config, preflight


def verify_study(cfg):
    if cfg.get('study_kind') == 'jepa_response_followup':
        from .followup import verify_followup
        return verify_followup(cfg)
    from .scheduler import _verify_frozen
    from .data import load_dataset,select_rows
    import numpy as np
    import pandas as pd
    from .evaluation import measurement_rows,load_predictions
    work=Path(cfg['work']);_verify_frozen(cfg)
    state=read_json(work/'ledger.json');count=0
    for phase,value in state['completed'].items():
        if 'receipt' not in value:
            continue
        path=Path(value['receipt'])
        if sha256(path)!=value['sha256']:
            raise RuntimeError(f'Completion receipt changed: {phase}')
        for artifact,expected in read_json(path)['artifacts'].items():
            if not Path(artifact).is_file() or sha256(artifact)!=expected:
                raise RuntimeError(f'Artifact changed: {artifact}')
            count+=1
    bundle=load_dataset(state['completed']['prepare']['result']['bundle'])
    checked=0
    table_path=work/'evaluation/per-window.csv'
    if table_path.exists():
        dev=bundle.subset('development')
        phases={(p['recipe']['recipe_id'],p['seed']):p for p in read_json(work/'plan.json')['phases'] if p['phase']!='pretrain'}
        expected=np.flatnonzero([r['split']=='development' for r in bundle.records])
        ids=['person_id','motion_id','window_id','variant','extractor']
        positions={tuple(str(r[k]) for k in ids):i for i,r in enumerate(dev.records)}
        if len(positions)!=len(dev.records):
            raise RuntimeError('Development record identities are duplicated')
        families={}
        for index,row in enumerate(dev.records):
            families.setdefault(row['source_family_id'],[]).append(index)
        predictions={};seen={key:np.zeros(len(dev.records),bool) for key in phases}
        for chunk in pd.read_csv(table_path,keep_default_na=False,chunksize=10000,dtype={k:str for k in ids}):
            for (method,seed,family),saved in chunk.groupby(['method','seed','source_family_id'],sort=False):
                key=(method,int(seed))
                if key not in phases:
                    continue # Spatial/filter exports are verified by their artifact receipts.
                if key not in predictions:
                    xy,indices=load_predictions(state['completed'][phases[key]['phase_id']]['result'])
                    if not np.array_equal(indices,expected):
                        raise RuntimeError('Prediction record order differs from the saved development population')
                    predictions[key]=xy
                rows=families[family]
                fresh=measurement_rows(select_rows(dev,rows),predictions[key][rows],cfg,method,int(seed))
                fresh[ids]=fresh[ids].astype(str)
                retained=saved.set_index(ids)
                if retained.index.has_duplicates:
                    raise RuntimeError('Evaluation population contains duplicated records')
                row_positions=np.asarray([positions[tuple(index)] for index in retained.index])
                if seen[key][row_positions].any():
                    raise RuntimeError('Evaluation population contains repeated saved rows')
                seen[key][row_positions]=True
                reconstructed=fresh.set_index(ids).loc[retained.index]
                for metric in ('A_error','waveform_error','support_frames'):
                    values=pd.to_numeric(retained[metric],errors='coerce').to_numpy()
                    if not np.allclose(values,reconstructed[metric].to_numpy(float),equal_nan=True,atol=1e-8,rtol=1e-7):
                        raise RuntimeError(f'Saved metric does not reconstruct: {method} {seed} {metric}')
                checked+=len(saved)
        if any(not population.all() for population in seen.values()):
            raise RuntimeError('Evaluation population is incomplete')
    return dict(status='GAIT_FIDELITY_VERIFIED',artifacts_checked=count,reconstructed_rows=checked,
                evidence_status=bundle.evidence_status,cuda_verified_here=False)


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    for command in ('init','setup-followup','preflight','prepare','validate','plan','cohort','run','worker','evaluate','status','report','verify','lock-confirmation','evaluate-confirmation'):
        sub=commands.add_parser(command)
        sub.add_argument('--work',type=Path,required=True)
        if command=='setup-followup':
            sub.add_argument('--parent-work',type=Path,required=True)
            sub.add_argument('--deadline-utc',help='Explicit offset-aware cutoff for a new child; saved cutoffs cannot be edited.')
            sub.add_argument('--include-base-readouts',action='store_true',default=None,
                             help='Add nine coordinate-only readouts using the same frozen encoders; 27 optimization phases.')
        if command=='init':
            sub.add_argument('--root',type=Path)
            sub.add_argument('--fixture',action='store_true')
            sub.add_argument('--source-config',type=Path)
            sub.add_argument('--source-bundle',type=Path)
            sub.add_argument('--source-selection',choices=('full_manifest','legacy_roster'),default='full_manifest')
            sub.add_argument('--cohort-preset',choices=('named_walking','treadmill_walking','all_eligible','reviewed'))
            sub.add_argument('--reservation-csv',type=Path)
            sub.add_argument('--motion-review-csv',type=Path)
            sub.add_argument('--num-shards',type=int)
            sub.add_argument('--experiment-set',choices=('core','full'))
        if command=='preflight':
            sub.add_argument('--gpu',action='store_true')
        if command=='run':
            sub.add_argument('--local',action='store_true')
            sub.add_argument('--max-jobs',type=int,default=8)
            stage=sub.add_mutually_exclusive_group()
            stage.add_argument('--prepare-only',action='store_true')
            stage.add_argument('--gavd-only',action='store_true')
            stage.add_argument('--gavd-confirmation',action='store_true')
            stage.add_argument('--confirmation-only',action='store_true')
        if command=='lock-confirmation':
            sub.add_argument('--reviewed-by',required=True)
            sub.add_argument('--evidence',required=True)
            sub.add_argument('--exposure-ledger',type=Path)
        if command=='evaluate-confirmation':
            sub.add_argument('--bundle',type=Path)
            sub.add_argument('--lock',type=Path)
            sub.add_argument('--output',type=Path)
        if command=='worker':
            sub.add_argument('--phase-id',required=True)
            sub.add_argument('--attempt',type=Path,required=True)
    from .gavd import add_commands
    add_commands(commands)
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    if args.command.startswith('gavd-'):
        from .gavd import run_command
        result=run_command(args)
        print(json.dumps(result,indent=2,allow_nan=False))
        return result
    if args.command=='setup-followup':
        from .followup import initialize_followup
        cfg=initialize_followup(args.work,parent_work=args.parent_work,deadline_utc=args.deadline_utc,
                                include_base_readouts=args.include_base_readouts)
        counts=read_json(Path(cfg['work'])/'plan.json')['counts']
        result=dict(status='RESPONSE_FOLLOWUP_CONFIGURED', work=cfg['work'],
                    parent_work=cfg['followup']['parent_binding']['parent_work'],
                    prepared_bundle=cfg['followup']['parent_binding']['bundle'],
                    actual_updates=cfg['followup']['parent_binding']['actual_updates'],
                    optimization_phases=counts['optimization_phases'], final_models=counts['final_fits'], seeds=cfg['seeds'],
                    deadline_utc=cfg['followup']['deadline_utc'],
                    maximum_gpu_hours=cfg['resources']['gpu_hours'],
                    session=str(Path(cfg['work'])/'session.env'), parent_modified=False)
    elif args.command=='init':
        result=initialize(args.work,root=args.root,fixture=args.fixture,source_config=args.source_config,source_bundle=args.source_bundle,
                          source_selection=args.source_selection,cohort_preset=args.cohort_preset,
                          reservation_csv=args.reservation_csv,motion_review_csv=args.motion_review_csv,
                          num_shards=args.num_shards,experiment_set=args.experiment_set)
    else:
        cfg=load_config(args.work)
        followup=cfg.get('study_kind')=='jepa_response_followup'
        if followup and args.command in {'prepare','lock-confirmation','evaluate-confirmation'}:
            raise ValueError('The response follow-up uses only its bound prepared parent; confirmation is excluded')
        if args.command=='preflight':
            if args.gpu and not cfg['fixture']:
                import os
                if not os.environ.get('SLURM_JOB_ID'):
                    raise RuntimeError('GPU preflight belongs in a Slurm GPU allocation')
            result=preflight(cfg,gpu=args.gpu)
            atomic_json(args.work/'preflight.json',result)
        elif args.command=='plan':
            result=read_json(args.work/'plan.json')
        elif args.command=='cohort':
            if followup:
                result=dict(status='READ_ONLY_PARENT_DEPENDENCY',parent=cfg['followup']['parent_binding']['parent_work'],
                            bundle=cfg['followup']['parent_binding']['bundle'])
            elif cfg['fixture']:
                result={'status':'SOFTWARE_FIXTURE','message':'No AMASS source cohort is used by the fixture.'}
            elif cfg['data'].get('source_selection','legacy_roster')=='legacy_roster':
                result={'status':'LEGACY_ROSTER','source_bundle':cfg['source_bundle']}
            else:
                path=Path(cfg['cohort']['plan_path'])
                cohort=read_json(path)
                result=dict(plan=str(path),identity=cohort.get('identity'),summary=cohort.get('summary',{}),
                            note='Full selection and exclusion records are retained alongside the plan.')
        elif args.command in {'run','prepare'}:
            from .scheduler import run
            if args.command=='prepare':
                if not cfg['fixture']:
                    raise RuntimeError('For source preparation use run.sh launch --prepare-only')
                result=run(cfg,local=True,prepare_only=True)
            else:
                result=run(cfg,local=args.local,max_jobs=args.max_jobs,prepare_only=args.prepare_only,
                           gavd_only=args.gavd_only,gavd_confirmation=args.gavd_confirmation,
                           confirmation_only=args.confirmation_only)
        elif args.command=='lock-confirmation':
            from .confirmation import lock_confirmation
            result=lock_confirmation(cfg,args.work/'confirmation/lock.json',reviewed_by=args.reviewed_by,
                                     evidence=args.evidence,exposure_ledger=args.exposure_ledger)
        elif args.command=='evaluate-confirmation':
            from .confirmation import evaluate_confirmation
            result=evaluate_confirmation(cfg,args.bundle or args.work/'confirmation/bundle',
                                         args.lock or args.work/'confirmation/lock.json',
                                         args.output or args.work/'confirmation/evaluation')
        elif args.command=='worker':
            from .scheduler import execute_worker
            result=execute_worker(cfg,args.phase_id,args.attempt)
        elif args.command=='validate':
            from .data import load_dataset,validate_bundle
            if followup:
                from .followup import verify_parent
                path=verify_parent(cfg)['bundle']
            else:
                state=read_json(args.work/'ledger.json')
                path=state['completed']['prepare']['result']['bundle']
            result=dict(bundle=path,**validate_bundle(load_dataset(path)))
        elif args.command=='status':
            from .scheduler import status
            result=status(cfg)
        elif args.command=='evaluate':
            if followup:
                from .response_evaluation import evaluate_response_followup
                result=evaluate_response_followup(cfg)
            else:
                from .evaluation import evaluate_study
                result=evaluate_study(cfg)
        elif args.command=='verify':
            result=verify_study(cfg)
        else:
            print((args.work/'report.md').read_text())
            return
    print(json.dumps(result,indent=2,allow_nan=False))
    return result


if __name__=='__main__':
    main()
