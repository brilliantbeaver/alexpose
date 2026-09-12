"""Explicit source-learning-curve-v1 CLI; historical gate commands stay unchanged."""
import argparse
import json
from pathlib import Path

from ..future_innovation.fi_contracts import stage_lock, read_json, write_json
from .fi_scaling_cohort import freeze, read_study


def status(root,parent=None):
    root=Path(root)
    read_study(root,parent)
    from .fi_scaling_data import verify_seal
    verify_seal(root,'reports/cohort-audit-complete.json')
    stages=['data/cohort-complete.json','data/cache-complete.json','data/audit-complete.json','manifests/plan-complete.json','reports/complete.json']
    missing=next((s for s in stages if not (root/s).exists()),None)
    return dict(status='complete_report_present' if missing is None else 'incomplete_execution',
                next_missing_stage=missing,measurement_complete=None if missing is None else False,
                numerical_verification_performed=False,
                inventory=read_json(root/'reports/cohort-audit.json'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ('freeze','prepare','cache','audit-teacher','plan','run','report','verify','status'):
        p=sub.add_parser(name)
        p.add_argument('--run-root',type=Path,required=True)
        p.add_argument('--parent-root',type=Path)
        if name=='freeze':
            p.add_argument('--sequence-manifest',type=Path,required=True)
            p.add_argument('--video-manifest',type=Path,required=True)
            p.add_argument('--inspected-manifest',type=Path,action='append',required=True)
            p.add_argument('--participant-registry',type=Path)
            p.add_argument('--calibration',type=Path,required=True)
            p.add_argument('--protocol-document',type=Path,default=Path('docs/studies/future-innovation/source-learning-curve-protocol.md'))
        if name=='prepare':
            p.add_argument('--annotations',type=Path,nargs='+',required=True)
            p.add_argument('--video-root',type=Path,required=True)
            p.add_argument('--pose-model',type=Path,required=True)
            p.add_argument('--teacher-root',type=Path,required=True)
            p.add_argument('--checkpoint',type=Path,required=True)
        if name in ('cache','audit-teacher'): p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
        if name=='run': p.add_argument('--fold',type=int,choices=range(5))
    args=parser.parse_args(argv)
    kwargs=vars(args).copy(); command=kwargs.pop('command'); root=kwargs.pop('run_root'); parent=kwargs.pop('parent_root')
    if command=='freeze':
        if parent is None: parser.error('freeze requires --parent-root')
        kwargs['inspected_manifests']=kwargs.pop('inspected_manifest')
        with stage_lock(root,'scaling-freeze'): result=freeze(root,parent,**kwargs)
    elif command=='status': result=status(root,parent)
    elif command=='verify':
        from .fi_scaling_training import report
        result=report(root,parent,verify=True)
    else:
        from . import fi_scaling_data as data, fi_scaling_training as training
        operation={'prepare':data.prepare,'cache':data.cache,'audit-teacher':data.audit_teacher,
                   'plan':training.freeze_plan,'run':training.run,'report':training.report}[command]
        key=f'scaling-{command}'+(f'-{args.fold}' if command=='run' else '')
        with stage_lock(root,key): result=operation(root,parent,**kwargs)
    print(json.dumps(result or {'status':'stage_complete'},indent=2))
    return 0


if __name__=='__main__':
    main()
