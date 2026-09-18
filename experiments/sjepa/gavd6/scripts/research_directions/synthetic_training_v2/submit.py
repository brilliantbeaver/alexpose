#!/usr/bin/env python3
"""Sequential batch stages, conservative worst-case reservation, no implicit expansion."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training_v2.contracts import atomic_json
from gavd6_sjepa.research_directions.temporal_gait.contracts import stage_lock

def commands(config,*,dependency=None):
    common=['sbatch','--parsable','--no-requeue','--export=ALL',f'--chdir={ROOT}']
    for variable,option in (('ST_ACCOUNT','account'),('ST_PARTITION','partition')):
        value=os.environ.get(variable,'').strip()
        if value:common.append(f'--{option}={value}')
    result=[];previous=dependency
    for stage in ('audit','data','adaptation','information','direct','jepa','evaluate','optional','freeze','report'):
        gpu=stage in {'direct','jepa'}
        cmd=common+([f'--dependency=afterok:{previous}'] if previous else [])
        cmd+=['--gres=gpu:h100:1','--time=01:00:00'] if gpu else ['--time=00:20:00']
        cmd+=[str(ROOT/'slurm/synthetic-training-v2/stage.sbatch'),str(Path(config).resolve()),stage]
        result.append(cmd);previous=f'JOB_{stage}'
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--dependency');a=p.parse_args()
    cfg=RunConfig.load(a.config)
    if cfg.mode!='source':raise ValueError('Batch launcher accepts explicitly configured source runs only')
    if a.dry_run:print(json.dumps({'commands':commands(a.config,dependency=a.dependency),'gpu_job_bound':2,'max_parallel':1,'reserved_gpu_hours':2,'submits_jobs':False},indent=2));return
    cfg.require_gpu_scope()
    if cfg.device!='cuda':raise ValueError('HAIC batch training requires configured CUDA')
    if cfg.measured_gpu_hours+2>cfg.authorized_gpu_hours:raise PermissionError('Two1-hour sequential GPU jobs exceed authorized remaining budget')
    if not Path(cfg.bundle,'manifest.json').is_file():raise FileNotFoundError('Audited paired bundle is required before submitting training')
    if not os.environ.get('STV2_PYTHON'):raise ValueError('Source study env before sbatch')
    with stage_lock(cfg.root,'submit'):
        record=cfg.root/'submission.json'
        logs=cfg.root/'logs'
        pending=logs/'submission-pending.json'
        if pending.exists():
            raise FileExistsError(
                f'Uncertain previous submission recorded at {pending}. Inspect squeue/sacct '
                'and reconcile whether Slurm accepted that command before any retry; '
                'retain this marker and the existing job history.')
        if record.exists() and record.stat().st_size:
            raise FileExistsError(
                f'Submission already recorded at {record}. Inspect those jobs with squeue/sacct; '
                'retain this record and recover only the remaining stages. '
                'Do not repeat the full launcher or delete its job history.')
        logs.mkdir(parents=True,exist_ok=True)
        jobs=[];previous=a.dependency
        submission_env={key:value for key,value in os.environ.items() if not key.startswith('SBATCH_')}
        for cmd in commands(a.config,dependency=a.dependency):
            cmd=[x for x in cmd if not x.startswith('--dependency=')]
            if previous:cmd.insert(1,f'--dependency=afterok:{previous}')
            cmd.insert(1,f'--output={logs}/%x-%j.out')
            # A failed client response does not establish scheduler rejection.
            # Keep an uncertainty record even when the first call loses its ID.
            atomic_json(pending,{'stage':cmd[-1],'command':cmd})
            value=subprocess.check_output(cmd,env=submission_env,text=True).strip().split(';')[0]
            if not value.isdigit():raise RuntimeError(f'Unexpected sbatch response {value!r}; inspect squeue before retrying')
            previous=value;jobs.append({'stage':cmd[-1],'job_id':value})
            # Persist each accepted allocation before requesting another one.
            # A mid-chain scheduler failure must retain all earlier job IDs.
            atomic_json(record,jobs)
            pending.unlink()
    print(json.dumps(jobs,indent=2))
if __name__=='__main__':main()
