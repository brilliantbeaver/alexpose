#!/usr/bin/env python3
"""Sequential batch stages, conservative worst-case reservation, no implicit expansion."""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'src'))
from gavd6_sjepa.research_directions.synthetic_training_v2.config import RunConfig

def commands(config,*,dependency=None):
    common=['sbatch','--parsable','--export=ALL',f'--chdir={ROOT}']
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
    logs=cfg.root/'logs';logs.mkdir(parents=True,exist_ok=True)
    jobs=[];previous=a.dependency
    for cmd in commands(a.config,dependency=a.dependency):
        cmd=[x for x in cmd if not x.startswith('--dependency=')]
        if previous:cmd.insert(1,f'--dependency=afterok:{previous}')
        cmd.insert(1,f'--output={logs}/%x-%j.out')
        value=subprocess.check_output(cmd,text=True).strip().split(';')[0]
        if not value.isdigit():raise RuntimeError(f'Unexpected sbatch response {value!r}')
        previous=value;jobs.append({'stage':cmd[-1],'job_id':value})
        (cfg.root/'submission.json').write_text(json.dumps(jobs,indent=2)+'\n')
    print(json.dumps(jobs,indent=2))
if __name__=='__main__':main()
