"""Dependency scheduling, immutable attempts and shared allocation accounting."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import uuid

from .common import atomic_json, locked, read_json, sha256, utc_now, verify_code
from .config import freeze, load_config, preflight

TERMINAL = {'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','BOOT_FAIL','DEADLINE','REVOKED'}


def preparation_phase_ids(cfg,confirmation=False):
    """Partition the frozen cohort independently of the concurrency limit."""
    if cfg['fixture']:
        return ['prepare']
    count = cfg['data'].get('num_shards', 8)
    if type(count) is not int or count < 1:
        raise ValueError('The preparation shard count must be a positive integer')
    prefix='confirmation' if confirmation else 'prepare'
    return [f'{prefix}-shard-{i:05d}' for i in range(count)] if count > 100 else [
        f'{prefix}-shard-{i:02d}' for i in range(count)]


def _state(work):
    path = Path(work)/'ledger.json'
    return read_json(path) if path.exists() else dict(schema='gait-fidelity-ledger-v1', attempts=[], completed={}, created_utc=utc_now())


def reserve(work, phase_id, gpu_hours, *, limit, max_jobs, budget_category=None, category_limits=None):
    """Commit before sbatch. Unresolved submissions remain charged, preventing duplicates."""
    if not math.isfinite(gpu_hours) or gpu_hours < 0:
        raise ValueError('Finite nonnegative allocation reservation required')
    with locked(Path(work)/'locks/ledger.lock'):
        state = _state(work)
        active = [a for a in state['attempts'] if a['status'] in {'reserved','submitted','running','accounting_pending'}]
        if phase_id in state['completed'] or any(a['phase_id']==phase_id for a in active):
            raise RuntimeError('Phase already completed or has an unresolved attempt')
        if len(active) >= max_jobs:
            raise RuntimeError('Combined worker concurrency reached')
        used = sum(a.get('allocated_gpu_hours',a['reserved_gpu_hours']) for a in state['attempts'])
        if used + gpu_hours > limit + 1e-8:
            raise RuntimeError(f'Study allocation budget exhausted: {used:.3f}+{gpu_hours:.3f}>{limit:.3f}')
        if category_limits is not None:
            if budget_category not in category_limits:
                raise ValueError('A registered follow-up accounting category is required')
            category_used = sum(a.get('allocated_gpu_hours', a['reserved_gpu_hours']) for a in state['attempts']
                                if a.get('budget_category') == budget_category)
            if category_used + gpu_hours > category_limits[budget_category] + 1e-8:
                raise RuntimeError(f'Follow-up {budget_category} allocation allowance exhausted')
        identifier = uuid.uuid4().hex[:16]
        attempt = dict(id=identifier,phase_id=phase_id,job_id=None,status='reserved',
                       reserved_gpu_hours=gpu_hours,path=str(Path(work)/'attempts'/phase_id/identifier),
                       submitted_utc=utc_now(),job_name='gf-'+identifier)
        if budget_category is not None:
            attempt['budget_category'] = budget_category
        state['attempts'].append(attempt)
        atomic_json(Path(work)/'ledger.json',state)
        return attempt


def update_attempt(work, identifier, values):
    with locked(Path(work)/'locks/ledger.lock'):
        state = _state(work)
        match = next(a for a in state['attempts'] if a['id']==identifier)
        match.update(values)
        atomic_json(Path(work)/'ledger.json',state)


def complete_attempt(work, attempt, accounting):
    receipt_path = Path(attempt['path'])/'complete.json'
    receipt = read_json(receipt_path)
    if receipt['phase_id'] != attempt['phase_id'] or receipt['config_sha256'] != sha256(Path(work)/'config.json'):
        raise RuntimeError('Worker completion identity mismatch')
    for name, expected in receipt['artifacts'].items():
        if not Path(name).is_file() or sha256(name) != expected:
            raise RuntimeError(f'Worker artifact changed: {name}')
    with locked(Path(work)/'locks/ledger.lock'):
        state = _state(work)
        row = next(a for a in state['attempts'] if a['id']==attempt['id'])
        row.update(accounting,status='complete')
        state['completed'][attempt['phase_id']] = dict(receipt=str(receipt_path),sha256=sha256(receipt_path),result=receipt['result'])
        atomic_json(Path(work)/'ledger.json',state)


def verify_completed(value):
    if 'receipt' in value:
        path=Path(value['receipt'])
        if not path.is_file() or sha256(path)!=value['sha256']:
            raise RuntimeError('Prerequisite completion receipt changed')
        for artifact,expected in read_json(path)['artifacts'].items():
            if not Path(artifact).is_file() or sha256(artifact)!=expected:
                raise RuntimeError(f'Prerequisite artifact changed: {artifact}')
    elif 'bundle_manifest_sha256' in value:
        if sha256(Path(value['result']['bundle'])/'manifest.json')!=value['bundle_manifest_sha256']:
            raise RuntimeError('Merged data manifest changed')
    else:
        raise RuntimeError('Prerequisite lacks a retained artifact identity')


def preparation_inspection(cfg, bundle_path,confirmation=False):
    from .data import load_dataset,validate_bundle
    from .visualization import build_viewer
    from .masking import audit_matching
    import numpy as np
    work=Path(cfg['work']);bundle=load_dataset(bundle_path,allow_confirmation=confirmation)
    output=work/('confirmation/data' if confirmation else 'data');output.mkdir(parents=True,exist_ok=True)
    viewer=build_viewer(bundle,output/'viewer.html')
    indices=np.unique(np.linspace(0,len(bundle.records)-1,min(16,len(bundle.records))).round().astype(int))
    audit=audit_matching(bundle.inputs['observed'][indices],draws=128,seed=17,patch_size=cfg['model']['patch_size'])
    atomic_json(output/'mask-audit.json',audit)
    result=dict(bundle=str(bundle_path),viewer=str(viewer),counts=validate_bundle(bundle),mask_audit=str(output/'mask-audit.json'),
                review_status='automated_only; human review is not inferred from software completion')
    atomic_json(output/'admission.json',result)
    return result


def parse_accounting(text, job_id):
    """Only the parent allocation row counts; batch/extern are never added."""
    matches=[]
    for line in text.splitlines():
        fields=line.split('|')
        if len(fields)<5 or fields[0]!=str(job_id):
            continue
        identifier,state,exitcode,seconds,tres=fields[:5]
        state=state.split()[0].rstrip('+')
        if not seconds.isdigit():
            raise ValueError('sacct ElapsedRaw is not an integer')
        resources=dict(item.split('=',1) for item in tres.split(',') if '=' in item)
        if 'gres/gpu' in resources:
            count=int(resources['gres/gpu'])
        else:
            count=sum(int(v) for k,v in resources.items() if k.startswith(('gres/gpu:','gres/gpu/')))
        if state in TERMINAL and count < 1 and int(seconds)>0:
            raise ValueError('GPU allocation accounting lacks allocated GPU count')
        matches.append(dict(scheduler_state=state,exit_code=exitcode,elapsed_seconds=int(seconds),
                            allocated_gpus=count,allocated_gpu_hours=count*int(seconds)/3600))
    if len(matches)>1:
        raise ValueError('Ambiguous duplicate Slurm allocation rows')
    return matches[0] if matches else None


def _reconcile(cfg):
    work=Path(cfg['work'])
    for attempt in _state(work)['attempts']:
        if attempt['status'] not in {'reserved','submitted','running','accounting_pending'}:
            continue
        if not attempt['job_id']:
            # Lost sbatch responses are recovered by the unique persisted job name.
            result=subprocess.run(['sacct','-n','-P','--name',attempt['job_name'],'--starttime',attempt['submitted_utc'][:19],
                                   '--format=JobIDRaw,JobName%80'],capture_output=True,text=True,check=True,env={**os.environ,'TZ':'UTC'})
            ids={line.split('|')[0] for line in result.stdout.splitlines()
                 if len(line.split('|'))>=2 and line.split('|')[1]==attempt['job_name'] and line.split('|')[0].isdigit()}
            if len(ids)!=1:
                raise RuntimeError(f"Submission {attempt['id']} is unresolved. Preserve ledger; inspect sbatch/sacct before retrying.")
            attempt['job_id']=ids.pop()
            update_attempt(work,attempt['id'],dict(job_id=attempt['job_id'],status='submitted'))
        result=subprocess.run(['sacct','-n','-P','-j',str(attempt['job_id']),
                               '--format=JobIDRaw,State,ExitCode,ElapsedRaw,AllocTRES%200'],capture_output=True,text=True,check=True)
        account=parse_accounting(result.stdout,attempt['job_id'])
        if account is None:
            update_attempt(work,attempt['id'],dict(status='accounting_pending'))
        elif account['scheduler_state'] in TERMINAL:
            if account['scheduler_state']=='COMPLETED' and account['exit_code']=='0:0':
                complete_attempt(work,attempt,account)
            else:
                update_attempt(work,attempt['id'],dict(**account,status='failed'))
        else:
            update_attempt(work,attempt['id'],dict(status='running',scheduler_state=account['scheduler_state']))


def _submit(cfg, phase_id):
    work=Path(cfg['work']);r=cfg['resources']
    minutes=r['prepare_wall_minutes'] if phase_id.startswith(('prepare','confirmation','gavd-')) else r['phase_wall_minutes']
    kwargs = {}
    if cfg.get('study_kind') == 'jepa_response_followup':
        from .followup import allocation_minutes, category
        previous = sum(a['phase_id'] == phase_id for a in _state(work)['attempts'])
        minutes = allocation_minutes(cfg, phase_id, previous)
        kwargs = dict(budget_category=category(phase_id, previous), category_limits=cfg['followup']['budgets_gpu_hours'])
    attempt=reserve(work,phase_id,minutes/60,limit=r['gpu_hours'],max_jobs=r['max_jobs'], **kwargs)
    Path(attempt['path']).mkdir(parents=True)
    wall_minutes = minutes-1 if cfg.get('study_kind') == 'jepa_response_followup' else minutes
    command=['sbatch','--parsable',f"--job-name={attempt['job_name']}",f"--account={r['account']}",
             '--nodes=1','--ntasks=1','--export=ALL','--no-requeue',
             f"--partition={r['partition']}",f"--gres=gpu:{r['gpu']}",f'--time={wall_minutes}',
             f"--cpus-per-task={r['cpu_workers']}",f"--mem={r['memory']}",
             f"--output={attempt['path']}/slurm-%j.out",f"--chdir={cfg['code_root']}",
             str(Path(cfg['code_root'])/'slurm/gait-fidelity/worker.sbatch'),str(work),phase_id,attempt['path']]
    result=subprocess.run(command,capture_output=True,text=True)
    (Path(attempt['path'])/'submission.json').write_text(json.dumps(dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2))
    if result.returncode:
        # Nonzero sbatch may still be ambiguous after a transport interruption.
        raise RuntimeError(f'Submission failed; reservation retained for reconciliation: {result.stderr.strip()}')
    job_id=result.stdout.strip().split(';')[0]
    if not re.fullmatch(r'\d+',job_id):
        raise RuntimeError('Ambiguous sbatch response; reservation retained')
    update_attempt(work,attempt['id'],dict(job_id=job_id,status='submitted'))
    return job_id


def _verify_frozen(cfg):
    frozen=read_json(Path(cfg['work'])/'frozen.json')
    if sha256(Path(cfg['work'])/'config.json')!=frozen['config_sha256'] or sha256(Path(cfg['work'])/'plan.json')!=frozen['plan_sha256']:
        raise RuntimeError('Frozen configuration or phase plan changed')
    verify_code(cfg['code_root'],frozen['code'])


def execute_worker(cfg, phase_id, attempt_path):
    """Run exactly one reserved phase; no scheduler calls from GPU workers."""
    from .data import load_dataset, fixture_bundle, save_dataset
    work=Path(cfg['work']);attempt_path=Path(attempt_path).resolve()
    if not attempt_path.is_relative_to(work/'attempts'):
        raise ValueError('Worker attempt must be inside this run attempts directory')
    _verify_frozen(cfg)
    attempts=[a for a in _state(work)['attempts'] if a['phase_id']==phase_id and Path(a['path'])==attempt_path]
    if len(attempts)!=1:
        raise ValueError('Worker requires one persisted allocation reservation')
    attempt_path.mkdir(parents=True,exist_ok=True)
    with locked(attempt_path/'worker.lock',nonblocking=True):
        if (attempt_path/'complete.json').exists():
            raise FileExistsError('Completed attempts are immutable')
        started=time.monotonic()
        if cfg.get('study_kind') == 'jepa_response_followup':
            from .followup import execute_followup_phase
            result, artifact_root = execute_followup_phase(cfg, phase_id, attempt_path, started=started)
            artifacts = {str(p):sha256(p) for p in artifact_root.rglob('*') if p.is_file() and p.suffix != '.lock'}
        elif phase_id.startswith('gavd-'):
            from argparse import Namespace
            from .gavd import run_command
            atomic_json(attempt_path/'preflight.json',preflight(cfg,gpu=True))
            result=run_command(Namespace(command='gavd-extract',work=work,
                                         shard_index=int(phase_id.rsplit('-',1)[1]),
                                         split=phase_id.split('-')[1]))
            artifact_root=Path(result['output'])
            artifacts={str(p):sha256(p) for p in artifact_root.rglob('*') if p.is_file() and p.suffix!='.lock'}
        elif phase_id.startswith(('prepare','confirmation')):
            atomic_json(attempt_path/'preflight.json',preflight(cfg,gpu=not cfg['fixture']))
            if cfg['fixture']:
                result_path=attempt_path/'bundle'
                save_dataset(fixture_bundle(samples=cfg['data']['samples']),result_path)
            else:
                from .preparation import prepare
                prepared=copy.deepcopy(cfg)
                prepared['data']['shard_index']=int(phase_id.rsplit('-',1)[1])
                prepared['data']['num_shards']=cfg['data'].get('num_shards',8)
                if phase_id.startswith('confirmation'):
                    prepared['data']['partition']='confirmation'
                    prepared['cohort']['confirmation_lock']=str(work/'confirmation/lock.json')
                prepared['data']['resume_preparation_paths']=[str(Path(a['path'])/'prepared')
                    for a in _state(work)['attempts']
                    if a['phase_id']==phase_id and a['status']=='failed'
                    and (Path(a['path'])/'prepared').is_dir()]
                result_path=prepare(prepared,attempt_path/'prepared')
            result=dict(bundle=str(result_path))
            artifact_root=Path(result_path) if cfg['fixture'] else attempt_path/'prepared'
            artifacts={str(p):sha256(p) for p in artifact_root.rglob('*') if p.is_file()} if artifact_root.is_dir() else {str(artifact_root):sha256(artifact_root)}
        elif phase_id=='profile':
            from .profiling import profile_training
            state=_state(work);verify_completed(state['completed']['prepare'])
            bundle=load_dataset(state['completed']['prepare']['result']['bundle'])
            worker_setup_seconds=time.monotonic()-started
            spent=sum(a.get('allocated_gpu_hours',0.) for a in state['attempts'])
            result=profile_training(bundle,cfg,attempt_path/'profile',spent,
                                    worker_setup_seconds=worker_setup_seconds)
            artifacts={str(p):sha256(p) for p in (attempt_path/'profile').rglob('*') if p.is_file()}
        else:
            from .training import train_phase
            phase=next(p for p in read_json(work/'plan.json')['phases'] if p['phase_id']==phase_id)
            state=_state(work)
            for parent in phase['depends_on']:
                if parent not in state['completed']:
                    raise RuntimeError(f'Incomplete prerequisite: {parent}')
                verify_completed(state['completed'][parent])
            verify_completed(state['completed']['prepare'])
            data_path=state['completed']['prepare']['result']['bundle']
            bundle=load_dataset(data_path)
            parent=phase['depends_on'][0]
            upstream=Path(state['completed'][parent]['result']['checkpoint']) if parent!='prepare' else None
            fit_cfg=copy.deepcopy(cfg)
            if not cfg['fixture']:
                verify_completed(state['completed']['profile'])
                budget=state['completed']['profile']['result']['budget']
                if not budget['selected']:
                    raise RuntimeError('Profile does not admit the complete matrix within allowance')
                fit_cfg['training'].update(budget['selected']['updates'])
            previous=[a for a in state['attempts'] if a['phase_id']==phase_id and a['status']=='failed'
                      and (Path(a['path'])/'fit/checkpoint.pt').is_file()]
            if previous:
                fit_cfg['training']['resume_from']=str(Path(previous[-1]['path'])/'fit/checkpoint.pt')
            result=train_phase(bundle,phase['recipe'],phase['phase'],phase['seed'],fit_cfg,attempt_path/'fit',upstream)
            result=dict(result)
            artifacts={str(p):sha256(p) for p in (attempt_path/'fit').rglob('*') if p.is_file()}
        _verify_frozen(cfg)
        atomic_json(attempt_path/'complete.json',dict(phase_id=phase_id,config_sha256=sha256(work/'config.json'),
                    completed_utc=utc_now(),worker_seconds=time.monotonic()-started,result=result,artifacts=artifacts))
        return result


def _merge_preparation(cfg,confirmation=False):
    from .data import merge_datasets,load_dataset
    work=Path(cfg['work']);state=_state(work)
    keys=preparation_phase_ids(cfg,confirmation)
    for key in keys:
        verify_completed(state['completed'][key])
    paths=[Path(state['completed'][k]['result']['bundle']) for k in keys]
    result=work/('confirmation/bundle' if confirmation else 'data/bundle')
    if result.exists():
        existing=load_dataset(result,allow_confirmation=confirmation)
        expected={str(path.resolve()):sha256(path if path.suffix=='.json' else path/'manifest.json') for path in paths}
        retained={row['path']:row['manifest_sha256'] for row in existing.provenance['shard_receipts']}
        if retained!=expected:
            raise RuntimeError('Existing merged data do not match completed preparation shards')
    else:
        result=merge_datasets(paths,result,allow_confirmation=confirmation)
    inspection=preparation_inspection(cfg,result,confirmation)
    with locked(work/'locks/ledger.lock'):
        state=_state(work)
        state['completed']['confirmation_prepare' if confirmation else 'prepare']=dict(result=inspection,merged_from=keys,bundle_manifest_sha256=sha256(Path(result)/'manifest.json'))
        atomic_json(work/'ledger.json',state)


def _run_gavd(cfg,max_jobs,partition='development'):
    """Use the same coordinator lock, allocation budget and worker cap as AMASS."""
    work=Path(cfg['work']);plan_path=work/'gavd/plan.json'
    if not plan_path.is_file():
        raise FileNotFoundError('Run gavd-plan before launch --gavd-only')
    plan=read_json(plan_path)
    if partition=='confirmation':
        from .gavd import _check_confirmation
        _check_confirmation(work/'gavd',plan)
    count=plan['num_shards']
    if type(count) is not int or count<1:
        raise ValueError('GAVD plan requires a positive shard count')
    keys=[f'gavd-{partition}-shard-{index:05d}' for index in range(count)]
    identity=sha256(plan_path)
    receipt=work/f'gavd/scheduler-plan-{partition}.json'
    if receipt.exists():
        if read_json(receipt)['plan_sha256']!=identity:
            raise RuntimeError('GAVD extraction plan changed after submission')
    else:
        atomic_json(receipt,dict(plan_sha256=identity,phases=keys))
    while True:
        if sha256(plan_path)!=identity:
            raise RuntimeError('GAVD extraction plan changed while running')
        _reconcile(cfg);state=_state(work)
        if all(key in state['completed'] for key in keys):
            return dict(status='GAVD_EXTRACTION_COMPLETE',shards=count,
                        outputs=[state['completed'][key]['result']['output'] for key in keys])
        active={a['phase_id'] for a in state['attempts'] if a['status'] in {'reserved','submitted','running','accounting_pending'}}
        ready=[key for key in keys if key not in state['completed'] and key not in active]
        for key in ready[:max(0,max_jobs-len(active))]:
            if sum(a['phase_id']==key for a in state['attempts'])>=cfg['resources']['max_attempts']:
                raise RuntimeError(f'Phase {key} exhausted retries; inspect retained attempt logs')
            _submit(cfg,key)
        time.sleep(min(60,cfg['resources']['poll_seconds']))


def _run_confirmation(cfg,max_jobs):
    work=Path(cfg['work'])
    lock=work/'confirmation/lock.json'
    if not lock.is_file():
        raise FileNotFoundError('Run lock-confirmation before launch --confirmation-only')
    from .cohort import load_cohort
    check=copy.deepcopy(cfg);check['cohort']['confirmation_lock']=str(lock)
    load_cohort(check,partition='confirmation',require_available=False)
    keys=preparation_phase_ids(cfg,confirmation=True)
    while True:
        _reconcile(cfg);state=_state(work)
        if all(key in state['completed'] for key in keys):
            if 'confirmation_prepare' not in state['completed']:
                _merge_preparation(cfg,confirmation=True)
            return dict(status='CONFIRMATION_PREPARATION_COMPLETE',bundle=str(work/'confirmation/bundle'))
        active={a['phase_id'] for a in state['attempts'] if a['status'] in {'reserved','submitted','running','accounting_pending'}}
        ready=[key for key in keys if key not in state['completed'] and key not in active]
        for key in ready[:max(0,max_jobs-len(active))]:
            if sum(a['phase_id']==key for a in state['attempts'])>=cfg['resources']['max_attempts']:
                raise RuntimeError(f'Phase {key} exhausted retries; inspect retained attempt logs')
            _submit(cfg,key)
        time.sleep(min(60,cfg['resources']['poll_seconds']))


def run(cfg, *, local=False, max_jobs=8, prepare_only=False, gavd_only=False,gavd_confirmation=False,confirmation_only=False):
    work=Path(cfg['work'])
    if local != cfg['fixture']:
        raise ValueError('Use --local only for CPU fixtures; real experiments run through Slurm')
    if not local and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Launch the CPU Slurm coordinator with slurm/gait-fidelity/run.sh launch')
    if (gavd_only or gavd_confirmation or confirmation_only) and (local or prepare_only):
        raise ValueError('GAVD extraction is a separate source allocation stage')
    if not 1 <= max_jobs <= min(8,cfg['resources']['max_jobs']):
        raise ValueError('Concurrency exceeds frozen study cap')
    freeze(cfg)
    with locked(work/'locks/controller.lock',nonblocking=True):
        if cfg.get('study_kind') == 'jepa_response_followup':
            if prepare_only or gavd_only or gavd_confirmation or confirmation_only:
                raise ValueError('The follow-up uses its bound parent; preparation/confirmation/GAVD branches are disabled')
            from .followup import run_followup
            return run_followup(cfg, local=local, max_jobs=max_jobs)
        if gavd_only or gavd_confirmation:
            return _run_gavd(cfg,max_jobs,'confirmation' if gavd_confirmation else 'development')
        if confirmation_only:
            return _run_confirmation(cfg,max_jobs)
        phases=read_json(work/'plan.json')['phases']
        prep_ids=preparation_phase_ids(cfg)
        while True:
            if not local:
                _reconcile(cfg)
            state=_state(work)
            if not local and 'prepare' not in state['completed'] and all(k in state['completed'] for k in prep_ids):
                _merge_preparation(cfg);state=_state(work)
            if prepare_only and 'prepare' in state['completed']:
                if not (work/'data/admission.json').is_file():
                    preparation_inspection(cfg,state['completed']['prepare']['result']['bundle'])
                return dict(status='PREPARATION_COMPLETE',bundle=state['completed']['prepare']['result']['bundle'])
            if all(p['phase_id'] in state['completed'] for p in phases):
                break
            active={a['phase_id'] for a in state['attempts'] if a['status'] in {'reserved','submitted','running','accounting_pending'}}
            ready=[key for key in prep_ids if key not in state['completed'] and key not in active]
            if not prepare_only:
                if not local and 'prepare' in state['completed'] and 'profile' not in state['completed']:
                    if 'profile' not in active:
                        ready.append('profile')
                elif local or 'profile' in state['completed']:
                    if not local and not state['completed']['profile']['result']['budget']['selected']:
                        raise RuntimeError('Measured profile exceeds the full and half-budget schedules. Inspect profile.json; no final fits launched.')
                    ready += [p['phase_id'] for p in phases if p['phase_id'] not in state['completed'] and p['phase_id'] not in active
                              and all(k in state['completed'] for k in p['depends_on'])]
            slots=max_jobs-len(active)
            for phase_id in ready[:slots]:
                attempts=[a for a in state['attempts'] if a['phase_id']==phase_id]
                if len(attempts)>=cfg['resources']['max_attempts']:
                    raise RuntimeError(f'Phase {phase_id} exhausted retries; inspect retained attempt logs')
                if local:
                    attempt=reserve(work,phase_id,0.,limit=0.,max_jobs=max_jobs)
                    try:
                        execute_worker(cfg,phase_id,attempt['path'])
                        complete_attempt(work,attempt,dict(allocated_gpu_hours=0.,scheduler_state='LOCAL_CPU'))
                    except Exception:
                        update_attempt(work,attempt['id'],dict(status='failed',allocated_gpu_hours=0.))
                        raise
                else:
                    _submit(cfg,phase_id)
            if not local:
                time.sleep(min(60,cfg['resources']['poll_seconds']))
        from .evaluation import evaluate_study
        if not (work/'data/admission.json').is_file():
            preparation_inspection(cfg,_state(work)['completed']['prepare']['result']['bundle'])
        return evaluate_study(cfg)


def status(cfg):
    state=_state(cfg['work']);phases=read_json(Path(cfg['work'])/'plan.json')['phases']
    result = dict(work=cfg['work'],fixture=cfg['fixture'],completed_fits=sum(p['phase']!='pretrain' and p['phase_id'] in state['completed'] for p in phases),
                completed_phases=sum(p['phase_id'] in state['completed'] for p in phases),total_phases=len(phases),
                active=[dict(phase=a['phase_id'],job_id=a['job_id'],status=a['status']) for a in state['attempts'] if a['status'] not in {'complete','failed'}],
                failed=[a['phase_id'] for a in state['attempts'] if a['status']=='failed'],
                charged_or_reserved_gpu_hours=sum(a.get('allocated_gpu_hours',a['reserved_gpu_hours']) for a in state['attempts']),
                preparation=state['completed'].get('prepare',{}).get('result'),viewer=str(Path(cfg['work'])/'data/viewer.html'),
                gavd_completed_shards=sum(key.startswith('gavd-') for key in state['completed']),
                report=str(Path(cfg['work'])/'report.md'),report_available=(Path(cfg['work'])/'report.md').is_file())
    if cfg.get('study_kind') == 'jepa_response_followup':
        from .followup import budget_usage
        result.update(study_kind=cfg['study_kind'], parent_work=cfg['followup']['parent_binding']['parent_work'],
                      deadline_utc=cfg['followup']['deadline_utc'], budget_categories=budget_usage(state),
                      admission=state['completed'].get('followup-profile',{}).get('result',{}).get('admission'),
                      parent_bundle=cfg['followup']['parent_binding']['bundle'],
                      viewer=str(Path(cfg['followup']['parent_binding']['parent_work'])/'data/viewer.html'))
        if (Path(cfg['work'])/'followup-status.json').is_file():
            result['execution_result'] = read_json(Path(cfg['work'])/'followup-status.json')
    return result
