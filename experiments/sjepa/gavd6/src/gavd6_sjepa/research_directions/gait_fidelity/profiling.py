"""Measure representative H100 work before selecting a common update budget."""
from __future__ import annotations
import copy
import math
from pathlib import Path
import time

from .common import atomic_json,read_json,sha256


def select_budget(cfg, timings, spent_gpu_hours):
    """Use timing only, never model ranking, for the prespecified fallback."""
    def project(name,updates,default_updates):
        measured=timings[name]
        if isinstance(measured,dict):
            # Hashing the cohort, preparing donor tables and exporting the
            # complete development population happen once per phase.
            return measured['fixed_seconds']+measured['optimization_seconds']*updates/measured['profile_updates']
        return float(measured)*updates/default_updates  # historical timing-only fixtures
    from .spec import build_plan
    plan=build_plan(cfg.get('seeds',[17,29,43]),cfg.get('experiment_set','full'))
    counts={phase:sum(p['phase']==phase for p in plan['phases']) for phase in ('pretrain','readout','end_to_end')}
    choices=[]
    for divisor in (1,2):
        updates={k:max(1,cfg['training'][k]//divisor) for k in
                 ('pretraining_updates','readout_updates','end_to_end_updates')}
        phases=dict(pretrain=max(project(n,updates['pretraining_updates'],20) for n in ('coordinate_pretrain','jepa_pretrain')),
                    readout=project('readout',updates['readout_updates'],20),
                    end_to_end=max(project(n,updates['end_to_end_updates'],40) for n in ('direct','refiner') if n in timings))
        projected=sum(counts[key]*phases[key] for key in counts)/3600
        reserve=float(cfg['resources'].get('evaluation_reserve_gpu_hours',60))
        fits=spent_gpu_hours+projected+reserve<=cfg['resources']['gpu_hours']
        fits=bool(fits and max(phases.values())<=cfg['resources']['phase_wall_minutes']*60*.9)
        choices.append(dict(divisor=divisor,updates=updates,projected_training_gpu_hours=projected,
                            projected_phase_seconds=phases,fits=fits))
    selected=next((c for c in choices if c['fits']),None)
    return dict(status='PROFILE_BUDGET_SELECTED' if selected else 'PROFILE_EXCEEDS_ALLOWANCE',
                selected=selected,alternatives=choices,charged_before_profile=spent_gpu_hours,
                phase_counts=counts,
                criterion='Runtime only; preserve every frozen recipe and seed; no metric-ranking input',
                limitation='Fixed cohort hashing/export overhead is counted once; measured optimization time scales with updates. Longer-run throughput and memory can differ.')


def profile_training(bundle,cfg,output,spent_gpu_hours,*,worker_setup_seconds=0.):
    from .training import train_phase
    if not math.isfinite(worker_setup_seconds) or worker_setup_seconds < 0:
        raise ValueError('Worker setup time must be finite and nonnegative')
    profile_started=time.monotonic()
    output=Path(output);output.mkdir(parents=True)
    local=copy.deepcopy(cfg)
    local['training'].update(pretraining_updates=20,readout_updates=20,end_to_end_updates=40)
    recipes={r['recipe_id']:r for r in read_json(Path(cfg['work'])/'plan.json')['recipes']}
    timings={};receipts={}
    def measure(name,recipe,phase,parent=None):
        start=time.monotonic()
        value=train_phase(bundle,recipes[recipe],phase,17,local,output/name,parent)
        wall=time.monotonic()-start
        optimization=min(wall,float(value['elapsed_seconds']))
        hash_started=time.monotonic()
        for path in sorted((output/name).rglob('*')):
            if path.is_file(): sha256(path)
        artifact_hash_seconds=time.monotonic()-hash_started
        timings[name]=dict(wall_seconds=wall,optimization_seconds=optimization,
                           train_phase_fixed_seconds=max(0.,wall-optimization),
                           artifact_hash_seconds=artifact_hash_seconds,
                           profile_updates=int(value['updates']))
        receipts[name]=value
        return Path(value['checkpoint'])
    coordinate=measure('coordinate_pretrain','M-coordinate-graph_time-base','pretrain')
    measure('jepa_pretrain','M-paired_jepa-graph_time-base','pretrain')
    measure('readout','M-coordinate-graph_time-paired_change','readout',coordinate)
    measure('direct','P-direct-none-paired_change','end_to_end')
    if 'P-temporal_refiner-none-base' in recipes:
        measure('refiner','P-temporal_refiner-none-base','end_to_end')
    # Final workers verify the complete profile artifact tree, not just their
    # selected timing row. Measure that recurring read before admitting fits.
    hash_started=time.monotonic()
    for path in sorted(output.rglob('*')):
        if path.is_file(): sha256(path)
    profile_verification_seconds=time.monotonic()-hash_started
    for measured in timings.values():
        # Direct/pretraining workers check preparation both as a dependency
        # and as the dataset authority. Twice the measured verification/load
        # setup is conservative and also covers the readout parent check.
        measured['worker_setup_allowance_seconds']=2*worker_setup_seconds
        measured['profile_verification_seconds']=profile_verification_seconds
        measured['fixed_seconds']=(measured['train_phase_fixed_seconds']+
            measured['artifact_hash_seconds']+measured['worker_setup_allowance_seconds']+
            profile_verification_seconds)
    profile_elapsed_seconds=time.monotonic()-profile_started
    measured_profile_gpu_hours=(worker_setup_seconds+profile_elapsed_seconds)/3600
    result=select_budget(cfg,timings,spent_gpu_hours+measured_profile_gpu_hours)
    result.update(timings_seconds=timings,profile_receipts=receipts,
                  measured_worker_setup_seconds=worker_setup_seconds,
                  measured_profile_seconds=profile_elapsed_seconds,
                  measured_profile_gpu_hours=measured_profile_gpu_hours,
                  profile_verification_seconds=profile_verification_seconds,
                  elapsed_basis='Measured update-loop time scales with updates; recurring worker setup, artifact verification and exports are fixed per phase',
                  allocation_basis='Profile elapsed time plus its measured worker setup is charged once at admission; final Slurm allocation accounting remains authoritative, including publication and process overhead',
                  checkpoint_reuse='Profile checkpoints are never reused by final model phases')
    atomic_json(output/'profile.json',result)
    # An over-budget profile completes as a measurement; the coordinator stops
    # before all final fits without throwing away the allocation receipt.
    return dict(profile=str(output/'profile.json'),budget=result)
