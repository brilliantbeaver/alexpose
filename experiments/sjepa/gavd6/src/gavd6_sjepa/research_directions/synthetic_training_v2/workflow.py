"""Artifact-first stage graph. Fixture execution never opens confirmation labels."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import pandas as pd
from .config import RunConfig, STAGES
from .contracts import (TrackBundle,atomic_json,digest,sha256_file,code_identity,
                        verify_preservation,array_digest)
from .data import fixture_bundle,normalize_inputs,filter_tracks,shuffled_donors
from ..temporal_gait.contracts import stage_lock

DEPENDENCIES={'audit':(), 'data':('audit',), 'adaptation':('data',), 'information':('data',),
 'direct':('information',),'jepa':('direct',),'evaluate':('jepa',),'optional':('audit',),
 'freeze':('evaluate',),'report':('evaluate','adaptation','optional','freeze')}


def _json(path):return json.loads(Path(path).read_text())

def _safe(value):
    if isinstance(value,dict):return {str(k):_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe(v) for v in value]
    if isinstance(value,np.ndarray):return _safe(value.tolist())
    if isinstance(value,(np.integer,)):return int(value)
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def _repo():return Path(__file__).resolve().parents[4]


def initialize(cfg,repo):
    root=cfg.root;root.mkdir(parents=True,exist_ok=True)
    identity=dict(configuration=cfg.as_dict(),code=code_identity(repo),
                  protocol=sha256_file(repo/'docs/studies/synthetic-training-v2/protocol.md'))
    if cfg.mode=='source':identity['source_manifest']=sha256_file(Path(cfg.bundle)/'manifest.json')
    if cfg.decision_spec:
        from .decisions import load_decision_spec
        spec=load_decision_spec(cfg.decision_spec)
        identity['decision_spec']=sha256_file(cfg.decision_spec)
        identity['calibration_artifact']=spec['calibration_artifact_sha256']
    if cfg.cost_ledger:identity['prior_cost_ledger']=sha256_file(cfg.cost_ledger)
    signature=digest(identity);path=root/'identity.json'
    if path.exists():
        if _json(path)['signature']!=signature:raise ValueError('Run identity changed; use a new unique run ID')
    else:
        atomic_json(path,dict(signature=signature,**identity))
        atomic_json(root/'effective-config.json',cfg.as_dict())
        atomic_json(root/'environment.json',dict(interpreter=sys.executable,platform=platform.platform(),
             python=sys.version,evidence_status='fixture-tested' if cfg.mode=='fixture' else 'source-run',
             torch=__import__('importlib.metadata',fromlist=['version']).version('torch'),
             cuda_verified=False,slurm_job=os.environ.get('SLURM_JOB_ID')))
    return signature


def _receipt(cfg,stage):
    p=cfg.root/'receipts'/f'{stage}.json'
    if not p.exists():raise ValueError(f'Missing prerequisite stage {stage}')
    value=_json(p)
    if value.get('stage')!=stage or value.get('identity')!=_json(cfg.root/'identity.json')['signature']:
        raise ValueError('Receipt identity/stage differs from current run')
    for name,h in value['files'].items():
        if not (cfg.root/name).is_file() or sha256_file(cfg.root/name)!=h:raise ValueError(f'Stale/corrupt stage artifact {name}')
    for dependency in DEPENDENCIES[stage]:_receipt(cfg,dependency)
    return value


def _bundle(cfg):return TrackBundle.load(cfg.root/'data'/'bundle')


def _record_groups(records,seed):
    # Alias-safe independent unit. Keep the original identifier for audit.
    return [{**r,'source_person_id':r['person_id'],'person_id':r['canonical_person_id'],'seed':seed} for r in records]


def _write_prediction(folder,method,seed,pred,bundle):
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/f'{method}-{seed}.npz',prediction=pred,
        timestamps=bundle.inputs['timestamps'],target_xy=bundle.targets['xy'],target_valid=bundle.targets['valid'],
        target_visible=bundle.targets['visible'],eval_scale=bundle.targets['eval_scale'],
        input_xy=bundle.inputs['xy'],input_observed=bundle.inputs['observed'],input_confidence=bundle.inputs['confidence'])
    atomic_json(folder/f'{method}-{seed}.json',dict(method=method,seed=seed,records=_record_groups(bundle.records,seed),evidence_status=bundle.evidence_status))


def reconstruct_metrics(root):
    from .evaluation import evaluate_predictions
    rows=[]
    for path in sorted((Path(root)/'predictions').glob('*.npz')):
        meta=_json(path.with_suffix('.json'))
        with np.load(path,allow_pickle=False) as z:
            rows.append(evaluate_predictions(z['prediction'],dict(xy=z['target_xy'],valid=z['target_valid'],visible=z['target_visible'],eval_scale=z['eval_scale']),
                z['timestamps'],meta['records'],method=meta['method'],evidence_status=meta['evidence_status']))
    if not rows:raise ValueError('No saved predictions; means cannot stand in for evidence')
    return pd.concat(rows,ignore_index=True)


def _fit(cfg,arms,identity,*,resume=False,stage='training'):
    from .runtime import require_torch26,require_haic_runtime
    torch=require_torch26();torch.set_num_threads(1 if cfg.mode=='fixture' else int(os.environ.get('OMP_NUM_THREADS','4')))
    cfg.require_gpu_scope()
    if cfg.device=='cuda':
        import uuid
        atomic_json(cfg.root/'runtime'/f'{stage}-{uuid.uuid4().hex}.json',require_haic_runtime())
    from .models import ModelConfig
    from .training import TrainConfig,fit_arm,predict
    bundle=_bundle(cfg);train=bundle.subset('train');dev=bundle.subset('development')
    train.validate(cfg.held_extractor)
    if not len(dev.records):raise ValueError('Development inputs absent')
    train_x,norm=normalize_inputs(train.inputs);dev_x,dev_norm=normalize_inputs(dev.inputs)
    train_y=dict(xy=norm.apply(train.targets['xy']).astype(np.float32),valid=train.targets['valid'])
    reports=[]
    for seed in cfg.seeds:
        for arm in arms:
            directory=cfg.root/'fits'/f'{arm}-{seed}'
            tc=TrainConfig(updates=cfg.updates,readout_updates=cfg.readout_updates,batch_size=cfg.batch_size,
                seed=int(seed),device=cfg.device,log_every=1 if cfg.mode=='fixture' else 20,checkpoint_every=cfg.updates)
            if cfg.resource_contrast=='equal_total_compute' and arm in {'direct','smoothnet','static'}:
                from dataclasses import replace
                tc=replace(tc,total_seconds_budget=cfg.total_compute_seconds)
            if resume:
                from dataclasses import replace
                checkpoints=sorted(directory.glob('checkpoint-*.pt'))
                if checkpoints:tc=replace(tc,resume_from=str(checkpoints[-1]))
            model=fit_arm(arm,train_x,train_y,ModelConfig(**cfg.model),tc,directory,
                          dict(run=identity,data=array_digest(train.inputs),source=bundle.provenance),
                          donor_indices=shuffled_donors(train.records,int(seed)) if arm=='shuffled_jepa' else None,
                          roles=[r['split'] for r in train.records])
            pred=dev_norm.invert(predict(model,dev_x,batch_size=cfg.batch_size))
            _write_prediction(cfg.root/'predictions',arm,int(seed),pred,dev)
            reports.append({**model.training_report,'arm':arm,'seed':int(seed)})
    return dict(evidence_status=bundle.evidence_status,fits=reports,resource_contrast=cfg.resource_contrast,
                equal_compute_claim=False,reason='Actual cost tables must establish matching; step equality alone does not.')


def _evaluate(cfg):
    from .evaluation import aggregate_metrics,paired_cluster_bootstrap,scientific_gate
    folder=cfg.root/'evaluation';folder.mkdir(exist_ok=True)
    metrics=reconstruct_metrics(cfg.root);metrics.to_csv(folder/'per-window.csv',index=False)
    metric_names=['visible_nle','lower_limb_nle','p95_nle','missing_rate','displacement_nle','ankle_separation_mae','amplitude_error','event_timing_mae_s']
    aggregates=pd.concat([aggregate_metrics(metrics,m) for m in metric_names],ignore_index=True)
    aggregates.to_csv(folder/'per-person-balanced-summary.csv',index=False)
    def complete_mean(x):return float(x.mean()) if np.isfinite(x).all() else np.nan
    strata=['method','extractor','seed','evidence_status','split']
    windows=metrics.groupby(strata+['person_id','motion_id','window_id'])[metric_names].agg(complete_mean)
    motions=windows.groupby(level=strata+['person_id','motion_id']).agg(complete_mean)
    people=motions.groupby(level=strata+['person_id']).agg(complete_mean)
    people.reset_index().to_csv(folder/'per-person.csv',index=False)
    aggregates.groupby(['method','extractor','metric']).value.agg(['count','mean','std']).to_csv(folder/'training-seed-variability.csv')
    # Explicit nuisance/clean strata are supplemental; never replace the primary.
    stratified=[]
    for variant,group in metrics.groupby('variant'):
        a=aggregate_metrics(group);a['variant']=variant;stratified.append(a)
    pd.concat(stratified).to_csv(folder/'nuisance-strata.csv',index=False)
    contrasts=[]
    for (extractor,seed),group in metrics.groupby(['extractor','seed']):
        for comparator in ('coordinate','direct','smoothnet','initialized'):
            if {'paired_jepa',comparator}<=set(group.method):
                for metric in ('visible_nle','displacement_nle','amplitude_error','event_timing_mae_s'):
                    result=paired_cluster_bootstrap(group,'paired_jepa',comparator,metric,draws=cfg.bootstrap_draws,seed=cfg.seed)
                    name=f'{extractor}-{seed}-{comparator}-{metric}.json'
                    atomic_json(folder/name,_safe(result));contrasts.append({**{k:v for k,v in result.items() if k not in {'draws','matched_keys'}},'extractor':extractor,'seed':int(seed),'comparator':comparator,'metric':metric,'artifact':name})
    status=_bundle(cfg).evidence_status
    # Confirmation-level preservation margins and reference annotations are
    # deliberately not invented from fixture or single-seed screens.
    from .decisions import adjudicate_gate_b
    gate=adjudicate_gate_b(metrics,cfg)
    gates=dict(A=dict(status='insufficient_evidence',reason='Augmented-COCO versus synthetic image adaptation is a separate pending branch'),
               B=gate,real_transfer=dict(status='insufficient_evidence',reason='Independent real temporal annotations and held recording evidence absent'),
               personalization=dict(status='fail',reason='Closed on historical development panel;0.0406% retrospective incremental oracle headroom; fresh panel needed to reopen'),
               video=dict(status='insufficient_evidence',reason='Optional incremental-benefit branch not run'))
    atomic_json(folder/'contrasts.json',_safe(contrasts));atomic_json(folder/'gates.json',gates)
    plot_tradeoff(metrics,folder/'accuracy-preservation.png')
    return dict(evidence_status=status,gates=gates,windows=len(metrics),independent_people=metrics.person_id.nunique(),extractors=sorted(metrics.extractor.unique()),seeds=sorted(int(s) for s in metrics.seed.unique()))


def plot_tradeoff(metrics,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .evaluation import aggregate_metrics
    x=aggregate_metrics(metrics,'visible_nle');y=aggregate_metrics(metrics,'displacement_nle')
    joined=x.merge(y,on=['method','split','extractor','seed','evidence_status'],suffixes=('_x','_y'))
    extractors=sorted(joined.extractor.unique());methods=sorted(joined.method.unique())
    fig,axes=plt.subplots(1,len(extractors),figsize=(6*len(extractors),5),squeeze=False)
    colors=plt.get_cmap('tab20');markers=['o','s','^','D','v','P','X','<','>','h','*','p']
    for ax,extractor in zip(axes.flat,extractors):
        for index,method in enumerate(methods):
            values=joined[(joined.extractor==extractor)&(joined.method==method)]
            ax.scatter(values.value_x,values.value_y,s=65,color=colors(index),marker=markers[index%len(markers)],label=method,alpha=.8)
        ax.set(xlabel='Visible coordinate error / box diagonal',ylabel='0.20 s displacement error / box diagonal',title=extractor)
        ax.grid(alpha=.2);ax.margins(.15)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False,fontsize=8)
    fig.suptitle('Accuracy and motion error — '+str(metrics.evidence_status.iloc[0]))
    fig.tight_layout(rect=(0,.16,1,.95));fig.savefig(path,dpi=140);plt.close(fig)


def _stage(cfg,stage,repo,identity,*,resume=False):
    root=cfg.root
    if stage=='audit':
        from .audit import reconstruct_pilot
        result=reconstruct_pilot(repo,root/'audit');result['preservation']=verify_preservation(repo);return result
    if stage=='data':
        data=fixture_bundle(cfg.seed) if cfg.mode=='fixture' else TrackBundle.load(cfg.bundle)
        counts=data.validate(cfg.held_extractor)
        if cfg.mode=='source' and data.evidence_status!='source-run':raise ValueError('Source fitting refuses fixtures')
        (root/'data').mkdir(exist_ok=True);data.save(root/'data'/'bundle')
        _,normalization=normalize_inputs(data.inputs)
        np.savez_compressed(root/'data'/'input-normalization.npz',origin=normalization.origin,scale=normalization.scale)
        atomic_json(root/'data'/'achieved-size.json',counts)
        # Auditable skeleton contact sheet for fixtures/imported track bundles.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,i in zip(axes.flat,np.linspace(0,len(data.records)-1,6,dtype=int)):
            ax.scatter(data.inputs['xy'][i,::8,:,0],-data.inputs['xy'][i,::8,:,1],s=3)
            ax.set_title(data.records[i]['variant']+' / '+data.records[i]['window_id'],fontsize=7);ax.set_aspect('equal')
        fig.suptitle('Input track audit — '+data.evidence_status);fig.tight_layout();fig.savefig(root/'data'/'track-contact-sheet.png');plt.close(fig)
        return dict(evidence_status=data.evidence_status,counts=counts,landmark_gate='synthetic_proxy_only; independent anatomical audit pending')
    if stage=='adaptation':return dict(status='insufficient_evidence',evidence_status='planned',reason='Supporting Gate A pending independent adaptation run: unchanged, replay, augmented-COCO, matched synthetic, pooled synthetic; include gradients, output changes and retention.')
    if stage=='information':
        data=_bundle(cfg);dev=data.subset('development')
        for seed in cfg.seeds:
            _write_prediction(root/'predictions','unchanged',int(seed),dev.inputs['xy'],dev)
            for strength in (0,1,2):_write_prediction(root/'predictions',f'filter{strength}',int(seed),filter_tracks(dev.inputs,strength),dev)
        return dict(evidence_status=data.evidence_status,latency='whole64-sample observed window',filter_strengths=[0,1,2],cadence='pending adequate cycles')
    if stage=='direct':return _fit(cfg,[a for a in cfg.arms if a in {'smoothnet','direct','static','initialized'}],identity,resume=resume,stage=stage)
    if stage=='jepa':return _fit(cfg,[a for a in cfg.arms if a in {'coordinate','ordinary_jepa','paired_jepa','shuffled_jepa'}],identity,resume=resume,stage=stage)
    if stage=='evaluate':return _evaluate(cfg)
    if stage=='optional':return dict(personalization=dict(status='fail',execution='skipped',reason='Historical oracle diagnostic insufficient to justify expansion; new panel required'),video=dict(status='insufficient_evidence',execution='skipped',reason='No incremental-benefit evidence or allocated extension budget'))
    if stage=='freeze':
        gate=_json(root/'evaluation/gates.json')
        result=dict(status='insufficient_evidence',confirmation_opened=False,reason='No qualifying real-development references and calibrated confirmation margins',gates=gate,
                    development_snapshot=dict(run_identity=identity,predictions={p.name:sha256_file(p) for p in (root/'predictions').glob('*')}))
        atomic_json(root/'development-snapshot.json',result);return result
    if stage=='report':return build_report(cfg)
    raise ValueError(f'Unknown stage {stage}')


def build_report(cfg):
    root=cfg.root;metrics=reconstruct_metrics(root)
    from .evaluation import aggregate_metrics
    table=aggregate_metrics(metrics);gates=_json(root/'evaluation/gates.json')
    rows=['# Paired synthetic restoration: executed evidence', '',f'Run `{cfg.run_id}`. Evidence: **{metrics.evidence_status.iloc[0]}**. This run is not a real-transfer or confirmation result.','',
          'The question is whether aligned synthetic supervision improves 2D restoration without erasing timing or side-specific variation, and whether JEPA adds value beyond coordinate learning.','',
          'Each number below is recomputed from per-window saved predictions, target masks, physical timestamps and independent evaluation scales. People are outer sampling groups; render variants and frames are not independent samples.','',
          '| Method | Extractor | Seed | Visible normalized error | Group IDs |','|---|---|---:|---:|---:|']
    for r in table.itertuples():rows.append(f'| {r.method} | {r.extractor} | {r.seed} | {r.value:.8f} | {r.people} |')
    evidence_note='Fixture trajectories are analytic software cases; method ordering is not empirical evidence.' if cfg.mode=='fixture' else 'Source projections are synthetic anatomical proxies; they do not establish real measurement accuracy.'
    rows += ['',f'Errors are Euclidean pixels divided by reference-box diagonal. {evidence_note} Fixture group IDs are analytic trajectories, not sampled people. The reference scale is independent of model predictions, not an independent anatomical annotation. Intervals and paired resampling draws are in `evaluation/`. The plot includes the predeclared filter strengths; no strength is selected for confirmation.','',
             'Method key: `direct` learns coordinate restoration end to end; `coordinate` reconstructs clean coordinates before a separately fitted frozen-encoder readout; `initialized` fits only that readout; `ordinary_jepa` predicts observed-track latent targets; `paired_jepa` uses aligned clean synthetic targets; `shuffled_jepa` changes pretraining pairing only. `smoothnet` is the documented temporal MLP adaptation. `static` removes neighboring coordinates after shared whole-window normalization.','',
             'Motion quantities are 0.20-second displacement, absolute signed ankle-separation error, the RMS amplitude of that demeaned separation, and operational positive-maxima timing. They are not metric stride length, heel strikes or clinical outcomes. Missing event/trajectory support prevents preservation success.','',
             '| Scientific gate | Decision | Reason |','|---|---|---|']
    for k,v in gates.items():rows.append(f"| {k} | {v['status']} | {v['reason']} |")
    contrasts=_json(root/'evaluation/contrasts.json')
    rows+=['','The candidate in each interval below is `paired_jepa`; positive reduction means comparator error minus candidate error, in reference-box-diagonal units. The bootstrap resamples canonical people and nested motions together, retaining render variants. It excludes training and model-selection uncertainty. For fixtures it demonstrates resampling arithmetic over analytic IDs and does not estimate population uncertainty.','',
           '| Extractor | Seed | Comparator | Absolute error reduction | 95% interval |','|---|---:|---|---:|---|']
    for c in contrasts:
        if c['metric']=='visible_nle' and c.get('ci95'):
            rows.append(f"| {c['extractor']} | {c['seed']} | {c['comparator']} | {c['improvement']:.8f} | [{c['ci95'][0]:.8f}, {c['ci95'][1]:.8f}] |")
    supported=metrics.groupby('method').agg(windows=('window_id','size'),amplitude_windows=('amplitude_count',lambda x:int((x>0).sum())),event_windows=('event_count',lambda x:int((x>0).sum())))
    rows+=['', 'Support counts (correlated windows; not independent sample sizes):','', '| Method | Scored windows | Amplitude supported | Event supported |','|---|---:|---:|---:|']
    for method,r in supported.iterrows():rows.append(f'| {method} | {r.windows} | {r.amplitude_windows} | {r.event_windows} |')
    rows += ['','![Accuracy and motion error](evaluation/accuracy-preservation.png)','',
             'The strongest competing explanation for a future gain is clean-target supervision or ordinary denoising. The matched clean-coordinate, initialized-readout and SmoothNet-style controls test that explanation. Low latent loss or rank diagnostics cannot establish useful gait measurement.','',
             ('HAIC runtime and rendered source data remain pending. ' if cfg.mode=='fixture' else 'Source/environment provenance is recorded in this run. ')+
             'Independent landmark-convention references, audited held extractor exposure and real temporal annotations must support their corresponding claims. A GPU source run requires the exact Torch 2.6.0+cu124 environment and explicit all-stage compute scope. This report does not submit jobs.','',
             'Cost: see per-fit `training.json`, stage receipts and environment record. CPU fixture time is not H100 throughput. Equal steps are not equal compute. Extraction/body-model/rendering cost must be imported before any empirical expansion.','']
    (root/'report.md').write_text('\n'.join(rows))
    return dict(report=str(root/'report.md'),evidence_status=metrics.evidence_status.iloc[0],gates=gates)


def run_stage(cfg,stage,repo_root=None,*,resume=False):
    if stage not in STAGES:raise ValueError('Unknown study stage')
    repo=Path(repo_root).resolve() if repo_root else _repo()
    identity=initialize(cfg,repo)
    with stage_lock(cfg.root,'run'):
        if (cfg.root/'receipts'/f'{stage}.json').exists():return _receipt(cfg,stage)['result']
        for dependency in DEPENDENCIES[stage]:_receipt(cfg,dependency)
        owned=set()
        for old in (cfg.root/'receipts').glob('*.json'):
            owned.update(_receipt(cfg,old.stem)['files'])
        start=time.perf_counter()
        from contextlib import nullcontext
        from .runtime import budgeted_gpu_stage
        context=budgeted_gpu_stage(cfg,stage) if cfg.device=='cuda' and stage in {'direct','jepa'} else nullcontext()
        with context:result=_safe(_stage(cfg,stage,repo,identity,resume=resume))
        paths=[p for p in cfg.root.rglob('*') if p.is_file() and str(p.relative_to(cfg.root)) not in owned
               and 'receipts' not in p.parts and 'notebook_runs' not in p.parts and not p.name.endswith('.lock')]
        receipt=dict(stage=stage,identity=identity,result=result,elapsed_seconds=time.perf_counter()-start,
                     files={str(p.relative_to(cfg.root)):sha256_file(p) for p in sorted(paths)})
        atomic_json(cfg.root/'receipts'/f'{stage}.json',receipt)
        return result
