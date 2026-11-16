"""Gated residual-forecast research extension; frozen modules stay unchanged."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from .data import sha256_file
from .masking import forecast_mask, uniform_mask
from .model import ModelConfig, SkeletonJEPA, loss_terms
from .training import TrainConfig, fit, seed_everything, representation_diagnostics
from .workflow import load_prepared

@dataclass(frozen=True)
class RevisionProtocol:
    identity: str = 'normal-articulation-residual-forecast-v1'
    frames: int = 32
    context_frames: int = 16
    seeds: tuple[int,...] = (27,41,73)
    arms: tuple[str,...] = ('direct_det','residual_det','residual_slv')
    pretrain_steps: int = 200
    forecast_steps: int = 300
    batch_size: int = 8
    dim: int = 48
    encoder_layers: int = 2
    learning_rate: float = 3e-4
    variance_weight: float = 1.
    covariance_weight: float = .04
    coordinate_weight: float = 1.
    latent_mse_weight: float = 1.
    kl_weight: float = .01
    latent_dim: int = 8
    prior_samples: int = 16
    scoring_batch_size: int = 8
    horizons: tuple[int,...] = (4,8,16)
    def __post_init__(self):
        if not 1<self.context_frames<self.frames: raise ValueError('Need prefix and future')
        if not self.seeds or len(set(self.seeds))!=len(self.seeds): raise ValueError('Need distinct seeds')
        if set(self.arms)!={'direct_det','residual_det','residual_slv'}: raise ValueError('Retain matched three arms')
        if any(h<1 or h>self.frames-self.context_frames for h in self.horizons): raise ValueError('Invalid horizons')
        if min(self.pretrain_steps,self.forecast_steps)<1 or self.batch_size<2: raise ValueError('Invalid training budget')
        if self.dim%4 or self.prior_samples<2 or self.scoring_batch_size<1: raise ValueError('Invalid model/sampling')

def _write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')

def _json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()

def _code_identity():
    root=Path(__file__).parent
    return {name:sha256_file(root/name) for name in ('revision.py','model.py','training.py','masking.py','pose.py','data.py','workflow.py')}

def last_observed(x,valid,context_frames):
    """Last observed prefix pose, presence mask, and original prefix indexes."""
    if x.ndim!=4 or valid.shape!=x.shape[:-1]: raise ValueError('Expected x[B,T,J,2],valid[B,T,J]')
    anchor=x.new_zeros((len(x),x.shape[2],2))
    known=torch.zeros((len(x),x.shape[2]),dtype=torch.bool,device=x.device)
    last=torch.full_like(known,-1,dtype=torch.long)
    for t in range(context_frames):
        anchor=torch.where(valid[:,t,:,None],x[:,t],anchor)
        last=torch.where(valid[:,t],t,last);known|=valid[:,t]
    return anchor,known,last

def common_support(valid,context_frames):
    target=valid.clone();target[:,:context_frames]=False
    return target & valid[:,:context_frames].any(1)[:,None]

class RevisionJEPA(SkeletonJEPA):
    def __init__(self,config,context_frames,residual=False):
        super().__init__(config);self.context_frames=context_frames;self.residual=residual
        nn.init.zeros_(self.coordinate_head.weight);nn.init.zeros_(self.coordinate_head.bias)
    def forward(self,x,valid,target,*,posterior=False,sample_latent=False,anchor_override=None):
        if target[:,:self.context_frames].any() or (valid[:,self.context_frames:] & ~target[:,self.context_frames:]).any():
            raise ValueError('All observed future tokens must be hidden in forward')
        out=super().forward(x,valid,target,posterior=posterior,sample_latent=sample_latent)
        if self.residual:
            anchor=last_observed(x,valid,self.context_frames)[0] if anchor_override is None else anchor_override
            out['coordinates']=out['coordinates']+anchor[:,None]
        return out

def clone_forecaster(base,p,arm,seed):
    seed_everything(seed+1009)
    config=replace(base.config,latent_dim=p.latent_dim if arm=='residual_slv' else 0)
    model=RevisionJEPA(config,p.context_frames,residual=arm.startswith('residual'))
    mismatch=model.load_state_dict(base.state_dict(),strict=False)
    expected={n for n,_ in model.named_parameters() if n.startswith(('prior.','posterior.','z_projection.'))}
    if set(mismatch.missing_keys)!=expected or mismatch.unexpected_keys: raise ValueError('Pretrained state mismatch')
    nn.init.zeros_(model.coordinate_head.weight);nn.init.zeros_(model.coordinate_head.bias)
    return model

def source_schedule(meta,steps,batch_size,seed):
    """Source, sequence, then window sampling; every arm uses identical rows."""
    if set(meta.label)!={'normal'} or set(meta.split)!={'train'}: raise ValueError('Only normal training records allowed')
    if meta.source_id.nunique()<2: raise ValueError('Need two independent normal training sources')
    rng=np.random.default_rng(seed)
    groups={s:{q:np.flatnonzero((meta.source_id==s)&(meta.sequence_id==q))
               for q in sorted(meta.loc[meta.source_id==s,'sequence_id'].unique())} for s in sorted(meta.source_id.unique())}
    schedule=[]
    for _ in range(steps):
        sources=rng.choice(list(groups),size=min(batch_size,len(groups)),replace=False)
        schedule.append([int(rng.choice(groups[s][rng.choice(list(groups[s]))])) for s in sources])
    return schedule

def train_forecaster(model,x,valid,meta,p,schedule,seed,uniform_pretrain=False):
    if set(meta.label)!={'normal'} or set(meta.split)!={'train'}: raise ValueError('Held-out or abnormal training prohibited')
    seed_everything(seed+2027)
    x,valid=torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(valid,dtype=torch.bool)
    optimizer=torch.optim.AdamW((v for v in model.parameters() if v.requires_grad),lr=p.learning_rate)
    history=[];model.train()
    mask_generator=torch.Generator().manual_seed(seed+102)
    for step,rows in enumerate(schedule):
        if len(set(meta.iloc[rows].source_id))!=len(rows): raise ValueError('Batch repeats a source')
        bx,bv=x[rows],valid[rows]
        target=uniform_mask(bv,.5,generator=mask_generator) if uniform_pretrain else forecast_mask(bv,p.context_frames)
        support=target if uniform_pretrain else common_support(bv,p.context_frames)
        if not support.flatten(1).any(1).all(): raise ValueError('No common forecast support')
        out=model(bx,bv,target,posterior=bool(model.config.latent_dim),sample_latent=bool(model.config.latent_dim))
        terms=loss_terms(model,out,bx,bv,support,variance_weight=p.variance_weight,
            covariance_weight=p.covariance_weight,coordinate_weight=0. if uniform_pretrain else p.coordinate_weight,
            latent_mse_weight=0.,kl_weight=p.kl_weight)
        token_mse=(out['latent']-out['teacher'].detach()).square().mean(-1)
        balanced_mse=torch.stack([token_mse[i][support[i]].mean() for i in range(len(rows))]).mean()
        terms['latent_mse']=balanced_mse
        terms['loss']=terms['loss']+p.latent_mse_weight*balanced_mse
        if not torch.isfinite(terms['loss']): raise FloatingPointError('Nonfinite loss')
        optimizer.zero_grad(set_to_none=True);terms['loss'].backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise FloatingPointError('Nonfinite gradient')
        optimizer.step();model.update_teacher()
        history.append(dict(step=step,**{k:float(v.detach()) for k,v in terms.items()}))
    model.eval();return history

def _control_prefix(x,valid,context_frames,control,seed):
    altered,observed=x.clone(),valid.clone();anchor=last_observed(x,valid,context_frames)[0]
    if control=='static': altered[:,:context_frames]=anchor[:,None]
    elif control=='shuffled':
        rng=np.random.default_rng(seed)
        for i in range(len(x)):
            perm=np.r_[rng.permutation(context_frames-1),context_frames-1]
            altered[i,:context_frames]=x[i,perm];observed[i,:context_frames]=valid[i,perm]
    elif control!='actual': raise ValueError('Unknown prefix control')
    return altered,observed,anchor

@torch.no_grad()
def predict(model,x,valid,p,*,seed,control='actual'):
    model.eval();chunks=[]
    count=p.prior_samples if model.config.latent_dim else 1
    for start in range(0,len(x),p.scoring_batch_size):
        bx=torch.as_tensor(x[start:start+p.scoring_batch_size],dtype=torch.float32)
        bv=torch.as_tensor(valid[start:start+p.scoring_batch_size],dtype=torch.bool)
        altered,observed,anchor=_control_prefix(bx,bv,p.context_frames,control,seed+start)
        target=forecast_mask(observed,p.context_frames)
        with torch.random.fork_rng():
            torch.manual_seed(seed+start)
            draws=[model(altered,observed,target,posterior=False,sample_latent=count>1,
                         anchor_override=anchor)['coordinates'].numpy() for _ in range(count)]
        chunks.append(np.stack(draws))
    samples=np.concatenate(chunks,axis=1)
    return samples.mean(0),samples.var(0),samples

def baselines(x,valid,context_frames,static_pose,sample_fps):
    tx,tv=torch.as_tensor(x,dtype=torch.float32),torch.as_tensor(valid,dtype=torch.bool)
    anchor,known,last=last_observed(tx,tv,context_frames)
    previous=tx.new_zeros(anchor.shape);previous_index=torch.full_like(last,-1)
    for t in range(context_frames):
        use=tv[:,t] & (last>t)
        previous=torch.where(use[...,None],tx[:,t],previous);previous_index=torch.where(use,t,previous_index)
    rates=torch.as_tensor(sample_fps,dtype=torch.float32)
    dt=(last-previous_index).clamp_min(1)/rates[:,None]
    velocity=(anchor-previous)/dt[...,None]
    velocity=torch.where((previous_index>=0)[...,None],velocity,0.)
    future_dt=(torch.arange(x.shape[1])[None,:,None]-last[:,None])/rates[:,None,None]
    return dict(persistence=np.broadcast_to(anchor.numpy()[:,None],x.shape).copy(),
                velocity=(anchor[:,None]+velocity[:,None]*future_dt[...,None]).numpy(),
                population_static=np.broadcast_to(static_pose,x.shape).copy())

def normal_static_pose(x,valid,meta):
    means=[]
    for source in sorted(meta.source_id.unique()):
        rows=np.asarray(meta.source_id==source)
        means.append(np.nanmean(np.where(valid[rows,...,None],x[rows],np.nan),axis=(0,1)))
    return np.nan_to_num(np.nanmean(means,axis=0),nan=0.).astype(np.float32)

def energy_score(samples,y,mask):
    """MC energy score with RMS distance on each window's common observed support."""
    count=mask.sum((1,2))*y.shape[-1]
    def distance(delta):
        squared=np.where(mask[...,None],delta**2,0).sum((1,2,3))
        return np.sqrt(np.divide(squared,count,out=np.full(len(y),np.nan),where=count>0))
    score=np.mean([distance(draw-y) for draw in samples],axis=0);pair=np.zeros(len(y))
    for i in range(len(samples)):
        for j in range(i): pair+=distance(samples[i]-samples[j])
    return score-pair/(len(samples)*(len(samples)-1)) if len(samples)>1 else score

def score_windows(x,valid,meta,prediction,variance,samples,p):
    support=common_support(torch.as_tensor(valid),p.context_frames).numpy()
    output=meta[['source_id','sequence_id','label','split']].copy().reset_index(drop=True)
    error=((prediction-x)**2).mean(-1);uncertainty=variance.mean(-1)
    for horizon in p.horizons:
        interval=slice(p.context_frames,p.context_frames+horizon);mask=support[:,interval];count=mask.sum((1,2))
        for name,values in [('mse',error),('variance',uncertainty)]:
            output[f'{name}_h{horizon}']=np.divide(np.where(mask,values[:,interval],0).sum((1,2)),count,
                out=np.full(len(x),np.nan),where=count>0)
        output[f'energy_h{horizon}']=energy_score(samples[:,:,interval],x[:,interval],mask)
        output[f'horizon_seconds_h{horizon}']=horizon/meta.sample_fps.to_numpy(float)
        output[f'target_count_h{horizon}']=count
    output['context_valid_fraction']=valid[:,:p.context_frames].mean((1,2))
    output['future_valid_fraction']=valid[:,p.context_frames:].mean((1,2))
    output['sample_fps']=meta.sample_fps.to_numpy(float)
    for key in ('prefix_mean_confidence','future_mean_confidence','unresolved_person_fraction'):
        if key in meta: output[key]=meta[key].to_numpy()
    return output

def source_means(windows):
    if windows.groupby('source_id').label.nunique().max()!=1: raise ValueError('Mixed source labels')
    numeric=windows.select_dtypes(include='number').columns.tolist()
    sequence=windows.groupby(['source_id','sequence_id','label','split'],as_index=False)[numeric].mean()
    return sequence.groupby(['source_id','label','split'],as_index=False)[numeric].mean()

def uncertainty_summary(sources,horizon):
    from scipy.stats import rankdata
    error=sources[f'mse_h{horizon}'].to_numpy(float)
    dispersion=sources[f'variance_h{horizon}'].to_numpy(float)
    missing=1-sources.context_valid_fraction.to_numpy(float)
    def risk_curve(score):
        ordered=error[np.argsort(score,kind='stable')]
        # Equal-score uncertainty has no justified ranking. Average all within
        # each tied score block before computing the retained-risk curve.
        order=np.argsort(score,kind='stable');sorted_score=score[order]
        for value in np.unique(sorted_score):
            mask=sorted_score==value;ordered[mask]=ordered[mask].mean()
        risk=np.cumsum(ordered)/np.arange(1,len(ordered)+1)
        return risk
    def aurc(score): return float(risk_curve(score).mean())
    def retained(score):
        curve=risk_curve(score)
        return {str(fraction):dict(mse=float(curve[max(1,int(np.ceil(fraction*len(curve))))-1]),
                                  actual_coverage=max(1,int(np.ceil(fraction*len(curve))))/len(curve))
                for fraction in (.5,.75,1.)}
    def correlation(a,b):
        if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12: return None
        return float(np.corrcoef(a,b)[0,1])
    nuisance=[missing,sources.sample_fps.to_numpy(float)]
    if 'prefix_mean_confidence' in sources: nuisance.append(1-sources.prefix_mean_confidence.to_numpy(float))
    nuisance.append(1-sources.future_valid_fraction.to_numpy(float))
    if 'future_mean_confidence' in sources: nuisance.append(1-sources.future_mean_confidence.to_numpy(float))
    design=np.column_stack([np.ones(len(error))]+[rankdata(v) for v in nuisance])
    u,e=rankdata(dispersion),rankdata(error)
    ur=u-design@np.linalg.lstsq(design,u,rcond=None)[0]
    er=e-design@np.linalg.lstsq(design,e,rcond=None)[0]
    return dict(n_sources=len(error),retained_risk=dict(uncertainty=retained(dispersion),prefix_missingness=retained(missing),
                prefix_confidence=retained(1-sources.prefix_mean_confidence.to_numpy(float)) if 'prefix_mean_confidence' in sources else None),aurc_uncertainty=aurc(dispersion),aurc_prefix_missingness=aurc(missing),
                aurc_prefix_confidence=aurc(1-sources.prefix_mean_confidence.to_numpy(float)) if 'prefix_mean_confidence' in sources else None,
                uncertainty_error_spearman=correlation(u,e),partial_rank_correlation=correlation(ur,er) if len(error)-np.linalg.matrix_rank(design)>=3 else None,
                partial_rank_residual_dof=int(len(error)-np.linalg.matrix_rank(design)),
                partial_correlation_is_descriptive=True,nuisance_design_rank=int(np.linalg.matrix_rank(design)),
                confidence_fields_present=('prefix_mean_confidence' in sources and 'future_mean_confidence' in sources))

def _save_model(model,path,metadata):
    torch.save(dict(config=asdict(model.config),context_frames=model.context_frames,residual=model.residual,
                    state=model.state_dict(),metadata=metadata),path)

def load_revision_checkpoint(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    saved['config']['joint_ids']=tuple(saved['config']['joint_ids'])
    model=RevisionJEPA(ModelConfig(**saved['config']),saved['context_frames'],saved['residual'])
    model.load_state_dict(saved['state']);model.eval();return model

def _score_arm(model,x,valid,meta,p,seed,name,output,controls=True):
    frames={}
    for control in ('actual','static','shuffled') if controls else ('actual',):
        prediction,variance,samples=predict(model,x,valid,p,seed=seed+7001,control=control)
        if not all(np.isfinite(a).all() for a in (prediction,variance,samples)): raise FloatingPointError('Nonfinite inference output')
        windows=score_windows(x,valid,meta,prediction,variance,samples,p)
        sources=source_means(windows)
        windows.to_csv(output/f'{name}_{control}_windows.csv',index=False)
        sources.to_csv(output/f'{name}_{control}_sources.csv',index=False)
        frames[control]=sources
    return frames

def _filtered_subset(x,valid,meta,p,mask):
    x,valid,meta=x[mask],valid[mask],meta.loc[mask].reset_index(drop=True)
    support=common_support(torch.as_tensor(valid),p.context_frames).numpy()
    eligible=np.ones(len(x),bool)
    for h in p.horizons: eligible &= support[:,p.context_frames:p.context_frames+h].any((1,2))
    dropped=meta.loc[~eligible,['source_id','sequence_id','label','split']].to_dict('records')
    return x[eligible],valid[eligible],meta.loc[eligible].reset_index(drop=True),dropped

def _load_inputs(input_dir,p):
    x,valid,meta=load_prepared(input_dir)
    if x.shape[1]!=p.frames: raise ValueError('Prepared frames do not match protocol')
    prep=json.loads((Path(input_dir)/'preparation.json').read_text())
    if prep.get('context_frames')!=p.context_frames: raise ValueError('Normalization prefix differs from forecasting prefix')
    if np.any(~np.isfinite(meta.sample_fps) | (meta.sample_fps<=0)): raise ValueError('Missing sampled frame rates')
    return x,valid,meta

def hypothesis_summary(records):
    """Prespecified scientific criteria; negative findings never remove arms."""
    deterministic={r['seed']:r for r in records if r['arm']=='residual_det'}
    stochastic={r['seed']:r for r in records if r['arm']=='residual_slv'}
    details=[]
    for seed,det in deterministic.items():
        slv=stochastic[seed]
        ratio=slv['normal_mse']/max(det['normal_mse'],1e-12)
        uncertainty=slv.get('uncertainty',slv.get('normal_uncertainty'))
        details.append(dict(seed=seed,energy_gain=det['normal_energy']-slv['normal_energy'],
            mse_ratio=ratio,mse_within_10pct=ratio<=1.1,
            aurc_gain_over_prefix_missingness=uncertainty['aurc_prefix_missingness']-uncertainty['aurc_uncertainty'],
            aurc_gain_over_prefix_confidence=(uncertainty['aurc_prefix_confidence']-uncertainty['aurc_uncertainty'])
                if uncertainty['aurc_prefix_confidence'] is not None else None,
            static_context_penalty=det['normal_control_mse']['static']-det['normal_control_mse']['actual'],
            shuffled_context_penalty=det['normal_control_mse']['shuffled']-det['normal_control_mse']['actual']))
    return dict(by_seed=details,mean_energy_gain=float(np.mean([d['energy_gain'] for d in details])),
        all_seed_mse_within_10pct=all(d['mse_within_10pct'] for d in details),
        interpretation='Criteria limit claims; all declared arms remain in final evaluation.')


def develop(input_dir,output_dir,protocol=None):
    """Fit train-only models and score validation only; never compute test outputs."""
    p=protocol or RevisionProtocol();output=Path(output_dir)
    if output.exists(): raise FileExistsError('Use a new revision development directory')
    x,valid,meta=_load_inputs(input_dir,p)
    trainmask=(meta.split=='train')&(meta.label=='normal');valmask=meta.split=='validation'
    tx,tv,tm,train_drops=_filtered_subset(x,valid,meta,p,trainmask.to_numpy())
    vx,vv,vm,val_drops=_filtered_subset(x,valid,meta,p,valmask.to_numpy())
    if tm.source_id.nunique()<2 or vm.loc[vm.label=='normal','source_id'].nunique()<2:
        raise ValueError('Need at least two normal sources in training and validation')
    output.mkdir(parents=True);torch.set_num_threads(2);started=time.monotonic()
    _write(output/'protocol.json',asdict(p));_write(output/'attrition.json',dict(train=train_drops,validation=val_drops))
    static=normal_static_pose(tx,tv,tm);np.save(output/'normal_static_pose.npy',static)
    baseline_predictions=baselines(vx,vv,p.context_frames,static,vm.sample_fps.to_numpy())
    baseline_predictions['untrained_direct']=np.zeros_like(vx)
    baseline_predictions['untrained_residual']=baseline_predictions['persistence']
    base_sources={}
    for name,pred in baseline_predictions.items():
        windows=score_windows(vx,vv,vm,pred,np.zeros_like(pred),pred[None],p)
        sources=source_means(windows);base_sources[name]=sources
        windows.to_csv(output/f'{name}_validation_windows.csv',index=False)
        sources.to_csv(output/f'{name}_validation_sources.csv',index=False)
    code=_code_identity();protocol_document=Path(__file__).parents[1]/'docs/revision_protocol.md'
    protocol_document_sha256=sha256_file(protocol_document);records=[];h=max(p.horizons)
    for seed in p.seeds:
        seed_dir=output/f'seed_{seed}';seed_dir.mkdir();seed_everything(seed)
        base=SkeletonJEPA(ModelConfig(frames=p.frames,dim=p.dim,encoder_layers=p.encoder_layers))
        pre_schedule=source_schedule(tm,p.pretrain_steps,p.batch_size,seed+101)
        _write(seed_dir/'pretraining_schedule.json',pre_schedule)
        prehistory=train_forecaster(base,tx,tv,tm,p,pre_schedule,seed,uniform_pretrain=True)
        base.run_metadata=dict(seed=seed,regime='normal_uniform_pretraining',steps=p.pretrain_steps,
            train_sources=sorted(map(str,tm.source_id.unique())),source_schedule_sha256=_json_hash(pre_schedule),
            objective='CE + source-balanced latent MSE + variance/covariance; coordinate weight zero')
        _write(seed_dir/'pretraining_history.json',prehistory)
        torch.save(dict(state=base.state_dict(),config=asdict(base.config),metadata=base.run_metadata),seed_dir/'uniform_pretrain.pt')
        schedule=source_schedule(tm,p.forecast_steps,p.batch_size,seed+401)
        _write(seed_dir/'source_schedule.json',schedule)
        source_rows=tm.reset_index().groupby('source_id',sort=True).head(1)['index'].to_numpy()
        for arm in p.arms:
            model=clone_forecaster(base,p,arm,seed)
            stage_start=time.monotonic()
            history=train_forecaster(model,tx,tv,tm,p,schedule,seed)
            training_seconds=time.monotonic()-stage_start
            _write(seed_dir/f'{arm}_history.json',history)
            _save_model(model,seed_dir/f'{arm}.pt',dict(protocol=asdict(p),seed=seed,arm=arm,
                source_schedule_sha256=_json_hash(schedule),code_sha256=code,
                train_sources=sorted(map(str,tm.source_id.unique())),normal_pretrain_sha256=sha256_file(seed_dir/'uniform_pretrain.pt')))
            score_start=time.monotonic()
            frames=_score_arm(model,vx,vv,vm,p,seed,arm,seed_dir)
            scoring_seconds=time.monotonic()-score_start
            normal=frames['actual'].query("label == 'normal'").set_index('source_id')
            baseline=base_sources['persistence'].query("label == 'normal'").set_index('source_id').loc[normal.index]
            gain=baseline[f'mse_h{h}']-normal[f'mse_h{h}']
            control_means={c:float(frame.query("label == 'normal'")[f'mse_h{h}'].mean()) for c,frame in frames.items()}
            row=dict(seed=seed,arm=arm,training_seconds=training_seconds,scoring_seconds=scoring_seconds,parameters_total=sum(v.numel() for v in model.parameters()),parameters_trainable=sum(v.numel() for v in model.parameters() if v.requires_grad),normal_validation_sources=len(normal),normal_mse=float(normal[f'mse_h{h}'].mean()),
                normal_energy=float(normal[f'energy_h{h}'].mean()),persistence_gain=float(gain.mean()),
                normal_control_mse=control_means,uncertainty=uncertainty_summary(normal.reset_index(),h),
                representation_diagnostics=representation_diagnostics(model,tx[source_rows],tv[source_rows]))
            records.append(row);_write(output/'progress.json',dict(completed=records,elapsed_seconds=time.monotonic()-started))
    residual=[r for r in records if r['arm']=='residual_det'];gains=[r['persistence_gain'] for r in residual]
    required=int(np.ceil(2*len(p.seeds)/3))
    gate=dict(passed=bool(np.isfinite(gains).all() and np.mean(gains)>0 and sum(g>0 for g in gains)>=required),
        gains_by_seed=gains,mean_gain=float(np.mean(gains)),positive_seeds=sum(g>0 for g in gains),required_positive_seeds=required)
    identity=dict(input_dir=str(Path(input_dir).resolve()),windows_sha256=sha256_file(Path(input_dir)/'windows.npz'),
                  metadata_sha256=sha256_file(Path(input_dir)/'windows.csv'),preparation_sha256=sha256_file(Path(input_dir)/'preparation.json'))
    summary=dict(status='development_complete_test_unexamined',protocol=asdict(p),inputs=identity,records=records,
        validation_gate=gate,hypotheses=hypothesis_summary(records),code_sha256=code,protocol_document_sha256=protocol_document_sha256,seconds=time.monotonic()-started,
        versions={n:importlib.metadata.version(n) for n in ('torch','numpy','pandas','scikit-learn','scipy')},
        train_sources=sorted(map(str,tm.source_id.unique())),validation_sources=sorted(map(str,vm.source_id.unique())),
        fixed_frame_task=True,seconds_depend_on_sample_fps=True,
        artifact_sha256={str(path.relative_to(output)):sha256_file(path) for path in output.rglob('*') if path.is_file()})
    _write(output/'development.json',summary)
    return summary

def freeze(development_dir):
    """Seal all complete arms, retaining failed scientific criteria as evidence."""
    directory=Path(development_dir);path=directory/'freeze.json'
    if path.exists(): raise FileExistsError('A freeze decision already exists')
    record=json.loads((directory/'development.json').read_text())
    if record['code_sha256']!=_code_identity(): raise ValueError('Code changed after development')
    for package,version in record['versions'].items():
        if importlib.metadata.version(package)!=version: raise ValueError(f'Changed runtime: {package}')
    if record['protocol_document_sha256']!=sha256_file(Path(__file__).parents[1]/'docs/revision_protocol.md'): raise ValueError('Protocol document changed after development')
    for name,digest in record['artifact_sha256'].items():
        if sha256_file(directory/name)!=digest: raise ValueError(f'Changed artifact: {name}')
    expected={(seed,arm) for seed in record['protocol']['seeds'] for arm in record['protocol']['arms']}
    actual={(row['seed'],row['arm']) for row in record['records']}
    if actual!=expected or len(record['records'])!=len(expected): raise ValueError('Incomplete seed/arm execution')
    if not all(np.isfinite(row[key]) for row in record['records'] for key in ('normal_mse','normal_energy','persistence_gain')):
        raise ValueError('Nonfinite development outcomes')
    if set(record['train_sources']) & set(record['validation_sources']): raise ValueError('Source leakage')
    decision=dict(status='accepted',scientific_h1_supported=record['validation_gate']['passed'],
        development_sha256=sha256_file(directory/'development.json'),protocol_sha256=_json_hash(record['protocol']),
        code_sha256=record['code_sha256'],validation_gate=record['validation_gate'],
        selection='All declared arms retained; residual_det versus persistence is primary')
    _write(path,decision);return decision

def paired_gain_summary(gains,seed=2027,repetitions=2000):
    values=np.asarray(gains,float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError('Need finite paired source gains')
    rng=np.random.default_rng(seed)
    means=np.array([rng.choice(values,size=len(values),replace=True).mean() for _ in range(repetitions)])
    return dict(mean_gain=float(values.mean()),interval95=list(map(float,np.quantile(means,[.025,.975]))),
                n_sources=len(values),unit='source after averaging model seeds',repetitions=repetitions)

def evaluate_once(development_dir,output_dir):
    """Open held-out outcomes only after an accepted, immutable validation gate."""
    directory=Path(development_dir);output=Path(output_dir)
    decision=json.loads((directory/'freeze.json').read_text())
    if decision['status']!='accepted': raise ValueError('Execution/provenance gate failed; final test remains sealed')
    if decision['code_sha256']!=_code_identity(): raise ValueError('Code changed after freeze')
    if decision['development_sha256']!=sha256_file(directory/'development.json'): raise ValueError('Changed development')
    if (directory/'test_access.json').exists(): raise FileExistsError('Final test already accessed')
    if output.exists(): raise FileExistsError('Use a new evaluation directory')
    record=json.loads((directory/'development.json').read_text())
    for package,version in record['versions'].items():
        if importlib.metadata.version(package)!=version: raise ValueError(f'Changed runtime: {package}')
    if record['protocol_document_sha256']!=sha256_file(Path(__file__).parents[1]/'docs/revision_protocol.md'): raise ValueError('Protocol document changed after freeze')
    for name,digest in record['artifact_sha256'].items():
        if sha256_file(directory/name)!=digest: raise ValueError(f'Changed frozen artifact: {name}')
    p=RevisionProtocol(**record['protocol']);input_dir=Path(record['inputs']['input_dir'])
    for name,key in [('windows.npz','windows_sha256'),('windows.csv','metadata_sha256'),('preparation.json','preparation_sha256')]:
        if sha256_file(input_dir/name)!=record['inputs'][key]: raise ValueError('Changed evaluation data')
    x,valid,meta=_load_inputs(input_dir,p)
    sx,sv,sm,dropped=_filtered_subset(x,valid,meta,p,(meta.split=='test').to_numpy())
    if not len(sx): raise ValueError('No eligible final-test windows')
    output.mkdir(parents=True);torch.set_num_threads(2)
    _write(directory/'test_access.json',dict(status='started',output=str(output.resolve()),freeze_sha256=sha256_file(directory/'freeze.json')))
    _write(output/'attrition.json',dropped)
    static=np.load(directory/'normal_static_pose.npy',allow_pickle=False)
    base=baselines(sx,sv,p.context_frames,static,sm.sample_fps.to_numpy());sources={}
    base['untrained_direct']=np.zeros_like(sx);base['untrained_residual']=base['persistence']
    for name,pred in base.items():
        windows=score_windows(sx,sv,sm,pred,np.zeros_like(pred),pred[None],p)
        frame=source_means(windows);sources[name]=frame
        frame.to_csv(output/f'{name}_sources.csv',index=False)
    all_records=[];h=max(p.horizons);paired=[]
    for seed in p.seeds:
        seed_dir=output/f'seed_{seed}';seed_dir.mkdir()
        for arm in p.arms:
            model=load_revision_checkpoint(directory/f'seed_{seed}'/f'{arm}.pt')
            frames=_score_arm(model,sx,sv,sm,p,seed,arm,seed_dir)
            actual=frames['actual'];normal=actual.query("label == 'normal'").set_index('source_id')
            persistence=sources['persistence'].set_index('source_id').loc[normal.index]
            gain=persistence[f'mse_h{h}']-normal[f'mse_h{h}']
            if arm=='residual_det': paired.append(pd.DataFrame(dict(source_id=normal.index,gain=gain.to_numpy(),seed=seed)))
            all_records.append(dict(seed=seed,arm=arm,normal_mse=float(normal[f'mse_h{h}'].mean()),
                normal_energy=float(normal[f'energy_h{h}'].mean()),normal_persistence_gain=float(gain.mean()),
                normal_control_mse={c:float(f.query("label == 'normal'")[f'mse_h{h}'].mean()) for c,f in frames.items()},
                normal_uncertainty=uncertainty_summary(normal.reset_index(),h),
                all_source_uncertainty=uncertainty_summary(actual,h)))
    pooled=pd.concat(paired).groupby('source_id').gain.mean()
    result=dict(status='final_test_complete',protocol=record['protocol'],frozen_development_sha256=decision['development_sha256'],
        records=all_records,hypotheses=hypothesis_summary(all_records),primary=paired_gain_summary(pooled.to_numpy()),n_test_sources=int(sm.source_id.nunique()),
        normal_test_sources=int(sm.loc[sm.label=='normal','source_id'].nunique()),
        limitations=['Source bootstrap conditions on three fixed trained seeds.',
            'Relative 2D articulation forecasts do not establish physical-world or clinical validity.',
            'Native-frame sampling gives different time horizons across sources.'])
    _write(output/'results.json',result)
    _write(directory/'test_access.json',dict(status='complete',output=str(output.resolve()),result_sha256=sha256_file(output/'results.json')))
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    dev=sub.add_parser('develop');dev.add_argument('--input',required=True);dev.add_argument('--output',required=True)
    dev.add_argument('--protocol',help='JSON protocol, otherwise fixed defaults')
    seal=sub.add_parser('freeze');seal.add_argument('--development',required=True)
    test=sub.add_parser('evaluate');test.add_argument('--development',required=True);test.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    if args.command=='develop':
        p=RevisionProtocol(**json.loads(Path(args.protocol).read_text())) if args.protocol else RevisionProtocol()
        result=develop(args.input,args.output,p)
        print(json.dumps(dict(status=result['status'],validation_gate=result['validation_gate'])))
    elif args.command=='freeze': print(json.dumps(freeze(args.development)))
    else: print(json.dumps(evaluate_once(args.development,args.output)['primary']))

if __name__=='__main__': main()
