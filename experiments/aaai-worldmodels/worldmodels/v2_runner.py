"""Immutable staged runner for medwm-reference-masking-v2.

Real outer fitting requires a matching independently accepted protocol freeze.
Every fit records normal training membership, configuration, and artifact hashes.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits
from .data import sha256_file
from .settings import PROJECT
from . import v2_data
from . import v2_evaluation as ev
from . import v2_models as vm
from .v2_masks import POLICIES
from . import restart_experiment as legacy

METHOD_ID='medwm-reference-masking-v2'
STAGES=('refs','masks','calibration','nuisances','retraining','summarize')
DEFAULT=PROJECT/'artifacts/runs/reference-v2-core-v1'
IMPLEMENTATIONS=('worldmodels/v2_models.py','worldmodels/v2_masks.py','worldmodels/v2_data.py',
                 'worldmodels/v2_evaluation.py','worldmodels/v2_runner.py',
                 'worldmodels/restart_experiment.py','worldmodels/data.py',
                 'worldmodels/pose.py','worldmodels/restart_data.py')

@dataclass(frozen=True)
class RunConfig:
    identity:str='reference-v2-core-v1'
    alphas:tuple[float,...]=(.1,1.,10.,100.)
    mask_steps:tuple[int,...]=(100,200)
    seeds:tuple[int,...]=(27,41,73)
    policies:tuple[str,...]=POLICIES
    inner_folds:int=3
    inner_seed:int=6701
    forecast_reps:int=5000
    primary_reps:int=5000
    pooled_reps:int=2000
    core_seconds:float=3600.
    untrained_probe:bool=True
    partition_seeds:tuple[int,...]=(2702,2703)
    def __post_init__(self):
        if not self.identity or not self.alphas or min(self.alphas)<=0 or not self.mask_steps or min(self.mask_steps)<1:
            raise ValueError('Nonempty identity and valid fitting grids required')
        if not self.seeds or len(set(self.seeds))!=len(self.seeds) or not self.policies or any(p not in POLICIES for p in self.policies):
            raise ValueError('Invalid seed or policy family')
        if self.inner_folds!=3 or min(self.forecast_reps,self.primary_reps,self.pooled_reps)<2 or self.core_seconds<=0:
            raise ValueError('Three inner folds and positive execution/uncertainty budgets required')

def _json_default(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(f'Unsupported JSON value: {type(value)}')

def write_json(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False,default=_json_default)+'\n');tmp.replace(p)

def _hash_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,default=_json_default,separators=(',',':')).encode()).hexdigest()

def verify_freeze(path=None,*,project=None):
    project=Path(project or PROJECT);path=Path(path or project/'docs/reference_v2/protocol_freeze.json')
    if not path.is_file():raise ValueError('Real fitting requires protocol_freeze.json')
    freeze=json.loads(path.read_text())
    if freeze.get('status')!='accepted_before_outer_fitting':raise ValueError('Protocol freeze is not accepted before outer fitting')
    if freeze.get('protocol_sha256')!=sha256_file(project/'docs/reference_v2/protocol.md'):
        raise ValueError('Protocol bytes differ from accepted freeze')
    recorded=freeze.get('implementation_sha256',{})
    for name in IMPLEMENTATIONS:
        if recorded.get(name)!=sha256_file(project/name):raise ValueError(f'Implementation differs from accepted freeze: {name}')
    if not freeze.get('independent_review'):raise ValueError('Protocol freeze lacks independent review record')
    return freeze

def window_errors(x,valid,prediction,*,horizon=8):
    x,v,p=np.asarray(x),np.asarray(valid),np.asarray(prediction,float)
    if x.shape[1:]!=(24,16,2) or v.shape!=x.shape[:-1] or p.shape!=(len(x),8,8,2):
        raise ValueError('Expected common-support24 observations and fixed8x8 forecasts')
    if horizon not in (4,8) or not np.isfinite(p).all():raise ValueError('Finite predictions and declared horizon required')
    target=x[:,16:16+horizon,vm.TARGET_JOINTS]
    observed=v[:,16:16+horizon,vm.TARGET_JOINTS]&v[:,:16,vm.TARGET_JOINTS].any(1)[:,None]
    losses=[]
    for leg in (slice(0,4),slice(4,8)):
        m=np.repeat(observed[:,:,leg,None],2,axis=-1);count=m.sum((1,2,3))
        if np.any(count==0):raise ValueError('Common target support failed for one leg')
        safe=np.where(m,target[:,:,leg],0.)
        losses.append(np.where(m,(safe-p[:,:horizon,leg])**2,0.).sum((1,2,3))/count)
    return np.mean(losses,axis=0)

def basic_predictions(x,valid,times):
    _,anchor,_=vm.causal_features(x[:,:16],valid[:,:16],times[:,:16])
    persistence=np.repeat(anchor[:,None],8,axis=1);velocity=np.zeros_like(anchor)
    last_time=np.repeat(times[:,15,None],8,axis=1).copy()
    for i in range(len(x)):
        for out,j in enumerate(vm.TARGET_JOINTS):
            indexes=np.flatnonzero(valid[i,:16,j])
            if len(indexes):last_time[i,out]=times[i,indexes[-1]]
            if len(indexes)>1:
                a,b=indexes[-2:];velocity[i,out]=(x[i,b,j]-x[i,a,j])/(times[i,b]-times[i,a])
    offsets=times[:,15,None,None]+np.arange(1,9)[None,:,None]*.1-last_time[:,None]
    return dict(persistence=persistence.astype(np.float32),
                damped_velocity=(persistence+.5*offsets[...,None]*velocity[:,None]).astype(np.float32))

def inner_source_splits(metadata,*,n_splits=3,seed=6701):
    meta=metadata[['source_id','group_id','label']].drop_duplicates().sort_values('source_id')
    if not meta.label.eq('normal').all() or meta.groupby('group_id').source_id.nunique().max()!=1:
        raise ValueError('Inner selection requires singleton normal components')
    groups=np.array(sorted(meta.group_id.unique()))
    if len(groups)<n_splits+1:raise ValueError('Insufficient normal groups for inner fitting')
    return [dict(inner_fold=f,train_sources=sorted(meta.loc[meta.group_id.isin(groups[tr]),'source_id']),
                 validation_sources=sorted(meta.loc[meta.group_id.isin(groups[va]),'source_id']))
            for f,(tr,va) in enumerate(KFold(n_splits,shuffle=True,random_state=seed).split(groups))]

class Study:
    """Request-identified observations and cache; no implicit outer selection."""
    def __init__(self,output,x,valid,metadata,times,config=None,*,request=None,test_only=False):
        self.output=Path(output);self.x=np.asarray(x);self.valid=np.asarray(valid)
        self.meta=metadata.copy().reset_index(drop=True);self.times=np.asarray(times,float)
        self.config=config or RunConfig();self.test_only=test_only
        if not test_only and request is None:raise ValueError('Real Study initialization requires a freeze-verified request; call run()')
        if len(self.meta)!=len(self.x) or self.x.shape[1:]!=(24,16,2) or self.valid.shape!=self.x.shape[:-1] or self.times.shape!=(len(x),24):
            raise ValueError('Arrays and metadata must share fixed observation contract')
        if not {'window_id','source_id','sequence_id','label','component_id','outer_fold'}.issubset(self.meta) or self.meta.window_id.duplicated().any():
            raise ValueError('Missing cohort identity or repeated window')
        if self.meta.groupby('component_id').source_id.nunique().max()!=1:
            raise ValueError('Multi-source components require reviewed source-weighting extension')
        if self.meta.groupby('component_id').outer_fold.nunique().max()!=1:raise ValueError('Component crossed outer folds')
        if not v2_data.common_support(self.valid).all():raise ValueError('All arms require common target support')
        if not np.isfinite(self.times).all() or np.any(np.diff(self.times,axis=1)<=0):raise ValueError('Decoder times must increase')
        self.meta['group_id']=self.meta.component_id
        self.sources=self.meta[['source_id','label','component_id','outer_fold']].drop_duplicates().sort_values('source_id').reset_index(drop=True)
        self.source_covariates=ev.source_covariates(self.x,self.valid,self.times,self.meta)
        self.output.mkdir(parents=True,exist_ok=True);self.started=time.monotonic()
        identity=request or dict(identity=self.config.identity,method=METHOD_ID,test_only=bool(test_only),config=asdict(self.config),
            array_sha256=vm._identity_arrays(self.x,self.valid,self.times),cohort_sha256=hashlib.sha256(self.meta.to_csv(index=False).encode()).hexdigest())
        self.request=json.loads(json.dumps(identity,default=_json_default));self.request_hash=_hash_json(self.request)
        request_path=self.output/'request.json'
        if (self.output/'results.json').exists():raise FileExistsError('Completed core runs are immutable')
        if request_path.exists() and json.loads(request_path.read_text())!=self.request:raise ValueError('Resume request changed')
        if not request_path.exists():write_json(request_path,self.request)
        self.checkpoints=self.output/'models';self.checkpoints.mkdir(exist_ok=True);self._memory={}
        ledger=self.output/'model_fit_ledger.json';self.ledger=json.loads(ledger.read_text()) if ledger.exists() else {}
        self.selection_path=self.output/'selection.json';self.selections=json.loads(self.selection_path.read_text()) if self.selection_path.exists() else {}
        self._write_progress('initialized')
    def _budget(self):
        if time.monotonic()-self.started>self.config.core_seconds:
            self._write_progress('resource_limit');raise TimeoutError('Core fitting time budget exhausted')
    def _write_progress(self,status,**extra):
        write_json(self.output/'progress.json',dict(status=status,identity=self.config.identity,request_sha256=self.request_hash,
            seconds_this_invocation=time.monotonic()-self.started,fits=len(getattr(self,'ledger',{})),**extra))
    def indices(self,sources):
        given=set(sources)
        if not given or not given<=set(self.sources.source_id):raise ValueError('Unknown or empty source pool')
        return np.flatnonzero(self.meta.source_id.isin(given))
    def normal_indices(self,sources):
        ii=self.indices(sources)
        if not self.meta.iloc[ii].label.eq('normal').all():raise ValueError('Fitted pool contains abnormal observations')
        return ii
    def _artifact(self,path):
        path=Path(path);side=path.with_name(path.name+'.identity.json')
        if path.exists():
            if not side.exists():raise ValueError(f'Incomplete artifact identity: {path}')
            record=json.loads(side.read_text())
            if record.get('request_sha256')!=self.request_hash or record.get('sha256')!=sha256_file(path):raise ValueError(f'Artifact identity mismatch: {path}')
            return True
        if side.exists():raise ValueError(f'Artifact bytes missing: {path}')
        return False
    def save_npz(self,path,arrays):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if self._artifact(path):return
        tmp=path.with_name(path.name+'.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
        write_json(path.with_name(path.name+'.identity.json'),dict(request_sha256=self.request_hash,sha256=sha256_file(path)))
    def read_npz(self,path):
        if not self._artifact(path):raise FileNotFoundError(path)
        with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
    def _cached_fit(self,identity,role,build):
        self._budget();key=_hash_json(identity);path=self.checkpoints/f'{key}.joblib'
        if key not in self._memory:
            if self._artifact(path):
                checkpoint=joblib.load(path)
                if checkpoint.get('cache_identity')!=identity:raise ValueError('Checkpoint request changed')
            else:
                checkpoint=build();checkpoint.update(cache_identity=identity,request_sha256=self.request_hash)
                tmp=path.with_name(path.name+'.tmp');joblib.dump(checkpoint,tmp);tmp.replace(path)
                write_json(path.with_name(path.name+'.identity.json'),dict(request_sha256=self.request_hash,sha256=sha256_file(path)))
            self._memory[key]=checkpoint
        checkpoint=self._memory[key]
        row=self.ledger.setdefault(key,dict(checkpoint=str(path.relative_to(self.output)),checkpoint_sha256=sha256_file(path),
            identity=identity,training_source_ids=checkpoint['training_source_ids'],roles=[],seconds=checkpoint.get('seconds'),parameter_count=checkpoint.get('parameter_count')))
        if role not in row['roles']:row['roles'].append(role);write_json(self.output/'model_fit_ledger.json',self.ledger)
        return checkpoint
    def fit_new(self,sources,config,role):
        sources=sorted(set(sources));ii=self.normal_indices(sources)
        identity=dict(kind='compact',sources=sources,config=asdict(config))
        return self._cached_fit(identity,role,lambda:vm.fit_forecaster(self.x[ii,:16],self.valid[ii,:16],
            self.times[ii,:16],self.x[ii,16:],self.valid[ii,16:],self.meta.iloc[ii],config))
    def predict_new(self,checkpoint,indexes):
        return vm.predict_forecaster(checkpoint,self.x[indexes,:16],self.valid[indexes,:16],self.times[indexes,:16])
    def fit_probe(self,sources,alpha,role):
        sources=sorted(set(sources));ii=self.normal_indices(sources)
        identity=dict(kind='untrained_encoder_coordinate_probe',sources=sources,alpha=float(alpha),seed=27)
        def build():
            started=time.perf_counter();torch.manual_seed(27);torch.set_num_threads(1)
            f,a,k=vm.causal_features(self.x[ii,:16],self.valid[ii,:16],self.times[ii,:16])
            w=vm.hierarchical_weights(self.meta.iloc[ii]);mu,sd=vm._scale_fit(f,w);network=vm.ResidualMLP(832,32)
            with torch.no_grad():h=network.encoder(torch.as_tensor(((f-mu)/sd).astype(np.float32))).numpy()
            y,m=vm.target_values(self.x[ii,16:],self.valid[ii,16:],a,k);ridge=vm._fit_ridge(h,y,m,w,alpha)
            return dict(kind=identity['kind'],training_source_ids=sources,training_group_ids=sorted(self.meta.iloc[ii].group_id.unique()),
                input_mean=mu,input_scale=sd,state_dict=network.state_dict(),ridge=ridge,
                training_input_sha256=vm._identity_arrays(self.x[ii,:16],self.valid[ii,:16],self.times[ii,:16],y,m),
                diagnostics=vm._representation_health(h,w),seconds=time.perf_counter()-started,
                parameter_count=sum(p.numel() for p in network.parameters())+ridge['coef'].size+ridge['intercept'].size)
        return self._cached_fit(identity,role,build)
    def predict_probe(self,checkpoint,indexes):
        f,a,_=vm.causal_features(self.x[indexes,:16],self.valid[indexes,:16],self.times[indexes,:16])
        model=vm.ResidualMLP(832,32);model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.no_grad():h=model.encoder(torch.as_tensor(((f-checkpoint['input_mean'])/checkpoint['input_scale']).astype(np.float32))).numpy()
        r=checkpoint['ridge'];res=(((h-r['mean'])/r['scale'])@r['coef']+r['intercept']).reshape(-1,8,8,2)
        return (res+a[:,None]).astype(np.float32)
    def fit_legacy(self,sources,arm,role):
        sources=sorted(set(sources));ii=self.normal_indices(sources)
        identity=dict(kind='historical_ridge_grid_refit',sources=sources,arm=arm,alphas=list(self.config.alphas))
        def build():
            started=time.perf_counter();legs=[]
            for leg in (0,1):
                f,_,_=legacy.features(self.x[ii,:16],self.valid[ii,:16],leg,arm,self.meta.iloc[ii].window_id)
                y,m=legacy.targets(self.x[ii],self.valid[ii],leg)
                legs.append(legacy.fit_grid(f,y,m,legacy.hierarchical_weights(self.meta.iloc[ii]),alphas=self.config.alphas))
            return dict(kind=identity['kind'],training_source_ids=sources,arm=arm,legs=legs,
                training_input_sha256=vm._identity_arrays(self.x[ii],self.valid[ii]),seconds=time.perf_counter()-started)
        return self._cached_fit(identity,role,build)
    def predict_legacy(self,checkpoint,indexes,alphas):
        predictions=[]
        for leg,alpha in enumerate(alphas):
            ck=dict(arm=checkpoint['arm'],leg=leg,beta=1.,model=checkpoint['legs'][leg][float(alpha)])
            predictions.append(legacy.predict_checkpoint(ck,self.x[indexes,:16],self.valid[indexes,:16],self.meta.iloc[indexes].window_id))
        return np.concatenate(predictions,axis=2).astype(np.float32)
    def source_error_values(self,indexes,prediction):
        return ev.aggregate_windows(self.meta.iloc[indexes],window_errors(self.x[indexes],self.valid[indexes],prediction))
    def mean_error(self,indexes,prediction):
        return float(self.source_error_values(indexes,prediction).value.mean())
    def choose_alpha(self,sources,mode,role):
        sources=sorted(set(sources));ii=self.normal_indices(sources)
        key=_hash_json(dict(kind='compact_alpha',sources=sources,mode=mode,alphas=self.config.alphas,seed=self.config.inner_seed))
        if key in self.selections:return self.selections[key]
        losses={float(a):[] for a in self.config.alphas};membership=[];source_errors={str(a):[] for a in losses}
        for split in inner_source_splits(self.meta.iloc[ii],seed=self.config.inner_seed):
            vi=self.indices(split['validation_sources']);membership.append(split)
            for alpha in losses:
                if mode=='untrained_probe':
                    ck=self.fit_probe(split['train_sources'],alpha,role+':inner_alpha');pred=self.predict_probe(ck,vi)
                else:
                    cfg=vm.ForecastConfig(mode=mode,ridge_alpha=alpha,seed=27)
                    ck=self.fit_new(split['train_sources'],cfg,role+':inner_alpha');pred=self.predict_new(ck,vi)
                values=self.source_error_values(vi,pred)
                losses[alpha].extend(values.value.tolist());source_errors[str(alpha)].extend(values[['source_id','value']].to_dict('records'))
        means={a:float(np.mean(v)) for a,v in losses.items()};alpha=min(means,key=lambda a:(means[a],-a))
        record=dict(mode=mode,training_sources=sources,selected_alpha=alpha,inner_source_mse={str(a):v for a,v in means.items()},
            membership=membership,validation_source_errors=source_errors,selection='Equal normal-source OOF MSE; exact ties choose larger penalty')
        self.selections[key]=record;write_json(self.selection_path,self.selections);return record
    def choose_legacy(self,sources,arm,role):
        sources=sorted(set(sources));ii=self.normal_indices(sources)
        key=_hash_json(dict(kind='legacy_alpha',sources=sources,arm=arm,alphas=self.config.alphas,seed=self.config.inner_seed))
        if key in self.selections:return self.selections[key]
        losses=[{float(a):[] for a in self.config.alphas} for _ in (0,1)];membership=[]
        source_errors=[{str(a):[] for a in losses[0]} for _ in (0,1)]
        for split in inner_source_splits(self.meta.iloc[ii],seed=self.config.inner_seed):
            vi=self.indices(split['validation_sources']);membership.append(split);ck=self.fit_legacy(split['train_sources'],arm,role+':inner_alpha')
            for leg in (0,1):
                target,mask=legacy.targets(self.x[vi],self.valid[vi],leg)
                f,_,_=legacy.features(self.x[vi,:16],self.valid[vi,:16],leg,arm,self.meta.iloc[vi].window_id)
                for alpha in losses[leg]:
                    pred=legacy.predict(ck['legs'][leg][alpha],f);err=legacy.error_by_window(target,pred,mask)
                    values=ev.aggregate_windows(self.meta.iloc[vi],err)
                    losses[leg][alpha].extend(values.value.tolist());source_errors[leg][str(alpha)].extend(values[['source_id','value']].to_dict('records'))
        means=[{a:float(np.mean(v)) for a,v in loss.items()} for loss in losses]
        selected=[min(v,key=lambda a:(v[a],-a)) for v in means]
        record=dict(mode=arm,training_sources=sources,selected_alphas=selected,
            inner_source_mse=[{str(a):v for a,v in m.items()} for m in means],membership=membership,validation_source_errors=source_errors)
        self.selections[key]=record;write_json(self.selection_path,self.selections);return record
    def fit_selected_ridge(self,sources,mode,role):
        choice=self.choose_alpha(sources,mode,role)
        return self.fit_new(sources,vm.ForecastConfig(mode=mode,ridge_alpha=choice['selected_alpha'],seed=27),role+':refit')
    def choose_steps(self,sources,role):
        sources=sorted(set(sources));ii=self.normal_indices(sources)
        key=_hash_json(dict(kind='mlp_steps',sources=sources,steps=self.config.mask_steps,seed=27,inner_seed=self.config.inner_seed))
        if key in self.selections:return self.selections[key]
        losses={int(s):[] for s in self.config.mask_steps};membership=[];source_errors={str(s):[] for s in losses}
        for split in inner_source_splits(self.meta.iloc[ii],seed=self.config.inner_seed):
            vi=self.indices(split['validation_sources']);membership.append(split)
            for steps in losses:
                cfg=vm.ForecastConfig(mode='mlp',mask_policy='none',steps=steps,seed=27)
                ck=self.fit_new(split['train_sources'],cfg,role+':inner_steps')
                values=self.source_error_values(vi,self.predict_new(ck,vi))
                losses[steps].extend(values.value.tolist());source_errors[str(steps)].extend(values[['source_id','value']].to_dict('records'))
        means={s:float(np.mean(v)) for s,v in losses.items()};selected=min(means,key=lambda s:(means[s],s))
        record=dict(mode='mlp_M0',training_sources=sources,selected_steps=selected,inner_source_mse={str(s):v for s,v in means.items()},
            membership=membership,validation_source_errors=source_errors,selection='Equal normal-source OOF MSE from M0 seed27; ties choose fewer updates')
        self.selections[key]=record;write_json(self.selection_path,self.selections);return record
    def _folds(self):return sorted(int(f) for f in self.sources.outer_fold.unique())
    def _training_sources(self,fold):return sorted(self.sources.loc[(self.sources.outer_fold!=fold)&self.sources.label.eq('normal'),'source_id'])
    def run_refs(self):
        for fold in self._folds():
            path=self.output/'refs'/f'fold_{fold}.npz'
            if self._artifact(path):continue
            self._write_progress('running_refs',outer_fold=fold);train=self._training_sources(fold);ii=np.flatnonzero(self.meta.outer_fold.eq(fold))
            predictions=basic_predictions(self.x[ii],self.valid[ii],self.times[ii])
            for arm in ('own','bilateral','shifted','static'):
                choice=self.choose_legacy(train,arm,f'outer{fold}:ref_{arm}');ck=self.fit_legacy(train,arm,f'outer{fold}:ref_{arm}:refit')
                predictions[arm]=self.predict_legacy(ck,ii,choice['selected_alphas'])
            for mode in ('ridge_posture','ridge_posture_velocity'):
                ck=self.fit_selected_ridge(train,mode,f'outer{fold}:ref_{mode}');predictions[mode]=self.predict_new(ck,ii)
            if self.config.untrained_probe:
                choice=self.choose_alpha(train,'untrained_probe',f'outer{fold}:untrained_probe')
                ck=self.fit_probe(train,choice['selected_alpha'],f'outer{fold}:untrained_probe:refit');predictions['untrained_probe']=self.predict_probe(ck,ii)
            self.save_npz(path,dict(indexes=ii,**predictions))
        self._consolidate_predictions('refs')
    def run_masks(self):
        for fold in self._folds():
            path=self.output/'masks'/f'fold_{fold}.npz'
            if self._artifact(path):continue
            self._write_progress('running_masks',outer_fold=fold);train=self._training_sources(fold);ii=np.flatnonzero(self.meta.outer_fold.eq(fold))
            steps=self.choose_steps(train,f'outer{fold}:mlp')['selected_steps'];predictions={}
            for policy in self.config.policies:
                seed_predictions=[]
                for seed in self.config.seeds:
                    cfg=vm.ForecastConfig(mode='mlp',mask_policy=policy,seed=seed,steps=steps)
                    ck=self.fit_new(train,cfg,f'outer{fold}:mask_{policy}:seed{seed}');pred=self.predict_new(ck,ii)
                    predictions[f'mlp_{policy}_seed{seed}']=pred;seed_predictions.append(pred)
                predictions[f'mlp_{policy}']=np.mean(seed_predictions,axis=0).astype(np.float32)
            if 'motion_balanced' in self.config.policies:
                cfg=vm.ForecastConfig(mode='mlp',mask_policy='motion_balanced',seed=27,steps=steps,pattern_only=True)
                ck=self.fit_new(train,cfg,f'outer{fold}:M4_pattern_only_seed27');predictions['mlp_pattern_only']=self.predict_new(ck,ii)
            self.save_npz(path,dict(indexes=ii,**predictions))
        self._consolidate_predictions('masks')
    def _consolidate_predictions(self,stage):
        output=self.output/f'{stage}_predictions.npz'
        if self._artifact(output):
            full=self.read_npz(output);self._write_forecast_scores(stage,full);return full
        full={};seen=np.zeros(len(self.meta),int)
        for fold in self._folds():
            data=self.read_npz(self.output/stage/f'fold_{fold}.npz');ii=data.pop('indexes');seen[ii]+=1
            for arm,p in data.items():
                if arm not in full:full[arm]=np.zeros((len(self.meta),8,8,2),np.float32)
                full[arm][ii]=p
        if not np.all(seen==1):raise ValueError('Each window needs exactly one held-out forecast')
        self.save_npz(output,full);self._write_forecast_scores(stage,full);return full
    def _write_forecast_scores(self,stage,predictions):
        rows=[];sources=[]
        for arm,pred in predictions.items():
            for horizon in (4,8):
                errors=window_errors(self.x,self.valid,pred,horizon=horizon)
                frame=self.meta[['window_id','source_id','sequence_id','label','component_id','outer_fold']].copy()
                frame['arm'],frame['horizon'],frame['mse']=arm,horizon,errors;rows.append(frame)
                agg=ev.aggregate_windows(self.meta,errors,name='mse');agg['arm'],agg['horizon']=arm,horizon;sources.append(agg)
        pd.concat(rows,ignore_index=True).to_csv(self.output/f'{stage}_window_scores.csv',index=False)
        pd.concat(sources,ignore_index=True).to_csv(self.output/f'{stage}_source_scores.csv',index=False)
    def source_errors(self,indexes,prediction,training_sources):
        frame=ev.aggregate_windows(self.meta.iloc[indexes],window_errors(self.x[indexes],self.valid[indexes],prediction),name='mse')
        frame=frame.merge(self.source_covariates[['source_id','prefix_motion','prefix_missingness']],on='source_id',validate='one_to_one')
        frame['forecast_training_sources']=json.dumps(sorted(training_sources));return frame
    def _calibrate_family(self,mode):
        scores=[];ledgers=[]
        for fold in self._folds():
            rotations=[]
            for schedule in ev.calibration_schedule(self.meta,fold):
                rotation=schedule['rotation'];base=self.output/'calibration'/mode/f'fold_{fold}_rotation_{rotation}'
                score_path=base.with_suffix('.scores.csv');done=base.with_suffix('.done.json')
                if done.exists():
                    record=json.loads(done.read_text())
                    if record['request_sha256']!=self.request_hash:raise ValueError('Calibration resume identity changed')
                    for name,h in record['artifacts'].items():
                        if sha256_file(self.output/name)!=h:raise ValueError('Calibration artifact hash mismatch')
                    rotations.append(pd.read_csv(score_path,keep_default_na=False));ledgers.append(record['ledger']);continue
                self._write_progress('running_calibration',family=mode,outer_fold=fold,rotation=rotation);oof=[]
                for split in schedule['nuisance_oof']:
                    train=split['train_sources'];vi=self.indices(split['validation_sources'])
                    ck=self.fit_selected_ridge(train,mode,f'outer{fold}:{mode}:cal{rotation}:gOOF{split["nuisance_fold"]}')
                    oof.append(self.source_errors(vi,self.predict_new(ck,vi),train))
                oof=pd.concat(oof,ignore_index=True)
                train=schedule['predictor_fit_sources'];ci=self.indices(schedule['calibration_sources']);ti=self.indices(schedule['heldout_sources'])
                ck=self.fit_selected_ridge(train,mode,f'outer{fold}:{mode}:cal{rotation}:R_forecast')
                cal_pred=self.predict_new(ck,ci);test_pred=self.predict_new(ck,ti)
                calibration=self.source_errors(ci,cal_pred,train);test=self.source_errors(ti,test_pred,train)
                calibrator=ev.fit_source_calibrator(oof,calibration,schedule=schedule,alphas=self.config.alphas)
                scored=ev.score_source_calibrator(calibrator,test);scored['one_sided']=scored.cdf
                marginal=ev.empirical_mid_cdf(np.log(calibration.mse.to_numpy()+ev.FLOOR),np.log(test.mse.to_numpy()+ev.FLOOR))
                scored['marginal_two_sided']=np.abs(2*marginal-1);scored['outer_fold'],scored['rotation']=fold,rotation
                paths=[]
                for ending,frame in [('.oof.csv',oof),('.calibration.csv',calibration),('.scores.csv',scored)]:
                    path=base.with_suffix(ending);path.parent.mkdir(parents=True,exist_ok=True);frame.to_csv(path,index=False);paths.append(path)
                cal_path=base.with_suffix('.calibrator.json');write_json(cal_path,calibrator);paths.append(cal_path)
                pred_path=base.with_suffix('.predictions.npz')
                self.save_npz(pred_path,dict(calibration_indexes=ci,calibration_prediction=cal_pred,heldout_indexes=ti,heldout_prediction=test_pred))
                paths.extend([pred_path,pred_path.with_name(pred_path.name+'.identity.json')])
                key=_hash_json(ck['cache_identity']);fit_record=self.ledger[key]
                ledger=dict(family=mode,outer_fold=fold,rotation=rotation,schedule=schedule,
                    forecast_checkpoint=fit_record['checkpoint'],forecast_checkpoint_sha256=fit_record['checkpoint_sha256'],
                    calibrator_path=str(cal_path.relative_to(self.output)),calibrator_sha256=sha256_file(cal_path),
                    nuisance_reference_path=str(base.with_suffix('.oof.csv').relative_to(self.output)),
                    calibration_path=str(base.with_suffix('.calibration.csv').relative_to(self.output)),
                    scores_path=str(score_path.relative_to(self.output)),predictions_path=str(pred_path.relative_to(self.output)),
                    nuisance_training_sources=sorted(oof.source_id),calibration_sources=sorted(calibration.source_id),test_sources=sorted(test.source_id),
                    predictor_selected_alpha=ck['config']['ridge_alpha'],nuisance_selected_alpha=calibrator['nuisance_model']['alpha'])
                write_json(done,dict(request_sha256=self.request_hash,ledger=ledger,artifacts={str(p.relative_to(self.output)):sha256_file(p) for p in paths}))
                rotations.append(scored);ledgers.append(ledger)
            combined=ev.ensemble_rotation_scores(rotations).rename(columns={'two_sided':'calibrated_two_sided'})
            for column in ('one_sided','marginal_two_sided','mse'):
                part=ev.ensemble_rotation_scores(rotations,score=column)
                combined[column if column!='mse' else 'calibration_forecast_mse']=part[column]
            scores.append(combined)
        return pd.concat(scores,ignore_index=True).sort_values('source_id'),ledgers
    def run_calibration(self):
        primary,ledgers=self._calibrate_family('ridge_posture_velocity')
        posture,posture_ledgers=self._calibrate_family('ridge_posture')
        posture=posture.rename(columns={c:'posture_'+c for c in posture if c not in ('source_id','label','component_id','outer_fold')})
        posture=posture.rename(columns={'posture_calibrated_two_sided':'posture_two_sided'})
        combined=primary.merge(posture.drop(columns=['label','component_id','outer_fold']),on='source_id',validate='one_to_one')
        combined.to_csv(self.output/'calibrated_source_scores.csv',index=False)
        write_json(self.output/'calibration_models.json',ledgers)
        write_json(self.output/'posture_calibration_models.json',posture_ledgers)
        self._write_progress('calibration_complete');return combined
    def run_nuisances(self):
        features,_,_=vm.causal_features(self.x[:,:16],self.valid[:,:16],self.times[:,:16],mode='ridge_posture')
        source_features=self.sources.copy();columns=[]
        for j in range(features.shape[1]):
            col=f'posture_{j}';columns.append(col);vals=ev.aggregate_windows(self.meta,features[:,j],name=col)
            source_features=source_features.merge(vals[['source_id',col]],on='source_id',validate='one_to_one')
        frame=self.source_covariates.copy();frame['posture_distance']=0.;frame['motion_quality_distance']=0.
        frame['negative_prefix_motion']=-frame.prefix_motion;frame['high_prefix_missingness']=frame.prefix_missingness
        for fold in self._folds():
            train=self._training_sources(fold);ti=np.flatnonzero(source_features.source_id.isin(train));vi=np.flatnonzero(source_features.outer_fold.eq(fold))
            posture=ev.fit_normal_distance(source_features.iloc[ti][columns],train,shrinkage=.2)
            frame.loc[frame.outer_fold.eq(fold),'posture_distance']=ev.score_normal_distance(posture,source_features.iloc[vi][columns])
            tc=frame[frame.source_id.isin(train)];cov=ev.fit_normal_distance(tc[['log_prefix_motion','prefix_missingness']],train,shrinkage=.2)
            frame.loc[frame.outer_fold.eq(fold),'motion_quality_distance']=ev.score_normal_distance(cov,frame[frame.outer_fold.eq(fold)][['log_prefix_motion','prefix_missingness']])
            write_json(self.output/'nuisance_models'/f'fold_{fold}.json',dict(posture=posture,motion_quality=cov,
                posture_columns=columns,covariate_columns=['log_prefix_motion','prefix_missingness'],normal_training_sources=train))
        frame.to_csv(self.output/'nuisance_source_scores.csv',index=False);source_features.to_csv(self.output/'source_posture_features.csv',index=False)
        return frame
    def run_retraining(self):
        summaries=[];tables=[]
        for seed in self.config.partition_seeds:
            meta=self.meta.copy()
            meta['outer_fold']=v2_data.component_folds(meta,n_splits=len(self._folds()),seed=int(seed))
            predictions={name:np.zeros((len(meta),8,8,2),np.float32) for name in ('persistence','ridge_posture','ridge_posture_velocity')}
            for fold in self._folds():
                path=self.output/'retraining'/f'seed_{seed}_fold_{fold}.npz'
                ii=np.flatnonzero(meta.outer_fold.eq(fold))
                if self._artifact(path):
                    saved=self.read_npz(path)
                    if not np.array_equal(saved.pop('indexes'),ii):raise ValueError('Retraining partition changed')
                else:
                    train=sorted(meta.loc[meta.outer_fold.ne(fold)&meta.label.eq('normal'),'source_id'].unique())
                    saved={'persistence':basic_predictions(self.x[ii],self.valid[ii],self.times[ii])['persistence']}
                    for mode in ('ridge_posture','ridge_posture_velocity'):
                        ck=self.fit_selected_ridge(train,mode,f'partition{seed}:outer{fold}:{mode}')
                        saved[mode]=self.predict_new(ck,ii)
                    self.save_npz(path,dict(indexes=ii,**saved))
                for name,pred in saved.items():predictions[name][ii]=pred
            source=meta[['source_id','label','component_id','outer_fold']].drop_duplicates().sort_values('source_id')
            for name,pred in predictions.items():
                values=ev.aggregate_windows(meta,window_errors(self.x,self.valid,pred),name=name)
                source=source.merge(values[['source_id',name]],on='source_id',validate='one_to_one')
            source['partition_seed']=seed;tables.append(source)
            comparisons=[ev.normal_forecast_contrast(source,'ridge_posture_velocity',reference,
                reps=self.config.forecast_reps,seed=5702) for reference in ('persistence','ridge_posture')]
            summaries.append(dict(partition_seed=seed,contrasts=comparisons,
                note='Complete normal-only refitting on a correlated alternate source partition; not an independent study.'))
        if tables:pd.concat(tables,ignore_index=True).to_csv(self.output/'retraining_source_scores.csv',index=False)
        write_json(self.output/'retraining_summary.json',summaries)
        return summaries
    def summarize(self):
        refs=pd.read_csv(self.output/'refs_source_scores.csv');masks=pd.read_csv(self.output/'masks_source_scores.csv')
        forecast=pd.concat([refs,masks],ignore_index=True);meta_columns=['source_id','label','component_id','outer_fold']
        frame=forecast[forecast.horizon.eq(8)].pivot(index=meta_columns,columns='arm',values='mse').reset_index()
        calibration=pd.read_csv(self.output/'calibrated_source_scores.csv');nuisance=pd.read_csv(self.output/'nuisance_source_scores.csv')
        frame=frame.merge(calibration.drop(columns=['label','component_id','outer_fold']),on='source_id',validate='one_to_one')
        frame=frame.merge(nuisance.drop(columns=['label','component_id','outer_fold']),on='source_id',validate='one_to_one');frame['low_motion']=-frame.persistence
        score_names=['calibrated_two_sided','posture_two_sided','low_motion','one_sided','marginal_two_sided',
            'posture_distance','motion_quality_distance','negative_prefix_motion','high_prefix_missingness']
        forecast_arms=sorted(forecast.arm.unique())
        for arm in forecast_arms:
            if '_seed' in arm:continue
            name=f'excess_{arm}';frame[name]=frame[arm]-frame.persistence;score_names.extend([arm,name])
        frame.to_csv(self.output/'source_scores.csv',index=False)
        frame.to_csv(self.output/'source_anomaly_scores.csv',index=False)
        source_forecasts=frame[meta_columns+forecast_arms]
        primary=ev.paired_within_fold_bootstrap(frame,'calibrated_two_sided','low_motion',reps=self.config.primary_reps,seed=5701)
        posture_comparison=ev.paired_within_fold_bootstrap(frame,'calibrated_two_sided','posture_two_sided',reps=self.config.primary_reps,seed=5701)
        comparisons=[(arm,'persistence') for arm in forecast_arms if arm!='persistence']
        comparisons += [('ridge_posture_velocity','ridge_posture'),('bilateral','own'),('bilateral','shifted'),('bilateral','static')]
        for a,b in [('mlp_anatomical_tubes','mlp_uniform'),('mlp_anatomical_tubes','mlp_random_groups'),
                    ('mlp_motion_balanced','mlp_uniform'),('mlp_motion_balanced','mlp_motion_focused')]:
            if a in frame and b in frame:comparisons.append((a,b))
        contrasts=[ev.normal_forecast_contrast(source_forecasts,a,b,reps=self.config.forecast_reps,seed=5702) for a,b in comparisons]
        normal_means=[]
        for (arm,horizon),group in forecast[forecast.label.eq('normal')].groupby(['arm','horizon']):
            values=group.mse.to_numpy();rng=np.random.default_rng(5702)
            boot=values[rng.integers(len(values),size=(self.config.forecast_reps,len(values)))].mean(1)
            normal_means.append(dict(arm=arm,horizon=int(horizon),mse=float(values.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),normal_sources=len(values)))
        rankings=[]
        for score in score_names:
            self._budget();rankings.append(dict(score=score,within_fold=ev.within_fold_summary(frame,score),
                pooled=ev.pooled_rank_summary(frame,score,reps=self.config.pooled_reps,seed=5704)))
        from sklearn.metrics import roc_auc_score,average_precision_score
        conditions=[]
        for label in sorted(set(frame.label)-{'normal'}):
            part=frame[frame.label.isin(['normal',label])];target=part.label.ne('normal')
            for score in score_names:
                conditions.append(dict(label=label,score=score,normal_sources=int((~target).sum()),abnormal_sources=int(target.sum()),
                    auroc=float(roc_auc_score(target,part[score])),average_precision=float(average_precision_score(target,part[score]))))
        pd.DataFrame(conditions).to_csv(self.output/'condition_scores.csv',index=False);write_json(self.output/'forecast_contrasts.json',contrasts)
        result=dict(identity=self.config.identity,method=METHOD_ID,status='complete',test_only=self.test_only,request_sha256=self.request_hash,
            windows=len(self.meta),sequences=int(self.meta.sequence_id.nunique()),sources=len(self.sources),components=int(self.sources.component_id.nunique()),
            normal_sources=int(self.sources.label.eq('normal').sum()),primary_anomaly=primary,posture_calibration_contrast=posture_comparison,
            normal_forecast=normal_means,forecast_contrasts=contrasts,rankings=rankings,
            retraining_sensitivity=json.loads((self.output/'retraining_summary.json').read_text()) if (self.output/'retraining_summary.json').exists() else [],fits=len(self.ledger),selection_records=len(self.selections),
            seconds_this_invocation=time.monotonic()-self.started,score_direction='Larger anomaly score means more abnormal; prefix motion uses negative direction; MSE/excess/missingness/distance/CDF tails keep positive direction.',
            limitations=['Exploratory GAVD reuse after outcome exposure.','Sources/components do not establish independent participants or clinical diagnoses.',
                'Intervals condition on saved predictions, not retraining variability.','Small calibration groups yield discrete tail ranks, without conformal coverage.',
                'Finite four-bank corruption regime and correlated optimization seeds.'])
        self.meta.to_csv(self.output/'cohort.csv',index=False);write_json(self.output/'results.json',result);self._write_progress('complete')
        write_json(self.output/'artifact_hashes.json',{str(p.relative_to(self.output)):sha256_file(p) for p in sorted(self.output.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'})
        return result
    def run(self,stage='all'):
        selected=STAGES if stage=='all' else (stage,)
        if any(s not in STAGES for s in selected):raise ValueError('Unknown execution stage')
        with threadpool_limits(limits=1):
            for s in selected:
                self._budget();getattr(self,'run_'+s if s!='summarize' else 'summarize')()
        return self.output

def run(output=None,*,cohort=None,config=None,stage='all',freeze=None):
    accepted=verify_freeze(freeze);config=config or RunConfig();cohort=Path(cohort or v2_data.DEFAULT)
    normalized=json.loads(json.dumps(asdict(config),default=_json_default))
    if accepted.get('run_config')!=normalized:raise ValueError('Run configuration differs from accepted protocol freeze')
    for filename,key in [('identity.json','cohort_identity_sha256'),('cohort.csv','cohort_membership_sha256')]:
        if accepted.get(key)!=sha256_file(cohort/filename):
            raise ValueError('Cohort identity or membership differs from accepted protocol freeze')
    x,valid,meta,times=v2_data.load_cohort(cohort)
    request=dict(identity=config.identity,method=METHOD_ID,config=asdict(config),test_only=False,
        cohort_directory=str(cohort.resolve().relative_to(PROJECT)),cohort_identity_sha256=sha256_file(cohort/'identity.json'),
        cohort_membership_sha256=sha256_file(cohort/'cohort.csv'),protocol_sha256=accepted['protocol_sha256'],
        freeze_sha256=sha256_file(freeze or PROJECT/'docs/reference_v2/protocol_freeze.json'),
        implementation_sha256={name:sha256_file(PROJECT/name) for name in IMPLEMENTATIONS},
        legacy_code_sha256=sha256_file(PROJECT/'worldmodels/restart_experiment.py'),
        code_revision=__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=PROJECT,text=True).strip(),
        versions={name:importlib.metadata.version(name) for name in ['numpy','pandas','torch','scikit-learn','joblib']})
    study=Study(output or DEFAULT,x,valid,meta,times,config,request=request);return study.run(stage)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=DEFAULT)
    p.add_argument('--cohort',type=Path,default=v2_data.DEFAULT);p.add_argument('--stage',choices=('all',)+STAGES,default='all')
    p.add_argument('--freeze',type=Path);args=p.parse_args();print(run(args.output,cohort=args.cohort,stage=args.stage,freeze=args.freeze))

if __name__=='__main__':main()
