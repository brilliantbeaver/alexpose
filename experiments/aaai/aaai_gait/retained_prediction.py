"""Retained coordinate prediction from explicitly permitted leg observations.

Inputs already use the caller's fixed body reference Q. This module never
recomputes Q, phase, or any input feature from an artificially hidden value.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
from .masking import MaskBank, draw_masks, make_mask_bank, rng_stream
from .model import JOINT_INDICES

READOUT_JOINTS = tuple(range(25, 33))
READOUT_TOKENS = np.asarray([16*t+j for j in range(8,16) for t in range(8)])
LEG_FEATURE_DIM, RESIDUAL_FEATURE_DIM = 77, 154
FEATURE_VERSION = "permitted_leg_interpolation_moments_77_v1"

def patches(x):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (32,16,2) or len(x)==0:
        raise ValueError("Expected nonempty [windows,32,16,2] in JOINT_INDICES order")
    if np.isinf(x).any(): raise ValueError("Use NaN for missing coordinates, never infinity")
    return x.reshape(-1,8,4,16,2).transpose(0,1,3,2,4).reshape(-1,128,8)

def _normal_sources(x, source_ids, labels):
    ids, labels = np.asarray(source_ids).astype(str), np.asarray(labels).astype(str)
    if ids.shape != (len(x),) or labels.shape != (len(x),):
        raise ValueError("One source ID and label per window is required")
    if not len(ids) or set(labels) != {"normal"}:
        raise ValueError("Fitting requires explicitly labeled normal sources only")
    if np.any(ids == ""): raise ValueError("Empty source ID")
    return ids

def _support(p, hidden, targets):
    valid = np.isfinite(p).all(-1); visible = (~hidden)&valid
    slots = targets%16; side = slots%2; grid_slots=np.arange(128)[None,:]%16
    own = visible & (grid_slots>=6) & (grid_slots%2==side[:,None])
    other = visible & (grid_slots>=6) & (grid_slots%2!=side[:,None])
    result = dict(target_valid=valid[np.arange(len(p)),targets], own_count=own.sum(1),
                other_count=other.sum(1),own_times=own.reshape(-1,8,16).any(2).sum(1),
                other_times=other.reshape(-1,8,16).any(2).sum(1),
                own_joint_count=(visible&(grid_slots==slots[:,None])).sum(1))
    token_times=np.arange(128)[None,:]//16;query_times=targets[:,None]//16
    for name,mask in (("own",own),("other",other)):
        for period,selection in (("before",token_times<query_times),("within",token_times==query_times),
                                 ("after",token_times>query_times)):
            result[f"{name}_{period}"]=(mask&selection).sum(1)
    return result

def eligibility(support, rule="paired_legs"):
    if rule=="historical_own":
        return support["target_valid"]&(support["own_joint_count"]>=2)
    if rule=="paired_legs":
        return (support["target_valid"]&(support["own_count"]>=2)&(support["other_count"]>=2)
                &(support["own_times"]>=2)&(support["other_times"]>=2))
    raise ValueError("Unknown eligibility rule")

@dataclass
class FeatureBatch:
    own: np.ndarray
    own_extra: np.ndarray
    other: np.ndarray
    target: np.ndarray
    support: dict
    def residual(self, branch):
        if branch not in ("own","both"): raise ValueError("Unknown correction branch")
        return np.concatenate((self.own,self.own_extra if branch=="own" else self.other),axis=1)
    def subset(self, use):
        return FeatureBatch(self.own[use],self.own_extra[use],self.other[use],self.target[use],
                            {k:v[use] for k,v in self.support.items()})

def _leg_basis(p, hidden, targets, opposite=False):
    n=len(p); target_slot=targets%16; target_time=targets//16
    kind=(target_slot-8)//2; side=(target_slot%2)^int(opposite)
    orders=np.asarray([[k+1]+[j for j in range(5) if j!=k+1] for k in range(4)])
    slots=6+2*orders[kind]+side[:,None]
    grid=p.reshape(n,8,16,4,2)
    leg=grid[np.arange(n)[:,None,None],np.arange(8)[None,:,None],slots[:,None,:]]
    allowed=(~hidden.reshape(n,8,16)[np.arange(n)[:,None,None],np.arange(8)[None,:,None],slots[:,None,:]])&np.isfinite(leg).all((3,4))
    values=np.where(allowed[...,None,None],leg,0.).astype(np.float64)
    time=np.arange(8)[None,:,None]
    prev=np.where(allowed&(time<target_time[:,None,None]),time,-1).max(1)
    nxt=np.where(allowed&(time>target_time[:,None,None]),time,8).min(1)
    has_prev,has_next=prev>=0,nxt<8
    row,role=np.arange(n)[:,None],np.arange(5)[None,:]
    vprev=values[row,np.maximum(prev,0),role,3];vnext=values[row,np.minimum(nxt,7),role,0]
    start,end=4*prev+3,4*nxt
    query=4*target_time[:,None,None]+np.arange(4)[None,None,:]
    frac=(query-start[...,None])/np.maximum((end-start)[...,None],1)
    estimate=vprev[:,:,None]+frac[...,None]*(vnext-vprev)[:,:,None]
    estimate=np.where((has_prev&has_next)[...,None,None],estimate,
                      np.where(has_prev[...,None,None],vprev[:,:,None],np.where(has_next[...,None,None],vnext[:,:,None],0.)))
    same=allowed[np.arange(n),target_time]
    estimate=np.where(same[...,None,None],values[np.arange(n),target_time],estimate)
    support=np.stack((same,has_prev,has_next,allowed.mean(1),
                      np.where(has_prev,target_time[:,None]-prev,0)/8.,
                      np.where(has_next,nxt-target_time[:,None],0)/8.),axis=2)
    local=np.concatenate((estimate.reshape(n,5,8),support),axis=2)
    delta=(np.arange(8)[None,:,None]-target_time[:,None,None])/8.
    mean_xy=values.mean(3); denominator=np.maximum(allowed.sum(1),1)[...,None]
    moments=np.concatenate([(mean_xy*delta[...,None]**power).sum(1)/denominator for power in range(4)],axis=2)
    extra=np.concatenate((moments,support),axis=2)
    identity=np.concatenate((np.eye(4)[kind],(2*(target_slot%2)-1)[:,None],
                             (target_time/7.)[:,None],((target_time/7.)**2)[:,None]),axis=1)
    return (np.concatenate((local.reshape(n,70),identity),axis=1),
            np.concatenate((extra.reshape(n,70),identity),axis=1))

def build_features(p, hidden, target_indices):
    p=np.asarray(p);hidden=np.asarray(hidden);targets=np.asarray(target_indices,dtype=np.int64)
    if p.ndim!=3 or p.shape[1:]!=(128,8) or hidden.shape!=p.shape[:2] or hidden.dtype!=bool:
        raise ValueError("Expected patches[B,128,8] and Boolean hidden[B,128]")
    if targets.shape!=(len(p),) or np.any((targets<0)|(targets>=128)) or np.any(targets%16<8):
        raise ValueError("Targets must be lower-limb readout tokens")
    if not hidden[np.arange(len(p)),targets].all():
        raise ValueError("Every requested prediction target must be hidden")
    if np.isinf(p).any(): raise ValueError("Infinite coordinates")
    own,extra=_leg_basis(p,hidden,targets);other,_=_leg_basis(p,hidden,targets,True)
    return FeatureBatch(own,extra,other,p[np.arange(len(p)),targets].astype(float),_support(p,hidden,targets))

def construct_features(windows, hidden, joint, patch, validity=None):
    """Timing adapter. Joint is a BlazePose index, patch is0..7; scalars broadcast.

    ``hidden`` may include the pair's common-support restriction. Optional validity
    is Boolean[B,128] and can only remove measurements, never restore missing ones.
    """
    p=patches(windows);n=len(p);hidden=np.asarray(hidden,dtype=bool)
    if hidden.shape==(128,): hidden=np.broadcast_to(hidden,(n,128))
    if hidden.shape==(n,8,16): hidden=hidden.reshape(n,128)
    if validity is not None:
        validity=np.asarray(validity,dtype=bool)
        if validity.shape!=(n,128): raise ValueError("validity must be[B,128]")
        p=np.where(validity[...,None],p,np.nan)
    jj=np.broadcast_to(np.asarray(joint),(n,));tt=np.broadcast_to(np.asarray(patch),(n,))
    if not np.isin(jj,READOUT_JOINTS).all() or np.any((tt<0)|(tt>7)):
        raise ValueError("Invalid target joint or patch")
    targets=tt.astype(int)*16+np.asarray([JOINT_INDICES.index(int(j)) for j in jj])
    return build_features(p,hidden,targets)

@dataclass
class RidgeStats:
    dimension: int
    outputs: int=8
    total_weight: float=0.
    rows: int=0
    sx: np.ndarray=field(init=False)
    sy: np.ndarray=field(init=False)
    xx: np.ndarray=field(init=False)
    xy: np.ndarray=field(init=False)
    yy: float=0.
    def __post_init__(self):
        self.sx,self.sy=np.zeros(self.dimension),np.zeros(self.outputs)
        self.xx=np.zeros((self.dimension,self.dimension));self.xy=np.zeros((self.dimension,self.outputs))
    def add(self,x,y,weights=None):
        x,y=np.asarray(x,float),np.asarray(y,float)
        w=np.ones(len(x)) if weights is None else np.asarray(weights,float)
        if x.shape!=(len(x),self.dimension) or y.shape!=(len(x),self.outputs) or w.shape!=(len(x),):
            raise ValueError("Incorrect sufficient-statistic shapes")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(w).all() or np.any(w<=0):
            raise ValueError("Ridge values and positive weights must be finite")
        self.total_weight+=w.sum();self.rows+=len(x)
        self.sx+=w@x;self.sy+=w@y;self.xx+=x.T@(x*w[:,None]);self.xy+=x.T@(y*w[:,None])
        self.yy+=float(np.sum(w[:,None]*y*y))
    def fit(self,alpha=1.):
        if self.rows<8 or self.total_weight<=0 or alpha<=0:
            raise ValueError("Need at least8 rows and positive ridge regularization")
        mx,my=self.sx/self.total_weight,self.sy/self.total_weight
        centered=self.xx-np.outer(self.sx,self.sx)/self.total_weight
        scale=np.sqrt(np.maximum(np.diag(centered)/self.total_weight,0));scale[scale<1e-8]=1.
        gram=centered/np.outer(scale,scale)
        rhs=(self.xy-np.outer(self.sx,self.sy)/self.total_weight)/scale[:,None]
        coef=np.linalg.solve(gram+alpha*np.eye(self.dimension),rhs)/scale[:,None]
        eigen=np.linalg.eigvalsh(gram);rank=int((eigen>max(1e-10,eigen.max()*1e-10)).sum())
        return RidgePredictor(coef,my-mx@coef,float(alpha),rank,self.rows,self.total_weight,scale,mx)
    def risk(self,model=None,multiplier=1.):
        if model is None or multiplier==0: return self.yy/(self.total_weight*self.outputs)
        coef,intercept=model.coef*multiplier,model.intercept*multiplier
        sse=(self.yy-2*np.sum(coef*self.xy)-2*intercept@self.sy+np.sum(coef*(self.xx@coef))
             +2*(self.sx@coef)@intercept+self.total_weight*(intercept@intercept))
        return float(max(0,sse)/(self.total_weight*self.outputs))

@dataclass(frozen=True)
class RidgePredictor:
    coef: np.ndarray
    intercept: np.ndarray
    alpha: float
    feature_rank: int
    training_rows: int
    total_weight: float
    scale: np.ndarray
    center: np.ndarray
    def predict(self,x): return np.asarray(x)@self.coef+self.intercept

@dataclass
class LocalBaseline:
    ridge: RidgePredictor|None
    method: str
    metadata: dict
    def predict(self,own_features):
        return np.asarray(own_features)[:,:8].copy() if self.method=="interpolation" else self.ridge.predict(own_features)

@dataclass
class TrainingPlan:
    p: np.ndarray
    presentation_windows: np.ndarray
    hidden: np.ndarray
    row_presentations: np.ndarray
    target_indices: np.ndarray
    weights: np.ndarray
    metadata: dict
    def batches(self,batch_examples=2048):
        for begin in range(0,len(self.target_indices),batch_examples):
            rows=slice(begin,begin+batch_examples);presentations=self.row_presentations[rows]
            feat=build_features(self.p[self.presentation_windows[presentations]],self.hidden[presentations],self.target_indices[rows])
            yield feat,self.weights[rows]

def make_training_plan(x,*,source_ids,labels,policy="U",draws=512,seed=42,rule="paired_legs"):
    ids=_normal_sources(x,source_ids,labels);p=patches(x);unique=np.unique(ids)
    if draws<len(unique): raise ValueError("Need at least one presentation per fitting source")
    order_rng,window_rng=rng_stream(seed,"source_order"),rng_stream(seed,"window_selection")
    schedule=np.concatenate([order_rng.permutation(len(unique)) for _ in range((draws+len(unique)-1)//len(unique))])[:draws]
    groups=[np.flatnonzero(ids==source) for source in unique]
    chosen=np.asarray([window_rng.choice(groups[k]) for k in schedule])
    hidden=draw_masks(policy,draws,rng=rng_stream(seed,"training_masks"))
    valid=np.isfinite(p[chosen]).all(-1);visible=(~hidden&valid).reshape(draws,8,16)
    counts=np.stack([visible[:,:,6+side::2].sum((1,2)) for side in (0,1)],axis=1)
    times=np.stack([visible[:,:,6+side::2].any(2).sum(1) for side in (0,1)],axis=1)
    slots,sides=READOUT_TOKENS%16,READOUT_TOKENS%2
    support=dict(target_valid=valid[:,READOUT_TOKENS],own_count=counts[:,sides],other_count=counts[:,1-sides],
                 own_times=times[:,sides],other_times=times[:,1-sides],own_joint_count=visible.sum(1)[:,slots])
    requested=hidden[:,READOUT_TOKENS];use=requested&eligibility(support,rule)
    presentations,columns=np.nonzero(use);target_indices=READOUT_TOKENS[columns]
    row_sources=ids[chosen[presentations]];row_joints=target_indices%16
    counter=pd.DataFrame(dict(source=row_sources,joint=row_joints)).groupby(["source","joint"]).size()
    if not len(counter): raise ValueError("No eligible training targets; masks were not redrawn")
    if set(counter.index.get_level_values(0))!=set(unique):
        raise ValueError("A fitting source has no eligible target; revise the protocol explicitly")
    nj=counter.groupby(level=0).size()
    weight_lookup={(s,j):len(target_indices)/(len(unique)*nj[s]*v) for (s,j),v in counter.items()}
    weights=np.asarray([weight_lookup[s,j] for s,j in zip(row_sources,row_joints)])
    metadata=dict(policy=policy,seed=seed,presentations=draws,eligibility=rule,normal_sources=unique.tolist(),
                  feature_version=FEATURE_VERSION,eligible_target_rows=len(target_indices),requested_target_rows=int(requested.sum()),
                  source_joint_counts=[dict(source_id=s,joint=JOINT_INDICES[int(j)],rows=int(v)) for (s,j),v in counter.items()])
    return TrainingPlan(p,chosen,hidden,presentations,target_indices,weights,metadata)

def fit_local_baseline(x,*,source_ids,labels,draws=512,alpha=1.,seed=42,method="ridge",batch_examples=2048):
    if method not in ("ridge","interpolation"): raise ValueError("Unknown baseline method")
    plan=make_training_plan(x,source_ids=source_ids,labels=labels,policy="U",draws=draws,seed=seed)
    stats=RidgeStats(77)
    for feat,w in plan.batches(batch_examples): stats.add(feat.own,feat.target,w)
    ridge=stats.fit(alpha) if method=="ridge" else None
    meta=dict(plan.metadata,method=method,feature_dimension=77,feature_rank=0 if ridge is None else ridge.feature_rank,
              regularization=alpha,objective="source_joint_weighted_SSE_plus_alpha_standardized_coef_L2")
    return LocalBaseline(ridge,method,meta)

@dataclass
class RetainedBundle:
    base: LocalBaseline
    own: RidgePredictor
    both: RidgePredictor
    policy: str
    own_multiplier: float=1.
    both_multiplier: float=1.
    metadata: dict=field(default_factory=dict)
    def predict_features(self,feat):
        base=self.base.predict(feat.own)
        own=base.copy() if self.own_multiplier==0 else base+self.own_multiplier*self.own.predict(feat.residual("own"))
        both=base.copy() if self.both_multiplier==0 else base+self.both_multiplier*self.both.predict(feat.residual("both"))
        return dict(base=base,own=own,both=both)
    predict=predict_features
    def with_multipliers(self,own=1.,both=1.):
        if own not in (0.,.5,1.) or both not in (0.,.5,1.): raise ValueError("Multiplier grid is0,.5,1")
        return replace(self,own_multiplier=float(own),both_multiplier=float(both))

def fit_correction_grid(base,x,*,source_ids,labels,policy="U",alphas=(.1,1.,10.),draws=512,seed=42,batch_examples=2048):
    ids=_normal_sources(x,source_ids,labels)
    if set(base.metadata["normal_sources"])!=set(ids): raise ValueError("Baseline and corrections need the identical fitting-source role")
    plan=make_training_plan(x,source_ids=ids,labels=labels,policy=policy,draws=draws,seed=seed)
    own_stats,both_stats=RidgeStats(154),RidgeStats(154)
    for feat,w in plan.batches(batch_examples):
        residual=feat.target-base.predict(feat.own)
        own_stats.add(feat.residual("own"),residual,w);both_stats.add(feat.residual("both"),residual,w)
    result={}
    for alpha in alphas:
        own,both=own_stats.fit(alpha),both_stats.fit(alpha);zero=own_stats.risk()
        meta=dict(plan.metadata,regularization=float(alpha),own_feature_dimension=154,both_feature_dimension=154,
                  own_feature_rank=own.feature_rank,both_feature_rank=both.feature_rank,coefficient_count_per_branch=155*8,
                  zero_training_risk=zero,own_training_risk=own_stats.risk(own),both_training_risk=both_stats.risk(both),
                  rank_note="Equal widths do not imply equal rank or inductive bias.")
        result[float(alpha)]=RetainedBundle(base,own,both,policy,float(own_stats.risk(own)<=zero+1e-12),
                                          float(both_stats.risk(both)<=zero+1e-12),meta)
    return result

def fit_corrections(base,x,*,alpha=1.,own_multiplier=1.,both_multiplier=1.,**kwargs):
    result=fit_correction_grid(base,x,alphas=(alpha,),**kwargs)[float(alpha)]
    return result.with_multipliers(own=result.own_multiplier*own_multiplier,both=result.both_multiplier*both_multiplier)

def iter_feature_batches(x,bank=None,*,rule="historical_own",batch_windows=64,batch_examples=2048,include_invalid=False):
    """Yield bounded target batches; only requested targets enter the ledger."""
    bank=make_mask_bank() if bank is None else bank;p=patches(x)
    if batch_windows<1 or batch_examples<1: raise ValueError("Batch sizes must be positive")
    for start in range(0,len(p),batch_windows):
        windows=np.arange(start,min(start+batch_windows,len(p)))
        ww,mm,tt=np.meshgrid(windows,np.arange(len(bank.hidden)),READOUT_TOKENS,indexing="ij")
        ww,mm,tt=ww.ravel(),mm.ravel(),tt.ravel();requested=bank.targets[mm,tt]
        ww,mm,tt=ww[requested],mm[requested],tt[requested]
        for begin in range(0,len(ww),batch_examples):
            sl=slice(begin,begin+batch_examples);w,m,t=ww[sl],mm[sl],tt[sl]
            feat=build_features(p[w],bank.hidden[m],t);valid=eligibility(feat.support,rule)
            meta=pd.DataFrame(dict(window_index=w,mask_id=m.astype(np.int16),
                                   joint=np.asarray(JOINT_INDICES)[t%16].astype(np.int16),
                                   patch_index=(t//16).astype(np.int8),valid=valid,**feat.support))
            if not include_invalid:
                if not valid.any(): continue
                meta=meta.loc[valid].reset_index(drop=True);feat=feat.subset(valid)
            yield meta,feat

def iter_score_batches(bundle,x,bank=None,*,eligibility="historical_own",batch_windows=64,
                       batch_examples=2048,include_invalid=False,return_predictions=False):
    for meta,feat in iter_feature_batches(x,bank,rule=eligibility,batch_windows=batch_windows,
                                         batch_examples=batch_examples,include_invalid=include_invalid):
        predictions=bundle.predict_features(feat);valid=meta.valid.to_numpy()
        for branch,value in predictions.items():
            error=np.square(value-feat.target).mean(1);meta[f"loss_{branch}"]=np.where(valid,error,np.nan)
        meta["benefit"]=meta.loss_own-meta.loss_both;meta["sham_benefit"]=meta.loss_base-meta.loss_own
        if return_predictions:
            for name,values in {"Y":feat.target,**predictions}.items():
                for k in range(8): meta[f"{name}_{k}"]=values[:,k]
        meta.attrs.update(policy=bundle.policy,feature_version=FEATURE_VERSION)
        yield meta

def inner_candidate_report(base,grids,x,*,source_ids,labels,bank=None,rule="historical_own",
                           multipliers=(0.,.5,1.),batch_examples=2048,inner_fold=None):
    """One source-level candidate error, sharing features across all alpha/gates.

    Root orchestrates disjoint internal folds and concatenates these reports.
    Every ``grids[policy][alpha]`` must retain the exact supplied baseline.
    """
    ids=_normal_sources(x,source_ids,labels)
    if set(ids)&set(base.metadata["normal_sources"]): raise ValueError("Internal validation sources overlap fitting")
    unique=np.unique(ids);lookup={s:i for i,s in enumerate(unique)}
    counts=np.zeros(len(unique)*8);sums={("shared","base",0.,1.):np.zeros_like(counts)}
    for policy,grid in grids.items():
        for alpha in grid:
            for branch in ("own","both"):
                for gate in multipliers: sums[policy,branch,float(alpha),float(gate)]=np.zeros_like(counts)
    for meta,feat in iter_feature_batches(x,bank,rule=rule,batch_examples=batch_examples):
        codes=np.asarray([lookup[s] for s in ids[meta.window_index]])*8+meta.joint.to_numpy()-25
        counts+=np.bincount(codes,minlength=len(counts));pred=base.predict(feat.own)
        err=np.square(pred-feat.target).mean(1)
        sums["shared","base",0.,1.]+=np.bincount(codes,weights=err,minlength=len(counts))
        features={branch:feat.residual(branch) for branch in ("own","both")}
        for policy,grid in grids.items():
            for alpha,bundle in grid.items():
                if bundle.base is not base: raise ValueError("Candidates must retain the same baseline object")
                for branch in ("own","both"):
                    correction=getattr(bundle,branch).predict(features[branch])
                    for gate in multipliers:
                        err=np.square(pred+gate*correction-feat.target).mean(1)
                        sums[policy,branch,float(alpha),float(gate)]+=np.bincount(codes,weights=err,minlength=len(counts))
    records=[]
    for (policy,branch,alpha,gate),total in sums.items():
        cell=np.divide(total,counts,out=np.full_like(total,np.nan),where=counts>0).reshape(-1,8)
        for i,source in enumerate(unique):
            present=np.isfinite(cell[i])
            records.append(dict(source_id=source,policy=policy,branch=branch,regularization=alpha,
                                multiplier=gate,loss=float(cell[i,present].mean()) if present.any() else np.nan,
                                readout_count=int(present.sum()),inner_fold=inner_fold))
    return pd.DataFrame(records)

def select_correction_recipe(report,policy,*,required_positive_folds=2):
    """Select own and both separately on normal internal reports, then gate both.

    Mean risk weights validation sources equally across folds. The operational
    two-fold gate also requires improved mean risk against f and selected f+h.
    It is not a significance test or a held-out-generalization guarantee.
    """
    needed={"source_id","policy","branch","regularization","multiplier","loss","inner_fold"}
    if not needed.issubset(report): raise ValueError("Incomplete internal candidate report")
    if report.inner_fold.isna().any() or not np.isfinite(report.loss).all():
        raise ValueError("Every candidate needs finite source losses and an internal fold")
    choices={};fold_loss={}
    for branch in ("own","both"):
        table=report[(report.policy==policy)&(report.branch==branch)]
        means=table.groupby(["regularization","multiplier"]).loss.mean().reset_index()
        if means.empty: raise ValueError(f"No {policy}/{branch} candidates")
        # Deterministic ties prefer the smaller gate, then stronger regularization.
        best=means.sort_values(["loss","multiplier","regularization"],ascending=[True,True,False]).iloc[0]
        chosen=table[(table.regularization==best.regularization)&(table.multiplier==best.multiplier)]
        choices[branch]=dict(alpha=float(best.regularization),multiplier=float(best.multiplier),mean_loss=float(best.loss))
        fold_loss[branch]=chosen.groupby("inner_fold").loss.mean()
    baseline=report[report.branch=="base"].groupby("inner_fold").loss.mean()
    frame=pd.concat([baseline.rename("base"),fold_loss["own"].rename("own"),fold_loss["both"].rename("both")],axis=1)
    if frame.isna().any().any() or len(frame)<required_positive_folds:
        raise ValueError("Missing common internal-fold losses")
    positive=(frame.both<frame.base)&(frame.both<frame.own)
    base_mean=report[report.branch=="base"].loss.mean()
    supported=bool(choices["both"]["multiplier"]>0 and positive.sum()>=required_positive_folds
                   and choices["both"]["mean_loss"]<min(base_mean,choices["own"]["mean_loss"]))
    selected_both=choices["both"]["multiplier"] if supported else 0.
    return dict(policy=policy,own_alpha=choices["own"]["alpha"],own_multiplier=choices["own"]["multiplier"],
                both_alpha=choices["both"]["alpha"],both_multiplier=selected_both,
                candidate_both_multiplier=choices["both"]["multiplier"],relational_supported=supported,
                positive_internal_folds=int(positive.sum()),internal_fold_count=len(frame),
                internal_losses=frame.reset_index().to_dict("records"),candidate_choices=choices)

def assemble_selected_bundle(grid,recipe):
    """Own and both may select different regularizations, with the same base."""
    own=grid[float(recipe["own_alpha"])];both=grid[float(recipe["both_alpha"])]
    if own.base is not both.base or own.policy!=both.policy: raise ValueError("Incompatible fitted grids")
    return RetainedBundle(own.base,own.own,both.both,own.policy,float(recipe["own_multiplier"]),
                          float(recipe["both_multiplier"]),dict(own.metadata,selected_recipe=recipe,
                                                                both_feature_rank=both.both.feature_rank))

def known_dependency_fixtures(seed=19,samples=1024):
    """Five paired seed fixtures with identical Y, O and C marginal laws.

    Only the visibility of W through C changes. Intervals summarize software
    Monte Carlo repetitions, never patients, gait accuracy or disease evidence.
    """
    result=[]
    for replicate in range(5):
        rng=rng_stream(seed+replicate,"known_dependency_fixture")
        own,independent=rng.normal(size=(2*samples,2,77)).transpose(1,0,2)
        innovation=rng.normal(size=(2*samples,1));extra=(own**2-1.)/np.sqrt(2.)
        y=.8*own[:,:1]-.4*own[:,1:2]+1.5*innovation+rng.normal(scale=.1,size=(2*samples,8))
        train,test=slice(0,samples),slice(samples,None);stat=RidgeStats(77);stat.add(own[train],y[train]);base=stat.fit(1.)
        residual=y[train]-base.predict(own[train])
        for coupled in (False,True):
            other=independent.copy()
            if coupled: other[:,0]=innovation[:,0]
            predictions=dict(base=base.predict(own[test]));ranks={}
            for branch,second in (("own",extra),("both",other)):
                features=np.concatenate((own,second),axis=1);stat=RidgeStats(154);stat.add(features[train],residual)
                model=stat.fit(1.);predictions[branch]=predictions["base"]+model.predict(features[test]);ranks[branch]=model.feature_rank
            losses={k:float(np.square(v-y[test]).mean()) for k,v in predictions.items()}
            result.append(dict(replicate=replicate,coupled=coupled,losses=losses,
                               benefit=losses["own"]-losses["both"],ranks=ranks,
                               train_samples=samples,test_samples=samples))
    summary={}
    for coupled in (False,True):
        values=np.array([r["benefit"] for r in result if r["coupled"]==coupled])
        mean=float(values.mean());se=float(values.std(ddof=1)/np.sqrt(len(values)))
        summary["coupled" if coupled else "independent"]=dict(mean=mean,standard_error=se,
              monte_carlo_interval=[mean-2.776*se,mean+2.776*se],replicates=len(values))
    passed=summary["coupled"]["monte_carlo_interval"][0]>.5 and summary["independent"]["monte_carlo_interval"][0]<=0
    return dict(scope="feature_level_estimator_fixture_not_gait",seed=seed,cases=result,
                marginal_laws="Y=.8O0-.4O1+1.5W+.1noise; only C0=W versus independent changes",
                summary=summary,passed=bool(passed))
