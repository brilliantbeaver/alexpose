"""Source-balanced ridge completion with the neural study's exact masks.

Three independently fitted linear context branches predict local coordinates.
Availability indicators describe observation support; hidden measurements never
enter features. This is a deliberately simple alternative to neural prediction.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from .context_prediction import ContextConfig, make_mask_bank
from .model import JOINT_INDICES, uniform_mask
import torch


def patches(x):
    x=np.asarray(x,dtype=np.float32)
    if x.ndim!=4 or x.shape[1:]!=(32,16,2) or np.isinf(x).any():
        raise ValueError('Expected finite or NaN [windows,32,16,2]')
    return x.reshape(-1,8,4,16,2).transpose(0,1,3,2,4).reshape(-1,128,8)


def context_features(p,mask,joint_slot,patch_index,branch):
    """The only information passed to a fitted linear predictor."""
    valid=np.isfinite(p).all(-1)
    allowed=(~np.asarray(mask,dtype=bool))&valid
    slots=np.arange(128)%16;times=np.arange(128)//16
    if branch=='own':allowed&=slots[None,:]==joint_slot
    elif branch=='patch':allowed&=((slots==joint_slot)|(times==patch_index))[None,:]
    elif branch!='full':raise ValueError('Unknown context branch')
    values=np.where(allowed[:,:,None],p,0).reshape(len(p),-1)
    return np.concatenate([values,allowed.astype(np.float32)],axis=1)


@dataclass
class RidgeContext:
    estimators:dict
    counts:dict
    config:ContextConfig
    seed:int
    alpha:float


def fit_ridge_context(x,*,source_ids,labels,config=None,draws=256,alpha=10.,seed=42):
    """Fit only normal train sources; regularization is fixed before scoring."""
    cfg=config or ContextConfig()
    if len(labels)!=len(x) or set(labels)!={'normal'}:raise ValueError('Ridge fitting needs normal labels only')
    if len(source_ids)!=len(x) or draws<8 or alpha<=0:raise ValueError('Invalid source IDs, sample count, or regularization')
    ids=np.asarray(source_ids);groups=[np.flatnonzero(ids==s) for s in sorted(set(ids))]
    rng=np.random.default_rng(seed)
    chosen=np.array([rng.choice(groups[rng.integers(len(groups))]) for _ in range(draws)])
    p=patches(np.asarray(x)[chosen]);valid=np.isfinite(p).all(-1)
    masks=uniform_mask(draws,128,cfg.mask_ratio,generator=torch.Generator().manual_seed(seed+1000)).numpy()
    fitted={};counts={}
    for j in range(16):
        own_count=((~masks)&valid)[:,j::16].sum(1)
        for t in range(8):
            token=16*t+j;use=masks[:,token]&valid[:,token]&(own_count>=2)
            counts[(j,t)]=int(use.sum())
            if use.sum()<8:continue
            for branch in ('own','patch','full'):
                features=context_features(p[use],masks[use],j,t,branch)
                fitted[(j,t,branch)]=Ridge(alpha=alpha,solver='lsqr',tol=1e-4,max_iter=200).fit(features,p[use,token])
    return RidgeContext(fitted,counts,cfg,seed,float(alpha))


def score_ridge_context(model,x,scoring_rows,*,mask_seed=17):
    """Reuse requested neural coordinate targets, eligibility and mask IDs."""
    rows=scoring_rows.loc[scoring_rows.target_type.eq('coordinate')].copy()
    if rows.empty:raise ValueError('Need neural coordinate target identities')
    rows=rows.reset_index(drop=True);p=patches(x)
    if 'exclusion_reason' in rows:rows['exclusion_reason']=rows.exclusion_reason.astype(object)
    bank=make_mask_bank(model.config,mask_seed,int(rows.mask_id.max())+1).numpy()
    for col in ('loss_own','loss_patch','loss_full'):rows[col]=np.nan
    for (joint,t,mask_id),group in rows.groupby(['joint','patch_index','mask_id'],observed=True):
        j=JOINT_INDICES.index(int(joint));t=int(t);token=t*16+j
        use=group[group.valid].index.to_numpy()
        if len(use)==0:continue
        if any((j,t,b) not in model.estimators for b in ('own','patch','full')):
            rows.loc[use,'valid']=False;rows.loc[use,'exclusion_reason']='ridge_training_support';continue
        windows=rows.loc[use,'window_index'].to_numpy(int)
        pp=p[windows];mask=np.broadcast_to(bank[int(mask_id)],(len(pp),128))
        for branch in ('own','patch','full'):
            prediction=model.estimators[(j,t,branch)].predict(context_features(pp,mask,j,t,branch))
            rows.loc[use,'loss_'+branch]=np.square(prediction-pp[:,token]).mean(axis=1)
    rows.attrs.update(method='source_balanced_ridge',alpha=model.alpha,seed=model.seed)
    return rows
