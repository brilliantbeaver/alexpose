"""Small deterministic normal-only forecasters for reference-masking-v2.

The public inference boundary receives exactly sixteen observed-prefix samples.
Future outcomes and their visibility enter fitting/loss only through separate
arguments. Predictions always request eight future times and eight distal joints.
This module does not assign folds, tune against anomalies, or calculate AUROC.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from threadpoolctl import threadpool_limits

from .data import JOINT_IDS
from .v2_masks import MaskConfig, batch_masks

C, H = 16, 8
TARGET_IDS = (25, 27, 29, 31, 26, 28, 30, 32)
TARGET_JOINTS = np.array([JOINT_IDS.index(j) for j in TARGET_IDS])
MODES = ('mlp', 'ridge_posture', 'ridge_posture_velocity')


@dataclass(frozen=True)
class ForecastConfig:
    mode: str = 'mlp'
    mask_policy: str = 'none'
    mask_ratio: float = .4
    seed: int = 27
    hidden_dim: int = 32
    steps: int = 200
    batch_sources: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    ridge_alpha: float = 10.
    mask_banks: int = 4
    pattern_only: bool = False
    objective: str = 'coordinate'

    def __post_init__(self):
        if self.mode not in MODES or self.objective != 'coordinate':
            raise ValueError('Unknown mode or unsupported objective; this stage is coordinate-only')
        MaskConfig(self.mask_policy, self.mask_ratio)
        if self.hidden_dim < 1 or self.steps < 1 or self.batch_sources < 2 or self.mask_banks < 1:
            raise ValueError('Positive dimensions, steps and banks; at least two sources per batch')
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.ridge_alpha <= 0:
            raise ValueError('Invalid learning or regularization parameters')
        if self.mode != 'mlp' and (self.mask_policy != 'none' or self.pattern_only):
            raise ValueError('Ridge reference modes use the complete observed prefix')


def _prefix(prefix, observed, times, hidden=None):
    x, v, t = np.asarray(prefix, float), np.asarray(observed), np.asarray(times, float)
    if x.ndim != 4 or x.shape[1:] != (C, len(JOINT_IDS), 2) or not len(x):
        raise ValueError('Inference must receive a nonempty exact [N,16,16,2] prefix')
    if v.shape != x.shape[:-1] or v.dtype != np.bool_:
        raise ValueError('Natural support must be Boolean [N,16,16]')
    if t.shape != (len(x), C) or not np.isfinite(t).all() or np.any(np.diff(t, axis=1) <= 0):
        raise ValueError('Actual prefix timestamps must be finite and increasing')
    if hidden is None:
        h = np.zeros_like(v)
    else:
        h = np.asarray(hidden)
        if h.shape != v.shape or h.dtype != np.bool_ or np.any(h & ~v):
            raise ValueError('Hidden mask must be Boolean and a subset of natural support')
    visible = v & ~h
    # Deliberately hidden values need not be finite: they are erased before any
    # coordinate operation. Policy construction has its own observed-value API.
    if not np.isfinite(x[visible]).all():
        raise ValueError('Visible coordinates must be finite')
    return x, v, t, h, visible


def causal_features(prefix, observed, times, *, mode='mlp', hidden=None, pattern_only=False):
    """Return feature matrix, target anchors and known-target flags.

    Causal filling happens AFTER hiding. Anchors are derived from the actually
    visible prefix. Canonical training masks protect latest anchors; deliberately
    different stress masks may change the visible anchor and remain causal.
    """
    if mode not in MODES:
        raise ValueError('Unknown feature mode')
    x, natural, t, hidden, visible = _prefix(prefix, observed, times, hidden)
    n = len(x)
    safe = np.where(visible[..., None], x, 0.)
    filled = np.zeros_like(safe)
    last = np.zeros((n,16,2))
    for k in range(C):
        last = np.where(visible[:, k, :, None], safe[:, k], last)
        filled[:, k] = last
    known = visible.any(axis=1)
    target_anchor, target_known = last[:, TARGET_JOINTS], known[:, TARGET_JOINTS]
    if mode == 'ridge_posture':
        f = np.concatenate([target_anchor.reshape(n,-1), target_known.astype(float)],axis=1)
    elif mode == 'ridge_posture_velocity':
        # The median of recent observed adjacent-pair velocities resists one
        # implausible increment. Unknown velocity remains distinct via counts.
        velocity, dispersion, count = np.zeros((n,8,2)), np.zeros((n,8,2)), np.zeros((n,8))
        for i in range(n):
            for out, j in enumerate(TARGET_JOINTS):
                paired = visible[i, 1:, j] & visible[i, :-1, j]
                positions = np.flatnonzero(paired & (np.arange(C-1) >= C-6))
                if len(positions):
                    slopes = (safe[i, positions+1, j]-safe[i, positions, j]) / np.diff(t[i])[positions,None]
                    velocity[i,out] = np.median(slopes,axis=0)
                    dispersion[i,out] = np.median(np.abs(slopes-velocity[i,out]),axis=0)
                    count[i,out] = len(positions)/5.
        f = np.concatenate([target_anchor.reshape(n,-1), target_known.astype(float),
                            velocity.reshape(n,-1),dispersion.reshape(n,-1),count],axis=1)
    else:
        relative = filled-last[:,None]
        f = np.concatenate([relative.reshape(n,-1), visible.reshape(n,-1), last.reshape(n,-1),
                            known, (t-t[:,-1:])/1.5],axis=1).astype(float)
        if pattern_only:
            # Preserve the visibility/time pattern but remove coordinate values
            # from the learned residual input. Persistence anchors remain added
            # after prediction, so this is a mask-pattern residual diagnostic.
            f[:, :C*16*2] = 0
            anchor_start = C*16*2 + C*16
            f[:, anchor_start:anchor_start+16*2] = 0
    if not np.isfinite(f).all():
        raise ValueError('Feature construction produced nonfinite values')
    return f, target_anchor, target_known


def target_values(future, future_valid, anchors, known):
    """Build common observed residual targets in declared left/right order."""
    y, v = np.asarray(future, float), np.asarray(future_valid)
    if y.ndim != 4 or y.shape[1:] not in ((H,16,2),(H,8,2)):
        raise ValueError('Future must be [N,8,16,2] or already ordered [N,8,8,2]')
    if v.shape != y.shape[:-1] or v.dtype != np.bool_ or anchors.shape != (len(y),8,2) or known.shape != (len(y),8):
        raise ValueError('Invalid future support or target anchors')
    if y.shape[2] == 16:
        y, v = y[:,:,TARGET_JOINTS], v[:,:,TARGET_JOINTS]
    observed = v & known[:,None]
    if not np.isfinite(y[observed]).all():
        raise ValueError('Observed targets must be finite')
    for leg in (slice(0,4),slice(4,8)):
        if np.any(observed[:,:,leg].sum((1,2))==0):
            raise ValueError('Every training window needs observed targets for both legs')
    residual = np.where(observed[...,None], y-anchors[:,None], 0.)
    coordinate_mask = np.repeat(observed[...,None],2,axis=-1)
    return residual.astype(np.float32), coordinate_mask


def hierarchical_weights(metadata):
    """Equal group mass, then equal sequence, then equal window mass."""
    meta = metadata.copy().reset_index(drop=True)
    if 'group_id' not in meta:
        meta['group_id'] = meta.source_id.astype(str)
    counts = meta.groupby(['group_id','sequence_id']).group_id.transform('size').to_numpy()
    sequences = meta[['group_id','sequence_id']].drop_duplicates().groupby('group_id').size()
    w = 1/(counts*meta.group_id.map(sequences).to_numpy())
    return w/w.mean()


def _training_meta(metadata, n):
    meta = pd.DataFrame(metadata).copy().reset_index(drop=True)
    if len(meta)!=n or not {'source_id','sequence_id','label'}.issubset(meta):
        raise ValueError('Training metadata needs source_id,sequence_id,label for every window')
    if set(meta.label.astype(str)) != {'normal'}:
        raise ValueError('Forecasters fit normal training sources only')
    if meta[['source_id','sequence_id']].isna().any().any() or (meta.source_id.astype(str)=='').any():
        raise ValueError('Missing source or sequence identity')
    if 'partition' in meta and not meta.partition.eq('train').all():
        raise ValueError('Only declared training observations may enter fitting')
    if 'group_id' not in meta:
        meta['group_id'] = meta.source_id.astype(str)
    if meta.group_id.nunique()<2:
        raise ValueError('At least two independent training groups are required')
    return meta


def masked_window_loss(prediction, target, mask):
    """Observed coordinate mean per leg, then equal legs and windows."""
    losses=[]
    for leg in (slice(0,4),slice(4,8)):
        m=mask[:,:,leg]
        count=m.flatten(1).sum(1)
        if torch.any(count == 0):
            raise ValueError('Each leg needs target support')
        safe_target=torch.where(m,target[:,:,leg],torch.zeros_like(target[:,:,leg]))
        sq=torch.where(m,(prediction[:,:,leg]-safe_target).square(),torch.zeros_like(safe_target))
        losses.append(sq.flatten(1).sum(1)/count)
    return torch.stack(losses,1).mean(1)


class ResidualMLP(nn.Module):
    """Two32-unit hidden layers and a persistence-initialized residual head."""
    def __init__(self,input_dim=832,hidden_dim=32):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),
                                   nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        self.head=nn.Linear(hidden_dim,H*8*2)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,x):
        return self.head(self.encoder(x)).reshape(-1,H,8,2)


def _scale_fit(features, weights):
    mu=np.average(features,axis=0,weights=weights)
    sd=np.sqrt(np.average((features-mu)**2,axis=0,weights=weights))
    sd[sd<1e-6]=1.
    return mu,sd


def _fit_ridge(f,y,m,weights,alpha):
    # Each leg has half the total loss mass regardless of its observation count.
    yy, mm=y.reshape(len(y),-1),m.reshape(len(y),-1)
    mu,sd=_scale_fit(f,weights);z=(f-mu)/sd
    coef=np.zeros((z.shape[1],yy.shape[1]));intercept=np.zeros(yy.shape[1])
    target_leg=np.tile(np.repeat([0,1],4*2),H)
    observed_per_leg=np.stack([m[:,:,leg].reshape(len(y),-1).sum(1)
                            for leg in (slice(0,4),slice(4,8))],axis=1)
    for k in range(yy.shape[1]):
        use=mm[:,k]
        if not use.any():raise ValueError('No observed target for one output coordinate')
        w=weights[use]*(H*4*2)/observed_per_leg[use,target_leg[k]]
        xx,yt=z[use],yy[use,k]
        xm,ym=np.average(xx,axis=0,weights=w),np.average(yt,weights=w)
        centered=xx-xm
        coef[:,k]=np.linalg.solve((centered*w[:,None]).T@centered+alpha*np.eye(z.shape[1]),
                                 (centered*w[:,None]).T@(yt-ym))
        intercept[k]=ym-xm@coef[:,k]
    return dict(mean=mu,scale=sd,coef=coef,intercept=intercept)


def _identity_arrays(*arrays):
    digest=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        digest.update(str((a.shape,str(a.dtype))).encode());digest.update(a.tobytes())
    return digest.hexdigest()


def _representation_health(h, weights):
    centered=h-np.average(h,axis=0,weights=weights)
    singular=np.linalg.svd(centered*np.sqrt(weights[:,None]),compute_uv=False)
    spectrum=singular**2
    probabilities=spectrum/max(spectrum.sum(),1e-30)
    return {'mean_std':float(np.sqrt(np.average(centered**2,axis=0,weights=weights)).mean()),
            'effective_rank':float(np.exp(-np.sum(probabilities*np.log(np.maximum(probabilities,1e-30)))))}


def _group_sampler(meta,rng,count):
    groups=sorted(meta.group_id.unique())
    chosen=rng.choice(groups,size=min(len(groups),count),replace=False)
    rows=[]
    for group in chosen:
        gm=meta[meta.group_id==group]
        seq=rng.choice(sorted(gm.sequence_id.unique()))
        rows.append(int(rng.choice(gm.index[gm.sequence_id==seq])))
    return np.array(rows)


def fit_forecaster(prefix, observed, times, future, future_valid, metadata,
                   config: ForecastConfig | None = None):
    """Fit one fixed configuration using only explicitly supplied normal groups.

    The caller owns group-isolated inner selection and final evaluation. Four
    seeded mask banks are built once. Every optimization step samples groups,
    sequences and windows uniformly, then a mask bank independently. This keeps
    exposure and sampling seeds identical across policies while bounding compute.
    """
    c=config or ForecastConfig();started=time.perf_counter()
    x,v,t,_,_=_prefix(prefix,observed,times)
    meta=_training_meta(metadata,len(x))
    torch.set_num_threads(1)
    torch.manual_seed(c.seed)
    sampler_rng=np.random.default_rng(np.random.SeedSequence([c.seed,1]))
    bank_rng=np.random.default_rng(np.random.SeedSequence([c.seed,2]))
    weights=hierarchical_weights(meta)
    _,anchors,known=causal_features(x,v,t,mode='mlp')
    y,m=target_values(future,future_valid,anchors,known)
    checkpoint=dict(schema='reference-v2-forecaster-v1',config=asdict(c),
        target_joint_ids=list(TARGET_IDS),training_source_ids=sorted(meta.source_id.astype(str).unique()),
        training_group_ids=sorted(meta.group_id.astype(str).unique()),training_sequences=int(meta.sequence_id.nunique()),
        training_windows=len(meta),training_input_sha256=_identity_arrays(x,v,t,y,m),
        training_metadata_sha256=hashlib.sha256(meta.to_csv(index=False).encode()).hexdigest(),
        mask_audit=[],diagnostics={},history=[])
    with threadpool_limits(limits=1):
        if c.mode != 'mlp':
            f,_,_=causal_features(x,v,t,mode=c.mode)
            checkpoint['ridge']=_fit_ridge(f,y,m,weights,c.ridge_alpha)
            checkpoint['parameter_count']=int(checkpoint['ridge']['coef'].size+checkpoint['ridge']['intercept'].size)
            checkpoint['diagnostics']={'objective':'coordinate ridge; no latent representation',
                'feature_dimension':int(f.shape[1])}
        else:
            n_banks=1 if c.mask_policy=='none' else c.mask_banks
            banks=[]
            for b in range(n_banks):
                hidden,audit=batch_masks(x,v,t,MaskConfig(c.mask_policy,c.mask_ratio),
                    seed=np.random.SeedSequence([c.seed,3,b]).generate_state(1)[0])
                f,aa,kk=causal_features(x,v,t,hidden=hidden,pattern_only=c.pattern_only)
                if not np.array_equal(aa,anchors) or not np.array_equal(kk,known):
                    raise AssertionError('Training corruption changed a protected target anchor')
                banks.append(f)
                checkpoint['mask_audit'].append({'bank':b,'windows':len(audit),
                    'hidden_total':sum(a['hidden_tokens'] for a in audit),
                    'eligible_total':sum(a['eligible_tokens'] for a in audit),
                    'partial_final_blocks':sum(a['partial_final_block'] for a in audit),
                    'mask_sha256':_identity_arrays(hidden)})
            bank_array=np.stack(banks)
            mu,sd=_scale_fit(bank_array.reshape(-1,bank_array.shape[-1]),np.tile(weights,n_banks))
            ff=torch.as_tensor(((bank_array-mu)/sd).astype(np.float32))
            target=torch.as_tensor(y);mask=torch.as_tensor(m)
            model=ResidualMLP(ff.shape[-1],c.hidden_dim)
            full_features,_,_=causal_features(x,v,t,pattern_only=c.pattern_only)
            full_z=torch.as_tensor(((full_features-mu)/sd).astype(np.float32))
            with torch.no_grad():
                initial_health=_representation_health(model.encoder(full_z).numpy(),weights)
            optimizer=torch.optim.AdamW(model.parameters(),lr=c.learning_rate,weight_decay=c.weight_decay)
            for step in range(c.steps):
                ii=_group_sampler(meta,sampler_rng,c.batch_sources)
                bb=bank_rng.integers(n_banks,size=len(ii))
                optimizer.zero_grad(set_to_none=True)
                residual=model(ff[bb,ii])
                loss=masked_window_loss(residual,target[ii],mask[ii]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite forecast loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                if step in {0,c.steps-1} or (step+1)%50==0:
                    checkpoint['history'].append({'step':step+1,'sampled_source_loss':float(loss.detach())})
            model.eval()
            with torch.no_grad():
                h=model.encoder(full_z).numpy()
            final_health=_representation_health(h,weights)
            checkpoint['diagnostics']=dict(feature_dimension=int(ff.shape[-1]),
                mean_feature_std=final_health['mean_std'], effective_rank=final_health['effective_rank'],
                initial_mean_feature_std=initial_health['mean_std'],
                initial_effective_rank=initial_health['effective_rank'],
                diagnostic_population='normal training prefix only',latent_objective=False,
                maximum_absolute_parameter=max(float(p.detach().abs().max()) for p in model.parameters()))
            checkpoint['mean'],checkpoint['scale']=mu,sd
            checkpoint['state_dict']={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            checkpoint['parameter_count']=sum(p.numel() for p in model.parameters())
    checkpoint['seconds']=time.perf_counter()-started
    checkpoint['versions']={'numpy':np.__version__,'torch':torch.__version__}
    return checkpoint


def predict_forecaster(checkpoint, prefix, observed, times, *, hidden=None):
    """Predict fixed [N,8,8,2] queries using exactly the visible prefix only."""
    c=ForecastConfig(**checkpoint['config'])
    f,anchor,_=causal_features(prefix,observed,times,mode=c.mode,hidden=hidden,pattern_only=c.pattern_only)
    if c.mode == 'mlp':
        model=ResidualMLP(f.shape[-1],c.hidden_dim)
        model.load_state_dict(checkpoint['state_dict']);model.eval()
        with torch.no_grad():
            z=torch.as_tensor(((f-checkpoint['mean'])/checkpoint['scale']).astype(np.float32))
            residual=model(z).numpy()
    else:
        r=checkpoint['ridge']
        residual=(((f-r['mean'])/r['scale'])@r['coef']+r['intercept']).reshape(-1,H,8,2)
    prediction=residual+anchor[:,None]
    if not np.isfinite(prediction).all():raise FloatingPointError('Prediction is nonfinite')
    return prediction.astype(np.float32)
