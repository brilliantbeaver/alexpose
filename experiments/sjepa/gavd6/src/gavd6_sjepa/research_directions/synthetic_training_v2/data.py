"""Input-only normalization, physical time, paired-factor checks and explicit fixtures."""
from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
from .contracts import TrackBundle, JOINTS, digest, array_digest, validate_inputs


def fixed_time_grid(source_times,start,*,samples=64,hz=25):
    t=np.asarray(source_times,float)
    if t.ndim!=1 or len(t)<2 or not np.isfinite(t).all() or np.any(np.diff(t)<=0):
        raise ValueError('Source timestamps must be finite and strictly increasing')
    grid=float(start)+np.arange(samples)/hz
    if grid[0]<t[0]-1e-9 or grid[-1]>t[-1]+1e-9:raise ValueError('Window exceeds real duration; do not stretch')
    return grid


def sample_observations(xy,confidence,observed,source_times,start,*,samples=64,hz=25,tolerance=.02):
    grid=fixed_time_grid(source_times,start,samples=samples,hz=hz)
    source_times=np.asarray(source_times)
    ix=np.searchsorted(source_times,grid).clip(0,len(source_times)-1)
    prev=(ix-1).clip(0);ix=np.where(abs(source_times[prev]-grid)<=abs(source_times[ix]-grid),prev,ix)
    support=np.asarray(observed,bool)[ix] & (abs(source_times[ix]-grid)<=tolerance)[:,None]
    return dict(xy=np.where(support[...,None],np.asarray(xy)[ix],np.nan),
                confidence=np.where(support,np.asarray(confidence)[ix],np.nan),
                observed=support,timestamps=grid),source_times[ix]


@dataclass
class Normalization:
    origin: np.ndarray
    scale: np.ndarray
    def apply(self,xy):return (xy-self.origin[:,None,None,:])/self.scale[:,None,None,None]
    def invert(self,xy):return xy*self.scale[:,None,None,None]+self.origin[:,None,None,:]


def normalize_inputs(inputs):
    validate_inputs(inputs)
    origins=[];scales=[]
    for xy,observed in zip(inputs['xy'],inputs['observed']):
        points=xy[observed]
        if len(points)<2:raise ValueError('Insufficient observed coordinates for input-only normalization')
        origin=np.median(points,axis=0);span=np.quantile(points,.95,axis=0)-np.quantile(points,.05,axis=0)
        scale=float(np.linalg.norm(span))
        if not np.isfinite(scale) or scale<1e-6:raise ValueError('Degenerate input-derived scale')
        origins.append(origin);scales.append(scale)
    norm=Normalization(np.asarray(origins,np.float32),np.asarray(scales,np.float32))
    result={k:v.copy() for k,v in inputs.items()};result['xy']=norm.apply(result['xy']).astype(np.float32)
    # Numeric fill only for absent score channels; the observed flag retains
    # missingness and artifacts retain the original NaN score. This is not an
    # estimated confidence or a target-derived calibration.
    result['confidence']=np.where(np.isfinite(result['confidence']),result['confidence'],0).astype(np.float32)
    return result,norm


PAIR_FACTORS=('motion_hash','shape_hash','timestamps_hash','camera_hash','background_hash','lighting_hash','appearance_hash','render_seed')

def audit_pair(clean,intervened,changed):
    if not set(changed)<=set(PAIR_FACTORS)|{'blur_px','occlusion_fraction','downsample'}:
        raise ValueError('Unknown intervention factor')
    for name in PAIR_FACTORS:
        if name not in clean or name not in intervened:raise ValueError(f'Missing paired factor {name}')
        if name not in changed and clean[name]!=intervened[name]:raise ValueError(f'Undeclared pair change: {name}')
    if any(clean.get(k)==intervened.get(k) for k in changed):raise ValueError('Declared intervention did not change')
    if 'camera_hash' in changed and clean.get('target_hash')==intervened.get('target_hash'):
        raise ValueError('Changed camera requires separately projected targets')
    return {'status':'pass','changed':list(changed),'fixed':list(set(PAIR_FACTORS)-set(changed))}


def shuffled_donors(records,seed):
    rng=np.random.default_rng(seed);donors=[]
    for r in records:
        if r['split']!='train':raise ValueError('Shuffled pairing only uses source training')
        choices=[i for i,s in enumerate(records) if s['canonical_person_id']==r['canonical_person_id']
                 and s['variant']==r['variant'] and s['extractor']==r['extractor']
                 and s['window_id']!=r['window_id'] and s['split']=='train']
        if not choices:raise ValueError('No distinct training-window donor in matched person/nuisance stratum')
        donors.append(int(rng.choice(choices)))
    return np.asarray(donors)


def filter_tracks(inputs,strength=1):
    """Offline interpolation plus centered triangular filter, same declared window.

    Entirely missing joints remain NaN and receive evaluation penalties.
    Interpolated values do not change observed masks in saved input records.
    """
    if strength not in {0,1,2}:raise ValueError('Predeclared filter strengths are0,1,2')
    out=np.full_like(inputs['xy'],np.nan)
    for n in range(len(out)):
        t=inputs['timestamps'][n]
        for j in range(12):
            ok=inputs['observed'][n,:,j]
            if not ok.any():continue
            for c in range(2):
                v=np.interp(t,t[ok],inputs['xy'][n,ok,j,c])
                if strength:
                    w=np.r_[np.arange(1,strength+2),np.arange(strength,0,-1)];w=w/w.sum()
                    v=np.convolve(np.pad(v,(strength,strength),mode='edge'),w,mode='valid')
                out[n,:,j,c]=v
    return out


def fixture_bundle(seed=17):
    """Analytic sine trajectories for SOFTWARE evidence only; no licensed data/humans."""
    rng=np.random.default_rng(seed);inputs={k:[] for k in ('xy','confidence','observed','timestamps')}
    targets={k:[] for k in ('xy','valid','visible','eval_scale')};records=[]
    base=np.array([[220,100],[300,100],[210,160],[310,160],[200,220],[320,220],
                   [235,230],[285,230],[230,315],[290,315],[220,400],[300,400]],float)
    for p in range(7):
        split='train' if p<4 else 'development'
        for m in range(2):
            times=np.arange(64)/25+m*3.;phase=2*np.pi*(.8+.05*p)*(times-times[0])+.25*m
            clean=np.broadcast_to(base,(64,12,2)).copy()
            clean[:,:,0]+=8*np.sin(phase)[:,None]
            for j in range(6,12):clean[:,j,0]+=(12+p)*np.sin(phase+(j%2)*np.pi)
            for variant in ('clean','blur','obstruction'):
                for extractor in (('fixture-source',) if split=='train' else ('fixture-source','fixture-held')):
                    noisy=clean+rng.normal(0,.3 if variant=='clean' else 3.0,clean.shape)
                    observed=np.ones((64,12),bool);confidence=np.full((64,12),.9,np.float32)
                    if variant=='obstruction':observed[18:28,10:]=False
                    noisy[~observed]=np.nan;confidence[~observed]=np.nan
                    visible=np.ones((64,12),bool)
                    if variant=='obstruction':visible[18:28,10:]=False
                    for k,v in dict(xy=noisy.astype(np.float32),confidence=confidence,observed=observed,timestamps=times).items():inputs[k].append(v)
                    for k,v in dict(xy=clean.astype(np.float32),valid=np.ones_like(visible),visible=visible,eval_scale=np.full(64,360.)).items():targets[k].append(v)
                    records.append(dict(person_id=f'fixture-person{p}',canonical_person_id=f'fixture-person{p}',
                        motion_id=f'fixture-motion{p}-{m}',motion_hash=digest([p,m]),window_id=f'fixture-window{p}-{m}',
                        split=split,variant=variant,extractor=extractor,box_source='fixture-fixed-independent-scale',
                        exposure='software_fixture',original_split='train' if p<4 else 'validation',
                        locomotion_status='fixture',audit_reviewer='analytic fixture generator',audit_evidence='no human annotation',
                        audit_date='2026-09-18',target_kind='synthetic_proxy',seed=seed,start_s=float(times[0]),end_s=float(times[-1])))
    inputs={k:np.stack(v) for k,v in inputs.items()};targets={k:np.stack(v) for k,v in targets.items()}
    bundle=TrackBundle(inputs,targets,records,'fixture-tested',dict(kind='analytic software fixture',identity=array_digest(inputs)))
    bundle.validate('fixture-held');return bundle
