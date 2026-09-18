"""Validated, explicit scope. No empirical GPU allocation is authorized by default."""
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
import math

ARMS=('smoothnet','direct','static','initialized','coordinate','ordinary_jepa','paired_jepa','shuffled_jepa')
STAGES=('audit','data','adaptation','information','direct','jepa','evaluate','optional','freeze','report')

@dataclass(frozen=True)
class RunConfig:
    run_id: str
    output_root: str='outputs/synthetic-training-v2'
    mode: str='fixture'
    bundle: str=''
    device: str='cpu'
    seed: int=17
    seeds: tuple=(17,)
    updates: int=200
    readout_updates: int=200
    batch_size: int=64
    bootstrap_draws: int=2000
    authorized_gpu_hours: float=0.
    measured_gpu_hours: float=0.
    projected_gpu_hours: float=0.
    cost_ledger: str=''
    held_extractor: str='unverified-held-family'
    model: dict=field(default_factory=dict)
    arms: tuple=ARMS
    resource_contrast: str='matched_data_steps'
    total_compute_seconds: float=0.
    decision_spec: str=''
    margins: dict=field(default_factory=lambda:dict(coordinate=.02,displacement=.05,clean=.01))

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{2,90}',self.run_id):raise ValueError('Use a unique safe run ID')
        if self.mode not in {'fixture','source'}:raise ValueError('Only fixture/source modes; confirmation is separate')
        if self.device not in {'cpu','cuda'}:raise ValueError('Explicit cpu/cuda required')
        if not 1<=self.updates<=2000 or not 1<=self.readout_updates<=2000:raise ValueError('Screening update cap is 2000 per phase')
        if self.batch_size<1 or self.bootstrap_draws<10:raise ValueError('Invalid batch/bootstrap size')
        if not set(self.arms)<=set(ARMS):raise ValueError('Unknown arm')
        for name in ('authorized_gpu_hours','measured_gpu_hours','projected_gpu_hours','total_compute_seconds'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<0:raise ValueError('Budget fields must be finite and nonnegative')
        if not 0<=self.authorized_gpu_hours<=48:raise ValueError('GPU hours must lie in [0,48]')
        if type(self.seed) is not int or not 0<=self.seed<=2**32-1:
            raise ValueError('Seed must be a nonboolean integer in [0, 2**32-1]')
        if not isinstance(self.seeds,(list,tuple)) or not self.seeds:
            raise ValueError('Seeds must be a nonempty list or tuple of integers')
        if any(type(value) is not int or not 0<=value<=2**32-1 for value in self.seeds):
            raise ValueError('Seeds must be nonboolean integers in [0, 2**32-1]')
        if len(set(self.seeds))!=len(self.seeds):raise ValueError('Seeds must be unique')
        if self.resource_contrast not in {'matched_data_steps','equal_total_compute'}:raise ValueError('Unknown resource comparison')
        if self.resource_contrast=='equal_total_compute' and self.total_compute_seconds<=0:raise ValueError('Compute comparison needs a measured per-arm budget')
        if self.mode=='source' and not self.bundle:raise ValueError('Source mode requires prepared audited paired bundle')
        if self.mode=='fixture' and self.device!='cpu':raise ValueError('Fixtures use CPU only')
        if self.mode=='fixture' and (self.updates>10 or self.readout_updates>10):raise ValueError('Fixture updates capped at10; no scientific fitting')

    @property
    def root(self):return Path(self.output_root).expanduser().resolve()/self.run_id
    def as_dict(self):return asdict(self)
    @classmethod
    def load(cls,path):return cls(**json.loads(Path(path).read_text()))
    @classmethod
    def fixture(cls,run_id,output_root='outputs/synthetic-training-v2'):
        return cls(run_id,output_root=output_root,updates=2,readout_updates=2,batch_size=4,
                   bootstrap_draws=100,model=dict(width=16,encoder_layers=1,predictor_layers=1,heads=2,patch_size=4,window_size=64),
                   held_extractor='fixture-held')

    def require_gpu_scope(self):
        if self.device!='cuda':return
        if self.authorized_gpu_hours<=0 or self.projected_gpu_hours<=0 or not self.cost_ledger:
            raise PermissionError('GPU stage needs explicit authorized budget, measured projection and all-stage cost ledger')
        if self.measured_gpu_hours+self.projected_gpu_hours>self.authorized_gpu_hours:
            raise PermissionError('Projected stage exceeds total GPU budget')
        ledger=json.loads(Path(self.cost_ledger).read_text())
        if not ledger.get('scope_authorized',False):raise PermissionError('Cost ledger must record explicit user-authorized scope')
        entries=ledger.get('entries')
        if not isinstance(entries,list):raise ValueError('Cost ledger requires entries including preparation and retries')
        seconds=0.
        for row in entries:
            value=float(row['gpu_seconds'])
            if not math.isfinite(value) or value<0:raise ValueError('Invalid measured GPU cost')
            seconds+=value
        if abs(seconds/3600-self.measured_gpu_hours)>1e-6:raise ValueError('Declared prior cost differs from measured ledger')
