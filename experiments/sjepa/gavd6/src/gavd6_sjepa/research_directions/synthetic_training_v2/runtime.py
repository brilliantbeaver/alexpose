"""Environment verification is evidence only when actually run in an allocation."""
import os
import sys

def require_torch26():
    import torch
    if torch.__version__.split('+')[0]!='2.6.0':raise RuntimeError('Study requires Torch2.6.0; do not sync the root environment')
    return torch


def require_haic_runtime():
    torch=require_torch26()
    if torch.__version__!='2.6.0+cu124':raise RuntimeError('HAIC requires exactly torch2.6.0+cu124')
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('HAIC GPU checks require Slurm batch allocation')
    if not torch.cuda.is_available():raise RuntimeError('Allocated CUDA device unavailable')
    import torchvision,mmcv
    if torchvision.__version__!='0.21.0+cu124' or mmcv.__version__!='2.1.0':raise RuntimeError('Preserve torchvision0.21.0+cu124 and compatible mmcv2.1.0')
    from torchvision.ops import nms
    from mmcv.ops import nms as mmcv_nms
    boxes=torch.tensor([[0.,0.,4.,4.],[1.,1.,3.,3.]],device='cuda');scores=torch.tensor([.9,.8],device='cuda')
    assert nms(boxes,scores,.5).numel()>0
    assert mmcv_nms(boxes,scores,.5)[1].numel()>0
    assert torch.isfinite(torch.randn(4,4,device='cuda')@torch.randn(4,4,device='cuda')).all()
    torch.cuda.synchronize()
    return {'interpreter':sys.executable,'torch':torch.__version__,'torchvision':torchvision.__version__,
            'mmcv':mmcv.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
            'job_id':os.environ['SLURM_JOB_ID'],'operators':'pass'}


from contextlib import contextmanager
import json
from pathlib import Path
import signal
import time
import uuid
from .contracts import atomic_json

@contextmanager
def _budgeted_gpu_stage_unlocked(cfg,stage):
    """Account for every allocated attempt, including failed loads and retries.

    Scheduler limits are the hard allocation bounds. A Python alarm additionally
    stops work at the smaller projected-stage or remaining study budget; one
    native operator may finish before the exception is delivered.
    """
    cfg.require_gpu_scope()
    if cfg.device!='cuda':yield;return
    folder=cfg.root/'costs';folder.mkdir(parents=True,exist_ok=True)
    prior=0.
    for path in folder.glob('*.json'):
        row=json.loads(path.read_text());value=float(row['gpu_seconds'])
        if not __import__('math').isfinite(value) or value<0:raise ValueError('Invalid attempt cost')
        prior+=value
    remaining=cfg.authorized_gpu_hours*3600-cfg.measured_gpu_hours*3600-prior
    projection=cfg.projected_gpu_hours*3600
    if projection>remaining+1e-6:raise PermissionError('Cumulative attempts plus proposed stage exceed total authorized GPU hours')
    def exhausted(signum,frame):raise TimeoutError('Declared GPU stage budget exhausted; retain checkpoint and count this attempt')
    old=signal.signal(signal.SIGALRM,exhausted);signal.setitimer(signal.ITIMER_REAL,min(remaining,projection))
    started=time.perf_counter();status='failed'
    try:yield;status='completed'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        atomic_json(folder/f'{stage}-{uuid.uuid4().hex}.json',dict(stage=stage,status=status,
            gpu_seconds=time.perf_counter()-started,slurm_job=os.environ.get('SLURM_JOB_ID'),
            includes='loading, prediction, diagnostics, checkpoints and retries',projected_seconds=projection))


@contextmanager
def budgeted_gpu_stage(cfg,stage):
    # Preparation and training may use different stage locks. This additional
    # scope-wide lock prevents concurrent attempts from spending the same
    # remaining budget. The shared helper fails fast instead of queueing GPUs.
    from ..temporal_gait.contracts import stage_lock
    with stage_lock(cfg.root,'gpu-budget'):
        with _budgeted_gpu_stage_unlocked(cfg,stage):
            yield
