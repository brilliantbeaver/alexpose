"""Bounded, explicitly adapted STG-NF comparator from the authors' code.

The graph, missing-data adapter, coordinate normalization and fitting budget
differ from the published surveillance-video recipe. This is not a faithful
reproduction and does not resolve clinical external-validity limitations.
"""
from pathlib import Path
import sys, time, warnings, subprocess
import numpy as np
import pandas as pd
import torch
from .data import JOINTS, file_sha256
from .study_runner import dump_json

UPSTREAM = 'https://github.com/orhir/STG-NF'
REVISION = 'edb5f3220332e160e4d20ce258787d5e2d7e0200'


def _load_model(vendor):
    vendor = Path(vendor)
    observed = subprocess.check_output(['git','-C',str(vendor),'rev-parse','HEAD'],text=True).strip()
    if observed != REVISION:
        raise ValueError('The inspected STG-NF revision changed')
    sys.path.insert(0,str(vendor))
    import models.STG_NF.model_pose as original
    from models.STG_NF.graph import get_hop_distance, normalize_digraph
    class BlazeGraph:
        def __init__(self, strategy='uniform', max_hop=1, **kwargs):
            if strategy != 'uniform':
                raise ValueError('Only the frozen uniform graph partition is supported')
            edges = [(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),
                     (23,25),(25,27),(27,29),(29,31),(27,31),
                     (24,26),(26,28),(28,30),(30,32),(28,32)]
            idx = {j:k for k,j in enumerate(JOINTS)}
            links = [(idx[a],idx[b]) for a,b in edges]+[(j,j) for j in range(16)]
            hops = get_hop_distance(16,links,max_hop=max_hop)
            self.A = normalize_digraph((hops<=max_hop).astype(float))[None].astype(np.float32)
    # Adapt only the graph factory at runtime; retain upstream files unchanged.
    original.Graph = BlazeGraph
    return original.STG_NF


def run_pose_baseline(data, fold, folder, *, steps=128, seed=42, batch_size=32):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    vendor=Path(__file__).resolve().parents[1]/'vendor/STG-NF'
    started=time.perf_counter(); registry=data.source_registry
    roles=registry.set_index('source_id')[f'role_fold_{fold}']
    fit_sources=sorted(roles[roles.eq('train')].index); test_sources=sorted(roles[roles.eq('test')].index)
    meta=data.metadata.reset_index(drop=True);fit=meta.source_id.isin(fit_sources).to_numpy()
    test=meta.source_id.isin(test_sources).to_numpy();x=np.asarray(data.x,float)
    if meta.loc[fit,'binary_label'].ne(0).any() or set(fit_sources)&set(test_sources):
        raise ValueError('Pose baseline roles violate normal-only source separation')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        # One median XY for each joint, using only fitting-normal observations.
        medians=np.nanmedian(x[fit].reshape(-1,16,2),axis=0)
    if not np.isfinite(medians).all():
        raise ValueError('A fitted joint median is undefined; no zero substitution is allowed')
    filled=np.where(np.isfinite(x),x,medians).transpose(0,3,1,2).astype(np.float32)
    tensor=torch.from_numpy(filled);torch.manual_seed(seed+fold);torch.set_num_threads(2)
    Model=_load_model(vendor)
    model=Model(pose_shape=(2,32,16),hidden_channels=0,K=4,L=1,actnorm_scale=1.,
                flow_permutation='reverse',flow_coupling='affine',LU_decomposed=False,
                learn_top=False,R=0,edge_importance=False,temporal_kernel_size=9,
                strategy='uniform',max_hops=1,device='cpu')
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    rng=np.random.default_rng(seed+fold+30000)
    source_rows={s:np.flatnonzero(meta.source_id.eq(s).to_numpy()) for s in fit_sources}
    history=[];model.train()
    for step in range(steps):
        selected=rng.choice(fit_sources,size=batch_size,replace=True)
        ix=np.array([rng.choice(source_rows[s]) for s in selected])
        optimizer.zero_grad();_,nll=model(tensor[ix]);loss=nll.mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite STG-NF loss at step {step}')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        history.append(dict(step=step+1,mean_nll=float(loss.detach())))
    model.eval();indices=np.flatnonzero(test);pred=[]
    with torch.no_grad():
        for start in range(0,len(indices),64):
            _,value=model(tensor[indices[start:start+64]])
            pred.extend(value.numpy().tolist())
    table=meta.loc[test,['source_id','sequence_id','window_index']].copy()
    table['stgnf_adapted']=pred;table['missing_fraction']=np.isnan(x[test]).mean(axis=(1,2,3))
    seq=table.groupby(['source_id','sequence_id'])[['stgnf_adapted','missing_fraction']].mean()
    result=seq.groupby('source_id').mean().reset_index();result['outer_fold']=fold
    if not np.isfinite(result.stgnf_adapted).all():
        raise FloatingPointError('STG-NF test predictions contain nonfinite values')
    table.to_csv(folder/'window_scores.csv.gz',index=False,compression='gzip')
    result.to_csv(folder/'source_scores.csv',index=False);pd.DataFrame(history).to_csv(folder/'training.csv',index=False)
    torch.save(dict(state_dict=model.state_dict(),medians=medians,fit_sources=fit_sources,
                    model_config=dict(K=4,L=1,hidden_channels=0,temporal_kernel_size=9,max_hops=1)),folder/'model.pt')
    audit=dict(status='complete_adapted',upstream=UPSTREAM,revision=REVISION,
               upstream_file_sha256=file_sha256(vendor/'models/STG_NF/model_pose.py'),
               source_code_unmodified=True,fit_sources=fit_sources,test_sources=test_sources,
               steps=steps,batch_size=batch_size,seed=seed+fold,parameter_count=sum(p.numel() for p in model.parameters()),
               normal_only=True,selection='fixed engineering budget, no test selection',
               adaptations=['16-joint BlazePose anatomical graph','pelvis/torso-normalized 15 Hz coordinates',
                            'training-normal per-joint median imputation, not native missing-data inference',
                            'four flow steps, one level, temporal kernel9, maxhop1',
                            'source-balanced batches and equal sequence/source aggregation'],
               limitations=['density can respond to imputation artifacts','budget-limited adapted comparator, not faithful reproduction'],
               imputation_medians=medians.tolist(),wall_seconds=time.perf_counter()-started)
    dump_json(folder/'status.json',audit)
    return result,audit
