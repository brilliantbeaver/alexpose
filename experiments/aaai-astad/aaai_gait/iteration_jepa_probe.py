"""Separate supervised five-class probe with freshly fitted normal-only JEPA.

This is not an encoder ablation for the retained coordinate estimator. It
refits both learned and random embedding probes under the new outer roles.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
import joblib
from .model import ModelConfig,SJEPA,uniform_mask,_visible_indices,_valid_pool,vicreg
from .training import embed_windows,representation_diagnostics
from .iteration_scoring import aggregate_descriptors,FOCAL
from .masking import rng_stream,stream_seed
from .study_runner import dump_json
from .data import file_sha256


def run_probe_fold(data,fold,folder,*,updates=512,seed=42):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    config=ModelConfig(frames=32,patch_frames=4,dim=64,depth=2,heads=4,mask_ratio=.5,vicreg_weight=.1)
    registry=data.source_registry.set_index('source_id');role=f'role_fold_{fold}'
    fit=registry.index[registry[role].eq('train')].tolist()
    supervised=registry.index[registry.five_class_eligible.astype(bool)&registry[role].isin(['train','auxiliary'])].tolist()
    test=registry.index[registry.five_class_eligible.astype(bool)&registry[role].eq('test')].tolist()
    identity=dict(outer_fold=fold,model=asdict(config),updates=updates,batch_size=16,seed=seed+fold,
        normal_fit_sources=fit,supervised_fit_sources=supervised,test_sources=test,
        code_sha256=file_sha256(Path(__file__)),pretrain_mask_policy='U',vicreg_view_policy='U',
        head_mask_policy='not_applicable_frozen_full_observation_linear_probe',
        evaluation_bank_hash='not_applicable_five_class_full_observation',
        selection='fixed budget and C1; no architecture or stopping selection',
        prior_results='historical draft01 five-class results were seen; exploratory refit')
    dump_json(folder/'protocol.json',identity)
    torch.set_num_threads(2);torch.manual_seed(seed+fold)
    model=SJEPA(config);random_model=SJEPA(config);random_model.load_state_dict(model.state_dict())
    source_rng=rng_stream(seed+fold,'jepa_source_order');window_rng=rng_stream(seed+fold,'jepa_window_selection')
    generators={name:torch.Generator().manual_seed(stream_seed(seed+fold,name)) for name in
                ['pretrain_mask','vicreg_mask','coordinate_noise','positional_noise']}
    pools={s:np.flatnonzero(data.metadata.source_id.eq(s).to_numpy()) for s in fit}
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.05)
    history=[];model.train()
    for step in range(updates):
        sources=source_rng.choice(fit,size=16,replace=True)
        indices=[window_rng.choice(pools[s]) for s in sources]
        x=torch.as_tensor(data.x[indices],dtype=torch.float32)
        first=uniform_mask(16,128,.5,generator=generators['pretrain_mask'])
        second=uniform_mask(16,128,.5,generator=generators['vicreg_mask'])
        optimizer.zero_grad(set_to_none=True)
        out=model(x,first,stochastic_positions=True,generator=generators['positional_noise'])
        prob=F.softmax((out['target']-model.target_center)/config.target_temperature,dim=-1)
        ce=-(prob*F.log_softmax(out['predicted']/config.student_temperature,dim=-1)).sum(-1)
        valid=out['target_mask'];prediction=((ce*valid).sum(1)/valid.sum(1)).mean()
        visible=_visible_indices(second);second_valid=out['valid'].gather(1,visible)
        noise=torch.randn(x.shape,generator=generators['coordinate_noise'])*.005
        context2=model.view_encoder(x+noise,visible)
        pooled=_valid_pool(out['context'],out['context_valid'])
        aux=vicreg(model.vicreg_projector(pooled),model.vicreg_projector(_valid_pool(context2,second_valid)))
        loss=prediction+.1*aux['vicreg']
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite JEPA objective')
        loss.backward();torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad),1.,error_if_nonfinite=True)
        optimizer.step();model.update_target(.996,out['target'][valid].mean(0).detach())
        history.append(dict(step=step+1,loss=float(loss.detach()),prediction=float(prediction.detach()),
            variance=float(aux['variance'].detach()),covariance=float(aux['covariance'].detach()),
            embedding_std=float(pooled.std(0,unbiased=False).mean().detach())))
    needed=set(fit+supervised+test);selected=data.metadata.source_id.isin(needed).to_numpy()
    meta=data.metadata.loc[selected].reset_index(drop=True);x=data.x[selected];tables=[];metrics={};probes={}
    for name,encoder in [('learned_jepa_probe',model),('random_encoder_probe',random_model)]:
        z=embed_windows(encoder,x,batch_size=64);features=aggregate_descriptors(z,list(range(z.shape[1])),meta)
        scaler=StandardScaler().fit(features.loc[supervised]);probe=LogisticRegression(C=1.,max_iter=2000,class_weight='balanced',random_state=seed+fold)
        probe.fit(scaler.transform(features.loc[supervised]),registry.loc[supervised,'five_class_label'])
        p=probe.predict_proba(scaler.transform(features.loc[test]));pred=np.asarray(probe.classes_)[p.argmax(1)]
        truth=registry.loc[test,'five_class_label'].to_numpy()
        table=pd.DataFrame(p,index=test,columns=probe.classes_).rename_axis('source_id').reset_index()
        table['truth']=truth;table['prediction']=pred;table['outer_fold']=fold;table['method']=name;tables.append(table)
        metrics[name]=dict(macro_f1=float(f1_score(truth,pred,labels=list(FOCAL),average='macro',zero_division=0)),
            balanced_accuracy=float(balanced_accuracy_score(truth,pred)),labels=list(FOCAL),
            confusion_matrix=confusion_matrix(truth,pred,labels=list(FOCAL)).tolist(),
            fit_normal_representation_diagnostics=representation_diagnostics(features.loc[fit].to_numpy()))
        probes[name]=dict(scaler=scaler,probe=probe)
        features.reset_index().to_csv(folder/f'{name}_source_embeddings.csv',index=False)
    result=pd.concat(tables,ignore_index=True);result.to_csv(folder/'five_class_predictions.csv',index=False)
    pd.DataFrame(history).to_csv(folder/'training.csv',index=False);joblib.dump(probes,folder/'probes.joblib')
    torch.save(dict(config=asdict(config),state_dict=model.state_dict(),normal_fit_sources=fit),folder/'learned_model.pt')
    torch.save(dict(config=asdict(config),state_dict=random_model.state_dict(),normal_fit_sources=[]),folder/'random_model.pt')
    status=dict(status='complete',metrics=metrics,wall_seconds=time.perf_counter()-started,
                interpretation='separate supervised classification; no coordinate-information or encoder-mask ablation claim')
    dump_json(folder/'status.json',status);return result,status
