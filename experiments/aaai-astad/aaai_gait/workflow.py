"""Notebook orchestration with separate demo and GAVD artifacts.

Every run reads current CSVs. Checkpoint identity also includes exact retained
window bytes, split assignments, configuration, and implementation file hashes.
"""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import copy
import numpy as np
import pandas as pd
import torch
from .settings import ROOT, EXPERIMENT, Settings
from .data import build_manifest, grouped_split, manifest_digest, load_windows, LABELS, assert_group_disjoint
from .demo import synthetic_corpus
from .model import SJEPA, ModelConfig, save_checkpoint, load_checkpoint
from .training import seed_everything, train_model, progressive_train, embed_windows, fit_frozen_probe
from .features import build_features, fit_rf, equal_source_weights
from .evaluation import (align_probabilities, aggregate_sequence_predictions,
                         classification_metrics, cluster_bootstrap, calibrate_threshold,
                         anomaly_metrics)


def json_ready(value):
    if isinstance(value, dict): return {str(k):json_ready(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)): return [json_ready(v) for v in value]
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (float,np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, Path): return str(value)
    return value


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(json_ready(value),indent=2,allow_nan=False))


def prepare_corpus(settings=None):
    settings = settings or Settings()
    manifest = grouped_split(build_manifest(ROOT),seed=settings.seed)
    if settings.demo:
        x, meta = synthetic_corpus(settings.seed,settings.frames)
    else:
        x, meta = load_windows(manifest,settings.pose_dir,frames=settings.frames,
                              stride=settings.stride,target_fps=settings.target_fps,
                              expected_frame_number_base=settings.frame_number_base)
    meta['camera_views']=('synthetic' if settings.demo else meta.sequence_id.map(manifest.set_index('sequence_id').camera_views).fillna('unknown'))
    assert_group_disjoint(meta)
    h=hashlib.sha256()
    h.update(x.tobytes())
    h.update(meta.attrs.get('corpus_sha256','synthetic').encode())
    h.update(meta.to_csv(index=False).encode())
    h.update(manifest_digest(manifest).encode())
    config={**asdict(settings),'model':asdict(ModelConfig(frames=settings.frames))}
    h.update(json.dumps(json_ready(config),sort_keys=True).encode())
    # Reject stale checkpoints after a change to preprocessing or training code.
    for name in ('model.py','training.py','data.py','features.py','workflow.py','demo.py'):
        h.update((EXPERIMENT/'aaai_gait'/name).read_bytes())
    corpus_hash=h.hexdigest()
    split_hash=hashlib.sha256(meta[['sequence_id','source_id','split']].drop_duplicates().sort_values('sequence_id').to_csv(index=False).encode()).hexdigest()
    run_dir=settings.run_dir/corpus_hash[:16]
    run_dir.mkdir(parents=True,exist_ok=True)
    manifest.to_csv(run_dir/'fresh_manifest_and_split.csv',index=False)
    meta.to_csv(run_dir/'window_index.csv',index=False)
    write_json(run_dir/'provenance.json',dict(synthetic=settings.demo,corpus_sha256=corpus_hash,
        annotations_sha256=manifest_digest(manifest),split_sha256=split_hash,configuration=config,
        n_windows=len(x),n_sequences=meta.sequence_id.nunique(),n_sources=meta.source_id.nunique(),
        exclusions=meta.attrs.get('exclusions',[]),pose_model_sha256=meta.attrs.get('pose_model_sha256',[]),
        corpus_preprocessing_sha256=meta.attrs.get('corpus_sha256','synthetic')))
    return x,meta,dict(manifest=manifest,manifest_hash=corpus_hash,split_hash=split_hash,
                       run_dir=run_dir,synthetic=settings.demo)


def normal_anchor(x,meta,provenance,settings=None,*,spe=False,mode='masked'):
    settings=settings or Settings()
    cfg=ModelConfig(frames=settings.frames,spe=spe)
    path=provenance['run_dir']/f'normal_{mode}_{"spe" if spe else "fixed"}.pt'
    if path.exists():
        model,record=load_checkpoint(path,manifest_hash=provenance['manifest_hash'],split_hash=provenance['split_hash'])
        if record['config']!=asdict(cfg) or record['stage']!=f'normal_{mode}':
            raise ValueError('Incompatible model configuration or stage')
        return model,record['history']
    seed_everything(settings.seed)
    model=SJEPA(cfg)
    history=train_model(model,x,meta,allowed_labels=('normal',),epochs=settings.epochs,
                        batch_size=settings.batch_size,seed=settings.seed,mode=mode)
    save_checkpoint(model,path,manifest_hash=provenance['manifest_hash'],split_hash=provenance['split_hash'],stage=f'normal_{mode}',history=history)
    return model,history


def progressive_model(anchor,x,meta,provenance,settings=None):
    settings=settings or Settings()
    anchor_hash=hashlib.sha256(b''.join(value.detach().cpu().numpy().tobytes() for value in anchor.state_dict().values())).hexdigest()[:16]
    path=provenance['run_dir']/f'progressive_{anchor_hash}.pt'
    if path.exists():
        model,record=load_checkpoint(path,manifest_hash=provenance['manifest_hash'],split_hash=provenance['split_hash'])
        if record['config']!=asdict(anchor.cfg) or record['stage']!='progressive':
            raise ValueError('Incompatible progressive checkpoint')
        return model,record['history']
    model,history=progressive_train(anchor,x,meta,epochs=settings.epochs,
                                    batch_size=settings.batch_size,seed=settings.seed)
    save_checkpoint(model,path,manifest_hash=provenance['manifest_hash'],split_hash=provenance['split_hash'],stage='progressive',history=history)
    return model,history


def _evaluate(meta,probabilities):
    seq,p=aggregate_sequence_predictions(meta,probabilities,LABELS)
    weights=equal_source_weights(seq.source_id)
    return seq,p,dict(sequence_weighted=classification_metrics(seq.label,p,LABELS),
                     source_weighted=classification_metrics(seq.label,p,LABELS,sample_weight=weights))


def select_probe(z,meta,seed=42):
    """Choose regularization on sequence-level validation macro-F1 only."""
    val=meta.split.eq('val').to_numpy()
    candidates=[]
    for c in (.1,1.,10.):
        probe=fit_frozen_probe(z,meta,c=c,seed=seed)
        p=align_probabilities(probe.predict_proba(z[val]),probe.classes_,LABELS)
        _,_,metrics=_evaluate(meta.loc[val],p)
        candidates.append((metrics['source_weighted']['macro_f1'],c,probe))
    # Stable order favors stronger regularization for exact ties.
    score,c,probe=max(candidates,key=lambda row:row[0])
    return probe,dict(C=c,validation_source_macro_f1=score)


def capstone(x,meta,provenance,settings=None,bootstrap_repeats=1000):
    """Fit fixed candidate methods, select settings on validation, score test.

    Test results are descriptive for prespecified methods. The chosen JEPA stage
    uses validation macro-F1. Comparing reported test methods must not be used
    to retrospectively choose a new primary method.
    """
    settings=settings or Settings()
    anchor,_=normal_anchor(x,meta,provenance,settings)
    adapted,_=progressive_model(anchor,x,meta,provenance,settings)
    seed_everything(settings.seed)
    untrained=SJEPA(ModelConfig(frames=settings.frames))
    features,names=build_features(x,fps=settings.target_fps)
    train=meta.split.eq('train').to_numpy();val=meta.split.eq('val').to_numpy();test=meta.split.eq('test').to_numpy()
    rf_candidates=[]
    for depth in (4,8,None):
        estimator=fit_rf(features[train],meta.loc[train,'label'],seed=settings.seed,
                         groups=meta.loc[train,'source_id'],max_depth=depth)
        p=align_probabilities(estimator.predict_proba(features[val]),estimator.classes_,LABELS)
        _,_,metrics=_evaluate(meta.loc[val],p)
        rf_candidates.append((metrics['source_weighted']['macro_f1'],depth,estimator))
    score,depth,rf=max(rf_candidates,key=lambda row:row[0])
    methods={'random_forest':(rf,features,dict(max_depth=depth,validation_source_macro_f1=score))}
    for name,model in [('normal_jepa',anchor),('progressive_jepa',adapted),('untrained_encoder',untrained)]:
        z=embed_windows(model,x,batch_size=settings.batch_size)
        probe,selection=select_probe(z,meta,settings.seed)
        methods[name]=(probe,z,selection)
    observed=np.isfinite(x).sum(axis=1)
    mean_pose=np.divide(np.nansum(x,axis=1),observed,out=np.full(x.shape[:1]+x.shape[2:],np.nan,dtype=np.float32),where=observed>0).reshape(len(x),-1)
    pose_probe,pose_selection=select_probe(mean_pose,meta,settings.seed)
    methods['mean_pose']=(pose_probe,mean_pose,pose_selection)
    valid=np.isfinite(x).all(axis=-1).reshape(len(x),settings.frames//4,4,16).all(axis=2)
    availability=np.concatenate([valid.mean(axis=1),valid.mean(axis=2)],axis=1)
    from sklearn.preprocessing import OneHotEncoder
    view_encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
    view_encoder.fit(meta.loc[train,['camera_views']])
    availability=np.concatenate([availability,view_encoder.transform(meta[['camera_views']])],axis=1)
    missing_probe,missing_selection=select_probe(availability,meta,settings.seed)
    methods['pose_availability']=(missing_probe,availability,missing_selection)
    from sklearn.dummy import DummyClassifier
    dummy=DummyClassifier(strategy='prior').fit(features[train],meta.loc[train,'label'],sample_weight=equal_source_weights(meta.loc[train,'source_id']))
    methods['prior_baseline']=(dummy,features,dict(selection='training prior'))
    chosen=max(('normal_jepa','progressive_jepa'),key=lambda name:methods[name][2]['validation_source_macro_f1'])
    results={};predictions={};sequence_table=None
    for name,(estimator,z,selection) in methods.items():
        p=align_probabilities(estimator.predict_proba(z[test]),estimator.classes_,LABELS)
        seq,p,metrics=_evaluate(meta.loc[test],p)
        metrics['selection']=selection
        metrics['cluster_bootstrap_sequence_metrics']=cluster_bootstrap(seq.label,p,seq.source_id,LABELS,repeats=bootstrap_repeats,seed=settings.seed)
        results[name]=metrics;predictions[name]=p;sequence_table=seq
        table=seq.copy()
        for i,label in enumerate(LABELS):table[f'p_{label}']=p[:,i]
        table.to_csv(provenance['run_dir']/f'test_predictions_{name}.csv',index=False)
    paired=cluster_bootstrap(sequence_table.label,predictions[chosen],sequence_table.source_id,LABELS,
                              repeats=bootstrap_repeats,seed=settings.seed,other_probabilities=predictions['random_forest'])
    result=dict(synthetic=provenance['synthetic'],selected_jepa=chosen,methods=results,
                paired_selected_minus_rf_sequence_weighted=paired,feature_names=names,
                nuisance_features='valid token fractions by joint and time patch plus training-fitted camera-view one-hot',
                interpretation='Software demonstration only' if settings.demo else 'Internal source-held-out evaluation; participant independence unverified')
    write_json(provenance['run_dir']/'capstone_metrics.json',result)
    import joblib
    joblib.dump(dict(rf=rf,probe=methods[chosen][0],selected_jepa=chosen,classes=list(LABELS),
                     features=names,manifest_hash=provenance['manifest_hash'],split_hash=provenance['split_hash'],
                     synthetic=settings.demo,training_sequences=meta.loc[train,'sequence_id'].unique().tolist()),
                provenance['run_dir']/'classifiers.joblib')
    inference_path=provenance['run_dir']/'selected_encoder.pt'
    if not inference_path.exists():
        save_checkpoint(anchor if chosen=='normal_jepa' else adapted,inference_path,
                        manifest_hash=provenance['manifest_hash'],split_hash=provenance['split_hash'],stage=chosen)
    return result


def anomaly_evaluation(anchor,x,meta,*,batch_size=16,alpha=.05,mode='masked'):
    anchor=anchor.to('cpu').float()
    chunks=[]
    for start in range(0,len(x),batch_size):
        chunks.append(anchor.latent_residual(torch.as_tensor(x[start:start+batch_size],dtype=torch.float32),mode=mode)['score'].numpy())
    window=meta.copy();window['score']=np.concatenate(chunks)
    sequence=window.groupby('sequence_id').agg(label=('label','first'),source_id=('source_id','first'),split=('split','first'),score=('score','mean')).reset_index()
    # A source contributes one score per label, preserving mixed-label recordings.
    source=sequence.groupby(['source_id','label','split'],as_index=False).score.mean()
    val=source[(source.split=='val')&(source.label=='normal')]
    threshold=calibrate_threshold(val.score.to_numpy(),alpha)
    test=source[source.split=='test']
    metrics=anomaly_metrics(test.label.to_numpy()!='normal',test.score.to_numpy(),threshold)
    metrics.update(n_normal_calibration_sources=len(val),alpha=alpha,finite_threshold=bool(np.isfinite(threshold)),mode=mode,
                   unit='source and label; mixed-label source rows are dependent')
    return source,metrics
