"""One declared, source-disjoint matched RF versus JEPA experiment."""
import copy
import json
import joblib
from pathlib import Path
import time
import importlib.metadata
import ambient.pose.joint_angles as shared_angles
import ambient.pose.keypoint_extractor as shared_extractor
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, average_precision_score, confusion_matrix
from .data import LABELS, sha256_file
from .settings import PROJECT
from .workflow import load_prepared, json_write
from .features import build_features
from .evaluation import fit_rf, fit_probe, align_probabilities, aggregate_sources, classification_report, paired_source_bootstrap, validation_threshold
from .model import SkeletonJEPA, ModelConfig
from .training import TrainConfig, fit, encode, anomaly_scores, save_checkpoint, seed_everything, representation_diagnostics


def compare(input_dir, output_dir, epochs=5, seed=27):
    if epochs<1: raise ValueError('epochs must be positive')
    output=Path(output_dir)
    if output.exists(): raise FileExistsError('Choose a new result directory for each experiment')
    x,valid,meta=load_prepared(input_dir)
    for split in ('train','validation','test'):
        if set(meta.loc[meta.split==split,'label'])!=set(LABELS):
            raise ValueError(f'{split} lost a class during extraction; inspect attrition')
    output.mkdir(parents=True)
    pose_models=set()
    for fingerprint in meta.pose_fingerprint.unique():
        matches=list((PROJECT/'artifacts/poses').glob(f'*-{fingerprint[:20]}.npz'))
        if len(matches)!=1: raise ValueError('Pose provenance cache required for model identity')
        with np.load(matches[0],allow_pickle=False) as cached:
            pose_models.add(json.loads(str(cached['provenance_json']))['model_sha256'])
    if len(pose_models)!=1: raise ValueError('Mixed pose model weights in cohort')
    started=time.monotonic(); torch.set_num_threads(2); seed_everything(seed)
    train=(meta.split=='train').to_numpy(); normal=train & (meta.label=='normal').to_numpy()
    val=(meta.split=='validation').to_numpy(); test=(meta.split=='test').to_numpy()
    features,names=build_features(x,valid,fps=meta.sample_fps.to_numpy())
    model=SkeletonJEPA(ModelConfig(frames=x.shape[1],dim=48))
    random_embeddings=encode(model,x,valid)
    save_checkpoint(model,output/'untrained_encoder.pt')
    steps=max(2,epochs*int(np.ceil(normal.sum()/8)))
    def stage(net,selected,identity,regime,forecast_fraction=0.,updates=None):
        return fit(net,x[selected],valid[selected],TrainConfig(steps=updates or steps,seed=seed,regime=regime,
            run_id=identity,forecast_fraction=forecast_fraction),labels=meta.label[selected],
            source_ids=meta.source_id[selected],partitions=meta.split[selected])
    stage(model,normal,'normal-completion-v1','normal')
    save_checkpoint(model,output/'normal_completion.pt')
    forecast=copy.deepcopy(model);stage(forecast,normal,'normal-forecast-v1','forecast',1.)
    save_checkpoint(forecast,output/'normal_forecast.pt')
    adapted=copy.deepcopy(model); stages=[]
    for i,condition in enumerate(LABELS[1:],1):
        selected=train & meta.label.isin(LABELS[:i+1]).to_numpy()
        stage(adapted,selected,f'adaptation-stage-{i}-v1','adaptation')
        save_checkpoint(adapted,output/f'adaptation_{i}.pt');stages.append(copy.deepcopy(adapted.run_metadata))
    mixed=copy.deepcopy(model);stage(mixed,train,'mixed-adaptation-v1','adaptation',updates=steps*4)
    save_checkpoint(mixed,output/'mixed_adaptation.pt')
    matrices={'rf':features,'untrained_probe':random_embeddings,'jepa_normal':encode(model,x,valid),'jepa_adapted':encode(adapted,x,valid),'jepa_mixed':encode(mixed,x,valid)}
    reports={}; predictions={}; validation=[]
    for name,matrix in matrices.items():
        factory=fit_rf if name=='rf' else fit_probe
        candidates=[dict(max_depth=d) for d in (4,8)] if name=='rf' else [dict(C=c) for c in (.1,1.)]
        fitted=[]
        for params in candidates:
            estimator=factory(matrix[train],meta.label[train],meta.source_id[train],seed=seed,**params)
            _,vy,vp=aggregate_sources(align_probabilities(estimator,matrix[val]),meta.label[val],meta.source_id[val],meta.sequence_id[val])
            score=balanced_accuracy_score(vy!='normal',(1-vp[:,0])>=.5)
            validation.append(dict(model=name,parameters=params,binary_balanced_accuracy=float(score)))
            fitted.append((score,estimator,params))
        _,estimator,chosen=max(fitted,key=lambda item:item[0])
        joblib.dump(estimator,output/f'{name}_classifier.joblib')
        ids,y,p=aggregate_sources(align_probabilities(estimator,matrix[test]),meta.label[test],meta.source_id[test],meta.sequence_id[test])
        reports[name]=dict(classification_report(y,p),selected_parameters=chosen);predictions[name]=(ids,y,p)
        frame=pd.DataFrame(p,columns=LABELS);frame.insert(0,'label',y);frame.insert(0,'source_id',ids)
        frame.to_csv(output/f'{name}_source_predictions.csv',index=False)
    a,b=predictions['jepa_adapted'],predictions['rf']
    assert np.array_equal(a[0],b[0]) and np.array_equal(a[1],b[1])
    paired=paired_source_bootstrap(a[1],a[2],b[2],repetitions=1000,seed=seed)
    paired_macro=paired_source_bootstrap(a[1],a[2],b[2],repetitions=1000,seed=seed,metric='macro_f1')
    forecast_output=anomaly_scores(forecast,x,valid,context_frames=8)
    scores=forecast_output['coordinate_mse']
    persistence=[]; matched_forecast=[]
    for window,mask,prediction in zip(x,valid,forecast_output['coordinates']):
        last=np.zeros((x.shape[2],2)); known=np.zeros(x.shape[2],bool)
        for t in range(8):
            last[mask[t]]=window[t,mask[t]]; known|=mask[t]
        target=mask[8:] & known[None,:]
        persistence.append(float(((window[8:]-last)**2).mean(-1)[target].mean()))
        matched_forecast.append(float(((window[8:]-prediction[8:])**2).mean(-1)[target].mean()))
    nval=val & (meta.label=='normal').to_numpy()
    _,_,cal=aggregate_sources(scores[nval,None],meta.label[nval],meta.source_id[nval],meta.sequence_id[nval])
    threshold=validation_threshold(cal[:,0],false_positive_rate=.05)
    ids,truth,s=aggregate_sources(scores[test,None],meta.label[test],meta.source_id[test],meta.sequence_id[test])
    ybin=truth!='normal';pred=s[:,0]>threshold
    tn,fp,fn,tp=confusion_matrix(ybin,pred,labels=[False,True]).ravel()
    anomaly=dict(score='future_coordinate_mse',auroc=float(roc_auc_score(ybin,s[:,0])),
        average_precision=float(average_precision_score(ybin,s[:,0])),prevalence=float(ybin.mean()),
        threshold=float(threshold) if np.isfinite(threshold) else None,
        threshold_status='finite' if np.isfinite(threshold) else 'insufficient_normal_sources_for_finite_5pct_threshold',
        normal_calibration_sources=len(cal),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)))
    pd.DataFrame(dict(source_id=ids,label=truth,coordinate_mse=s[:,0])).to_csv(output/'anomaly_source_scores.csv',index=False)
    _,_,forecast_source_errors=aggregate_sources(np.array(matched_forecast)[test,None],meta.label[test],meta.source_id[test],meta.sequence_id[test])
    _,_,persistence_source_errors=aggregate_sources(np.array(persistence)[test,None],meta.label[test],meta.source_id[test],meta.sequence_id[test])
    normal_diagnostic_rows=meta[normal].groupby('source_id',sort=True).head(1).index.to_numpy()
    train_diagnostic_rows=meta[train].groupby('source_id',sort=True).head(1).index.to_numpy()
    preparation=json.loads((Path(input_dir)/'preparation.json').read_text())
    summary=dict(identity='medwm-gait-v1',status='completed_pilot' if preparation['mode']!='all-sequences' or preparation['max_frames'] else 'completed_declared_run',
        pose_model_sha256=next(iter(pose_models)),
        seed=seed,epochs_parameter=epochs,steps_per_stage=steps,n_windows=len(x),
        support=meta.groupby(['split','label']).agg(windows=('label','size'),sources=('source_id','nunique')).reset_index().to_dict('records'),
        representation_diagnostics={'normal':representation_diagnostics(model,x[normal_diagnostic_rows],valid[normal_diagnostic_rows]),'adapted':representation_diagnostics(adapted,x[train_diagnostic_rows],valid[train_diagnostic_rows])},
        classifiers=reports,paired_jepa_minus_rf=paired,paired_macro_f1=paired_macro,anomaly=anomaly,seconds=time.monotonic()-started,
        input_sha256=sha256_file(Path(input_dir)/'windows.npz'),metadata_sha256=sha256_file(Path(input_dir)/'windows.csv'),
        features=names,validation_selection=validation,
        forecast_mse_mean=float(forecast_source_errors.mean()),persistence_mse_mean=float(persistence_source_errors.mean()),
        artifact_sha256={p.name:sha256_file(p) for p in output.iterdir() if p.suffix in ('.pt','.joblib')},
        person_selection='full_image_first_detection_requires_manual_review',
        preprocessing='pre-normalized articulation; random masks are not raw-coordinate conditional predictions',
        versions={name:importlib.metadata.version(name) for name in ('torch','numpy','scikit-learn','mediapipe')},
        ambient_source_sha256={'joint_angles':sha256_file(shared_angles.__file__),'keypoint_extractor':sha256_file(shared_extractor.__file__)},
        code_sha256={p.name:sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
        limitations=['Single-run pilot is not clinical validation.', 'Source bootstrap does not cover training-seed uncertainty.',
                    'Curriculum and mixed adaptation match updates but not per-condition exposure.'])
    json_write(output/'results.json',summary);json_write(output/'adaptation_stages.json',stages)
    print(f'Completed {len(x)} windows; results: {output / "results.json"}')
    return summary
