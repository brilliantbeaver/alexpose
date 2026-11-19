"""Separate, explicitly supervised condition recognition on the new outer folds."""
from pathlib import Path
from dataclasses import asdict
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,balanced_accuracy_score,confusion_matrix
from .features import build_features,fit_rf,equal_source_weights
from .training import embed_windows
from .data import file_sha256

FOCAL=('normal','parkinsons','stroke','cerebral palsy','myopathic')


def condition_capstone(bundle,x,metadata,folder,seed=42):
    """Fixed-hyperparameter RF and frozen normal-JEPA probe, with equal sources.

    This task uses abnormal training labels. It is never described as a normal-
    only anomaly comparator. Inner normal reference sources stay excluded.
    A recording with several focal labels has one prediction row per label.
    These rows divide one source's evaluation weight equally, rather than
    treating its labels as additional independent recordings.
    """
    from .study_runner import dump_json
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=True)
    required={'label','source_id','sequence_id','outer_fold','split'}
    if len(x)!=len(metadata) or not required.issubset(metadata) or metadata[list(required)].isna().any().any():
        raise ValueError('Aligned nonmissing source, label and fold metadata are required')
    if metadata.groupby('source_id').split.nunique().gt(1).any():
        raise ValueError('A condition-classifier source occurs in multiple split roles')
    selected=metadata.label.isin(FOCAL).to_numpy();m=metadata.loc[selected].reset_index(drop=True);a=x[selected]
    fit=m.split.isin(['train','auxiliary']).to_numpy();test=m.split.eq('test').to_numpy()
    if set(m.loc[fit,'label'])!=set(FOCAL):raise ValueError('Five-class fitting needs every focal label')
    if not test.any():raise ValueError('Condition classifier needs outer-test sources')
    if set(m.loc[fit,'source_id']) & set(m.loc[test,'source_id']):
        raise ValueError('Condition fitting and test sources overlap')
    features,names=build_features(a);z=embed_windows(bundle.backbone,a,batch_size=32)
    rf=fit_rf(features[fit],m.loc[fit,'label'].to_numpy(),seed,groups=m.loc[fit,'source_id'].to_numpy())
    weights=equal_source_weights(m.loc[fit,'source_id'].to_numpy())
    y=m.loc[fit,'label'].to_numpy();total=weights.sum()
    for label in FOCAL:weights[y==label]*=total/(len(FOCAL)*weights[y==label].sum())
    scaler=StandardScaler().fit(z[fit],sample_weight=weights)
    probe=LogisticRegression(C=1.,max_iter=2000,random_state=seed).fit(scaler.transform(z[fit]),y,sample_weight=weights)
    allrows=[];metrics={}
    for name,model,values in [('random_forest',rf,features[test]),('normal_jepa_probe',probe,scaler.transform(z[test]))]:
        probabilities=model.predict_proba(values);labels=list(model.classes_)
        table=m.loc[test,['source_id','sequence_id','label','outer_fold']].reset_index(drop=True).copy()
        table[labels]=probabilities
        sequences=table.groupby(['source_id','sequence_id','label','outer_fold'],as_index=False)[labels].mean()
        sources=sequences.groupby(['source_id','label','outer_fold'],as_index=False)[labels].mean()
        sources['prediction']=np.array(labels)[sources[labels].to_numpy().argmax(1)]
        sources['source_weight']=1./sources.groupby('source_id').source_id.transform('size')
        sources['method']=name;allrows.append(sources)
        truth=sources.label.to_numpy();prediction=sources.prediction.to_numpy()
        weights=sources.source_weight.to_numpy()
        metrics[name]=dict(macro_f1=f1_score(truth,prediction,labels=list(FOCAL),average='macro',zero_division=0,sample_weight=weights),
            balanced_accuracy=balanced_accuracy_score(truth,prediction,sample_weight=weights),
            n_test_sources=int(sources.source_id.nunique()),n_test_source_label_pairs=len(sources),
            n_mixed_label_sources=int(sources.groupby('source_id').label.nunique().gt(1).sum()),
            confusion_matrix=confusion_matrix(truth,prediction,labels=list(FOCAL),sample_weight=weights).tolist(),labels=list(FOCAL),
            evaluation_unit='Source-label pairs with inverse label-count weights; each source totals one')
    keys=['source_id','label','outer_fold','source_weight']
    pd.testing.assert_frame_equal(allrows[0][keys],allrows[1][keys])
    result=pd.concat(allrows,ignore_index=True);result.to_csv(folder/'condition_predictions.csv',index=False)
    provenance=dict(supervision='normal and abnormal labels',metrics=metrics,
        fit_source_ids=sorted(m.loc[fit,'source_id'].unique()),test_source_ids=sorted(m.loc[test,'source_id'].unique()),
        normal_reference_sources_excluded=True,feature_names=names,probe_C=1.,hyperparameters_selected_from_test=False,
        training_weight_rule='Inverse source window count followed by focal-class balancing',
        evaluation_weight_rule='Each unique test source totals one across its focal source-label pairs',
        paired_methods_share_identical_source_label_support=True,seed=int(seed),
        encoder_provenance=bundle.provenance,encoder_config=asdict(bundle.backbone.cfg))
    checkpoint=folder/'trained.pt'
    provenance['encoder_checkpoint']=str(checkpoint.resolve()) if checkpoint.exists() else None
    provenance['encoder_checkpoint_sha256']=file_sha256(checkpoint) if checkpoint.exists() else None
    payload=dict(format_version=1,random_forest=rf,embedding_scaler=scaler,embedding_probe=probe,
                 feature_names=names,class_labels=list(FOCAL),provenance=provenance)
    temporary=folder/'condition_classifiers.joblib.tmp';destination=folder/'condition_classifiers.joblib'
    joblib.dump(payload,temporary);temporary.replace(destination)
    provenance['classifier_artifact']=str(destination.resolve())
    provenance['classifier_artifact_sha256']=file_sha256(destination)
    dump_json(folder/'condition_metrics.json',provenance)
    return result,metrics
