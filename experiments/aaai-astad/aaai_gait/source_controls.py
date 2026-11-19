"""Supplementary source-level motion and observation controls.

These controls preserve the primary study's windows, folds and common test
sources. Two distances fit normal sources only. The nuisance Random Forest
uses abnormal training labels and is reported as a different supervision task.
No model score, test outcome or reference-source measurement selects settings.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from .features import build_features
from .model import JOINT_INDICES
from .study_data import source_table

AVAILABILITY_COLUMNS=tuple(f'availability_{joint}' for joint in JOINT_INDICES)
SCORE_COLUMNS=('motion_normal_distance','nuisance_normal_distance','nuisance_supervised_probability')


def _hierarchical_mean(values,metadata):
    table=pd.DataFrame(values,index=metadata.index)
    table['source_id']=metadata.source_id.to_numpy();table['sequence_id']=metadata.sequence_id.to_numpy()
    columns=list(values.columns) if isinstance(values,pd.DataFrame) else list(range(np.shape(values)[1]))
    sequence=table.groupby(['source_id','sequence_id'],sort=True)[columns].mean()
    return sequence.groupby('source_id',sort=True)[columns].mean()


def aggregate_source_features(x,metadata):
    """Equal windows within sequence, then equal sequences within source.

    Motion features summarize observed values only. Joint availability is the
    proportion of frames with both XY coordinates finite, not a confidence or
    disease feature. Entirely unavailable motion features remain NaN until the
    training-only imputer. Every recording appears once, including recordings
    carrying more than one abnormal category.
    """
    x=np.asarray(x);meta=metadata.reset_index(drop=True).copy()
    if x.shape!=(len(meta),32,16,2) or not len(meta) or np.isinf(x).any():
        raise ValueError('Aligned nonempty [windows,32,16,2] with finite or NaN coordinates required')
    if not {'source_id','sequence_id','label','fps','camera_views'}.issubset(meta):
        raise ValueError('Source, sequence, label, fps and camera-view metadata are required')
    fps=meta.fps.to_numpy(float)
    if not np.isfinite(fps).all() or not np.allclose(fps,fps[0],atol=0,rtol=0):
        raise ValueError('The shared windows require one fixed positive resampling FPS')
    values,names=build_features(x,float(fps[0]))
    motion=_hierarchical_mean(pd.DataFrame(values,columns=names),meta)
    available=np.isfinite(x).all(axis=-1).mean(axis=1)
    availability=_hierarchical_mean(pd.DataFrame(available,columns=AVAILABILITY_COLUMNS),meta)
    info=source_table(meta).set_index('source_id')
    result=info.join(motion).join(availability).reset_index()
    result.attrs['motion_columns']=names
    result.attrs['availability_columns']=list(AVAILABILITY_COLUMNS)
    result.attrs['aggregation']='Mean observed window features within sequence, then equal sequence means within source'
    return result,names


def _views(value):
    if pd.isna(value) or not str(value).strip():return ('<missing_view>',)
    # The refreshed manifest joins distinct annotation camera values by comma.
    return tuple(sorted({part.strip() for part in str(value).split(',') if part.strip()}))


def fit_view_vocabulary(metadata,fit_source_ids):
    fit=set(map(str,fit_source_ids));rows=metadata[metadata.source_id.isin(fit)]
    if not fit or set(rows.source_id)!=fit:raise ValueError('View fitting source IDs are missing')
    return tuple(sorted({view for value in rows.camera_views for view in _views(value)}))


def source_view_features(metadata,vocabulary):
    """Train-vocabulary one-hot fractions, with unseen views reported separately.

    A sequence annotated with multiple camera views gives those views equal
    weight. Windows are averaged within sequence, then sequences within source.
    Unknown categories map to zero, as handle_unknown='ignore'. The unknown
    flags are diagnostic outputs, not additional predictor inputs.
    Camera views come from GAVD annotations, not a video view estimator. A
    multi-view sequence therefore uses its annotation-level view mixture.
    """
    meta=metadata.reset_index(drop=True);vocabulary=tuple(vocabulary)
    if len(set(vocabulary))!=len(vocabulary):raise ValueError('Duplicate view vocabulary')
    parsed=[_views(value) for value in meta.camera_views]
    values=np.zeros((len(meta),len(vocabulary)),float);unknown=[];unknown_names={}
    index={view:i for i,view in enumerate(vocabulary)}
    for row,views in enumerate(parsed):
        for view in views:
            if view in index:values[row,index[view]]+=1/len(views)
            else:unknown_names.setdefault(str(meta.iloc[row].source_id),set()).add(view)
        unknown.append(sum(view not in index for view in views)/len(views))
    columns=[f'view_{view}' for view in vocabulary]
    features=_hierarchical_mean(pd.DataFrame(values,columns=columns),meta)
    diagnostics=_hierarchical_mean(pd.DataFrame({'unknown_view_fraction':unknown}),meta)
    diagnostics['unknown_view_present']=diagnostics.index.map(lambda s:bool(unknown_names.get(str(s))))
    diagnostics['unknown_views']=diagnostics.index.map(lambda s:'|'.join(sorted(unknown_names.get(str(s),set()))))
    return features,diagnostics


@dataclass
class TrainOnlyTransform:
    """Finite training medians, explicit zero fallback, and training-only scale."""
    medians:np.ndarray
    scaler:StandardScaler
    all_missing_columns:np.ndarray

    @classmethod
    def fit(cls,values):
        array=np.asarray(values,float)
        if array.ndim!=2 or len(array)<2 or np.isinf(array).any():
            raise ValueError('At least two training sources and finite/NaN features required')
        medians=np.array([np.median(column[np.isfinite(column)]) if np.isfinite(column).any() else 0. for column in array.T])
        missing=~np.isfinite(array).any(axis=0)
        filled=np.where(np.isfinite(array),array,medians)
        return cls(medians,StandardScaler().fit(filled),missing)

    def transform(self,values):
        array=np.asarray(values,float)
        if array.ndim!=2 or array.shape[1]!=len(self.medians) or np.isinf(array).any():
            raise ValueError('Feature dimension changed or contains infinite values')
        return self.scaler.transform(np.where(np.isfinite(array),array,self.medians))


def _distance_fit(values):
    transform=TrainOnlyTransform.fit(values);model=LedoitWolf().fit(transform.transform(values))
    return transform,model


def _distance_score(transform,model,values):
    scores=model.mahalanobis(transform.transform(values))
    if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite source-control distance')
    return np.maximum(scores,0.)


def fit_source_controls(features,window_metadata,roles,fold,seed=42,motion_columns=None):
    """Fit fixed controls on frozen scientific roles and score outer-test sources.

    Normal reference sources are excluded from every fit. The supervised
    nuisance classifier additionally uses auxiliary abnormal training sources.
    Its probabilities have different supervision from the two anomaly scores.
    """
    if set(roles)!={'train','val','test','auxiliary'} or fold not in range(4):
        raise ValueError('Four frozen source roles and a declared outer fold required')
    flattened=[str(source) for values in roles.values() for source in values]
    if len(flattened)!=len(set(flattened)):raise ValueError('Scientific source roles overlap')
    if features.source_id.duplicated().any() or set(features.source_id)!=set(flattened):
        raise ValueError('Each frozen source must occur exactly once in the feature table')
    observed=source_table(window_metadata).set_index('source_id')
    table=features.set_index('source_id')
    if set(observed.index)!=set(table.index) or not observed.binary_label.eq(table.binary_label.reindex(observed.index)).all():
        raise ValueError('Source features disagree with the fresh metadata labels')
    train=sorted(roles['train']);test=sorted(roles['test']);supervised=sorted(train+list(roles['auxiliary']))
    if len(train)<2 or not test or table.loc[train+list(roles['val']),'binary_label'].ne(0).any():
        raise ValueError('Normal-only fitting/reference roles or test support are invalid')
    if table.loc[list(roles['auxiliary']),'binary_label'].ne(1).any():raise ValueError('Auxiliary fitting sources must be abnormal')
    if set(table.loc[supervised,'binary_label'])!={0,1}:raise ValueError('Supervised nuisance fitting needs both binary classes')
    names=list(motion_columns or features.attrs.get('motion_columns',[]))
    if len(names)!=36 or not set(names+list(AVAILABILITY_COLUMNS)).issubset(table):
        raise ValueError('The declared 36 motion features and sixteen availability fields are required')
    motion_transform,motion_model=_distance_fit(table.loc[train,names].to_numpy())
    normal_views=fit_view_vocabulary(window_metadata,train)
    view_features,view_diagnostics=source_view_features(window_metadata,normal_views)
    nuisance=table[list(AVAILABILITY_COLUMNS)].join(view_features)
    nuisance_names=list(nuisance.columns)
    nuisance_transform,nuisance_model=_distance_fit(nuisance.loc[train].to_numpy())
    supervised_views=fit_view_vocabulary(window_metadata,supervised)
    supervised_features,supervised_diagnostics=source_view_features(window_metadata,supervised_views)
    supervised_matrix=table[list(AVAILABILITY_COLUMNS)].join(supervised_features)
    supervised_names=list(supervised_matrix.columns)
    rf_transform=TrainOnlyTransform.fit(supervised_matrix.loc[supervised].to_numpy())
    rf=RandomForestClassifier(n_estimators=300,max_depth=4,min_samples_leaf=2,
        max_features='sqrt',class_weight='balanced',random_state=int(seed)+int(fold),n_jobs=2)
    rf.fit(rf_transform.transform(supervised_matrix.loc[supervised].to_numpy()),table.loc[supervised,'binary_label'].to_numpy(int))
    out=table.loc[test,['binary_label','categories','n_sequences']].copy()
    out['outer_fold']=int(fold);out['split']='test'
    out['motion_normal_distance']=_distance_score(motion_transform,motion_model,table.loc[test,names].to_numpy())
    out['nuisance_normal_distance']=_distance_score(nuisance_transform,nuisance_model,nuisance.loc[test].to_numpy())
    out['nuisance_supervised_probability']=rf.predict_proba(rf_transform.transform(supervised_matrix.loc[test].to_numpy()))[:,list(rf.classes_).index(1)]
    out=out.join(view_diagnostics.add_prefix('normal_')).join(supervised_diagnostics.add_prefix('supervised_')).reset_index()
    if not np.isfinite(out[list(SCORE_COLUMNS)].to_numpy()).all():raise FloatingPointError('Nonfinite supplementary score')
    models=dict(motion_transform=motion_transform,motion_model=motion_model,
        nuisance_transform=nuisance_transform,nuisance_model=nuisance_model,
        supervised_transform=rf_transform,supervised_model=rf,
        motion_features=names,nuisance_features=nuisance_names,supervised_features=supervised_names,
        normal_view_vocabulary=normal_views,supervised_view_vocabulary=supervised_views)
    audit=dict(outer_fold=int(fold),seed=int(seed),source_roles={k:sorted(v) for k,v in roles.items()},
        normal_fit_source_ids=train,supervised_fit_source_ids=supervised,normal_reference_sources_excluded=True,
        fit_unit='One recording-source row; windows within sequence then equal sequences within source',
        normal_motion_supervision='Normal fitting sources only',normal_nuisance_supervision='Normal fitting sources only',
        nuisance_rf_supervision='Binary normal and abnormal training labels; different supervision from anomaly detectors',
        camera_view_source='GAVD sequence annotation; not inferred from the video',
        normal_view_vocabulary=list(normal_views),supervised_view_vocabulary=list(supervised_views),
        unknown_view_rule='Ignore unseen categories in features; report unknown flags outside predictors',
        all_missing_motion_features=[name for name,missing in zip(names,motion_transform.all_missing_columns) if missing],
        motion_training_medians=motion_transform.medians.tolist(),nuisance_training_medians=nuisance_transform.medians.tolist(),
        motion_covariance_shrinkage=float(motion_model.shrinkage_),nuisance_covariance_shrinkage=float(nuisance_model.shrinkage_),
        motion_covariance_rank=int(np.linalg.matrix_rank(motion_model.covariance_)),
        nuisance_covariance_rank=int(np.linalg.matrix_rank(nuisance_model.covariance_)),
        rf_parameters=rf.get_params(),hyperparameters_selected_from_test=False,
        imputation='Training-source median; all-missing training columns become zero, without reading evaluation rows',
        distance='Squared Mahalanobis distance from LedoitWolf covariance on training-standardized source features')
    return out,models,audit


def align_frozen_windows(x,current_metadata,frozen_metadata):
    """Recover exactly the recorded window order, with nanosecond time checks."""
    keys=['window_id','source_id','sequence_id']
    if len(x)!=len(current_metadata) or not set(keys+['start_seconds']).issubset(current_metadata) or not set(keys+['start_seconds']).issubset(frozen_metadata):
        raise ValueError('Frozen alignment requires window, source, sequence and start time')
    if current_metadata.duplicated(keys).any() or frozen_metadata.duplicated(keys).any():
        raise ValueError('Window identity must be unique')
    current=current_metadata.reset_index(drop=True).copy();current['_array_index']=np.arange(len(current))
    joined=frozen_metadata[keys+['start_seconds']].merge(current[keys+['start_seconds','_array_index']],
        on=keys,how='left',suffixes=('_frozen','_current'),sort=False,validate='one_to_one')
    if joined._array_index.isna().any():raise ValueError('A frozen window is missing from the current corpus')
    if not np.allclose(joined.start_seconds_frozen,joined.start_seconds_current,atol=.5e-9,rtol=0):
        raise ValueError('A frozen window start time differs')
    indices=joined._array_index.to_numpy(int)
    for name in ('label','binary_label','fps','camera_views'):
        if name in frozen_metadata and name in current:
            a=frozen_metadata[name].reset_index(drop=True).fillna('')
            b=current.iloc[indices][name].reset_index(drop=True).fillna('')
            if not a.eq(b).all():raise ValueError(f'Frozen window {name} differs')
    return np.asarray(x)[indices],frozen_metadata.reset_index(drop=True).copy()


def run_source_controls(run_dir):
    """Evaluate supplements after a completed real four-fold primary run."""
    import importlib.metadata
    import joblib
    from .study_runner import prepare_study,dump_json
    from .data import file_sha256
    from .predictive_profile import fold_ranking_metrics,paired_fold_bootstrap
    run_dir=Path(run_dir).resolve()
    protocol=json.loads((run_dir/'protocol.json').read_text())
    summary=json.loads((run_dir/'summary.json').read_text())
    if protocol['synthetic'] or summary.get('status')!='completed':
        raise ValueError('Supplementary execution requires a completed real four-fold study')
    x,meta,_,_=prepare_study(demo=False,seed=int(protocol['seed']))
    if meta.attrs['corpus_sha256']!=protocol.get('pre_profile_corpus_sha256'):
        raise ValueError('Current annotation or pose corpus differs from the frozen primary study')
    index_path=run_dir/'window_index.csv'
    if file_sha256(index_path)!=protocol.get('retained_metadata_sha256'):
        raise ValueError('Frozen window-index fingerprint changed')
    if protocol.get('code',{}).get('features.py')!=file_sha256(Path(__file__).parent/'features.py'):
        raise ValueError('The primary motion-feature implementation changed')
    frozen=pd.read_csv(index_path,dtype={'source_id':str,'sequence_id':str,'window_id':str})
    x,meta=align_frozen_windows(x,meta,frozen)
    features,motion_names=aggregate_source_features(x,meta)
    primary_path=run_dir/'paired_source_scores.csv'
    primary=pd.read_csv(primary_path,dtype={'source_id':str})
    primary_columns=['trained_latent_all_profile_score','trained_latent_all_raw_full']
    if not set(primary_columns+['all_methods_scorable']).issubset(primary):
        raise ValueError('Primary common-source scores are missing')
    if primary.source_id.duplicated().any():raise ValueError('Primary test sources are duplicated')
    primary=primary[primary.all_methods_scorable.eq(True)].copy()
    keys=['source_id','outer_fold','binary_label','categories']
    identity=dict(parent_run_id=protocol['run_id'],parent_protocol_sha256=file_sha256(run_dir/'protocol.json'),
        corpus_sha256=protocol['corpus'],window_index_sha256=file_sha256(index_path),
        primary_scores_sha256=file_sha256(primary_path),
        source_control_code_sha256=file_sha256(Path(__file__)),
        cli_code_sha256=file_sha256(Path(__file__).parents[1]/'scripts/run_source_controls.py'),
        roles_sha256={str(f):file_sha256(run_dir/f'fold_{f}/source_roles.json') for f in range(4)},
        software={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn')},
        settings=dict(covariance='LedoitWolf defaults',scaler='StandardScaler training only',
            rf_trees=300,rf_depth=4,rf_min_leaf=2,rf_class_weight='balanced',seed=int(protocol['seed'])))
    control_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    out=run_dir/'source_controls'/control_id
    if (out/'summary.json').exists():
        stored=json.loads((out/'summary.json').read_text())
        if not all(file_sha256(out/name)==digest for name,digest in stored['artifact_sha256'].items()):
            raise ValueError('Existing supplementary artifact changed')
        return out
    if out.exists():raise FileExistsError('Incomplete supplementary output exists; preserve it before retrying')
    out.mkdir(parents=True)
    dump_json(out/'protocol.json',dict(**identity,control_id=control_id,primary_hypothesis_unchanged=True,
        source_unit='One source video; not a verified participant',
        normal_only_controls=list(SCORE_COLUMNS[:2]),separately_supervised_control=SCORE_COLUMNS[2],
        supervision_caveat='The supervised nuisance RF is a shortcut diagnostic, not a normal-only anomaly method',
        interpretation_limit='Nuisance prediction establishes an observation/label association, not proof that JEPA uses that shortcut',
        multiplicity='Supplementary descriptive comparisons; no multiplicity-adjusted significance claim',
        selection='Fixed primary windows, folds, fitting roles and primary common-source cohort; no outcome tuning'))
    features.to_csv(out/'source_features.csv',index=False)
    scores=[]
    for fold in range(4):
        roles=json.loads((run_dir/f'fold_{fold}/source_roles.json').read_text())
        result,models,audit=fit_source_controls(features,meta,roles,fold,int(protocol['seed']),motion_names)
        scores.append(result);dump_json(out/f'fold_{fold}_fitting.json',audit)
        joblib.dump(dict(models=models,audit=audit,identity=identity),out/f'fold_{fold}_models.joblib')
    scores=pd.concat(scores,ignore_index=True)
    if scores.source_id.duplicated().any() or set(scores.source_id)!=set(features.source_id):
        raise ValueError('Frozen folds do not yield exactly one outer-test prediction per source')
    scores.to_csv(out/'source_scores.csv',index=False)
    paired=primary[keys+primary_columns].merge(scores,on=keys,how='left',validate='one_to_one')
    if len(paired)!=len(primary) or paired[list(SCORE_COLUMNS)].isna().any().any():
        raise ValueError('A supplementary control lost a primary common source')
    paired.to_csv(out/'paired_common_source_scores.csv',index=False)
    metrics=fold_ranking_metrics(paired,primary_columns+list(SCORE_COLUMNS))
    dump_json(out/'ranking_metrics.json',metrics)
    pd.DataFrame([dict(method=name,auroc=value['primary_auroc'],average_precision=value['normal_weighted_ap'],
        supervision='binary source labels' if name==SCORE_COLUMNS[2] else 'normal fitting sources only')
        for name,value in metrics['methods'].items()]).to_csv(out/'ranking_metrics.csv',index=False)
    intervals={name:paired_fold_bootstrap(paired,primary_columns[0],name,repeats=1000,seed=int(protocol['seed'])) for name in SCORE_COLUMNS}
    dump_json(out/'paired_bootstrap.json',intervals)
    artifacts={p.name:file_sha256(p) for p in out.iterdir() if p.is_file()}
    dump_json(out/'summary.json',dict(status='completed',control_id=control_id,parent_run_id=protocol['run_id'],
        n_sources=len(scores),n_primary_common_sources=len(paired),artifact_sha256=artifacts,
        unknown_normal_view_sources=int(scores.normal_unknown_view_present.sum()),
        unknown_supervised_view_sources=int(scores.supervised_unknown_view_present.sum()),
        interpretation='Supplementary motion and observation controls; supervised RF uses additional abnormal labels. No change to the primary hypothesis.'))
    dump_json(out.parent/'latest.json',dict(control_dir=str(out),control_id=control_id))
    return out
