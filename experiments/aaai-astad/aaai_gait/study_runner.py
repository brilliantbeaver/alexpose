"""Source-held-out predictive profile experiment, separate from old capstones."""
from dataclasses import asdict,replace
from pathlib import Path
import hashlib,json,time,os,uuid
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
import torch
from .settings import ROOT,EXPERIMENT,Settings
from .study_data import build_study_manifest,load_study_windows,make_outer_folds,fold_metadata,source_table
from .context_prediction import ContextConfig,READOUT_JOINTS,fit_context_models,score_context_models,save_context_bundle,load_context_bundle
from .predictive_profile import (aggregate_profiles,compress_loss_rows,ProfileReference,
    score_profiles,leave_one_reference_out,freeze_profile_eligibility)
from .linear_context import fit_ridge_context,score_ridge_context
from .data import file_sha256
from .model import JOINT_INDICES
from .study_baselines import condition_capstone


def dump_json(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,Path):return str(v)
        return v
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    temporary.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temporary.replace(path)


def _initialize_protocol(path,value):
    """Atomic create-once publication lets independent folds share one run."""
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.new')
    dump_json(temporary,value)
    try:
        try:os.link(temporary,path)
        except FileExistsError:pass
    finally:temporary.unlink(missing_ok=True)
    existing=json.loads(path.read_text())
    for key in ('run_id','corpus','profile_eligibility_sha256','fold_registry_sha256'):
        if existing.get(key)!=value.get(key):raise ValueError('Existing study protocol identity differs')


def _write_shared_csv(path,table):
    path=Path(path);temporary=path.with_name(path.name+f'.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    table.to_csv(temporary,index=False);temporary.replace(path)


def prepare_study(*,demo=False,seed=42):
    if demo:
        from .demo import synthetic_corpus
        x,meta=synthetic_corpus(seed,sources_per_class=12)
        meta=meta.drop(columns=['split']);meta['binary_label']=meta.label.ne('normal').astype(int)
        meta['start_seconds']=0.;meta['end_seconds']=31/15;meta['window_index']=np.arange(len(meta))
        meta['window_id']=meta.sequence_id+':0';meta['camera_views']='synthetic';meta['prior_use']='synthetic'
        meta.attrs['corpus_sha256']=hashlib.sha256(x.tobytes()+meta.to_csv(index=False).encode()).hexdigest()
        return x,meta,pd.DataFrame(),pd.DataFrame()
    settings=Settings(demo=False)
    manifest=build_study_manifest(ROOT,settings.pose_dir)
    x,meta,ledger=load_study_windows(manifest,settings.pose_dir,frames=32,stride=16,target_fps=15.,
                                    expected_frame_number_base=settings.frame_number_base)
    return x,meta,manifest,ledger


def _identity(meta,config,seed,masks,ridge_draws,untrained):
    names=('study_runner.py','study_data.py','context_prediction.py','predictive_profile.py',
           'linear_context.py','study_baselines.py','model.py','data.py','training.py','features.py','demo.py')
    value=dict(protocol='neuromechanics_completion_v1',corpus=meta.attrs['corpus_sha256'],
        pre_profile_corpus_sha256=meta.attrs.get('pre_profile_corpus_sha256'),
        profile_eligibility_sha256=meta.attrs.get('profile_eligibility_sha256'),
        retained_metadata_sha256=meta.attrs.get('retained_metadata_sha256'),
        config=asdict(config),seed=seed,mask_seed=17,mask_repeats=masks,ridge_draws=ridge_draws,
        untrained_control=untrained,code={n:file_sha256(Path(__file__).parent/n) for n in names},
        software={name:importlib.metadata.version(name) for name in ('numpy','pandas','torch','scikit-learn')})
    value['run_id']=hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:16]
    return value


def _freeze_profile_cohort(x,meta,config,mask_repeats):
    """Apply label-independent source support QC before any source receives a fold."""
    original_attrs=dict(meta.attrs)
    keep,ledger=freeze_profile_eligibility(x,meta,config=config,mask_seed=17,mask_repeats=mask_repeats)
    if not keep.any():raise ValueError('No sources have the required fixed-mask readout support')
    retained=meta.loc[keep].reset_index(drop=True).copy()
    retained['pre_profile_window_index']=retained['window_index'].to_numpy() if 'window_index' in retained else np.flatnonzero(keep)
    retained['window_index']=np.arange(len(retained),dtype=np.int64)
    values=x[keep]
    identity=dict(pre_profile_corpus_sha256=original_attrs['corpus_sha256'],
        profile_eligibility_sha256=ledger.attrs['eligibility_sha256'],
        retained_metadata_sha256=hashlib.sha256(retained.to_csv(index=False).encode()).hexdigest(),
        retained_array_sha256=hashlib.sha256(np.asarray(values).tobytes()).hexdigest())
    retained.attrs={**original_attrs,**identity,
        'corpus_sha256':hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()}
    return values,retained,ledger


def _validate_ridge_support(ridge):
    """Do not shrink the primary neural cohort to accommodate an underfit control."""
    missing=[]
    for joint in READOUT_JOINTS:
        j=JOINT_INDICES.index(joint)
        for t in range(8):
            count=ridge.counts.get((j,t),0)
            if count<8 or any((j,t,b) not in ridge.estimators for b in ('own','patch','full')):
                missing.append(dict(joint=joint,patch_index=t,eligible_training_draws=count))
    if missing:
        raise ValueError('Ridge control lacks eight eligible training examples for every readout target. '
                         'Increase the prespecified ridge_draws budget for the entire study before scoring. '
                         f'Unsupported targets: {missing}')


def _score_batches(bundle,x,meta,*,masks,ridge=None,chunk_size=64):
    if len(x)!=len(meta) or len(x)==0:raise ValueError('Scoring arrays and positional metadata must match and be nonempty')
    if not isinstance(chunk_size,int) or chunk_size<1:raise ValueError('Positive integer chunk_size required')
    if ridge is not None:_validate_ridge_support(ridge)
    neural=[];linear=[]
    for start in range(0,len(x),chunk_size):
        part=score_context_models(bundle,x[start:start+chunk_size],mask_repeats=masks,batch_size=16)
        if ridge is not None:
            r=score_ridge_context(ridge,x[start:start+chunk_size],part)
            coordinate=part.loc[part.target_type.eq('coordinate')].reset_index(drop=True)
            keys=['window_index','joint','patch_index','mask_id','valid']
            if not r[keys].reset_index(drop=True).equals(coordinate[keys]):
                raise ValueError('Ridge and neural targets or eligibility differ before aggregation')
            r['window_index']+=start;linear.append(compress_loss_rows(r))
        part['window_index']+=start;neural.append(compress_loss_rows(part))
    return pd.concat(neural,ignore_index=True),pd.concat(linear,ignore_index=True) if linear else None


def _write_score_cache(path,table,identity):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    table.to_csv(temporary,index=False,compression='gzip');temporary.replace(path)
    dump_json(path.with_name(path.name+'.json'),dict(**identity,sha256=file_sha256(path),rows=len(table)))


def _read_score_cache(path,identity):
    path=Path(path);sidecar=path.with_name(path.name+'.json')
    if not path.exists() or not sidecar.exists():return None
    stored=json.loads(sidecar.read_text())
    if any(stored.get(k)!=v for k,v in identity.items()) or stored.get('sha256')!=file_sha256(path):
        raise ValueError(f'Score cache provenance or contents changed: {path}')
    table=pd.read_csv(path)
    if len(table)!=stored.get('rows'):raise ValueError('Score cache row count differs from its provenance')
    return table


def _assert_compact_support(first,second):
    """Predictor training and latent units cannot change observation eligibility."""
    keys=['window_index','joint','patch_index','target_type']
    fields=keys+['target_count','requested_count']
    left,right=compress_loss_rows(first),compress_loss_rows(second)
    for value in (left,right):value['target_type']=value.target_type.astype(str)
    left=left[fields].sort_values(keys).reset_index(drop=True)
    right=right[fields].sort_values(keys).reset_index(drop=True)
    try:pd.testing.assert_frame_equal(left,right,check_dtype=False)
    except AssertionError as error:
        raise ValueError('Methods have different compact target identities or support weights') from error


def _assert_budget_selection(selection,reference):
    keys=['source_id','joint','sequence_id','patch_start_ns','patch_end_ns']
    left,right=selection[keys].copy(),reference[keys].copy()
    for value in (left,right):
        for field in ('source_id','sequence_id'):value[field]=value[field].astype(str)
    try:pd.testing.assert_frame_equal(left.sort_values(keys).reset_index(drop=True),
                                      right.sort_values(keys).reset_index(drop=True),check_dtype=False)
    except AssertionError as error:
        raise ValueError('Fixed four-patch observation selections differ across methods') from error


def _profiles_and_scores(compact,meta,folder,*,method,train_ids,test_ids):
    all_scores=[];sensitivity=[];reference_dict={}
    for budget in (None,4):
        profiles,support,selection=aggregate_profiles(compact,meta,fixed_budget=budget)
        if not profiles.scorable.all():
            raise ValueError('A source failed profile support after eligibility was frozen before folds')
        suffix='all' if budget is None else 'budget4'
        profiles.to_csv(folder/f'{method}_{suffix}_profiles.csv',index=False)
        support.to_csv(folder/f'{method}_{suffix}_support.csv',index=False)
        if budget:
            if method!='trained':
                _assert_budget_selection(selection,pd.read_csv(folder/'trained_budget4_selection.csv',dtype={'source_id':str,'sequence_id':str}))
            selection.to_csv(folder/f'{method}_{suffix}_selection.csv',index=False)
        for kind in sorted(profiles.target_type.unique()):
            subset=profiles[profiles.target_type.eq(kind)].copy()
            refdata=subset[subset.split.eq('val')];testdata=subset[subset.split.eq('test')]
            refs={field:ProfileReference.fit(refdata,field,kind,training_source_ids=train_ids,
                    test_source_ids=test_ids) for field in ('benefit','own','full','patch')}
            scored=score_profiles(testdata,refs);scored['method']=method;scored['observation_budget']=suffix
            all_scores.append(scored)
            reference_dict[f'{method}_{suffix}_{kind}']={k:v.to_dict() for k,v in refs.items()}
            if budget is None:
                loo=leave_one_reference_out(refdata,testdata,refs['benefit'],target_type=kind)
                loo['method']=method;loo['outer_fold']=int(meta.outer_fold.iloc[0]);sensitivity.append(loo)
    dump_json(folder/f'{method}_references.json',reference_dict)
    if sensitivity:pd.concat(sensitivity).to_csv(folder/f'{method}_reference_sensitivity.csv',index=False)
    result=pd.concat(all_scores,ignore_index=True);result.to_csv(folder/f'{method}_scores.csv',index=False)
    return result


def run_study(*,demo=False,seed=42,config=None,mask_repeats=16,ridge_draws=256,
              include_untrained=True,folds=(0,1,2,3),output_root=None,finalize=True):
    """Freeze a corpus, fit independent folds, and persist every scientific role.

    No abnormal source enters model or profile-reference fitting. The existing
    supervised five-class RF capstone is separate and has different supervision.
    Separate processes may fit disjoint folds with finalize=False. Summarize
    once after every process exits successfully, with pose inputs held fixed.
    """
    cfg=config or ContextConfig();torch.set_num_threads(2)
    if isinstance(mask_repeats,bool) or not isinstance(mask_repeats,int) or mask_repeats!=16:
        raise ValueError('The study fixes exactly 16 scoring masks before eligibility and folds')
    if isinstance(ridge_draws,bool) or not isinstance(ridge_draws,int) or ridge_draws<8:
        raise ValueError('ridge_draws must be an integer of at least eight')
    folds=tuple(folds)
    if not folds or len(set(folds))!=len(folds) or any(f not in range(4) for f in folds):
        raise ValueError('Provide unique folds from 0, 1, 2, 3')
    x,meta,manifest,ledger=prepare_study(demo=demo,seed=seed)
    pre_profile_meta=meta.copy()
    x,meta,profile_ledger=_freeze_profile_cohort(x,meta,cfg,mask_repeats)
    identity=_identity(meta,cfg,seed,mask_repeats,ridge_draws,include_untrained)
    base=Path(output_root or EXPERIMENT/'artifacts/context_study'/('synthetic' if demo else 'gavd'))
    out=base/identity['run_id'];out.mkdir(parents=True,exist_ok=True)
    registry=make_outer_folds(meta,seed=seed)
    _initialize_protocol(out/'protocol.json',dict(**identity,synthetic=demo,started_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        scoring='source-level; within-fold AUROC; fixed 12-joint support; normal-reference distance',
        status='planned',fold_registry_sha256=registry.attrs['registry_sha256']))
    _write_shared_csv(out/'source_folds.csv',registry);_write_shared_csv(out/'window_index.csv',meta)
    _write_shared_csv(out/'window_index_before_profile_qc.csv',pre_profile_meta)
    _write_shared_csv(out/'profile_eligibility.csv',profile_ledger)
    dump_json(out/'profile_eligibility.json',dict(**profile_ledger.attrs,
        pre_profile_sources=pre_profile_meta.source_id.nunique(),retained_sources=meta.source_id.nunique(),
        pre_profile_windows=len(pre_profile_meta),retained_windows=len(meta),
        excluded_source_ids=profile_ledger.loc[~profile_ledger.profile_eligible,'source_id'].tolist(),
        retained_metadata_sha256=meta.attrs['retained_metadata_sha256'],corpus_sha256=meta.attrs['corpus_sha256']))
    if len(manifest):
        _write_shared_csv(out/'fresh_manifest.csv',manifest);_write_shared_csv(out/'eligibility.csv',ledger)
    print('RUN_DIRECTORY',out,flush=True)
    for fold in folds:
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True)
        fm=fold_metadata(meta,registry,fold);train=fm.split.eq('train').to_numpy();val=fm.split.eq('val').to_numpy()
        evaluate=fm.split.isin(['val','test']).to_numpy();em=fm.loc[evaluate].reset_index(drop=True)
        fm.to_csv(folder/'window_roles.csv',index=False)
        roles={s:sorted(fm.loc[fm.split.eq(s),'source_id'].unique()) for s in ('train','val','test','auxiliary')}
        dump_json(folder/'source_roles.json',roles)
        for method in ('trained','untrained') if include_untrained else ('trained',):
            checkpoint=folder/f'{method}.pt'
            provenance=dict(run_id=identity['run_id'],fold=fold,method=method,roles=roles)
            print('FOLD',fold,method,'fit',flush=True)
            if checkpoint.exists():bundle=load_context_bundle(checkpoint,expected_provenance=provenance)
            else:
                modelcfg=cfg if method=='trained' else replace(cfg,pretrain_updates=0)
                bundle,history=fit_context_models(x[train],x[val],config=modelcfg,seed=seed+fold,
                    train_source_ids=fm.loc[train,'source_id'].to_numpy(),validation_source_ids=fm.loc[val,'source_id'].to_numpy(),
                    train_labels=fm.loc[train,'label'].to_numpy(),validation_labels=fm.loc[val,'label'].to_numpy(),provenance=provenance)
                save_context_bundle(bundle,checkpoint,provenance=provenance)
            dump_json(folder/f'{method}_history.json',bundle.history)
            compact_path=folder/f'{method}_losses.csv.gz'
            cache_identity=dict(**provenance,checkpoint_sha256=file_sha256(checkpoint),mask_seed=17,
                mask_repeats=mask_repeats,evaluation_metadata_sha256=hashlib.sha256(em.to_csv(index=False).encode()).hexdigest())
            compact=_read_score_cache(compact_path,cache_identity)
            linear_path=folder/'ridge_losses.csv.gz'
            linear=_read_score_cache(linear_path,cache_identity) if method=='trained' else None
            if compact is None or (method=='trained' and linear is None):
                ridge=None
                if method=='trained':
                    ridge=fit_ridge_context(x[train],source_ids=fm.loc[train,'source_id'].to_numpy(),
                        labels=fm.loc[train,'label'].to_numpy(),config=cfg,draws=ridge_draws,seed=seed+fold)
                    _validate_ridge_support(ridge)
                    ridge_path=folder/'ridge.joblib';temporary=folder/'ridge.joblib.tmp'
                    joblib.dump(dict(model=ridge,provenance=cache_identity),temporary);temporary.replace(ridge_path)
                compact,linear=_score_batches(bundle,x[evaluate],em,masks=mask_repeats,ridge=ridge)
                _write_score_cache(compact_path,compact,cache_identity)
                if linear is not None:_write_score_cache(linear_path,linear,cache_identity)
            if method=='trained':
                trained_compact=compact
                _assert_compact_support(compact.loc[compact.target_type.eq('coordinate')],linear)
            else:
                _assert_compact_support(trained_compact,compact)
            _profiles_and_scores(compact,em,folder,method=method,train_ids=roles['train'],test_ids=roles['test'])
            if method=='trained':
                _profiles_and_scores(linear,em,folder,method='ridge',train_ids=roles['train'],test_ids=roles['test'])
                condition_capstone(bundle,x,fm,folder,seed+fold)
            print('FOLD',fold,method,'complete',flush=True)
    if finalize:summarize_study(out,meta,identity,synthetic=demo)
    return out


def summarize_study(out,meta=None,identity=None,*,synthetic=False):
    from .predictive_profile import fold_ranking_metrics,paired_fold_bootstrap
    out=Path(out);tables=[]
    for folder in sorted(out.glob('fold_*')):
        for method in ('trained','untrained','ridge'):
            path=folder/f'{method}_scores.csv'
            if path.exists():tables.append(pd.read_csv(path))
    scores=pd.concat(tables,ignore_index=True);scores.to_csv(out/'source_scores.csv',index=False)
    # Complete-case source intersection is explicit, not method-specific attrition.
    pieces=[]
    for (method,kind,budget),group in scores.groupby(['method','target_type','observation_budget']):
        keep=group[['source_id','outer_fold','binary_label','categories']].copy()
        for column in ('profile_score','raw_full','raw_own','anomaly_contrast','own_distance','full_distance','patch_distance'):
            keep[f'{method}_{kind}_{budget}_{column}']=group[column].to_numpy()
        pieces.append(keep)
    wide=pieces[0]
    for piece in pieces[1:]:wide=wide.merge(piece,on=['source_id','outer_fold','binary_label','categories'],how='outer',validate='one_to_one')
    columns=[c for c in wide if c not in ('source_id','outer_fold','binary_label','categories')]
    complete=wide[columns].notna().all(axis=1);wide['all_methods_scorable']=complete
    wide.to_csv(out/'paired_source_scores.csv',index=False)
    metrics=fold_ranking_metrics(wide.loc[complete],columns,expected_folds=tuple(sorted(wide.outer_fold.unique())))
    dump_json(out/'ranking_metrics.json',metrics)
    pd.DataFrame([dict(method=name,auroc=value['primary_auroc'],average_precision=value['normal_weighted_ap'])
                  for name,value in metrics['methods'].items()]).to_csv(out/'ranking_metrics.csv',index=False)
    dump_json(out/'evaluation_coverage.json',dict(sources=len(wide),common_sources=int(complete.sum()),excluded=wide.loc[~complete,'source_id'].tolist()))
    primary='trained_latent_all_profile_score';raw='trained_latent_all_raw_full'
    bootstrap=paired_fold_bootstrap(wide.loc[complete],primary,raw,repeats=1000,seed=42,
                                   expected_folds=tuple(sorted(wide.outer_fold.unique())))
    dump_json(out/'primary_paired_bootstrap.json',bootstrap)
    if meta is None:meta=pd.read_csv(out/'window_index.csv')
    completed=sorted(scores.outer_fold.unique().astype(int).tolist())
    summary=dict(status='completed' if completed==[0,1,2,3] else 'partial',run_name=out.name,synthetic=synthetic,
        cohort=dict(sources=meta.source_id.nunique(),normal_sources=meta.loc[meta.binary_label.eq(0),'source_id'].nunique(),
            sequences=meta.sequence_id.nunique(),windows=len(meta)),completed_folds=completed,
        artifact_path=str(out),metrics=[dict(method=name,value=metrics['methods'][name]['primary_auroc'])
          for name in ('trained_latent_all_profile_score','trained_latent_all_raw_full',
                       'trained_coordinate_all_profile_score','ridge_coordinate_all_profile_score',
                       'untrained_latent_all_profile_score') if name in metrics['methods']],
        notes=['Masked completion, not future prediction or diagnosis.',
            'Confidence intervals condition on fitted models, references, folds and observed source IDs.',
            'Supervised five-class Random Forest comparison remains a separate task.'])
    dump_json(out/'summary.json',summary)
    dump_json(out.parent/'latest.json',dict(run_dir=str(out),synthetic=synthetic))
    return summary
