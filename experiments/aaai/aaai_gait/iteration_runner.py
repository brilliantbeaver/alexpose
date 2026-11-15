"""Execute the frozen retained-prediction study without touching draft01 runs."""
from __future__ import annotations
import argparse, hashlib, importlib.metadata, json, platform, sys, time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix

from .data import file_sha256
from .study_runner import dump_json
from .iteration_data import load_iteration_data
from .masking import make_mask_bank
from .retained_prediction import (fit_local_baseline, fit_correction_grid,
    inner_candidate_report, select_correction_recipe, assemble_selected_bundle,
    iter_score_batches, known_dependency_fixtures)
from .iteration_scoring import (source_descriptor_tables, NormalDistance,
    compress_target_rows, aggregate_target_profiles, bilateral_profiles,
    BilateralReference, endpoint_frame, ranking_metrics, paired_bootstrap,
    collapse_recording_duplicates, FOCAL)
from .source_controls import fit_source_controls, aggregate_source_features

HERE=Path(__file__).resolve().parent
EXPERIMENT=HERE.parent
ROOT=EXPERIMENT.parents[1]
CONFIG=dict(protocol='retained_opposite_leg_coordinate_iteration02_v1',seed=42,
    primary_endpoint='normal_vs_parkinsons_stroke_cerebral_palsy_myopathic',
    secondary_endpoint='normal_vs_all_other_folders',readouts=list(range(25,33)),
    allowed_joints=[11,12,13,14,15,16,23,24,25,26,27,28,29,30,31,32],
    window_frames=32,sampling_hz=15,patch_frames=4,mask_ratio=.5,
    policies=['U','J','A'],evaluation_banks={'U':17,'E_leg':117},mask_repeats=16,
    train_draws=512,base_method='ridge',base_alpha=1.,residual_alphas=[.1,1.,10.],
    residual_multipliers=[0.,.5,1.],outer_folds=4,inner_folds=3,
    reference_count=5,minimum_distinct_patches=4,
    primary_policy_contrast=['A','U'],primary_method_contrast=['S','M'],
    score_formula='S=.5*(M+R); failed normal adequacy gives R=0 for every source in that fold',
    local_distance='16 descriptors, source aggregation before RobustScaler/LedoitWolf fit-normal distance/16',
    relational_reference='five separate normals; variance=.5*s2+.5*mean(s2)+1e-8; ddof1',
    selection='source-grouped internal normals only; positive benefit over f and f+h in at least2of3folds',
    timing=dict(parent_frames=48,central=[8,40],offsets=[-4,4],parent_mask_seed=217,
                mask_ratio=.5,mask_draws=16,normal_floor=20,test_normal_floor_per_fold=4,
                scored_parents_per_source=8,parent_selection='evenly spaced source-time order'),
    elementary_timing=dict(lags=list(range(-8,9)),minimum_pairs=8,features=8),
    pose_baseline=dict(name='adapted_STG_NF',steps=128,batch_size=32,K=4,L=1,
        upstream_revision='edb5f3220332e160e4d20ce258787d5e2d7e0200',
        imputation='fitting-normal per-joint median XY',graph='16 BlazePose anatomical nodes'),
    binary_unit='one recording cluster; confirmed duplicate score rows averaged before ranking',
    five_class_conflict='exclude sources with multiple folder memberships before fitting',
    bootstrap=dict(repeats=1000,seed=42,strata='outer_fold_and_binary_label',
        weights='fixed_original_normal_times_abnormal_pair_count'),
    exploratory=True,all_normals_previously_explored=True)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def freeze_protocol(data, cohort_dir, mode='full'):
    config=dict(CONFIG)
    if mode=='development':
        config.update(train_draws=64,outer_folds=1,mode='development_incomplete')
    names=['iteration_runner.py','iteration_scoring.py','iteration_data.py','retained_prediction.py',
           'masking.py','iteration_pose_baseline.py','data.py','model.py','features.py','study_data.py']
    identity=dict(config=config,cohort_sha256=data.provenance['cohort_sha256'],
                  registry_sha256=file_sha256(Path(cohort_dir)/'source_registry.csv'),
                  code={n:file_sha256(HERE/n) for n in names},
                  banks={n:make_mask_bank(n,16,s).metadata for n,s in CONFIG['evaluation_banks'].items()})
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    out=EXPERIMENT/'artifacts/iteration02'/digest;out.mkdir(parents=True,exist_ok=True)
    path=out/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['identity']!=identity:raise ValueError('Existing protocol identity changed')
    else:
        dump_json(path,dict(protocol_hash=digest,identity=identity,frozen_at_utc=utcnow(),
                           data_preparation=str(Path(cohort_dir).resolve()),
                           prior_results_seen='draft01 and its sensitivities; no iteration02 test scores used',
                           deviations=['confirmed duplicate review occurred before fitting; retained cohort has no confirmed duplicates']))
    for name,seed in CONFIG['evaluation_banks'].items():
        bank=make_mask_bank(name,16,seed)
        np.savez_compressed(out/f'mask_bank_{name}.npz',hidden=bank.hidden,targets=bank.targets)
    return out,identity


def _features_and_baselines(data,out):
    tables=source_descriptor_tables(data.x,data.metadata)
    for name,table in tables.items():table.reset_index().to_csv(out/f'descriptors_{name}.csv',index=False)
    return tables


def _normal_arrays(data,sources):
    selected=data.metadata.source_id.isin(sources).to_numpy()
    return data.x[selected],data.metadata.loc[selected].reset_index(drop=True)


def fit_outer(data,fold,out,draws):
    """Internal model choices never see reference or test sources."""
    folder=out/f'fold_{fold}';folder.mkdir(parents=True,exist_ok=True)
    registry=data.source_registry; role=f'role_fold_{fold}'; inner=f'inner_fold_{fold}'
    train=registry.loc[registry[role].eq('train')]
    reports=[]; fit_audit=[];start=time.perf_counter()
    for internal in range(3):
        fit_ids=train.loc[train[inner].ne(internal),'source_id'].tolist()
        val_ids=train.loc[train[inner].eq(internal),'source_id'].tolist()
        if set(fit_ids)&set(val_ids):raise ValueError('Internal source leakage')
        x,meta=_normal_arrays(data,fit_ids);v,vm=_normal_arrays(data,val_ids)
        seed=42+fold*100+internal
        base=fit_local_baseline(x,source_ids=meta.source_id,labels=meta.label,
                                draws=draws,alpha=1.,seed=seed)
        grids={p:fit_correction_grid(base,x,source_ids=meta.source_id,labels=meta.label,
                policy=p,draws=draws,seed=seed) for p in CONFIG['policies']}
        report=inner_candidate_report(base,grids,v,source_ids=vm.source_id,labels=vm.label,
                                       bank=make_mask_bank(),inner_fold=internal)
        report['outer_fold']=fold;reports.append(report)
        joblib.dump(dict(base=base,grids=grids,fit_sources=fit_ids,validation_sources=val_ids),folder/f'inner_{internal}_models.joblib')
        fit_audit.append(dict(inner_fold=internal,fit_sources=fit_ids,validation_sources=val_ids,
                              base=base.metadata,corrections={p:grids[p][1.].metadata for p in grids}))
        print(f'fold{fold} internal{internal} completed in {time.perf_counter()-start:.1f}s',flush=True)
    report=pd.concat(reports,ignore_index=True);report.to_csv(folder/'internal_candidates.csv',index=False)
    recipes={p:select_correction_recipe(report,p) for p in CONFIG['policies']}
    dump_json(folder/'selection.json',recipes);dump_json(folder/'internal_fit_audit.json',fit_audit)
    fit_ids=train.source_id.tolist();x,meta=_normal_arrays(data,fit_ids)
    base=fit_local_baseline(x,source_ids=meta.source_id,labels=meta.label,draws=draws,alpha=1.,seed=42+fold*100+99)
    bundles={}
    for policy in CONFIG['policies']:
        grid=fit_correction_grid(base,x,source_ids=meta.source_id,labels=meta.label,policy=policy,draws=draws,seed=42+fold*100+99)
        bundles[policy]=assemble_selected_bundle(grid,recipes[policy])
        joblib.dump(bundles[policy],folder/f'model_{policy}.joblib')
    dump_json(folder/'fitting_record.json',dict(status='complete',outer_fold=fold,started_sources=fit_ids,
        wall_seconds=time.perf_counter()-start,base=base.metadata,recipes=recipes,
        corrections={p:b.metadata for p,b in bundles.items()},model_fits_are_real=True))
    return bundles,recipes,report


def score_prediction_bundle(bundle,x,metadata,bank,folder):
    """Save every requested eligible prediction, and a compact target ledger."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    metadata.to_csv(folder/'window_metadata.csv',index=False)
    raw_parts=[]; examples=[]; total=0
    # Save original mask-specific predictions in bounded files. Compressed
    # target losses are then aggregated independently of repeated-mask counts.
    numeric=[]
    for i,rows in enumerate(iter_score_batches(bundle,x,bank,eligibility='historical_own' if bank.policy=='U' else 'paired_legs',
                                               return_predictions=True,batch_examples=8192)):
        if not len(rows):continue
        if not numeric:numeric=list(rows.columns)
        arrays={c:rows[c].to_numpy() for c in numeric}
        np.savez_compressed(folder/f'predictions_{i:04d}.npz',**arrays)
        raw_parts.append(rows[['window_index','joint','patch_index','mask_id','valid','loss_base','loss_own','loss_both']])
        total+=len(rows)
        if len(examples)<8:
            examples.extend(rows.head(8-len(examples)).to_dict('records'))
    if not raw_parts:raise ValueError(f'No eligible real targets for {bundle.policy}/{bank.policy}')
    raw=pd.concat(raw_parts,ignore_index=True);compact=compress_target_rows(raw)
    compact.to_csv(folder/'target_losses.csv.gz',index=False,compression='gzip')
    profiles,coverage=aggregate_target_profiles(compact,metadata)
    profiles.to_csv(folder/'source_joint_profiles.csv',index=False);coverage.to_csv(folder/'coverage.csv',index=False)
    ex=pd.DataFrame(examples).merge(metadata,on='window_index',how='left',validate='many_to_one')
    ex['policy']=bundle.policy;ex['bank']=bank.policy;ex.to_csv(folder/'example_predictions.csv',index=False)
    dump_json(folder/'prediction_manifest.json',dict(policy=bundle.policy,bank=bank.metadata,
        raw_prediction_rows=total,unique_window_targets=len(compact),eligible_sources=profiles.source_id.nunique() if len(profiles) else 0,
        raw_files=sorted(p.name for p in folder.glob('predictions_*.npz')),
        raw_columns=numeric,producer='iteration_runner.score_prediction_bundle'))
    return profiles,coverage,ex


def _reference_channels(profiles,reference_ids,test_ids,folder):
    values={};references={};sensitivity=[]
    for field in ('benefit','sham_benefit','loss_base','loss_own','loss_both'):
        bilateral=bilateral_profiles(profiles,field)
        reference=BilateralReference.fit(bilateral,reference_ids)
        values[field]=reference.score(bilateral).reindex(test_ids)
        references[field]=reference.audit()
        if field=='benefit':
            for removed in reference_ids:
                ref=BilateralReference.fit(bilateral,[s for s in reference_ids if s!=removed])
                for source,value in ref.score(bilateral).reindex(test_ids).items():
                    sensitivity.append(dict(source_id=source,removed_reference=removed,variance_floor=1e-8,R=value))
            for floor in (1e-10,1e-6):
                ref=BilateralReference.fit(bilateral,reference_ids,floor=floor)
                for source,value in ref.score(bilateral).reindex(test_ids).items():
                    sensitivity.append(dict(source_id=source,removed_reference='',variance_floor=floor,R=value))
    dump_json(folder/'references.json',references);pd.DataFrame(sensitivity).to_csv(folder/'reference_sensitivity.csv',index=False)
    return values


def fit_tabular_comparators(data,fold,tables,folder):
    registry=data.source_registry.set_index('source_id');role=f'role_fold_{fold}'
    fit=sorted(registry.index[registry[role].eq('train')]);test=sorted(registry.index[registry[role].eq('test')])
    results=registry.loc[test,['binary_label','categories','cluster_id','is_focal']].rename(columns={'is_focal':'focal_eligible'}).copy()
    results['outer_fold']=fold;models={};audit={}
    for key in ('M','motion36','timing8','quality16'):
        model=NormalDistance.fit(tables[key],fit,robust=key!='motion36')
        results[key]=model.score(tables[key]).reindex(test);models[key]=model;audit[key]=model.audit()
    # Isolation Forest shares M16 inputs and normal source roles. It is a
    # generic published anomaly control, not a replacement for a pose method.
    dist=models['M'];filled=np.where(np.isfinite(tables['M']),tables['M'],dist.medians)
    values=pd.DataFrame(dist.scaler.transform(filled),index=tables['M'].index)
    iso=IsolationForest(n_estimators=300,max_samples='auto',random_state=42+fold,n_jobs=2).fit(values.loc[fit])
    results['isolation_forest']=-iso.score_samples(values.loc[test]);models['isolation_forest']=iso
    features,names=aggregate_source_features(data.x,data.metadata)
    roles={r:registry.index[registry[role].eq(r)].tolist() for r in ('train','reference','test','auxiliary')}
    roles['val']=roles.pop('reference')
    nuisance,control_models,control_audit=fit_source_controls(features,data.metadata,roles,fold,motion_columns=names)
    nuisance=nuisance.set_index('source_id')
    results['nuisance_distance']=nuisance.nuisance_normal_distance
    results['nuisance_supervised']=nuisance.nuisance_supervised_probability
    audit['nuisance']=control_audit;models['nuisance']=control_models
    # Explicitly supervised five-class, one descriptor row per source.
    eligible=registry.five_class_eligible.astype(bool)
    fitclass=registry.index[eligible&registry[role].isin(['train','auxiliary'])].tolist()
    testclass=registry.index[eligible&registry[role].eq('test')].tolist()
    rfprep=NormalDistance.fit(tables['motion36'],fitclass,robust=False)
    a=np.where(np.isfinite(tables['motion36']),tables['motion36'],rfprep.medians)
    a=pd.DataFrame(a,index=tables['motion36'].index)
    rf=RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=2,max_features='sqrt',
                              class_weight='balanced',random_state=42+fold,n_jobs=2)
    rf.fit(a.loc[fitclass],registry.loc[fitclass,'five_class_label'])
    probability=rf.predict_proba(a.loc[testclass]);pred=np.asarray(rf.classes_)[probability.argmax(1)]
    truth=registry.loc[testclass,'five_class_label'].to_numpy()
    classes=pd.DataFrame(probability,index=testclass,columns=rf.classes_).rename_axis('source_id').reset_index()
    classes['truth']=truth;classes['prediction']=pred;classes['outer_fold']=fold
    classes.to_csv(folder/'five_class_predictions.csv',index=False)
    audit['five_class_rf']=dict(supervision='normal and abnormal training labels',fit_sources=fitclass,test_sources=testclass,
        macro_f1=float(f1_score(truth,pred,labels=list(FOCAL),average='macro',zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(truth,pred)),labels=list(FOCAL),
        confusion_matrix=confusion_matrix(truth,pred,labels=list(FOCAL)).tolist(),
        adaptation='36 exp5-inspired observed descriptors aggregated to source before the RF',
        conflict_rule=CONFIG['five_class_conflict'])
    models['five_class_rf']=rf;models['five_class_imputer']=rfprep
    joblib.dump(models,folder/'tabular_models.joblib');dump_json(folder/'tabular_audit.json',audit)
    return results.reset_index(),audit


def score_outer(data,fold,bundles,recipes,tables,out):
    folder=out/f'fold_{fold}';registry=data.source_registry.set_index('source_id')
    role=f'role_fold_{fold}';references=sorted(registry.index[registry[role].eq('reference')])
    test=sorted(registry.index[registry[role].eq('test')]);sources=set(references+test)
    select=data.metadata.source_id.isin(sources)
    meta=data.metadata.loc[select].copy().sort_values(['binary_label','source_id','sequence_id','start_seconds'])
    indices=meta.index.to_numpy();x=data.x[indices]
    meta=meta.reset_index(drop=True);meta['cohort_window_index']=meta.window_index;meta['window_index']=np.arange(len(meta))
    baseline,audit=fit_tabular_comparators(data,fold,tables,folder)
    baseline.to_csv(folder/'baseline_source_scores.csv',index=False)
    source_rows=[];examples=[];bank_status=[]
    for bank_name,seed in CONFIG['evaluation_banks'].items():
        bank=make_mask_bank(bank_name,16,seed);previous_support=None
        for policy in CONFIG['policies']:
            dest=folder/f'{policy}_{bank_name}'
            profiles,coverage,ex=score_prediction_bundle(bundles[policy],x,meta,bank,dest)
            ex['outer_fold']=fold;examples.append(ex)
            support=profiles[['source_id','joint','distinct_patches']].sort_values(['source_id','joint']).reset_index(drop=True)
            if previous_support is not None:pd.testing.assert_frame_equal(previous_support,support)
            previous_support=support
            eligible=set(profiles.source_id);missing_ref=sorted(set(references)-eligible);missing_test=sorted(set(test)-eligible)
            if bank_name=='U' and (missing_ref or missing_test):
                raise ValueError(f'Historical U target support changed: refs{missing_ref}, test{missing_test}')
            record=dict(outer_fold=fold,policy=policy,bank=bank_name,reference_sources=references,
                        eligible_sources=len(eligible),missing_reference_sources=missing_ref,missing_test_sources=missing_test)
            rows=baseline.copy().set_index('source_id');rows['policy']=policy;rows['bank']=bank_name
            rows['fallback']=not recipes[policy]['relational_supported']
            if missing_ref:
                rows['R']=np.nan;rows['S']=np.nan;record.update(status='unavailable_reference_support')
            else:
                channels=_reference_channels(profiles,references,test,dest)
                rows['R_raw']=channels['benefit'];rows['R']=channels['benefit'] if recipes[policy]['relational_supported'] else 0.
                rows['S']=.5*(rows.M+rows.R)
                rows['sham_R']=channels['sham_benefit'];rows['S_sham']=.5*(rows.M+rows.sham_R)
                rows['base_error_distance']=channels['loss_base'];rows['own_error_distance']=channels['loss_own']
                rows['both_error_distance']=channels['loss_both']
                rows['S_constituent']=.5*rows.M+.25*(rows.own_error_distance+rows.both_error_distance)
                rows['S_base_error']=.5*(rows.M+rows.base_error_distance)
                rows['S_richer_motion']=.5*(rows.motion36+rows.R)
                for field in ('benefit','sham_benefit','loss_base','loss_own','loss_both'):
                    raw=bilateral_profiles(profiles,field)
                    rows[f'left_{field}']=raw.left.reindex(test);rows[f'right_{field}']=raw.right.reindex(test)
                rows.loc[missing_test,['R','S']]=np.nan
                record.update(status='complete' if not missing_test else 'partial_target_support')
            rows.reset_index().to_csv(dest/'source_scores.csv',index=False)
            dump_json(dest/'status.json',record);source_rows.append(rows.reset_index());bank_status.append(record)
            print(f'fold{fold} {policy}/{bank_name}: {record["status"]}, {len(eligible)} reference/test sources',flush=True)
    return pd.concat(source_rows,ignore_index=True),pd.concat(examples,ignore_index=True),bank_status


def summarize_results(out, *, bootstrap_repeats=1000):
    """Reconstruct metrics from persisted source rows, never notebook state."""
    out=Path(out);scores=pd.read_csv(out/'source_scores.csv')
    methods=['M','S','motion36','timing8','isolation_forest','nuisance_distance','nuisance_supervised',
             'S_sham','S_constituent','S_base_error','S_richer_motion']
    if 'stgnf_adapted' in scores:methods.append('stgnf_adapted')
    metrics=[];detail={};comparisons=[]
    for (policy,bank),group in scores.groupby(['policy','bank']):
        for endpoint in ('focal','broad'):
            table=collapse_recording_duplicates(endpoint_frame(group,endpoint),[c for c in methods if c in group])
            if bank=='U' and (not np.isfinite(table[['M','S']]).all().all() or table.outer_fold.nunique()!=4):
                raise ValueError('Primary frozen endpoint is incomplete')
            key=f'{policy}/{bank}/{endpoint}';detail[key]={}
            for method in methods:
                if method not in table:continue
                values=ranking_metrics(table,method);detail[key][method]=values
                if values['status']=='unavailable':continue
                for metric,field in [('auroc','weighted_auroc'),('average_precision','equal_fold_average_precision')]:
                    metrics.append(dict(policy=policy,bank=bank,endpoint=endpoint,method=method,metric=metric,
                                        value=values[field],n_sources=values['n_sources'],status=values['status']))
            if policy=='U' and bank=='U':
                for comparator in ['M','motion36','S_sham','S_constituent','timing8','isolation_forest']:
                    result=paired_bootstrap(table,'S',comparator,repeats=bootstrap_repeats)
                    result.update(endpoint=endpoint,bank=bank,contrast='method',policy=policy);comparisons.append(result)
    for bank in ('U','E_leg'):
        for endpoint in ('focal','broad'):
            frames=[]
            for policy in ('U','J','A'):
                t=endpoint_frame(scores.loc[scores.policy.eq(policy)&scores.bank.eq(bank)],endpoint)
                t=collapse_recording_duplicates(t,['S'])
                frames.append(t[['source_id','cluster_id','outer_fold','binary_label','focal_eligible','S']].rename(columns={'S':policy}))
            merged=frames[0]
            for t in frames[1:]:merged=merged.merge(t,on=['source_id','cluster_id','outer_fold','binary_label','focal_eligible'],validate='one_to_one')
            for a,b in [('A','U'),('A','J'),('J','U')]:
                result=paired_bootstrap(merged,a,b,repeats=bootstrap_repeats)
                result.update(endpoint=endpoint,bank=bank,contrast='mask_policy');comparisons.append(result)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    dump_json(out/'rawmetrics.json',detail);dump_json(out/'paired_intervals.json',comparisons)
    return detail,comparisons


def run_study(cohort_dir, *, mode='full', only_fold=None, resume=False):
    started=time.perf_counter();data=load_iteration_data(cohort_dir)
    out,identity=freeze_protocol(data,cohort_dir,mode);draws=identity['config']['train_draws']
    producer=f'uv run --project experiments/aaai python -m aaai_gait.iteration_runner --cohort {Path(cohort_dir)} --mode {mode}'
    manifest=dict(protocol_hash=out.name,mode=mode,data_kind='real_gavd',status='running',producer_command=producer,
                  started_utc=utcnow(),python=sys.executable,python_version=platform.python_version(),
                  packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','torch','scikit-learn','python-dotenv']},
                  cohort_directory=str(Path(cohort_dir).resolve()),cohort_sha256=data.provenance['cohort_sha256'])
    dump_json(out/'run_manifest.json',manifest)
    data.source_registry.to_csv(out/'source_registry.csv',index=False);data.metadata.to_csv(out/'window_metadata.csv',index=False)
    fixtures=known_dependency_fixtures();dump_json(out/'fixtures.json',fixtures)
    if not fixtures['passed']:raise AssertionError('Known-dependency software gate failed')
    tables=_features_and_baselines(data,out)
    folds=list(range(4)) if mode=='full' else [0]
    if only_fold is not None:folds=[only_fold]
    all_scores=[];all_examples=[];all_status=[];adequacy=[]
    for fold in folds:
        folder=out/f'fold_{fold}';fold_start=time.perf_counter()
        if resume and (folder/'fitting_record.json').exists():
            bundles={p:joblib.load(folder/f'model_{p}.joblib') for p in CONFIG['policies']}
            recipes=json.loads((folder/'selection.json').read_text());report=pd.read_csv(folder/'internal_candidates.csv')
        else:bundles,recipes,report=fit_outer(data,fold,out,draws)
        for policy,recipe in recipes.items():
            for row in recipe['internal_losses']:
                adequacy.append(dict(outer_fold=fold,policy=policy,relational_supported=recipe['relational_supported'],
                                     positive_internal_folds=recipe['positive_internal_folds'],**row))
        if mode=='development':
            manifest.update(status='development_complete_not_efficacy',wall_seconds=time.perf_counter()-started)
            dump_json(out/'run_manifest.json',manifest);pd.DataFrame(adequacy).to_csv(out/'adequacy.csv',index=False)
            print(out,flush=True);return out
        if resume and (folder/'fold_source_scores.csv').exists():
            scores=pd.read_csv(folder/'fold_source_scores.csv');examples=pd.read_csv(folder/'fold_examples.csv')
            status=json.loads((folder/'bank_status.json').read_text())
        else:
            scores,examples,status=score_outer(data,fold,bundles,recipes,tables,out)
            scores.to_csv(folder/'fold_source_scores.csv',index=False);examples.to_csv(folder/'fold_examples.csv',index=False)
            dump_json(folder/'bank_status.json',status)
        all_scores.append(scores);all_examples.append(examples);all_status.extend(status)
        print(f'fold{fold} total wall {time.perf_counter()-fold_start:.1f}s',flush=True)
        if time.perf_counter()-started>6*3600:raise TimeoutError('The frozen first-core six-hour ceiling was reached')
    pd.concat(all_scores,ignore_index=True).to_csv(out/'source_scores.csv',index=False)
    pd.concat(all_examples,ignore_index=True).to_csv(out/'target_predictions.csv',index=False)
    pd.DataFrame(adequacy).to_csv(out/'adequacy.csv',index=False);dump_json(out/'bank_status.json',all_status)
    if len(folds)==4:summarize_results(out,bootstrap_repeats=CONFIG['bootstrap']['repeats'])
    manifest.update(status='core_complete' if len(folds)==4 else 'partial_fold_complete',
                    wall_seconds=time.perf_counter()-started,completed_folds=folds,completed_utc=utcnow())
    dump_json(out/'run_manifest.json',manifest);print(out,flush=True);return out


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cohort',type=Path,required=True)
    parser.add_argument('--mode',choices=['full','development'],default='full')
    parser.add_argument('--only-fold',type=int);parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();run_study(args.cohort,mode=args.mode,only_fold=args.only_fold,resume=args.resume)

if __name__=='__main__':main()
