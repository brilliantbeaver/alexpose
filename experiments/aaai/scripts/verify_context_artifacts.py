"""Independently verify a completed four-fold study before using its artifacts.

Run with an artifact directory and --expect-real for paper evidence. --self-test
uses copies of a completed four-fold demo to test missing and tampered files.
The verifier re-reads the current corpus and code; changed inputs fail checks.
Joblib artifacts must originate from this trusted local experiment. Checksums
provide consistency, not signatures against coordinated malicious replacement.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict,replace
import hashlib,importlib.metadata,json,shutil,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import torch
from aaai_gait.context_prediction import ContextConfig,READOUT_JOINTS,load_context_bundle,make_mask_bank
from aaai_gait.data import file_sha256
from aaai_gait.model import JOINT_INDICES
from aaai_gait.predictive_profile import aggregate_profiles,ProfileReference,compress_loss_rows
from aaai_gait.study_baselines import FOCAL
from aaai_gait.study_data import make_outer_folds,fold_metadata
from aaai_gait.study_runner import (prepare_study,_freeze_profile_cohort,_identity,
    _assert_compact_support,_assert_budget_selection,_validate_ridge_support,dump_json)

SCORES=('profile_score','raw_full','raw_own','anomaly_contrast','own_distance','full_distance','patch_distance')
METHODS={'trained':('coordinate','latent'),'untrained':('coordinate','latent'),'ridge':('coordinate',)}
FIELDS=('benefit','own','full','patch')
IDS=('source_id','outer_fold','binary_label','categories')

def require(condition,message):
    if not bool(condition):raise ValueError(message)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def read_json(path):return json.loads(Path(path).read_text())

def read_csv(path):return pd.read_csv(path,float_precision='round_trip',dtype={'source_id':str,'sequence_id':str})

def same_table(a,b,name,keys=None,atol=1e-10,rtol=1e-10):
    require(set(a.columns)==set(b.columns),name+' has different columns')
    a,b=a.copy(),b[a.columns].copy()
    for table in (a,b):
        for col in table:
            if isinstance(table[col].dtype,pd.CategoricalDtype):table[col]=table[col].astype(object)
    if keys:a,b=[t.sort_values(list(keys)).reset_index(drop=True) for t in (a,b)]
    try:pd.testing.assert_frame_equal(a,b,check_dtype=False,check_categorical=False,atol=atol,rtol=rtol)
    except AssertionError:raise ValueError(name+' values or identities differ') from None

def file_matches_table(path,table):
    require(file_sha256(path)==hashlib.sha256(table.to_csv(index=False).encode()).hexdigest(),
            Path(path).name+' differs from freshly reproduced metadata')

def expected_support(x,cfg):
    bank=make_mask_bank(cfg,17,16).numpy().reshape(16,8,16)
    valid=np.isfinite(x).reshape(len(x),8,4,16,2).all(axis=(2,4));counts=np.zeros_like(valid,dtype=np.int64)
    for mask in bank:counts+=mask & valid & ((valid & ~mask).sum(axis=1)[:,None,:]>=2)
    slots=[JOINT_INDICES.index(j) for j in READOUT_JOINTS];w,t,j=np.indices((len(x),8,len(slots)))
    requested=np.broadcast_to(bank[:,:,slots].sum(0),(len(x),8,len(slots))).reshape(-1)
    result=pd.DataFrame(dict(window_index=w.reshape(-1),joint=np.asarray(READOUT_JOINTS)[j.reshape(-1)],
        patch_index=t.reshape(-1),target_count=counts[:,:,slots].reshape(-1),requested_count=requested))
    return result[result.requested_count.gt(0)].reset_index(drop=True)

def required_files():
    common=['protocol.json','summary.json','source_folds.csv','window_index.csv','window_index_before_profile_qc.csv',
        'profile_eligibility.csv','profile_eligibility.json','source_scores.csv','paired_source_scores.csv',
        'evaluation_coverage.json','ranking_metrics.json','ranking_metrics.csv','primary_paired_bootstrap.json']
    fold=['source_roles.json','window_roles.csv','trained.pt','untrained.pt','trained_history.json',
          'untrained_history.json','ridge.joblib','condition_classifiers.joblib','condition_metrics.json','condition_predictions.csv']
    for method in METHODS:
        fold += [f'{method}_losses.csv.gz',f'{method}_losses.csv.gz.json',f'{method}_references.json',
                 f'{method}_scores.csv',f'{method}_reference_sensitivity.csv',f'{method}_budget4_selection.csv']
        for budget in ('all','budget4'):fold += [f'{method}_{budget}_profiles.csv',f'{method}_{budget}_support.csv']
    return common+[f'fold_{f}/{name}' for f in range(4) for name in fold]

def verify_run(run_dir,*,expect_real=False,write=True):
    """Save checks, counts and hashes; pass only if every check succeeds."""
    run=Path(run_dir).resolve();torch.set_num_threads(2);state={}
    report=dict(format_version=1,run_dir=str(run),passed=False,expect_real=bool(expect_real),checks=[],counts={},artifact_sha256={})
    def check(name,operation):
        try:
            detail=operation();report['checks'].append(dict(name=name,passed=True,detail=detail));return True
        except Exception as error:
            report['checks'].append(dict(name=name,passed=False,error=f'{type(error).__name__}: {error}'));return False
    def inventory():
        missing=[n for n in required_files() if not (run/n).is_file()]
        require(not missing,'Missing artifacts: '+', '.join(missing))
        report['artifact_sha256'].update({n:file_sha256(run/n) for n in required_files()})
        return dict(required_files=len(required_files()),folds=4)
    inventory_ok=check('required_artifact_inventory',inventory)
    def protocol():
        p=read_json(run/'protocol.json');state['protocol']=p
        require(p['protocol']=='neuromechanics_completion_v1','Unsupported study protocol')
        require(p['run_id']==run.name,'Directory differs from declared run identity')
        require(p['mask_seed']==17 and p['mask_repeats']==16,'Exactly sixteen seed-17 masks required')
        require(p['untrained_control'] is True,'Complete control set requires untrained models')
        require(not expect_real or p['synthetic'] is False,'--expect-real rejects synthetic demonstrations')
        state['config']=ContextConfig(**p['config']);base=Path(__file__).resolve().parents[1]/'aaai_gait'
        for name,sha in p['code'].items():require(file_sha256(base/name)==sha,'Code changed: '+name)
        for name,version in p['software'].items():require(importlib.metadata.version(name)==version,'Software changed: '+name)
        return dict(run_id=p['run_id'],synthetic=p['synthetic'],code_files=len(p['code']),mask_repeats=16)
    protocol_ok=check('protocol_code_software_and_real_data_gate',protocol)
    def corpus():
        p,cfg=state['protocol'],state['config'];x,pre,manifest,ledger=prepare_study(demo=p['synthetic'],seed=p['seed'])
        x,meta,qc=_freeze_profile_cohort(x,pre,cfg,16)
        for k,v in _identity(meta,cfg,p['seed'],16,p['ridge_draws'],True).items():require(p.get(k)==v,'Fresh identity differs: '+k)
        for name,table in [('window_index_before_profile_qc.csv',pre),('window_index.csv',meta),('profile_eligibility.csv',qc)]:file_matches_table(run/name,table)
        q=read_json(run/'profile_eligibility.json')
        for k in ('eligibility_sha256','decision_inputs'):require(q[k]==qc.attrs[k],'QC provenance differs: '+k)
        counts=dict(pre_profile_sources=pre.source_id.nunique(),retained_sources=meta.source_id.nunique(),pre_profile_windows=len(pre),retained_windows=len(meta))
        for k,v in counts.items():require(q[k]==v,'QC counts differ: '+k)
        require(q['excluded_source_ids']==qc.loc[~qc.profile_eligible,'source_id'].tolist(),'QC exclusion list differs')
        require(q['corpus_sha256']==p['corpus'] and q['retained_metadata_sha256']==p['retained_metadata_sha256'],'QC corpus linkage differs')
        bank=make_mask_bank(cfg,17,16).numpy()
        require(qc.mask_bank_sha256.eq(hashlib.sha256(bank.tobytes()).hexdigest()).all(),'QC mask bank differs')
        registry=make_outer_folds(meta,seed=p['seed']);file_matches_table(run/'source_folds.csv',registry)
        require(registry.attrs['registry_sha256']==p['fold_registry_sha256'],'Registry hash differs')
        require(registry.source_id.is_unique,'Registry repeats source identities')
        if not p['synthetic']:
            for name,table in [('fresh_manifest.csv',manifest),('eligibility.csv',ledger)]:
                file_matches_table(run/name,table);report['artifact_sha256'][name]=file_sha256(run/name)
        state.update(x=x,meta=meta,registry=registry)
        report['counts'].update(sources=int(meta.source_id.nunique()),normal_sources=int(meta.loc[meta.binary_label.eq(0),'source_id'].nunique()),
                               sequences=int(meta.sequence_id.nunique()),windows=len(meta),**{k:int(v) for k,v in counts.items()})
        return dict(corpus_sha256=p['corpus'],eligibility_sha256=p['profile_eligibility_sha256'],**counts)
    corpus_ok=protocol_ok and check('fresh_corpus_qc_masks_and_fold_hashes',corpus)
    if inventory_ok and corpus_ok:
        for f in range(4):check(f'fold_{f}_models_roles_losses_references_and_condition_artifacts',lambda f=f:verify_fold(run,f,state))
        check('paired_out_of_fold_cohort_and_summary_counts',lambda:verify_cohort(run,state))
    else:report['checks'].append(dict(name='dependent_fold_and_cohort_checks',passed=False,error='Inventory or corpus identity failed'))
    report['passed']=all(c['passed'] for c in report['checks']);report['verified_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    report['verifier_sha256']=file_sha256(Path(__file__))
    report['limitations']=['Verifies artifact consistency and current inputs, not clinical validity, generalization or publication readiness.',
        'No performance sign or preferred method is selected. Unknown repeated-person dependence remains unresolved.']
    if write:dump_json(run/'verification.json',report)
    return report

def verify_fold(run,fold,state):
    p,cfg,x,meta,registry=[state[k] for k in ('protocol','config','x','meta','registry')]
    folder=run/f'fold_{fold}';fm=fold_metadata(meta,registry,fold);file_matches_table(folder/'window_roles.csv',fm)
    roles={r:sorted(fm.loc[fm.split.eq(r),'source_id'].unique()) for r in ('train','val','test','auxiliary')}
    require(read_json(folder/'source_roles.json')==roles,'Saved source roles differ from registry')
    for a in roles:
        require(len(roles[a])==len(set(roles[a])),'Duplicate source in a role')
        for b in roles:
            if a!=b:require(not set(roles[a])&set(roles[b]),'Source roles overlap')
    require(fm.loc[fm.split.isin(['train','val']),'binary_label'].eq(0).all(),'Abnormal fitting/reference source')
    require(len(roles['val'])>=4,'Insufficient normal references')
    train=fm.split.eq('train').to_numpy();val=fm.split.eq('val').to_numpy();evaluate=fm.split.isin(['val','test']).to_numpy()
    em=fm.loc[evaluate].reset_index(drop=True);expected_counts=expected_support(x[evaluate],cfg)
    checkpoint_hash={};payloads={};losses={}
    for method in ('trained','untrained'):
        path=folder/f'{method}.pt';checkpoint_hash[method]=file_sha256(path)
        bundle=load_context_bundle(path,expected_provenance=dict(run_id=p['run_id'],fold=fold,method=method,roles=roles))
        payload=torch.load(path,map_location='cpu',weights_only=True);payloads[method]=payload
        modelcfg=cfg if method=='trained' else replace(cfg,pretrain_updates=0)
        require(payload['config']==asdict(modelcfg),'Checkpoint configuration differs')
        provenance=payload['provenance']
        require(provenance['normal_only_labels_checked'] and provenance['source_disjointness_checked'],'Checkpoint omitted normal/source assertions')
        require(provenance['train_source_ids']==roles['train'] and provenance['validation_source_ids']==roles['val'],'Checkpoint source IDs differ')
        require(provenance['joint_indices']==list(JOINT_INDICES),'Checkpoint joint whitelist differs')
        require(provenance['context_implementation_sha256']==p['code']['context_prediction.py'],'Checkpoint implementation differs')
        for key,values in [('train_array_sha256',x[train]),('validation_array_sha256',x[val])]:
            require(provenance[key]==hashlib.sha256(np.asarray(values,dtype=np.float32).tobytes()).hexdigest(),'Checkpoint input-array hash differs: '+key)
        h=hashlib.sha256()
        for name,value in payload['state_dict'].items():
            require(torch.isfinite(value).all().item(),'Nonfinite checkpoint tensor')
            if name.startswith('backbone.'):h.update(name[9:].encode());h.update(value.cpu().contiguous().numpy().tobytes())
        require(h.hexdigest()==provenance['frozen_backbone_sha256'],'Frozen backbone changed during head fitting')
        require(set(bundle.heads)=={f'{k}_{b}' for k in ('coordinate','latent') for b in ('own','patch','full')},'Missing prediction heads')
        require(all(not q.requires_grad for q in bundle.backbone.parameters()),'Loaded backbone is not frozen')
        history=read_json(folder/f'{method}_history.json');require(history==payload['history'],'Checkpoint/history JSON differ')
        require(history['pretrain_updates_completed']==modelcfg.pretrain_updates,'Encoder budget incomplete')
        require(history['predictor_updates_completed_per_head']==cfg.predictor_updates,'Prediction-head budget incomplete')
        require(history['predictors'][-1]['update']==cfg.predictor_updates,'Final head diagnostic missing')
        if modelcfg.pretrain_updates:require(history['pretrain'][-1]['update']==modelcfg.pretrain_updates,'Final encoder diagnostic missing')
        for branch in bundle.heads:
            for prefix in ('train_','validation_'):
                value=history['predictors'][-1].get(prefix+branch);require(value is not None and np.isfinite(value),'Missing/nonfinite final head diagnostic')
    for method,kinds in METHODS.items():
        path=folder/f'{method}_losses.csv.gz';sidecar=read_json(folder/(path.name+'.json'));cm='trained' if method=='ridge' else method
        expected=dict(run_id=p['run_id'],fold=fold,method=cm,roles=roles,checkpoint_sha256=checkpoint_hash[cm],mask_seed=17,mask_repeats=16,
                      evaluation_metadata_sha256=hashlib.sha256(em.to_csv(index=False).encode()).hexdigest())
        for k,v in expected.items():require(sidecar.get(k)==v,'Loss provenance differs: '+method+'/'+k)
        require(sidecar['sha256']==file_sha256(path),'Loss cache bytes changed: '+method)
        table=read_csv(path);require(len(table)==sidecar['rows'],'Loss row count differs');compact=compress_loss_rows(table);losses[method]=compact
        losscols=['loss_own','loss_patch','loss_full']
        require(compact.loc[compact.target_count.eq(0),losscols].isna().all().all(),'Unobserved targets contain reported losses')
        require(compact.loc[compact.target_count.gt(0),losscols].ge(0).all().all(),'Squared-error/KL loss is negative')
        require(set(compact.target_type)==set(kinds),'Loss target types differ')
        for kind in kinds:
            same_table(compact[compact.target_type.eq(kind)][expected_counts.columns],expected_counts,method+' fixed-mask support',
                       keys=['window_index','joint','patch_index'],atol=0,rtol=0)
    _assert_compact_support(losses['trained'],losses['untrained'])
    _assert_compact_support(losses['trained'][losses['trained'].target_type.eq('coordinate')],losses['ridge'])
    ridge=joblib.load(folder/'ridge.joblib');_validate_ridge_support(ridge['model'])
    require(ridge['provenance']=={k:v for k,v in read_json(folder/'ridge_losses.csv.gz.json').items() if k not in ('sha256','rows')},'Ridge fitted-model provenance differs')
    for estimator in ridge['model'].estimators.values():require(np.isfinite(estimator.coef_).all(),'Nonfinite ridge coefficients')
    score_tables=verify_profiles(folder,fold,roles,em,losses)
    verify_condition(folder,fm,checkpoint_hash['trained'],payloads['trained']['provenance'])
    state.setdefault('fold_scores',[]).append(pd.concat(score_tables,ignore_index=True))
    return dict(normal_train_sources=len(roles['train']),normal_reference_sources=len(roles['val']),outer_test_sources=len(roles['test']),
                models=3,prediction_heads=12,compact_rows={k:len(v) for k,v in losses.items()})

def verify_profiles(folder,fold,roles,em,losses):
    selections={};tables=[]
    for method,kinds in METHODS.items():
        refs=read_json(folder/f'{method}_references.json');scores=read_csv(folder/f'{method}_scores.csv')
        require(not scores.duplicated(['source_id','target_type','observation_budget']).any(),'Repeated source score identity')
        require(set(scores.source_id)==set(roles['test']) and scores.split.eq('test').all(),'Scores include wrong outer-test sources')
        require(scores.outer_fold.eq(fold).all(),'Scores belong to wrong fold')
        require(np.isfinite(scores[list(SCORES)]).all().all(),'Nonfinite source scores')
        require(set(refs)=={f'{method}_{b}_{k}' for b in ('all','budget4') for k in kinds},'Missing/extra frozen references')
        for budget in ('all','budget4'):
            regenerated,support,selection=aggregate_profiles(losses[method],em,fixed_budget=4 if budget=='budget4' else None)
            saved=read_csv(folder/f'{method}_{budget}_profiles.csv')
            # CSV float32 loss serialization limits reconstruction precision.
            same_table(regenerated.fillna(''),saved.fillna(''),method+' '+budget+' profiles',keys=['source_id','target_type'],atol=2e-6,rtol=2e-5)
            require(saved.scorable.eq(True).all(),'A frozen eligible source became unscorable')
            same_table(support.fillna(''),read_csv(folder/f'{method}_{budget}_support.csv').fillna(''),method+' '+budget+' joint support',
                       keys=['source_id','target_type','joint'],atol=2e-6,rtol=2e-5)
            if budget=='budget4':
                selections[method]=read_csv(folder/f'{method}_budget4_selection.csv');_assert_budget_selection(selection,selections[method])
                require(selections[method].groupby(['source_id','joint']).size().eq(4).all(),'Budget does not contain exactly four physical patches')
            for kind in kinds:
                group=saved[saved.target_type.eq(kind)];normal=group[group.split.eq('val')];outer=group[group.split.eq('test')].sort_values('source_id')
                current=scores[scores.target_type.eq(kind)&scores.observation_budget.eq(budget)].sort_values('source_id')
                require(len(current)==len(roles['test']),'Missing test source/target/budget score')
                references=refs[f'{method}_{budget}_{kind}'];require(set(references)==set(FIELDS),'Missing profile-reference field')
                for field in FIELDS:
                    raw=references[field];sha=raw['parameters_sha256']
                    require(canonical_hash({k:v for k,v in raw.items() if k!='parameters_sha256'})==sha,'Frozen reference SHA invalid')
                    require(raw['outer_fold']==fold and raw['source_ids']==roles['val'],'Reference fold/source identity differs')
                    require(raw['field']==field and raw['target_type']==kind and raw['sensitivity'] is False,'Reference semantics differ')
                    ref=ProfileReference.fit(normal,field,kind,training_source_ids=roles['train'],test_source_ids=roles['test'])
                    require(ref.parameters_sha256==sha,'Reference differs from normal-only validation profiles')
                    name={'benefit':'profile_score','own':'own_distance','full':'full_distance','patch':'patch_distance'}[field]
                    require(np.allclose(ref.score(outer),current[name],rtol=1e-10,atol=1e-10),'Score differs from frozen reference')
                    require(current[field+'_reference_sha256'].eq(sha).all(),'Score cites different reference SHA')
        tables.append(scores)
    for method in ('ridge','untrained'):_assert_budget_selection(selections['trained'],selections[method])
    return tables

def verify_condition(folder,metadata,encoder_sha,encoder_provenance):
    report=read_json(folder/'condition_metrics.json');path=folder/'condition_classifiers.joblib'
    require(file_sha256(path)==report['classifier_artifact_sha256'],'Condition model artifact hash differs')
    require(report['encoder_checkpoint_sha256']==encoder_sha,'Condition probe uses different encoder')
    require(report['encoder_provenance']==encoder_provenance,'Condition encoder provenance differs')
    require(report['normal_reference_sources_excluded'] and report['hyperparameters_selected_from_test'] is False,'Condition fitting protocol differs')
    focal=metadata[metadata.label.isin(FOCAL)]
    train=set(focal.loc[focal.split.isin(['train','auxiliary']),'source_id']);test=set(focal.loc[focal.split.eq('test'),'source_id'])
    require(set(report['fit_source_ids'])==train and set(report['test_source_ids'])==test,'Condition fit/test sources differ')
    require(not train&test and not train&set(metadata.loc[metadata.split.eq('val'),'source_id']),'Condition source leakage')
    artifact=joblib.load(path)
    for key in ('random_forest','embedding_scaler','embedding_probe'):require(key in artifact,'Missing condition model '+key)
    require(set(artifact['random_forest'].classes_)==set(FOCAL) and set(artifact['embedding_probe'].classes_)==set(FOCAL),'Condition class coverage differs')
    require(artifact['provenance']['encoder_checkpoint_sha256']==encoder_sha,'Saved classifier encoder linkage differs')
    rows=read_csv(folder/'condition_predictions.csv');expected=focal[focal.split.eq('test')][['source_id','label','outer_fold']].drop_duplicates()
    require(set(rows.method)=={'random_forest','normal_jepa_probe'},'Missing condition comparator')
    for method,part in rows.groupby('method'):
        require(not part.duplicated(['source_id','label','outer_fold']).any(),'Duplicate condition prediction')
        same_table(part[expected.columns],expected,'Condition held-out identities',keys=['source_id','label'])
        require(np.isfinite(part[list(FOCAL)]).all().all(),'Nonfinite condition probabilities')
        require(part[list(FOCAL)].ge(0).all().all() and part[list(FOCAL)].le(1).all().all(),'Condition probabilities outside zero to one')
        require(np.allclose(part[list(FOCAL)].sum(axis=1),1.),'Condition probabilities not normalized')
        require(np.allclose(part.groupby('source_id').source_weight.sum(),1.),'Condition source weights do not each sum to one')
        require(np.allclose(part.source_weight,1./part.groupby('source_id').source_id.transform('size')),'Mixed-label source weights are unequal')
        require(report['metrics'][method]['n_test_sources']==len(test),'Condition unique source count differs')
        require(report['metrics'][method]['n_test_source_label_pairs']==len(expected),'Condition source-label count differs')

def verify_cohort(run,state):
    registry,meta,p=state['registry'],state['meta'],state['protocol']
    require(len(state.get('fold_scores',[]))==4,'Not every fold passed independent verification')
    scores=read_csv(run/'source_scores.csv');combined=pd.concat(state['fold_scores'],ignore_index=True)
    same_table(scores,combined,'Combined source scores',keys=['source_id','method','target_type','observation_budget'])
    wide=read_csv(run/'paired_source_scores.csv');require(wide.source_id.is_unique,'Paired cohort repeats source IDs')
    same_table(wide[list(IDS)],registry[list(IDS)],'Paired cohort versus complete registry',keys=['source_id'])
    require(wide.all_methods_scorable.eq(True).all(),'Paired cohort excludes declared eligible sources')
    columns=[]
    for method,kinds in METHODS.items():
        for kind in kinds:
            for budget in ('all','budget4'):
                group=scores[scores.method.eq(method)&scores.target_type.eq(kind)&scores.observation_budget.eq(budget)].sort_values('source_id')
                require(group.source_id.is_unique and set(group.source_id)==set(registry.source_id),'Method lacks unique complete OOF coverage')
                for field in SCORES:
                    column=f'{method}_{kind}_{budget}_{field}';columns.append(column);require(column in wide,'Missing paired comparator')
                    require(np.allclose(wide.sort_values('source_id')[column],group[field],rtol=1e-12,atol=1e-12),'Paired scores differ from fold artifacts')
    require(np.isfinite(wide[columns]).all().all(),'Nonfinite paired scores')
    summary=read_json(run/'summary.json');coverage=read_json(run/'evaluation_coverage.json')
    expected=dict(sources=int(meta.source_id.nunique()),normal_sources=int(meta.loc[meta.binary_label.eq(0),'source_id'].nunique()),sequences=int(meta.sequence_id.nunique()),windows=len(meta))
    require(summary['status']=='completed' and summary['completed_folds']==[0,1,2,3],'Summary not complete four-fold run')
    require(summary['run_name']==p['run_id'] and summary['synthetic']==p['synthetic'],'Summary identity differs')
    require(summary['cohort']==expected,'Summary cohort counts differ')
    require(coverage==dict(sources=len(wide),common_sources=len(wide),excluded=[]),'Coverage differs from complete registry')
    ranking=read_json(run/'ranking_metrics.json');require(set(ranking['methods'])==set(columns),'Ranking omits paired comparator')
    cohort_hash=canonical_hash(registry[['source_id','outer_fold','binary_label']].sort_values('source_id').to_dict('records'))
    require(ranking['source_cohort_sha256']==cohort_hash,'Ranking source-cohort hash differs')
    expected_coverage=dict(n_sources_supplied=len(wide),n_common_scorable_sources=len(wide),n_excluded_sources=0,excluded_source_ids=[])
    require(ranking['coverage']==expected_coverage,'Ranking coverage differs')
    for result in ranking['methods'].values():
        require(result['all_folds_have_binary_support'],'Ranking lacks binary support')
        require(np.isfinite([result['primary_auroc'],result['normal_weighted_ap']]).all(),'Nonfinite ranking summary')
        require([r['outer_fold'] for r in result['folds']]==[0,1,2,3],'Ranking fold list differs')
        for row in result['folds']:
            group=registry[registry.outer_fold.eq(row['outer_fold'])]
            require(row['n_normal']==int(group.binary_label.eq(0).sum()) and row['n_abnormal']==int(group.binary_label.eq(1).sum()),'Ranking source support differs')
    ranking_csv=read_csv(run/'ranking_metrics.csv')
    require(ranking_csv.method.is_unique and set(ranking_csv.method)==set(columns),'Ranking CSV method coverage differs')
    require(np.isfinite(ranking_csv[['auroc','average_precision']]).all().all(),'Nonfinite ranking CSV values')
    bootstrap=read_json(run/'primary_paired_bootstrap.json')
    require(bootstrap['paired'] is True and bootstrap['bootstrap_unit']=='source within outer fold and binary label','Bootstrap is not paired by source/fold/label')
    require(bootstrap['score_a']=='trained_latent_all_profile_score' and bootstrap['score_b']=='trained_latent_all_raw_full','Primary bootstrap comparator differs')
    require(bootstrap['n_sources']==len(wide) and bootstrap['coverage']==expected_coverage,'Bootstrap source coverage differs')
    expected_normals={str(f):int((registry.outer_fold.eq(f)&registry.binary_label.eq(0)).sum()) for f in range(4)}
    require(bootstrap['normal_sources_by_fold']==expected_normals,'Bootstrap stratum sizes differ')
    require(bootstrap['repeats']==1000 and np.isfinite([bootstrap['point_difference'],*bootstrap['interval_95']]).all(),'Bootstrap is incomplete or nonfinite')
    return dict(unique_out_of_fold_sources=len(wide),paired_score_columns=len(columns),**expected)

def self_test(run_dir):
    """Reject missing/tampered artifacts on isolated copies of a valid demo."""
    root=Path(run_dir).resolve();require(read_json(root/'protocol.json')['synthetic'] is True,'Self-tests require synthetic run')
    baseline=verify_run(root,write=False);require(baseline['passed'],'Unmodified demo failed: '+str([c for c in baseline['checks'] if not c['passed']]))
    results=[]
    for mutation in ('missing_checkpoint','changed_loss_bytes','changed_reference'):
        with tempfile.TemporaryDirectory(prefix='aaai-artifact-verification-') as directory:
            copy=Path(directory)/root.name;shutil.copytree(root,copy)
            if mutation=='missing_checkpoint':(copy/'fold_2/untrained.pt').unlink()
            elif mutation=='changed_loss_bytes':
                with (copy/'fold_1/trained_losses.csv.gz').open('ab') as handle:handle.write(b'changed')
            else:
                path=copy/'fold_3/trained_references.json';value=read_json(path)
                value['trained_all_latent']['benefit']['mean'][0]+=1.;dump_json(path,value)
            result=verify_run(copy,write=True);require(not result['passed'],'Tampered artifact incorrectly passed: '+mutation)
            results.append(dict(case=mutation,rejected=True,failed_checks=[c['name'] for c in result['checks'] if not c['passed']]))
    return dict(passed=True,cases=results)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run_dir',type=Path)
    parser.add_argument('--expect-real',action='store_true',help='Reject synthetic demonstrations')
    parser.add_argument('--self-test',action='store_true',help='Test missing/tampered artifacts on copies of completed four-fold demo')
    args=parser.parse_args()
    if args.self_test:
        require(not args.expect_real,'--self-test and --expect-real are incompatible')
        print(json.dumps(self_test(args.run_dir),indent=2));return 0
    report=verify_run(args.run_dir,expect_real=args.expect_real)
    print(json.dumps(dict(passed=report['passed'],report=str(args.run_dir/'verification.json'),checks=len(report['checks']),
                         failed=[c for c in report['checks'] if not c['passed']],counts=report['counts']),indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(main())
