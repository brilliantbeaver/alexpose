"""Nested source learning curves, paired conditional uncertainty and reconstruction."""
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import atomic_save_joblib
from ..future_innovation.fi_contracts import DIRECT_ARMS, read_json, write_json, write_once_json, sha256_file, save_npz, equal_source_weights, stage_lock
from ..future_innovation.fi_joint_models import JointModelContract, baseline_schema, SupportedInput, TargetStandardizer, JointRidge, skeleton_schema, temporal_features
from .fi_scaling_nested import nested_partition
from ..future_innovation.fi_joint_training import predict_selected, choose_candidate, summarize, candidate_specs, error_record
from ..future_innovation.fi_nested_training import partition_control
from ..future_innovation.fi_metrics import score_arrays, source_error_sums, source_bootstrap_counts, score_source_sums
from .fi_scaling_cohort import POLICY, PROTOCOL, read_study, make_plan
from .fi_scaling_data import load_expanded, seal, verify_seal
from ..future_innovation.fi_cohort import assign_source_folds


def freeze_plan(root,parent=None):
    root=Path(root)
    read_study(root,parent,require_software=True)
    cohort,_=load_expanded(root,parent)
    plan=make_plan(cohort,pd.read_csv(root/'config/source-reservation.csv'))
    write_once_json(root/'manifests/learning-plan.json',plan)
    seal(root,'manifests/plan-complete.json',[root/'manifests/learning-plan.json'],
         cache_sha256=sha256_file(root/'data/cache-complete.json'),
         study_sha256=sha256_file(root/'config/study.json'))
    return dict(unique_fits=len(plan['fits']),logical_entries=len(plan['entries']),
                unavailable_entries=sum(e['status']=='unavailable' for e in plan['entries']))


def checked_plan(root,cohort):
    receipt=verify_seal(root,'manifests/plan-complete.json')
    if receipt['cache_sha256']!=sha256_file(Path(root)/'data/cache-complete.json') or receipt['study_sha256']!=sha256_file(Path(root)/'config/study.json'):
        raise ValueError('Learning plan cache/study binding changed')
    expected=make_plan(cohort,pd.read_csv(Path(root)/'config/source-reservation.csv'))
    actual=read_json(Path(root)/'manifests/learning-plan.json')
    if actual!=expected: raise ValueError('Frozen nested subsets/splits changed')
    return actual


def indices(cohort,windows):
    lookup=dict(zip(cohort.window_id,range(len(cohort))))
    if len(set(windows))!=len(windows) or not set(windows)<=set(lookup):
        raise ValueError('Wrong window identities')
    return np.array([lookup[w] for w in windows],dtype=int)


def prediction_payload(result,cohort,arrays,train,test):
    base=result['baseline']
    payload=dict(window_id=cohort.iloc[test].window_id.to_numpy(dtype=str),
                 video_id=cohort.iloc[test].video_id.to_numpy(dtype=str),
                 y_true=base.y_scaler.transform(arrays['person'][test]),
                 y_raw=arrays['person'][test], target_mean=base.y_scaler.mean,
                 target_scale=base.y_scaler.scale,valid_features=base.valid_features,
                 baseline=base.predict(arrays['baseline'][test]))
    for arm in DIRECT_ARMS:
        skeleton=partition_control(arm,test,cohort,arrays,f'{arm}/reload-test',[])
        payload[arm]=predict_selected(result['models'][arm],base,arrays['baseline'][test],skeleton)
    return payload


def run(root,parent=None,fold=None):
    root=Path(root)
    read_study(root,parent,require_software=True)
    cohort,arrays=load_expanded(root,parent)
    plan=checked_plan(root,cohort)
    if fold is not None and fold not in range(5): raise ValueError('Fold must be 0..4')
    schema=baseline_schema(read_json(root/'config/nuisance-schema.json'))
    for identity,spec in plan['fits'].items():
        if fold is not None and spec['outer_fold']!=fold: continue
        with stage_lock(root,f'scaling-fit-{identity}'):
            receipt=f'models/{identity}/complete.json'
            if (root/receipt).exists():
                verify_fit(root,identity,cohort,arrays,plan)
                continue
            if (root/'reports/complete.json').exists(): raise ValueError('Report is sealed; no fitting')
            train,test=indices(cohort,spec['train_windows']),indices(cohort,spec['test_windows'])
            started=time.monotonic()
            result=nested_partition(cohort,arrays,train,test,schema,JointModelContract(),
                                    penalty_reference_windows=POLICY['penalty_reference_windows'],group_column='source_group')
            directory=root/f'models/{identity}'
            directory.mkdir(parents=True,exist_ok=True)
            # Save selected objects and every rejected/failed candidate as distinct evidence.
            modelpath=directory/'selected.joblib'
            atomic_save_joblib(modelpath,{'baseline':result['baseline'],'models':result['models']})
            files=[modelpath]
            for key in ('selection','baseline_selection','split','preprocessing','diagnostics'):
                path=directory/f'{key}.json'
                write_json(path,result[key]); files.append(path)
            payload=prediction_payload(joblib.load(modelpath),cohort,arrays,train,test)
            for arm in DIRECT_ARMS:
                np.testing.assert_array_equal(payload[arm],result['predictions'][arm])
            pred=root/f'predictions/{identity}.npz'
            save_npz(pred,**payload); files.append(pred)
            seal(root,receipt,files,fit_id=identity,plan_sha256=sha256_file(root/'manifests/learning-plan.json'),
                 specification=spec,all_candidates_valid=all(r['complete'] for r in result['diagnostics']),
                 elapsed_seconds=time.monotonic()-started,fit_identity='deterministic; source subset is replication unit')
            verify_fit(root,identity,cohort,arrays,plan)
            print(f'Completed {identity}: fold {spec["outer_fold"]}, {len(spec["train_sources"])} sources/{len(train)} clips',flush=True)


def assert_tree(actual,expected):
    if isinstance(expected,dict):
        if not isinstance(actual,dict) or set(actual)!=set(expected): raise ValueError('Artifact schema mismatch')
        for k in expected: assert_tree(actual[k],expected[k])
    elif isinstance(expected,(tuple,list)):
        if len(actual)!=len(expected): raise ValueError('Artifact length mismatch')
        for a,b in zip(actual,expected): assert_tree(a,b)
    elif isinstance(expected,(float,np.floating)):
        if not np.isclose(actual,expected,rtol=1e-10,atol=1e-12): raise ValueError('Numerical artifact mismatch')
    elif actual!=expected: raise ValueError('Artifact identity/value mismatch')


def verify_fit(root,identity,cohort,arrays,plan):
    root=Path(root)
    directory=root/f'models/{identity}'
    receipt=verify_seal(root,f'models/{identity}/complete.json')
    spec=plan['fits'][identity]
    expected_files={f'models/{identity}/{k}.json' for k in ('selection','baseline_selection','split','preprocessing','diagnostics')}
    expected_files|={f'models/{identity}/selected.joblib',f'predictions/{identity}.npz'}
    if set(receipt['artifacts'])!=expected_files or receipt['specification']!=spec or receipt['plan_sha256']!=sha256_file(root/'manifests/learning-plan.json'):
        raise ValueError('Fit identities/inventory changed')
    selected=joblib.load(directory/'selected.joblib')
    train,test=indices(cohort,spec['train_windows']),indices(cohort,spec['test_windows'])
    base=selected['baseline']
    if set(selected['models'])!=set(DIRECT_ARMS): raise ValueError('Missing/unknown arm')
    weights=equal_source_weights(cohort.iloc[train].video_id)
    xs=SupportedInput.fit(arrays['baseline'][train],weights,cohort.iloc[train].window_id,cohort.iloc[train].video_id,
                          *baseline_schema(read_json(root/'config/nuisance-schema.json')))
    assert_tree(base.x_scaler.record(),xs.record())
    ys=TargetStandardizer.fit(arrays['person'][train],weights,cohort.iloc[train].window_id)
    for k in ('mean','scale','variance'):
        np.testing.assert_allclose(getattr(base.y_scaler,k),getattr(ys,k),rtol=1e-12,atol=1e-12)
    if tuple(base.y_scaler.training_window_ids)!=tuple(cohort.iloc[train].window_id) or not np.array_equal(base.valid_features,ys.variance>1e-10):
        raise ValueError('Target training identity/mask changed')
    records=read_json(directory/'selection.json')
    config=JointModelContract()
    expected_ids={s['candidate_id'] for s in candidate_specs(config)}|{'baseline_only'}
    ref_records=read_json(directory/'baseline_selection.json')
    if [r['lambda_x'] for r in ref_records]!=list(config.ridge_alphas): raise ValueError('RGB grid changed')
    for r in ref_records:
        assert_tree(r,summarize({k:r[k] for k in ('candidate_id','candidate_type','lambda_x','lambda_s')},r['inner_folds']))
    best=min(r['pooled_loss'] for r in ref_records)
    alpha=max(r['lambda_x'] for r in ref_records if r['pooled_loss']<=best+config.tie_atol+config.tie_rtol*abs(best))
    if base.ridge.lambda_x != alpha*len(train)/40: raise ValueError('Effective reference penalty changed')
    refit=JointRidge.fit(xs.transform(arrays['baseline'][train]),None,ys.transform(arrays['person'][train]),weights,base.ridge.lambda_x)
    for key in ('x_weight','s_weight','intercept'):
        np.testing.assert_allclose(getattr(base.ridge,key),getattr(refit,key),rtol=1e-10,atol=1e-10)
    # Rebuild every learned input/target boundary and all partition-local donors.
    # Rejected coefficient fits are not repeated; their pooling and selection are.
    preprocessing=read_json(directory/'preprocessing.json')
    split=read_json(directory/'split.json')
    groups=assign_source_folds(cohort.iloc[train].source_group,3)
    labels=np.array([groups[g] for g in cohort.iloc[train].source_group])
    expected_inner=[]; expected_donors=[]
    for f in range(3):
        fit,val=train[labels!=f],train[labels==f]
        expected_inner.append(dict(fold=f,train=cohort.iloc[fit].window_id.tolist(),validation=cohort.iloc[val].window_id.tolist()))
        w=equal_source_weights(cohort.iloc[fit].video_id)
        xi=SupportedInput.fit(arrays['baseline'][fit],w,cohort.iloc[fit].window_id,cohort.iloc[fit].video_id,
                              *baseline_schema(read_json(root/'config/nuisance-schema.json')))
        yi=TargetStandardizer.fit(arrays['person'][fit],w,cohort.iloc[fit].window_id)
        assert_tree(preprocessing[f'inner-{f}']['x'],xi.record())
        for key in ('mean','scale','variance'):
            np.testing.assert_allclose(preprocessing[f'inner-{f}']['y'][key],getattr(yi,key),rtol=1e-12,atol=1e-12)
        if preprocessing[f'inner-{f}']['y']['training_window_ids']!=list(yi.training_window_ids):
            raise ValueError('Inner target training IDs changed')
        assert_tree(preprocessing[f'inner-{f}']['heldout_x'],xi.diagnostics(arrays['baseline'][val]))
        for candidate in ref_records:
            row=candidate['inner_folds'][f]
            assert_tree(row['effective_lambda_x'],candidate['lambda_x']*len(fit)/40)
            if row['fit_windows']!=len(fit): raise ValueError('Inner fit size changed')
        for arm in DIRECT_ARMS:
            sf,sv=[temporal_features(partition_control(arm,ids,cohort,arrays,f'{arm}/inner-{f}/{label}',expected_donors))
                   for ids,label in ((fit,'train'),(val,'validation'))]
            si=SupportedInput.fit(sf,w,cohort.iloc[fit].window_id,cohort.iloc[fit].video_id,*skeleton_schema())
            assert_tree(preprocessing[f'{arm}/inner-{f}']['s'],si.record())
            assert_tree(preprocessing[f'{arm}/inner-{f}']['heldout_s'],si.diagnostics(sv))
    if split['outer_train']!=spec['train_windows'] or split['outer_test']!=spec['test_windows'] or split['inner']!=expected_inner:
        raise ValueError('Saved source/inner split audit changed')
    assert_tree(preprocessing['outer']['x'],xs.record())
    assert_tree(preprocessing['outer']['heldout_x'],xs.diagnostics(arrays['baseline'][test]))
    expected_specs={s['candidate_id']:s for s in candidate_specs(config)}
    reference=next(r for r in ref_records if r['lambda_x']==alpha)
    complete=True
    diagnostics=read_json(directory/'diagnostics.json')
    if [d['arm'] for d in diagnostics]!=list(DIRECT_ARMS): raise ValueError('Diagnostic arm inventory changed')
    for arm in DIRECT_ARMS:
        candidates=records[arm]['candidates']
        if len(candidates)!=len(expected_ids) or {r['candidate_id'] for r in candidates}!=expected_ids: raise ValueError('Candidate inventory changed')
        winner,reason=choose_candidate(candidates,config)
        if records[arm]['winner']!=winner['candidate_id'] or records[arm]['reason']!=reason: raise ValueError('Selection changed')
        baseline_record=next(c for c in candidates if c['candidate_type']=='baseline_only')
        assert_tree(baseline_record['inner_folds'],reference['inner_folds'])
        for c in candidates:
            if c['candidate_id']!='baseline_only':
                for k,v in expected_specs[c['candidate_id']].items(): assert_tree(c[k],v)
                for row in c['inner_folds']:
                    n=len(expected_inner[row['inner_fold']]['train'])
                    for key in ('x','s'): assert_tree(row['effective_lambda_'+key],c['lambda_'+key]*n/40)
                    if row['fit_windows']!=n: raise ValueError('Joint inner fit size changed')
            summarized=summarize({k:c[k] for k in ('candidate_id','candidate_type','lambda_x','lambda_s')},c['inner_folds'])
            for k in ('valid','pooled_loss'): assert_tree(c[k],summarized[k])
            improvement=baseline_record['pooled_loss']-c['pooled_loss'] if c['valid'] else None
            assert_tree(c['improvement_over_baseline'],improvement)
            if c['selected']!=(c['candidate_id']==winner['candidate_id']): raise ValueError('Selected candidate label changed')
            expected_reason=reason if c['selected'] else ('rejected: invalid numerical fit' if not c['valid'] else 'rejected by pooled inner loss/tie rule')
            if c['selection_reason']!=expected_reason: raise ValueError('Candidate selection reason changed')
        complete &= all(c['valid'] for c in candidates)
        saved=selected['models'][arm]
        if saved['candidate_id']!=winner['candidate_id'] or saved['training_window_ids']!=spec['train_windows']:
            raise ValueError('Wrong selected checkpoint or training rows')
        sf=temporal_features(partition_control(arm,train,cohort,arrays,f'{arm}/outer/train',expected_donors))
        st=temporal_features(partition_control(arm,test,cohort,arrays,f'{arm}/outer/test',expected_donors))
        ss=SupportedInput.fit(sf,weights,cohort.iloc[train].window_id,cohort.iloc[train].video_id,*skeleton_schema())
        assert_tree(saved['s_scaler'].record(),ss.record())
        assert_tree(preprocessing[f'{arm}/outer']['s'],ss.record())
        assert_tree(preprocessing[f'{arm}/outer']['heldout_s'],ss.diagnostics(st))
        for key in ('x','s'):
            nominal=winner['lambda_'+key]
            expected=None if nominal is None else nominal*len(train)/40
            assert_tree(saved['effective_lambda_'+key],expected)
        if saved['checkpoint_type']=='joint_ridge':
            refit=JointRidge.fit(xs.transform(arrays['baseline'][train]),ss.transform(sf),ys.transform(arrays['person'][train]),weights,
                                saved['effective_lambda_x'],saved['effective_lambda_s'])
            for key in ('x_weight','s_weight','intercept'):
                np.testing.assert_allclose(getattr(saved['model'],key),getattr(refit,key),rtol=1e-10,atol=1e-10)
        diagnostic=next(d for d in diagnostics if d['arm']==arm)
        training_prediction=predict_selected(saved,base,arrays['baseline'][train],sf,features=True)
        expected_mse=error_record(ys.transform(arrays['person'][train]),training_prediction,cohort.iloc[train].video_id,base.valid_features)['validation_mse']
        assert_tree(diagnostic['training_mse'],expected_mse)
        assert_tree(diagnostic['selected_inner_mse'],winner['pooled_loss'])
        for key,value in dict(selected_type=saved['checkpoint_type'],selected_candidate=saved['candidate_id'],
                              x_features=len(xs.mask),x_supported=int(xs.mask.sum()),s_features=len(ss.mask),
                              s_supported=int(ss.mask.sum()),complete=all(c['valid'] for c in candidates)).items():
            assert_tree(diagnostic[key],value)
    if receipt['all_candidates_valid'] is not complete: raise ValueError('Completeness flag changed')
    if sorted(split['donors'],key=lambda x:x['scope'])!=sorted(expected_donors,key=lambda x:x['scope']):
        raise ValueError('Partition-local donors changed')
    expected=prediction_payload(selected,cohort,arrays,train,test)
    with np.load(root/f'predictions/{identity}.npz',allow_pickle=False) as payload:
        if set(payload.files)!=set(expected): raise ValueError('Prediction schema changed')
        for k,v in expected.items():
            if np.asarray(v).dtype.kind in 'USb': np.testing.assert_array_equal(payload[k],v)
            else: np.testing.assert_allclose(payload[k],v,rtol=1e-10,atol=1e-10)
    return expected


def compute_curve(plan,payloads,policy=None,synthetic=False):
    """Same paired source draws across sizes and source-subset repetitions."""
    policy=policy or POLICY
    mask=np.logical_and.reduce([v['valid_features'] for v in payloads.values()])
    if not mask.any(): raise ValueError('No common training-valid targets across learning curve')
    sizes=[str(s) for s in policy['nominal_source_sizes']]+['all']
    seeds=policy['subset_seeds']
    metrics,raw_rows,boot_rows=[],[],[]
    available=[]
    for size in sizes:
        entries=[e for e in plan['entries'] if e['size']==size]
        if any(e['status']!='planned' for e in entries): continue
        available.append(size)
        for seed in seeds:
            subset=sorted([e for e in entries if e['subset_seed']==seed],key=lambda e:e['outer_fold'])
            if len(subset)!=policy['outer_folds']: raise ValueError('Missing fold entry')
            payload={k:np.concatenate([payloads[e['fit_id']][k] for e in subset]) for k in ('window_id','video_id','y_true','y_raw','baseline',*DIRECT_ARMS)}
            if len(set(payload['window_id']))!=len(payload['window_id']) or set(payload['window_id'])!=set(plan['common_evaluation_windows']):
                raise ValueError('Missing/duplicate common evaluation predictions')
            videos=payload['video_id']; weights=equal_source_weights(videos)
            sums={}
            for arm in DIRECT_ARMS:
                score,*_=score_arrays(payload['y_true'],payload['baseline'],payload[arm],weights,mask)
                metrics.append(dict(size=size,subset_seed=seed,arm=arm,**score))
                sums[arm]=source_error_sums(payload['y_true'],payload['baseline'],payload[arm],weights,videos)
                # Inverse-transform fold by fold; standardized targets differ by subset.
                raw_preds=np.concatenate([payloads[e['fit_id']][arm]*payloads[e['fit_id']]['target_scale']+payloads[e['fit_id']]['target_mean'] for e in subset])
                raw_base=np.concatenate([payloads[e['fit_id']]['baseline']*payloads[e['fit_id']]['target_scale']+payloads[e['fit_id']]['target_mean'] for e in subset])
                raw_rows.append(dict(size=size,subset_seed=seed,arm=arm,
                    mse=float(np.average(((payload['y_raw'][:,mask]-raw_preds[:,mask])**2).mean(axis=1),weights=weights)),
                    baseline_mse=float(np.average(((payload['y_raw'][:,mask]-raw_base[:,mask])**2).mean(axis=1),weights=weights))))
            for draw,multiplicity in enumerate(source_bootstrap_counts(videos,policy['bootstrap_repetitions'],policy['bootstrap_seed'])):
                scores={arm:score_source_sums(s,multiplicity,mask) for arm,s in sums.items()}
                for arm,s in scores.items():
                    boot_rows.append(dict(size=size,subset_seed=seed,draw=draw,arm=arm,**s))
    if 'all' not in available: raise ValueError('Required full-training endpoint unavailable')
    frame=pd.DataFrame(metrics); boot=pd.DataFrame(boot_rows); raw=pd.DataFrame(raw_rows)
    summary=[]; paired={}
    for size in available:
        point=frame[frame['size']==size].groupby('arm')[['r2_full','r2_baseline','delta_r2']].mean()
        draws=boot[boot['size']==size].groupby(['draw','arm'])[['r2_full','r2_baseline','delta_r2']].mean()
        paired[size]={}
        for name,values,estimate in (
            ('real_gain',draws.xs('real-skeleton',level='arm').delta_r2,point.loc['real-skeleton','delta_r2']),
            ('matched_increment',draws.xs('real-skeleton',level='arm').r2_full-draws.xs('no-skeleton',level='arm').r2_full,
             point.loc['real-skeleton','r2_full']-point.loc['no-skeleton','r2_full']),
            ('real_minus_shuffle',draws.xs('real-skeleton',level='arm').r2_full-draws.xs('time-shuffle',level='arm').r2_full,
             point.loc['real-skeleton','r2_full']-point.loc['time-shuffle','r2_full'])):
            paired[size][name]=values.to_numpy()
            summary.append(dict(size=size,contrast=name,estimate=float(estimate),ci95=np.quantile(values,[.025,.975]).tolist(),positive_fraction=float((values>0).mean())))
    largest=frame[frame['size']=='all'].groupby('arm')[['r2_full','r2_baseline','delta_r2']].mean()
    gain=float(largest.loc['real-skeleton','delta_r2'])
    matched=float(largest.loc['real-skeleton','r2_full']-largest.loc['no-skeleton','r2_full'])
    checks=dict(real_gain=gain>=.05,shuffle=gain>=2*max(float(largest.loc['time-shuffle','delta_r2']),0),
                mismatch=float(largest.loc['clip-mismatch','delta_r2'])<=.01,matched_increment=matched>0,
                real_bootstrap=float((paired['all']['real_gain']>0).mean())>=.9,
                matched_bootstrap=float((paired['all']['matched_increment']>0).mean())>=.9)
    point_pass=all(checks[k] for k in ('real_gain','shuffle','mismatch','matched_increment'))
    status='development_advance_requires_confirmation' if all(checks.values()) else ('inconclusive' if point_pass else 'development_stop')
    if synthetic: status='synthetic_calibration_only'
    elif len(available)<2: status='insufficient_training_sizes'
    growth=None
    if len(available)>1:
        small=available[0]
        values=paired['all']['matched_increment']-paired[small]['matched_increment']
        estimates={(r['size'],r['contrast']):r['estimate'] for r in summary}
        growth=dict(comparison=f'all minus {small}',estimate=estimates[('all','matched_increment')]-estimates[(small,'matched_increment')],
                    ci95=np.quantile(values,[.025,.975]).tolist(),positive_fraction=float((values>0).mean()))
    report=dict(protocol=PROTOCOL,status=status,measurement_complete=True,synthetic=synthetic,
                training_target_mask=mask.tolist(),available_sizes=available,
                scaling_question_evaluable=len(available)>1,
                unavailable_sizes=[s for s in sizes if s not in available],
                unique_fits=len(payloads),evaluation_sources=len(plan['common_evaluation_sources']),
                subset_seeds=seeds,seed_meaning='source-subset sampling; deterministic solver; identical endpoints referenced once',
                checks=checks,contrasts=summary,growth=growth,
                arm_means=frame.groupby(['size','arm'])[['r2_full','r2_baseline','delta_r2']].mean().reset_index().to_dict('records'),
                raw_error_means=raw.groupby(['size','arm'])[['mse','baseline_mse']].mean().reset_index().to_dict('records'),
                uncertainty='conditional paired whole-source bootstrap; excludes retraining, subset selection and adaptive development',
                action='confirmation design only if largest endpoint passes; otherwise inspect raw-error and matched-increment curves; no automatic student training')
    individual=[]
    for (size,seed),rows in frame.groupby(['size','subset_seed']):
        rows=rows.set_index('arm')
        individual.append(dict(size=size,subset_seed=int(seed),real_gain=float(rows.loc['real-skeleton','delta_r2']),
            matched_increment=float(rows.loc['real-skeleton','r2_full']-rows.loc['no-skeleton','r2_full']),
            real_minus_shuffle=float(rows.loc['real-skeleton','r2_full']-rows.loc['time-shuffle','r2_full']),
            mismatch_gain=float(rows.loc['clip-mismatch','delta_r2'])))
    report['per_subset_contrasts']=individual
    report['training_counts']=[{k:e[k] for k in ('outer_fold','subset_seed','size','status','actual_sources','actual_windows') if k in e} for e in plan['entries']]
    return report,frame,boot,raw


def plot_curve(root,result):
    """A rendered curve exists only after all required numerical results exist."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sizes=result['available_sizes']; x=np.arange(len(sizes))
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    means=pd.DataFrame(result['arm_means'])
    raw=pd.DataFrame(result['raw_error_means'])
    for arm,label,color in [('real-skeleton','Real history','#087e8b'),('no-skeleton','Validity control','#bd632f'),('time-shuffle','Time shuffle','#7570b3'),('clip-mismatch','Clip mismatch','#666666')]:
        table=means[means.arm==arm].set_index('size').loc[sizes]
        axes[0].plot(x,table.r2_full,'o-',label=label,color=color)
    table=means[means.arm=='real-skeleton'].set_index('size').loc[sizes]
    axes[0].plot(x,table.r2_baseline,'s--',label='RGB reference',color='#222222')
    contrast={r['size']:r for r in result['contrasts'] if r['contrast']=='matched_increment'}
    values=np.array([contrast[s]['estimate'] for s in sizes]); intervals=np.array([contrast[s]['ci95'] for s in sizes])
    axes[1].plot(x,values,'o-',color='#087e8b')
    # Percentile intervals need not contain the point estimate; plot endpoints directly.
    axes[1].vlines(x,intervals[:,0],intervals[:,1],color='#087e8b',linewidth=2)
    axes[1].axhline(0,color='#666666',linewidth=1,linestyle='--')
    r=raw[raw.arm=='real-skeleton'].set_index('size').loc[sizes]
    axes[2].plot(x,r.mse,'o-',color='#087e8b',label='Real history')
    axes[2].plot(x,r.baseline_mse,'s--',color='#222222',label='RGB reference')
    for ax,title,ylabel in zip(axes,['Contextual feature prediction','Matched skeleton increment','Raw teacher-unit error'],['Predictive R²','Real − validity control R²','Source-weighted MSE']):
        ax.set(title=title,ylabel=ylabel,xlabel='Requested training sources per outer fold')
        ax.set_xticks(x,sizes); ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8); axes[2].legend(fontsize=8)
    fig.suptitle('Development learning curve; intervals condition on saved fits')
    path=Path(root)/'reports/learning-curve.svg'; fig.savefig(path); plt.close(fig)
    return path


def report(root,parent=None,verify=False):
    root=Path(root)
    study,_=read_study(root,parent)
    cohort,arrays=load_expanded(root,parent)
    plan=checked_plan(root,cohort)
    payloads={}
    for identity in plan['fits']:
        payloads[identity]=verify_fit(root,identity,cohort,arrays,plan)
        if not read_json(root/f'models/{identity}/complete.json')['all_candidates_valid']:
            raise ValueError('Required candidate failed; scientific measurement incomplete')
    computed,metrics,bootstrap,raw=compute_curve(plan,payloads,synthetic=study['synthetic'])
    if study.get('cohort_policy') is not None:
        computed['cohort_policy'] = study['cohort_policy']
        computed['cohort_availability'] = read_json(root/'config/availability-contract.json')['summary']
    computed['fallback_count']=sum(m['checkpoint_type']=='baseline_only' for identity in plan['fits'] for m in joblib.load(root/f'models/{identity}/selected.joblib')['models'].values())
    if verify or (root/'reports/complete.json').exists():
        verify_seal(root,'reports/complete.json')
        assert_tree(read_json(root/'reports/learning-curve.json'),computed)
        for name,expected in (('per-subset.csv',metrics),('source-bootstrap.parquet',bootstrap),('raw-errors.csv',raw)):
            actual=pd.read_parquet(root/'reports'/name) if name.endswith('parquet') else pd.read_csv(root/'reports'/name,dtype={'size':str})
            pd.testing.assert_frame_equal(actual,expected,check_dtype=False,rtol=1e-10,atol=1e-12)
        return dict(status='numerically_verified',result=computed)
    write_json(root/'reports/learning-curve.json',computed)
    metrics.to_csv(root/'reports/per-subset.csv',index=False)
    bootstrap.to_parquet(root/'reports/source-bootstrap.parquet',index=False)
    raw.to_csv(root/'reports/raw-errors.csv',index=False)
    plot_curve(root,computed)
    seal(root,'reports/complete.json',[root/'reports'/name for name in ('learning-curve.json','per-subset.csv','source-bootstrap.parquet','raw-errors.csv','learning-curve.svg')],
         plan_sha256=sha256_file(root/'manifests/learning-plan.json'))
    return computed
