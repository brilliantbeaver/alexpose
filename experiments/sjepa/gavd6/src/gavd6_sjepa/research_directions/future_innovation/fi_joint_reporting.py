"""Read-only numerical reconstruction and honest deterministic development decisions."""
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .fi_contracts import (DIRECT_ARMS, GateThresholds, code_fingerprint, equal_source_weights, load_model_contract,
                           read_json, sha256_file, stable_key, verified_report_decision, write_json, write_once_json)
from .fi_cache_reuse import checked_file, load_reused_cache, verify_reused_readiness
from .fi_joint_training import verify_joint_fold
from .fi_metrics import score_arrays, source_bootstrap_counts, source_error_sums, score_source_sums
from .fi_gate_decision import decide_direct_gate


def decide_joint_gate(metrics, thresholds=None):
    """Keep direct-v2 point/bootstrap criteria; stochastic seeds are inapplicable."""
    thresholds = thresholds or GateThresholds()
    result = decide_direct_gate(metrics, thresholds, 1)
    if result['metrics'] is None: return {**result, 'protocol': 'direct-v3'}
    checks = result['checks']
    checks.pop('seeds_stable')
    valid_names = ('data_contract_valid','evaluation_contract_valid','controls_complete','input_audit_complete',
                   'target_variance_valid','teacher_stable','causal_leakage_absent')
    points = ('real_gain','time_shuffle','clip_mismatch','skeleton_increment')
    stability = ('bootstrap_positive','skeleton_increment_stable')
    decision = 'STOP' if not all(checks[k] for k in (*valid_names,*points)) else ('ADVANCE' if all(checks[k] for k in stability) else 'INCONCLUSIVE')
    return {**result, 'protocol': 'direct-v3', 'decision': decision, 'checks': checks,
            'stochastic_seed_stability': 'not_applicable: deterministic primary model',
            'allow_full_experiment': False, 'allow_jepa_training_comparison': False, 'allow_adapter_training': False,
            'development_criteria_passed': decision == 'ADVANCE',
            'next_action': {'ADVANCE':'independent_source_confirmation_before_scaling', 'STOP':'stop_this_predictor_target_comparison',
                            'INCONCLUSIVE':'plan_independent_source_confirmation'}[decision]}


def _close(actual, expected, label):
    try:
        np.testing.assert_allclose(np.asarray(actual,dtype=float), np.asarray(expected,dtype=float), rtol=1e-10, atol=1e-12, equal_nan=True)
    except (AssertionError, TypeError, ValueError) as e:
        raise ValueError(f'Numerical verification failed: {label}') from e


def verify_numerics(root, *, require_report=True):
    """No writes, hashes plus fresh prediction/score/bootstrap arithmetic.

    Scoring functions are the same verified primitives used by the forensic
    evaluator; saved summary flags and hashes never replace this recomputation.
    """
    from .fi_reporting import assemble_oof
    root = Path(root)
    readiness = verify_reused_readiness(root)
    cohort, cache = load_reused_cache(root)
    config = load_model_contract(root)
    run = read_json(root / 'config/run-contract.json')
    tables = [verify_joint_fold(root,f,loaded=(cohort,cache)) for f in range(5)]
    if not all(read_json(root / f'models/fold-{f}/fold-complete.json')['all_candidates_valid'] for f in range(5)):
        raise ValueError('Required candidate failure makes measurement incomplete')
    predictions = pd.concat(tables,ignore_index=True)
    arrays = assemble_oof(cohort,cache,predictions,config,arms=DIRECT_ARMS)
    contract = read_json(root / 'reports/scores-contract.json')
    paths = ['config/run-contract.json','config/cache-contract.json','qc/readiness-summary.json'] + [f'models/fold-{f}/fold-complete.json' for f in range(5)]
    if contract['binding'] != stable_key(*[sha256_file(root / p) for p in paths]): raise ValueError('Score binding mismatch')
    required = {'reports/aggregate-metrics.csv','reports/featurewise-r2.csv','reports/source-bootstrap.csv','reports/paired-controls.csv','reports/uncertainty.json',
                'predictions/baseline.parquet', *(f'predictions/{arm}.parquet' for arm in DIRECT_ARMS)}
    if set(contract['artifacts']) != required: raise ValueError('Scores omit required evidence')
    for path,digest in contract['artifacts'].items(): checked_file(root,path,digest)
    frame = pd.read_csv(root / 'reports/aggregate-metrics.csv')
    if frame.duplicated(['arm','seed']).any() or set(zip(frame.arm,frame.seed)) != {(a,0) for a in DIRECT_ARMS}: raise ValueError('Invalid deterministic score identities')
    features = pd.read_csv(root / 'reports/featurewise-r2.csv')
    if features.duplicated(['arm','seed','target_feature']).any() or len(features) != 4*256 or set(features.seed) != {0}: raise ValueError('Invalid featurewise score identities')
    weights = equal_source_weights(cohort.video_id)
    scores, sufficient = {}, {}
    for key,(y,b,p,valid) in arrays.items():
        arm,seed = key
        computed,rbase,rfull,mask = score_arrays(y,b,p,weights,valid)
        row = frame[(frame.arm==arm)&(frame.seed==seed)].iloc[0]
        for name,value in computed.items(): _close(row[name], value if value is not None else np.nan, f'{arm}/{name}')
        f = features[features.arm==arm].set_index('target_feature').loc[range(256)]
        if not np.array_equal(f.valid.to_numpy(), mask): raise ValueError('Wrong featurewise mask')
        _close(f.r2_baseline,rbase,'feature baseline'); _close(f.r2_full,rfull,'feature full'); _close(f.delta_r2,rfull-rbase,'feature gain')
        scores[arm] = computed
        sufficient[key] = source_error_sums(y,b,p,weights,cohort.video_id)
        # Verify all derivative prediction exports, including raw target units.
        exported = pd.read_parquet(root / f'predictions/{arm}.parquet')
        expected = predictions[predictions.arm == arm].copy()
        for name in ('y_true','y_pred_baseline','y_pred_full'):
            expected[name+'_raw'] = expected[name] * expected.target_scale + expected.target_mean
        try:
            pd.testing.assert_frame_equal(exported.reset_index(drop=True),expected.reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
        except AssertionError as e: raise ValueError('Prediction export units/identity mismatch') from e
    expected_base = predictions[predictions.arm=='real-skeleton'].copy()
    for name in ('y_true','y_pred_baseline'): expected_base[name+'_raw'] = expected_base[name]*expected_base.target_scale+expected_base.target_mean
    expected_base = expected_base.drop(columns=['y_pred_full']).assign(arm='baseline')
    try: pd.testing.assert_frame_equal(pd.read_parquet(root / 'predictions/baseline.parquet').reset_index(drop=True),expected_base.reset_index(drop=True),check_dtype=False,rtol=1e-10,atol=1e-12)
    except AssertionError as e: raise ValueError('Shared baseline export mismatch') from e
    rows=[]
    for draw,counts in enumerate(source_bootstrap_counts(cohort.video_id,config.bootstrap_repetitions)):
        for arm in DIRECT_ARMS:
            values=score_source_sums(sufficient[(arm,0)],counts,arrays[(arm,0)][3])
            rows.append({'draw':draw,'arm':arm,**values})
    boot=pd.DataFrame(rows).set_index(['draw','arm']).sort_index()
    stored=pd.read_csv(root / 'reports/source-bootstrap.csv')
    if stored.duplicated(['draw','arm']).any(): raise ValueError('Duplicated bootstrap draws')
    stored=stored.set_index(['draw','arm']).sort_index()
    if not boot.index.equals(stored.index): raise ValueError('Missing or mismatched bootstrap draws')
    _close(stored[boot.columns],boot,'source bootstrap arithmetic/multiplicity')
    uncertainty=read_json(root / 'reports/uncertainty.json')
    if uncertainty['bootstrap_repetitions'] != 2000 or uncertainty['seed_aggregation'] != config.seed_aggregation:
        raise ValueError('Wrong bootstrap or deterministic policy')
    real=boot.xs('real-skeleton',level='arm').delta_r2.to_numpy()
    _close(uncertainty['bootstrap_positive_fraction'],np.mean(real>0),'positive gain fraction')
    for arm in DIRECT_ARMS:
        for metric in ('r2_baseline','r2_full','delta_r2','f8'):
            values=boot.xs(arm,level='arm')[metric].to_numpy(dtype=float)
            values=values[np.isfinite(values)]
            interval=uncertainty['intervals'][arm][metric]
            if interval['defined_draws'] != len(values): raise ValueError('Wrong interval draw count')
            for q,name in ((.025,'ci025'),(.975,'ci975')): _close(interval[name],np.quantile(values,q) if len(values) else np.nan,'uncertainty interval')
    paired=pd.read_csv(root / 'reports/paired-controls.csv')
    if len(paired)!=3 or set(paired.control)!=set(DIRECT_ARMS)-{'real-skeleton'} or not paired.control.is_unique: raise ValueError('Invalid paired controls')
    paired=paired.set_index('control')
    for arm in paired.index:
        diff=real-boot.xs(arm,level='arm').delta_r2.to_numpy()
        actual={'real_minus_control':scores['real-skeleton']['delta_r2']-scores[arm]['delta_r2'],
                'ci025':np.quantile(diff,.025),'ci975':np.quantile(diff,.975),'real_beats_control_fraction':np.mean(diff>0)}
        for name,value in actual.items(): _close(paired.loc[arm,name],value,'paired control summary')
    metrics={f'delta_r2_{key}':scores[arm]['delta_r2'] for key,arm in (('real','real-skeleton'),('time_shuffle','time-shuffle'),('clip_mismatch','clip-mismatch'),('no_skeleton','no-skeleton'))}
    metrics.update(seed_real_gains=[metrics['delta_r2_real']],seed_skeleton_increments=[metrics['delta_r2_real']-metrics['delta_r2_no_skeleton']],
                   bootstrap_positive_fraction=float(np.mean(real>0)),skeleton_increment_positive_fraction=float(paired.loc['no-skeleton','real_beats_control_fraction']),
                   evaluation_contract_valid=True,controls_complete=True,**readiness['checks'])
    decision=decide_joint_gate(metrics,GateThresholds(**read_json(root / 'config/thresholds.json')))
    if run['synthetic']: decision.update(decision='STOP',development_criteria_passed=False,next_action='synthetic_implementation_check_only')
    increment={name:float(paired.loc['no-skeleton',column]) for name,column in (('estimate','real_minus_control'),('ci025','ci025'),('ci975','ci975'),('positive_fraction','real_beats_control_fraction'))}
    if require_report:
        saved=verified_report_decision(root)
        if saved is None or saved['measurement_complete'] is not True or saved['protocol']!='direct-v3' or saved['run_id']!=run['run_id']:
            raise ValueError('Missing or invalid final report identity')
        for name in ('decision','checks','metrics','stochastic_seed_stability','development_criteria_passed','next_action',
                     'allow_full_experiment','allow_adapter_training','allow_jepa_training_comparison'):
            if saved[name]!=decision[name]: raise ValueError(f'Final decision arithmetic differs: {name}')
        if saved['skeleton_increment']!=increment: raise ValueError('Final matched increment differs')
    return {'passed':True,'read_only':True,'run_id':run['run_id'],'protocol':'direct-v3','prediction_rows':len(predictions),
            'model_count':20,'bootstrap_rows':len(boot),'source_count':cohort.video_id.nunique(),'window_count':len(cohort),
            'target_feature_count':int(arrays[('real-skeleton',0)][3].sum()),
            'score_contract_sha256':sha256_file(root / 'reports/scores-contract.json'),
            'prediction_tolerance':{'rtol':1e-10,'atol':1e-10},'score_tolerance':{'rtol':1e-10,'atol':1e-12},
            'scores':scores,'skeleton_increment':increment,'decision':decision}


def build_joint_report(root):
    root=Path(root)
    run=read_json(root / 'config/run-contract.json')
    previous=verified_report_decision(root)
    if previous and previous.get('measurement_complete') and (root / 'reports/final-report-contract.json').exists():
        verify_numerics(root)
        return previous
    try:
        verification=verify_numerics(root,require_report=False)
        result={**verification['decision'],'measurement_complete':True,'skeleton_increment':verification['skeleton_increment']}
        reason='; '.join(k for k,v in result['checks'].items() if not v) or 'All development criteria passed; independent-source confirmation required.'
    except (ValueError,OSError,KeyError,AssertionError) as e:
        verification=None
        result={'decision':'STOP','measurement_complete':False,'metrics':None,'checks':{'complete_valid_evidence':False},
                'allow_full_experiment':False,'allow_jepa_training_comparison':False,'allow_adapter_training':False,
                'next_action':'repair_missing_or_invalid_evidence'}
        reason=str(e)
    result.update(run_id=run['run_id'],protocol='direct-v3',experiment=run['experiment'],synthetic=run['synthetic'],development=True,
                  reason=reason,seed_policy='one deterministic fit; no optimization seed replicates',
                  teacher_evidence='reused from direct-v2',run_contract_sha256=sha256_file(root / 'config/run-contract.json'),
                  config_sha256=run['config_sha256'],parent_lineage_sha256=sha256_file(root / 'config/parent-lineage.json'),
                  report_code_sha256=code_fingerprint(),stage='all source-held-out comparisons numerically verified' if verification else 'incomplete predictive measurement')
    lines=[f"# Experiment 0 direct-v3 development result: {result['decision']}",'',reason,'',
           'Past skeleton coordinates and confidence are tested beyond safely transformed RGB/nuisance inputs and the matched validity-only arm. The deterministic joint model controls RGB and skeleton coefficients with separate penalties. Inner source validation can select the exact shared RGB baseline.',
           '','Inputs use frames 0–31. The person-region target at frames 38–39 is contextualized by all 64 frames. This is prediction of contextual teacher features. The 50 clips from 43 sources have been inspected during development; source separation does not establish participant separation.','']
    if verification:
        inc=verification['skeleton_increment']
        lines += ['| Arm | Shared RGB R² | Full R² | Gain over RGB |','|---|---:|---:|---:|']
        for arm,row in verification['scores'].items(): lines.append(f"| {arm} | {row['r2_baseline']:.6f} | {row['r2_full']:.6f} | {row['delta_r2']:+.8f} |")
        lines += ['',f"Real minus no-skeleton: **{inc['estimate']:+.8f} R²**, 95% paired source-bootstrap interval **[{inc['ci025']:+.8f}, {inc['ci975']:+.8f}]**.",
                  f"Positive real-gain draws: {result['metrics']['bootstrap_positive_fraction']:.2%}; positive matched-increment draws: {inc['positive_fraction']:.2%}. Each must reach 90%. The 95% interval is a separate uncertainty summary.",
                  '', 'One deterministic solution is fitted per fold and arm. Stochastic three-seed criteria are inapplicable, not passed. The bootstrap resamples whole source videos with replacement, preserving clips and multiplicity in 2,000 paired draws. These intervals are conditional on fitted models and omit repeated training, selection and adaptive redesign. The draws add no independent sources.',
                  '', '| Criterion | Passed |','|---|---|']
        lines += [f'| {k} | {v} |' for k,v in result['checks'].items()]
        diagnostics=[r for f in range(5) for r in read_json(root / f'models/fold-{f}/fit-diagnostics.json')]
        lines += ['',f"Baseline-only selected in {sum(r['selected_type']=='baseline_only' for r in diagnostics)}/20 fold-arm fits. An inner-selected joint model may still lose on outer sources.",
                  '',f"Numerical reconstruction checked {verification['prediction_rows']:,} predictions, 20 typed selected models and {verification['bootstrap_rows']:,} bootstrap rows; all {verification['target_feature_count']} target dimensions survive the training-mask intersection.",
                  '', '[Numerical verification](numerical-verification.json) · [Full scores](aggregate-metrics.csv) · [Paired controls](paired-controls.csv) · [Uncertainty](uncertainty.json) · [Bootstrap draws](source-bootstrap.csv) · [Parent lineage](../config/parent-lineage.json)',
                  '', 'All selection candidates, fitted transforms and fold-local diagnostics are retained under `../models/fold-0/` through `fold-4/`. Teacher encoding and pixel audits were reused after integrity and arithmetic checks; raw video and teacher inference were not rerun.']
    lines += ['',f"Next action: **{result['next_action']}**.",
              'A negative result applies to this specified predictor, representation, contextual target and small data regime. It does not prove that all skeleton representations lack value. A passing result is development evidence requiring independent-source confirmation. Full S-JEPA training, adapter training and skeleton-only distillation remain separate studies and are not authorized by this run.','']
    if previous and not previous.get('measurement_complete'):
        archive=root/'reports/attempts'/sha256_file(root/'reports/gate-decision.json'); archive.mkdir(parents=True,exist_ok=True)
        for name in ('gate-decision.json','gate-report.md'):
            if (root/'reports'/name).exists(): shutil.copyfile(root/'reports'/name,archive/name)
    if verification: write_once_json(root/'reports/numerical-verification.json',verification)
    write_json(root/'reports/gate-decision.json',result)
    (root/'reports/gate-report.md').write_text('\n'.join(lines))
    if verification:
        write_once_json(root/'reports/final-report-contract.json',{'artifacts':{p:sha256_file(root/p) for p in ('reports/gate-decision.json','reports/gate-report.md','reports/numerical-verification.json')}})
        verify_numerics(root)
    return result
