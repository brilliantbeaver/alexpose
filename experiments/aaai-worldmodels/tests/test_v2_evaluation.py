"""Independent-group calibration, nuisance, and uncertainty invariants."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

from worldmodels.v2_evaluation import (
    FLOOR, aggregate_windows, calibration_schedule, empirical_mid_cdf,
    ensemble_rotation_scores, fit_normal_distance, fit_nuisance_reference,
    fit_source_calibrator, independent_score_table, normal_forecast_contrast,
    paired_within_fold_bootstrap, score_normal_distance, score_source_calibrator,
    source_covariates, validate_calibration_schedule, window_covariates,
    within_fold_summary,
)


def metadata():
    rows = []
    for label, count in [('normal', 29), ('stroke', 20)]:
        for i in range(count):
            rows.append({'source_id': f'{label}-{i:02d}', 'component_id': f'{label}-{i:02d}',
                         'sequence_id': f'{label}-{i:02d}-seq', 'label': label, 'outer_fold': i % 5})
    return pd.DataFrame(rows)


def reference_data(schedule):
    rng = np.random.default_rng(51)
    def rows(sources):
        motion = rng.uniform(.05, 2., len(sources))
        missing = rng.uniform(0, .2, len(sources))
        return pd.DataFrame({'source_id': sources, 'component_id': sources, 'label': 'normal',
                             'mse': np.exp(-3+.5*np.log(motion)+missing+rng.normal(0,.1,len(sources))),
                             'prefix_motion': motion, 'prefix_missingness': missing})
    reference = rows(schedule['predictor_fit_sources'])
    calibration = rows(schedule['calibration_sources'])
    expected = {source: fold['train_sources'] for fold in schedule['nuisance_oof']
                for source in fold['validation_sources']}
    reference['forecast_training_sources'] = [json.dumps(expected[s]) for s in reference.source_id]
    calibration['forecast_training_sources'] = [json.dumps(schedule['predictor_fit_sources'])]*len(calibration)
    return reference, calibration


def test_schedule_disjoint_roles_and_rotation_coverage():
    meta = metadata()
    for outer in range(5):
        schedules = calibration_schedule(meta, outer)
        expected = set(meta.loc[meta.label.eq('normal') & meta.outer_fold.ne(outer), 'source_id'])
        cal = []
        for s in schedules:
            assert validate_calibration_schedule(s)
            assert 15 <= len(s['predictor_fit_sources']) <= 16
            assert 7 <= len(s['calibration_sources']) <= 8
            assert set(s['calibration_sources']) | set(s['predictor_fit_sources']) == expected
            cal.extend(s['calibration_sources'])
        assert len(cal) == len(set(cal)) and set(cal) == expected
        assert schedules == calibration_schedule(meta.sample(frac=1, random_state=14), outer)


def test_related_sources_stay_together_and_split_corruption_fails():
    meta = metadata()
    extra = meta.iloc[[0]].copy(); extra['source_id'] = 'linked-upload'
    combined = pd.concat([meta,extra],ignore_index=True)
    for s in calibration_schedule(combined, 1):
        for role in ['predictor_fit_sources','calibration_sources','heldout_sources']:
            assert ('linked-upload' in s[role]) == ('normal-00' in s[role])
    s = calibration_schedule(meta, 0)[0]
    s['nuisance_oof'][0]['train_sources'].append(s['calibration_sources'][0])
    with pytest.raises(ValueError, match='out-of-fit'):
        validate_calibration_schedule(s)


def test_hierarchical_aggregation_and_component_endpoint():
    meta = pd.DataFrame({'source_id':['a','a','a','b','c'],
                         'sequence_id':['a1','a1','a2','b1','c1'],
                         'label':['normal']*5,'component_id':['ab','ab','ab','ab','c']})
    source = aggregate_windows(meta,[0,0,10,1,2])
    assert source.set_index('source_id').value.to_dict() == {'a':5,'b':1,'c':2}
    independent = independent_score_table(source,['value']).set_index('component_id').value
    assert independent.to_dict() == {'ab':3,'c':2}
    with pytest.raises(ValueError,match='finite'):
        aggregate_windows(meta,[0,0,np.nan,1,2])


def test_prefix_covariates_use_actual_time_and_ignore_future_and_missing():
    n = 3
    t = np.broadcast_to(np.arange(24)*.1,(n,24)).copy()
    x = np.zeros((n,24,16,2)); x[:,:,:,0] = t[:,:,None]
    v = np.ones(x.shape[:-1],bool);v[:,5,8]=False
    expected = window_covariates(x,v,t)
    np.testing.assert_allclose(expected.prefix_motion,1.)
    changed = x.copy();changed[:,16:]=np.nan;changed[:,5,8]=1e8
    vv=v.copy();vv[:,16:]=~vv[:,16:]
    tt=t.copy();tt[:,16:]=np.nan
    pd.testing.assert_frame_equal(expected,window_covariates(changed,vv,tt))
    half = window_covariates(x,v,t*2)
    np.testing.assert_allclose(half.prefix_motion,.5)
    meta=metadata().iloc[:3].copy()
    summary=source_covariates(x,v,t,meta)
    np.testing.assert_allclose(summary.log_prefix_motion,np.log(1+FLOOR))


def test_nuisance_ridge_matches_independent_solver_and_uses_only_normal():
    schedule=calibration_schedule(metadata(),0)[0]
    ref,_=reference_data(schedule)
    model=fit_nuisance_reference(ref,alphas=[2.])
    z=np.column_stack([np.log(ref.prefix_motion+FLOOR),ref.prefix_missingness])
    zz=(z-np.asarray(model['mean']))/np.asarray(model['scale'])
    independent=Ridge(alpha=2.).fit(zz,np.log(ref.mse+FLOOR))
    np.testing.assert_allclose(model['coefficient'],independent.coef_)
    np.testing.assert_allclose(model['intercept'],independent.intercept_)
    np.testing.assert_allclose(model['covariate_min'],z.min(0))
    np.testing.assert_allclose(model['covariate_max'],z.max(0))
    ref.loc[0,'label']='stroke'
    with pytest.raises(ValueError,match='normal'):
        fit_nuisance_reference(ref)


def test_calibration_exclusion_and_response_isolation():
    schedule=calibration_schedule(metadata(),0)[0]
    ref,cal=reference_data(schedule)
    model=fit_source_calibrator(ref,cal,schedule=schedule)
    assert len(model['nuisance_reference_rows'])==len(ref)
    altered=cal.copy();altered['mse']*=500
    other=fit_source_calibrator(ref,altered,schedule=schedule)
    assert model['nuisance_model']==other['nuisance_model']
    assert model['calibration_residuals']!=other['calibration_residuals']
    with pytest.raises(ValueError,match='overlap'):
        score_source_calibrator(model,cal)
    bad=ref.copy();bad.at[0,'forecast_training_sources']=json.dumps(schedule['predictor_fit_sources'])
    with pytest.raises(ValueError,match='provenance'):
        fit_source_calibrator(bad,cal,schedule=schedule)
    with pytest.raises(ValueError,match='provenance'):
        fit_source_calibrator(ref.drop(columns='forecast_training_sources'),cal,schedule=schedule)
    alias=cal.iloc[[0]].copy();alias['source_id']='renamed-upload'
    with pytest.raises(ValueError,match='components overlap'):
        score_source_calibrator(model,alias)


def test_midcdf_ties_endpoint_smoothing_and_symmetric_tail_scores():
    values=empirical_mid_cdf([-1,0,0,1],[-10,-1,0,1,10])
    np.testing.assert_allclose(values,[.1,.2,.5,.8,.9])
    tails=np.abs(2*values-1)
    np.testing.assert_allclose(tails,tails[::-1])
    assert tails[2]==0 and tails.max()<1
    with pytest.raises(ValueError):empirical_mid_cdf([],0)


def test_calibrator_scoring_and_three_rotation_ensemble():
    schedules=calibration_schedule(metadata(),0)
    query=pd.DataFrame({'source_id':['external-a','external-b'],'label':['normal','stroke'],
                        'mse':[.01,.3],'prefix_motion':[.1,1.],'prefix_missingness':[0.,.1]})
    outputs=[]
    for schedule in schedules:
        ref,cal=reference_data(schedule)
        model=fit_source_calibrator(ref,cal,schedule=schedule)
        scored=score_source_calibrator(model,query)
        assert np.all((scored.two_sided>=0)&(scored.two_sided<1))
        outputs.append(scored)
    combined=ensemble_rotation_scores(outputs)
    expected=np.mean([o.set_index('source_id').two_sided for o in outputs],axis=0)
    np.testing.assert_allclose(combined.two_sided,expected)
    with pytest.raises(ValueError,match='three'):ensemble_rotation_scores(outputs[:2])


def score_frame():
    # Two folds deliberately differ in score offsets, making pooled and
    # within-fold rankings different and exposing accidental cross-fold pairs.
    return pd.DataFrame({'source_id':list('abcdefgh'), 'label':['normal','normal','stroke','stroke']*2,
                         'outer_fold':[0]*4+[1]*4,
                         'candidate':[0,2,1,3,100,102,101,103],
                         'reference':[0,0,1,1,100,100,101,101]})


def test_within_fold_auc_and_paired_bootstrap_exact_point_and_ties():
    frame=score_frame()
    summary=within_fold_summary(frame,'candidate')
    assert summary['pair_weighted_auroc']==.75
    assert roc_auc_score(frame.label.ne('normal'),frame.candidate)!=.75
    result=paired_within_fold_bootstrap(frame,'candidate','reference',reps=500,seed=1)
    assert result['estimate']==-.25
    assert result['candidate_auroc']==.75 and result['reference_auroc']==1.
    same=paired_within_fold_bootstrap(frame,'candidate','candidate',reps=500,seed=1)
    assert same['estimate']==0 and same['ci95']==[0.,0.]
    assert result==paired_within_fold_bootstrap(frame,'candidate','reference',reps=500,seed=1)


def test_related_component_count_and_conflicting_fold_guard():
    frame=score_frame();frame['component_id']=frame.source_id
    frame.loc[1,'component_id']='a'
    summary=within_fold_summary(frame,'candidate')
    assert summary['independent_components']==7
    frame.loc[1,'outer_fold']=1
    with pytest.raises(ValueError,match='conflicting'):
        within_fold_summary(frame,'candidate')


def test_normal_forecast_pair_and_distance_controls():
    frame=pd.DataFrame({'source_id':list('abc'),'label':'normal','method':[1.,2.,3.],'reference':[2.,4.,6.]})
    result=normal_forecast_contrast(frame,'method','reference',reps=100)
    assert result['estimate']==2 and result['relative_reduction']==.5
    x=np.array([[-1.,0],[0.,0],[1.,0]])
    model=fit_normal_distance(x,list('abc'))
    distance=score_normal_distance(model,np.array([[0.,0],[2.,0],[0.,2]]))
    assert distance[0]==0 and distance[1]>0 and distance[2]>0
    assert model['training_sources']==list('abc')


def test_schedule_refuses_too_few_normal_sources():
    meta=metadata().iloc[:5]
    with pytest.raises(ValueError,match='Insufficient'):
        calibration_schedule(meta,0)
