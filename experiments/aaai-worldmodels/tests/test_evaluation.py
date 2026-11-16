import numpy as np
import pytest
from worldmodels.data import LABELS
from worldmodels.evaluation import (aggregate_sources, classification_report, paired_source_bootstrap,
                                   validation_threshold, equal_source_weights, fit_rf, align_probabilities)
from worldmodels.features import build_features


def test_source_aggregation_equalizes_sequence_duration():
    ids,y,p=aggregate_sources(np.array([[1.,0],[1,0],[0,1]]),['normal']*3,['s']*3,['a','a','b'])
    np.testing.assert_allclose(p,[[.5,.5]])
    assert ids.tolist()==['s']


def test_matched_bootstrap_null_difference():
    y=np.repeat(LABELS,3)
    p=np.eye(5)[np.repeat(np.arange(5),3)]
    report=classification_report(y,p)
    assert report['macro_f1']==1
    assert report['abnormal_auroc']==1
    result=paired_source_bootstrap(y,p,p,repetitions=30)
    assert result['estimate']==0 and result['ci95']==[0,0]


def test_small_calibration_cannot_guarantee_five_percent():
    assert validation_threshold(np.arange(6))==float('inf')
    assert np.isfinite(validation_threshold(np.arange(6),false_positive_rate=.2))


def test_weights_equalize_sources_within_class():
    weights=equal_source_weights(['a','a','b','c'],['normal','normal','normal','stroke'])
    assert weights[:2].sum()==pytest.approx(weights[2])
    assert weights[:3].sum()==pytest.approx(weights[3])


def test_missing_features_not_zero_and_rf_imputation_train_only():
    x=np.ones((5,16,16,2)); valid=np.zeros((5,16,16),bool)
    features,names=build_features(x,valid)
    assert features.shape==(5,37)
    assert np.isnan(features).all()
    estimator=fit_rf(np.eye(5),np.array(LABELS),np.arange(5),n_estimators=10,min_samples_leaf=1)
    p=align_probabilities(estimator,np.eye(5))
    assert p.shape==(5,5)
    np.testing.assert_allclose(p.sum(1),1)
