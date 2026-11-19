"""Condition recognition keeps paired source weights and deployable artifacts."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
import torch

from aaai_gait.context_prediction import ContextBundle,ContextConfig,save_context_bundle
from aaai_gait.demo import synthetic_corpus
from aaai_gait.features import build_features
from aaai_gait.model import SJEPA
from aaai_gait.study_baselines import condition_capstone


def test_mixed_condition_source_has_one_total_weight_and_saved_classifiers(tmp_path):
    torch.set_num_threads(2)
    x,meta=synthetic_corpus(sources_per_class=3);meta['outer_fold']=0
    # Two abnormal conditions belong to one recording. They are dependent rows.
    meta.loc[meta.source_id.eq('synthetic-4-2'),'source_id']='synthetic-2-2'
    cfg=ContextConfig(pretrain_updates=0,predictor_updates=1,dim=16,depth=2,heads=2)
    bundle=ContextBundle(SJEPA(cfg.model_config()),cfg,provenance={'run_id':'unit-test'}).eval()
    save_context_bundle(bundle,tmp_path/'trained.pt')
    predictions,metrics=condition_capstone(bundle,x,meta,tmp_path)
    for method,rows in predictions.groupby('method'):
        np.testing.assert_allclose(rows.groupby('source_id').source_weight.sum(),1.)
        assert metrics[method]['n_test_sources']==4
        assert metrics[method]['n_test_source_label_pairs']==5
        assert metrics[method]['n_mixed_label_sources']==1
        assert np.isclose(np.asarray(metrics[method]['confusion_matrix']).sum(),4.)
    a=predictions[predictions.method.eq('random_forest')]
    b=predictions[predictions.method.eq('normal_jepa_probe')]
    pd.testing.assert_frame_equal(a[['source_id','label','source_weight']].reset_index(drop=True),
                                  b[['source_id','label','source_weight']].reset_index(drop=True))
    saved=joblib.load(tmp_path/'condition_classifiers.joblib')
    assert saved['provenance']['encoder_checkpoint_sha256']
    assert saved['provenance']['encoder_provenance']['run_id']=='unit-test'
    assert saved['embedding_scaler'].n_features_in_==16
    assert len(saved['embedding_probe'].classes_)==5
    features,_=build_features(x[:1])
    assert saved['random_forest'].predict_proba(features).shape==(1,5)
    report=json.loads((tmp_path/'condition_metrics.json').read_text())
    assert not set(report['fit_source_ids']) & set(report['test_source_ids'])
    assert report['paired_methods_share_identical_source_label_support']


def test_condition_sources_cannot_cross_fit_and_test(tmp_path):
    x,meta=synthetic_corpus(sources_per_class=3);meta['outer_fold']=0
    meta.loc[meta.index[-1],'source_id']=meta.source_id.iloc[0]
    with pytest.raises(ValueError,match='multiple split roles'):
        condition_capstone(None,x,meta,tmp_path)
