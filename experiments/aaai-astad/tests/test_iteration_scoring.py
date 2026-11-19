import numpy as np
import pandas as pd
from aaai_gait.iteration_scoring import (local_movement_features, aggregate_descriptors,
    compress_target_rows, aggregate_target_profiles, BilateralReference, NormalDistance,
    ranking_metrics, paired_bootstrap)


def test_mask_repetition_does_not_reweight_targets():
    # First target is requested three times; second only once. Each target
    # still has equal weight after the mask-within-target aggregation.
    rows = pd.DataFrame(dict(window_index=[0]*4, joint=[25]*4,
        patch_index=[0,0,0,1], valid=[True]*4, loss_base=[1,1,1,9],
        loss_own=[1,1,1,9], loss_both=[0,0,0,0]))
    compact = compress_target_rows(rows)
    assert compact.benefit.mean() == 5
    assert compact.mask_draws.tolist() == [3,1]


def test_source_descriptors_aggregate_before_distance():
    meta = pd.DataFrame({'source_id':['a','a','a','b'], 'sequence_id':['s1','s1','s2','s3']})
    out = aggregate_descriptors(np.array([[0.],[2.],[9.],[4.]]),['f'],meta)
    assert out.loc['a','f'] == 5
    model = NormalDistance.fit(pd.DataFrame({'f':[0.,2.,1000.]}, index=['a','b','test']), ['a','b'])
    assert model.medians.tolist() == [1.]
    assert model.fit_sources == ('a','b')


def test_reference_requires_exact_reserved_sources():
    values = pd.DataFrame({'left':[0.,1.,2.], 'right':[0.,2.,4.]},index=['a','b','c'])
    ref = BilateralReference.fit(values,['a','b'])
    assert ref.sources == ('a','b')
    assert np.allclose(ref.variance,[.875+1e-8,1.625+1e-8])
    try:
        BilateralReference.fit(values,['a','missing'])
    except ValueError:
        pass
    else:
        raise AssertionError('A missing reserved reference must not be replaced')


def test_metrics_compare_within_model_and_bootstrap_pairs():
    t = pd.DataFrame({'source_id':list('abcdefgh'), 'outer_fold':[0]*4+[1]*4,
                      'binary_label':[0,0,1,1]*2, 'a':[0,1,2,3,100,101,102,103],
                      'b':[3,2,1,0,103,102,101,100]})
    assert ranking_metrics(t,'a')['weighted_auroc'] == 1
    assert ranking_metrics(t,'a')['pooled_secondary_auroc'] < 1
    interval = paired_bootstrap(t,'a','b',repeats=20)
    assert interval['observed'] == 1
    assert interval['conditional_95_interval'] == [1.,1.]
