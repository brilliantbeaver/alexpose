import pandas as pd
from aaai_gait.robustness import anomaly_bootstrap

def test_anomaly_bootstrap_ignores_validation_and_counts_invalid_draws():
    rows=pd.DataFrame([
        dict(source_id='a',label='normal',split='test',score=0.),
        dict(source_id='b',label='stroke',split='test',score=1.),
        dict(source_id='c',label='normal',split='val',score=99.),
    ])
    report=anomaly_bootstrap(rows,repeats=300,seed=7)
    assert report['n_test_source_clusters']==2
    assert 0<report['valid_repeats']<300
    assert report['auroc_ci95']==[1.,1.]
    assert report['average_precision_ci95']==[1.,1.]
    assert anomaly_bootstrap(rows,repeats=10,seed=7)['auroc_ci95'] is None

def test_mixed_label_source_stays_in_one_bootstrap_cluster():
    rows=pd.DataFrame([
        dict(source_id='same',label='normal',split='test',score=.2),
        dict(source_id='same',label='stroke',split='test',score=.8),
        dict(source_id='other',label='normal',split='test',score=.1),
        dict(source_id='other',label='stroke',split='test',score=.9),
    ])
    report=anomaly_bootstrap(rows,repeats=30)
    assert report['valid_repeats']==30
    assert report['n_test_source_clusters']==2
