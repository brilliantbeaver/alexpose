"""Simple normal-space baseline proposed before opening model test scores."""
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from .training import embed_windows
from .evaluation import calibrate_threshold,anomaly_metrics
from .robustness import anomaly_bootstrap

def normal_distance(anchor,x,meta,batch_size=16):
    z=embed_windows(anchor,x,batch_size=batch_size)
    cols=[f'z{i}' for i in range(z.shape[1])]
    table=meta[['sequence_id','source_id','label','split']].reset_index(drop=True).copy()
    table[cols]=z
    # Equal sequence contribution within each source, then one normal fit row per source.
    sequence=table.groupby(['sequence_id','source_id','label','split'],as_index=False)[cols].mean()
    source=sequence.groupby(['source_id','label','split'],as_index=False)[cols].mean()
    train=source.split.eq('train')&source.label.eq('normal')
    if train.sum()<2:raise ValueError('Need at least two normal training sources')
    covariance=LedoitWolf().fit(source.loc[train,cols].to_numpy())
    source['score']=covariance.mahalanobis(source[cols].to_numpy())
    val=source[source.split.eq('val')&source.label.eq('normal')]
    threshold=calibrate_threshold(val.score.to_numpy())
    test=source[source.split.eq('test')]
    metrics=anomaly_metrics(test.label.to_numpy()!='normal',test.score.to_numpy(),threshold)
    metrics.update(n_normal_training_sources=int(train.sum()),n_normal_calibration_sources=len(val),finite_threshold=bool(np.isfinite(threshold)),source_bootstrap=anomaly_bootstrap(source))
    return source[['source_id','label','split','score']],metrics
