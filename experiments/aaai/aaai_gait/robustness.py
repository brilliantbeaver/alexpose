"""Source-cluster uncertainty for the primary normal-only anomaly task."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def anomaly_bootstrap(source_scores, *, repeats=1000, seed=42):
    """Bootstrap test source clusters, preserving mixed-label source rows.

    Rank metrics only. This interval excludes model-training and unknown
    participant-dependence uncertainty. Missing-class resamples are counted.
    """
    table=source_scores[source_scores.split.eq('test')].reset_index(drop=True)
    groups=table.source_id.unique()
    if len(groups)<2 or repeats<1:
        raise ValueError('Bootstrap requires at least two source clusters and positive repeats')
    rng=np.random.default_rng(seed)
    draws=[]
    for _ in range(repeats):
        index=np.concatenate([np.flatnonzero(table.source_id.eq(g)) for g in rng.choice(groups,len(groups),replace=True)])
        rows=table.iloc[index];y=rows.label.to_numpy()!='normal'
        if len(np.unique(y))<2:continue
        draws.append((roc_auc_score(y,rows.score),average_precision_score(y,rows.score)))
    interval=np.quantile(draws,[.025,.975],axis=0).T.tolist() if len(draws)>=20 else [None,None]
    return dict(auroc_ci95=interval[0],average_precision_ci95=interval[1],valid_repeats=len(draws),
                requested_repeats=repeats,n_test_source_clusters=len(groups),unit='source-label ranking with source-cluster resampling')
