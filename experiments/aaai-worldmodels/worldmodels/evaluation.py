"""Matched input comparisons with source-level endpoints and uncertainty."""
from __future__ import annotations
import numpy as np
from .data import LABELS


def equal_source_weights(source_ids, labels=None):
    groups = np.asarray(source_ids)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0/counts[inverse]
    if labels is not None:
        labels = np.asarray(labels)
        if len(labels) != len(groups):
            raise ValueError('Labels and sources must align')
        for group in np.unique(groups):
            if len(np.unique(labels[groups==group])) != 1:
                raise ValueError('Mixed labels within source')
        # Equal total weight per class, equal total weight per source within class.
        total = weights.sum()
        classes = np.unique(labels)
        for label in classes:
            selected = labels == label
            weights[selected] *= total/(len(classes)*weights[selected].sum())
    return weights/weights.mean()


def fit_rf(x_train, y_train, source_ids=None, seed=27, **kwargs):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    params = dict(n_estimators=200,max_depth=8,min_samples_leaf=2,random_state=seed,n_jobs=2)
    params.update(kwargs)
    estimator = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),RandomForestClassifier(**params))
    sample_weight = equal_source_weights(source_ids,y_train) if source_ids is not None else None
    estimator.fit(x_train,y_train,randomforestclassifier__sample_weight=sample_weight)
    return estimator


def fit_probe(x_train, y_train, source_ids=None, seed=27, C=1.0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    estimator = make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000,random_state=seed))
    weights = equal_source_weights(source_ids,y_train) if source_ids is not None else None
    estimator.fit(x_train,y_train,logisticregression__sample_weight=weights)
    return estimator


def align_probabilities(estimator, x, classes=LABELS):
    """Preserve a declared probability-column order and fail on absent classes."""
    if set(estimator.classes_) != set(classes):
        raise ValueError('All declared classes must occur in training; absent classes are not zeros')
    p = estimator.predict_proba(x)
    return p[:,[list(estimator.classes_).index(c) for c in classes]]


def aggregate_sources(probabilities, labels, source_ids, sequence_ids=None):
    """Mean windows within sequence, then mean sequences within source.

    Returns sorted source IDs, their labels, and aggregated predictions. Supplying
    sequence IDs prevents long sequences from receiving more source weight.
    """
    p, y, groups = np.asarray(probabilities,float), np.asarray(labels), np.asarray(source_ids)
    if len(p)!=len(y) or len(p)!=len(groups) or not len(p) or not np.isfinite(p).all():
        raise ValueError('Nonempty aligned finite inputs required')
    seq = np.asarray(sequence_ids) if sequence_ids is not None else np.arange(len(p))
    if len(seq)!=len(p):
        raise ValueError('Sequence IDs must align')
    sources, truth, predictions = [], [], []
    for source in sorted(set(groups)):
        mask = groups==source
        unique = np.unique(y[mask])
        if len(unique)!=1:
            raise ValueError('Mixed labels in source')
        sequence_means = [p[mask & (seq==sequence)].mean(0) for sequence in np.unique(seq[mask])]
        sources.append(source); truth.append(unique[0]); predictions.append(np.mean(sequence_means,axis=0))
    return np.asarray(sources), np.asarray(truth), np.asarray(predictions)


def classification_report(labels, probabilities, classes=LABELS):
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score
    y, p = np.asarray(labels), np.asarray(probabilities,float)
    if p.shape!=(len(y),len(classes)) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Expected finite class probabilities with rows summing to one')
    predicted = np.asarray(classes)[p.argmax(1)]
    abnormal = (y!='normal').astype(int)
    abnormal_probability = 1-p[:,list(classes).index('normal')]
    report = dict(n_sources=len(y),accuracy=float(accuracy_score(y,predicted)),
                  balanced_accuracy=float(balanced_accuracy_score(y,predicted)),
                  macro_f1=float(f1_score(y,predicted,labels=list(classes),average='macro',zero_division=0)),
                  classes=list(classes), confusion_matrix=confusion_matrix(y,predicted,labels=list(classes)).tolist())
    binary_predicted = (abnormal_probability >= 0.5).astype(int)
    report['abnormal_prevalence'] = float(abnormal.mean())
    report['binary_threshold'] = 0.5
    report['binary_balanced_accuracy'] = float(balanced_accuracy_score(abnormal,binary_predicted)) if len(np.unique(abnormal))==2 else None
    report['sensitivity'] = float(binary_predicted[abnormal==1].mean()) if (abnormal==1).any() else None
    report['specificity'] = float((binary_predicted[abnormal==0]==0).mean()) if (abnormal==0).any() else None
    report['per_class_recall'] = {label: float((predicted[y==label]==label).mean()) if (y==label).any() else None for label in classes}
    report['abnormal_auroc'] = float(roc_auc_score(abnormal,abnormal_probability)) if len(np.unique(abnormal))==2 else None
    report['abnormal_average_precision'] = float(average_precision_score(abnormal,abnormal_probability)) if len(np.unique(abnormal))==2 else None
    return report


def paired_source_bootstrap(labels, probabilities_a, probabilities_b, classes=LABELS, repetitions=1000, seed=27, metric="binary_balanced_accuracy"):
    """Paired class-stratified source bootstrap of metric(A)-metric(B).

    Inputs must already contain one row per source in identical order. Intervals
    are descriptive conditional on this test cohort, not new-patient guarantees.
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score
    y, a, b = np.asarray(labels), np.asarray(probabilities_a), np.asarray(probabilities_b)
    if a.shape!=b.shape or a.shape!=(len(y),len(classes)) or repetitions<10:
        raise ValueError('Aligned source-level probabilities and >=10 repetitions required')
    if set(y)!=set(classes):
        raise ValueError('Every class needs test sources for a five-class interval')
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(y==label) for label in classes]
    pred_a, pred_b = np.asarray(classes)[a.argmax(1)], np.asarray(classes)[b.argmax(1)]
    if metric not in {"macro_f1", "binary_balanced_accuracy"}:
        raise ValueError("Unknown comparison metric")
    def difference(index):
        if metric == "binary_balanced_accuracy":
            truth = (y[index] != "normal").astype(int)
            column = list(classes).index("normal")
            return balanced_accuracy_score(truth, (1-a[index,column]>=0.5).astype(int))-balanced_accuracy_score(truth, (1-b[index,column]>=0.5).astype(int))
        return f1_score(y[index],pred_a[index],labels=list(classes),average='macro',zero_division=0)-f1_score(y[index],pred_b[index],labels=list(classes),average='macro',zero_division=0)
    values = [difference(np.concatenate([rng.choice(s,len(s),replace=True) for s in strata])) for _ in range(repetitions)]
    return dict(metric=metric+'_A_minus_B',estimate=float(difference(np.arange(len(y)))),
                ci95=np.quantile(values,[0.025,0.975]).tolist(),repetitions=repetitions,seed=seed,
                unit='source',resampling='paired within label',n_sources=len(y))


def validation_threshold(normal_validation_scores, false_positive_rate=0.05):
    """Calibrate on held-out normal SOURCE scores only; never inspect test scores."""
    scores = np.asarray(normal_validation_scores,float)
    if scores.ndim!=1 or len(scores)<2 or not np.isfinite(scores).all() or not 0<false_positive_rate<1:
        raise ValueError('Need >=2 finite normal source scores and a valid target rate')
    # Conservative finite-sample order statistic. With six calibration sources,
    # no finite threshold can guarantee a 5% conformal false-positive rate.
    order = int(np.ceil((len(scores)+1)*(1-false_positive_rate)))
    return float(np.sort(scores)[order-1]) if order<=len(scores) else float('inf')


def validation_threshold_report(normal_validation_scores, false_positive_rate=0.05):
    """JSON-safe calibration record; null means no finite calibrated threshold."""
    threshold = validation_threshold(normal_validation_scores, false_positive_rate)
    return dict(threshold=threshold if np.isfinite(threshold) else None,
                status='finite' if np.isfinite(threshold) else 'insufficient_normal_calibration_sources',
                n_sources=len(normal_validation_scores), target_false_positive_rate=false_positive_rate,
                decision_rule='score > threshold; an infinite threshold flags no sources')
