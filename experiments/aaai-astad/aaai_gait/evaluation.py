"""Sequence-level evaluation and recording-cluster uncertainty.

Average windows within an annotated sequence. Never average across differently
labeled sequences just because they share a YouTube recording.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import LABELS


def _checked(y, probabilities, classes):
    y, p, classes = np.asarray(y), np.asarray(probabilities, float), np.asarray(classes)
    if p.shape != (len(y), len(classes)) or len(set(classes)) != len(classes):
        raise ValueError("Probability columns must correspond exactly to unique classes")
    if not set(y).issubset(set(classes)):
        raise ValueError("A true label has no probability column")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Expected finite nonnegative row-normalized probabilities")
    return y,p,classes


def align_probabilities(probabilities, observed_classes, classes=LABELS):
    """Reorder explicit class columns and reject missing training classes."""
    if set(observed_classes) != set(classes):
        raise ValueError("Training data do not cover every evaluation class")
    return np.asarray(probabilities)[:, [list(observed_classes).index(c) for c in classes]]


def aggregate_sequence_predictions(metadata: pd.DataFrame, probabilities, classes=LABELS):
    """Return (one-row-per-sequence metadata, mean probabilities)."""
    _, p, classes = _checked(metadata.label.to_numpy(), probabilities, classes)
    table = metadata.reset_index(drop=True).copy()
    for key in ("label","source_id","split"):
        if table.groupby("sequence_id")[key].nunique().max() > 1:
            raise ValueError(f"Inconsistent {key} within a sequence")
    probability_names = [f"p_{i}" for i in range(len(classes))]
    table[probability_names] = p
    records = table.groupby("sequence_id", sort=True).agg(label=("label","first"), source_id=("source_id","first"), split=("split","first"), n_windows=("sequence_id","size"))
    means = table.groupby("sequence_id", sort=True)[probability_names].mean()
    return records.reset_index(), means.to_numpy()


def classification_metrics(y, probabilities, classes=LABELS, sample_weight=None) -> dict:
    """Multiclass metrics plus abnormal score = 1 - P(normal)."""
    from sklearn.metrics import (balanced_accuracy_score, f1_score, classification_report,
                                 confusion_matrix, roc_auc_score, average_precision_score)
    y,p,classes = _checked(y,probabilities,classes)
    predicted = classes[p.argmax(axis=1)]
    result = dict(n_sequences=len(y), macro_f1=float(f1_score(y,predicted,labels=classes,average="macro",zero_division=0,sample_weight=sample_weight)),
                  balanced_accuracy=float(balanced_accuracy_score(y,predicted,sample_weight=sample_weight)),
                  per_class=classification_report(y,predicted,labels=classes,output_dict=True,zero_division=0,sample_weight=sample_weight),
                  confusion_matrix=confusion_matrix(y,predicted,labels=classes,sample_weight=sample_weight).tolist(),
                  class_order=classes.tolist())
    if "normal" in classes:
        abnormal = (y != "normal").astype(int)
        score = 1 - p[:,list(classes).index("normal")]
        result["binary_balanced_accuracy"] = float(balanced_accuracy_score(abnormal, score >= 0.5, sample_weight=sample_weight))
        result["abnormal_auroc"] = float(roc_auc_score(abnormal,score,sample_weight=sample_weight)) if len(set(abnormal))==2 else None
        result["abnormal_average_precision"] = float(average_precision_score(abnormal,score,sample_weight=sample_weight)) if len(set(abnormal))==2 else None
    return result


def cluster_bootstrap(y, probabilities, groups, classes=LABELS, *, repeats: int = 1000,
                      seed: int = 42, other_probabilities=None) -> dict:
    """Percentile CIs by resampling complete source groups with replacement.

    Paired comparison resamples identical groups for both approaches and reports
    first-minus-other differences. Replicates missing any prespecified class are
    omitted and their number is reported. A wide interval is evidence of limited
    source diversity. This quantifies evaluation sampling, not training-seed
    uncertainty or dependence across unidentified repeated participants.
    """
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    y,p,classes = _checked(y,probabilities,classes)
    groups = np.asarray(groups)
    if len(groups) != len(y) or repeats < 1:
        raise ValueError("Invalid group count or repeat count")
    q = None if other_probabilities is None else _checked(y,other_probabilities,classes)[1]
    group_names = np.unique(groups)
    if len(group_names) < 2:
        raise ValueError("Cluster uncertainty needs at least two source recordings")
    lookup = {g: np.flatnonzero(groups == g) for g in group_names}
    rng = np.random.default_rng(seed)
    scores = []
    def summarize(index, proba):
        true = y[index]
        pred = classes[proba[index].argmax(axis=1)]
        out = [f1_score(true,pred,labels=classes,average="macro",zero_division=0),balanced_accuracy_score(true,pred)]
        if "normal" in classes:
            out.append(roc_auc_score(true != "normal",1-proba[index,list(classes).index("normal")]))
        return np.asarray(out)
    for _ in range(repeats):
        index = np.concatenate([lookup[g] for g in rng.choice(group_names,len(group_names),replace=True)])
        if set(y[index]) != set(classes):
            continue
        value = summarize(index,p)
        if q is not None:
            value -= summarize(index,q)
        scores.append(value)
    names = ["macro_f1", "balanced_accuracy"] + (["abnormal_auroc"] if "normal" in classes else [])
    intervals = {name: None for name in names}
    if len(scores) >= 20:
        quantiles = np.quantile(scores,[0.025,0.975],axis=0)
        intervals = {name: quantiles[:,i].tolist() for i,name in enumerate(names)}
    return dict(intervals_95=intervals, valid_repeats=len(scores), requested_repeats=repeats,
                n_source_groups=len(group_names), paired_difference=q is not None,
                minimum_20_valid_repeats_met=len(scores)>=20)


def calibrate_threshold(normal_val_scores, alpha: float = 0.05) -> float:
    """Upper finite-sample normal calibration quantile; never fit on test scores.

    Input should be one score per independent normal source recording. With
    fewer than ceil(1/alpha)-1 observations, the requested finite-sample quantile
    is infinity. This correctly exposes that 5% FPR calibration needs more data
    than the small default normal validation split provides. Exchangeability
    remains an assumption and is not established by GAVD's source IDs.
    """
    values = np.asarray(normal_val_scores,float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 < alpha < 1:
        raise ValueError("Need finite normal validation scores and 0<alpha<1")
    rank = int(np.ceil((len(values)+1)*(1-alpha)))
    return float(np.sort(values)[rank-1]) if rank <= len(values) else float("inf")


def anomaly_metrics(y_is_abnormal, scores, threshold: float) -> dict:
    """Thresholded sensitivity/FPR and ranking metrics for latent surprise."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y, scores = np.asarray(y_is_abnormal,bool), np.asarray(scores,float)
    if y.shape != scores.shape or not np.isfinite(scores).all() or len(np.unique(y)) != 2:
        raise ValueError("Need finite scores and both normal and abnormal samples")
    flagged = scores > threshold
    return dict(auroc=float(roc_auc_score(y,scores)), average_precision=float(average_precision_score(y,scores)),
                sensitivity=float(flagged[y].mean()), false_positive_rate=float(flagged[~y].mean()),
                threshold=float(threshold), n_normal=int((~y).sum()), n_abnormal=int(y.sum()))
