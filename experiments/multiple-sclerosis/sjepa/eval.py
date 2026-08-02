"""Evaluation helpers shared by the capstone notebook.

Everything here works on whole-sequence records, not windows, so the Random
Forest branch and the S-JEPA branch are scored the exact same way on the exact
same held-out videos. Metrics are macro-averaged so a rare class counts as much
as a common one, matching the exp5 methodology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Metrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusion: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "confusion": self.confusion,
            "labels": self.labels,
        }


def evaluate(y_true: Sequence, y_pred: Sequence, labels: Sequence[str]) -> Metrics:
    """Compute accuracy, macro P/R/F1, per-class scores, and a confusion matrix."""
    from sklearn.metrics import (
        accuracy_score, precision_recall_fscore_support, confusion_matrix,
    )

    y_true = list(y_true)
    y_pred = list(y_pred)
    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average=None, zero_division=0
    )
    mp, mr, mf1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(labels), average="macro", zero_division=0
    )
    per_class = {
        lab: {"precision": float(p[i]), "recall": float(r[i]),
              "f1": float(f1[i]), "support": int(support[i])}
        for i, lab in enumerate(labels)
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(labels)).tolist()
    return Metrics(acc, float(mp), float(mr), float(mf1), per_class, cm, list(labels))


def aggregate_folds(metrics_list: Sequence[Metrics]) -> Dict[str, Dict[str, float]]:
    """Mean and std across folds for the headline scalar metrics."""
    keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        vals = np.array([getattr(m, k) for m in metrics_list], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


def silhouette(embeddings: np.ndarray, labels: Sequence) -> float:
    """Silhouette score: how well separated the class clusters are (higher better)."""
    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2 or len(labels) <= 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels))
