from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def source_weights(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups).astype(str)
    unique, counts = np.unique(values, return_counts=True)
    lookup = dict(zip(unique, counts))
    return np.asarray([1.0 / lookup[group] for group in values], dtype=np.float64)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(np.asarray(values, dtype=np.float64), weights=weights))


def weighted_r2(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    truth = np.asarray(y_true, dtype=np.float64)
    prediction = np.asarray(y_pred, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    mean = weighted_mean(truth, weight)
    denominator = float(np.sum(weight * np.square(truth - mean)))
    numerator = float(np.sum(weight * np.square(truth - prediction)))
    reference = max(float(np.sum(weight * np.square(truth))), 1.0)
    return (
        float(1.0 - numerator / denominator)
        if denominator > 1e-15 * reference
        else float("nan")
    )


def weighted_mae(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    return weighted_mean(np.abs(np.asarray(y_true) - np.asarray(y_pred)), weights)


def source_medians(
    y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray
) -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "video_id": np.asarray(groups).astype(str),
                "target": np.asarray(y_true, dtype=np.float64),
                "prediction": np.asarray(y_pred, dtype=np.float64),
            }
        )
        .groupby("video_id", as_index=False)
        .agg(target=("target", "median"), prediction=("prediction", "median"))
    )


def metric_bundle(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    *,
    neutral_threshold: float | np.ndarray,
) -> dict[str, Any]:
    weights = source_weights(groups)
    medians = source_medians(y_true, y_pred, groups)
    source_r2 = weighted_r2(
        medians["target"].to_numpy(),
        medians["prediction"].to_numpy(),
        np.ones(len(medians)),
    )
    threshold_values = np.asarray(neutral_threshold, dtype=np.float64)
    if threshold_values.ndim == 0:
        source_threshold = np.full(len(medians), float(threshold_values))
    else:
        if threshold_values.shape != np.asarray(y_true).shape:
            raise ValueError("Per-row neutral_threshold shape must match y_true")
        source_threshold = (
            pd.DataFrame(
                {
                    "video_id": np.asarray(groups).astype(str),
                    "threshold": threshold_values,
                }
            )
            .groupby("video_id")["threshold"]
            .median()
            .reindex(medians["video_id"])
            .to_numpy()
        )
    eligible = np.abs(medians["target"].to_numpy()) > source_threshold
    sign_accuracy = (
        float(
            np.mean(
                np.sign(medians.loc[eligible, "target"])
                == np.sign(medians.loc[eligible, "prediction"])
            )
        )
        if eligible.any()
        else float("nan")
    )
    return {
        "source_balanced_r2": weighted_r2(y_true, y_pred, weights),
        "source_balanced_mae": weighted_mae(y_true, y_pred, weights),
        "source_median_r2": source_r2,
        "source_level_sign_accuracy": sign_accuracy,
        "sign_coverage": float(eligible.mean()),
        "sequence_count": int(len(y_true)),
        "source_count": int(len(medians)),
    }


def normalized_antisymmetry_error(
    prediction: np.ndarray,
    mirror_prediction: np.ndarray,
    training_target_scale: float,
) -> float:
    numerator = float(
        np.sqrt(
            np.mean(
                np.square(
                    np.asarray(prediction, dtype=np.float64)
                    + np.asarray(mirror_prediction, dtype=np.float64)
                )
            )
        )
    )
    denominator = 2.0 * float(training_target_scale)
    return numerator / denominator if denominator > 0 else float("nan")
