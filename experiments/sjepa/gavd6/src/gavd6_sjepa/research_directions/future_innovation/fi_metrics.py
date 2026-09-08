"""Predictive featurewise R² and paired source bootstraps of saved OOF predictions."""

import numpy as np


def featurewise_predictive_r2(
    y_true, y_pred, y_reference, sample_weight, training_variance_valid
):
    if (
        y_true.ndim != 2
        or y_pred.shape != y_true.shape
        or y_reference.shape != y_true.shape
    ):
        raise ValueError("Metric arrays must have identical [windows,features] shapes")
    if not all(np.isfinite(a).all() for a in (y_true, y_pred, y_reference)):
        raise ValueError("Non-finite targets or predictions")
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    if sample_weight.ndim != 1:
        raise ValueError("Source weights must be one-dimensional")
    weights = sample_weight[:, None]
    if (
        len(weights) != len(y_true)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("Source weights must be finite and positive")
    training_variance_valid = np.asarray(training_variance_valid, dtype=bool)
    if training_variance_valid.shape != (y_true.shape[1],):
        raise ValueError("Invalid target variance mask shape")
    error = np.sum(weights * (y_true - y_pred) ** 2, axis=0)
    reference_error = np.sum(weights * (y_true - y_reference) ** 2, axis=0)
    valid = training_variance_valid & (reference_error > 1e-12)
    if not valid.any():
        raise ValueError("No valid target features")
    result = np.full(y_true.shape[1], np.nan)
    result[valid] = 1 - error[valid] / reference_error[valid]
    return result, valid


def score_arrays(y_true, base, full, weights, valid):
    reference = np.zeros_like(y_true)
    rbase, mask = featurewise_predictive_r2(y_true, base, reference, weights, valid)
    rfull, mask_full = featurewise_predictive_r2(
        y_true, full, reference, weights, valid
    )
    if not np.array_equal(mask, mask_full):
        raise ValueError("Arms used different feature masks")
    rb, rf = float(rbase[mask].mean()), float(rfull[mask].mean())
    delta = rf - rb
    gains = rfull - rbase
    if not np.isclose(delta, np.mean(gains[mask]), rtol=1e-9, atol=1e-12):
        raise ValueError("Aggregate and featurewise gains disagree")
    return (
        {
            "r2_baseline": rb,
            "r2_full": rf,
            "delta_r2": delta,
            "f8": delta / (1 - rb) if 1 - rb > 1e-8 else None,
            "valid_feature_count": int(mask.sum()),
            "delta_feature_mean": float(gains[mask].mean()),
            "delta_feature_median": float(np.median(gains[mask])),
            "delta_feature_q25": float(np.quantile(gains[mask], 0.25)),
            "delta_feature_q75": float(np.quantile(gains[mask], 0.75)),
            "positive_feature_fraction": float(np.mean(gains[mask] > 0)),
            "baseline_ceiling_warning": rb >= 0.95,
        },
        rbase,
        rfull,
        mask,
    )


def source_bootstrap_indices(video_ids, repetitions=2000, seed=260905):
    values = np.asarray(video_ids, dtype=str)
    unique = np.array(sorted(set(values)))
    if not len(unique) or repetitions < 1:
        raise ValueError("Bootstrap needs sources and positive repetitions")
    rng = np.random.default_rng(seed)
    groups = {source: np.flatnonzero(values == source) for source in unique}
    for _ in range(repetitions):
        yield np.concatenate(
            [groups[source] for source in rng.choice(unique, len(unique), replace=True)]
        )
