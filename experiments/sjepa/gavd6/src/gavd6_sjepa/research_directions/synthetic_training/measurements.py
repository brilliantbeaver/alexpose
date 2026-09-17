"""Reference accuracy and label-free target response measurements.

Target prediction summaries never receive target coordinates. Diagnostic errors
are a separate, explicitly labeled source-data input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def normalized_coordinates(predictions, boxes):
    """Express full-image predictions relative to fixed crop width and height."""
    x, boxes = np.asarray(predictions, float), np.asarray(boxes, float)
    if x.shape != (len(boxes), 12, 2) or boxes.shape[1:] != (4,):
        raise ValueError("Expected predictions [N,12,2] and xyxy boxes [N,4].")
    size = boxes[:, 2:] - boxes[:, :2]
    if not np.isfinite(boxes).all() or np.any(size <= 0):
        raise ValueError("Fixed person boxes must be finite and have positive area.")
    return (x - boxes[:, None, :2]) / size[:, None, :]


def prediction_summary(predictions, boxes):
    """Per-joint location, variation, and missingness; no target labels required."""
    x = normalized_coordinates(predictions, boxes)
    valid = np.isfinite(x).all(axis=-1)
    safe = np.where(valid[..., None], x, 0)
    count = valid.sum(axis=0).clip(1)
    mean = safe.sum(axis=0) / count[:, None]
    variance = np.where(valid[..., None], (safe - mean) ** 2, 0).sum(axis=0) / count[:, None]
    return np.concatenate([mean.ravel(), np.sqrt(variance).ravel(), valid.mean(axis=0)]).astype(np.float32)


def response_summary(before, after, boxes):
    """Signed and unsigned per-joint changes, plus a separate magnitude control."""
    delta = normalized_coordinates(after, boxes) - normalized_coordinates(before, boxes)
    valid = np.isfinite(delta).all(axis=-1)
    safe = np.where(valid[..., None], delta, 0)
    count = valid.sum(axis=0).clip(1)
    signed = safe.sum(axis=0) / count[:, None]
    absolute = np.abs(safe).sum(axis=0) / count[:, None]
    rms = np.sqrt((safe ** 2).sum(axis=0) / count[:, None])
    summary = np.concatenate([signed.ravel(), absolute.ravel(), rms.ravel(), valid.mean(axis=0)])
    magnitude = np.linalg.norm(safe, axis=-1).sum() / max(valid.sum(), 1)
    return summary.astype(np.float32), np.asarray([magnitude], np.float32)


def landmark_errors(predictions, keypoints, visible, boxes, missing_penalty=1.0):
    """Visible-joint Euclidean errors with a model-independent person scale.

    A nonfinite prediction receives a fixed normalized penalty; finite reference
    labels are mandatory wherever visibility is true. Hidden joints contribute
    no weight. A frame with no visible references is not an accuracy observation.
    """
    pred, target = np.asarray(predictions, float), np.asarray(keypoints, float)
    visible, boxes = np.asarray(visible, bool), np.asarray(boxes, float)
    if pred.shape != target.shape or pred.shape[1:] != (12, 2) or visible.shape != pred.shape[:2]:
        raise ValueError("Landmark arrays must be [N,12,2] with visibility [N,12].")
    if boxes.shape != (len(pred), 4) or not np.isfinite(boxes).all():
        raise ValueError("Reference boxes must be finite xyxy [N,4].")
    sides = boxes[:, 2:] - boxes[:, :2]
    if np.any(sides <= 0) or np.any(visible & ~np.isfinite(target).all(axis=-1)):
        raise ValueError("Visible reference coordinates and positive boxes are required.")
    scale = np.linalg.norm(sides, axis=-1)
    failed = ~np.isfinite(pred).all(axis=-1)
    distance = np.linalg.norm(pred - target, axis=-1)
    distance = np.where(failed, missing_penalty * scale[:, None], distance)
    pixels = np.where(visible, distance, np.nan)
    return pixels / scale[:, None], pixels, failed & visible


def frame_scores(predictions, keypoints, visible, boxes, index, *, missing_penalty=1.0):
    """Keep exactly the reference visibility mask for every estimator."""
    error, pixels, failed = landmark_errors(predictions, keypoints, visible, boxes, missing_penalty)
    count = np.asarray(visible, bool).sum(axis=1)
    result = index.reset_index(drop=True).copy()
    if len(result) != len(error):
        raise ValueError("Predictions and frame index have different lengths.")
    result["visible_joints"] = count
    result["nle"] = np.divide(np.nansum(error, axis=1), count, out=np.full(len(count), np.nan), where=count > 0)
    result["pixel_error"] = np.divide(np.nansum(pixels, axis=1), count, out=np.full(len(count), np.nan), where=count > 0)
    result["failed_joints"] = failed.sum(axis=1)
    return result


def reference_error(predictions, keypoints, visible, boxes, index, *, group="person_id", missing_penalty=1.0):
    """Equal frame weight within a group, then equal person/recording weight."""
    scores = frame_scores(predictions, keypoints, visible, boxes, index, missing_penalty=missing_penalty)
    means = scores.groupby(group, dropna=False).nle.mean()
    if means.empty or means.isna().any():
        raise ValueError("Every reference group needs at least one visible annotated landmark.")
    return float(means.mean())


def diagnostic_summary(predictions, keypoints, visible, boxes, lesson_ids, lesson_order, *, missing_penalty=1.0):
    """Known source weakness for every lesson and joint, with coverage indicators."""
    error, _, _ = landmark_errors(predictions, keypoints, visible, boxes, missing_penalty)
    result = []
    ids = np.asarray(lesson_ids)
    for lesson in lesson_order:
        selected = error[ids == lesson]
        if len(selected) == 0:
            raise ValueError(f"No labeled diagnostic frames for lesson {lesson!r}.")
        finite = np.isfinite(selected)
        means = np.nansum(selected, axis=0) / finite.sum(axis=0).clip(1)
        result.extend(means)
        result.extend(finite.mean(axis=0))
    return np.asarray(result, np.float32)


def paired_recording_interval(scores, method, baseline, *, samples=2000, seed=17):
    """Bootstrap recording means, preserving every method pair and collection.

    Positive differences mean that the method improves on the baseline. The
    interval is conditional on these students and the selected lessons, not a
    confidence interval over unseen architectures or deployment collections.
    """
    grouped = scores.groupby(["recording_id", "domain_id", "method"], dropna=False).nle.mean().unstack("method")
    if method not in grouped or baseline not in grouped:
        raise ValueError("Both methods need scores on the same recordings.")
    pair = grouped[[method, baseline]]
    if pair.isna().any().any():
        raise ValueError("Missing recording pairs cannot be dropped silently.")
    delta = (pair[baseline] - pair[method]).to_numpy()
    rng = np.random.default_rng(seed)
    # Resample recordings within each preselected collection, not individual frames.
    domains = pair.index.get_level_values("domain_id").to_numpy()
    positions = [np.flatnonzero(domains == domain) for domain in pd.unique(domains)]
    estimates = np.zeros(samples)
    for pos in positions:
        estimates += delta[rng.choice(pos, (samples, len(pos)), replace=True)].sum(axis=1) / len(delta)
    low, high = np.quantile(estimates, [0.025, 0.975])
    return dict(method=method, baseline=baseline, improvement=float(delta.mean()),
                ci_low=float(low), ci_high=float(high), n_recordings=len(delta), n_domains=len(positions))
