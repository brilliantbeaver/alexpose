"""Observable forecast arithmetic with equal-video and clustered estimands.

No target is invented and no missing prediction is silently removed from a
method comparison. Bootstrap draws preserve all videos of connected groups.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np


PRIMARY_PAIRS = ((25, 26), (27, 28), (29, 30), (31, 32))


def primary_window_errors(prediction, endpoint, endpoint_valid, scale_valid,
                          *, minimum_pairs=3):
    """Errors are already in the prefix-normalized coordinate frame.

    Returns [window,horizon] values and a [window,horizon,33] common target
    support. Invalid target support is distinct from a missing model output.
    """
    p, y = np.asarray(prediction, float), np.asarray(endpoint, float)
    valid, scale_ok = np.asarray(endpoint_valid), np.asarray(scale_valid)
    if (p.shape != y.shape or y.ndim != 4 or y.shape[-2:] != (33, 2)
            or valid.shape != y.shape[:-1] or valid.dtype != bool
            or scale_ok.shape != (len(y),) or scale_ok.dtype != bool):
        raise ValueError("Expected predictions/targets [N,H,33,2], boolean validity and scale_valid")
    if not 1 <= minimum_pairs <= len(PRIMARY_PAIRS):
        raise ValueError("minimum_pairs must lie between one and four")
    if not np.isfinite(y[valid]).all():
        raise ValueError("An observed endpoint has nonfinite coordinates")
    support = np.zeros_like(valid)
    pair_count = np.zeros(valid.shape[:2], int)
    for left, right in PRIMARY_PAIRS:
        pair = valid[:, :, left] & valid[:, :, right]
        pair_count += pair
        support[:, :, left] = support[:, :, right] = pair
    eligible = (pair_count >= minimum_pairs) & scale_ok[:, None]
    support &= eligible[..., None]
    finite = np.isfinite(p).all(-1)
    complete = (~support | finite).all(-1)
    delta = np.where(support[..., None] & finite[..., None], p - y, 0)
    errors = np.linalg.norm(delta, axis=-1).sum(-1) / np.maximum(support.sum(-1), 1)
    errors[~eligible | ~complete] = np.nan
    return {"errors": errors, "eligible": eligible, "prediction_complete": complete,
            "joint_support": support, "valid_pair_count": pair_count}


def secondary_motion_errors(prediction, endpoint, endpoint_valid, scale_valid,
                            *, minimum_pairs=3):
    """Separate projected root and articulation errors; never a primary gate.

    Pelvis is the midpoint of observed/predicted hips 23 and 24. The origin and
    body-length scale were fixed from the prefix for BOTH endpoints: therefore
    the pelvis endpoint error is also its displacement-from-prefix error. The
    root-relative metric subtracts each skeleton's own future pelvis before
    comparing paired lower-limb joints. No future-fitted scale is introduced.

    Both secondaries require primary target eligibility plus both observed
    hips. That extra condition does not alter the primary target support.
    """
    primary = primary_window_errors(prediction, endpoint, endpoint_valid, scale_valid,
                                     minimum_pairs=minimum_pairs)
    p, y = np.asarray(prediction, float), np.asarray(endpoint, float)
    valid = np.asarray(endpoint_valid)
    eligible = primary["eligible"] & valid[:, :, 23] & valid[:, :, 24]
    predicted_hips = p[:, :, [23, 24]]
    observed_hips = y[:, :, [23, 24]]
    hips_complete = np.isfinite(predicted_hips).all((-2, -1))
    pelvis_delta = predicted_hips.mean(-2) - observed_hips.mean(-2)
    root_error = np.linalg.norm(pelvis_delta, axis=-1)
    root_error[~eligible | ~hips_complete] = np.nan
    root_support = np.zeros_like(valid)
    root_support[:, :, 23] = root_support[:, :, 24] = eligible

    limb_support = primary["joint_support"] & eligible[..., None]
    limb_complete = primary["prediction_complete"] & hips_complete
    # (predicted joint - predicted pelvis) - (true joint - true pelvis).
    relative_delta = p - y - pelvis_delta[:, :, None]
    safe_delta = np.where(limb_support[..., None] & np.isfinite(relative_delta), relative_delta, 0)
    relative_error = np.linalg.norm(safe_delta, axis=-1).sum(-1) / np.maximum(limb_support.sum(-1), 1)
    relative_error[~eligible | ~limb_complete] = np.nan
    return {
        "root_displacement_2d": {"errors": root_error, "eligible": eligible,
            "prediction_complete": hips_complete, "joint_support": root_support},
        "root_relative_lower_limb_2d": {"errors": relative_error, "eligible": eligible,
            "prediction_complete": limb_complete, "joint_support": limb_support},
    }


def _identities(records):
    seen, videos, bouts = set(), {}, {}
    for row in records:
        required = ("window_id", "video_id", "sequence_id", "group_id")
        if any(not str(row.get(key, "")) for key in required):
            raise ValueError("Every window needs window, video, bout and group identities")
        if row["window_id"] in seen:
            raise ValueError("Duplicate window_id in evaluation")
        seen.add(row["window_id"])
        video, group, bout = str(row["video_id"]), str(row["group_id"]), str(row["sequence_id"])
        if video in videos and videos[video] != group:
            raise ValueError("One video belongs to multiple connected groups")
        if bout in bouts and bouts[bout] != video:
            raise ValueError("One bout belongs to multiple videos")
        videos[video], bouts[bout] = group, video
    return videos


def aggregate_window_errors(records, measured, horizons, *, method="", seed=0):
    """Mean joints→windows→bouts→videos, retaining unavailable predictions."""
    videos = _identities(records)
    errors = np.asarray(measured["errors"], float)
    eligible = np.asarray(measured["eligible"], bool)
    complete = np.asarray(measured["prediction_complete"], bool)
    if (errors.shape != (len(records), len(horizons)) or eligible.shape != errors.shape
            or complete.shape != errors.shape):
        raise ValueError("Window errors and declared horizons disagree")
    source_rows, summaries = [], []
    for h, horizon in enumerate(horizons):
        by_video = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(records):
            by_video[str(row["video_id"])][str(row["sequence_id"])].append(index)
        h_rows = []
        for video, bouts in sorted(by_video.items()):
            bout_means, scored_windows, eligible_windows, absent_predictions = [], 0, 0, 0
            for indices in bouts.values():
                index = np.asarray(indices, int)
                selected = index[eligible[index, h]]
                eligible_windows += len(selected)
                absent_predictions += int((~complete[selected, h]).sum())
                finite = errors[selected, h][np.isfinite(errors[selected, h])]
                scored_windows += len(finite)
                if len(finite):
                    bout_means.append(float(finite.mean()))
            score = (float(np.mean(bout_means)) if bout_means and not absent_predictions else None)
            row = {"method": method, "seed": int(seed), "horizon_seconds": float(horizon),
                   "video_id": video, "group_id": videos[video], "score": score,
                   "eligible_windows": eligible_windows, "scored_windows": scored_windows,
                   "missing_predictions": absent_predictions, "scored_bouts": len(bout_means)}
            source_rows.append(row)
            h_rows.append(row)
        observed = [r["score"] for r in h_rows if r["score"] is not None]
        missing_predictions = sum(r["missing_predictions"] for r in h_rows)
        summaries.append({"method": method, "seed": int(seed), "horizon_seconds": float(horizon),
            "score": float(np.mean(observed)) if observed and not missing_predictions else None,
            "sources": len(observed), "declared_sources": len(videos),
            "groups": len({r["group_id"] for r in h_rows if r["score"] is not None}),
            "eligible_windows": int(eligible[:, h].sum()),
            "missing_predictions": missing_predictions,
            "status": ("missing_predictions" if missing_predictions else
                       "evaluated" if observed else "no_observed_support"),
            "estimand": "mean windows within bout; mean bouts within video; equal videos"})
    return {"sources": source_rows, "summary": summaries}


def paired_group_bootstrap(first: Sequence[dict], reference: Sequence[dict], *,
                           repetitions=2000, seed=812):
    """Paired source scores for one horizon, jointly resampled across seeds.

    Rows require seed/video_id/group_id/score. Difference is first-reference,
    so negative error differences favor the first model. The fitted pipeline
    remains fixed and seed standard deviation is reported separately.
    """
    if repetitions < 2:
        raise ValueError("At least two bootstrap repetitions are required")

    def indexed(rows):
        result = {}
        for row in rows:
            key = (int(row["seed"]), str(row["video_id"]))
            if key in result:
                raise ValueError("Duplicate seed/video score")
            value = row.get("score")
            if value is None or not np.isfinite(value):
                raise ValueError("Paired bootstrap requires complete finite source scores")
            result[key] = (str(row["group_id"]), float(value))
        return result

    a, b = indexed(first), indexed(reference)
    if not a or set(a) != set(b):
        raise ValueError("Paired methods must cover identical seeds and videos")
    seeds = sorted({key[0] for key in a})
    videos = sorted({key[1] for key in a})
    if set(a) != {(s, v) for s in seeds for v in videos}:
        raise ValueError("Every seed must cover the same source videos")
    groups = {v: a[(seeds[0], v)][0] for v in videos}
    for key in a:
        if a[key][0] != b[key][0] or a[key][0] != groups[key[1]]:
            raise ValueError("Source group assignments differ across methods or seeds")
    group_names = sorted(set(groups.values()))
    if len(group_names) < 2:
        raise ValueError("At least two independent source groups are required")
    group_index = np.array([group_names.index(groups[v]) for v in videos])
    av = np.array([[a[(s, v)][1] for v in videos] for s in seeds])
    bv = np.array([[b[(s, v)][1] for v in videos] for s in seeds])
    deltas = (av - bv).mean(1)
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, float)
    gains = np.empty(repetitions, float)
    for i in range(repetitions):
        counts = np.bincount(rng.integers(0, len(group_names), len(group_names)),
                             minlength=len(group_names))
        weights = counts[group_index]
        draws[i] = np.average(av - bv, weights=weights, axis=1).mean()
        reference_draw = np.average(bv, weights=weights, axis=1).mean()
        gains[i] = -draws[i] / reference_draw if reference_draw > 0 else np.nan
    lower, upper = np.quantile(draws, [0.025, 0.975])
    baseline = float(bv.mean())
    return {"difference": float(deltas.mean()), "lower_95": float(lower), "upper_95": float(upper),
            "first_mean": float(av.mean()), "reference_mean": baseline,
            "relative_improvement": float((bv.mean() - av.mean()) / baseline) if baseline > 0 else None,
            "relative_improvement_lower_95": float(np.quantile(gains, .025)) if np.isfinite(gains).all() else None,
            "relative_improvement_upper_95": float(np.quantile(gains, .975)) if np.isfinite(gains).all() else None,
            "seed_differences": dict(zip(map(str, seeds), map(float, deltas))),
            "seed_sd_difference": float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0,
            "videos": len(videos), "groups": len(group_names), "seeds": len(seeds),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": int(seed),
            "bootstrap_differences": draws.tolist(),
            "bootstrap_relative_improvements": [float(g) if np.isfinite(g) else None for g in gains],
            "scope": "paired connected-group resampling conditional on fitted models; equal-video estimand"}
