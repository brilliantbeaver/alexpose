"""Pure past-only preparation; hidden feature masking is a later, distinct step."""
import numpy as np


def prefix_geometry(coords, valid, width, height):
    """Median projected shoulder→hip→knee→ankle chain, ≥8 frames EACH side.

    Arrays here contain distinct ORIGINAL prefix observations, not held grid bins.
    The fallback is for SSL input only. It never certifies endpoint scale support.
    """
    minimum = max(20., .02 * np.hypot(width, height))
    pelvis_ok = valid[:, 23] & valid[:, 24]
    origin = ((coords[pelvis_ok, 23] + coords[pelvis_ok, 24]) * .5)[-1] if pelvis_ok.any() else np.zeros(2)
    chains, counts = [], []
    for joints in ([11, 23, 25, 27], [12, 24, 26, 28]):
        ok = valid[:, joints].all(1)
        lengths = np.linalg.norm(np.diff(coords[ok][:, joints], axis=1), axis=-1).sum(1)
        lengths = lengths[np.isfinite(lengths)]
        counts.append(len(lengths))
        if len(lengths):
            chains.append(lengths)
    scale = float(np.median(np.concatenate(chains))) if len(chains) == 2 else float("nan")
    supported = min(counts) >= 8 and np.isfinite(scale) and scale >= minimum and pelvis_ok.any()
    reason = "supported" if supported else "insufficient_distinct_chain_observations_or_scale_or_pelvis"
    if not supported:
        # Explicit label-blind normalization fallback; primary metric is missing.
        scale = float(np.hypot(width, height))
    return origin, scale, bool(supported), reason


def context_input(pose, cfg, issue, *, index_resized=False):
    keep = (pose.times >= issue - cfg.prefix_seconds - 1e-10) & (pose.times < issue)
    times = pose.times[keep]
    coords = pose.coords[keep]
    valid = pose.observed[keep]
    if not len(times):
        raise ValueError("Empty half-open prefix")
    origin, scale, scale_valid, reason = prefix_geometry(coords, valid, pose.width, pose.height)
    count = round(cfg.prefix_seconds * cfg.grid_hz)
    queries = issue - cfg.prefix_seconds + np.arange(count) / cfg.grid_hz
    out = np.zeros((count, 33, 2), np.float32)
    mask = np.zeros((count, 33), bool)
    actual = np.zeros((count, 33), np.float32)
    age = np.zeros((count, 33), np.float32)
    if index_resized:
        # Named index-resize common-window adaptation, NOT historical whole-clip replay.
        positions = np.linspace(0, len(times) - 1, count)
        for i, pos in enumerate(positions):
            lo, hi = int(np.floor(pos)), int(np.ceil(pos))
            fraction = pos - lo
            ok = valid[lo] & valid[hi]
            mask[i] = ok
            out[i, ok] = ((1 - fraction) * coords[lo, ok] + fraction * coords[hi, ok] - origin) / scale
            actual[i, ok] = (1 - fraction) * times[lo] + fraction * times[hi] - issue
    else:
        for j in range(33):
            observed = np.flatnonzero(valid[:, j])
            if not len(observed):
                continue
            positions = np.searchsorted(times[observed], queries + 1e-10, side="right") - 1
            indices = observed[np.maximum(positions, 0)]
            ages = queries - times[indices]
            ok = (positions >= 0) & (ages >= -1e-9) & (ages <= cfg.max_observation_age_seconds + 1e-10)
            mask[:, j] = ok
            out[ok, j] = (coords[indices[ok], j] - origin) / scale
            actual[ok, j] = times[indices[ok]] - issue
            age[ok, j] = np.maximum(0, ages[ok])
    return dict(context=out, context_valid=mask, context_times=actual, context_age=age,
                query_times=(queries - issue).astype(np.float32), origin=origin, scale=scale,
                scale_valid=scale_valid, scale_reason=reason)


def nearest_endpoint(pose, query, issue, tolerance, *, interval=None):
    """Nearest original FRAME, tie earlier; no per-joint visibility cherry-picking."""
    coords = np.zeros((33, 2), np.float32)
    valid = np.zeros(33, bool)
    times = np.full(33, np.nan)
    candidates = (pose.times > issue) & (np.abs(pose.times - query) <= tolerance + 1e-10)
    if interval is not None:
        start, end = interval
        candidates &= (pose.times > start) & (pose.times <= end + 1e-10)
    eligible = np.flatnonzero(candidates)
    if len(eligible):
        distance = np.abs(pose.times[eligible] - query)
        # Numerical near-ties toward earlier original PTS.
        index = eligible[np.flatnonzero(distance <= distance.min() + 1e-10)[0]]
        valid = pose.observed[index].copy()
        coords[valid] = pose.coords[index, valid]
        times[valid] = pose.times[index]
    return coords, valid, times


def signed_speed_contrast(coords, valid, times, pairs=((25, 26), (27, 28), (29, 30), (31, 32))):
    """Diagnostic arithmetic only; legacy target remains in historical geometry.py."""
    dt = np.diff(times)
    if np.any(dt <= 0):
        raise ValueError("Increasing original time required")
    contrasts = []
    for left, right in pairs:
        ok = valid[:-1, left] & valid[1:, left] & valid[:-1, right] & valid[1:, right]
        if not ok.any():
            continue
        speeds = np.linalg.norm(np.diff(coords[:, [left, right]], axis=0), axis=-1) / dt[:, None]
        means = speeds[ok].mean(0)
        if means.sum() > 0:
            contrasts.append(float((means[0] - means[1]) / means.sum()))
    return float(np.mean(contrasts)) if contrasts else float("nan")
