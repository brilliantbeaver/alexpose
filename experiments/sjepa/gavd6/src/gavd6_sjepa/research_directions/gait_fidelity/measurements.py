"""Projected gait measurements with reference-only support and explicit failures.

Angles are measured in original image geometry. The primary outcome is right
minus left P95-P5 knee-angle excursion; it is not anatomical 3D range of motion.
"""
from __future__ import annotations

import numpy as np
import torch

KNEES = ((6, 8, 10), (7, 9, 11))  # left/right hip, knee, ankle in body12


def knee_angles(xy, valid=None, *, min_segment_px=1.0):
    """Return [..., time, left/right] degrees and geometric support."""
    xy = np.asarray(xy, float)
    if xy.shape[-2:] != (12, 2):
        raise ValueError("Body12 xy[...,time,12,2] is required")
    if min_segment_px <= 0:
        raise ValueError("A positive projected segment threshold is required")
    valid = np.isfinite(xy).all(-1) if valid is None else np.asarray(valid, bool) & np.isfinite(xy).all(-1)
    result, support = [], []
    for hip, knee, ankle in KNEES:
        u, v = xy[..., hip, :] - xy[..., knee, :], xy[..., ankle, :] - xy[..., knee, :]
        length_u, length_v = np.linalg.norm(u, axis=-1), np.linalg.norm(v, axis=-1)
        ok = valid[..., [hip, knee, ankle]].all(-1) & (length_u >= min_segment_px) & (length_v >= min_segment_px)
        cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        angle = np.degrees(np.arctan2(np.abs(cross), (u * v).sum(-1)))
        result.append(np.where(ok, angle, np.nan))
        support.append(ok)
    return np.stack(result, -1), np.stack(support, -1)


def uniform_time_check(timestamps):
    times = np.asarray(timestamps, float)
    if times.shape[-1] < 2 or not np.isfinite(times).all():
        raise ValueError("At least two finite physical timestamps are required")
    dt = np.diff(times, axis=-1)
    if (dt <= 0).any() or not np.allclose(dt, dt[..., :1], rtol=1e-4, atol=1e-6):
        raise ValueError("Percentile endpoint requires a uniform physical-time grid")


def reference_support(xy, valid, *, groups=None, min_segment_px=1.0):
    """Common reference-valid frames across both legs and every row in a group.

    A group may be a physical pair or all four physical/nuisance cells. The
    caller forms groups from metadata before examining method predictions.
    """
    _, geometry = knee_angles(xy, valid, min_segment_px=min_segment_px)
    support = geometry.all(-1)
    if groups is not None:
        for rows in groups:
            rows = np.asarray(rows, int)
            if not len(rows):
                raise ValueError("A reference contrast cannot contain an empty group")
            support[rows] = np.logical_and.reduce(support[rows], axis=0)
    return support


def knee_excursion(xy, valid=None, *, support=None, timestamps=None,
                   min_segment_px=1.0, min_frames=16, min_fraction=.75):
    """Evaluate on fixed reference support, keeping prediction failures visible.

    Output arrays preserve leading example dimensions. A prediction with any
    missing/degenerate limb on the admitted reference frames fails; its value
    remains NaN, never a success computed on a convenient subset of frames.
    """
    if not 0 < min_fraction <= 1 or min_frames < 2:
        raise ValueError("Invalid minimum reference coverage")
    if timestamps is not None:
        uniform_time_check(timestamps)
    angles, geometric = knee_angles(xy, valid, min_segment_px=min_segment_px)
    if angles.ndim == 2:
        angles, geometric = angles[None], geometric[None]
        if support is not None:
            support = np.asarray(support)[None]
    if angles.ndim != 3:
        raise ValueError("Use xy[examples,time,12,2] or xy[time,12,2]")
    fixed = geometric.all(-1) if support is None else np.asarray(support, bool)
    if fixed.shape != angles.shape[:2]:
        raise ValueError("Fixed reference support must be [examples,time]")
    count = fixed.sum(1)
    reference_eligible = (count >= min_frames) & (count >= min_fraction * angles.shape[1])
    prediction_failure = reference_eligible & (fixed & ~geometric.all(-1)).any(1)
    admitted = reference_eligible & ~prediction_failure
    q = np.full((len(angles), 2), np.nan)
    for i in np.flatnonzero(admitted):
        p = np.quantile(angles[i, fixed[i]], [.05, .95], axis=0, method="linear")
        q[i] = p[1] - p[0]
    return {"left": q[:, 0], "right": q[:, 1], "asymmetry": q[:, 1] - q[:, 0],
            "supported": admitted, "reference_eligible": reference_eligible,
            "prediction_failure": prediction_failure, "reference_frames": count,
            "angles": angles, "frame_support": fixed}


def torch_knee_angles(xy):
    """Differentiable atan2 angle; finite at straight and degenerate limbs.

    Exactly straight knees have a zero subgradient through abs(cross). Coordinate
    loss is retained. Degenerate limbs receive a separate length penalty during
    training; evaluation marks them as failures instead of assigning an angle.
    """
    xy = xy.float()
    values, lengths = [], []
    for hip, knee, ankle in KNEES:
        u, v = xy[..., hip, :] - xy[..., knee, :], xy[..., ankle, :] - xy[..., knee, :]
        cross = u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
        dot = (u * v).sum(-1)
        # atan2(0,0) has undefined backward even if masked later.
        dot = torch.where((cross.abs() + dot.abs()) > 1e-12, dot, torch.ones_like(dot) * 1e-12)
        values.append(torch.atan2(cross.abs(), dot) * (180.0 / np.pi))
        lengths.append(torch.stack(((u.square().sum(-1) + 1e-12).sqrt(),
                                    (v.square().sum(-1) + 1e-12).sqrt()), -1))
    return torch.stack(values, -1), torch.stack(lengths, -2)


def torch_knee_excursion(xy, support, *, min_frames=16, min_fraction=.75):
    """Exact linear percentiles on fixed reference support, with sparse gradients."""
    angles, lengths = torch_knee_angles(xy)
    support = torch.as_tensor(support, dtype=torch.bool, device=xy.device)
    if support.shape != xy.shape[:2]:
        raise ValueError("Reference support differs from the prediction time grid")
    count = support.sum(1)
    eligible = (count >= min_frames) & (count >= min_fraction * xy.shape[1])
    values = []
    for i in range(len(xy)):
        if bool(eligible[i]):
            q = torch.quantile(angles[i, support[i]], angles.new_tensor([.05, .95]), dim=0, interpolation="linear")
            excursion = q[1] - q[0]
            values.append(excursion[1] - excursion[0])
        else:
            # Numeric zero is an internal placeholder excluded by eligible.
            values.append(xy[i].sum() * 0)
    return torch.stack(values), eligible, lengths


def measurement_loss(predicted_px, target_px, valid, *, objective,
                     min_segment_px=1.0, min_frames=16, min_fraction=.75):
    """Return angle/180 squared loss and diagnostics for [B,2,T,12,2] pairs.

    Both endpoint labels are recomputed from their actual tensors. Per-example
    and paired terms use the identical marginal endpoints and common reference
    timestamps. Neither target geometry nor validity is a model input.
    """
    if predicted_px.shape != target_px.shape or predicted_px.ndim != 5 or predicted_px.shape[1] != 2:
        raise ValueError("Measurement supervision requires paired endpoint tensors")
    b, _, t, _, _ = predicted_px.shape
    truth = target_px.reshape(2 * b, t, 12, 2)
    flags = valid.reshape(2 * b, t, 12)
    ref_angles, ref_lengths = torch_knee_angles(torch.where(flags[..., None], truth, 0))
    del ref_angles
    finite = flags[..., [6, 8, 10, 7, 9, 11]].all(-1)
    geometry = (ref_lengths >= min_segment_px).all(-1).all(-1)
    fixed = (finite & geometry).reshape(b, 2, t).all(1)
    fixed_endpoints = fixed[:, None].expand(-1, 2, -1).reshape(2 * b, t)
    predicted = predicted_px.reshape(2 * b, t, 12, 2)
    predicted_a, eligible, lengths = torch_knee_excursion(predicted, fixed_endpoints,
        min_frames=min_frames, min_fraction=min_fraction)
    target_a, _, _ = torch_knee_excursion(torch.where(flags[..., None], truth, 0), fixed_endpoints,
        min_frames=min_frames, min_fraction=min_fraction)
    ok = eligible.reshape(b, 2).all(1)
    p, y = predicted_a.reshape(b, 2), target_a.detach().reshape(b, 2)
    if objective in {"paired_change", "repaired_change"}:
        error = ((p[:, 1] - p[:, 0]) - (y[:, 1] - y[:, 0])) / 180.
        error = error.square()
    elif objective in {"per_example", "per_example_measurement"}:
        error = ((p - y) / 180.).square().mean(1)
    else:
        raise ValueError(f"Unknown measurement objective: {objective}")
    if not ok.any():
        return predicted_px.sum() * 0, {"supported_pairs": 0, "unsupported_pairs": b,
            "measurement_loss": None, "short_segment_penalty": None}
    # Short predicted limbs cannot remove training support. Penalize them using
    # the same fixed reference frames; report the term separately.
    short = torch.relu((min_segment_px - lengths) / min_segment_px).square().mean((-1, -2))
    short = (short * fixed_endpoints).sum(1) / fixed_endpoints.sum(1).clamp_min(1)
    short = short.reshape(b, 2).mean(1)
    loss = (error + short)[ok].mean()
    return loss, {"supported_pairs": int(ok.sum()), "unsupported_pairs": int((~ok).sum()),
        "measurement_loss": float(error[ok].detach().mean()),
        "short_segment_penalty": float(short[ok].detach().mean()),
        "gradient_convention": "linear_percentile_atan2_abs; coordinate_MSE_retained",
        "units": "squared(angle_degrees/180) plus squared_relative_segment_deficit"}
