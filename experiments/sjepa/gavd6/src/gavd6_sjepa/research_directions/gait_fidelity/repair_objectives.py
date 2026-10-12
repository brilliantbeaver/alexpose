"""Separated scalar, temporal and geometric terms for the readout repair study.

The scalar term is exactly the existing paired excursion error.  Dense change
supervises framewise changes across the same physical intervention endpoints;
it is not a velocity loss or an anatomical 3-D measurement.
"""
from __future__ import annotations

import torch

from .measurements import torch_knee_angles

FORMULAS = {
    "scalar": "mean_pairs((((Ahat_b-Ahat_a)-(A_b-A_a))/180)^2); A=right(P95-P5)-left(P95-P5)",
    "dense": "mean_pairs(mean_common_frames_and_legs((((theta_hat_b-theta_hat_a)-(theta_b-theta_a))/180)^2))",
    "geometry": "mean_pairs_endpoints_common_frames_legs_segments(relu((min_segment-length)/min_segment)^2)",
}


def _angle_terms(predicted_angles, reference_angles, fixed, eligible):
    """Equal-pair reduction; both objectives use identical fixed reference support."""
    scalar, dense = [], []
    for i in range(len(predicted_angles)):
        if not bool(eligible[i]):
            continue
        p = predicted_angles[i, :, fixed[i]]
        y = reference_angles[i, :, fixed[i]]
        pq = torch.quantile(p, p.new_tensor([.05, .95]), dim=1, interpolation="linear")
        yq = torch.quantile(y, y.new_tensor([.05, .95]), dim=1, interpolation="linear")
        pe, ye = pq[1] - pq[0], yq[1] - yq[0]
        pa, ya = pe[:, 1] - pe[:, 0], ye[:, 1] - ye[:, 0]
        scalar.append((((pa[1] - pa[0]) - (ya[1] - ya[0])) / 180.).square())
        dense.append((((p[1] - p[0]) - (y[1] - y[0])) / 180.).square().mean())
    if not scalar:
        zero = predicted_angles.sum() * 0
        return zero, zero
    return torch.stack(scalar).mean(), torch.stack(dense).mean()


def repair_measurement_terms(predicted_px, target_px, valid, *, min_segment_px=1.,
                             min_frames=16, min_fraction=.75):
    """Separate exact scalar, dense temporal, and unchanged short-limb penalties.

    Shapes are [pairs, endpoints=2, frames, body12, xy].  Reference validity and
    geometry determine the common support before examining predictions. Invalid
    reference coordinates are zeroed before arithmetic; prediction failures
    cannot remove frames from the supervised support.
    """
    if predicted_px.shape != target_px.shape or predicted_px.ndim != 5 or predicted_px.shape[1] != 2:
        raise ValueError("Repair supervision requires adjacent paired endpoints")
    if valid.shape != predicted_px.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("Repair supervision requires an explicit boolean reference mask")
    if min_segment_px <= 0 or min_frames < 2 or not 0 < min_fraction <= 1:
        raise ValueError("Invalid reference support settings")
    if not torch.isfinite(predicted_px).all():
        raise FloatingPointError("Repair prediction is nonfinite; it cannot remove reference support")
    if (valid & ~torch.isfinite(target_px).all(-1)).any():
        raise ValueError("A valid reference coordinate is nonfinite")
    b, _, t, _, _ = predicted_px.shape
    safe_truth = torch.where(valid[..., None], target_px, 0).reshape(2*b, t, 12, 2)
    angles, lengths = torch_knee_angles(predicted_px.reshape(2*b, t, 12, 2))
    truth_angles, truth_lengths = torch_knee_angles(safe_truth)
    flags = valid.reshape(2*b, t, 12)
    common = (flags[..., [6, 8, 10, 7, 9, 11]].all(-1)
              & (truth_lengths >= min_segment_px).all(-1).all(-1)).reshape(b, 2, t).all(1)
    count = common.sum(1)
    eligible = (count >= min_frames) & (count >= min_fraction*t)
    angles = angles.reshape(b, 2, t, 2)
    scalar, dense = _angle_terms(angles, truth_angles.detach().reshape(b, 2, t, 2), common, eligible)
    short = torch.relu((min_segment_px-lengths)/min_segment_px).square().mean((-1, -2)).reshape(b, 2, t)
    per_pair_short = (short * common[:, None]).sum((1, 2)) / (2*count).clamp_min(1)
    geometry = per_pair_short[eligible].mean() if eligible.any() else predicted_px.sum()*0
    short_count = (((lengths < min_segment_px).reshape(b, 2, t, 2, 2))
                   & common[:, None, :, None, None] & eligible[:, None, None, None, None]).sum()
    result = dict(scalar=scalar, dense=dense, geometry=geometry, angles=angles,
                  frame_support=common, eligible=eligible)
    result["diagnostic"] = dict(supported_pairs=int(eligible.sum()), unsupported_pairs=int((~eligible).sum()),
        common_reference_frames=int(count[eligible].sum()), scalar_loss=float(scalar.detach()),
        dense_loss=float(dense.detach()), short_segment_penalty=float(geometry.detach()),
        predicted_short_segments=int(short_count), support="common reference frames; both endpoints and legs",
        units="squared(angle_degrees/180); geometry=squared relative segment deficit")
    return result


def angle_gradient_support(terms):
    """Measure realized angle-gradient support, without a causal attribution."""
    support = (terms["frame_support"][:, None, :, None]
               & terms["eligible"][:, None, None, None]).expand_as(terms["angles"])
    denominator = int(support.sum())
    result = {}
    for name in ("scalar", "dense"):
        gradient = torch.autograd.grad(terms[name], terms["angles"], retain_graph=True, allow_unused=True)[0]
        active = 0 if gradient is None else int(((gradient.abs() > 1e-12) & support).sum())
        result[name] = dict(nonzero_angle_gradients=active, eligible_angle_entries=denominator,
                            fraction=active/denominator if denominator else None,
                            zero_threshold=1e-12)
    return result
