"""Post-run diagnostics for operational 2D ankle-separation maxima.

This module does not change the experiment's retained metrics or scientific
gates. Its tolerance-based event matching is a diagnostic, not a clinical event
definition. Eligibility depends on references alone, while the original metric
also required a complete prediction with exactly the reference peak count.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def _positive_peaks(centered):
    """Exactly the local-maximum rule used by the original evaluator."""
    return np.flatnonzero(
        (centered[1:-1] > centered[:-2])
        & (centered[1:-1] >= centered[2:])
        & (centered[1:-1] > 0)
    ) + 1


def _ordered_matches(reference_times, prediction_times, tolerance_s):
    """Maximize ordered one-to-one matches, then minimize total absolute error.

    The dynamic program permits a match only inside the stated time tolerance;
    skipping either sequence cannot reuse an event. The return value contains
    indices into the two supplied arrays, rather than frame indices.
    """
    a = np.asarray(reference_times, dtype=float)
    z = np.asarray(prediction_times, dtype=float)
    if (a.ndim != 1 or z.ndim != 1 or not np.isfinite(a).all()
            or not np.isfinite(z).all() or np.any(np.diff(a) <= 0)
            or np.any(np.diff(z) <= 0)):
        raise ValueError("Event times must be finite, one-dimensional and strictly increasing")
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("Event tolerance must be finite and nonnegative")
    n, m = len(a), len(z)
    counts = np.zeros((n + 1, m + 1), dtype=np.int64)
    costs = np.zeros((n + 1, m + 1), dtype=float)
    choices = np.zeros((n, m), dtype=np.int8)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            count, cost, choice = counts[i + 1, j], costs[i + 1, j], 0
            candidate = (counts[i, j + 1], costs[i, j + 1])
            if candidate[0] > count or (candidate[0] == count and candidate[1] < cost):
                count, cost, choice = *candidate, 1
            distance = abs(a[i] - z[j])
            # Floating point rounding at the tolerance boundary is not a new
            # scientific matching allowance (one picosecond at most).
            if distance <= tolerance_s + 1e-12:
                candidate = (counts[i + 1, j + 1] + 1,
                             costs[i + 1, j + 1] + distance)
                if candidate[0] > count or (candidate[0] == count and candidate[1] <= cost):
                    count, cost, choice = *candidate, 2
            counts[i, j], costs[i, j], choices[i, j] = count, cost, choice
    pairs = []
    i = j = 0
    while i < n and j < m:
        choice = choices[i, j]
        if choice == 2:
            pairs.append((i, j))
            i += 1
            j += 1
        elif choice == 0:
            i += 1
        else:
            j += 1
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def timing_diagnostics(predictions, targets, timestamps, records, *,
                       method, evidence_status, tolerance_s=.12):
    """Return one diagnostic row per input record using only NumPy and pandas.

    Shapes match ``evaluate_predictions``: coordinates [B,T,12,2], Boolean
    ``valid``/``visible`` [B,T,12], and ``eval_scale``/timestamps [B,T]. Body-12
    ankles are indices 10 and 11. Times are physical seconds; scales are finite
    positive independent reference-box diagonals in pixels.

    A reference is eligible only with both ankles valid, visible and finite for
    the full window, at least 8 frames, duration >= 1 second, normalized RMS
    amplitude > 1e-8, and at least two positive maxima. These are the original
    reference conditions, separated from prediction conditions. Invalid or
    nonfinite reference coordinates are diagnosed rather than silently filled.

    ``original_support`` reproduces the old timing eligibility; old timing MAE
    is independent of this diagnostic's matching tolerance. Matching maximizes
    cardinality, then minimizes total absolute timing error. A missing predicted
    ankle anywhere fails the full window: all eligible reference events are
    missed, recall is zero, and predicted/extra counts and precision are unknown.
    Ineligible references receive no precision/recall or matched timing score.
    ``matched_timing_mae_s`` is conditional on matches and must be read together
    with missed/extra counts and the explicit coverage denominators.

    Passing ``targets['xy']`` as predictions provides the reference oracle and
    the original metric's attainable support ceiling. It is not a learned
    method. No cadence, heel-strike or clinical interpretation is implied.
    """
    pred = np.asarray(predictions, dtype=float)
    truth = np.asarray(targets["xy"], dtype=float)
    valid = np.asarray(targets["valid"])
    visible = np.asarray(targets["visible"])
    scale = np.asarray(targets["eval_scale"], dtype=float)
    times = np.asarray(timestamps, dtype=float)
    records = list(records)
    if pred.ndim != 4 or pred.shape[-2:] != (12, 2) or truth.shape != pred.shape:
        raise ValueError("Coordinates must share shape [B,T,12,2]")
    b, t = pred.shape[:2]
    if b == 0 or t < 2:
        raise ValueError("At least one window and two physical-time samples are required")
    if (valid.shape != (b, t, 12) or visible.shape != valid.shape
            or valid.dtype != bool or visible.dtype != bool):
        raise ValueError("Target valid/visible must be Boolean [B,T,12]")
    if scale.shape != (b, t) or times.shape != (b, t) or len(records) != b:
        raise ValueError("Scale/timestamp/record shape mismatch")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("Records must be mappings")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Independent evaluation scale must be finite positive pixels")
    if not np.isfinite(times).all() or np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("Physical timestamps must be finite and strictly increasing")
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("Event tolerance must be finite and nonnegative")

    rows = []
    for i, record in enumerate(records):
        reference_finite = np.isfinite(truth[i, :, 10:12]).all(axis=(1, 2))
        prediction_finite = np.isfinite(pred[i, :, 10:12]).all(axis=(1, 2))
        bilateral_visible = visible[i, :, 10:12].all(axis=1)
        bilateral_valid = valid[i, :, 10:12].all(axis=1)
        duration = float(times[i, -1] - times[i, 0])
        scale_mean = float(scale[i].mean())
        original_input_valid = bool(
            not np.any(visible[i] & ~valid[i])
            and not np.any(valid[i] & ~np.isfinite(truth[i]).all(axis=-1))
        )
        row = {
            **record, "method": method, "evidence_status": evidence_status,
            "diagnostic_kind": "postrun_2d_ankle_maxima",
            "tolerance_s": float(tolerance_s),
            "frames": t, "duration_s": duration,
            "reference_visible_frames": int(bilateral_visible.sum()),
            "reference_valid_frames": int(bilateral_valid.sum()),
            "reference_finite_frames": int(reference_finite.sum()),
            "prediction_finite_frames": int(prediction_finite.sum()),
            "original_metric_input_valid": original_input_valid,
            "reference_amplitude_raw_px": np.nan,
            "reference_amplitude_normalized": np.nan,
            "reference_amplitude_constant_scale": np.nan,
            "prediction_amplitude_normalized": np.nan,
            "reference_event_count": np.nan,
            "reference_raw_event_count": np.nan,
            "predicted_event_count": np.nan,
            "bbox_scale_mean_px": scale_mean,
            "bbox_scale_min_px": float(scale[i].min()),
            "bbox_scale_max_px": float(scale[i].max()),
            "bbox_scale_cv": float(scale[i].std() / scale_mean),
            "bbox_scale_relative_range": float(np.ptp(scale[i]) / scale_mean),
            "original_support": False, "original_event_count": 0,
            "original_event_timing_mae_s": np.nan,
            "original_event_status": "insufficient_reference_support",
            "matched_event_count": np.nan, "missed_event_count": np.nan,
            "extra_event_count": np.nan, "matched_timing_mae_s": np.nan,
            "event_recall": np.nan, "event_precision": np.nan,
            "event_f1": np.nan,
            # Sum these denominators explicitly; absent metrics are never
            # implicitly successful windows or zero-error observations.
            "window_denominator": 1,
            "eligible_window_denominator": 0,
            "complete_prediction_window_denominator": 0,
            "reference_event_denominator": 0,
            "prediction_event_denominator": 0,
            "matched_timing_denominator": 0,
        }
        a = z = None
        if reference_finite.all():
            raw = truth[i, :, 10, 0] - truth[i, :, 11, 0]
            raw_centered = raw - raw.mean()
            ref = raw / scale[i]
            ref -= ref.mean()
            a = _positive_peaks(ref)
            raw_amp = float(np.sqrt(np.mean(raw_centered ** 2)))
            ref_amp = float(np.sqrt(np.mean(ref ** 2)))
            row.update(reference_amplitude_raw_px=raw_amp,
                       reference_amplitude_normalized=ref_amp,
                       reference_amplitude_constant_scale=raw_amp / scale_mean,
                       reference_event_count=len(a),
                       reference_raw_event_count=len(_positive_peaks(raw_centered)))
        if prediction_finite.all():
            out = (pred[i, :, 10, 0] - pred[i, :, 11, 0]) / scale[i]
            out -= out.mean()
            z = _positive_peaks(out)
            row.update(predicted_event_count=len(z),
                       prediction_amplitude_normalized=float(np.sqrt(np.mean(out ** 2))))

        reasons = []
        if not bilateral_valid.all():
            reasons.append("reference_invalid_ankles")
        if not bilateral_visible.all():
            reasons.append("reference_visibility_gap")
        if not reference_finite.all():
            reasons.append("reference_nonfinite_ankles")
        if t < 8:
            reasons.append("fewer_than_eight_frames")
        if duration < 1:
            reasons.append("duration_below_one_second")
        if reference_finite.all():
            if row["reference_amplitude_normalized"] <= 1e-8:
                reasons.append("reference_amplitude_too_small")
            if len(a) < 2:
                reasons.append("fewer_than_two_reference_peaks")
        eligible = not reasons
        row.update(reference_eligible=eligible,
                   reference_status="eligible" if eligible else reasons[0],
                   reference_reasons=";".join(reasons))

        # Reconstruct the original status as well as support for valid original
        # inputs; the source evaluator rejects invalid inputs before scoring.
        if not original_input_valid:
            row["original_event_status"] = "invalid_original_metric_input"
        elif (bilateral_visible.all() and prediction_finite.all() and t >= 8
              and row["reference_amplitude_normalized"] > 1e-8 and duration >= 1):
            row["original_event_status"] = "insufficient_cycles_or_prediction_event_mismatch"
            if len(a) >= 2 and len(z) == len(a):
                row.update(original_support=True, original_event_count=len(a),
                           original_event_status="supported_2d_ankle_maxima",
                           original_event_timing_mae_s=float(np.abs(times[i, a] - times[i, z]).mean()))

        if not prediction_finite.all():
            row["prediction_status"] = "incomplete_ankle_predictions"
        elif not eligible:
            row["prediction_status"] = "complete_reference_ineligible"
        elif len(z) != len(a):
            row["prediction_status"] = "event_count_mismatch"
        else:
            row["prediction_status"] = "complete_matching_event_count"
        if not eligible:
            row["matching_status"] = "reference_ineligible"
        else:
            row.update(eligible_window_denominator=1,
                       reference_event_denominator=len(a))
            if not prediction_finite.all():
                row.update(matching_status="prediction_incomplete", matched_event_count=0,
                           missed_event_count=len(a), event_recall=0.)
            else:
                pairs = _ordered_matches(times[i, a], times[i, z], tolerance_s)
                matched = len(pairs)
                row.update(
                    complete_prediction_window_denominator=1,
                    prediction_event_denominator=len(z),
                    matched_event_count=matched, missed_event_count=len(a) - matched,
                    extra_event_count=len(z) - matched,
                    matched_timing_denominator=matched,
                    event_recall=matched / len(a),
                    event_precision=matched / len(z) if len(z) else np.nan,
                    event_f1=2 * matched / (len(a) + len(z)),
                    matching_status=("all_events_matched" if matched == len(a) == len(z)
                                     else "partial_matches" if matched else "no_events_matched"),
                )
                if matched:
                    row["matched_timing_mae_s"] = float(np.abs(
                        times[i, a[pairs[:, 0]]] - times[i, z[pairs[:, 1]]]
                    ).mean())
        rows.append(row)
    return pd.DataFrame(rows)
