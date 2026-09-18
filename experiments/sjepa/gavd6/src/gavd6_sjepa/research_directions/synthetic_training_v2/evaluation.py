"""Observable 2D errors, nested aggregation and paired uncertainty.

Pixels and independent evaluation scales are supplied explicitly. Reference
masks are scoring information only; this module never constructs model inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KEYS = ["person_id", "motion_id", "window_id", "variant", "extractor", "split", "seed"]
STRATA = ["method", "split", "seed", "extractor", "evidence_status"]


def _score(error, mask):
    count = mask.sum(axis=1)
    frame = np.divide(np.where(mask, error, 0).sum(axis=1), count,
                      out=np.full(len(count), np.nan), where=count > 0)
    return float(np.nanmean(frame)) if np.any(count) else float("nan")


def evaluate_predictions(predictions, targets, timestamps, records, *,
                         method="candidate", evidence_status="source-run",
                         missing_penalty=1.0, displacement_seconds=0.2):
    """Return one row per window, with explicit metric support counts.

    Arrays: predictions/targets['xy'] [B,T,12,2], valid/visible [B,T,12],
    targets['eval_scale'] [B,T] in pixels, timestamps [B,T] in seconds.
    COCO body-12 order is shoulders, elbows, wrists, hips, knees, ankles with
    left/right interleaved. Synthetic all/occluded metrics require each record
    to explicitly set target_kind='synthetic_proxy'; real hidden joints cannot
    enter those outputs merely because a target array contains values.
    """
    pred = np.asarray(predictions, dtype=float)
    truth = np.asarray(targets["xy"], dtype=float)
    valid = np.asarray(targets["valid"])
    visible = np.asarray(targets["visible"])
    scale = np.asarray(targets["eval_scale"], dtype=float)
    times = np.asarray(timestamps, dtype=float)
    if pred.ndim != 4 or pred.shape[-2:] != (12, 2) or truth.shape != pred.shape:
        raise ValueError("Coordinates must share shape [B,T,12,2]")
    b, t = pred.shape[:2]
    if b == 0 or t < 2:
        raise ValueError("At least one window and two physical-time samples are required")
    if valid.shape != (b, t, 12) or visible.shape != valid.shape or valid.dtype != bool or visible.dtype != bool:
        raise ValueError("Target valid/visible must be Boolean [B,T,12]")
    if scale.shape != (b, t) or times.shape != (b, t) or len(records) != b:
        raise ValueError("Scale/timestamp/record shape mismatch")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Independent evaluation scale must be finite positive pixels")
    if not np.isfinite(times).all() or np.any(np.diff(times, axis=1) <= 0):
        raise ValueError("Physical timestamps must be finite and strictly increasing")
    if np.any(visible & ~valid) or np.any(valid & ~np.isfinite(truth).all(axis=-1)):
        raise ValueError("Visible targets must be valid; valid targets must be finite")
    if not np.isfinite(missing_penalty) or missing_penalty <= 0 or displacement_seconds <= 0:
        raise ValueError("Missing penalty and displacement horizon must be positive")
    metadata = pd.DataFrame(records)
    if not set(KEYS) <= set(metadata) or metadata[KEYS].isna().any().any():
        raise ValueError(f"Every record requires {KEYS}")
    if metadata.duplicated(KEYS).any():
        raise ValueError("Duplicate per-window metric identity")
    finite = np.isfinite(pred).all(axis=-1)
    error = np.linalg.norm(pred - truth, axis=-1) / scale[..., None]
    error = np.where(finite, error, missing_penalty)
    rows = []
    for i, record in enumerate(records):
        mask = visible[i]
        lower = mask.copy()
        lower[:, :6] = False
        row = {**record, "method": method, "evidence_status": evidence_status,
               "visible_nle": _score(error[i], mask),
               "lower_limb_nle": _score(error[i], lower),
               "p95_nle": float(np.quantile(error[i][mask], 0.95)) if mask.any() else np.nan,
               "missing_rate": float((~finite[i] & mask).sum() / mask.sum()) if mask.any() else np.nan,
               "visible_count": int(mask.sum()), "visible_frame_count": int(mask.any(axis=1).sum()),
               "lower_limb_count": int(lower.sum()), "missing_count": int((~finite[i] & mask).sum()),
               "target_kind": record.get("target_kind", "unspecified"),
               "synthetic_all_nle": np.nan, "synthetic_occluded_nle": np.nan,
               "synthetic_all_count": 0, "synthetic_occluded_count": 0}
        if row["target_kind"] == "synthetic_proxy":
            row.update(synthetic_all_nle=_score(error[i], valid[i]),
                       synthetic_occluded_nle=_score(error[i], valid[i] & ~mask),
                       synthetic_all_count=int(valid[i].sum()),
                       synthetic_occluded_count=int((valid[i] & ~mask).sum()))
        # Both endpoints must have visible independent references; no inferred
        # visibility, time stretching, or time warping enters the comparison.
        displacements, support, failed_pairs = [], 0, 0
        for start in range(t):
            desired = times[i, start] + displacement_seconds
            end = int(np.argmin(np.abs(times[i] - desired)))
            if end <= start or abs(times[i, end] - desired) > 1e-6:
                continue
            supported = mask[start] & mask[end]
            if not supported.any():
                continue
            good = finite[i, start] & finite[i, end]
            delta = (pred[i, end] - pred[i, start]) - (truth[i, end] - truth[i, start])
            e = np.linalg.norm(delta, axis=-1) / scale[i, start]
            e = np.where(good, e, missing_penalty)
            displacements.extend(e[supported])
            support += int(supported.sum())
            failed_pairs += int((supported & ~good).sum())
        row.update(displacement_nle=float(np.mean(displacements)) if support else np.nan,
                   displacement_count=support, displacement_missing_count=failed_pairs,
                   displacement_seconds=float(displacement_seconds))
        bilateral = mask[:, 10] & mask[:, 11]
        predicted = finite[i, :, 10] & finite[i, :, 11]
        ref_sep = (truth[i, :, 10, 0] - truth[i, :, 11, 0]) / scale[i]
        out_sep = (pred[i, :, 10, 0] - pred[i, :, 11, 0]) / scale[i]
        sep_error = np.where(predicted, np.abs(out_sep - ref_sep), missing_penalty)
        row.update(ankle_separation_mae=float(np.mean(sep_error[bilateral])) if bilateral.any() else np.nan,
                   ankle_separation_count=int(bilateral.sum()), amplitude_error=np.nan,
                   amplitude_ratio=np.nan, amplitude_count=0, event_timing_mae_s=np.nan,
                   event_count=0, event_status="insufficient_reference_support")
        # Full-window support is intentionally required: selecting only detected
        # samples could make a failing method appear to preserve amplitude.
        if bilateral.all() and predicted.all() and t >= 8:
            ref = ref_sep - ref_sep.mean()
            out = out_sep - out_sep.mean()
            ref_amp, out_amp = float(np.sqrt(np.mean(ref ** 2))), float(np.sqrt(np.mean(out ** 2)))
            row.update(amplitude_error=abs(out_amp - ref_amp), amplitude_count=t)
            if ref_amp > 1e-8:
                row["amplitude_ratio"] = out_amp / ref_amp
            if ref_amp > 1e-8 and times[i, -1] - times[i, 0] >= 1:
                # Strict positive local maxima define an operational 2D event,
                # not clinical heel strikes/cadence. No elastic alignment.
                peaks = lambda x: np.flatnonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > 0)) + 1
                a, z = peaks(ref), peaks(out)
                row["event_status"] = "insufficient_cycles_or_prediction_event_mismatch"
                if len(a) >= 2 and len(z) == len(a):
                    row.update(event_timing_mae_s=float(np.abs(times[i, a] - times[i, z]).mean()),
                               event_count=len(a), event_status="supported_2d_ankle_maxima")
        rows.append(row)
    return pd.DataFrame(rows)


def _motion_scores(frame, metric):
    required = set(KEYS + STRATA + [metric])
    if not required <= set(frame) or frame[list(required - {metric})].isna().any().any():
        raise ValueError("Metrics lack complete identity, method or evidence fields")
    if frame.duplicated(["method", *KEYS]).any():
        raise ValueError("Duplicate method/window/variant metric")
    # An unsupported window makes its motion unsupported. Silently dropping
    # hard windows would change the scientific population separately by method.
    def complete_mean(series):
        return float(series.mean()) if np.isfinite(series).all() else np.nan
    window = frame.groupby(STRATA + ["person_id", "motion_id", "window_id"], dropna=False)[metric].agg(complete_mean)
    return window.groupby(level=STRATA + ["person_id", "motion_id"]).agg(complete_mean).reset_index()


def aggregate_metrics(frame, metric="visible_nle"):
    """Balance variants within windows, windows within motions, then people.

    Seeds and extractors remain separate. Unsupported motions invalidate that
    stratum's aggregate instead of disappearing from the denominator.
    """
    motion = _motion_scores(frame, metric)
    rows = []
    for strata, group in motion.groupby(STRATA, dropna=False):
        supported = np.isfinite(group[metric])
        people = group.groupby("person_id")[metric].mean()
        rows.append({**dict(zip(STRATA, strata)), "metric": metric,
                     "value": float(people.mean()) if supported.all() else np.nan,
                     "people": int(group.person_id.nunique()), "motions": len(group),
                     "supported_motions": int(supported.sum()),
                     "status": "supported" if supported.all() else "insufficient_evidence"})
    return pd.DataFrame(rows)


def paired_cluster_bootstrap(frame, method, comparator, metric="visible_nle", *,
                             draws=2000, seed=17):
    """Pair exact window identities, resample people and nested motions.

    Rendering variants/windows stay together. One split/extractor/training seed
    per call prevents treating training repeats or architectures as people.
    Returns saved draws and exact matched keys for independent reconstruction.
    """
    if draws < 2:
        raise ValueError("At least two bootstrap draws are required")
    chosen = frame.loc[frame.method.isin([method, comparator])].copy()
    if any(chosen[column].nunique() != 1 for column in ["split", "extractor", "seed", "evidence_status"]):
        raise ValueError("Compare one split, extractor, training seed and evidence status at a time")
    if method == comparator or set(chosen.method) != {method, comparator}:
        raise ValueError("Two distinct observed methods are required")
    _motion_scores(chosen, metric)  # Validate duplicate identities first.
    left = chosen.loc[chosen.method.eq(method)].set_index(KEYS)
    right = chosen.loc[chosen.method.eq(comparator)].set_index(KEYS)
    if set(left.index) != set(right.index):
        raise ValueError("Methods have different evaluation keys; no silent intersection")
    matched = left.index.sort_values()
    if not np.isfinite(left[metric]).all() or not np.isfinite(right[metric]).all():
        return {"status": "insufficient_evidence", "reason": "unsupported_matched_metric",
                "draws": [], "matched_keys": [dict(zip(KEYS, key)) for key in matched]}
    motion = _motion_scores(chosen, metric)
    wide = motion.pivot(index=["person_id", "motion_id"], columns="method", values=metric)
    persons = list(wide.index.get_level_values("person_id").unique())
    if len(persons) < 2:
        return {"status": "insufficient_evidence", "reason": "fewer_than_two_independent_groups",
                "draws": [], "matched_keys": [dict(zip(KEYS, key)) for key in matched]}
    blocks = [wide.loc[person, [method, comparator]].to_numpy(float).reshape(-1, 2) for person in persons]
    point = np.stack([block.mean(axis=0) for block in blocks]).mean(axis=0)
    rng = np.random.default_rng(seed)
    sampled = []
    for draw in range(draws):
        means = []
        for index in rng.integers(0, len(blocks), len(blocks)):
            block = blocks[index]
            means.append(block[rng.integers(0, len(block), len(block))].mean(axis=0))
        candidate, baseline = np.mean(means, axis=0)
        sampled.append({"draw": draw, "candidate": float(candidate), "comparator": float(baseline),
                        "improvement": float(baseline - candidate),
                        "relative_improvement": float((baseline - candidate) / baseline) if baseline > 0 else None})
    absolute = np.array([row["improvement"] for row in sampled])
    relative = [row["relative_improvement"] for row in sampled]
    relative_ci = np.quantile(relative, [0.025, 0.975]).tolist() if all(x is not None for x in relative) else None
    return {"status": "estimated", "metric": metric, "method": method, "comparator": comparator,
            "groups": len(persons), "motions": len(wide), "bootstrap_seed": seed,
            "candidate_mean": float(point[0]), "comparator_mean": float(point[1]),
            "improvement": float(point[1] - point[0]),
            "relative_improvement": float((point[1] - point[0]) / point[1]) if point[1] > 0 else None,
            "ci95": np.quantile(absolute, [0.025, 0.975]).tolist(),
            "relative_ci95": relative_ci, "draws": sampled,
            "matched_keys": [dict(zip(KEYS, key)) for key in matched],
            "limitations": "Conditional on fitted models; training/selection uncertainty is excluded. Person IDs must represent audited independent groups."}


def scientific_gate(*, evidence_status, relative_improvement=None, ci95=None,
                    meaningful_margin=0.02, preservation_ok=None,
                    clean_retention_ok=None, complete=True):
    """Apply an explicitly supplied development margin, never promote fixtures.

    ci95 is the paired interval for relative improvement, not absolute NLE.
    Preservation/retention decisions must come from separately specified
    supported endpoints; absent evidence is not an automatic pass.
    """
    result = {"status": "insufficient_evidence", "evidence_status": evidence_status,
              "meaningful_margin": meaningful_margin}
    if evidence_status not in {"source-run", "real-development", "confirmed"}:
        return {**result, "reason": "fixture_or_planned_evidence_cannot_authorize_scientific_advance"}
    if not complete or relative_improvement is None or ci95 is None or preservation_ok is None or clean_retention_ok is None:
        return {**result, "reason": "missing_required_measurements"}
    if not np.isfinite([relative_improvement, *ci95, meaningful_margin]).all() or len(ci95) != 2 or ci95[0] > ci95[1] or meaningful_margin < 0:
        raise ValueError("Finite effect, ordered two-sided interval and nonnegative margin required")
    if not preservation_ok or not clean_retention_ok:
        return {**result, "status": "fail", "reason": "preservation_or_clean_retention_failed"}
    if relative_improvement < meaningful_margin or ci95[1] <= 0:
        return {**result, "status": "fail", "reason": "practical_benefit_rule_failed"}
    if ci95[0] <= 0:
        return {**result, "reason": "uncertain_improvement"}
    return {**result, "status": "pass", "reason": "practical_benefit_and_preservation_rules_satisfied"}
