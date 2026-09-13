"""Event fidelity, calibration-locked comparisons and person-level uncertainty."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .body_geometry import descriptor


MIN_EVENT_DELTA = {"arm_leg_timing": .01, "foot_clearance": .002, "trunk_pelvis_timing": .01}


def score_case(prediction, case, metadata, method, strength=1.0, probability=np.nan, threshold=.5):
    """Retention is not clipped. Missing observations still have reference targets."""
    family = metadata["event_family"]
    raw, truth, clean, event = (case[k] for k in ("raw", "truth", "clean", "event_reference"))
    d, d_truth, d_clean, d_event = [descriptor(x, family) for x in (prediction, truth, clean, event)]
    magnitude = d_event-d_clean
    mse = float(np.mean(np.sum((prediction-truth)**2, axis=-1)))
    raw_mse = float(np.mean(np.sum((raw-truth)**2, axis=-1)))
    present = bool(metadata["event_present"])
    eligible = present and abs(magnitude) >= MIN_EVENT_DELTA[family]
    probability = float(probability)
    return {
        **{k: metadata[k] for k in ("case_id", "person_id", "role", "event_family", "fixture", "event_present", "noise_present")},
        "method": method, "strength": float(strength), "mse_m2": mse, "raw_mse_m2": raw_mse,
        "noise_removal": 1-mse/raw_mse if raw_mse > 1e-10 else np.nan,
        "retention": 1-abs(d-d_event)/abs(magnitude) if eligible else np.nan,
        "descriptor_abs_error": abs(d-d_truth), "descriptor_signed_error": d-d_truth,
        "event_magnitude": magnitude, "event_eligible": eligible,
        "signed_event_error": (d-d_event)/magnitude if eligible else np.nan,
        "probability": probability,
        "brier": (probability-float(present))**2 if np.isfinite(probability) else np.nan,
        "decided": float(max(probability, 1-probability) >= threshold) if np.isfinite(probability) else np.nan,
        "correct": float((probability >= .5) == present) if np.isfinite(probability) else np.nan,
    }


def _primary(rows):
    # Same-clip event+noise retention and noise repair across both noisy factorial
    # groups. Matched and full-input-ambiguous fixtures are separate diagnostics.
    return rows.loc[rows.fixture.eq("factorial") & rows.noise_present.astype(bool)]


def summarize(scores):
    records = []
    for method, rows in scores.groupby("method", sort=False):
        primary = _primary(rows)
        person = primary.groupby("person_id")[["retention", "noise_removal", "descriptor_abs_error", "mse_m2"]].mean()
        record = dict(method=method, **person.mean().to_dict(), n_people=len(person))
        both=primary.loc[primary.event_present.astype(bool)]
        record["event_and_noise_removal"]=both.groupby("person_id").noise_removal.mean().mean()
        natural=rows.loc[rows.fixture.eq("factorial") & ~rows.event_present.astype(bool)]
        record["unedited_motion_descriptor_error"]=natural.groupby("person_id").descriptor_abs_error.mean().mean()
        probability = rows.loc[~rows.fixture.eq("ambiguous")]
        # Weight people equally, matching the motion metrics and uncertainty
        # unit; prolific participants should not dominate calibration reports.
        calibration = probability.groupby("person_id")[["decided", "brier"]].mean().mean()
        selected = probability.loc[probability.decided.eq(1)].groupby("person_id")["correct"].mean().mean()
        ambiguous = rows.loc[rows.fixture.eq("ambiguous")].groupby("person_id")[["brier", "decided"]].mean().mean()
        record.update(coverage=calibration.decided, brier=calibration.brier,
                      selective_accuracy=selected,
                      ambiguous_brier=ambiguous.brier, ambiguous_coverage=ambiguous.decided)
        records.append(record)
    return pd.DataFrame(records)


def lock_operating_points(calibration_scores, target=.25):
    """Select by calibration repair only, without looking at held-out outcomes."""
    if not set(calibration_scores.role) <= {"calibration"}:
        raise ValueError("Operating points must use calibration people only")
    records = []
    for method, rows in calibration_scores.groupby("method", sort=False):
        options = []
        for strength, subset in rows.groupby("strength"):
            mean = _primary(subset).groupby("person_id")[["noise_removal", "retention"]].mean().mean()
            options.append(dict(method=method, strength=float(strength), calibration_noise_removal=mean.noise_removal,
                                calibration_retention=mean.retention))
        table = pd.DataFrame(options)
        feasible = table.loc[table.calibration_noise_removal.ge(target)]
        pool = feasible if len(feasible) else table
        distance = (pool.calibration_noise_removal-target).abs().fillna(np.inf)
        chosen = pool.loc[distance.idxmin()].to_dict()
        chosen["feasible"] = bool(len(feasible))
        records.append(chosen)
    return pd.DataFrame(records)


def calibrate_probability(logits, labels):
    """Two-parameter Platt fit on calibration cases, with bounded coefficients."""
    logits, labels = np.asarray(logits, float), np.asarray(labels, float)
    def objective(ab):
        z = ab[0]*logits + ab[1]
        return float(np.mean(np.logaddexp(0, z)-labels*z) + .001*np.sum(np.asarray(ab)**2))
    result = minimize(objective, [1., 0.], bounds=[(-10., 10.), (-10., 10.)])
    a, b = result.x
    probability = expit(a*logits+b)
    confidence, correct = np.maximum(probability, 1-probability), (probability >= .5) == labels
    # Calibration-only abstention, requiring at least ten examples. All tests
    # report achieved error and coverage; no promised calibration on GAVD.
    threshold = 1.01
    for candidate in np.linspace(.5, .99, 50):
        mask = confidence >= candidate
        if mask.sum() >= 10 and correct[mask].mean() >= .9:
            threshold = float(candidate)
            break
    return {"a": float(a), "b": float(b), "threshold": threshold}


def paired_person_interval(scores, first, second, metric="retention", samples=1000, seed=17):
    eligible = _primary(scores)
    table = eligible.groupby(["person_id", "method"])[metric].mean().unstack()
    if first not in table or second not in table:
        return {"mean": np.nan, "low": np.nan, "high": np.nan, "n_people": 0}
    delta = (table[first]-table[second]).dropna().to_numpy()
    if len(delta) < 2:
        return {"mean": float(np.mean(delta)) if len(delta) else np.nan, "low": np.nan, "high": np.nan, "n_people": len(delta)}
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(samples, len(delta)), replace=True).mean(axis=1)
    return dict(mean=float(delta.mean()), low=float(np.quantile(draws, .025)),
                high=float(np.quantile(draws, .975)), n_people=len(delta))
