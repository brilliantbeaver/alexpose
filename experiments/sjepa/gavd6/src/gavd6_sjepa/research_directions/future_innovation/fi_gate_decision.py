"""Preregistered validity -> point thresholds -> stability decision."""

from dataclasses import asdict
from math import isfinite
from numbers import Real

from .fi_contracts import GateThresholds


def decide_gate(metrics, thresholds=None, expected_seed_count=3):
    thresholds = thresholds or GateThresholds()
    numeric = [
        "delta_r2_real",
        "delta_r2_time_shuffle",
        "delta_r2_clip_mismatch",
        "delta_r2_background_target",
        "delta_r2_no_skeleton",
        "motion_to_background_change_ratio",
        "person_edit_direction_fraction",
        "bootstrap_positive_fraction",
    ]
    seed_gains = metrics.get("seed_real_gains", [])
    numbers = [metrics.get(key) for key in numeric]
    valid = (
        isinstance(seed_gains, list)
        and len(seed_gains) == expected_seed_count
        and all(
            isinstance(v, Real) and not isinstance(v, bool) and isfinite(v)
            for v in numbers + seed_gains
        )
    )
    if valid:
        valid = (
            0 <= metrics["person_edit_direction_fraction"] <= 1
            and 0 <= metrics["bootstrap_positive_fraction"] <= 1
            and metrics["motion_to_background_change_ratio"] >= 0
            and abs(sum(seed_gains) / len(seed_gains) - metrics["delta_r2_real"])
            < 1e-10
        )
    if not valid:
        return {
            "decision": "STOP",
            "allow_full_experiment": False,
            "allow_adapter_training": False,
            "checks": {"complete_finite_metrics": False},
            "thresholds": asdict(thresholds),
            "metrics": None,
        }
    real = metrics["delta_r2_real"]
    validity = {
        name: metrics.get(name) is True
        for name in (
            "data_contract_valid",
            "evaluation_contract_valid",
            "controls_complete",
            "target_audit_complete",
            "target_variance_valid",
            "teacher_stable",
            "causal_leakage_absent",
        )
    }
    points = {
        "real_gain": real >= thresholds.real_delta_r2_min,
        "time_shuffle": real
        >= thresholds.real_to_shuffle_min * max(metrics["delta_r2_time_shuffle"], 0),
        "clip_mismatch": metrics["delta_r2_clip_mismatch"]
        <= thresholds.mismatch_delta_r2_max,
        "person_region": 1
        - max(metrics["delta_r2_background_target"], 0) / max(real, 1e-8)
        >= thresholds.person_ablation_reduction_min,
        "target_sensitivity": metrics["motion_to_background_change_ratio"]
        >= thresholds.motion_to_background_change_min,
        "person_edit_consistency": metrics["person_edit_direction_fraction"]
        >= thresholds.person_edit_direction_fraction_min,
    }
    stability = {
        "bootstrap_positive": metrics["bootstrap_positive_fraction"]
        >= thresholds.bootstrap_positive_fraction_min,
        "seeds_stable": all(gain > 0 for gain in seed_gains)
        and sum(gain >= thresholds.real_delta_r2_min for gain in seed_gains) >= 2,
    }
    decision = (
        "STOP"
        if not all(validity.values()) or not all(points.values())
        else ("INCONCLUSIVE" if not all(stability.values()) else "ADVANCE")
    )
    return {
        "decision": decision,
        "allow_full_experiment": decision == "ADVANCE",
        "allow_adapter_training": False,
        "checks": {**validity, **points, **stability},
        "thresholds": asdict(thresholds),
        "metrics": metrics,
    }
