"""Preregistered validity -> point thresholds -> stability decision."""

from dataclasses import asdict
from math import isfinite
from numbers import Real

from .fi_contracts import GateThresholds, DIRECT_PROTOCOL, LEGACY_PROTOCOL


def decide_gate(metrics, thresholds=None, expected_seed_count=3, protocol=LEGACY_PROTOCOL):
    thresholds = thresholds or GateThresholds()
    if protocol == DIRECT_PROTOCOL:
        return decide_direct_gate(metrics, thresholds, expected_seed_count)
    if protocol != LEGACY_PROTOCOL:
        raise ValueError("Unknown gate protocol")
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


def decide_direct_gate(metrics, thresholds, expected_seed_count):
    """Decide on a JEPA training comparison from the measured skeleton increment."""
    threshold_record = {name: value for name, value in asdict(thresholds).items()
                        if name not in {"person_ablation_reduction_min", "motion_to_background_change_min",
                                        "person_edit_direction_fraction_min"}}
    keys = ("delta_r2_real", "delta_r2_time_shuffle", "delta_r2_clip_mismatch",
            "delta_r2_no_skeleton", "bootstrap_positive_fraction", "skeleton_increment_positive_fraction")
    gains, increments = metrics.get("seed_real_gains"), metrics.get("seed_skeleton_increments")
    valid = (isinstance(gains, list) and isinstance(increments, list)
             and len(gains) == len(increments) == expected_seed_count
             and all(isinstance(v, Real) and not isinstance(v, bool) and isfinite(v)
                     for v in [*(metrics.get(k) for k in keys), *(gains or []), *(increments or [])]))
    if valid:
        increment = metrics["delta_r2_real"] - metrics["delta_r2_no_skeleton"]
        valid = (abs(sum(gains) / len(gains) - metrics["delta_r2_real"]) < 1e-10
                 and abs(sum(increments) / len(increments) - increment) < 1e-10
                 and 0 <= metrics["bootstrap_positive_fraction"] <= 1
                 and 0 <= metrics["skeleton_increment_positive_fraction"] <= 1)
    result = {"protocol": DIRECT_PROTOCOL, "decision": "STOP", "allow_full_experiment": False,
              "allow_jepa_training_comparison": False, "allow_adapter_training": False,
              "next_action": "inspect_failed_comparisons", "thresholds": threshold_record,
              "metrics": metrics if valid else None}
    if not valid:
        return {**result, "checks": {"complete_finite_metrics": False}, "next_action": "repair_missing_or_invalid_evidence"}
    validity = {name: metrics.get(name) is True for name in (
        "data_contract_valid", "evaluation_contract_valid", "controls_complete", "input_audit_complete",
        "target_variance_valid", "teacher_stable", "causal_leakage_absent")}
    real = metrics["delta_r2_real"]
    points = {"real_gain": real >= thresholds.real_delta_r2_min,
              "time_shuffle": real >= thresholds.real_to_shuffle_min * max(metrics["delta_r2_time_shuffle"], 0),
              "clip_mismatch": metrics["delta_r2_clip_mismatch"] <= thresholds.mismatch_delta_r2_max,
              "skeleton_increment": increment > 0}
    stability = {"bootstrap_positive": metrics["bootstrap_positive_fraction"] >= thresholds.bootstrap_positive_fraction_min,
                 "seeds_stable": all(g > 0 for g in gains) and sum(g >= thresholds.real_delta_r2_min for g in gains) >= 2,
                 "skeleton_increment_stable": (metrics["skeleton_increment_positive_fraction"] >= thresholds.bootstrap_positive_fraction_min
                                                and all(g > 0 for g in increments))}
    decision = "STOP" if not all(validity.values()) or not all(points.values()) else (
        "ADVANCE" if all(stability.values()) else "INCONCLUSIVE")
    result.update(decision=decision, checks={**validity, **points, **stability},
                  allow_full_experiment=decision == "ADVANCE",
                  allow_jepa_training_comparison=decision == "ADVANCE",
                  next_action={"ADVANCE": "design_full_gavd_jepa_comparison", "STOP": "inspect_failed_comparisons",
                               "INCONCLUSIVE": "plan_independent_confirmation"}[decision])
    return result
