"""Exact symmetry and temporal-observability teaching fixtures.

This module is a calibration, not a new GAVD or teacher-encoding experiment.
The choices below are fixed before outcomes are computed: NumPy seed 4107;
48 generated sources with two opposite-order clips each; sources 0--35 train
and 36--47 test; float64 arithmetic; ridge penalty 0.001 with source weights
normalized to the number of training clips; 1e-12 group/parity tolerance and
1e-8 numerical-invariance tolerance. The signed-order readout must achieve
R² > 0.99; order-even and support-only readouts must have |R²| < 1e-8.

Reflection exchanges anatomical sides and reverses centered horizontal x.
Time reversal reverses observations AND their physical inter-observation
intervals. It is a diagnostic transformation, not a physical reversibility
assumption or permission to use future observations in a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
import hashlib
import json

import numpy as np

from ..future_innovation.fi_joint_models import JointRidge, SupportedInput


CALIBRATION_VERSION = "reflection-time-observability-v1"
SEED = 4107
ALGEBRA_ATOL = 1e-12
INVARIANCE_ATOL = 1e-8
RIDGE_PENALTY = 0.001
FULL_MIRROR_PAIRS = (
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
    (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
    (27, 28), (29, 30), (31, 32),
)
OBSERVABLE_PAIRS = ((11, 12), (25, 26), (27, 28), (29, 30), (31, 32))
REFLECTION_INDEX = np.arange(33)
for _left, _right in FULL_MIRROR_PAIRS:
    REFLECTION_INDEX[[_left, _right]] = [_right, _left]
REFLECTION_INDEX.setflags(write=False)


@dataclass(frozen=True)
class PoseHistory:
    """One timed, already centered skeleton sequence with explicit support.

    No centering, interpolation or confidence threshold is learned here. The
    supplied validity is authoritative; finite observed coordinates and all
    confidence values are validated. Invalid coordinates may contain sentinels.
    Arrays are copied and made read-only so transformations cannot alter input.
    """

    xyz: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    times: np.ndarray

    def __post_init__(self):
        xyz = np.array(self.xyz, dtype=np.float64, copy=True)
        confidence = np.array(self.confidence, dtype=np.float64, copy=True)
        valid = np.array(self.valid, copy=True)
        times = np.array(self.times, dtype=np.float64, copy=True)
        if xyz.ndim != 3 or xyz.shape[1:] != (33, 3) or len(xyz) < 2:
            raise ValueError("Coordinates must have shape [T >= 2, 33, 3]")
        if valid.shape != xyz.shape[:2] or valid.dtype != bool:
            raise ValueError("Validity must be boolean with shape [T, 33]")
        if confidence.shape != valid.shape or not np.isfinite(confidence).all():
            raise ValueError("Confidence must be finite with shape [T, 33]")
        if np.any(confidence < 0) or np.any(confidence > 1):
            raise ValueError("Confidence must be between zero and one")
        if not np.isfinite(xyz[valid]).all():
            raise ValueError("Every observed coordinate must be finite")
        if (times.shape != (len(xyz),) or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0)):
            raise ValueError("Timestamps must be finite and strictly increasing")
        for name, array in (("xyz", xyz), ("confidence", confidence),
                            ("valid", valid), ("times", times)):
            array.setflags(write=False)
            object.__setattr__(self, name, array)


def reflect(history: PoseHistory) -> PoseHistory:
    """Anatomical M: swap all side channels and flip centered horizontal x."""
    xyz = history.xyz[:, REFLECTION_INDEX].copy()
    xyz[..., 0] *= -1
    return PoseHistory(xyz, history.confidence[:, REFLECTION_INDEX],
                       history.valid[:, REFLECTION_INDEX], history.times)


def reverse_time(history: PoseHistory) -> PoseHistory:
    """Reverse sequence and intervals while retaining increasing timestamps.

    t'_i = t_0 + (t_last - t_(last-i)); absolute origin and duration are kept.
    Keeping the original unequal intervals in their original order would change
    speeds and would invalidate the median-speed time-even claim.
    """
    times = history.times[0] + (history.times[-1] - history.times[::-1])
    return PoseHistory(history.xyz[::-1], history.confidence[::-1],
                       history.valid[::-1], times)


def motion_observables(history: PoseHistory, *, minimum_transitions: int = 8) -> dict:
    """Four scalar examples with known reflection/time-reversal parity.

    Speed uses only adjacent, common-valid left/right observations divided by
    their actual dt. Vertical displacement sums observed adjacent displacements;
    with gaps it is an observed-transition total, not an imputed endpoint shift.
    All five pairs require support. These are coordinate observables, not clinical
    scores. The signed vertical channel is image-coordinate y, not gravity.
    """
    if not isinstance(minimum_transitions, int) or minimum_transitions < 1:
        raise ValueError("minimum_transitions must be a positive integer")
    values, counts = [], []
    dt = np.diff(history.times)
    for left, right in OBSERVABLE_PAIRS:
        common = (history.valid[:-1, left] & history.valid[1:, left]
                  & history.valid[:-1, right] & history.valid[1:, right])
        counts.append(int(common.sum()))
        if common.sum() < minimum_transitions:
            raise ValueError(f"Insufficient common observed transitions for pair {left, right}")
        # Index before subtracting so invalid NaN/large sentinels never enter arithmetic.
        index = np.flatnonzero(common)
        left_delta = history.xyz[index + 1, left] - history.xyz[index, left]
        right_delta = history.xyz[index + 1, right] - history.xyz[index, right]
        ml = np.median(np.linalg.norm(left_delta, axis=1) / dt[index])
        mr = np.median(np.linalg.norm(right_delta, axis=1) / dt[index])
        dl, dr = left_delta[:, 1].sum(), right_delta[:, 1].sum()
        values.append(((ml + mr) / 2, (ml - mr) / (ml + mr + 1e-8),
                       (dl + dr) / 2, (dl - dr) / (abs(dl) + abs(dr) + 1e-8)))
    names = ("total_speed", "bilateral_speed_contrast", "mean_y_displacement",
             "bilateral_y_displacement_contrast")
    if not np.isfinite(values).all():
        raise ValueError("Motion arithmetic produced non-finite values")
    return {**dict(zip(names, map(float, np.mean(values, axis=0)))),
            "common_transition_counts": counts}


def parity_project(
    feature: Callable[[PoseHistory], np.ndarray], history: PoseHistory,
    reflection_sign: int, time_sign: int,
) -> np.ndarray:
    """Project an arbitrary common-basis feature onto one of four parities.

    P_ab h(x) = [h(x) + a h(Mx) + b h(Tx) + ab h(MTx)] / 4.
    M and T commute and square to identity, so P_ab h(Mx)=a P_ab h(x)
    and P_ab h(Tx)=b P_ab h(x). This is algebra, not learned equivariance.
    A token-valued feature must be aligned externally into a declared common
    basis first. This routine cannot infer teacher-token correspondence.
    """
    if reflection_sign not in (-1, 1) or time_sign not in (-1, 1):
        raise ValueError("Parity signs must each be -1 or +1")
    histories = (history, reflect(history), reverse_time(history),
                 reflect(reverse_time(history)))
    values = [np.asarray(feature(item), dtype=np.float64) for item in histories]
    if len({v.shape for v in values}) != 1 or not all(np.isfinite(v).all() for v in values):
        raise ValueError("Orbit features must have one common finite shape and basis")
    a, b = reflection_sign, time_sign
    result = values[0] / 4 + a * values[1] / 4 + b * values[2] / 4 + a * b * values[3] / 4
    if not np.isfinite(result).all():
        raise ValueError("Projected feature arithmetic produced non-finite values")
    return result


def _array_digest(values: np.ndarray) -> str:
    v = np.ascontiguousarray(values, dtype="<f8")
    return hashlib.sha256(str(v.shape).encode() + v.tobytes()).hexdigest()


def paired_teacher_components(
    original: np.ndarray, mirrored: np.ndarray, *, window_ids: Sequence[str],
    original_evidence_id: str, mirrored_evidence_id: str, shared_basis_id: str,
    evidence_kind: str = "synthetic_teaching",
) -> dict:
    """Even/odd components from TWO supplied pooled teacher feature arrays.

    Requires explicit row identities and separate evidence identifiers. This
    validates arithmetic and array schemas; it does not encode video or verify
    external evidence files. For real data, the caller must verify two actual
    encoder outputs from original/reflected video with matching teacher,
    projection, boundaries and row order. Existing pooled cache vectors cannot
    supply the reflected branch by swapping skeleton columns or changing signs.
    Raw spatial tokens are rejected: they require a separate alignment contract.
    """
    a, b = np.asarray(original, dtype=np.float64), np.asarray(mirrored, dtype=np.float64)
    if (a.ndim != 2 or a.shape != b.shape or min(a.shape) < 1
            or not np.isfinite(a).all() or not np.isfinite(b).all()):
        raise ValueError("Supply two finite, matching nonempty [windows, pooled features] arrays")
    ids = tuple(map(str, window_ids))
    if len(ids) != len(a) or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("Unique ordered window identities must match feature rows")
    if (not original_evidence_id or not mirrored_evidence_id
            or original_evidence_id == mirrored_evidence_id or not shared_basis_id):
        raise ValueError("Separate original/reflected evidence IDs and shared basis ID are required")
    if evidence_kind not in ("synthetic_teaching", "encoded_video"):
        raise ValueError("evidence_kind must explicitly identify synthetic or encoded-video arrays")
    return {"even": a / 2 + b / 2, "odd": a / 2 - b / 2,
            "provenance": {"window_ids": list(ids), "original_evidence_id": original_evidence_id,
                           "mirrored_evidence_id": mirrored_evidence_id, "shared_basis_id": shared_basis_id,
                           "evidence_kind": evidence_kind, "original_array_sha256": _array_digest(a),
                           "mirrored_array_sha256": _array_digest(b), "representation": "pooled_common_basis",
                           "external_lineage_verified_by_this_function": False}}


def _algebra_fixture() -> PoseHistory:
    # Dyadic irregular intervals make the algebra directly representable in float64.
    intervals = np.array([1, 2, 1, 3, 2, 1, 1, 2, 3, 1, 2, 1, 1, 2, 1, 3]) / 8
    times = np.r_[0.0, np.cumsum(intervals)]
    xyz = np.zeros((len(times), 33, 3))
    xyz[..., 0] = (np.arange(33) - 16)[None] / 32
    for index, (left, right) in enumerate(OBSERVABLE_PAIRS):
        xyz[:, left, 1] = (index + 2) * times
        xyz[:, right, 1] = (index + 1) * times
    rng = np.random.default_rng(SEED)
    confidence = rng.uniform(0.6, 1.0, size=xyz.shape[:2])
    return PoseHistory(xyz, confidence, np.ones(xyz.shape[:2], bool), times)


def _max_history_error(a: PoseHistory, b: PoseHistory) -> float:
    if not np.array_equal(a.valid, b.valid):
        return float("inf")
    return float(max(np.max(np.abs(a.xyz - b.xyz)),
                     np.max(np.abs(a.confidence - b.confidence)),
                     np.max(np.abs(a.times - b.times))))


def temporal_observability_fixture() -> dict:
    """Source-held proof that order-even summaries can miss a signed target.

    Each source contributes a path and its reversed path. Both end at the same
    current posture, have the same coordinate distribution and support, yet
    their last observed y velocities have opposite signs. A declared synthetic
    continuation moves one more step at that last velocity. This is not an
    empirical assertion that human motion follows constant velocity.
    """
    rng = np.random.default_rng(SEED)
    wave = np.array([0, 1, 2, 2, 1, 0, -1, -2, -2, -1, 0], dtype=float)
    times = np.arange(len(wave)) / 8
    histories, source_ids, clip_ids, targets = [], [], [], []
    for source in range(48):
        amplitude = rng.uniform(0.15, 0.5)
        offset = rng.uniform(-0.4, 0.4)
        confidence = np.full((len(wave), 33), rng.uniform(0.7, 1.0))
        valid = np.ones((len(wave), 33), bool)
        # An optional, time-symmetric interior gap changes support across sources
        # but cannot reveal direction within a source. Endpoints remain observed.
        if source % 3 == 0:
            valid[[3, 7], 27] = False
        xyz = np.zeros((len(wave), 33, 3))
        xyz[..., 0] = (np.arange(33) - 16)[None] / 32
        xyz[..., 1] = offset + amplitude * wave[:, None]
        original = PoseHistory(xyz, confidence, valid, times)
        for direction, history in ((1, original), (-1, reverse_time(original))):
            histories.append(history)
            source_ids.append(f"synthetic_source_{source:02d}")
            clip_ids.append(f"synthetic_source_{source:02d}_direction_{direction:+d}")
            # Exact last observed vertical velocity is the defined scalar target.
            targets.append(direction * amplitude * 8)
    sources = np.asarray(source_ids)
    rows = np.arange(len(sources))
    train, test = rows[:72], rows[72:]
    assert not set(sources[train]) & set(sources[test])
    even_rows, ordered_rows, support_rows, current_rows = [], [], [], []
    for history in histories:
        y = history.xyz[:, 27, 1]
        valid = history.valid[:, 27]
        adjacent = valid[:-1] & valid[1:]
        delta = np.diff(y)
        mean = y[valid].mean()
        even_rows.append([mean, y[valid].std(), np.abs(delta[adjacent]).mean(),
                          valid.mean(), adjacent.mean(), history.confidence[:, 27].mean()])
        ordered_rows.append(np.where(adjacent, delta / np.diff(history.times), np.nan))
        support_rows.append([valid.mean(), adjacent.mean(), history.confidence[:, 27].mean()])
        current_rows.append([y[-1], history.xyz[-1, 27, 0]])
    features = {"current_posture": np.asarray(current_rows),
                "order_even_mean_sd_absolute_change_support": np.asarray(even_rows),
                "support_only": np.asarray(support_rows),
                "ordered_signed_velocity": np.asarray(ordered_rows)}
    y = np.asarray(targets)[:, None]
    # Source pairs have opposite outcomes and identical order-even features.
    pair_difference = float(np.max(np.abs(features["order_even_mean_sd_absolute_change_support"][::2]
                                          - features["order_even_mean_sd_absolute_change_support"][1::2])))
    weights = np.ones(len(train))  # Two clips per source: normalized source-balanced weights.
    reference = np.average(y[train], axis=0, weights=weights)
    denominator = float(np.mean((y[test] - reference) ** 2))
    scores = []
    for name, matrix in features.items():
        schema = tuple(f"{name}_{i}" for i in range(matrix.shape[1]))
        if name == "ordered_signed_velocity":
            kinds = ("velocity",) * matrix.shape[1]
        elif name == "order_even_mean_sd_absolute_change_support":
            kinds = ("coordinate",) * 3 + ("fraction",) * 3
        elif name == "support_only":
            kinds = ("fraction",) * matrix.shape[1]
        else:
            kinds = ("coordinate",) * matrix.shape[1]
        scaler = SupportedInput.fit(matrix[train], weights, np.asarray(clip_ids)[train],
                                    sources[train], schema, kinds)
        model = JointRidge.fit(scaler.transform(matrix[train]), None, y[train], weights, RIDGE_PENALTY)
        prediction = model.predict(scaler.transform(matrix[test]))
        mse = float(np.mean((y[test] - prediction) ** 2))
        scores.append({"features": name, "r2_training_mean_reference": 1 - mse / denominator,
                       "mse": mse, "nominal_features": matrix.shape[1],
                       "supported_features": int(scaler.mask.sum()),
                       "test_predictions": prediction[:, 0].tolist(),
                       "preprocessing": scaler.record(),
                       "fit_sources": list(dict.fromkeys(scaler.training_video_ids)),
                       "fit_clip_ids": list(scaler.training_window_ids)})
    by_name = {r["features"]: r["r2_training_mean_reference"] for r in scores}
    assert pair_difference < ALGEBRA_ATOL
    assert by_name["ordered_signed_velocity"] > 0.99
    assert all(abs(value) < INVARIANCE_ATOL for name, value in by_name.items()
               if name != "ordered_signed_velocity")
    return {"evidence": "synthetic calibration only", "seed": SEED,
            "sources": 48, "clips": 96, "train_sources": list(dict.fromkeys(sources[train])),
            "test_sources": list(dict.fromkeys(sources[test])), "train_clips": len(train),
            "test_clips": len(test), "source_overlap": 0,
            "test_clip_ids": np.asarray(clip_ids)[test].tolist(),
            "test_source_ids": sources[test].tolist(), "test_targets": y[test, 0].tolist(),
            "training_target_mean": float(reference[0]),
            "target": "one-step synthetic continuation displacement rate = last observed y velocity",
            "time_units": "seconds", "score_reference": "outer-training target mean",
            "source_weight_normalization": "equal source totals; sum of fit weights = training clips",
            "ridge_penalty": RIDGE_PENALTY, "selection": "fixed fixture setting; no search or outer selection",
            "max_order_even_pair_difference": pair_difference, "scores": scores,
            "illustrative_wave": wave.tolist(), "illustrative_times": times.tolist(),
            "claim_limit": "This validates readout observability on generated paths; no encoder training or GAVD forecasting occurred."}


def run_calibration(output: str | Path | None = None) -> dict:
    """Run deterministic exact/statistical teaching checks; optionally save new JSON.

    Existing output files are rejected to prevent silently overwriting evidence.
    With no path this function has no filesystem side effects.
    """
    history = _algebra_fixture()
    m, t = reflect(history), reverse_time(history)
    group_errors = {"M_squared": _max_history_error(reflect(m), history),
                    "T_squared": _max_history_error(reverse_time(t), history),
                    "M_T_commute": _max_history_error(reflect(t), reverse_time(m)),
                    "reversed_interval_error": float(np.max(np.abs(np.diff(t.times)
                                                                           - np.diff(history.times)[::-1])))}
    assert max(group_errors.values()) <= ALGEBRA_ATOL
    expected = {"total_speed": (1, 1), "bilateral_speed_contrast": (-1, 1),
                "mean_y_displacement": (1, -1), "bilateral_y_displacement_contrast": (-1, -1)}
    observed = [motion_observables(h) for h in (history, m, t, reflect(t))]
    observable_rows = []
    for name, (a, b) in expected.items():
        values = [v[name] for v in observed]
        error = float(np.max(np.abs(np.asarray(values) - values[0] * np.array([1, a, b, a*b]))))
        assert abs(values[0]) > 0
        assert error <= ALGEBRA_ATOL
        observable_rows.append({"observable": name, "reflection_sign": a, "time_sign": b,
                                "original": values[0], "reflected": values[1],
                                "reversed": values[2], "reflected_reversed": values[3],
                                "maximum_parity_error": error})
    # An arbitrary nonlinear feature deliberately mixes all four observable lanes.
    def feature(h):
        z = motion_observables(h)
        v = np.array([z[name] for name in expected])
        return np.r_[v + 0.3, v**2 + v[::-1]]
    projector_rows = []
    for a in (1, -1):
        for b in (1, -1):
            z = parity_project(feature, history, a, b)
            error = max(np.max(np.abs(parity_project(feature, m, a, b) - a*z)),
                        np.max(np.abs(parity_project(feature, t, a, b) - b*z)))
            assert error <= ALGEBRA_ATOL
            projector_rows.append({"reflection_sign": a, "time_sign": b,
                                   "maximum_covariance_error": float(error),
                                   "projected_norm": float(np.linalg.norm(z))})
    zero = lambda h: np.zeros(4)
    zero_error = max(float(np.abs(parity_project(zero, history, a, b)).max())
                     for a in (1, -1) for b in (1, -1))
    rng = np.random.default_rng(SEED)
    original, mirrored = rng.normal(size=(6, 4)), rng.normal(size=(6, 4))
    arguments = dict(window_ids=[f"synthetic_{i}" for i in range(6)],
                     original_evidence_id="synthetic-original-seed4107",
                     mirrored_evidence_id="synthetic-mirrored-seed4107", shared_basis_id="synthetic-four-features")
    components = paired_teacher_components(original, mirrored, **arguments)
    swapped = paired_teacher_components(mirrored, original, **{
        **arguments, "original_evidence_id": arguments["mirrored_evidence_id"],
        "mirrored_evidence_id": arguments["original_evidence_id"]})
    decomposition_error = float(max(np.max(np.abs(components["even"] + components["odd"] - original)),
                                    np.max(np.abs(components["even"] - components["odd"] - mirrored)),
                                    np.max(np.abs(swapped["even"] - components["even"])),
                                    np.max(np.abs(swapped["odd"] + components["odd"]))))
    assert decomposition_error <= ALGEBRA_ATOL
    temporal = temporal_observability_fixture()
    zero_target = np.asarray(temporal["test_targets"])
    zero_r2 = 1 - np.mean(zero_target**2) / np.mean((zero_target - temporal["training_target_mean"])**2)
    result = {"version": CALIBRATION_VERSION, "status": "passed", "seed": SEED,
              "evidence_type": "synthetic mathematical and software calibration",
              "tolerances": {"algebra_absolute": ALGEBRA_ATOL, "zero_gain_absolute": INVARIANCE_ATOL,
                             "signed_order_r2_lower_bound": 0.99},
              "group_errors": group_errors, "observable_parities": observable_rows,
              "projector_checks": projector_rows,
              "zero_feature_control": {"symmetry_error": zero_error, "feature_variance": 0.0,
                                       "prediction_r2_on_balanced_nonzero_target": float(zero_r2),
                                       "interpretation": "Exact transformation consistency permits zero predictive information."},
              "teacher_pair_check": {"maximum_decomposition_error": decomposition_error,
                                     "provenance": components["provenance"],
                                     "real_transformed_teacher_cache_available": False},
              "temporal_observability": temporal}
    # One JSON-native return schema for direct notebook use and serialized reload.
    result = json.loads(json.dumps(result, allow_nan=False))
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return result
