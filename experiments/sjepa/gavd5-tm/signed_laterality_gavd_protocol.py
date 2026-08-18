"""Frozen protocol primitives for the GAVD signed-laterality audit.

This module deliberately implements an *audit*, not a clinical classifier.
GAVD rows are source-video grouped, the historical encoder may have seen them,
and the signed target is derived from the input coordinates.  Callers must
carry those limitations into every result they write.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - documentary environment only
    torch = None
    nn = object


CONDITIONS = ("normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy")
BLAZEPOSE_33 = (
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER", "RIGHT_EYE_INNER", "RIGHT_EYE",
    "RIGHT_EYE_OUTER", "LEFT_EAR", "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST", "LEFT_PINKY",
    "RIGHT_PINKY", "LEFT_INDEX", "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL", "RIGHT_HEEL",
    "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
)
TARGET_PAIRS = ((11, 12), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32))
MIRROR_PAIRS = ((1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                (27, 28), (29, 30), (31, 32))
AUDIT_LABELS = {
    "evidence_level": "internal code and representation audit",
    "evaluation": "transductive",
    "split_unit": "source-video-grouped",
    "target": "signed coordinate excursion (right minus left)",
    "checkpoint": "historically exposed hybrid JEPA checkpoint",
}


@dataclass(frozen=True)
class AuditConfig:
    """Versioned choices that cannot be inferred from an output file."""

    artifact_root: str | None
    mode: Literal["real", "smoke"] = "smoke"
    cohort: Literal["canonical", "sensitivity_full"] = "canonical"
    checkpoint_name: str = "sjepa_curriculum_final.pt"
    frame_count: int | None = None
    outer_folds: int = 5
    inner_folds: int = 3
    seed: int = 2027
    output_run_id: str | None = None
    input_frame: Literal["canonical_body", "historical_center_scale"] = "canonical_body"

    def validate(self) -> None:
        if self.mode == "real" and not self.artifact_root:
            raise ValueError(
                "Real GAVD audit requires GAVD_AUDIT_ARTIFACT_ROOT; it never falls back to smoke."
            )
        if self.outer_folds < 2 or self.inner_folds < 2:
            raise ValueError("At least two outer and inner folds are required.")
        if self.cohort not in {"canonical", "sensitivity_full"}:
            raise ValueError(f"Unknown cohort: {self.cohort}")


@dataclass
class PoseRecord:
    sequence_id: str
    video_id: str
    condition: str
    provenance: Literal["canonical", "augmented_normal"]
    raw: np.ndarray
    path: str

    def manifest_row(self) -> dict[str, str]:
        return {
            "sequence_id": self.sequence_id,
            "video_id": self.video_id,
            "condition": self.condition,
            "provenance": self.provenance,
            "path": self.path,
        }


def _load_pose_file(path: Path, condition: str, provenance: str) -> PoseRecord:
    with np.load(path, allow_pickle=False) as archive:
        raw = np.asarray(archive["sequence"], dtype=np.float32)
        sequence_id = str(archive["sequence_id"].item())
        video_id = str(archive["video_id"].item())
    if raw.ndim != 3 or raw.shape[1:] != (33, 4):
        raise ValueError(f"{path} has pose shape {raw.shape}; expected [T, 33, 4].")
    if not sequence_id or not video_id:
        raise ValueError(f"{path} is missing a sequence or source-video ID.")
    return PoseRecord(sequence_id, video_id, condition, provenance, raw, str(path))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen cohort manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_gavd_records(artifact_root: str | Path, cohort: str) -> list[PoseRecord]:
    """Load explicit provenance-separated GAVD cohorts.

    ``canonical`` reads only ``real/poses``. ``sensitivity_full`` appends
    ``real/poses_augmented/normal`` and is never a condition-effect cohort.
    """
    root = Path(artifact_root).expanduser().resolve() / "real"
    canonical_root = root / "poses"
    if not canonical_root.is_dir():
        raise FileNotFoundError(f"Canonical GAVD poses not found: {canonical_root}")
    records: list[PoseRecord] = []
    # The CSV is a membership manifest only: its historical csv_path is stale
    # and intentionally never used to find input poses.
    for row in _read_csv_rows(root / "manifest.csv"):
        condition, sequence_id, video_id = row["condition"], row["sequence_id"], row["video_id"]
        if condition not in CONDITIONS:
            raise ValueError(f"Manifest has unexpected condition {condition!r}")
        record = _load_pose_file(canonical_root / condition / f"{sequence_id}.npz", condition, "canonical")
        if (record.sequence_id, record.video_id) != (sequence_id, video_id):
            raise ValueError(f"NPZ metadata disagrees with frozen manifest for {sequence_id}")
        records.append(record)
    if cohort == "sensitivity_full":
        folder = root / "poses_augmented" / "normal"
        for row in _read_csv_rows(root / "augmented_normal_pose_coverage.csv"):
            sequence_id, video_id = row["sequence_id"], row["video_id"]
            path = folder / f"{sequence_id}.npz"
            if not path.is_file():
                # Coverage is the accepted-file manifest. Missing listed files
                # are an input corruption, not an invitation to glob a substitute.
                raise FileNotFoundError(f"Accepted augmented sequence is missing: {path}")
            record = _load_pose_file(path, "normal", "augmented_normal")
            if (record.sequence_id, record.video_id) != (sequence_id, video_id):
                raise ValueError(f"Augmented NPZ metadata disagrees with coverage manifest for {sequence_id}")
            records.append(record)
    if cohort not in {"canonical", "sensitivity_full"}:
        raise ValueError(f"Unknown cohort {cohort!r}")
    validate_records(records)
    return records


def validate_records(records: Sequence[PoseRecord]) -> None:
    if not records:
        raise ValueError("Cohort contains no records.")
    ids = [record.sequence_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Sequence IDs must be unique across canonical and sensitivity rows.")
    for record in records:
        if record.condition not in CONDITIONS:
            raise ValueError(f"Unexpected condition {record.condition!r}")
        if not np.isfinite(record.raw[..., 3]).any():
            raise ValueError(f"{record.sequence_id} has no finite visibility values.")
    if len({record.video_id for record in records}) < 2:
        raise ValueError("A source-video-disjoint audit requires at least two source videos.")


def make_smoke_records(seed: int = 2027, clips_per_source: int = 3) -> list[PoseRecord]:
    """Synthetic wiring fixture with an intentionally planted signed signal.

    It is not a disease simulation and must never be written as GAVD evidence.
    """
    generator = np.random.default_rng(seed)
    records: list[PoseRecord] = []
    phase = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    for condition_index, condition in enumerate(CONDITIONS):
        for source_index in range(2):
            source_id = f"synthetic-{condition}-{source_index}"
            direction = 1.0 if source_index == 0 else -1.0
            for clip in range(clips_per_source):
                raw = generator.normal(0.0, 0.002, size=(len(phase), 33, 4)).astype(np.float32)
                raw[..., 3] = 1.0
                raw[:, 23, :3], raw[:, 24, :3] = (-0.15, 0.0, 0.0), (0.15, 0.0, 0.0)
                raw[:, 11, :3], raw[:, 12, :3] = (-0.2, 0.65, 0.0), (0.2, 0.65, 0.0)
                for left, right in TARGET_PAIRS:
                    raw[:, left, 0] += (0.04 - 0.01 * direction) * np.sin(phase)
                    raw[:, right, 0] += (0.04 + 0.01 * direction) * np.sin(phase)
                records.append(PoseRecord(
                    f"smoke-{condition_index}-{source_index}-{clip}", source_id, condition, "canonical", raw,
                    "SYNTHETIC / NOT GAVD",
                ))
    validate_records(records)
    return records


def reflect_joint_values(values: np.ndarray, pairs: Sequence[tuple[int, int]] = MIRROR_PAIRS) -> np.ndarray:
    """Swap semantic left/right slots for masks, confidences, or arbitrary joint data."""
    reflected = np.asarray(values).copy()
    if reflected.ndim < 2 or reflected.shape[1] != 33:
        raise ValueError("Joint values must be [T, 33, ...].")
    for left, right in pairs:
        reflected[:, [left, right], ...] = reflected[:, [right, left], ...]
    return reflected


def anatomical_reflection(sequence: np.ndarray) -> np.ndarray:
    """Reflect geometry and jointly swap all semantic joint-associated columns.

    Coordinates occupy columns 0:3.  Visibility/confidence columns are not
    negated; they move with their anatomical joint.
    """
    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1] != 33 or sequence.shape[2] < 3:
        raise ValueError("Expected [T, 33, >=3] skeleton sequence.")
    reflected = reflect_joint_values(sequence)
    reflected[..., 0] *= -1.0
    return reflected


def historical_left_minus_right_excursion(sequence: np.ndarray) -> float:
    """The immutable historical coordinate diagnostic, retained for traceability."""
    xyz = np.asarray(sequence, dtype=np.float64)[..., :3]
    return float(sum(
        xyz[:, left, :].std(axis=0).sum() - xyz[:, right, :].std(axis=0).sum()
        for left, right in TARGET_PAIRS
    ))


def signed_right_minus_left_excursion(sequence: np.ndarray) -> float:
    """Reported GAVD target. Positive always means right excursion is greater."""
    return -historical_left_minus_right_excursion(sequence)


def even_total_excursion(sequence: np.ndarray) -> float:
    xyz = np.asarray(sequence, dtype=np.float64)[..., :3]
    return float(sum(
        xyz[:, left, :].std(axis=0).sum() + xyz[:, right, :].std(axis=0).sum()
        for left, right in TARGET_PAIRS
    ))


def interpolate_short_gaps(sequence: np.ndarray, threshold: float = 0.45, max_gap: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate only bounded low-confidence gaps; retain a semantic observed mask."""
    output = np.asarray(sequence, dtype=np.float32).copy()
    visibility = np.nan_to_num(output[..., 3], nan=0.0) if output.shape[-1] > 3 else np.ones(output.shape[:2])
    valid = (visibility >= threshold) & np.isfinite(output[..., :3]).all(axis=-1)
    retained = valid.copy()
    for joint in range(33):
        observed = np.flatnonzero(valid[:, joint])
        for start, end in zip(observed[:-1], observed[1:]):
            gap = int(end - start - 1)
            if not 0 < gap <= max_gap:
                continue
            fraction = (np.arange(1, gap + 1, dtype=np.float32) / (gap + 1))[:, None]
            output[start + 1:end, joint, :3] = (
                output[start, joint, :3] * (1.0 - fraction) + output[end, joint, :3] * fraction
            )
            retained[start + 1:end, joint] = True
        output[~retained[:, joint], joint, :3] = np.nan
    return output, valid


def _unit(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if np.isfinite(norm) and norm > 1e-6 else fallback.copy()


def canonical_body_frame(sequence: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center each frame and orient it to mediolateral, vertical, and forward axes.

    The axes use only within-trial skeleton geometry.  The returned 3x3 matrix
    is retained for audit; degenerate frames fall back deterministically.
    """
    xyz = np.asarray(sequence, dtype=np.float32)[..., :3].copy()
    finite = np.isfinite(xyz).all(axis=-1)
    hip_mid = np.nanmean(np.stack([xyz[:, 23], xyz[:, 24]], axis=1), axis=1)
    shoulder_mid = np.nanmean(np.stack([xyz[:, 11], xyz[:, 12]], axis=1), axis=1)
    pelvis = np.nanmedian(hip_mid[np.isfinite(hip_mid).all(axis=1)], axis=0) if np.isfinite(hip_mid).any() else np.zeros(3)
    pelvis_by_frame = hip_mid.copy()
    bad = ~np.isfinite(pelvis_by_frame).all(axis=1)
    pelvis_by_frame[bad] = pelvis
    centered = xyz - pelvis_by_frame[:, None, :]
    ml = np.nanmedian(xyz[:, 24] - xyz[:, 23], axis=0)
    vertical = np.nanmedian(shoulder_mid - hip_mid, axis=0)
    ml = _unit(ml, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    vertical = vertical - ml * np.dot(vertical, ml)
    vertical = _unit(vertical, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    forward = _unit(np.cross(vertical, ml), np.array([0.0, 0.0, 1.0], dtype=np.float32))
    # Align forward to robust pelvis travel when that signal is available.
    displacement = hip_mid[-1] - hip_mid[0]
    if np.isfinite(displacement).all() and np.dot(displacement, forward) < 0:
        forward *= -1.0
    rotation = np.stack([ml, vertical, forward], axis=1)
    canonical = np.einsum("tjc,ck->tjk", centered, rotation)
    canonical[~finite] = np.nan
    return canonical.astype(np.float32), rotation.astype(np.float32)


def robust_leg_scale(xyz: np.ndarray) -> float:
    lengths = []
    for hip, knee, ankle in ((23, 25, 27), (24, 26, 28)):
        lengths.extend(np.linalg.norm(xyz[:, hip] - xyz[:, knee], axis=1))
        lengths.extend(np.linalg.norm(xyz[:, knee] - xyz[:, ankle], axis=1))
    scale = float(np.nanmedian(np.asarray(lengths, dtype=np.float64)))
    return scale if np.isfinite(scale) and scale > 1e-6 else 1.0


def temporal_resize(array: np.ndarray, frames: int) -> np.ndarray:
    array = np.asarray(array)
    if frames < 2:
        raise ValueError("frame count must be at least two")
    if len(array) == frames:
        return array.copy()
    if len(array) < 2:
        return np.repeat(array, frames, axis=0)
    old = np.linspace(0.0, 1.0, len(array))
    new = np.linspace(0.0, 1.0, frames)
    flat = array.reshape(len(array), -1)
    resized = np.stack([np.interp(new, old, flat[:, column]) for column in range(flat.shape[1])], axis=1)
    return resized.reshape(frames, *array.shape[1:])


def prepare_for_encoder(sequence: np.ndarray, frames: int, input_frame: str = "canonical_body") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return normalized xyz, an observed mask, and the saved frame rotation."""
    filled, observed = interpolate_short_gaps(sequence)
    if input_frame == "canonical_body":
        xyz, rotation = canonical_body_frame(filled)
    elif input_frame == "historical_center_scale":
        xyz = filled[..., :3].copy()
        hip_mid = np.nanmean(np.stack([xyz[:, 23], xyz[:, 24]], axis=1), axis=1)
        xyz -= np.nan_to_num(hip_mid)[:, None, :]
        rotation = np.eye(3, dtype=np.float32)
    else:
        raise ValueError(f"Unknown input frame {input_frame!r}")
    xyz = xyz / robust_leg_scale(xyz)
    xyz = np.nan_to_num(xyz, nan=0.0, posinf=0.0, neginf=0.0)
    return temporal_resize(xyz, frames).astype(np.float32), temporal_resize(observed.astype(np.float32), frames) > 0.5, rotation


def yaw_rotate(prepared_xyz: np.ndarray, degrees: float) -> np.ndarray:
    """Rigid yaw around saved canonical vertical axis; no recanonicalization or side swap."""
    theta = np.deg2rad(degrees)
    rotation = np.array(((np.cos(theta), 0.0, np.sin(theta)), (0.0, 1.0, 0.0),
                         (-np.sin(theta), 0.0, np.cos(theta))), dtype=np.float32)
    return np.einsum("tjc,ck->tjk", prepared_xyz, rotation).astype(np.float32)


def source_group_folds(records: Sequence[PoseRecord], requested_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    from sklearn.model_selection import GroupKFold

    groups = np.asarray([record.video_id for record in records])
    unique = np.unique(groups)
    folds = min(requested_folds, len(unique))
    if folds < 2:
        raise ValueError("Need at least two source videos for grouped folds.")
    index = np.arange(len(records))
    result = list(GroupKFold(n_splits=folds).split(index, groups=groups))
    for train, test in result:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("Source video crosses a fold.")
    return result


def nested_group_rankings(groups: Sequence[str], seed: int, repetitions: int = 20) -> list[list[str]]:
    """Frozen nested source prefixes for audit-only label-efficiency sensitivity."""
    unique = np.asarray(sorted(set(map(str, groups))))
    if len(unique) < 2:
        raise ValueError("Need two groups to make rankings.")
    generator = np.random.default_rng(seed)
    return [generator.permutation(unique).tolist() for _ in range(repetitions)]


def validate_rankings(rankings: Sequence[Sequence[str]], groups: Sequence[str]) -> None:
    expected = set(map(str, groups))
    for ranking in rankings:
        if set(ranking) != expected or len(ranking) != len(expected):
            raise AssertionError("Each ranking must contain every source exactly once.")
        for prefix_size in range(1, len(ranking)):
            if not set(ranking[:prefix_size]) < set(ranking[:prefix_size + 1]):
                raise AssertionError("Label-budget prefixes must be nested.")


def source_manifest(records: Sequence[PoseRecord], folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, object]]:
    rows = [record.manifest_row() for record in records]
    for fold_index, (_, test) in enumerate(folds):
        for index in test:
            rows[int(index)]["outer_fold"] = fold_index
    return rows


def laterality_features(tokens: np.ndarray) -> np.ndarray:
    """Signed plus symmetric token features for the frozen representation readout."""
    features = []
    for left, right in TARGET_PAIRS:
        left_token = tokens[:, left, :].mean(axis=0)
        right_token = tokens[:, right, :].mean(axis=0)
        features.extend((right_token - left_token, right_token + left_token))
    return np.concatenate(features).astype(np.float64)


def side_agnostic_features(tokens: np.ndarray) -> np.ndarray:
    """Within-input side sum only; callers must symmetrize across M as well."""
    return np.concatenate([tokens[:, left, :].mean(axis=0) + tokens[:, right, :].mean(axis=0)
                           for left, right in TARGET_PAIRS]).astype(np.float64)


def nuisance_features(sequence: np.ndarray) -> np.ndarray:
    """M-invariant acquisition diagnostics; never encode a side difference."""
    raw = np.asarray(sequence, dtype=np.float64)
    visibility = np.nan_to_num(raw[..., 3], nan=0.0) if raw.shape[-1] > 3 else np.ones(raw.shape[:2])
    observed = np.isfinite(raw[..., :3]).all(axis=-1)
    return np.asarray((len(raw), visibility.mean(), visibility.std(), observed.mean(),
                       np.nanmedian(np.linalg.norm(raw[:, 23, :3] - raw[:, 24, :3], axis=1))), dtype=np.float64)


def coordinate_reference_features(xyz: np.ndarray) -> np.ndarray:
    return np.concatenate([xyz[:, right, :].std(axis=0) - xyz[:, left, :].std(axis=0)
                           for left, right in TARGET_PAIRS]).astype(np.float64)


def source_weights(groups: Sequence[str]) -> np.ndarray:
    """Every source contributes unit total fit weight irrespective of its clip count."""
    groups = np.asarray(groups)
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


@dataclass
class CVResult:
    arm: str
    prediction: np.ndarray
    mirrored_prediction: np.ndarray
    alpha_by_fold: list[float]


class _RMSScaler:
    """Origin-preserving scaling required for an exactly odd projected head."""

    def fit(self, x: np.ndarray) -> "_RMSScaler":
        self.scale_ = np.sqrt(np.mean(np.square(x), axis=0))
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x / self.scale_


@dataclass
class _TwoViewHead:
    """Arm D: a shared scalar q(h)=wᵀh, plus only a, b, and c."""

    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    a: float
    b: float
    c: float

    def predict(self, original: np.ndarray, mirrored: np.ndarray) -> np.ndarray:
        original = (original - self.mean) / self.scale
        mirrored = (mirrored - self.mean) / self.scale
        return self.a * (original @ self.weights) + self.b * (mirrored @ self.weights) + self.c


def _fit_two_view_head(original: np.ndarray, mirrored: np.ndarray, target: np.ndarray, groups: Sequence[str], alpha: float) -> _TwoViewHead:
    """Jointly optimize the parameter-matched Arm-D objective.

    This is intentionally not a concatenated Ridge model: both encoder passes
    must share q's weights, and D has exactly three additional scalars.
    """
    from scipy.optimize import minimize

    mean = np.mean(np.concatenate((original, mirrored)), axis=0)
    scale = np.std(np.concatenate((original, mirrored)), axis=0)
    scale[scale < 1e-12] = 1.0
    original = (original - mean) / scale
    mirrored = (mirrored - mean) / scale
    weights = source_weights(groups)
    width = original.shape[1]

    def loss_and_gradient(theta: np.ndarray) -> tuple[float, np.ndarray]:
        # Frozen unit-circle convention removes the a/b/w scale degeneracy:
        # a=cos(theta), b=sin(theta), so q's regularization has a meaning.
        w, angle, c = theta[:width], theta[width], theta[width + 1]
        a, b = np.cos(angle), np.sin(angle)
        q_original, q_mirrored = original @ w, mirrored @ w
        residual = a * q_original + b * q_mirrored + c - target
        total_weight = weights.sum()
        weighted_residual = weights * residual
        value = np.dot(weighted_residual, residual) / total_weight + alpha * np.mean(w * w)
        gradient_w = (2.0 / total_weight) * (a * original.T @ weighted_residual + b * mirrored.T @ weighted_residual)
        gradient_w += (2.0 * alpha / width) * w
        gradient = np.concatenate((gradient_w, np.asarray((
            2.0 * np.dot((-b * q_original + a * q_mirrored), weighted_residual) / total_weight,
            2.0 * weighted_residual.sum() / total_weight,
        ))))
        return float(value), gradient

    baseline = (original + mirrored) / 2.0
    weighted_baseline = baseline * np.sqrt(weights)[:, None]
    weighted_target = target * np.sqrt(weights)
    initial_w = np.linalg.lstsq(weighted_baseline, weighted_target, rcond=None)[0]
    start = np.concatenate((initial_w, np.array([np.pi / 4.0, 0.0])))
    optimized = minimize(loss_and_gradient, start, jac=True, method="L-BFGS-B", options={"maxiter": 1_000, "ftol": 1e-10})
    if not np.isfinite(optimized.x).all():
        raise RuntimeError(f"Arm D optimization produced non-finite parameters: {optimized.message}")
    theta = optimized.x
    return _TwoViewHead(mean, scale, theta[:width], float(np.cos(theta[width])), float(np.sin(theta[width])), float(theta[width + 1]))


def _choose_two_view_alpha(original: np.ndarray, mirrored: np.ndarray, y: np.ndarray, groups: np.ndarray, inner_folds: int) -> float:
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold

    candidates = np.logspace(-2, 2, 5)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return float(candidates[0])
    splitter = GroupKFold(n_splits=min(inner_folds, n_groups))
    scores = []
    for alpha in candidates:
        fold_scores = []
        for train, validation in splitter.split(original, y, groups):
            head = _fit_two_view_head(original[train], mirrored[train], y[train], groups[train], alpha)
            fold_scores.append(mean_absolute_error(y[validation], head.predict(original[validation], mirrored[validation]),
                                                  sample_weight=source_weights(groups[validation])))
        scores.append(np.mean(fold_scores))
    return float(candidates[int(np.argmin(scores))])


def _choose_alpha(x: np.ndarray, y: np.ndarray, groups: np.ndarray, odd: bool, inner_folds: int) -> float:
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    candidates = np.logspace(-3, 3, 13)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return float(candidates[0])
    splitter = GroupKFold(n_splits=min(inner_folds, n_groups))
    scores = []
    for alpha in candidates:
        fold_scores = []
        for train, validation in splitter.split(x, y, groups):
            scaler = _RMSScaler().fit(x[train]) if odd else StandardScaler().fit(x[train])
            weights = source_weights(groups[train])
            model = Ridge(alpha=alpha, fit_intercept=not odd).fit(scaler.transform(x[train]), y[train], sample_weight=weights)
            fold_scores.append(mean_absolute_error(y[validation], model.predict(scaler.transform(x[validation])),
                                                  sample_weight=source_weights(groups[validation])))
        scores.append(np.mean(fold_scores))
    return float(candidates[int(np.argmin(scores))])


def run_grouped_arm(
    arm: Literal["A", "B", "C", "D", "E", "F", "G_side_agnostic", "G_nuisance"],
    original: np.ndarray,
    mirrored: np.ndarray,
    target: np.ndarray,
    groups: Sequence[str],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    inner_folds: int,
    evaluation_original: np.ndarray | None = None,
    evaluation_mirrored: np.ndarray | None = None,
) -> CVResult:
    """Fit an OOF source-video-grouped GAVD probe with frozen arm semantics."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    original = np.asarray(original, dtype=np.float64)
    mirrored = np.asarray(mirrored, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    evaluation_original = original if evaluation_original is None else np.asarray(evaluation_original, dtype=np.float64)
    evaluation_mirrored = mirrored if evaluation_mirrored is None else np.asarray(evaluation_mirrored, dtype=np.float64)
    group_array = np.asarray(groups)
    prediction = np.full(len(target), np.nan)
    prediction_mirror = np.full(len(target), np.nan)
    alphas: list[float] = []
    for train, test in folds:
        if arm == "D":
            alpha = _choose_two_view_alpha(original[train], mirrored[train], target[train], group_array[train], inner_folds)
            head = _fit_two_view_head(original[train], mirrored[train], target[train], group_array[train], alpha)
            prediction[test] = head.predict(evaluation_original[test], evaluation_mirrored[test])
            prediction_mirror[test] = head.predict(evaluation_mirrored[test], evaluation_original[test])
            alphas.append(alpha)
            continue
        if arm == "C":
            x_train = (original[train] - mirrored[train]) / 2.0
            x_test = (evaluation_original[test] - evaluation_mirrored[test]) / 2.0
            x_test_mirror = -x_test
            y_train = target[train]
            odd = True
        elif arm == "B":
            x_train = np.concatenate((original[train], mirrored[train]))
            y_train = np.concatenate((target[train], -target[train]))
            x_test, x_test_mirror, odd = evaluation_original[test], evaluation_mirrored[test], False
        else:
            x_train, y_train = original[train], target[train]
            x_test, x_test_mirror, odd = evaluation_original[test], evaluation_mirrored[test], False
        train_groups = group_array[train] if arm != "B" else np.concatenate((group_array[train], group_array[train]))
        alpha = _choose_alpha(x_train, y_train, train_groups, odd, inner_folds)
        scaler = _RMSScaler().fit(x_train) if odd else StandardScaler().fit(x_train)
        model = Ridge(alpha=alpha, fit_intercept=not odd).fit(
            scaler.transform(x_train), y_train, sample_weight=source_weights(train_groups)
        )
        prediction[test] = model.predict(scaler.transform(x_test))
        prediction_mirror[test] = model.predict(scaler.transform(x_test_mirror))
        alphas.append(alpha)
    return CVResult(arm, prediction, prediction_mirror, alphas)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, r2_score

    finite = np.isfinite(prediction)
    return {"mae": float(mean_absolute_error(target[finite], prediction[finite])),
            "r2_untruncated": float(r2_score(target[finite], prediction[finite]))}


def source_balanced_regression_metrics(target: np.ndarray, prediction: np.ndarray, groups: Sequence[str]) -> dict[str, float]:
    """Metrics whose source, rather than clip, is the independent contribution."""
    from sklearn.metrics import mean_absolute_error, r2_score

    groups = np.asarray(groups)
    finite = np.isfinite(prediction)
    weight = source_weights(groups[finite])
    return {"source_balanced_mae": float(mean_absolute_error(target[finite], prediction[finite], sample_weight=weight)),
            "source_balanced_r2_untruncated": float(r2_score(target[finite], prediction[finite], sample_weight=weight))}


def mirror_metrics(prediction: np.ndarray, mirrored_prediction: np.ndarray, epsilon: float = 1e-12) -> dict[str, float]:
    finite = np.isfinite(prediction) & np.isfinite(mirrored_prediction)
    x, mirrored = prediction[finite], mirrored_prediction[finite]
    denominator = np.mean(np.abs(x)) + np.mean(np.abs(mirrored)) + epsilon
    slope, intercept = np.polyfit(x, mirrored, 1) if len(x) > 1 and np.std(x) > epsilon else (np.nan, np.nan)
    nonzero = (np.abs(x) > epsilon) & (np.abs(mirrored) > epsilon)
    flips = np.mean((x[nonzero] * mirrored[nonzero]) < 0) if nonzero.any() else np.nan
    return {"oddness_error": float(np.mean(np.abs(x + mirrored)) / denominator),
            "mirror_slope": float(slope), "mirror_intercept": float(intercept), "sign_flip_rate": float(flips)}


def source_balanced_mirror_metrics(prediction: np.ndarray, mirrored_prediction: np.ndarray, groups: Sequence[str]) -> dict[str, float]:
    """Mirror geometry after reducing paired clip predictions to one source each."""
    groups = np.asarray(groups)
    unique = np.unique(groups)
    source_prediction = np.asarray([np.mean(prediction[groups == source]) for source in unique])
    source_mirrored = np.asarray([np.mean(mirrored_prediction[groups == source]) for source in unique])
    return {f"source_balanced_{key}": value for key, value in mirror_metrics(source_prediction, source_mirrored).items()}


def source_bootstrap_mean(records: Sequence[PoseRecord], values: np.ndarray, repetitions: int = 10_000, seed: int = 2027) -> dict[str, float]:
    """Descriptive source-block bootstrap of a per-clip value, source-balanced first."""
    groups = np.asarray([record.video_id for record in records])
    unique = np.unique(groups)
    source_values = np.asarray([np.mean(values[groups == source]) for source in unique])
    generator = np.random.default_rng(seed)
    draws = np.mean(generator.choice(source_values, size=(repetitions, len(source_values)), replace=True), axis=1)
    return {"estimate": float(source_values.mean()), "ci_2_5": float(np.percentile(draws, 2.5)),
            "ci_97_5": float(np.percentile(draws, 97.5))}


def paired_source_bootstrap(
    records: Sequence[PoseRecord], values: np.ndarray, statistic: Callable[[np.ndarray], float],
    repetitions: int = 10_000, seed: int = 2027,
) -> dict[str, float]:
    """Resample sources, never clips, for a descriptive GAVD uncertainty interval."""
    groups = np.asarray([record.video_id for record in records])
    unique = np.unique(groups)
    generator = np.random.default_rng(seed)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        sampled = generator.choice(unique, size=len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == source) for source in sampled])
        estimates[repetition] = statistic(values[indices])
    return {"estimate": float(statistic(values)), "ci_2_5": float(np.percentile(estimates, 2.5)),
            "ci_97_5": float(np.percentile(estimates, 97.5))}


def corrupt_coordinates(xyz: np.ndarray, family: str, severity: float, seed: int) -> np.ndarray:
    """Shared, deterministic diagnostic corruption generator (not a clinical endpoint)."""
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must lie in [0, 1]")
    generator = np.random.default_rng(seed)
    output = np.asarray(xyz, dtype=np.float32).copy()
    if family == "coordinate_noise":
        output += generator.normal(0.0, 0.03 * severity, size=output.shape).astype(np.float32)
    elif family == "joint_dropout":
        mask = generator.random(output.shape[:2]) < 0.25 * severity
        output[mask] = 0.0
    elif family == "temporal_shift":
        output = np.roll(output, int(round(severity * max(1, len(output) // 4))), axis=0)
    else:
        raise ValueError(f"Unknown corruption family {family!r}")
    return output


def target_permutation(target: np.ndarray, groups: Sequence[str], seed: int) -> np.ndarray:
    """Destructive fixed clip-target permutation preserving the exact target vector.

    Grouped folds remain unchanged; this is explicitly a clip-label pairing
    falsification, not a source-level outcome permutation.
    """
    generator = np.random.default_rng(seed)
    return np.asarray(target)[generator.permutation(len(target))]


def left_right_label_shuffle(sequence: np.ndarray, seed: int, probability: float = 0.5) -> np.ndarray:
    """Destructively corrupt per-record semantic side labels without changing its target."""
    generator = np.random.default_rng(seed)
    shuffled = np.asarray(sequence, dtype=np.float32).copy()
    chosen = [pair for pair in MIRROR_PAIRS if generator.random() < probability]
    if not chosen:
        chosen = [MIRROR_PAIRS[int(generator.integers(len(MIRROR_PAIRS)))]]
    return reflect_joint_values(shuffled, chosen)


def build_result_bundle(config: AuditConfig, records: Sequence[PoseRecord], checkpoint: dict[str, object],
                        results: dict[str, dict[str, float]]) -> dict[str, object]:
    membership = [record.manifest_row() for record in records]
    digest = hashlib.sha256(json.dumps(membership, sort_keys=True).encode()).hexdigest()
    return {"protocol_version": "gavd-signed-laterality-audit-v2", "labels": AUDIT_LABELS,
            "config": asdict(config), "cohort_sha256": digest, "n_sequences": len(records),
            "n_source_videos": len({record.video_id for record in records}), "checkpoint": checkpoint,
            "results": results}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if torch is not None:
    class SkeletonPatchEncoder(nn.Module):
        def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4, embed_dim=64,
                     depth=2, heads=4, dropout=0.0):
            super().__init__()
            self.frames, self.joints, self.coordinate_dim = frames, joints, coordinate_dim
            self.segment_length, self.embed_dim = segment_length, embed_dim
            self.segments = frames // segment_length
            self.patch_embed = nn.Linear(segment_length * coordinate_dim, embed_dim)
            self.time_pos = nn.Parameter(torch.randn(self.segments, embed_dim) * 0.02)
            self.joint_pos = nn.Parameter(torch.randn(joints, embed_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(embed_dim, heads, embed_dim * 4, dropout, "gelu", batch_first=True,
                                               norm_first=True)
            self.blocks, self.norm = nn.TransformerEncoder(layer, depth), nn.LayerNorm(embed_dim)

        def forward(self, x):
            if x.shape[1:] != (self.frames, self.joints, self.coordinate_dim):
                raise ValueError(f"Unexpected encoder input shape {tuple(x.shape)}")
            batch = len(x)
            patches = x.reshape(batch, self.segments, self.segment_length, self.joints, self.coordinate_dim)
            patches = patches.permute(0, 1, 3, 2, 4).flatten(3)
            tokens = self.patch_embed(patches) + self.time_pos[None, :, None] + self.joint_pos[None, None, :]
            return self.norm(self.blocks(tokens.reshape(batch, -1, self.embed_dim)))


    class SkeletonPredictor(nn.Module):
        def __init__(self, segments, joints, encoder_dim=64, predictor_dim=64, depth=2, heads=4, dropout=0.0):
            super().__init__()
            self.encoder_to_predictor = nn.Linear(encoder_dim, predictor_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
            self.time_pos = nn.Parameter(torch.randn(segments, predictor_dim) * 0.02)
            self.joint_pos = nn.Parameter(torch.randn(joints, predictor_dim) * 0.02)
            layer = nn.TransformerEncoderLayer(predictor_dim, heads, predictor_dim * 4, dropout, "gelu",
                                               batch_first=True, norm_first=True)
            self.blocks, self.norm, self.output = nn.TransformerEncoder(layer, depth), nn.LayerNorm(predictor_dim), nn.Linear(predictor_dim, encoder_dim)


    class SJEPAGait(nn.Module):
        def __init__(self, frames=64, joints=33, coordinate_dim=3, segment_length=4, embed_dim=64,
                     encoder_depth=2, predictor_depth=2, heads=4):
            super().__init__()
            import copy
            self.view_encoder = SkeletonPatchEncoder(frames, joints, coordinate_dim, segment_length, embed_dim, encoder_depth, heads)
            self.target_encoder = copy.deepcopy(self.view_encoder)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.predictor = SkeletonPredictor(frames // segment_length, joints, embed_dim, embed_dim, predictor_depth, heads)
            self.register_buffer("target_center", torch.zeros(embed_dim))


def load_frozen_encoder(path: str | Path):
    if torch is None:
        raise RuntimeError("PyTorch is required for frozen encoder features.")
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "config" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("Checkpoint lacks config/model_state required by the historical S-JEPA architecture.")
    model = SJEPAGait(**checkpoint["config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    metadata = {key: checkpoint.get(key) for key in ("dataset_fingerprint", "mode", "conditions_seen", "curriculum_complete", "mask_keypoints")}
    return model.target_encoder, checkpoint["config"], metadata


def encode_tokens(encoder, xyz: np.ndarray, batch_size: int = 16) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for frozen encoder features.")
    batches = []
    with torch.no_grad():
        for start in range(0, len(xyz), batch_size):
            current = torch.as_tensor(xyz[start:start + batch_size], dtype=torch.float32)
            encoded = encoder(current)
            batches.append(encoded.cpu().numpy())
    flat = np.concatenate(batches)
    return flat.reshape(len(xyz), encoder.segments, encoder.joints, encoder.embed_dim)
