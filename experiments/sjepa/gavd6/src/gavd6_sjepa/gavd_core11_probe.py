"""Frozen AMASS-Core11 adapter and descriptive GAVD linear probes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier

from .amass_core11_jepa import *
from .gait_parity_jepa import *


MEDIAPIPE_CORE_INDEX = {
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_forefoot": 31,
    "right_forefoot": 32,
}
AMASS_UP = np.array([0.0, 0.0, 1.0], dtype=np.float64)
DEFAULT_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)


@dataclass(frozen=True)
class AdapterConfig:
    visibility_threshold: float = 0.45
    maximum_gap_frames: int = 4
    canonical_fps: float = 30.0
    minimum_travel_leg_lengths: float = 0.10
    minimum_travel_straightness: float = 0.20
    minimum_abs_lateral_hip_alignment: float = 0.95
    minimum_token_valid_fraction: float = 0.95
    frame_policy: str = "legacy-anatomical"

    def __post_init__(self) -> None:
        if self.frame_policy not in {
            "legacy-anatomical",
            "gauge-neutral-travel-or-image",
        }:
            raise ValueError(f"Unsupported GAVD frame policy: {self.frame_policy!r}")


def _video_aspect_ratio(path: str | Path) -> float:
    capture = cv2.VideoCapture(str(path))
    width = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if width <= 0 or height <= 0:
        raise ValueError(f"Could not read source-video dimensions: {path}")
    return width / height


def load_gavd_pose_records(pose_dir: str | Path) -> list[dict]:
    """Load the canonical 96-sequence cache, retaining FPS and video geometry."""

    records = []
    for path in sorted(Path(pose_dir).glob("*/*.npz")):
        with np.load(path, allow_pickle=False) as archive:
            required = {"sequence", "sequence_id", "video_id", "condition", "fps", "source_video"}
            missing = required.difference(archive.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            sequence = archive["sequence"].astype(np.float32)
            if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
                raise ValueError(f"Expected [T,33,4] in {path}, got {sequence.shape}")
            source_video = str(archive["source_video"].item())
            records.append(
                {
                    "condition": str(archive["condition"].item()),
                    "sequence_id": str(archive["sequence_id"].item()),
                    "video_id": str(archive["video_id"].item()),
                    "fps": float(archive["fps"].item()),
                    "aspect_ratio": _video_aspect_ratio(source_video),
                    "sequence": sequence,
                    "path": str(path),
                }
            )
    if not records:
        raise ValueError(f"No GAVD pose archives found below {pose_dir}")
    return records


def _fill_short_gaps(values: np.ndarray, valid: np.ndarray, maximum_gap: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float32).copy()
    usable = np.asarray(valid, dtype=bool).copy()
    for joint in range(values.shape[1]):
        observed = np.flatnonzero(usable[:, joint])
        for left, right in zip(observed[:-1], observed[1:]):
            gap = int(right - left - 1)
            if not 0 < gap <= maximum_gap:
                continue
            weight = (np.arange(1, gap + 1, dtype=np.float32) / (gap + 1))[:, None]
            values[left + 1 : right, joint] = (
                (1.0 - weight) * values[left, joint] + weight * values[right, joint]
            )
            usable[left + 1 : right, joint] = True
    values[~usable] = 0.0
    return values, usable


def mediapipe_to_core11_world(
    sequence: np.ndarray,
    *,
    aspect_ratio: float,
    config: AdapterConfig = AdapterConfig(),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map MediaPipe image landmarks into a right-handed pseudo-world Core11.

    MediaPipe x and z are already stored in full-frame-width units. Image y is
    converted from full-frame-height units before axes become [right, depth, up].
    MediaPipe foot-index is used as a forefoot proxy; AMASS instead averages its
    big- and small-toe surface landmarks. This is a monocular pseudo-3D
    approximation, not metric camera reconstruction.
    """

    sequence = np.asarray(sequence, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
        raise ValueError(f"Expected [T,33,4], got {sequence.shape}")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
        raise ValueError("aspect_ratio must be positive and finite")

    indices = list(MEDIAPIPE_CORE_INDEX.values())
    image_xyz = sequence[:, indices, :3]
    visibility = np.nan_to_num(sequence[:, indices, 3], nan=0.0)
    valid = (visibility >= config.visibility_threshold) & np.isfinite(image_xyz).all(axis=2)
    observed_valid = valid.copy()
    pseudo_world = np.stack(
        [image_xyz[..., 0], image_xyz[..., 2], -image_xyz[..., 1] / aspect_ratio],
        axis=-1,
    )
    pseudo_world, valid = _fill_short_gaps(
        pseudo_world, valid, config.maximum_gap_frames
    )

    coordinates = np.zeros((len(sequence), len(JOINT_NAMES), 3), dtype=np.float32)
    core_valid = np.zeros((len(sequence), len(JOINT_NAMES)), dtype=bool)
    core_observed_valid = np.zeros_like(core_valid)
    coordinates[:, 1:] = pseudo_world
    core_valid[:, 1:] = valid
    both_hips = valid[:, 0] & valid[:, 1]
    coordinates[both_hips, 0] = 0.5 * (
        pseudo_world[both_hips, 0] + pseudo_world[both_hips, 1]
    )
    core_valid[:, 0] = both_hips
    core_valid &= both_hips[:, None]
    observed_both_hips = observed_valid[:, 0] & observed_valid[:, 1]
    core_observed_valid[:, 0] = observed_both_hips
    core_observed_valid[:, 1:] = observed_valid
    core_observed_valid &= observed_both_hips[:, None]
    coordinates[~core_valid] = 0.0
    return coordinates, core_valid, core_observed_valid


def _safe_unit(vector: np.ndarray, label: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError(f"Cannot define {label}; norm={norm}")
    return vector / norm


def _leg_length(coordinates: np.ndarray, valid: np.ndarray) -> float:
    required = valid[:, 1] & valid[:, 2] & valid[:, 3] & valid[:, 4] & valid[:, 5] & valid[:, 6]
    if not required.any():
        raise ValueError("No frame has usable bilateral hips, knees, and ankles")
    xyz = coordinates[required].astype(np.float64)
    left = np.linalg.norm(xyz[:, 1] - xyz[:, 3], axis=1) + np.linalg.norm(
        xyz[:, 3] - xyz[:, 5], axis=1
    )
    right = np.linalg.norm(xyz[:, 2] - xyz[:, 4], axis=1) + np.linalg.norm(
        xyz[:, 4] - xyz[:, 6], axis=1
    )
    scale = float(np.median(0.5 * (left + right)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError(f"Invalid pseudo-3D leg length: {scale}")
    return scale


def _body_frame(
    coordinates: np.ndarray,
    valid: np.ndarray,
    leg_length: float,
    config: AdapterConfig,
) -> tuple[np.ndarray, str]:
    up = AMASS_UP.copy()
    pelvis = coordinates[valid[:, 0], 0].astype(np.float64)
    forward = None
    method = ""
    if len(pelvis) >= 2:
        horizontal = pelvis - np.outer(pelvis @ up, up)
        endpoint_width = max(1, min(15, len(horizontal) // 10))
        net = np.median(horizontal[-endpoint_width:], axis=0) - np.median(
            horizontal[:endpoint_width], axis=0
        )
        displacement = float(np.linalg.norm(net))
        path_length = float(np.linalg.norm(np.diff(horizontal, axis=0), axis=1).sum())
        straightness = displacement / max(path_length, 1e-12)
        if (
            displacement >= config.minimum_travel_leg_lengths * leg_length
            and straightness >= config.minimum_travel_straightness
        ):
            centered = horizontal - horizontal.mean(axis=0, keepdims=True)
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            forward = vh[0] - np.dot(vh[0], up) * up
            if np.dot(forward, net) < 0:
                forward = -forward
            forward = _safe_unit(forward, "pelvis-travel direction")
            method = "pelvis_travel_pca_signed_by_robust_displacement"

    if config.frame_policy == "gauge-neutral-travel-or-image":
        if forward is None:
            # The pseudo-world axes are [image-right, depth, image-up]. Keep a
            # declared image chart instead of manufacturing anatomy from named
            # hips. Odd outputs in this chart must be evaluated up to sign.
            return np.asarray(
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                dtype=np.float64,
            ), "declared_image_space_unanchored"
        lateral = _safe_unit(np.cross(up, forward), "mediolateral direction")
        forward = _safe_unit(np.cross(lateral, up), "orthogonalized forward direction")
        return (
            np.stack([forward, up, lateral], axis=0),
            "gauge_neutral_pelvis_travel_pca_signed_by_net_displacement",
        )

    hips_ok = valid[:, 1] & valid[:, 2]
    hip_axis = np.median(coordinates[hips_ok, 1] - coordinates[hips_ok, 2], axis=0)
    hip_axis = hip_axis.astype(np.float64) - np.dot(hip_axis, up) * up
    anatomical_left = _safe_unit(hip_axis, "anatomical hip axis")
    if forward is None:
        forward = _safe_unit(np.cross(anatomical_left, up), "hip-facing direction")
        method = "hip_facing_fallback"

    lateral = _safe_unit(np.cross(up, forward), "mediolateral direction")
    forward = _safe_unit(np.cross(lateral, up), "orthogonalized forward direction")
    if method.startswith("pelvis_travel") and abs(float(np.dot(lateral, anatomical_left))) < config.minimum_abs_lateral_hip_alignment:
        forward = _safe_unit(np.cross(anatomical_left, up), "hip-facing direction")
        lateral = _safe_unit(np.cross(up, forward), "mediolateral direction")
        method = "hip_facing_fallback_due_to_travel_anatomy_misalignment"
    return np.stack([forward, up, lateral], axis=0), method


def _canonical_times(frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    duration = (frames - 1) / source_fps
    count = int(np.floor(duration * target_fps + 1e-10)) + 1
    return np.arange(count, dtype=np.float64) / target_fps


def _resample(
    values: np.ndarray, valid: np.ndarray, source_fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray]:
    target_times = _canonical_times(len(values), source_fps, target_fps)
    source_times = np.arange(len(values), dtype=np.float64) / source_fps
    right = np.searchsorted(source_times, target_times, side="left")
    right = np.clip(right, 0, len(source_times) - 1)
    exact = np.isclose(source_times[right], target_times, rtol=0.0, atol=1e-10)
    left = np.where(exact, right, np.maximum(right - 1, 0))
    denominator = source_times[right] - source_times[left]
    weight = np.divide(
        target_times - source_times[left],
        denominator,
        out=np.zeros_like(target_times),
        where=denominator > 0,
    )
    output = (1.0 - weight[:, None, None]) * values[left] + weight[:, None, None] * values[right]
    output_valid = valid[left] & valid[right] & np.isfinite(output).all(axis=2)
    output[~output_valid] = 0.0
    return output.astype(np.float32), output_valid


def adapt_mediapipe_sequence(
    sequence: np.ndarray,
    *,
    source_fps: float,
    aspect_ratio: float,
    config: AdapterConfig = AdapterConfig(),
) -> dict:
    """Apply the frozen MediaPipe-to-AMASS-Core11 approximation."""

    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("source_fps must be positive and finite")
    world, valid, observed_valid = mediapipe_to_core11_world(
        sequence, aspect_ratio=aspect_ratio, config=config
    )
    source_confidence = np.zeros((len(sequence), len(JOINT_NAMES)), dtype=np.float32)
    indexed_confidence = np.nan_to_num(
        sequence[:, list(MEDIAPIPE_CORE_INDEX.values()), 3],
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clip(0.0, 1.0)
    source_confidence[:, 1:] = indexed_confidence
    source_confidence[:, 0] = np.minimum(
        indexed_confidence[:, 0], indexed_confidence[:, 1]
    )
    source_confidence[~valid] = 0.0
    leg_length = _leg_length(world, valid)
    transform, frame_method = _body_frame(world, valid, leg_length, config)
    centered = (world - world[:, 0:1]) / np.float32(leg_length)
    body = centered @ transform.T
    body[~valid] = 0.0
    coordinates, canonical_valid = _resample(
        body, valid, source_fps, config.canonical_fps
    )
    confidence, confidence_valid = _resample(
        source_confidence[..., None], valid, source_fps, config.canonical_fps
    )
    if not np.array_equal(confidence_valid, canonical_valid):
        raise ValueError("Detector confidence and coordinate validity diverged")
    confidence = confidence[..., 0]
    confidence[~canonical_valid] = 0.0
    return {
        "coordinates": coordinates,
        "valid": canonical_valid,
        "detector_confidence": confidence.astype(np.float32),
        "observed_core_joint_fraction": float(observed_valid.mean()),
        "leg_length_image_units": leg_length,
        "frame_method": frame_method,
        "odd_sign_anchored": False,
    }


def adapt_records(
    records: Sequence[Mapping], config: AdapterConfig = AdapterConfig()
) -> list[dict]:
    adapted = []
    for record in records:
        result = adapt_mediapipe_sequence(
            record["sequence"],
            source_fps=float(record["fps"]),
            aspect_ratio=float(record["aspect_ratio"]),
            config=config,
        )
        adapted.append({**{key: value for key, value in record.items() if key != "sequence"}, **result})
    return adapted


def build_probe_windows(
    records: Sequence[Mapping], config: AdapterConfig = AdapterConfig()
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Create 64-frame/32-stride windows and zero/invalid-pad short clips."""

    windows, token_valid, sequence_indices, rows = [], [], [], []
    for sequence_index, record in enumerate(records):
        coordinates = np.asarray(record["coordinates"], dtype=np.float32)
        valid = np.asarray(record["valid"], dtype=bool)
        starts = window_starts(len(coordinates)) if len(coordinates) >= WINDOW_FRAMES else [0]
        for start in starts:
            stop = min(start + WINDOW_FRAMES, len(coordinates))
            chunk = np.zeros((WINDOW_FRAMES, len(JOINT_NAMES), 3), dtype=np.float32)
            chunk_valid = np.zeros((WINDOW_FRAMES, len(JOINT_NAMES)), dtype=bool)
            observed = stop - start
            chunk[:observed] = coordinates[start:stop]
            chunk_valid[:observed] = valid[start:stop]
            patch_valid = (
                chunk_valid.reshape(
                    WINDOW_FRAMES // TIME_PATCH_FRAMES,
                    TIME_PATCH_FRAMES,
                    len(JOINT_NAMES),
                ).mean(axis=1)
                >= config.minimum_token_valid_fraction
            )
            mirrored_valid = patch_valid.copy()
            for left, right in MIRROR_PAIRS:
                mirrored_valid[:, left] = patch_valid[:, right]
                mirrored_valid[:, right] = patch_valid[:, left]
            patch_valid &= mirrored_valid
            windows.append(chunk)
            token_valid.append(patch_valid.reshape(-1))
            sequence_indices.append(sequence_index)
            rows.append(
                {
                    "window_id": f"{record['sequence_id']}:{start}",
                    "sequence_id": record["sequence_id"],
                    "video_id": record["video_id"],
                    "condition": record["condition"],
                    "start_frame": start,
                    "observed_frames": observed,
                }
            )
    return (
        np.stack(windows),
        np.stack(token_valid),
        np.asarray(sequence_indices, dtype=np.int64),
        pd.DataFrame(rows),
    )


def raw_coordinate_features(
    windows: np.ndarray,
    token_valid: np.ndarray,
    sequence_indices: np.ndarray,
    sequence_count: int,
) -> np.ndarray:
    """Pool body-frame coordinates into per-joint mean/std (66 features)."""

    valid = token_valid.reshape(-1, WINDOW_FRAMES // TIME_PATCH_FRAMES, len(JOINT_NAMES))
    valid = np.repeat(valid[:, :, None, :], TIME_PATCH_FRAMES, axis=2).reshape(
        -1, WINDOW_FRAMES, len(JOINT_NAMES)
    )
    total = np.zeros((sequence_count, len(JOINT_NAMES), 3), dtype=np.float64)
    square = np.zeros_like(total)
    count = np.zeros((sequence_count, len(JOINT_NAMES), 1), dtype=np.float64)
    for item, sequence_index in enumerate(sequence_indices):
        weights = valid[item, :, :, None]
        total[sequence_index] += (windows[item] * weights).sum(axis=0)
        square[sequence_index] += (np.square(windows[item]) * weights).sum(axis=0)
        count[sequence_index] += weights.sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(square, count, out=np.zeros_like(square), where=count > 0) - np.square(mean)
    return np.concatenate([mean.reshape(sequence_count, -1), np.sqrt(np.maximum(variance, 0)).reshape(sequence_count, -1)], axis=1).astype(np.float32)


def validity_only_features(
    token_valid: np.ndarray,
    sequence_indices: np.ndarray,
    sequence_count: int,
) -> np.ndarray:
    """Average the paired-valid token pattern without using any coordinates."""

    total = np.zeros((sequence_count, token_valid.shape[1]), dtype=np.float64)
    count = np.zeros(sequence_count, dtype=np.float64)
    for item, sequence_index in enumerate(sequence_indices):
        total[sequence_index] += token_valid[item]
        count[sequence_index] += 1
    return (total / count[:, None]).astype(np.float32)


@torch.inference_mode()
def parity_sequence_features(
    encoder: OrbitEncoder,
    windows: np.ndarray,
    token_valid: np.ndarray,
    sequence_indices: np.ndarray,
    sequence_count: int,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Pool tokenwise even/odd means and standard deviations into 4*D features."""

    device = torch.device(device)
    encoder = encoder.to(device).eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    dimension = encoder.config.embed_dim
    totals = np.zeros((2, sequence_count, dimension), dtype=np.float64)
    squares = np.zeros_like(totals)
    counts = np.zeros(sequence_count, dtype=np.float64)
    for start in range(0, len(windows), batch_size):
        stop = min(start + batch_size, len(windows))
        batch = torch.from_numpy(windows[start:stop]).to(device)
        orbit = lift_orbit(batch, encoder.config.mirror_pairs, encoder.config.mirror_channel)
        first, second = encoder(orbit)
        channels = ((first + second) * 0.5, (first - second) * 0.5)
        mask = torch.from_numpy(token_valid[start:stop]).to(device=device, dtype=first.dtype)
        for item, sequence_index in enumerate(sequence_indices[start:stop]):
            weight = mask[item, :, None]
            counts[sequence_index] += float(weight.sum().cpu())
            for parity, values in enumerate(channels):
                selected = values[item] * weight
                totals[parity, sequence_index] += selected.sum(dim=0).double().cpu().numpy()
                squares[parity, sequence_index] += (values[item].square() * weight).sum(dim=0).double().cpu().numpy()
    if np.any(counts == 0):
        raise ValueError("At least one sequence has no valid Core11 tokens")
    means = totals / counts[None, :, None]
    variances = squares / counts[None, :, None] - np.square(means)
    features = np.concatenate(
        [means[0], np.sqrt(np.maximum(variances[0], 0)), means[1], np.sqrt(np.maximum(variances[1], 0))],
        axis=1,
    )
    return features.astype(np.float32)


def frozen_target_encoder(checkpoint_path: str | Path) -> tuple[OrbitEncoder, dict]:
    model, _, metadata = load_checkpoint(Path(checkpoint_path))
    encoder = model.target_encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder, metadata


def random_target_encoder(metadata: Mapping, seed: int) -> OrbitEncoder:
    config = metadata["train_config"]
    from .gait_parity_jepa import TrainConfig

    model = build_model(TrainConfig(**config), str(metadata["variant"]), seed)
    encoder = model.target_encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    return encoder


def nested_ridge_probes(
    feature_sets: Mapping[str, np.ndarray],
    labels: Sequence[str],
    video_ids: Sequence[str],
    *,
    seed: int = 42,
    outer_splits: int = 5,
    inner_splits: int = 4,
    alphas: Sequence[float] = DEFAULT_ALPHAS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run identical nested sequence-stratified folds for every representation."""

    labels = np.asarray(labels)
    video_ids = np.asarray(video_ids)
    outer = list(
        StratifiedKFold(outer_splits, shuffle=True, random_state=seed).split(
            np.zeros(len(labels)), labels
        )
    )
    fold_rows, prediction_rows = [], []
    for representation, features in feature_sets.items():
        features = np.asarray(features)
        if features.shape[0] != len(labels) or not np.isfinite(features).all():
            raise ValueError(f"Invalid feature matrix for {representation}: {features.shape}")
        for fold, (train, test) in enumerate(outer):
            pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("ridge", RidgeClassifier(class_weight="balanced", solver="lsqr")),
                ]
            )
            inner = StratifiedKFold(inner_splits, shuffle=True, random_state=seed + fold + 1)
            search = GridSearchCV(
                pipeline,
                {"ridge__alpha": list(alphas)},
                scoring="balanced_accuracy",
                cv=inner,
                n_jobs=1,
                refit=True,
            )
            search.fit(features[train], labels[train])
            predicted = search.predict(features[test])
            shared_videos = set(video_ids[train]) & set(video_ids[test])
            fold_rows.append(
                {
                    "representation": representation,
                    "fold": fold,
                    "best_alpha": float(search.best_params_["ridge__alpha"]),
                    "accuracy": accuracy_score(labels[test], predicted),
                    "balanced_accuracy": balanced_accuracy_score(labels[test], predicted),
                    "macro_f1": f1_score(labels[test], predicted, average="macro", zero_division=0),
                    "shared_train_test_videos": len(shared_videos),
                }
            )
            prediction_rows.extend(
                {
                    "representation": representation,
                    "fold": fold,
                    "row_index": int(index),
                    "video_id": video_ids[index],
                    "true_condition": labels[index],
                    "predicted_condition": prediction,
                }
                for index, prediction in zip(test, predicted)
            )
    folds = pd.DataFrame(fold_rows)
    summary = (
        folds.groupby("representation", sort=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_std=("balanced_accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            median_best_alpha=("best_alpha", "median"),
            mean_shared_train_test_videos=("shared_train_test_videos", "mean"),
        )
        .reset_index()
    )
    summary.insert(
        1,
        "feature_dimension",
        summary["representation"].map(
            {name: values.shape[1] for name, values in feature_sets.items()}
        ),
    )
    return folds, summary, pd.DataFrame(prediction_rows)


def run_probe_experiment(
    *,
    pose_dir: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    device: str = "cpu",
    batch_size: int = 32,
    seed: int = 42,
    random_seeds: Sequence[int] = (7, 19, 31),
) -> dict:
    """Run the complete frozen-adapter, frozen-encoder, nested-probe experiment."""

    records = load_gavd_pose_records(pose_dir)
    adapted = adapt_records(records)
    windows, token_valid, sequence_indices, window_table = build_probe_windows(adapted)
    feature_sets = {
        "validity_only": validity_only_features(
            token_valid, sequence_indices, len(adapted)
        ),
        "raw_core11": raw_coordinate_features(
            windows, token_valid, sequence_indices, len(adapted)
        )
    }
    checkpoint_paths = sorted(Path(checkpoint_dir).glob("*_best.pt"))
    if len(checkpoint_paths) != 2:
        raise ValueError(f"Expected exactly two best checkpoints in {checkpoint_dir}")
    checkpoint_metadata = []
    for checkpoint_path in checkpoint_paths:
        encoder, metadata = frozen_target_encoder(checkpoint_path)
        variant = str(metadata["variant"])
        feature_sets[f"ema_{variant}"] = parity_sequence_features(
            encoder,
            windows,
            token_valid,
            sequence_indices,
            len(adapted),
            device=device,
            batch_size=batch_size,
        )
        for random_seed in random_seeds:
            random_encoder = random_target_encoder(metadata, int(random_seed))
            feature_sets[f"random_{variant}_seed{random_seed}"] = parity_sequence_features(
                random_encoder,
                windows,
                token_valid,
                sequence_indices,
                len(adapted),
                device=device,
                batch_size=batch_size,
            )
        checkpoint_metadata.append(metadata)

    labels = np.asarray([record["condition"] for record in adapted])
    video_ids = np.asarray([record["video_id"] for record in adapted])
    condition_video_counts = pd.DataFrame(
        {"condition": labels, "video_id": video_ids}
    ).groupby("condition")["video_id"].nunique()
    grouped_status = (
        "blocked_at_least_one_class_has_fewer_than_two_source_videos"
        if int(condition_video_counts.min()) < 2
        else "not_evaluated"
    )
    folds, summary, predictions = nested_ridge_probes(
        feature_sets, labels, video_ids, seed=seed
    )
    summary.insert(1, "cohort", f"all{len(adapted)}_zero_invalid_padded")
    summary.insert(2, "task", "five_class_condition")
    summary.insert(
        3,
        "evaluation_scope",
        "within_corpus_source_confounded_sequence_stratified",
    )
    summary.insert(
        4,
        "grouped_generalization_status",
        grouped_status,
    )
    sequence_table = pd.DataFrame(
        [
            {
                "row_index": index,
                "sequence_id": record["sequence_id"],
                "video_id": record["video_id"],
                "condition": record["condition"],
                "source_fps": record["fps"],
                "canonical_frames": len(record["coordinates"]),
                "short_clip_padding_frames": max(
                    WINDOW_FRAMES - len(record["coordinates"]), 0
                ),
                "observed_core_joint_fraction": record[
                    "observed_core_joint_fraction"
                ],
                "usable_core_joint_fraction": float(record["valid"].mean()),
                "paired_valid_token_fraction": float(
                    token_valid[sequence_indices == index].mean()
                ),
                "frame_method": record["frame_method"],
                "leg_length_image_units": record["leg_length_image_units"],
            }
            for index, record in enumerate(adapted)
        ]
    )
    condition_table = (
        sequence_table.groupby("condition", sort=True)
        .agg(
            sequences=("sequence_id", "size"),
            source_videos=("video_id", "nunique"),
            mean_observed_core_joint_fraction=("observed_core_joint_fraction", "mean"),
            mean_usable_core_joint_fraction=("usable_core_joint_fraction", "mean"),
            mean_paired_valid_token_fraction=("paired_valid_token_fraction", "mean"),
            padded_sequences=("short_clip_padding_frames", lambda value: int((value > 0).sum())),
        )
        .reset_index()
    )
    strict_mask = sequence_table["short_clip_padding_frames"].to_numpy() == 0
    strict_indices = np.flatnonzero(strict_mask)
    strict_feature_sets = {
        name: values[strict_mask] for name, values in feature_sets.items()
    }
    strict_folds, strict_summary, strict_predictions = nested_ridge_probes(
        strict_feature_sets,
        labels[strict_mask],
        video_ids[strict_mask],
        seed=seed,
    )
    strict_label = f"strict{int(strict_mask.sum())}_no_short_clip_padding"
    strict_summary.insert(1, "cohort", strict_label)
    strict_summary.insert(2, "task", "five_class_condition")
    strict_summary.insert(
        3,
        "evaluation_scope",
        "within_corpus_source_confounded_sequence_stratified",
    )
    strict_summary.insert(
        4,
        "grouped_generalization_status",
        grouped_status,
    )
    strict_predictions["row_index"] = strict_indices[
        strict_predictions["row_index"].to_numpy()
    ]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequence_table.to_csv(output_dir / "sequence_adapter_table.csv", index=False)
    condition_table.to_csv(output_dir / "adapter_condition_summary.csv", index=False)
    window_table.to_csv(output_dir / "window_table.csv", index=False)
    folds.to_csv(output_dir / "nested_probe_folds.csv", index=False)
    summary.to_csv(output_dir / "nested_probe_summary.csv", index=False)
    predictions.to_csv(output_dir / "nested_probe_predictions.csv", index=False)
    strict_folds.to_csv(output_dir / f"{strict_label}_nested_probe_folds.csv", index=False)
    strict_summary.to_csv(output_dir / f"{strict_label}_nested_probe_summary.csv", index=False)
    strict_predictions.to_csv(
        output_dir / f"{strict_label}_nested_probe_predictions.csv", index=False
    )
    np.savez_compressed(output_dir / "sequence_features.npz", **feature_sets)
    return {
        "records": records,
        "adapted": adapted,
        "windows": windows,
        "token_valid": token_valid,
        "window_table": window_table,
        "feature_sets": feature_sets,
        "folds": folds,
        "summary": summary,
        "predictions": predictions,
        "strict_folds": strict_folds,
        "strict_summary": strict_summary,
        "strict_predictions": strict_predictions,
        "sequence_table": sequence_table,
        "condition_table": condition_table,
        "checkpoint_metadata": checkpoint_metadata,
        "output_dir": output_dir,
    }
