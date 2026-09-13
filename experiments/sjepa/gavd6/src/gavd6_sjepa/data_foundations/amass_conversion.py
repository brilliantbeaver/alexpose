#!/usr/bin/env python3
"""Convert AMASS Extended SMPL+H sequences to GaitParity core-11 coordinates.

This script intentionally uses AMASS's reference ``human_body_prior.BodyModel``
path so that all 16 shape coefficients and all 8 DMPL coefficients are applied.
Licensed SMPL+H and DMPL model files stay outside this repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import numpy as np


CONVERTER_VERSION = "1.0.0"
CONVERTER_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
SCHEMA = "core11-v1"
COORDINATE_FRAME = "gait-parity-body-v1"
VALIDITY_SEMANTICS = "finite required AMASS parameters and finite SMPL+H output; not visibility"

CORE11_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_forefoot",
    "right_forefoot",
)
CHANNEL_NAMES = ("forward", "vertical_up", "mediolateral")

# Extended SMPL+H joint-regressor indices used by AMASS.
SMPLH_JOINT_INDEX = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
}

# Official SMPL-H surface landmarks from smplx/vertex_ids.py. Each forefoot is
# the midpoint of the big- and small-toe landmarks, rather than SMPL joint 10/11.
SMPLH_VERTEX_INDEX = {
    "left_big_toe": 3216,
    "left_small_toe": 3226,
    "left_heel": 3387,
    "right_big_toe": 6617,
    "right_small_toe": 6624,
    "right_heel": 6787,
}

AMASS_UP_WORLD = np.array([0.0, 0.0, 1.0], dtype=np.float64)
GAUGE_NEUTRAL_COORDINATE_FRAME = "gauge-neutral-travel-v1"


@dataclass(frozen=True)
class ConversionConfig:
    canonical_fps: float = 30.0
    batch_size: int = 256
    min_travel_m: float = 0.10
    min_travel_straightness: float = 0.20
    min_abs_lateral_hip_alignment: float = 0.95
    min_leg_length_m: float = 0.20
    max_leg_length_m: float = 2.00
    forward_policy: str = "travel-or-hips"

    def __post_init__(self) -> None:
        if not np.isclose(self.canonical_fps, 30.0, rtol=0.0, atol=1e-12):
            raise ValueError("core11-v1 freezes canonical_fps at exactly 30")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not np.isfinite(self.min_travel_m) or self.min_travel_m < 0:
            raise ValueError("min_travel_m must be finite and nonnegative")
        if not np.isfinite(self.min_travel_straightness) or not (
            0.0 <= self.min_travel_straightness <= 1.0
        ):
            raise ValueError("min_travel_straightness must be finite and in [0, 1]")
        if not np.isfinite(self.min_abs_lateral_hip_alignment) or not (
            0.0 <= self.min_abs_lateral_hip_alignment <= 1.0
        ):
            raise ValueError("min_abs_lateral_hip_alignment must be finite and in [0, 1]")
        if not 0 < self.min_leg_length_m < self.max_leg_length_m:
            raise ValueError("leg-length bounds must satisfy 0 < minimum < maximum")
        if self.forward_policy not in {
            "travel-or-hips",
            "require-travel",
            "gauge-neutral-travel",
        }:
            raise ValueError(f"unsupported forward_policy: {self.forward_policy!r}")


@dataclass(frozen=True)
class AmassSequence:
    poses: np.ndarray
    trans: np.ndarray
    betas: np.ndarray
    dmpls: np.ndarray
    gender: str
    mocap_framerate: float


@dataclass(frozen=True)
class ForwardFrame:
    forward_world: np.ndarray
    up_world: np.ndarray
    lateral_world: np.ndarray
    method: str
    displacement_m: float
    path_length_m: float
    straightness: float
    linearity: float
    lateral_hip_alignment: float | None

    @property
    def world_to_body_transform(self) -> np.ndarray:
        # Column-vector convention: body = T_world_to_body @ world.
        # Channels are stored [forward, up, lateral], while the physical
        # right-handed basis is ordered [forward, lateral, up]. The channel
        # transform is therefore orthogonal but is not itself a proper rotation.
        return np.stack(
            [self.forward_world, self.up_world, self.lateral_world], axis=0
        )


class ConversionError(RuntimeError):
    """A sequence cannot be converted without violating the data contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _scalar(archive: np.lib.npyio.NpzFile, key: str) -> Any:
    if key not in archive.files:
        raise ConversionError(f"missing required archive key: {key}")
    value = archive[key]
    if value.size != 1:
        raise ConversionError(f"{key} must be scalar, got shape {value.shape}")
    return value.item()


def _normalize_gender(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    gender = str(value).strip().lower()
    if gender not in {"male", "female"}:
        raise ConversionError(f"unsupported AMASS gender {gender!r}; expected male or female")
    return gender


def load_amass_sequence(path: Path) -> AmassSequence:
    """Load and structurally validate one standard AMASS ``*_poses.npz`` file."""

    with np.load(path, allow_pickle=False) as archive:
        missing = {"poses", "trans", "betas", "dmpls", "gender", "mocap_framerate"} - set(
            archive.files
        )
        if missing:
            raise ConversionError(f"missing required archive keys: {sorted(missing)}")

        poses = np.asarray(archive["poses"], dtype=np.float32)
        trans = np.asarray(archive["trans"], dtype=np.float32)
        betas = np.asarray(archive["betas"], dtype=np.float32).reshape(-1)
        dmpls = np.asarray(archive["dmpls"], dtype=np.float32)
        gender = _normalize_gender(_scalar(archive, "gender"))
        fps = float(_scalar(archive, "mocap_framerate"))

    if poses.ndim != 2 or poses.shape[1] != 156:
        raise ConversionError(f"poses must have shape [T, 156], got {poses.shape}")
    if poses.shape[0] < 1:
        raise ConversionError("poses contains no frames")
    if trans.shape != (poses.shape[0], 3):
        raise ConversionError(
            f"trans must have shape [{poses.shape[0]}, 3], got {trans.shape}"
        )
    if dmpls.ndim != 2 or dmpls.shape[0] != poses.shape[0] or dmpls.shape[1] < 8:
        raise ConversionError(
            f"dmpls must have shape [{poses.shape[0]}, >=8], got {dmpls.shape}"
        )
    if betas.size < 16:
        raise ConversionError(f"betas must contain at least 16 values, got {betas.size}")
    if not np.isfinite(betas[:16]).all():
        raise ConversionError("the required 16 beta coefficients are not all finite")
    if not np.isfinite(fps) or fps <= 0:
        raise ConversionError(f"invalid mocap_framerate: {fps}")

    return AmassSequence(
        poses=poses,
        trans=trans,
        betas=betas[:16].copy(),
        dmpls=dmpls[:, :8].copy(),
        gender=gender,
        mocap_framerate=fps,
    )


def _package_version(distribution_names: Sequence[str]) -> str:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


class HumanBodyPriorBackend:
    """Batched Extended SMPL+H + DMPL forward kinematics for AMASS."""

    def __init__(self, body_model_root: Path, device: str, batch_size: int) -> None:
        try:
            import torch
            from human_body_prior.body_model.body_model import BodyModel
        except ImportError as exc:
            raise RuntimeError(
                "human_body_prior is required on HAIC. Install the official package in the "
                "GaitParity environment before running this converter."
            ) from exc

        self.torch = torch
        self.BodyModel = BodyModel
        self.root = body_model_root.resolve()
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"requested CUDA device {self.device}, but torch.cuda.is_available() is false"
            )
        self.batch_size = batch_size
        self.package_version = _package_version(("human-body-prior", "human_body_prior"))
        backend_source = inspect.getsourcefile(BodyModel)
        self.backend_source_sha256 = (
            sha256_file(Path(backend_source)) if backend_source is not None else "unknown"
        )
        body_model_module = importlib.import_module("human_body_prior.body_model.body_model")
        lbs_source = inspect.getsourcefile(body_model_module.lbs)
        self.lbs_source_sha256 = (
            sha256_file(Path(lbs_source)) if lbs_source is not None else "unknown"
        )
        self._models: dict[str, Any] = {}
        self._model_info: dict[str, dict[str, str]] = {}

    def _paths(self, gender: str) -> tuple[Path, Path]:
        body_path = self.root / "smplh" / gender / "model.npz"
        dmpl_path = self.root / "dmpls" / gender / "model.npz"
        for kind, path in (("SMPL+H", body_path), ("DMPL", dmpl_path)):
            if not path.is_file():
                raise ConversionError(f"missing licensed {kind} model file: {path}")
        return body_path, dmpl_path

    def model_info(self, gender: str) -> Mapping[str, str]:
        self._load_model(gender)
        return self._model_info[gender]

    def _load_model(self, gender: str) -> Any:
        if gender in self._models:
            return self._models[gender]

        body_path, dmpl_path = self._paths(gender)
        model = self.BodyModel(
            bm_fname=str(body_path),
            num_betas=16,
            num_dmpls=8,
            dmpl_fname=str(dmpl_path),
        ).to(self.device)
        model.eval()

        if model.model_type != "smplh":
            raise ConversionError(f"body model type is {model.model_type!r}, expected 'smplh'")
        if model.shapedirs.shape[-1] != 16:
            raise ConversionError(
                f"body model exposes {model.shapedirs.shape[-1]} betas, expected 16"
            )
        if model.dmpldirs.shape[-1] != 8:
            raise ConversionError(
                f"DMPL model exposes {model.dmpldirs.shape[-1]} components, expected 8"
            )
        if model.shapedirs.shape[:2] != model.dmpldirs.shape[:2]:
            raise ConversionError(
                "SMPL+H shape directions and DMPL directions use different vertex topologies"
            )
        num_vertices = int(model.init_v_template.shape[1])
        if num_vertices != 6890 or max(SMPLH_VERTEX_INDEX.values()) >= num_vertices:
            raise ConversionError(
                f"body model has {num_vertices} vertices; expected the 6890-vertex SMPL-H topology"
            )
        if int(model.J_regressor.shape[0]) <= max(SMPLH_JOINT_INDEX.values()):
            raise ConversionError("SMPL+H joint regressor does not contain the required core joints")

        self._models[gender] = model
        self._model_info[gender] = {
            "backend": "human_body_prior.BodyModel",
            "backend_version": self.package_version,
            "backend_source_sha256": self.backend_source_sha256,
            "lbs_source_sha256": self.lbs_source_sha256,
            "torch_version": self.torch.__version__,
            "execution_device": str(self.device),
            "dtype": "float32",
            "body_model_family": "Extended SMPL+H",
            "body_model_sha256": sha256_file(body_path),
            "dmpl_model_sha256": sha256_file(dmpl_path),
        }
        return model

    def forward(self, sequence: AmassSequence) -> tuple[np.ndarray, np.ndarray]:
        """Return world-space core-11 coordinates and per-joint finite validity."""

        torch = self.torch
        model = self._load_model(sequence.gender)
        frame_valid = (
            np.isfinite(sequence.poses).all(axis=1)
            & np.isfinite(sequence.trans).all(axis=1)
            & np.isfinite(sequence.dmpls).all(axis=1)
        )
        valid_indices = np.flatnonzero(frame_valid)
        if valid_indices.size == 0:
            raise ConversionError("no frame has finite pose, translation, and DMPL parameters")

        num_frames = sequence.poses.shape[0]
        coordinates = np.zeros((num_frames, len(CORE11_NAMES), 3), dtype=np.float32)
        validity = np.zeros((num_frames, len(CORE11_NAMES)), dtype=bool)

        for start in range(0, valid_indices.size, self.batch_size):
            indices = valid_indices[start : start + self.batch_size]
            poses = torch.from_numpy(sequence.poses[indices]).to(self.device)
            trans = torch.from_numpy(sequence.trans[indices]).to(self.device)
            dmpls = torch.from_numpy(sequence.dmpls[indices]).to(self.device)
            betas = torch.from_numpy(
                np.repeat(sequence.betas[None, :], len(indices), axis=0)
            ).to(self.device)

            with torch.inference_mode():
                result = model(
                    root_orient=poses[:, 0:3],
                    pose_body=poses[:, 3:66],
                    pose_hand=poses[:, 66:156],
                    betas=betas,
                    dmpls=dmpls,
                    trans=trans,
                )
                joints = result.Jtr
                vertices = result.v
                core = torch.stack(
                    [
                        joints[:, SMPLH_JOINT_INDEX["pelvis"]],
                        joints[:, SMPLH_JOINT_INDEX["left_hip"]],
                        joints[:, SMPLH_JOINT_INDEX["right_hip"]],
                        joints[:, SMPLH_JOINT_INDEX["left_knee"]],
                        joints[:, SMPLH_JOINT_INDEX["right_knee"]],
                        joints[:, SMPLH_JOINT_INDEX["left_ankle"]],
                        joints[:, SMPLH_JOINT_INDEX["right_ankle"]],
                        vertices[:, SMPLH_VERTEX_INDEX["left_heel"]],
                        vertices[:, SMPLH_VERTEX_INDEX["right_heel"]],
                        0.5
                        * (
                            vertices[:, SMPLH_VERTEX_INDEX["left_big_toe"]]
                            + vertices[:, SMPLH_VERTEX_INDEX["left_small_toe"]]
                        ),
                        0.5
                        * (
                            vertices[:, SMPLH_VERTEX_INDEX["right_big_toe"]]
                            + vertices[:, SMPLH_VERTEX_INDEX["right_small_toe"]]
                        ),
                    ],
                    dim=1,
                )

            chunk = core.detach().cpu().numpy().astype(np.float32, copy=False)
            chunk_valid = np.isfinite(chunk).all(axis=2)
            chunk_valid &= chunk_valid[:, 0:1]
            chunk[~chunk_valid] = 0.0
            coordinates[indices] = chunk
            validity[indices] = chunk_valid

        return coordinates, validity


def _safe_unit(vector: np.ndarray, label: str, epsilon: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= epsilon:
        raise ConversionError(f"cannot define {label}: vector norm is {norm}")
    return vector / norm


def estimate_forward_frame(
    coordinates_world: np.ndarray,
    valid: np.ndarray,
    *,
    min_travel_m: float,
    min_travel_straightness: float,
    min_abs_lateral_hip_alignment: float,
    forward_policy: str,
) -> ForwardFrame:
    """Estimate one auditable trial-level frame without outcome labels."""

    up = AMASS_UP_WORLD.copy()
    pelvis_ok = valid[:, 0] & np.isfinite(coordinates_world[:, 0]).all(axis=1)
    pelvis = coordinates_world[pelvis_ok, 0].astype(np.float64)
    displacement = path_length = straightness = linearity = 0.0
    forward: np.ndarray | None = None
    method = ""

    if len(pelvis) >= 2:
        horizontal = pelvis - np.outer(pelvis @ up, up)
        endpoint_width = max(1, min(15, len(horizontal) // 10))
        net = np.median(horizontal[-endpoint_width:], axis=0) - np.median(
            horizontal[:endpoint_width], axis=0
        )
        displacement = float(np.linalg.norm(net))
        steps = np.diff(horizontal, axis=0)
        path_length = float(np.linalg.norm(steps, axis=1).sum())
        straightness = displacement / max(path_length, 1e-12)

        centered = horizontal - np.mean(horizontal, axis=0, keepdims=True)
        _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        variance = singular_values**2
        linearity = float(variance[0] / max(variance.sum(), 1e-12))

        if displacement >= min_travel_m and straightness >= min_travel_straightness:
            forward = vh[0]
            forward = forward - np.dot(forward, up) * up
            if np.dot(forward, net) < 0:
                forward = -forward
            forward = _safe_unit(forward, "pelvis-travel direction")
            method = "pelvis_travel_pca_signed_by_robust_displacement"

    if forward is None:
        if forward_policy in {"require-travel", "gauge-neutral-travel"}:
            raise ConversionError(
                "pelvis travel is insufficient or ambiguous under the traveling-only "
                f"forward policy {forward_policy!r}"
            )
        hips_ok = valid[:, 1] & valid[:, 2]
        hip_lateral = coordinates_world[hips_ok, 1] - coordinates_world[hips_ok, 2]
        hip_lateral = hip_lateral.astype(np.float64)
        hip_lateral -= np.outer(hip_lateral @ up, up)
        if len(hip_lateral) == 0:
            raise ConversionError("cannot define fallback facing: no valid bilateral hip frame")
        anatomical_left = _safe_unit(
            np.median(hip_lateral, axis=0), "anatomical left hip axis"
        )
        forward = _safe_unit(np.cross(anatomical_left, up), "hip-facing direction")
        method = "hip_facing_fallback"

    lateral = _safe_unit(np.cross(up, forward), "mediolateral direction")
    forward = _safe_unit(np.cross(lateral, up), "orthogonalized forward direction")
    if forward_policy == "gauge-neutral-travel":
        # This policy intentionally never reads side-named joints while choosing
        # or validating orientation. Its unresolved lateral sign is a sensor
        # chart choice, kept separate from the later semantic permutation P.
        hip_alignment = None
        method = "gauge_neutral_pelvis_travel_pca_signed_by_net_displacement"
    else:
        hips_ok = valid[:, 1] & valid[:, 2]
        if not hips_ok.any():
            raise ConversionError(
                "cannot verify body-frame laterality: no valid bilateral hip frame"
            )
        hip_axis = coordinates_world[hips_ok, 1] - coordinates_world[hips_ok, 2]
        hip_axis = np.median(hip_axis, axis=0).astype(np.float64)
        hip_axis -= np.dot(hip_axis, up) * up
        anatomical_left = _safe_unit(hip_axis, "hip alignment")
        hip_alignment = float(np.dot(lateral, anatomical_left))
        if (
            method.startswith("pelvis_travel")
            and abs(hip_alignment) < min_abs_lateral_hip_alignment
        ):
            if forward_policy == "require-travel":
                raise ConversionError(
                    "pelvis travel is not sufficiently orthogonal to the anatomical hip axis"
                )
            forward = _safe_unit(np.cross(anatomical_left, up), "hip-facing direction")
            lateral = _safe_unit(np.cross(up, forward), "mediolateral direction")
            hip_alignment = float(np.dot(lateral, anatomical_left))
            method = "hip_facing_fallback_due_to_travel_anatomy_misalignment"

    physical_basis = np.stack([forward, lateral, up], axis=1)
    if not np.allclose(physical_basis.T @ physical_basis, np.eye(3), atol=1e-6):
        raise ConversionError("constructed body-frame axes are not orthonormal")
    if not np.isclose(np.linalg.det(physical_basis), 1.0, atol=1e-6):
        raise ConversionError("constructed physical [forward, lateral, up] basis is not right-handed")

    return ForwardFrame(
        forward_world=forward,
        up_world=up,
        lateral_world=lateral,
        method=method,
        displacement_m=displacement,
        path_length_m=path_length,
        straightness=straightness,
        linearity=linearity,
        lateral_hip_alignment=hip_alignment,
    )


def robust_leg_length_m(
    coordinates_world: np.ndarray,
    valid: np.ndarray,
    *,
    minimum: float,
    maximum: float,
) -> float:
    required = valid[:, 1] & valid[:, 2] & valid[:, 3] & valid[:, 4] & valid[:, 5] & valid[:, 6]
    if not required.any():
        raise ConversionError("no frame has all bilateral hip, knee, and ankle joints valid")

    xyz = coordinates_world[required]
    left = np.linalg.norm(xyz[:, 1] - xyz[:, 3], axis=1) + np.linalg.norm(
        xyz[:, 3] - xyz[:, 5], axis=1
    )
    right = np.linalg.norm(xyz[:, 2] - xyz[:, 4], axis=1) + np.linalg.norm(
        xyz[:, 4] - xyz[:, 6], axis=1
    )
    scale = float(np.median(0.5 * (left + right)))
    if not np.isfinite(scale) or not minimum <= scale <= maximum:
        raise ConversionError(
            f"robust bilateral leg length {scale:.6g} m is outside [{minimum}, {maximum}]"
        )
    return scale


def transform_to_body_frame(
    coordinates_world: np.ndarray,
    valid: np.ndarray,
    frame: ForwardFrame,
) -> np.ndarray:
    effective_valid = valid & valid[:, 0:1]
    pelvis = coordinates_world[:, 0]
    centered = coordinates_world - pelvis[:, None, :]
    body = centered @ frame.world_to_body_transform.T
    body[~effective_valid] = 0.0
    return body.astype(np.float32)


def canonical_times(num_frames: int, source_fps: float, target_fps: float) -> np.ndarray:
    if num_frames < 1:
        raise ConversionError("cannot resample an empty sequence")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ConversionError(f"source_fps must be positive and finite, got {source_fps}")
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ConversionError(f"target_fps must be positive and finite, got {target_fps}")
    duration = (num_frames - 1) / source_fps
    count = int(np.floor(duration * target_fps + 1e-10)) + 1
    return np.arange(count, dtype=np.float64) / target_fps


def resample_at_times(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    source_fps: float,
    target_times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly resample all joints together; both brackets must be valid."""

    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError(
            f"expected values [T,J,C] and valid [T,J], got {values.shape} and {valid.shape}"
        )
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ConversionError(f"source_fps must be positive and finite, got {source_fps}")
    if target_times.ndim != 1:
        raise ConversionError(f"target_times must be one-dimensional, got {target_times.shape}")
    source_times = np.arange(values.shape[0], dtype=np.float64) / source_fps
    if not np.isfinite(target_times).all():
        raise ConversionError("target resampling times must all be finite")
    if target_times.size and (
        target_times[0] < -1e-10
        or target_times[-1] > source_times[-1] + 1e-10
        or np.any(np.diff(target_times) < 0)
    ):
        raise ConversionError("target resampling grid must be sorted and remain inside the source sequence")

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

    resampled = (1.0 - weight[:, None, None]) * values[left] + weight[:, None, None] * values[
        right
    ]
    resampled_valid = valid[left] & valid[right] & np.isfinite(resampled).all(axis=2)
    resampled[~resampled_valid] = 0.0
    return resampled.astype(np.float32), resampled_valid


def reconstruct_world(
    body_coordinates_m: np.ndarray,
    pelvis_world_m: np.ndarray,
    world_to_body_transform: np.ndarray,
) -> np.ndarray:
    """Reconstruct row-vector world coordinates for replay and tests."""

    centered_world = body_coordinates_m @ world_to_body_transform
    return centered_world + pelvis_world_m[:, None, :]


def convert_sequence_arrays(
    coordinates_world: np.ndarray,
    valid: np.ndarray,
    *,
    source_fps: float,
    config: ConversionConfig,
) -> dict[str, Any]:
    """Apply the frozen frame, robust scale, and canonical temporal grid."""

    if coordinates_world.shape[1:] != (len(CORE11_NAMES), 3):
        raise ConversionError(
            f"world coordinates must have shape [T, 11, 3], got {coordinates_world.shape}"
        )
    if valid.shape != coordinates_world.shape[:2]:
        raise ConversionError(f"valid must have shape {coordinates_world.shape[:2]}, got {valid.shape}")
    valid = np.asarray(valid, dtype=bool).copy()
    valid &= valid[:, 0:1]

    frame = estimate_forward_frame(
        coordinates_world,
        valid,
        min_travel_m=config.min_travel_m,
        min_travel_straightness=config.min_travel_straightness,
        min_abs_lateral_hip_alignment=config.min_abs_lateral_hip_alignment,
        forward_policy=config.forward_policy,
    )
    leg_length = robust_leg_length_m(
        coordinates_world,
        valid,
        minimum=config.min_leg_length_m,
        maximum=config.max_leg_length_m,
    )
    body_native_m = transform_to_body_frame(coordinates_world, valid, frame)
    times = canonical_times(len(coordinates_world), source_fps, config.canonical_fps)
    coordinates_m, canonical_valid = resample_at_times(
        body_native_m,
        valid,
        source_fps=source_fps,
        target_times=times,
    )
    pelvis_world, pelvis_valid = resample_at_times(
        coordinates_world[:, 0:1],
        valid[:, 0:1],
        source_fps=source_fps,
        target_times=times,
    )
    if not np.array_equal(pelvis_valid[:, 0], canonical_valid[:, 0]):
        raise ConversionError("pelvis validity diverged between coordinate and provenance streams")

    coordinates = coordinates_m / np.float32(leg_length)
    coordinates[~canonical_valid] = 0.0
    return {
        "coordinates": coordinates.astype(np.float32),
        "coordinates_m": coordinates_m,
        "valid": canonical_valid,
        "canonical_times_s": times,
        "pelvis_world_m": pelvis_world[:, 0],
        "leg_length_m": leg_length,
        "frame": frame,
    }


def _safe_relative_path(root: Path, relative_text: str) -> tuple[Path, PurePosixPath]:
    relative = PurePosixPath(relative_text)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ConversionError(f"unsafe inventory relative_path: {relative_text!r}")
    path = (root / Path(*relative.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ConversionError(f"source path escapes AMASS root: {relative_text!r}") from exc
    return path, relative


def _output_relative_path(relative: PurePosixPath) -> PurePosixPath:
    if not relative.name.endswith("_poses.npz"):
        raise ConversionError(f"AMASS filename must end with _poses.npz: {relative}")
    stem = relative.name[: -len("_poses.npz")]
    return relative.with_name(f"{stem}_core11.npz")


def _atomic_save_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp.npz"
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


MANIFEST_FIELDS = (
    "relative_path",
    "tensor_relative_path",
    "source_dataset",
    "subject_id_candidate",
    "identity",
    "split",
    "motion_id",
    "gender",
    "native_fps",
    "native_frames",
    "canonical_fps",
    "canonical_frames",
    "source_sha256",
    "source_sha256_verified",
    "body_model_sha256",
    "dmpl_model_sha256",
    "schema",
    "coordinate_frame",
    "valid_fraction",
    "forward_method",
    "travel_displacement_m",
    "travel_path_length_m",
    "travel_straightness",
    "travel_linearity",
    "lateral_hip_alignment",
    "leg_length_m",
    "conversion_fingerprint",
    "status",
    "error",
)


def _coordinate_frame_name(config: ConversionConfig) -> str:
    return (
        GAUGE_NEUTRAL_COORDINATE_FRAME
        if config.forward_policy == "gauge-neutral-travel"
        else COORDINATE_FRAME
    )


def _configuration_fingerprint(config: ConversionConfig, model_info: Mapping[str, str]) -> str:
    payload = {
        "converter_version": CONVERTER_VERSION,
        "converter_source_sha256": CONVERTER_SOURCE_SHA256,
        "schema": SCHEMA,
        "coordinate_frame": _coordinate_frame_name(config),
        "joint_names": CORE11_NAMES,
        "channel_names": CHANNEL_NAMES,
        "smplh_joint_indices": SMPLH_JOINT_INDEX,
        "smplh_vertex_indices": SMPLH_VERTEX_INDEX,
        "forefoot_definition": "midpoint(big_toe,small_toe)",
        "amass_up_world": AMASS_UP_WORLD.tolist(),
        "config": asdict(config),
        "model_info": dict(model_info),
        "dmpls_used": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _existing_output_matches(
    path: Path,
    source_sha256: str,
    fingerprint: str,
    row: Mapping[str, str],
    require_verified_source: bool,
) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "coordinates",
                "coordinates_m",
                "valid",
                "canonical_times_s",
                "pelvis_world_m",
                "world_to_body_transform",
                "body_to_world_transform",
                "physical_basis_world",
                "leg_length_m",
                "joint_names",
                "channel_names",
                "provenance_json",
            }
            if not required <= set(archive.files):
                return False
            provenance = json.loads(str(archive["provenance_json"].item()))
            coordinates = np.asarray(archive["coordinates"])
            coordinates_m = np.asarray(archive["coordinates_m"])
            valid = np.asarray(archive["valid"])
            times = np.asarray(archive["canonical_times_s"])
            pelvis = np.asarray(archive["pelvis_world_m"])
            world_to_body = np.asarray(archive["world_to_body_transform"])
            body_to_world = np.asarray(archive["body_to_world_transform"])
            physical_basis = np.asarray(archive["physical_basis_world"])
            leg_length = float(archive["leg_length_m"].item())
            joint_names = tuple(archive["joint_names"].tolist())
            channel_names = tuple(archive["channel_names"].tolist())
        expected_frames = coordinates.shape[0] if coordinates.ndim == 3 else -1
        return (
            provenance["source"]["sha256"] == source_sha256
            and provenance["conversion"]["fingerprint"] == fingerprint
            and (
                not require_verified_source
                or provenance["source"].get("sha256_verified_against_current_file") is True
            )
            and all(
                provenance["source"].get(key, "") == row.get(key, "")
                for key in (
                    "relative_path",
                    "source_dataset",
                    "subject_id_candidate",
                    "identity",
                    "split",
                    "motion_id",
                )
            )
            and coordinates.shape == (expected_frames, len(CORE11_NAMES), 3)
            and coordinates_m.shape == coordinates.shape
            and coordinates.dtype == np.float32
            and coordinates_m.dtype == np.float32
            and valid.shape == coordinates.shape[:2]
            and valid.dtype == np.bool_
            and times.shape == (expected_frames,)
            and pelvis.shape == (expected_frames, 3)
            and world_to_body.shape == (3, 3)
            and body_to_world.shape == (3, 3)
            and physical_basis.shape == (3, 3)
            and joint_names == CORE11_NAMES
            and channel_names == CHANNEL_NAMES
            and np.isfinite(coordinates).all()
            and np.isfinite(coordinates_m).all()
            and np.isfinite(times).all()
            and np.isfinite(pelvis).all()
            and np.isfinite(leg_length)
            and leg_length > 0
            and np.all(coordinates[~valid] == 0)
            and np.all(coordinates_m[~valid] == 0)
            and np.allclose(body_to_world, world_to_body.T, atol=1e-8)
            and np.allclose(world_to_body @ world_to_body.T, np.eye(3), atol=1e-8)
            and np.isclose(np.linalg.det(physical_basis), 1.0, atol=1e-8)
        )
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _provenance(
    row: Mapping[str, str],
    sequence: AmassSequence,
    source_sha256: str,
    source_sha256_verified: bool,
    converted: Mapping[str, Any],
    config: ConversionConfig,
    model_info: Mapping[str, str],
    fingerprint: str,
) -> dict[str, Any]:
    frame: ForwardFrame = converted["frame"]
    return {
        "source": {
            "relative_path": row["relative_path"],
            "sha256": source_sha256,
            "sha256_verified_against_current_file": source_sha256_verified,
            "source_dataset": row.get("source_dataset", ""),
            "subject_id_candidate": row.get("subject_id_candidate", ""),
            "identity": row.get("identity", ""),
            "split": row.get("split", ""),
            "subject_assignment_status": (
                "assigned" if row.get("identity") and row.get("split") else "unassigned"
            ),
            "motion_id": row.get("motion_id", ""),
            "gender": sequence.gender,
            "native_fps": sequence.mocap_framerate,
            "native_frames": int(sequence.poses.shape[0]),
            "pose_layout": {
                "width": 156,
                "representation": "axis-angle",
                "root_orient": [0, 3],
                "pose_body": [3, 66],
                "pose_hand": [66, 156],
                "left_hand": [66, 111],
                "right_hand": [111, 156],
            },
            "num_betas": 16,
            "num_dmpls": 8,
            "betas_sha256": sha256_array(sequence.betas),
            "dmpls_sha256": sha256_array(sequence.dmpls),
        },
        "body_model": {**dict(model_info), "num_betas": 16, "num_dmpls": 8, "dmpls_used": True},
        "schema": {
            "name": SCHEMA,
            "joint_names": CORE11_NAMES,
            "channel_names": CHANNEL_NAMES,
            "forefoot_definition": "midpoint of same-side SMPL-H big-toe and small-toe vertices",
            "joint_regressor_indices": SMPLH_JOINT_INDEX,
            "surface_vertex_indices": SMPLH_VERTEX_INDEX,
            "validity_semantics": VALIDITY_SEMANTICS,
        },
        "coordinate_frame": {
            "name": _coordinate_frame_name(config),
            "source_up_axis": "+z",
            "positive_mediolateral": (
                "up cross pelvis-travel forward; a sensor-chart direction with no "
                "anatomical-side claim"
                if config.forward_policy == "gauge-neutral-travel"
                else "up cross forward; aligns with anatomical left when the body faces travel direction"
            ),
            "centering": "per-frame SMPL-H pelvis",
            "scale": "median over frames of mean bilateral hip-knee plus knee-ankle length",
            "leg_length_m": converted["leg_length_m"],
            "forward_method": frame.method,
            "travel_displacement_m": frame.displacement_m,
            "travel_path_length_m": frame.path_length_m,
            "travel_straightness": frame.straightness,
            "travel_linearity": frame.linearity,
            "lateral_hip_alignment": frame.lateral_hip_alignment,
            "world_to_body_transform": frame.world_to_body_transform.tolist(),
            "body_to_world_transform": frame.world_to_body_transform.T.tolist(),
            "physical_basis_world": np.stack(
                [frame.forward_world, frame.lateral_world, frame.up_world], axis=1
            ).tolist(),
            "transform_convention": (
                "physical basis columns are [forward,lateral,up] and are right-handed; "
                "stored channels are [forward,up,lateral], so the channel transform is "
                "orthogonal but has determinant -1; row arrays use "
                "body=centered_world@world_to_body_transform.T and "
                "world=body@world_to_body_transform+pelvis_world"
            ),
        },
        "resampling": {
            "method": "joint linear interpolation on a shared physical-time grid; no extrapolation",
            "canonical_fps": config.canonical_fps,
            "canonical_frames": int(len(converted["coordinates"])),
            "valid_rule": "both source interpolation brackets must be valid",
        },
        "conversion": {
            "converter_version": CONVERTER_VERSION,
            "converter_source_sha256": CONVERTER_SOURCE_SHA256,
            "fingerprint": fingerprint,
            "config": asdict(config),
        },
    }


def _base_manifest_row(row: Mapping[str, str]) -> dict[str, Any]:
    return {
        "relative_path": row.get("relative_path", ""),
        "source_dataset": row.get("source_dataset", ""),
        "subject_id_candidate": row.get("subject_id_candidate", ""),
        "identity": row.get("identity", ""),
        "split": row.get("split", ""),
        "motion_id": row.get("motion_id", ""),
        "status": "error",
        "error": "",
    }


def convert_inventory_row(
    row: Mapping[str, str],
    *,
    amass_root: Path,
    output_root: Path,
    backend: HumanBodyPriorBackend,
    config: ConversionConfig,
    overwrite: bool,
    verify_source_sha256: bool,
) -> dict[str, Any]:
    result_row = _base_manifest_row(row)
    if row.get("status", "ok") != "ok":
        raise ConversionError(f"inventory row status is {row.get('status')!r}")

    source_path, relative = _safe_relative_path(amass_root, row["relative_path"])
    if not source_path.is_file():
        raise ConversionError(f"source file does not exist: {source_path}")
    output_relative = _output_relative_path(relative)
    output_path = (output_root / Path(*output_relative.parts)).resolve()
    try:
        output_path.relative_to(output_root)
    except ValueError as exc:
        raise ConversionError(f"output path escapes output root: {output_relative}") from exc

    sequence = load_amass_sequence(source_path)
    model_info = backend.model_info(sequence.gender)
    fingerprint = _configuration_fingerprint(config, model_info)
    inventory_sha256 = row.get("sha256", "").strip()
    source_sha256 = inventory_sha256 or sha256_file(source_path)
    source_sha256_verified = not bool(inventory_sha256)
    if verify_source_sha256:
        observed_sha256 = sha256_file(source_path)
        if observed_sha256 != source_sha256:
            raise ConversionError(
                f"source SHA-256 differs from inventory: {observed_sha256} != {source_sha256}"
            )
        source_sha256_verified = True

    if output_path.exists() and not overwrite:
        if not _existing_output_matches(
            output_path,
            source_sha256,
            fingerprint,
            row,
            require_verified_source=verify_source_sha256,
        ):
            raise ConversionError(
                f"existing output has stale or unreadable provenance: {output_path}; use --overwrite"
            )
        with np.load(output_path, allow_pickle=False) as archive:
            existing_provenance = json.loads(str(archive["provenance_json"].item()))
            existing_valid = np.asarray(archive["valid"], dtype=bool)
        frame_info = existing_provenance["coordinate_frame"]
        result_row.update(
            {
                "tensor_relative_path": output_relative.as_posix(),
                "gender": sequence.gender,
                "native_fps": sequence.mocap_framerate,
                "native_frames": sequence.poses.shape[0],
                "canonical_fps": config.canonical_fps,
                "canonical_frames": existing_valid.shape[0],
                "source_sha256": source_sha256,
                "source_sha256_verified": source_sha256_verified,
                "body_model_sha256": model_info["body_model_sha256"],
                "dmpl_model_sha256": model_info["dmpl_model_sha256"],
                "schema": SCHEMA,
                "coordinate_frame": _coordinate_frame_name(config),
                "valid_fraction": float(existing_valid.mean()),
                "forward_method": frame_info["forward_method"],
                "travel_displacement_m": frame_info["travel_displacement_m"],
                "travel_path_length_m": frame_info["travel_path_length_m"],
                "travel_straightness": frame_info["travel_straightness"],
                "travel_linearity": frame_info["travel_linearity"],
                "lateral_hip_alignment": frame_info["lateral_hip_alignment"],
                "leg_length_m": frame_info["leg_length_m"],
                "conversion_fingerprint": fingerprint,
                "status": "skipped_valid_existing",
            }
        )
        return result_row

    coordinates_world, valid = backend.forward(sequence)
    converted = convert_sequence_arrays(
        coordinates_world,
        valid,
        source_fps=sequence.mocap_framerate,
        config=config,
    )
    provenance = _provenance(
        row,
        sequence,
        source_sha256,
        source_sha256_verified,
        converted,
        config,
        model_info,
        fingerprint,
    )
    frame: ForwardFrame = converted["frame"]
    arrays = {
        "coordinates": converted["coordinates"],
        "coordinates_m": converted["coordinates_m"],
        "valid": converted["valid"],
        "canonical_times_s": converted["canonical_times_s"],
        "pelvis_world_m": converted["pelvis_world_m"],
        "world_to_body_transform": frame.world_to_body_transform.astype(np.float64),
        "body_to_world_transform": frame.world_to_body_transform.T.astype(np.float64),
        "physical_basis_world": np.stack(
            [frame.forward_world, frame.lateral_world, frame.up_world], axis=1
        ).astype(np.float64),
        "leg_length_m": np.asarray(converted["leg_length_m"], dtype=np.float64),
        "joint_names": np.asarray(CORE11_NAMES),
        "channel_names": np.asarray(CHANNEL_NAMES),
        "provenance_json": np.asarray(
            json.dumps(provenance, sort_keys=True, separators=(",", ":"))
        ),
    }
    _atomic_save_npz(output_path, arrays)

    result_row.update(
        {
            "tensor_relative_path": output_relative.as_posix(),
            "gender": sequence.gender,
            "native_fps": sequence.mocap_framerate,
            "native_frames": sequence.poses.shape[0],
            "canonical_fps": config.canonical_fps,
            "canonical_frames": len(converted["coordinates"]),
            "source_sha256": source_sha256,
            "source_sha256_verified": source_sha256_verified,
            "body_model_sha256": model_info["body_model_sha256"],
            "dmpl_model_sha256": model_info["dmpl_model_sha256"],
            "schema": SCHEMA,
            "coordinate_frame": _coordinate_frame_name(config),
            "valid_fraction": float(converted["valid"].mean()),
            "forward_method": frame.method,
            "travel_displacement_m": frame.displacement_m,
            "travel_path_length_m": frame.path_length_m,
            "travel_straightness": frame.straightness,
            "travel_linearity": frame.linearity,
            "lateral_hip_alignment": frame.lateral_hip_alignment,
            "leg_length_m": converted["leg_length_m"],
            "conversion_fingerprint": fingerprint,
            "status": "converted",
        }
    )
    return result_row


def _read_inventory(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "relative_path" not in reader.fieldnames:
            raise ConversionError("inventory must contain a relative_path column")
        rows = list(reader)
    relative_paths = []
    for row in rows:
        relative = PurePosixPath(row["relative_path"])
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ConversionError(f"unsafe inventory relative_path: {row['relative_path']!r}")
        row["relative_path"] = relative.as_posix()
        relative_paths.append(row["relative_path"])
    duplicates = sorted(path for path, count in Counter(relative_paths).items() if count > 1)
    if duplicates:
        preview = duplicates[:5]
        raise ConversionError(f"inventory contains duplicate relative paths: {preview}")
    return rows


def _attach_subject_splits(
    rows: list[dict[str, str]], split_path: Path
) -> list[dict[str, str]]:
    with split_path.open(newline="", encoding="utf-8") as handle:
        split_rows = list(csv.DictReader(handle))
    required = {"subject_id_candidate", "identity", "split"}
    if not split_rows or not required <= set(split_rows[0]):
        raise ConversionError(
            f"subject split manifest must contain {sorted(required)} and at least one row"
        )

    mapping: dict[str, tuple[str, str]] = {}
    identity_splits: dict[str, set[str]] = {}
    for split_row in split_rows:
        candidate = split_row["subject_id_candidate"].strip()
        identity = split_row["identity"].strip()
        split = split_row["split"].strip()
        if not candidate or not identity or split not in {"train", "validation", "test"}:
            raise ConversionError(f"invalid subject split row: {split_row}")
        value = (identity, split)
        if candidate in mapping and mapping[candidate] != value:
            raise ConversionError(f"subject {candidate!r} has conflicting identity/split rows")
        mapping[candidate] = value
        identity_splits.setdefault(identity, set()).add(split)

    crossed = sorted(identity for identity, splits in identity_splits.items() if len(splits) > 1)
    if crossed:
        raise ConversionError(f"identities cross subject splits: {crossed[:5]}")

    augmented: list[dict[str, str]] = []
    for row in rows:
        candidate = row.get("subject_id_candidate", "").strip()
        if candidate not in mapping:
            raise ConversionError(f"inventory subject is absent from split manifest: {candidate!r}")
        identity, split = mapping[candidate]
        augmented.append({**row, "identity": identity, "split": split})
    return augmented


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--subject-splits",
        type=Path,
        default=None,
        help="Optional audited manifest with subject_id_candidate, identity, and split",
    )
    parser.add_argument(
        "--body-model-root",
        type=Path,
        required=True,
        help="Root containing smplh/{male,female}/model.npz and dmpls/{male,female}/model.npz",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--rejects", type=Path, required=True)
    parser.add_argument("--canonical-fps", type=float, default=30.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda")
    parser.add_argument("--min-travel-m", type=float, default=0.10)
    parser.add_argument("--min-travel-straightness", type=float, default=0.20)
    parser.add_argument(
        "--forward-policy",
        choices=("travel-or-hips", "require-travel", "gauge-neutral-travel"),
        default="travel-or-hips",
    )
    parser.add_argument("--overwrite", action="store_true")
    hash_group = parser.add_mutually_exclusive_group()
    hash_group.add_argument(
        "--verify-source-sha256",
        dest="verify_source_sha256",
        action="store_true",
        help="Re-hash each current source file (default; required for frozen production data)",
    )
    hash_group.add_argument(
        "--trust-inventory-sha256",
        dest="verify_source_sha256",
        action="store_false",
        help="Trust the inventory hash without reading each source twice; provenance records it unverified",
    )
    parser.set_defaults(verify_source_sha256=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-rejects",
        action="store_true",
        help="Exit successfully after recording ineligible rows (for declared strata)",
    )
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N rows for a smoke test")
    parser.add_argument("--report-every", type=int, default=25)
    args = parser.parse_args(argv)
    if not np.isclose(args.canonical_fps, 30.0, rtol=0.0, atol=1e-12):
        parser.error("core11-v1 freezes --canonical-fps at exactly 30")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.report_every <= 0:
        parser.error("--report-every must be positive")
    if not np.isfinite(args.min_travel_m) or args.min_travel_m < 0:
        parser.error("--min-travel-m must be finite and nonnegative")
    if not np.isfinite(args.min_travel_straightness) or not (
        0.0 <= args.min_travel_straightness <= 1.0
    ):
        parser.error("--min-travel-straightness must be finite and in [0, 1]")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    amass_root = args.amass_root.resolve()
    output_root = args.output_root.resolve()
    inventory_path = args.inventory.resolve()
    output_manifest_path = args.output_manifest.resolve()
    rejects_path = args.rejects.resolve()
    control_paths = {
        "inventory": inventory_path,
        "output manifest": output_manifest_path,
        "reject manifest": rejects_path,
    }
    if args.subject_splits is not None:
        control_paths["subject split manifest"] = args.subject_splits.resolve()
    if len(set(control_paths.values())) != len(control_paths):
        raise SystemExit(f"manifest paths must be distinct: {control_paths}")
    if not amass_root.is_dir():
        raise SystemExit(f"AMASS root is not a directory: {amass_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    config = ConversionConfig(
        canonical_fps=args.canonical_fps,
        batch_size=args.batch_size,
        min_travel_m=args.min_travel_m,
        min_travel_straightness=args.min_travel_straightness,
        forward_policy=args.forward_policy,
    )
    rows = _read_inventory(inventory_path)
    if args.subject_splits is not None:
        rows = _attach_subject_splits(rows, args.subject_splits.resolve())
    if args.limit is not None:
        rows = rows[: args.limit]
    backend = HumanBodyPriorBackend(args.body_model_root, args.device, args.batch_size)

    results: list[dict[str, Any]] = []
    total = len(rows)
    for index, row in enumerate(rows, start=1):
        try:
            result = convert_inventory_row(
                row,
                amass_root=amass_root,
                output_root=output_root,
                backend=backend,
                config=config,
                overwrite=args.overwrite,
                verify_source_sha256=args.verify_source_sha256,
            )
        except Exception as exc:
            result = _base_manifest_row(row)
            result["error"] = f"{type(exc).__name__}: {exc}"
            if args.fail_fast:
                results.append(result)
                _atomic_write_csv(args.output_manifest, results, MANIFEST_FIELDS)
                _atomic_write_csv(
                    args.rejects,
                    [item for item in results if item["status"] == "error"],
                    MANIFEST_FIELDS,
                )
                raise
        results.append(result)

        if index % args.report_every == 0 or index == total:
            _atomic_write_csv(args.output_manifest, results, MANIFEST_FIELDS)
            rejects = [item for item in results if item["status"] == "error"]
            _atomic_write_csv(args.rejects, rejects, MANIFEST_FIELDS)
            converted = sum(item["status"] == "converted" for item in results)
            skipped = sum(item["status"] == "skipped_valid_existing" for item in results)
            print(
                f"[{index:,}/{total:,}] converted={converted:,} skipped={skipped:,} "
                f"rejected={len(rejects):,}",
                flush=True,
            )

    rejected = sum(item["status"] == "error" for item in results)
    print(f"Wrote conversion manifest: {args.output_manifest}")
    print(f"Wrote reject manifest: {args.rejects}")
    return 1 if rejected and not args.allow_rejects else 0


if __name__ == "__main__":
    sys.exit(main())
