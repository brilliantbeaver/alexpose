"""Matched GAVD feasibility training for the GaitParity encoder study.

This module is deliberately local in scope.  GAVD has no participant IDs and
the coordinate-derived diagnostics are not clinical targets.  The code below
therefore answers only whether the complete three-model JEPA loop runs,
avoids obvious collapse, and preserves (or deliberately violates) the declared
reflection geometry.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn


CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
EXPECTED_COUNTS = {"normal": 12, "parkinsons": 9, "stroke": 12, "myopathic": 47, "cerebralpalsy": 16}
LEFT_RIGHT_PAIRS = [
    (1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
    (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
    (27, 28), (29, 30), (31, 32),
]
MASK_JOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
VARIANTS = ["standard", "paired_unconstrained", "reflection_equivariant"]


@dataclass(frozen=True)
class TrainConfig:
    profile: str
    frames: int
    stride: int
    segment_length: int
    embed_dim: int
    encoder_depth: int
    predictor_depth: int
    heads: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    mask_fraction: float
    ema_start: float
    vicreg_weight: float
    odd_vicreg_weight: float
    max_yaw_degrees: float
    amp: bool


PROFILES = {
    # The CPU profile is a real, full-cohort feasibility run, not a paper-scale model.
    "cpu": TrainConfig("cpu", 64, 64, 8, 32, 2, 1, 4, 8, 6, 3e-4, 0.05, 0.50, 0.996, 0.05, 1.0, 8.0, False),
    # The CUDA profile keeps the original GAVD encoder scale and samples windows more densely.
    "gpu": TrainConfig("gpu", 64, 32, 4, 96, 4, 2, 8, 32, 100, 2e-4, 0.05, 0.60, 0.996, 0.05, 1.0, 8.0, True),
    # Smoke exists only for CI and notebook plumbing.
    "smoke": TrainConfig("smoke", 32, 32, 8, 16, 1, 1, 4, 4, 1, 8e-4, 0.01, 0.40, 0.99, 0.10, 1.0, 4.0, False),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_json(payload) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_pose_dir(project_dir: Path, explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.getenv("GAIT_PARITY_POSE_DIR"):
        candidates.append(Path(os.environ["GAIT_PARITY_POSE_DIR"]).expanduser())
    candidates.extend([
        Path(project_dir) / "work" / "artifacts" / "real" / "poses",
        Path(project_dir).parent / "gavd5-tm" / "work" / "artifacts" / "real" / "poses",
    ])
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*/*.npz")):
            return candidate.resolve()
    raise FileNotFoundError("No full GAVD pose cache found. Set GAIT_PARITY_POSE_DIR explicitly.")


def synthetic_records(n: int = 12, frames: int = 48, seed: int = 9) -> list[dict]:
    records = []
    phase = np.linspace(0, 4 * np.pi, frames, endpoint=False)
    for index in range(n):
        rng = np.random.default_rng(seed + index)
        sequence = np.zeros((frames, 33, 4), dtype=np.float32)
        sequence[..., 3] = 1.0
        for joint in range(33):
            sequence[:, joint, 0] = 0.05 * np.sin(phase + joint * 0.13)
            sequence[:, joint, 1] = 0.02 * np.cos(phase + joint * 0.07)
            sequence[:, joint, 2] = 0.01 * np.sin(2 * phase + joint * 0.05)
        sign = 1.0 if index % 2 == 0 else -1.0
        for left, right in LEFT_RIGHT_PAIRS[-5:]:
            sequence[:, left, 0] += (0.07 + 0.025 * sign) * np.sin(phase)
            sequence[:, right, 0] += (0.07 - 0.025 * sign) * np.sin(phase)
        sequence[..., :3] += rng.normal(0, 0.002, sequence[..., :3].shape)
        records.append({
            "condition": "illustrative",
            "sequence_id": f"smoke_{index:03d}",
            "video_id": f"smoke_video_{index // 2:03d}",
            "sequence": sequence,
            "path": None,
        })
    return records


def load_gavd_records(pose_dir: Path) -> list[dict]:
    records = []
    for condition in CONDITIONS:
        paths = sorted((Path(pose_dir) / condition).glob("*.npz"))
        if len(paths) != EXPECTED_COUNTS[condition]:
            raise ValueError(f"{condition}: expected {EXPECTED_COUNTS[condition]} cached sequences, found {len(paths)}")
        for path in paths:
            data = np.load(path, allow_pickle=False)
            required = {"sequence", "sequence_id", "video_id", "condition"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing {sorted(missing)}")
            sequence = data["sequence"].astype(np.float32)
            if sequence.ndim != 3 or sequence.shape[1:] != (33, 4):
                raise ValueError(f"Bad pose shape in {path}: {sequence.shape}")
            stored_condition = str(data["condition"].item())
            if stored_condition != condition:
                raise ValueError(f"Folder/record condition mismatch in {path}")
            records.append({
                "condition": condition,
                "sequence_id": str(data["sequence_id"].item()),
                "video_id": str(data["video_id"].item()),
                "sequence": sequence,
                "path": str(path),
            })
    if len(records) != sum(EXPECTED_COUNTS.values()):
        raise AssertionError("The locked full-GAVD cohort must contain 96 sequences")
    return records


def interpolate_low_visibility(sequence: np.ndarray, threshold: float = 0.45, max_gap: int = 4):
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    visibility = np.nan_to_num(sequence[..., 3], nan=0.0)
    finite = np.isfinite(sequence[..., :3]).all(axis=-1)
    valid = (visibility >= threshold) & finite
    filled = valid.copy()
    for joint in range(33):
        observed = np.flatnonzero(valid[:, joint])
        for left, right in zip(observed[:-1], observed[1:]):
            gap = int(right - left - 1)
            if not 0 < gap <= max_gap:
                continue
            fraction = (np.arange(1, gap + 1, dtype=np.float32) / (gap + 1))[:, None]
            sequence[left + 1:right, joint, :3] = (
                sequence[left, joint, :3][None] * (1 - fraction)
                + sequence[right, joint, :3][None] * fraction
            )
            filled[left + 1:right, joint] = True
        sequence[~filled[:, joint], joint, :3] = np.nan
    return sequence, valid


def center_and_scale(sequence: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    sequence = np.asarray(sequence, dtype=np.float32).copy()
    xyz = sequence[..., :3]
    left_hip, right_hip = xyz[:, 23], xyz[:, 24]
    left_ok = np.isfinite(left_hip).all(1)
    right_ok = np.isfinite(right_hip).all(1)
    pelvis = np.full((len(xyz), 3), np.nan, dtype=np.float32)
    pelvis[left_ok & right_ok] = 0.5 * (left_hip[left_ok & right_ok] + right_hip[left_ok & right_ok])
    pelvis[left_ok & ~right_ok] = left_hip[left_ok & ~right_ok]
    pelvis[right_ok & ~left_ok] = right_hip[right_ok & ~left_ok]
    good = np.isfinite(pelvis).all(1)
    fallback = np.median(pelvis[good], axis=0) if good.any() else np.zeros(3, dtype=np.float32)
    pelvis[~good] = fallback
    xyz = xyz - pelvis[:, None]
    shoulder = np.linalg.norm(xyz[:, 11, :2] - xyz[:, 12, :2], axis=-1)
    hip = np.linalg.norm(xyz[:, 23, :2] - xyz[:, 24, :2], axis=-1)
    scale = np.nanmedian(np.maximum(shoulder, hip))
    if not np.isfinite(scale) or scale < eps:
        scale = 1.0
    return np.nan_to_num(xyz / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def temporal_resize(array: np.ndarray, frames: int) -> np.ndarray:
    array = np.asarray(array)
    if len(array) == frames:
        return array.copy()
    if len(array) < 2:
        return np.repeat(array, frames, axis=0)
    old = np.linspace(0, 1, len(array))
    new = np.linspace(0, 1, frames)
    flat = array.reshape(len(array), -1)
    resized = np.stack([np.interp(new, old, flat[:, i]) for i in range(flat.shape[1])], axis=1)
    return resized.reshape(frames, *array.shape[1:]).astype(array.dtype)


def build_windows(records: list[dict], config: TrainConfig) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    windows, valid_windows, rows = [], [], []
    for record in records:
        cleaned, valid = interpolate_low_visibility(record["sequence"])
        xyz = center_and_scale(cleaned)
        if len(xyz) < config.frames:
            starts = [0]
            xyz_chunks = [temporal_resize(xyz, config.frames)]
            valid_chunks = [temporal_resize(valid.astype(np.float32), config.frames) >= 0.5]
        else:
            starts = list(range(0, len(xyz) - config.frames + 1, config.stride))
            if starts[-1] != len(xyz) - config.frames:
                starts.append(len(xyz) - config.frames)
            xyz_chunks = [xyz[start:start + config.frames] for start in starts]
            valid_chunks = [valid[start:start + config.frames] for start in starts]
        for start, chunk, chunk_valid in zip(starts, xyz_chunks, valid_chunks):
            patch_valid = chunk_valid.reshape(config.frames // config.segment_length, config.segment_length, 33).mean(1) >= 0.5
            if patch_valid[:, MASK_JOINTS].sum() < 2:
                continue
            windows.append(chunk)
            valid_windows.append(patch_valid)
            rows.append({
                "window_id": f"{record['sequence_id']}:{start}",
                "sequence_id": record["sequence_id"],
                "video_id": record["video_id"],
                "condition": record["condition"],
                "start_frame": int(start),
            })
    if not windows:
        raise ValueError("No eligible windows were constructed")
    return torch.tensor(np.stack(windows)), torch.tensor(np.stack(valid_windows)), pd.DataFrame(rows)


def cohort_manifest(records: list[dict], windows: pd.DataFrame, config: TrainConfig, mode: str) -> dict:
    members = [{
        "sequence_id": r["sequence_id"], "video_id": r["video_id"], "condition": r["condition"],
        "frames": int(len(r["sequence"])), "path": r.get("path"),
    } for r in records]
    identity = [{k: row[k] for k in ("sequence_id", "video_id", "condition", "frames")} for row in members]
    return {
        "scope": "local GAVD feasibility; transductive representation/geometry evidence only",
        "mode": mode,
        "profile": config.profile,
        "records": members,
        "record_count": len(members),
        "window_count": int(len(windows)),
        "source_video_count": int(windows.video_id.nunique()),
        "condition_counts": windows.groupby("condition").sequence_id.nunique().to_dict(),
        "cohort_sha256": sha256_json(identity),
        "window_sha256": sha256_json(windows.to_dict(orient="records")),
        "preprocessing": "short-gap interpolation; pelvis centering; shoulder/hip scale; sliding windows",
    }


def anatomical_mirror(x: torch.Tensor) -> torch.Tensor:
    mirrored = x.clone()
    mirrored[..., 0] *= -1
    original = mirrored.clone()
    for left, right in LEFT_RIGHT_PAIRS:
        mirrored[..., left, :] = original[..., right, :]
        mirrored[..., right, :] = original[..., left, :]
    return mirrored


def lift_orbit(x: torch.Tensor) -> torch.Tensor:
    return torch.stack([x, anatomical_mirror(x)], dim=1)


def augment_canonical(x: torch.Tensor, max_degrees: float) -> torch.Tensor:
    view = x.clone()
    present = view.abs().sum(dim=-1) > 1e-8
    angles = (torch.rand(len(x), device=x.device) * 2 - 1) * math.radians(max_degrees)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    old_x, old_z = view[..., 0].clone(), view[..., 2].clone()
    view[..., 0] = cosine[:, None, None] * old_x + sine[:, None, None] * old_z
    view[..., 2] = -sine[:, None, None] * old_x + cosine[:, None, None] * old_z
    return view.masked_fill(~present[..., None], 0.0)


def sample_mask(valid_patch: torch.Tensor, fraction: float, generator: torch.Generator) -> torch.Tensor:
    valid = valid_patch.bool().cpu()
    eligible_joint = torch.zeros(33, dtype=torch.bool)
    eligible_joint[MASK_JOINTS] = True
    eligible = valid & eligible_joint[None, None]
    counts = eligible.flatten(1).sum(1)
    n_mask = max(1, int(torch.floor(counts.min().float() * fraction).item()))
    n_mask = min(n_mask, int(counts.min().item()) - 1)
    result = torch.zeros_like(eligible)
    for index in range(len(result)):
        candidates = torch.where(eligible[index].flatten())[0]
        chosen = candidates[torch.randperm(len(candidates), generator=generator)[:n_mask]]
        result[index].view(-1)[chosen] = True
    return result


class SkeletonTokenizer(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        if config.frames % config.segment_length:
            raise ValueError("frames must be divisible by segment_length")
        self.frames = config.frames
        self.segment_length = config.segment_length
        self.segments = config.frames // config.segment_length
        self.joints = 33
        self.embed_dim = config.embed_dim
        self.patch = nn.Linear(config.segment_length * 3, config.embed_dim)
        self.time_pos = nn.Parameter(torch.randn(self.segments, config.embed_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(33, config.embed_dim) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = len(x)
        patches = x.reshape(batch, self.segments, self.segment_length, 33, 3)
        patches = patches.permute(0, 1, 3, 2, 4).flatten(3)
        return self.patch(patches) + self.time_pos[None, :, None] + self.joint_pos[None, None]


def transformer_layer(config: TrainConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        config.embed_dim, config.heads, config.embed_dim * 4, dropout=0.0,
        activation="gelu", batch_first=True, norm_first=True,
    )


class SymmetricPairedLayer(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.self_block = transformer_layer(config)
        self.cross_norm = nn.LayerNorm(config.embed_dim)
        self.cross = nn.MultiheadAttention(config.embed_dim, config.heads, dropout=0.0, batch_first=True)

    def forward(self, a, b):
        a0, b0 = self.self_block(a), self.self_block(b)
        qa, qb = self.cross_norm(a0), self.cross_norm(b0)
        a_cross = self.cross(qa, qb, qb, need_weights=False)[0]
        b_cross = self.cross(qb, qa, qa, need_weights=False)[0]
        return a0 + a_cross, b0 + b_cross


class UnconstrainedPairedLayer(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        self.self_a, self.self_b = transformer_layer(config), transformer_layer(config)
        self.norm_a, self.norm_b = nn.LayerNorm(config.embed_dim), nn.LayerNorm(config.embed_dim)
        self.cross_ab = nn.MultiheadAttention(config.embed_dim, config.heads, dropout=0.0, batch_first=True)
        self.cross_ba = nn.MultiheadAttention(config.embed_dim, config.heads, dropout=0.0, batch_first=True)

    def forward(self, a, b):
        a0, b0 = self.self_a(a), self.self_b(b)
        qa, qb = self.norm_a(a0), self.norm_b(b0)
        return a0 + self.cross_ab(qa, qb, qb, need_weights=False)[0], b0 + self.cross_ba(qb, qa, qa, need_weights=False)[0]


class OrbitEncoder(nn.Module):
    def __init__(self, config: TrainConfig, variant: str):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.config, self.variant = config, variant
        self.tokenizer = SkeletonTokenizer(config)
        if variant == "standard":
            self.layers = nn.ModuleList([transformer_layer(config) for _ in range(config.encoder_depth)])
        elif variant == "reflection_equivariant":
            self.layers = nn.ModuleList([SymmetricPairedLayer(config) for _ in range(config.encoder_depth)])
        else:
            self.layers = nn.ModuleList([UnconstrainedPairedLayer(config) for _ in range(config.encoder_depth)])
        self.norm = nn.LayerNorm(config.embed_dim)

    @property
    def segments(self):
        return self.tokenizer.segments

    def forward(self, orbit: torch.Tensor, keep_mask: torch.Tensor | None = None, return_states: bool = False):
        if orbit.ndim != 5 or orbit.shape[1] != 2:
            raise ValueError(f"Expected [B,2,T,J,C], got {tuple(orbit.shape)}")
        a, b = self.tokenizer(orbit[:, 0]), self.tokenizer(orbit[:, 1])
        a, b = a.flatten(1, 2), b.flatten(1, 2)
        if keep_mask is not None:
            keep = keep_mask.flatten(1)
            counts = keep.sum(1)
            if not torch.equal(counts, counts[:1].expand_as(counts)):
                raise ValueError("Every sample must keep the same token count")
            a = a[keep].reshape(len(a), int(counts[0]), -1)
            b = b[keep].reshape(len(b), int(counts[0]), -1)
        states = [(a, b)]
        for layer in self.layers:
            if self.variant == "standard":
                a, b = layer(a), layer(b)
            else:
                a, b = layer(a, b)
            states.append((a, b))
        a, b = self.norm(a), self.norm(b)
        states.append((a, b))
        return ((a, b), states) if return_states else (a, b)


class TokenPredictor(nn.Module):
    def __init__(self, config: TrainConfig):
        super().__init__()
        segments = config.frames // config.segment_length
        self.segments, self.joints, self.dim = segments, 33, config.embed_dim
        self.mask_token = nn.Parameter(torch.randn(1, 1, config.embed_dim) * 0.02)
        self.time_pos = nn.Parameter(torch.randn(segments, config.embed_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(33, config.embed_dim) * 0.02)
        self.layers = nn.ModuleList([transformer_layer(config) for _ in range(config.predictor_depth)])
        self.norm = nn.LayerNorm(config.embed_dim)

    def one_branch(self, visible: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        mask = target_mask.flatten(1)
        full = self.mask_token.expand(len(visible), self.segments * self.joints, -1).clone()
        full[~mask] = visible.flatten(0, 1)
        positions = (self.time_pos[:, None] + self.joint_pos[None]).reshape(1, self.segments * self.joints, -1)
        full = full + positions
        for layer in self.layers:
            full = layer(full)
        full = self.norm(full)
        return full[mask].reshape(len(full), -1, self.dim)

    def forward(self, state, target_mask):
        return torch.stack([self.one_branch(state[0], target_mask), self.one_branch(state[1], target_mask)], dim=1)


class OrbitJEPA(nn.Module):
    def __init__(self, config: TrainConfig, variant: str):
        super().__init__()
        self.config, self.variant = config, variant
        self.encoder = OrbitEncoder(config, variant)
        self.target_encoder = copy.deepcopy(self.encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = TokenPredictor(config)
        self.register_buffer("target_center", torch.zeros(config.embed_dim))

    def forward(self, orbit, target_mask):
        student = self.encoder(orbit, keep_mask=~target_mask)
        predicted = self.predictor(student, target_mask)
        with torch.no_grad():
            targets = self.target_encoder(orbit)
            mask = target_mask.flatten(1)
            selected = [branch[mask].reshape(len(orbit), -1, branch.shape[-1]) for branch in targets]
            selected = torch.stack(selected, dim=1)
        return predicted, selected

    @torch.no_grad()
    def update_target(self, momentum: float):
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)

    @torch.no_grad()
    def update_center(self, targets, beta: float = 0.9):
        self.target_center.mul_(beta).add_(targets.mean((0, 1, 2)), alpha=1 - beta)


class VICRegProjector(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):
        return self.net(x)


def sjepa_cross_entropy(predicted, targets, center, predictor_temperature=0.10, target_temperature=0.06):
    target_prob = torch.softmax((targets - center[None, None, None]) / target_temperature, dim=-1).detach()
    prediction_log_prob = torch.log_softmax(predicted / predictor_temperature, dim=-1)
    return -(target_prob * prediction_log_prob).sum(-1).mean()


def off_diagonal(matrix):
    n = matrix.shape[0]
    return matrix.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_terms(first, second, gamma=1.0, eps=1e-4):
    invariance = F.mse_loss(first, second)
    variance = 0.5 * (
        F.relu(gamma - torch.sqrt(first.var(0, unbiased=False) + eps)).mean()
        + F.relu(gamma - torch.sqrt(second.var(0, unbiased=False) + eps)).mean()
    )
    denominator = max(len(first) - 1, 1)
    first_centered, second_centered = first - first.mean(0), second - second.mean(0)
    first_cov = first_centered.T @ first_centered / denominator
    second_cov = second_centered.T @ second_centered / denominator
    covariance = (off_diagonal(first_cov).square().sum() + off_diagonal(second_cov).square().sum()) / (2 * first.shape[1])
    return 25 * invariance + 25 * variance + covariance, invariance, variance, covariance


def parity_channels(state):
    a, b = state[0].mean(1), state[1].mean(1)
    return (a + b) / 2, (a - b) / 2


def orbit_vicreg(model: OrbitJEPA, projector: VICRegProjector, first_orbit, second_orbit):
    first_even, first_odd = parity_channels(model.encoder(first_orbit))
    second_even, second_odd = parity_channels(model.encoder(second_orbit))
    even = vicreg_terms(projector(first_even), projector(second_even))
    odd = vicreg_terms(projector(first_odd), projector(second_odd))
    return even, odd


def cosine_ema(step, total_steps, start=0.996, end=1.0):
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1) / 2


def parameter_count(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only))


def compute_proxy_per_step(model: OrbitJEPA, config: TrainConfig) -> int:
    """A frozen analytic allocation proxy, explicitly not an exact FLOP count."""
    tokens = (config.frames // config.segment_length) * 33
    branch_passes = 8  # student+teacher JEPA and two complete-view VICReg forwards
    return int(branch_passes * config.batch_size * tokens * parameter_count(model.encoder))


def planned_updates(models: dict[str, OrbitJEPA], config: TrainConfig, n_windows: int, regime: str) -> dict[str, int]:
    base = config.epochs * math.ceil(n_windows / config.batch_size)
    if regime == "exposure":
        return {variant: base for variant in models}
    if regime != "compute":
        raise ValueError("matching regime must be exposure or compute")
    reference = compute_proxy_per_step(models["reflection_equivariant"], config) * base
    return {variant: max(1, round(reference / compute_proxy_per_step(model, config))) for variant, model in models.items()}


def device_from_environment() -> torch.device:
    requested = os.getenv("GAIT_PARITY_DEVICE", "cpu").strip().lower()
    if requested not in {"cpu", "cuda"}:
        raise ValueError("GAIT_PARITY_DEVICE must be cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return torch.device(requested)


def cyclic_batches(n_items: int, batch_size: int, updates: int, seed: int) -> Iterable[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_items)
    cursor = 0
    for _ in range(updates):
        if cursor + batch_size > n_items:
            tail = order[cursor:]
            order = rng.permutation(n_items)
            need = batch_size - len(tail)
            batch = np.concatenate([tail, order[:need]])
            cursor = need
        else:
            batch = order[cursor:cursor + batch_size]
            cursor += batch_size
        yield batch


def train_variant(
    model: OrbitJEPA,
    windows: torch.Tensor,
    valid_patch: torch.Tensor,
    config: TrainConfig,
    device: torch.device,
    updates: int,
    seed: int,
):
    seed_everything(seed)
    model = model.to(device)
    projector = VICRegProjector(config.embed_dim).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad] + list(projector.parameters())
    optimizer = torch.optim.AdamW(trainable, lr=config.learning_rate, betas=(0.9, 0.95), weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(updates, 1))
    amp_enabled = bool(config.amp and device.type == "cuda")
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - compatibility with the oldest supported torch builds
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    mask_generator = torch.Generator(device="cpu").manual_seed(seed + 100_000)
    history, start = [], time.perf_counter()
    model.train()
    for step, indices in enumerate(cyclic_batches(len(windows), config.batch_size, updates, seed + 17)):
        canonical = windows[indices].to(device)
        valid = valid_patch[indices]
        target_mask = sample_mask(valid, config.mask_fraction, mask_generator).to(device)
        first_orbit = lift_orbit(augment_canonical(canonical, config.max_yaw_degrees))
        second_orbit = lift_orbit(augment_canonical(canonical, config.max_yaw_degrees))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            predicted, targets = model(first_orbit, target_mask)
            jepa = sjepa_cross_entropy(predicted, targets, model.target_center)
            even_terms, odd_terms = orbit_vicreg(model, projector, first_orbit, second_orbit)
            vicreg = even_terms[0] + config.odd_vicreg_weight * odd_terms[0]
            total = jepa + config.vicreg_weight * vicreg
        if not torch.isfinite(total):
            raise FloatingPointError(f"Non-finite loss at step {step}")
        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        momentum = cosine_ema(step, updates, config.ema_start)
        model.update_target(momentum)
        model.update_center(targets.detach())
        history.append({
            "step": step + 1, "total_loss": float(total.detach().cpu()), "jepa_loss": float(jepa.detach().cpu()),
            "vicreg_loss": float(vicreg.detach().cpu()), "even_invariance": float(even_terms[1].detach().cpu()),
            "even_variance": float(even_terms[2].detach().cpu()), "even_covariance": float(even_terms[3].detach().cpu()),
            "odd_invariance": float(odd_terms[1].detach().cpu()), "odd_variance": float(odd_terms[2].detach().cpu()),
            "odd_covariance": float(odd_terms[3].detach().cpu()), "ema_momentum": momentum,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
    wall = time.perf_counter() - start
    return model.cpu(), projector.cpu(), pd.DataFrame(history), wall


@torch.no_grad()
def collect_parity_features(encoder: OrbitEncoder, windows: torch.Tensor, batch_size: int = 32):
    encoder.eval()
    evens, odds = [], []
    for start in range(0, len(windows), batch_size):
        orbit = lift_orbit(windows[start:start + batch_size])
        even, odd = parity_channels(encoder(orbit))
        evens.append(even.cpu())
        odds.append(odd.cpu())
    return torch.cat(evens), torch.cat(odds)


def representation_metrics(features: torch.Tensor) -> dict:
    features = features.float()
    centered = features - features.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(len(features) - 1, 1)
    eig = torch.linalg.eigvalsh(covariance).clamp_min(0)
    probability = eig / eig.sum().clamp_min(1e-12)
    nonzero = probability > 0
    rank = torch.exp(-(probability[nonzero] * torch.log(probability[nonzero])).sum())
    normalized = F.normalize(features, dim=1)
    cosine = normalized @ normalized.T
    offdiag = cosine[~torch.eye(len(features), dtype=torch.bool)].mean() if len(features) > 1 else torch.tensor(float("nan"))
    return {
        "energy": float(features.square().mean()),
        "feature_variance": float(features.var(0, unbiased=False).mean()),
        "effective_rank": float(rank),
        "mean_pairwise_cosine": float(offdiag),
    }


@torch.no_grad()
def commutation_report(
    encoder: OrbitEncoder,
    x: torch.Tensor,
    keep_mask: torch.Tensor | None = None,
    train_mode: bool = False,
    device: torch.device | None = None,
    mixed_precision: bool = False,
) -> pd.DataFrame:
    device = device or next(encoder.parameters()).device
    prior_mode = encoder.training
    encoder.train(train_mode)
    x = x.to(device)
    keep_mask = keep_mask.to(device) if keep_mask is not None else None
    amp_enabled = bool(mixed_precision and device.type == "cuda")
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        (_, states) = encoder(lift_orbit(x), keep_mask=keep_mask, return_states=True)
        (_, mirrored_states) = encoder(
            lift_orbit(anatomical_mirror(x)), keep_mask=keep_mask, return_states=True
        )
    rows = []
    for layer, ((a, b), (ma, mb)) in enumerate(zip(states, mirrored_states)):
        residual = max(float((ma - b).abs().max()), float((mb - a).abs().max()))
        scale = max(float(a.abs().max()), float(b.abs().max()), 1e-12)
        rows.append({"layer": layer, "max_abs": residual, "max_rel": residual / scale})
    encoder.train(prior_mode)
    return pd.DataFrame(rows)


def model_config_payload(config: TrainConfig, variant: str) -> dict:
    return {"variant": variant, "train_config": asdict(config)}


def build_model(config: TrainConfig, variant: str, seed: int) -> OrbitJEPA:
    seed_everything(seed)
    return OrbitJEPA(config, variant)


def save_checkpoint(path: Path, model: OrbitJEPA, projector: VICRegProjector, metadata: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "projector_state": projector.state_dict(),
        "metadata": metadata,
    }, path)


def load_checkpoint(path: Path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    config = TrainConfig(**payload["metadata"]["train_config"])
    model = build_model(config, payload["metadata"]["variant"], payload["metadata"]["seed"])
    model.load_state_dict(payload["model_state"])
    projector = VICRegProjector(config.embed_dim)
    projector.load_state_dict(payload["projector_state"])
    return model, projector, payload["metadata"]
