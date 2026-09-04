"""Validation-only swap-probe mechanism screen for AMASS Core11.

The module deliberately does not train a new JEPA.  It applies a controlled,
window-local bilateral naming corruption to non-overlapping AMASS windows,
learns lightweight relative-switch heads on the training identities, and
evaluates correction through one frozen standard S-JEPA encoder.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from torch import nn
from torch.utils.data import DataLoader, Dataset

from gavd6_sjepa.shared_infrastructure.artifact_io_operations import (
    atomic_save_joblib as atomic_joblib,
    atomic_write_json as atomic_json,
    sha256_file,
)
from .amass_core11_training_pipeline import (
    MIRROR_CHANNEL,
    MIRROR_PAIRS,
    TIME_PATCH_FRAMES,
    WINDOW_FRAMES,
    atomic_dataframe_to_csv,
)


SWAP_PROBE_VERSION = "swap-probe-v1"
SWAP_INDEX = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9)
DISTAL_JOINTS = (5, 6, 7, 8, 9, 10)
ARMS = (
    "corrupted_uncorrected",
    "continuity_map",
    "learned_map",
    "learned_structured_posterior",
    "oracle",
)
REFERENCE_ARMS = ("clean_reference",)


@dataclass(frozen=True)
class CorruptionConfig:
    """Frozen corruption settings for the bilateral-correction probe."""

    block_frames: int = TIME_PATCH_FRAMES
    swapped_blocks: int = 4
    noise_std: float = 0.01
    occlusion_probability: float = 0.10
    sensor_reflection_probability: float = 0.50
    event_probability: float = 0.80

    def validate(self) -> None:
        if self.block_frames != TIME_PATCH_FRAMES:
            raise ValueError(
                "The frozen Core11 probe requires exactly four frames per block"
            )
        blocks = WINDOW_FRAMES // self.block_frames
        if not 1 <= self.swapped_blocks <= blocks - 2:
            raise ValueError("swapped_blocks must leave an uncorrupted root and tail")
        for name, value in (
            ("occlusion_probability", self.occlusion_probability),
            ("sensor_reflection_probability", self.sensor_reflection_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        if not 0.0 < self.event_probability < 1.0:
            raise ValueError("event_probability must lie strictly between zero and one")
        if self.noise_std < 0.0:
            raise ValueError("noise_std must be non-negative")


@dataclass(frozen=True)
class WindowDraw:
    """All stochastic actions applied to one non-overlapping source window."""

    window_id: str
    semantic_path: tuple[int, ...]
    sensor_reflection_bit: int
    occlusion_keep: tuple[tuple[bool, ...], ...]
    noise_seed: int

    @property
    def switch_edges(self) -> tuple[int, ...]:
        return tuple(
            index
            for index in range(1, len(self.semantic_path))
            if self.semantic_path[index] != self.semantic_path[index - 1]
        )

    @property
    def event_present(self) -> bool:
        return any(self.semantic_path)


class GaugeWindowDataset(Dataset):
    """Attach deterministic corruption metadata to an existing window dataset."""

    def __init__(
        self,
        base: Dataset,
        rows: pd.DataFrame,
        draws: Mapping[str, WindowDraw],
    ) -> None:
        if len(base) != len(rows):
            raise ValueError("Base dataset and metadata rows must have equal length")
        required = {"window_id", "identity", "tensor_relative_path", "start_frame"}
        missing = required.difference(rows.columns)
        if missing:
            raise ValueError(f"Window metadata is missing columns: {sorted(missing)}")
        self.base = base
        self.rows = rows.reset_index(drop=True).to_dict(orient="records")
        self.draws = dict(draws)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, item: int) -> dict:
        sample = dict(self.base[item])
        row = self.rows[item]
        if sample["window_id"] != row["window_id"]:
            raise ValueError("Window dataset and metadata order disagree")
        draw = self.draws[row["window_id"]]
        sample.update(
            {
                "identity": row["identity"],
                "tensor_relative_path": row["tensor_relative_path"],
                "start_frame": int(row["start_frame"]),
                "semantic_path": torch.tensor(draw.semantic_path, dtype=torch.bool),
                "sensor_reflection_bit": bool(draw.sensor_reflection_bit),
                "occlusion_keep": torch.tensor(draw.occlusion_keep, dtype=torch.bool),
                "noise_seed": int(draw.noise_seed),
            }
        )
        return sample


def _stable_seed(seed: int, namespace: str, key: str) -> int:
    payload = f"{SWAP_PROBE_VERSION}|{seed}|{namespace}|{key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def make_window_draw(
    window_id: str,
    seed: int,
    config: CorruptionConfig,
) -> WindowDraw:
    """Generate domain-separated semantic and nuisance actions."""

    config.validate()
    blocks = WINDOW_FRAMES // config.block_frames
    semantic_rng = np.random.default_rng(_stable_seed(seed, "semantic", window_id))
    first_start = 1
    last_start = blocks - config.swapped_blocks - 1
    path = np.zeros(blocks, dtype=np.int8)
    if semantic_rng.random() < config.event_probability:
        event_start = int(semantic_rng.integers(first_start, last_start + 1))
        path[event_start : event_start + config.swapped_blocks] = 1

    sensor_rng = np.random.default_rng(_stable_seed(seed, "sensor", window_id))
    sensor_bit = int(sensor_rng.random() < config.sensor_reflection_probability)

    occlusion_rng = np.random.default_rng(_stable_seed(seed, "occlusion", window_id))
    keep = np.ones((blocks, len(SWAP_INDEX)), dtype=bool)
    draws = occlusion_rng.random((blocks, len(DISTAL_JOINTS)))
    keep[:, DISTAL_JOINTS] = draws >= config.occlusion_probability

    return WindowDraw(
        window_id=window_id,
        semantic_path=tuple(int(value) for value in path),
        sensor_reflection_bit=sensor_bit,
        occlusion_keep=tuple(tuple(bool(value) for value in row) for row in keep),
        noise_seed=_stable_seed(seed, "noise", window_id),
    )


def build_draws(
    rows: pd.DataFrame,
    seed: int,
    config: CorruptionConfig,
) -> dict[str, WindowDraw]:
    if not rows["window_id"].is_unique:
        raise ValueError("window_id must be unique")
    return {
        window_id: make_window_draw(str(window_id), seed, config)
        for window_id in rows["window_id"]
    }


def draws_to_frame(
    rows: pd.DataFrame,
    draws: Mapping[str, WindowDraw],
    *,
    split: str,
    config: CorruptionConfig,
) -> pd.DataFrame:
    records = []
    for row in rows.to_dict(orient="records"):
        draw = draws[row["window_id"]]
        dropped = [
            f"{block}:{joint}"
            for block, keep in enumerate(draw.occlusion_keep)
            for joint, present in enumerate(keep)
            if not present
        ]
        records.append(
            {
                "generator_version": SWAP_PROBE_VERSION,
                "split": split,
                "window_id": row["window_id"],
                "identity": row["identity"],
                "tensor_relative_path": row["tensor_relative_path"],
                "start_frame": int(row["start_frame"]),
                "block_frames": config.block_frames,
                "semantic_path": "".join(map(str, draw.semantic_path)),
                "switch_edges": ",".join(map(str, draw.switch_edges)),
                "event_present": draw.event_present,
                "root_anchor_bit": draw.semantic_path[0],
                "sensor_reflection_bit": draw.sensor_reflection_bit,
                "occluded_block_joints": ",".join(dropped),
                "noise_seed": draw.noise_seed,
                "noise_std": config.noise_std,
            }
        )
    return pd.DataFrame(records)


def semantic_permute_by_block(
    values: torch.Tensor,
    path: torch.Tensor,
    *,
    block_frames: int = TIME_PATCH_FRAMES,
) -> torch.Tensor:
    """Apply the semantic token action P to selected temporal blocks."""

    if values.ndim not in {3, 4}:
        raise ValueError("Expected [B,T,J] or [B,T,J,C]")
    if values.shape[2] != len(SWAP_INDEX):
        raise ValueError("Expected Core11 joint order")
    if path.ndim != 2 or path.shape[0] != len(values):
        raise ValueError("path must have shape [B,K]")
    if values.shape[1] != path.shape[1] * block_frames:
        raise ValueError("Path length does not match temporal dimension")
    swap = torch.tensor(SWAP_INDEX, device=values.device)
    exchanged = values.index_select(2, swap)
    frame_state = path.bool().repeat_interleave(block_frames, dim=1)
    while frame_state.ndim < values.ndim:
        frame_state = frame_state.unsqueeze(-1)
    return torch.where(frame_state, exchanged, values)


def semantic_permute_patch_validity(
    valid_patch: torch.Tensor,
    path: torch.Tensor,
    *,
    block_frames: int = TIME_PATCH_FRAMES,
) -> torch.Tensor:
    """Apply P to patch validity using the same block state as coordinates."""

    if valid_patch.ndim != 3 or valid_patch.shape[2] != len(SWAP_INDEX):
        raise ValueError("Expected patch validity [B,K,11]")
    if valid_patch.shape[:2] != path.shape:
        raise ValueError("validity and path must agree on batch and blocks")
    frames = valid_patch.repeat_interleave(block_frames, dim=1)
    exchanged = semantic_permute_by_block(frames, path, block_frames=block_frames)
    return exchanged.reshape(
        len(valid_patch), path.shape[1], block_frames, len(SWAP_INDEX)
    ).all(dim=2)


def sensor_reflect(
    coordinates: torch.Tensor,
    bits: torch.Tensor,
    *,
    channel: int = MIRROR_CHANNEL,
) -> torch.Tensor:
    """Reflect only the coordinate frame; anatomical token names do not move."""

    if coordinates.ndim != 4 or coordinates.shape[-1] != 3:
        raise ValueError("Expected coordinates [B,T,J,3]")
    if bits.shape != (len(coordinates),):
        raise ValueError("Sensor bits must have shape [B]")
    reflected = coordinates.clone()
    sign = torch.where(bits.bool(), -1.0, 1.0).to(coordinates.dtype)
    reflected[..., channel] *= sign[:, None, None]
    return reflected


def apply_corruption(
    clean: torch.Tensor,
    valid_patch: torch.Tensor,
    semantic_path: torch.Tensor,
    sensor_bits: torch.Tensor,
    occlusion_keep: torch.Tensor,
    noise_seeds: Sequence[int] | torch.Tensor,
    config: CorruptionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return observed and matched nuisance-only coordinates and validity.

    The order is clean -> independent sensor/noise/occlusion -> semantic P.
    Consequently, applying the saved semantic path to the observation exactly
    reconstructs the nuisance-only reference.
    """

    config.validate()
    blocks = WINDOW_FRAMES // config.block_frames
    if clean.shape[1:] != (WINDOW_FRAMES, len(SWAP_INDEX), 3):
        raise ValueError("clean must have shape [B,64,11,3]")
    if valid_patch.shape != (len(clean), blocks, len(SWAP_INDEX)):
        raise ValueError("valid_patch must have shape [B,16,11]")
    if occlusion_keep.shape != valid_patch.shape:
        raise ValueError("occlusion_keep must match valid_patch")

    reference_valid_patch = valid_patch.bool() & occlusion_keep.bool()
    reference_valid_frame = reference_valid_patch.repeat_interleave(
        config.block_frames, dim=1
    )
    reference = sensor_reflect(clean, sensor_bits)
    if config.noise_std:
        noise = torch.empty_like(reference)
        seeds = [int(value) for value in noise_seeds]
        for index, noise_seed in enumerate(seeds):
            generator = torch.Generator(device="cpu").manual_seed(noise_seed)
            noise[index] = torch.randn(
                reference[index].shape,
                generator=generator,
                dtype=reference.dtype,
            )
        reference = reference + config.noise_std * noise
    reference = reference.masked_fill(~reference_valid_frame[..., None], 0.0)

    observed = semantic_permute_by_block(
        reference, semantic_path, block_frames=config.block_frames
    )
    observed_valid_frame = semantic_permute_by_block(
        reference_valid_frame, semantic_path, block_frames=config.block_frames
    )
    observed_valid_patch = observed_valid_frame.reshape(
        len(clean), blocks, config.block_frames, len(SWAP_INDEX)
    ).all(dim=2)
    return observed, observed_valid_patch, reference, reference_valid_patch


def boundary_features(
    coordinates: torch.Tensor,
    valid_patch: torch.Tensor,
    *,
    block_frames: int = TIME_PATCH_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute global-chart-invariant continuity features for adjacent blocks."""

    if coordinates.ndim != 4 or coordinates.shape[2:] != (len(SWAP_INDEX), 3):
        raise ValueError("Expected coordinates [B,T,11,3]")
    blocks = coordinates.shape[1] // block_frames
    if coordinates.shape[1] != blocks * block_frames:
        raise ValueError("block_frames must divide the temporal dimension")
    if valid_patch.shape != (len(coordinates), blocks, len(SWAP_INDEX)):
        raise ValueError("valid_patch shape disagrees with coordinates")

    view = coordinates.reshape(len(coordinates), blocks, block_frames, 11, 3)
    predicted = 2.0 * view[:, :-1, -1] - view[:, :-1, -2]
    current = view[:, 1:, 0]
    prior_valid = valid_patch[:, :-1]
    current_valid = valid_patch[:, 1:]
    pair_features = []
    same_costs = []
    swap_costs = []
    coverages = []
    eps = torch.finfo(coordinates.dtype).eps

    for left, right in MIRROR_PAIRS:
        same_distances = torch.stack(
            (
                torch.linalg.vector_norm(predicted[..., left, :] - current[..., left, :], dim=-1),
                torch.linalg.vector_norm(predicted[..., right, :] - current[..., right, :], dim=-1),
            ),
            dim=-1,
        )
        swap_distances = torch.stack(
            (
                torch.linalg.vector_norm(predicted[..., left, :] - current[..., right, :], dim=-1),
                torch.linalg.vector_norm(predicted[..., right, :] - current[..., left, :], dim=-1),
            ),
            dim=-1,
        )
        same_valid = torch.stack(
            (
                prior_valid[..., left] & current_valid[..., left],
                prior_valid[..., right] & current_valid[..., right],
            ),
            dim=-1,
        )
        swap_valid = torch.stack(
            (
                prior_valid[..., left] & current_valid[..., right],
                prior_valid[..., right] & current_valid[..., left],
            ),
            dim=-1,
        )

        def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            count = mask.sum(dim=-1)
            total = (values * mask).sum(dim=-1)
            return torch.where(count > 0, total / count.clamp_min(1), torch.zeros_like(total))

        same = masked_mean(same_distances, same_valid)
        swapped = masked_mean(swap_distances, swap_valid)
        coverage = 0.25 * (same_valid.sum(dim=-1) + swap_valid.sum(dim=-1))
        margin = same - swapped
        normalized_margin = margin / (same.abs() + swapped.abs() + eps)
        pair_features.extend((same, swapped, margin, normalized_margin, coverage))
        same_costs.append(same)
        swap_costs.append(swapped)
        coverages.append(coverage)

    features = torch.stack(pair_features, dim=-1)
    coverage = torch.stack(coverages, dim=-1).mean(dim=-1)
    pair_present = torch.stack(coverages, dim=-1) > 0
    denominator = pair_present.sum(dim=-1).clamp_min(1)
    same_cost = (
        torch.stack(same_costs, dim=-1) * pair_present
    ).sum(dim=-1) / denominator
    swap_cost = (
        torch.stack(swap_costs, dim=-1) * pair_present
    ).sum(dim=-1) / denominator
    margin = same_cost - swap_cost
    aggregate = torch.stack(
        (
            same_cost,
            swap_cost,
            margin,
            margin / (same_cost.abs() + swap_cost.abs() + eps),
            coverage,
        ),
        dim=-1,
    )
    return torch.cat((features, aggregate), dim=-1), same_cost, swap_cost, coverage


def edge_labels(path: torch.Tensor) -> torch.Tensor:
    if path.ndim != 2:
        raise ValueError("path must have shape [B,K]")
    return torch.logical_xor(path[:, 1:].bool(), path[:, :-1].bool())


def anchored_path_from_edges(edges: torch.Tensor, root: torch.Tensor) -> torch.Tensor:
    """Integrate relative Z2 edges using an explicit root anchor."""

    if edges.ndim != 2 or root.shape != (len(edges),):
        raise ValueError("Expected edges [B,K-1] and root [B]")
    states = [root.bool()]
    for index in range(edges.shape[1]):
        states.append(torch.logical_xor(states[-1], edges[:, index].bool()))
    return torch.stack(states, dim=1)


def candidate_paths(config: CorruptionConfig) -> torch.Tensor:
    """Enumerate exactly the clean and one-contiguous-segment path family."""

    config.validate()
    blocks = WINDOW_FRAMES // config.block_frames
    candidates = [torch.zeros(blocks, dtype=torch.bool)]
    for start in range(1, blocks - config.swapped_blocks):
        path = torch.zeros(blocks, dtype=torch.bool)
        path[start : start + config.swapped_blocks] = True
        candidates.append(path)
    return torch.stack(candidates)


def candidate_prior(
    candidates: torch.Tensor, event_probability: float
) -> torch.Tensor:
    """Return the generator prior: clean versus uniformly located local event."""

    if candidates.ndim != 2 or len(candidates) < 2:
        raise ValueError("Expected clean plus at least one event candidate")
    if not 0.0 < event_probability < 1.0:
        raise ValueError("event_probability must lie strictly between zero and one")
    prior = torch.full(
        (len(candidates),), event_probability / (len(candidates) - 1), dtype=torch.float32
    )
    prior[0] = 1.0 - event_probability
    return prior


def structured_path_log_posterior(
    edge_probabilities: torch.Tensor,
    candidates: torch.Tensor,
    event_probability: float,
    *,
    edge_prior: float,
) -> torch.Tensor:
    """Return normalized log masses for the structured composite posterior.

    The calibrated head estimates an edge posterior under the training edge
    prevalence.  Convert it to a likelihood factor before applying the known
    clean-versus-event path prior once over the admissible path family.
    """

    if edge_probabilities.ndim != 2:
        raise ValueError("edge_probabilities must have shape [B,K-1]")
    if candidates.ndim != 2 or candidates.shape[1] != edge_probabilities.shape[1] + 1:
        raise ValueError("Candidate path geometry does not match edge probabilities")
    candidate_edges = edge_labels(candidates).to(edge_probabilities.device)
    probabilities = edge_probabilities.to(torch.float64).clamp(1e-7, 1.0 - 1e-7)
    marginal = torch.as_tensor(
        edge_prior, dtype=probabilities.dtype, device=probabilities.device
    ).clamp(1e-7, 1.0 - 1e-7)
    log_evidence = (
        candidate_edges[None].float()
        * (probabilities[:, None].log() - marginal.log())
        + (~candidate_edges)[None].float()
        * ((1.0 - probabilities[:, None]).log() - (1.0 - marginal).log())
    ).sum(dim=-1)
    prior = candidate_prior(candidates, event_probability).to(
        device=probabilities.device, dtype=probabilities.dtype
    )
    logits = log_evidence + prior.log()[None]
    return logits - torch.logsumexp(logits, dim=1, keepdim=True)


def structured_path_posterior(
    edge_probabilities: torch.Tensor,
    candidates: torch.Tensor,
    event_probability: float,
    *,
    edge_prior: float,
) -> torch.Tensor:
    """Return candidate masses for the structured composite posterior."""

    return structured_path_log_posterior(
        edge_probabilities,
        candidates,
        event_probability,
        edge_prior=edge_prior,
    ).exp().to(edge_probabilities.dtype)


def map_candidate_path(posterior: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    if posterior.ndim != 2 or posterior.shape[1] != len(candidates):
        raise ValueError("Posterior and candidate paths disagree")
    return candidates.to(posterior.device).index_select(0, posterior.argmax(dim=1))


def posterior_edge_marginals(posterior: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    candidate_edges = edge_labels(candidates).to(posterior.device).float()
    return posterior @ candidate_edges


def target_candidate_indices(target_path: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    matches = (target_path[:, None].bool() == candidates[None].bool()).all(dim=-1)
    if not matches.any(dim=1).all() or (matches.sum(dim=1) != 1).any():
        raise ValueError("Target path is outside the declared corruption support")
    return matches.float().argmax(dim=1).long()


def structured_path_nll(
    edge_probabilities: torch.Tensor,
    target_path: torch.Tensor,
    candidates: torch.Tensor,
    event_probability: float,
    *,
    edge_prior: float,
) -> torch.Tensor:
    """Mean negative log probability of complete paths under the structured posterior."""

    log_posterior = structured_path_log_posterior(
        edge_probabilities,
        candidates,
        event_probability,
        edge_prior=edge_prior,
    )
    targets = target_candidate_indices(target_path, candidates)
    return -log_posterior[torch.arange(len(target_path)), targets].mean()


def kinematic_targets(
    coordinates: torch.Tensor,
    valid_patch: torch.Tensor,
    *,
    block_mask: torch.Tensor | None = None,
    block_frames: int = TIME_PATCH_FRAMES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return right-minus-left and total distal within-block motion energy."""

    blocks = coordinates.shape[1] // block_frames
    view = coordinates.reshape(len(coordinates), blocks, block_frames, 11, 3)
    velocity = view[:, :, 1:] - view[:, :, :-1]
    energy = velocity.square().sum(dim=-1)
    if valid_patch.shape != (len(coordinates), blocks, 11):
        raise ValueError("valid_patch shape disagrees with coordinates")
    valid = valid_patch.bool()
    if block_mask is not None:
        if block_mask.shape != (len(coordinates), blocks):
            raise ValueError("block_mask must have shape [B,K]")
        valid = valid & block_mask[:, :, None]

    contributions = []
    available = []
    for left, right in ((5, 6), (7, 8), (9, 10)):
        pair_valid = valid[..., left] & valid[..., right]
        left_energy = energy[..., left].mean(dim=2)
        right_energy = energy[..., right].mean(dim=2)
        contributions.append(torch.stack((right_energy - left_energy, right_energy + left_energy), dim=-1))
        available.append(pair_valid)
    contribution = torch.stack(contributions, dim=2)
    pair_available = torch.stack(available, dim=2)
    count = pair_available.sum(dim=(1, 2))
    total = (contribution * pair_available[..., None]).sum(dim=(1, 2))
    values = torch.where(count[:, None] > 0, total / count[:, None].clamp_min(1), torch.zeros_like(total))
    return values[:, 0], values[:, 1], count > 0


@torch.inference_mode()
def encode_coordinates(
    encoder: nn.Module,
    coordinates: torch.Tensor,
    valid_patch: torch.Tensor,
    device: torch.device,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Pool frozen EMA tokens with validity-aware output weights."""

    prior_mode = encoder.training
    encoder.eval()
    outputs = []
    amp = device.type == "cuda"
    for start in range(0, len(coordinates), batch_size):
        stop = start + batch_size
        batch = coordinates[start:stop].to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            tokens = encoder(batch)
        tokens = tokens.float()
        weights = valid_patch[start:stop].flatten(1).to(device, non_blocking=True)
        if tokens.shape[:2] != weights.shape:
            raise ValueError("Encoder token geometry does not match validity patches")
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1)
        pooled = (tokens * weights[..., None]).sum(dim=1) / denominator
        outputs.append(pooled.cpu())
    encoder.train(prior_mode)
    return torch.cat(outputs, dim=0)


def path_hamming_up_to_global_flip(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    direct = (predicted.bool() != target.bool()).float().mean(dim=1)
    flipped = (predicted.bool() == target.bool()).float().mean(dim=1)
    return torch.minimum(direct, flipped)


def segment_iou_up_to_global_flip(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    scores = []
    for candidate in (predicted.bool(), ~predicted.bool()):
        intersection = (candidate & target.bool()).sum(dim=1).float()
        union = (candidate | target.bool()).sum(dim=1).float()
        scores.append(torch.where(union > 0, intersection / union, torch.ones_like(union)))
    return torch.maximum(scores[0], scores[1])


def segment_iou_anchored(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intersection = (predicted.bool() & target.bool()).sum(dim=1).float()
    union = (predicted.bool() | target.bool()).sum(dim=1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def tolerant_event_f1(
    predicted_edges: torch.Tensor,
    target_edges: torch.Tensor,
    *,
    tolerance: int = 1,
) -> torch.Tensor:
    """One-to-one switch-event F1 with a fixed edge tolerance."""

    values = []
    for predicted, target in zip(predicted_edges.bool(), target_edges.bool()):
        predicted_events = torch.where(predicted)[0].tolist()
        target_events = torch.where(target)[0].tolist()
        target_cursor = 0
        matches = 0
        for event in predicted_events:
            while (
                target_cursor < len(target_events)
                and target_events[target_cursor] < event - tolerance
            ):
                target_cursor += 1
            if (
                target_cursor < len(target_events)
                and target_events[target_cursor] <= event + tolerance
            ):
                target_cursor += 1
                matches += 1
        denominator = len(predicted_events) + len(target_events)
        values.append(1.0 if denominator == 0 else 2.0 * matches / denominator)
    return torch.tensor(values, dtype=torch.float32)


def reliability_frame(
    labels: np.ndarray,
    probabilities: np.ndarray,
    method: str,
    *,
    bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    records = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        selected = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if not selected.any():
            continue
        records.append(
            {
                "method": method,
                "bin": index,
                "lower": lower,
                "upper": upper,
                "count": int(selected.sum()),
                "mean_probability": float(probabilities[selected].mean()),
                "event_rate": float(labels[selected].mean()),
            }
        )
    return pd.DataFrame(records)


def edge_metric_row(
    labels: np.ndarray,
    probabilities: np.ndarray,
    predicted_path: torch.Tensor,
    target_path: torch.Tensor,
    method: str,
    *,
    structured_log_posterior: torch.Tensor,
    candidates: torch.Tensor,
) -> dict[str, float | str]:
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    true_edges = edge_labels(target_path)
    predicted_edges = edge_labels(predicted_path)
    target_candidate = target_candidate_indices(target_path, candidates)
    target_log_probability = structured_log_posterior[
        torch.arange(len(target_path)), target_candidate
    ]
    clean = ~target_path.any(dim=1)
    event = ~clean
    false_switch = predicted_path.any(dim=1)
    return {
        "method": method,
        "edge_prevalence": float(labels.mean()),
        "edge_auprc": float(average_precision_score(labels, probabilities)),
        "edge_brier": float(brier_score_loss(labels, probabilities)),
        "edge_log_loss": float(log_loss(labels, clipped, labels=[0, 1])),
        "structured_path_nll": float(-target_log_probability.mean()),
        "path_hamming_anchored": float((predicted_path != target_path).float().mean(dim=1).mean()),
        "path_hamming_up_to_flip": float(
            path_hamming_up_to_global_flip(predicted_path, target_path).mean()
        ),
        "switch_f1_tolerance_1": float(
            tolerant_event_f1(predicted_edges, true_edges, tolerance=1).mean()
        ),
        "switch_f1_event_windows_tolerance_1": (
            float(
                tolerant_event_f1(
                    predicted_edges[event], true_edges[event], tolerance=1
                ).mean()
            )
            if event.any()
            else math.nan
        ),
        "segment_iou_up_to_flip": float(
            segment_iou_up_to_global_flip(predicted_path, target_path).mean()
        ),
        "segment_iou_anchored": float(segment_iou_anchored(predicted_path, target_path).mean()),
        "clean_false_switch_rate": float(false_switch[clean].float().mean()) if clean.any() else math.nan,
    }


@dataclass(frozen=True)
class CalibratedProbabilityHead:
    """A training-identity classifier with held-out structured-path calibration."""

    classifier: object
    temperature: float
    edge_prior: float

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        raw = self.classifier.predict_proba(features)[:, 1]
        return _temperature_scale(raw, self.temperature, self.edge_prior)


def _temperature_scale(
    probabilities: np.ndarray,
    temperature: float,
    edge_prior: float,
) -> np.ndarray:
    """Temper likelihood evidence while preserving the training edge prior."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped) - np.log1p(-clipped)
    clipped_prior = float(np.clip(edge_prior, 1e-7, 1.0 - 1e-7))
    prior_logit = math.log(clipped_prior) - math.log1p(-clipped_prior)
    return expit(prior_logit + (logits - prior_logit) / temperature)


def _structured_path_nll_numpy(
    probabilities: np.ndarray,
    target_paths: torch.Tensor,
    candidates: torch.Tensor,
    event_probability: float,
    *,
    edge_prior: float,
) -> float:
    """Numpy form of structured path NLL used only in scalar calibration."""

    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    candidate_edges = edge_labels(candidates).numpy().astype(np.float64)
    marginal = float(np.clip(edge_prior, 1e-7, 1.0 - 1e-7))
    positive_evidence = np.log(clipped) - math.log(marginal)
    negative_evidence = np.log1p(-clipped) - math.log1p(-marginal)
    scores = (
        positive_evidence[:, None, :] * candidate_edges[None]
        + negative_evidence[:, None, :] * (1.0 - candidate_edges[None])
    ).sum(axis=-1)
    log_prior = np.log(candidate_prior(candidates, event_probability).numpy())
    target_indices = target_candidate_indices(target_paths, candidates).numpy()
    log_posterior = scores + log_prior[None] - logsumexp(scores + log_prior[None], axis=1)[:, None]
    return float(-log_posterior[np.arange(len(target_paths)), target_indices].mean())


def _fit_path_temperature(
    raw_probabilities: np.ndarray,
    calibration_paths: torch.Tensor,
    calibration_coverage: np.ndarray,
    candidates: torch.Tensor,
    event_probability: float,
    *,
    edge_prior: float,
) -> float:
    """Calibrate one temperature against held-out complete-path likelihood."""

    if raw_probabilities.ndim != 2 or calibration_coverage.shape != raw_probabilities.shape:
        raise ValueError("Calibration probabilities and coverage must have shape [N,K-1]")
    if calibration_paths.shape != (len(raw_probabilities), raw_probabilities.shape[1] + 1):
        raise ValueError("Calibration paths do not match edge probabilities")
    target_indices = target_candidate_indices(calibration_paths, candidates)
    if not ((target_indices == 0).any() and (target_indices != 0).any()):
        raise ValueError("Calibration identities must contain clean and event paths")

    def objective(log_temperature: float) -> float:
        temperature = math.exp(log_temperature)
        calibrated = _temperature_scale(raw_probabilities, temperature, edge_prior)
        calibrated = np.where(calibration_coverage > 0, calibrated, edge_prior)
        return _structured_path_nll_numpy(
            calibrated,
            calibration_paths,
            candidates,
            event_probability,
            edge_prior=edge_prior,
        )

    result = minimize_scalar(
        objective,
        bounds=(math.log(0.05), math.log(20.0)),
        method="bounded",
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("Temperature calibration failed")
    return float(math.exp(result.x))


def fit_calibrated_probability_head(
    fit_features: np.ndarray,
    fit_labels: np.ndarray,
    calibration_features: np.ndarray,
    calibration_paths: torch.Tensor,
    calibration_coverage: np.ndarray,
    candidates: torch.Tensor,
    event_probability: float,
    *,
    edge_prior: float,
) -> CalibratedProbabilityHead:
    if np.unique(fit_labels).size != 2:
        raise ValueError("Model-fitting edge labels must contain both classes")
    if calibration_features.ndim != 3:
        raise ValueError("Calibration features must have shape [N,K-1,F]")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=0),
    )
    model.fit(fit_features, fit_labels)
    raw_probabilities = model.predict_proba(
        calibration_features.reshape(-1, calibration_features.shape[-1])
    )[:, 1].reshape(calibration_features.shape[:2])
    temperature = _fit_path_temperature(
        raw_probabilities,
        calibration_paths,
        calibration_coverage,
        candidates,
        event_probability,
        edge_prior=edge_prior,
    )
    return CalibratedProbabilityHead(model, temperature, edge_prior)


def fit_probe(embeddings: np.ndarray, targets: np.ndarray):
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit(embeddings, targets)
    return model


def predict_edges(
    model: CalibratedProbabilityHead,
    features: torch.Tensor,
    coverage: torch.Tensor,
    prior: float,
) -> torch.Tensor:
    shape = features.shape[:2]
    probabilities = model.predict_proba(features.reshape(-1, features.shape[-1]).numpy())
    probabilities = torch.from_numpy(probabilities.astype(np.float32)).reshape(shape)
    return torch.where(coverage > 0, probabilities, torch.full_like(probabilities, prior))


def _batch_draw(batch: Mapping[str, object]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    return (
        torch.as_tensor(batch["semantic_path"]).bool(),
        torch.as_tensor(batch["sensor_reflection_bit"]).bool(),
        torch.as_tensor(batch["occlusion_keep"]).bool(),
        [int(value) for value in torch.as_tensor(batch["noise_seed"]).tolist()],
    )


def collect_training_data(
    loader: DataLoader,
    encoder: nn.Module,
    device: torch.device,
    corruption: CorruptionConfig,
    *,
    encoder_batch_size: int,
    progress: bool = False,
) -> dict[str, np.ndarray]:
    feature_parts = []
    label_parts = []
    clean_embedding_parts = []
    target_parts = []
    edge_identity_parts = []
    probe_identity_parts = []
    swapped_target_parts = []
    swapped_target_identity_parts = []
    for batch_index, batch in enumerate(loader, start=1):
        clean = batch["coordinates"].float()
        valid = batch["valid"].bool()
        path, sensor, occlusion, noise_seeds = _batch_draw(batch)
        observed, observed_valid, _, _ = apply_corruption(
            clean, valid, path, sensor, occlusion, noise_seeds, corruption
        )
        features, _, _, _ = boundary_features(observed, observed_valid)
        odd, even, usable = kinematic_targets(clean, valid)
        swapped_odd, _, swapped_usable = kinematic_targets(clean, valid, block_mask=path)
        embeddings = encode_coordinates(
            encoder, clean, valid, device, batch_size=encoder_batch_size
        )
        feature_parts.append(features.reshape(-1, features.shape[-1]))
        label_parts.append(edge_labels(path).reshape(-1))
        identities = np.asarray(batch["identity"], dtype=str)
        edge_identity_parts.append(np.repeat(identities, features.shape[1]))
        clean_embedding_parts.append(embeddings[usable])
        target_parts.append(torch.stack((odd[usable], even[usable]), dim=1))
        probe_identity_parts.append(identities[usable.numpy()])
        swapped_target_parts.append(swapped_odd[swapped_usable])
        swapped_target_identity_parts.append(identities[swapped_usable.numpy()])
        if progress and (batch_index == 1 or batch_index % 25 == 0 or batch_index == len(loader)):
            print(
                f"[swap-probe] encoded training batch {batch_index}/{len(loader)}",
                flush=True,
            )
    return {
        "edge_features": torch.cat(feature_parts).numpy(),
        "edge_labels": torch.cat(label_parts).numpy().astype(np.int64),
        "edge_identity": np.concatenate(edge_identity_parts),
        "clean_embeddings": torch.cat(clean_embedding_parts).numpy(),
        "targets": torch.cat(target_parts).numpy(),
        "probe_identity": np.concatenate(probe_identity_parts),
        "swapped_odd_targets": torch.cat(swapped_target_parts).numpy(),
        "swapped_target_identity": np.concatenate(swapped_target_identity_parts),
    }


def _arm_rows(
    *,
    arm: str,
    batch: Mapping[str, object],
    odd_true: torch.Tensor,
    even_true: torch.Tensor,
    odd_direct: torch.Tensor,
    even_direct: torch.Tensor,
    probe_prediction: np.ndarray,
    embedding: torch.Tensor,
    reference_embedding: torch.Tensor,
    usable: torch.Tensor,
    swapped_odd_true: torch.Tensor,
    swapped_odd_direct: torch.Tensor,
    swapped_usable: torch.Tensor,
    odd_scale: float,
    even_scale: float,
    swapped_odd_scale: float,
    sign_threshold: float,
) -> list[dict[str, object]]:
    cosine = 1.0 - F.cosine_similarity(embedding, reference_embedding, dim=1).numpy()
    records = []
    for index, window_id in enumerate(batch["window_id"]):
        is_usable = bool(usable[index])
        is_swapped_usable = bool(swapped_usable[index])
        sign_scored = bool(abs(float(odd_true[index])) > sign_threshold)
        direct_odd_error = (
            abs(float(odd_direct[index] - odd_true[index])) / odd_scale
            if is_usable
            else math.nan
        )
        direct_even_error = (
            abs(float(even_direct[index] - even_true[index])) / even_scale
            if is_usable
            else math.nan
        )
        direct_odd_squared_error = (
            float((odd_direct[index] - odd_true[index]).square()) / odd_scale**2
            if is_usable
            else math.nan
        )
        direct_even_squared_error = (
            float((even_direct[index] - even_true[index]).square()) / even_scale**2
            if is_usable
            else math.nan
        )
        record = {
            "window_id": window_id,
            "identity": batch["identity"][index],
            "tensor_relative_path": batch["tensor_relative_path"][index],
            "start_frame": int(torch.as_tensor(batch["start_frame"])[index]),
            "arm": arm,
            "condition": "swapped_event" if bool(batch["semantic_path"][index].any()) else "clean_no_switch",
            "odd_true": float(odd_true[index]),
            "even_true": float(even_true[index]),
            "direct_odd_prediction": float(odd_direct[index]),
            "direct_even_prediction": float(even_direct[index]),
            "probe_odd_prediction": float(probe_prediction[index, 0]),
            "probe_even_prediction": float(probe_prediction[index, 1]),
            "direct_odd_nmae": direct_odd_error,
            "direct_even_nmae": direct_even_error,
            "direct_odd_nmse": direct_odd_squared_error,
            "direct_even_nmse": direct_even_squared_error,
            "probe_odd_nmae": (
                abs(float(probe_prediction[index, 0] - odd_true[index])) / odd_scale
                if is_usable
                else math.nan
            ),
            "probe_even_nmae": (
                abs(float(probe_prediction[index, 1] - even_true[index])) / even_scale
                if is_usable
                else math.nan
            ),
            "probe_odd_nmse": (
                float((probe_prediction[index, 0] - odd_true[index].item()) ** 2) / odd_scale**2
                if is_usable
                else math.nan
            ),
            "probe_even_nmse": (
                float((probe_prediction[index, 1] - even_true[index].item()) ** 2) / even_scale**2
                if is_usable
                else math.nan
            ),
            "embedding_cosine_distance_to_nuisance_reference": float(cosine[index]),
            "odd_sign_scored": sign_scored,
            "direct_odd_sign_correct": (
                float(np.sign(float(odd_direct[index])) == np.sign(float(odd_true[index])))
                if sign_scored and is_usable
                else math.nan
            ),
            "probe_odd_sign_correct": (
                float(np.sign(probe_prediction[index, 0]) == np.sign(float(odd_true[index])))
                if sign_scored and is_usable
                else math.nan
            ),
            "usable": is_usable,
            "swapped_direct_odd_nmae": (
                abs(float(swapped_odd_direct[index] - swapped_odd_true[index])) / swapped_odd_scale
                if is_swapped_usable
                else math.nan
            ),
            "swapped_direct_odd_nmse": (
                float((swapped_odd_direct[index] - swapped_odd_true[index]).square())
                / swapped_odd_scale**2
                if is_swapped_usable
                else math.nan
            ),
        }
        records.append(record)
    return records


def evaluate_validation(
    loader: DataLoader,
    encoder: nn.Module,
    device: torch.device,
    corruption: CorruptionConfig,
    continuity_head: CalibratedProbabilityHead,
    learned_head: CalibratedProbabilityHead,
    mask_head: CalibratedProbabilityHead,
    probe,
    *,
    edge_prior: float,
    odd_scale: float,
    even_scale: float,
    swapped_odd_scale: float,
    sign_threshold: float,
    encoder_batch_size: int,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate only validation identities under the declared structured path family."""

    rows = []
    edge_batches = []
    candidates = candidate_paths(corruption)
    fixed_prior = candidate_prior(candidates, corruption.event_probability)
    for batch_index, batch in enumerate(loader, start=1):
        clean = batch["coordinates"].float()
        valid = batch["valid"].bool()
        path, sensor, occlusion, noise_seeds = _batch_draw(batch)
        observed, observed_valid, reference, reference_valid = apply_corruption(
            clean, valid, path, sensor, occlusion, noise_seeds, corruption
        )
        oracle = semantic_permute_by_block(observed, path)
        oracle_valid = semantic_permute_patch_validity(observed_valid, path)
        if not torch.equal(oracle, reference) or not torch.equal(oracle_valid, reference_valid):
            raise RuntimeError("Oracle inverse failed to reconstruct the nuisance reference")

        features, _, _, coverage = boundary_features(observed, observed_valid)
        continuity_probability = predict_edges(
            continuity_head, features[..., -3:-2], coverage, edge_prior
        )
        learned_probability = predict_edges(learned_head, features, coverage, edge_prior)
        mask_features = features[..., 4::5]
        mask_probability = predict_edges(mask_head, mask_features, coverage, edge_prior)
        continuity_log_posterior = structured_path_log_posterior(
            continuity_probability,
            candidates,
            corruption.event_probability,
            edge_prior=edge_prior,
        )
        learned_log_posterior = structured_path_log_posterior(
            learned_probability,
            candidates,
            corruption.event_probability,
            edge_prior=edge_prior,
        )
        mask_log_posterior = structured_path_log_posterior(
            mask_probability,
            candidates,
            corruption.event_probability,
            edge_prior=edge_prior,
        )
        continuity_posterior = continuity_log_posterior.exp().float()
        learned_posterior = learned_log_posterior.exp().float()
        mask_posterior = mask_log_posterior.exp().float()
        prior_posterior = fixed_prior[None].expand(len(clean), -1)
        prior_log_posterior = prior_posterior.double().log()
        continuity_path = map_candidate_path(continuity_posterior, candidates)
        learned_path = map_candidate_path(learned_posterior, candidates)
        mask_path = map_candidate_path(mask_posterior, candidates)
        deterministic = {
            "clean_reference": (clean, valid, torch.zeros_like(path)),
            "corrupted_uncorrected": (
                observed,
                observed_valid,
                torch.zeros_like(path),
            ),
            "continuity_map": (
                semantic_permute_by_block(observed, continuity_path),
                semantic_permute_patch_validity(observed_valid, continuity_path),
                continuity_path,
            ),
            "learned_map": (
                semantic_permute_by_block(observed, learned_path),
                semantic_permute_patch_validity(observed_valid, learned_path),
                learned_path,
            ),
            "oracle": (oracle, oracle_valid, path),
        }

        odd_true, even_true, usable = kinematic_targets(clean, valid)
        swapped_odd_true, _, swapped_usable = kinematic_targets(
            clean, valid, block_mask=path
        )
        reference_embedding = encode_coordinates(
            encoder, reference, reference_valid, device, batch_size=encoder_batch_size
        )

        for arm, (coordinates, arm_valid, estimated_path) in deterministic.items():
            odd_direct, even_direct, arm_usable = kinematic_targets(coordinates, arm_valid)
            swapped_odd_direct, _, arm_swapped_usable = kinematic_targets(
                coordinates, arm_valid, block_mask=path
            )
            embedding = encode_coordinates(
                encoder, coordinates, arm_valid, device, batch_size=encoder_batch_size
            )
            prediction = probe.predict(embedding.numpy())
            rows.extend(
                _arm_rows(
                    arm=arm,
                    batch=batch,
                    odd_true=odd_true,
                    even_true=even_true,
                    odd_direct=odd_direct,
                    even_direct=even_direct,
                    probe_prediction=prediction,
                    embedding=embedding,
                    reference_embedding=reference_embedding,
                    usable=usable & arm_usable,
                    swapped_odd_true=swapped_odd_true,
                    swapped_odd_direct=swapped_odd_direct,
                    swapped_usable=swapped_usable & arm_swapped_usable,
                    odd_scale=odd_scale,
                    even_scale=even_scale,
                    swapped_odd_scale=swapped_odd_scale,
                    sign_threshold=sign_threshold,
                )
            )

        candidate_count, blocks = candidates.shape
        batch_count = len(clean)
        candidate_batch_paths = candidates[:, None].expand(-1, batch_count, -1)
        flat_paths = candidate_batch_paths.reshape(candidate_count * batch_count, blocks)
        repeated_coordinates = observed.unsqueeze(0).expand(candidate_count, -1, -1, -1, -1)
        repeated_valid = observed_valid.unsqueeze(0).expand(candidate_count, -1, -1, -1)
        flat_coordinates = repeated_coordinates.reshape(
            candidate_count * batch_count, WINDOW_FRAMES, 11, 3
        )
        corrected_samples = semantic_permute_by_block(flat_coordinates, flat_paths)
        corrected_valid = semantic_permute_patch_validity(
            repeated_valid.reshape(candidate_count * batch_count, blocks, 11),
            flat_paths,
        )
        sampled_odd, sampled_even, sampled_usable = kinematic_targets(
            corrected_samples, corrected_valid
        )
        repeated_true_swap = path.unsqueeze(0).expand(candidate_count, -1, -1).reshape(
            candidate_count * batch_count, blocks
        )
        sampled_swapped_odd, _, sampled_swapped_usable = kinematic_targets(
            corrected_samples, corrected_valid, block_mask=repeated_true_swap
        )
        sampled_embedding = encode_coordinates(
            encoder,
            corrected_samples,
            corrected_valid,
            device,
            batch_size=encoder_batch_size,
        )
        embedding_shape = sampled_embedding.shape[-1]
        sampled_embedding = sampled_embedding.reshape(candidate_count, batch_count, embedding_shape)
        sampled_predictions = probe.predict(
            sampled_embedding.reshape(candidate_count * batch_count, embedding_shape).numpy()
        ).reshape(candidate_count, batch_count, 2)
        weights = learned_posterior.T
        rows.extend(
            _arm_rows(
                arm="learned_structured_posterior",
                batch=batch,
                odd_true=odd_true,
                even_true=even_true,
                odd_direct=(weights * sampled_odd.reshape(candidate_count, batch_count)).sum(dim=0),
                even_direct=(weights * sampled_even.reshape(candidate_count, batch_count)).sum(dim=0),
                probe_prediction=(weights[..., None].numpy() * sampled_predictions).sum(axis=0),
                embedding=(weights[..., None] * sampled_embedding).sum(dim=0),
                reference_embedding=reference_embedding,
                usable=usable & sampled_usable.reshape(candidate_count, batch_count).all(dim=0),
                swapped_odd_true=swapped_odd_true,
                swapped_odd_direct=(
                    weights * sampled_swapped_odd.reshape(candidate_count, batch_count)
                ).sum(dim=0),
                swapped_usable=(
                    swapped_usable
                    & sampled_swapped_usable.reshape(candidate_count, batch_count).all(dim=0)
                ),
                odd_scale=odd_scale,
                even_scale=even_scale,
                swapped_odd_scale=swapped_odd_scale,
                sign_threshold=sign_threshold,
            )
        )

        learned_edge_marginal = posterior_edge_marginals(learned_posterior, candidates)
        continuity_edge_marginal = posterior_edge_marginals(continuity_posterior, candidates)
        mask_edge_marginal = posterior_edge_marginals(mask_posterior, candidates)
        prior_edge_marginal = posterior_edge_marginals(prior_posterior, candidates)
        edge_batches.append(
            {
                "labels": edge_labels(path),
                "path": path,
                "continuity_raw_edge_probability": continuity_probability,
                "continuity_probability": continuity_edge_marginal,
                "learned_raw_edge_probability": learned_probability,
                "learned_probability": learned_edge_marginal,
                "mask_raw_edge_probability": mask_probability,
                "mask_probability": mask_edge_marginal,
                "prior_probability": prior_edge_marginal,
                "continuity_posterior": continuity_posterior,
                "continuity_log_posterior": continuity_log_posterior,
                "learned_posterior": learned_posterior,
                "learned_log_posterior": learned_log_posterior,
                "mask_posterior": mask_posterior,
                "mask_log_posterior": mask_log_posterior,
                "prior_posterior": prior_posterior,
                "prior_log_posterior": prior_log_posterior,
                "continuity_path": continuity_path,
                "learned_path": learned_path,
                "mask_path": mask_path,
                "prior_path": map_candidate_path(prior_posterior, candidates),
                "posterior_entropy": -(
                    learned_posterior.double() * learned_log_posterior
                ).sum(dim=1) / math.log(candidate_count),
                "window_id": list(batch["window_id"]),
            }
        )
        if progress and (batch_index == 1 or batch_index % 10 == 0 or batch_index == len(loader)):
            print(
                f"[swap-probe] evaluated validation batch {batch_index}/{len(loader)}",
                flush=True,
            )

    window_metrics = pd.DataFrame(rows)
    labels_tensor = torch.cat([value["labels"] for value in edge_batches])
    target_path = torch.cat([value["path"] for value in edge_batches])
    labels = labels_tensor.flatten().numpy().astype(np.int64)
    metric_rows = []
    reliability = []
    for method, probability_key, path_key, log_posterior_key in (
        (
            "input_free_path_prior",
            "prior_probability",
            "prior_path",
            "prior_log_posterior",
        ),
        (
            "continuity",
            "continuity_probability",
            "continuity_path",
            "continuity_log_posterior",
        ),
        ("learned", "learned_probability", "learned_path", "learned_log_posterior"),
        ("mask_only_control", "mask_probability", "mask_path", "mask_log_posterior"),
    ):
        probabilities = torch.cat([value[probability_key] for value in edge_batches])
        predicted_path = torch.cat([value[path_key] for value in edge_batches])
        log_posterior = torch.cat([value[log_posterior_key] for value in edge_batches])
        flat_probability = probabilities.flatten().numpy()
        metric_rows.append(
            edge_metric_row(
                labels,
                flat_probability,
                predicted_path,
                target_path,
                method,
                structured_log_posterior=log_posterior,
                candidates=candidates,
            )
        )
        reliability.append(reliability_frame(labels, flat_probability, method))

    prior_sensitivity_rows = []
    raw_probability_by_method = {
        "continuity": torch.cat(
            [value["continuity_raw_edge_probability"] for value in edge_batches]
        ),
        "learned": torch.cat(
            [value["learned_raw_edge_probability"] for value in edge_batches]
        ),
        "mask_only_control": torch.cat(
            [value["mask_raw_edge_probability"] for value in edge_batches]
        ),
    }
    for evaluation_event_probability in sorted(
        {0.20, 0.50, float(corruption.event_probability)}
    ):
        prior = candidate_prior(candidates, evaluation_event_probability)
        prior_posterior = prior[None].expand(len(target_path), -1)
        prior_sensitivity_rows.append(
            {
                **edge_metric_row(
                    labels,
                    posterior_edge_marginals(prior_posterior, candidates).flatten().numpy(),
                    map_candidate_path(prior_posterior, candidates),
                    target_path,
                    "input_free_path_prior",
                    structured_log_posterior=prior_posterior.double().log(),
                    candidates=candidates,
                ),
                "evaluation_event_probability": evaluation_event_probability,
            }
        )
        for method, raw_probability in raw_probability_by_method.items():
            log_posterior = structured_path_log_posterior(
                raw_probability,
                candidates,
                evaluation_event_probability,
                edge_prior=edge_prior,
            )
            posterior = log_posterior.exp().float()
            prior_sensitivity_rows.append(
                {
                    **edge_metric_row(
                        labels,
                        posterior_edge_marginals(posterior, candidates).flatten().numpy(),
                        map_candidate_path(posterior, candidates),
                        target_path,
                        method,
                        structured_log_posterior=log_posterior,
                        candidates=candidates,
                    ),
                    "evaluation_event_probability": evaluation_event_probability,
                }
            )

    entropy_by_window = {
        window_id: float(value)
        for batch in edge_batches
        for window_id, value in zip(batch["window_id"], batch["posterior_entropy"])
    }
    window_metrics["learned_structured_path_entropy"] = window_metrics["window_id"].map(
        entropy_by_window
    )
    return (
        window_metrics,
        pd.DataFrame(metric_rows),
        pd.concat(reliability, ignore_index=True),
        pd.DataFrame(prior_sensitivity_rows),
    )


def summarize_window_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "direct_odd_nmae",
        "direct_even_nmae",
        "direct_odd_nmse",
        "direct_even_nmse",
        "probe_odd_nmae",
        "probe_even_nmae",
        "probe_odd_nmse",
        "probe_even_nmse",
        "embedding_cosine_distance_to_nuisance_reference",
        "direct_odd_sign_correct",
        "probe_odd_sign_correct",
        "swapped_direct_odd_nmae",
    ]
    window = frame.groupby("arm", sort=False)[metrics].mean().add_prefix("window_mean_")
    sequence = frame.groupby(
        ["arm", "identity", "tensor_relative_path"], sort=False
    )[metrics].mean()
    identity = (
        sequence.groupby(["arm", "identity"], sort=False)
        .mean()
        .groupby("arm", sort=False)
        .mean()
        .add_prefix("identity_macro_")
    )
    counts = frame.groupby("arm", sort=False).agg(
        windows=("window_id", "nunique"), identities=("identity", "nunique")
    )
    return counts.join(window).join(identity).reset_index()


def summarize_condition_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Report clean/no-switch and event windows separately for false-correction checks."""

    metrics = [
        "direct_odd_nmae",
        "direct_even_nmae",
        "direct_odd_nmse",
        "probe_odd_nmae",
        "probe_even_nmae",
        "probe_odd_nmse",
        "embedding_cosine_distance_to_nuisance_reference",
    ]
    sequence = frame.groupby(
        ["condition", "arm", "identity", "tensor_relative_path"], sort=False
    )[metrics].mean()
    identity = sequence.groupby(["condition", "arm", "identity"], sort=False).mean()
    summary = identity.groupby(["condition", "arm"], sort=False).mean().reset_index()
    counts = frame.groupby(["condition", "arm"], sort=False).agg(
        windows=("window_id", "nunique"), identities=("identity", "nunique")
    )
    return counts.join(summary.set_index(["condition", "arm"])).reset_index()


def summarize_uncertainty(frame: pd.DataFrame) -> pd.DataFrame:
    """Report paired posterior-minus-MAP errors by fixed structured-entropy bin."""

    selected = frame.loc[
        frame["arm"].isin(("learned_map", "learned_structured_posterior"))
    ].copy()
    index = [
        "window_id",
        "identity",
        "tensor_relative_path",
        "condition",
        "learned_structured_path_entropy",
    ]
    metrics = ("direct_odd_nmse", "probe_odd_nmse", "probe_even_nmse")
    paired = selected.pivot(index=index, columns="arm", values=list(metrics))
    paired.columns = [f"{metric}_{arm}" for metric, arm in paired.columns]
    paired = paired.reset_index()
    for metric in metrics:
        paired[f"posterior_minus_map_{metric}"] = (
            paired[f"{metric}_learned_structured_posterior"]
            - paired[f"{metric}_learned_map"]
        )
    paired["entropy_bin"] = pd.cut(
        paired["learned_structured_path_entropy"],
        bins=(-1e-9, 0.25, 0.50, 0.75, 1.0 + 1e-9),
        labels=("[0,.25]", "(.25,.50]", "(.50,.75]", "(.75,1]"),
        include_lowest=True,
    )
    delta_columns = [f"posterior_minus_map_{metric}" for metric in metrics]
    sequence = paired.groupby(
        ["entropy_bin", "condition", "identity", "tensor_relative_path"], observed=True
    )[delta_columns].mean()
    identity = sequence.groupby(
        ["entropy_bin", "condition", "identity"], observed=True
    ).mean()
    summary = (
        identity.groupby(["entropy_bin", "condition"], observed=True)[delta_columns]
        .mean()
        .reset_index()
    )
    counts = (
        paired.groupby(["entropy_bin", "condition"], observed=True)["window_id"]
        .nunique()
        .rename("windows")
        .reset_index()
    )
    return counts.merge(summary, on=["entropy_bin", "condition"], validate="one_to_one")


def select_nonoverlapping_windows(
    window_index: pd.DataFrame,
    split: str,
    limit: int | None,
    seed: int,
) -> pd.DataFrame:
    """Select deterministic 64-frame-grid windows without source-frame overlap."""

    selected = window_index.loc[
        (window_index["split"] == split)
        & (window_index["start_frame"] % WINDOW_FRAMES == 0)
    ].copy()
    if selected.empty:
        raise ValueError(f"No non-overlapping windows for {split}")
    selected["_selection_key"] = selected["window_id"].map(
        lambda value: _stable_seed(seed, f"select-{split}", str(value))
    )
    selected = selected.sort_values("_selection_key", kind="stable")
    if limit is not None:
        if limit < 1:
            raise ValueError("Window limits must be positive")
        selected = selected.head(limit)
    return (
        selected.drop(columns="_selection_key")
        .sort_values(["sequence_index", "start_frame"], kind="stable")
        .reset_index(drop=True)
    )


def validate_encoder_contract(encoder: nn.Module) -> None:
    config = getattr(encoder, "config", None)
    if config is None:
        raise ValueError("Frozen encoder has no config")
    expected = {
        "frames": WINDOW_FRAMES,
        "segment_length": TIME_PATCH_FRAMES,
        "joints": len(SWAP_INDEX),
    }
    actual = {name: getattr(config, name, None) for name in expected}
    if actual != expected:
        raise ValueError(f"Checkpoint geometry {actual} does not match {expected}")
    if tuple(getattr(config, "mirror_pairs", ())) != tuple(MIRROR_PAIRS):
        raise ValueError("Checkpoint bilateral-pair contract does not match Core11")
    if getattr(config, "mirror_channel", None) != MIRROR_CHANNEL:
        raise ValueError("Checkpoint mediolateral reflection channel does not match Core11")


def split_calibration_identities(
    rows: pd.DataFrame,
    seed: int,
    *,
    fraction: float = 0.20,
    draws: Mapping[str, WindowDraw] | None = None,
) -> tuple[set[str], set[str]]:
    """Reserve whole identities while preserving required path support."""

    if not 0.0 < fraction < 1.0:
        raise ValueError("calibration fraction must lie strictly between zero and one")
    identities = sorted(set(rows["identity"].astype(str)))
    if len(identities) < 2:
        raise ValueError("At least two training identities are needed for calibration")
    ordered = sorted(
        identities, key=lambda value: _stable_seed(seed, "calibration-identity", value)
    )
    calibration_count = min(max(1, round(fraction * len(ordered))), len(ordered) - 1)
    calibration = set(ordered[:calibration_count])
    fit = set(ordered).difference(calibration)
    if draws is None:
        return fit, calibration

    required = {"window_id", "identity"}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Window metadata is missing columns: {sorted(missing)}")
    path_classes = {identity: set() for identity in ordered}
    for row in rows.loc[:, ["window_id", "identity"]].to_dict(orient="records"):
        window_id = row["window_id"]
        if window_id not in draws:
            raise ValueError(f"No corruption draw for training window {window_id}")
        path_classes[str(row["identity"])].add(bool(draws[window_id].event_present))

    def calibration_has_support(identities: set[str]) -> bool:
        support = set().union(*(path_classes[identity] for identity in identities))
        return support == {False, True}

    def fit_has_positive_edges(identities: set[str]) -> bool:
        return any(True in path_classes[identity] for identity in identities)

    if calibration_has_support(calibration) and fit_has_positive_edges(fit):
        return fit, calibration

    clean_only = [identity for identity in ordered if path_classes[identity] == {False}]
    event_only = [identity for identity in ordered if path_classes[identity] == {True}]

    def calibration_seed_candidates() -> Iterable[tuple[str, ...]]:
        for identity in ordered:
            if path_classes[identity] == {False, True}:
                yield (identity,)
        for clean_identity in clean_only:
            for event_identity in event_only:
                yield clean_identity, event_identity

    for calibration_seed in calibration_seed_candidates():
        if len(calibration_seed) > calibration_count:
            continue
        remaining = [identity for identity in ordered if identity not in calibration_seed]
        protected_fit_identity = next(
            (identity for identity in remaining if True in path_classes[identity]),
            None,
        )
        if protected_fit_identity is None:
            continue
        fillers_needed = calibration_count - len(calibration_seed)
        fillers = [
            identity for identity in remaining if identity != protected_fit_identity
        ][:fillers_needed]
        if len(fillers) != fillers_needed:
            continue
        calibration = set(calibration_seed).union(fillers)
        fit = set(ordered).difference(calibration)
        if calibration_has_support(calibration) and fit_has_positive_edges(fit):
            return fit, calibration

    raise ValueError(
        "Cannot split training identities so calibration contains clean and event "
        "paths while model fitting retains event paths"
    )


def run_probe(
    *,
    train_dataset: Dataset,
    train_rows: pd.DataFrame,
    validation_dataset: Dataset,
    validation_rows: pd.DataFrame,
    encoder: nn.Module,
    device: torch.device,
    output_dir: Path,
    corruption: CorruptionConfig,
    seed: int,
    batch_size: int,
    encoder_batch_size: int,
    num_workers: int,
    provenance: Mapping[str, object],
    progress: bool = False,
) -> dict[str, object]:
    """Fit lightweight heads on train identities and evaluate validation only."""

    validate_encoder_contract(encoder)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train_draws = build_draws(train_rows, seed, corruption)
    validation_draws = build_draws(validation_rows, seed, corruption)
    train = GaugeWindowDataset(train_dataset, train_rows, train_draws)
    validation = GaugeWindowDataset(validation_dataset, validation_rows, validation_draws)
    loader_options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
    }
    if num_workers:
        loader_options.update({"persistent_workers": False, "prefetch_factor": 1})
    train_loader = DataLoader(train, **loader_options)
    validation_loader = DataLoader(validation, **loader_options)

    fit_identities, calibration_identities = split_calibration_identities(
        train_rows, seed, draws=train_draws
    )
    if progress:
        print(
            "[swap-probe] fitting data: "
            f"{len(train)} windows, {len(fit_identities)} model-fit identities, "
            f"{len(calibration_identities)} calibration identities",
            flush=True,
        )
    training = collect_training_data(
        train_loader,
        encoder,
        device,
        corruption,
        encoder_batch_size=encoder_batch_size,
        progress=progress,
    )
    labels = training["edge_labels"]
    features = training["edge_features"]
    coverage_columns = np.arange(4, features.shape[1], 5)
    edge_fit = np.isin(training["edge_identity"], list(fit_identities))
    edge_calibration = ~edge_fit
    edge_prior = float(labels[edge_fit].mean())
    candidates = candidate_paths(corruption)
    edge_count = candidates.shape[1] - 1
    if edge_calibration.sum() % edge_count:
        raise RuntimeError("Calibration edge rows do not form complete windows")
    calibration_full_features = features[edge_calibration].reshape(
        -1, edge_count, features.shape[1]
    )
    calibration_paths = anchored_path_from_edges(
        torch.from_numpy(labels[edge_calibration].reshape(-1, edge_count).astype(bool)),
        torch.zeros(len(calibration_full_features), dtype=torch.bool),
    )
    calibration_coverage = calibration_full_features[..., -1]
    continuity_head = fit_calibrated_probability_head(
        features[edge_fit, -3:-2],
        labels[edge_fit],
        calibration_full_features[..., -3:-2],
        calibration_paths,
        calibration_coverage,
        candidates,
        corruption.event_probability,
        edge_prior=edge_prior,
    )
    learned_head = fit_calibrated_probability_head(
        features[edge_fit],
        labels[edge_fit],
        calibration_full_features,
        calibration_paths,
        calibration_coverage,
        candidates,
        corruption.event_probability,
        edge_prior=edge_prior,
    )
    mask_head = fit_calibrated_probability_head(
        features[edge_fit][:, coverage_columns],
        labels[edge_fit],
        calibration_full_features[..., coverage_columns],
        calibration_paths,
        calibration_coverage,
        candidates,
        corruption.event_probability,
        edge_prior=edge_prior,
    )
    probe_fit = np.isin(training["probe_identity"], list(fit_identities))
    targets = training["targets"][probe_fit]
    if not len(targets):
        raise ValueError("No usable clean targets in model-fitting identities")
    odd_scale = max(float(np.std(targets[:, 0])), 1e-8)
    even_scale = max(float(np.std(targets[:, 1])), 1e-8)
    swapped_fit = np.isin(training["swapped_target_identity"], list(fit_identities))
    swapped_odd_scale = max(
        float(np.std(training["swapped_odd_targets"][swapped_fit])), 1e-8
    )
    sign_threshold = 0.10 * odd_scale
    probe = fit_probe(training["clean_embeddings"][probe_fit], targets)

    if progress:
        print(
            f"[swap-probe] evaluating {len(validation)} validation windows over "
            f"{len(candidate_paths(corruption))} admissible paths per window",
            flush=True,
        )

    window_metrics, edge_metrics, reliability, prior_sensitivity = evaluate_validation(
        validation_loader,
        encoder,
        device,
        corruption,
        continuity_head,
        learned_head,
        mask_head,
        probe,
        edge_prior=edge_prior,
        odd_scale=odd_scale,
        even_scale=even_scale,
        swapped_odd_scale=swapped_odd_scale,
        sign_threshold=sign_threshold,
        encoder_batch_size=encoder_batch_size,
        progress=progress,
    )
    summary = summarize_window_metrics(window_metrics)
    condition_summary = summarize_condition_metrics(window_metrics)
    uncertainty = summarize_uncertainty(window_metrics)
    draw_manifest = pd.concat(
        (
            draws_to_frame(train_rows, train_draws, split="train", config=corruption),
            draws_to_frame(
                validation_rows,
                validation_draws,
                split="validation",
                config=corruption,
            ),
        ),
        ignore_index=True,
    )

    atomic_dataframe_to_csv(draw_manifest, output_dir / "corruption_manifest.csv")
    atomic_dataframe_to_csv(window_metrics, output_dir / "validation_window_metrics.csv")
    atomic_dataframe_to_csv(edge_metrics, output_dir / "validation_edge_metrics.csv")
    atomic_dataframe_to_csv(reliability, output_dir / "validation_reliability.csv")
    atomic_dataframe_to_csv(
        prior_sensitivity, output_dir / "validation_prior_sensitivity.csv"
    )
    atomic_dataframe_to_csv(uncertainty, output_dir / "validation_uncertainty.csv")
    atomic_dataframe_to_csv(condition_summary, output_dir / "validation_condition_summary.csv")
    atomic_dataframe_to_csv(summary, output_dir / "summary.csv")
    atomic_joblib(
        output_dir / "lightweight_models.joblib",
        {
            "continuity_head": continuity_head,
            "learned_edge_head": learned_head,
            "mask_only_control": mask_head,
            "clean_embedding_probe": probe,
        },
    )
    config_payload = {
        "version": SWAP_PROBE_VERSION,
        "scope": "validation-only anchored mechanism screen; not SG-JEPA training evidence",
        "seed": seed,
        "train_windows": len(train),
        "validation_windows": len(validation),
        "batch_size": batch_size,
        "encoder_batch_size": encoder_batch_size,
        "num_workers": num_workers,
        "inference": (
            "exact enumeration over clean plus one contiguous local segment paths "
            "under a calibrated composite posterior"
        ),
        "prior_sensitivity_event_probabilities": sorted(
            {0.20, 0.50, corruption.event_probability}
        ),
        "edge_training_prevalence": edge_prior,
        "model_fit_identities": len(fit_identities),
        "calibration_identities": len(calibration_identities),
        "path_temperature_calibration": {
            "objective": "held-out complete-path negative log likelihood",
            "temperatures": {
                "continuity": continuity_head.temperature,
                "learned": learned_head.temperature,
                "mask_only_control": mask_head.temperature,
            },
        },
        "odd_scale_train_std": odd_scale,
        "even_scale_train_std": even_scale,
        "swapped_odd_scale_train_std": swapped_odd_scale,
        "odd_sign_threshold": sign_threshold,
        "corruption": asdict(corruption),
        "arms": list(ARMS),
        "diagnostic_references": list(REFERENCE_ARMS),
        "provenance": dict(provenance),
    }
    atomic_json(output_dir / "effective_config.json", config_payload)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in output_dir.iterdir()
        if path.is_file()
    }
    complete = {
        "status": "complete",
        "test_split_evaluated": False,
        "artifact_sha256": artifact_hashes,
        "primary_caveat": "Legacy frozen encoder and synthetic root anchor make this exploratory.",
    }
    atomic_json(output_dir / "COMPLETE.json", complete)
    if progress:
        print(f"[swap-probe] sealed {output_dir}", flush=True)
    return {
        "summary": summary,
        "edge_metrics": edge_metrics,
        "uncertainty": uncertainty,
        "prior_sensitivity": prior_sensitivity,
        "config": config_payload,
    }
