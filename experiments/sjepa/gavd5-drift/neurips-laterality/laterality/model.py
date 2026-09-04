from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .geometry import FULL_MIRROR_PAIRS


class SkeletonPatchEncoder(nn.Module):
    def __init__(
        self,
        frames: int = 64,
        joints: int = 33,
        coordinate_dim: int = 3,
        segment_length: int = 4,
        embed_dim: int = 64,
        depth: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if frames % segment_length:
            raise ValueError("frames must be divisible by segment_length")
        self.frames = frames
        self.joints = joints
        self.coordinate_dim = coordinate_dim
        self.segment_length = segment_length
        self.embed_dim = embed_dim
        self.segments = frames // segment_length
        self.patch_embed = nn.Linear(segment_length * coordinate_dim, embed_dim)
        self.time_pos = nn.Parameter(torch.randn(self.segments, embed_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(joints, embed_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def patchify(self, coordinates: torch.Tensor) -> torch.Tensor:
        batch, frames, joints, channels = coordinates.shape
        expected = (self.frames, self.joints, self.coordinate_dim)
        if (frames, joints, channels) != expected:
            raise ValueError(f"Expected [B, {expected}], received {tuple(coordinates.shape)}")
        patches = coordinates.reshape(
            batch, self.segments, self.segment_length, joints, channels
        )
        return patches.permute(0, 1, 3, 2, 4).contiguous().flatten(3)

    def forward(
        self,
        coordinates: torch.Tensor,
        valid_patch: torch.Tensor | None = None,
        hide_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = self.patch_embed(self.patchify(coordinates))
        if hide_mask is not None:
            if hide_mask.shape != tokens.shape[:-1]:
                raise ValueError("hide_mask shape does not match patch tokens")
            tokens = tokens.masked_fill(hide_mask[..., None], 0.0)
        tokens = tokens + self.time_pos[None, :, None, :] + self.joint_pos[None, None, :, :]
        batch = len(tokens)
        flat = tokens.reshape(batch, self.segments * self.joints, self.embed_dim)
        padding_mask = None
        if valid_patch is not None:
            if valid_patch.shape != tokens.shape[:-1]:
                raise ValueError("valid_patch shape does not match patch tokens")
            padding_mask = ~valid_patch.reshape(batch, -1)
            if padding_mask.all(dim=1).any():
                raise ValueError("Every sequence needs at least one fully valid patch")
        encoded = self.norm(self.blocks(flat, src_key_padding_mask=padding_mask))
        if padding_mask is not None:
            encoded = encoded.masked_fill(padding_mask[..., None], 0.0)
        return encoded


class SkeletonPredictor(nn.Module):
    def __init__(
        self,
        segments: int,
        joints: int,
        encoder_dim: int,
        depth: int,
        heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.segments = segments
        self.joints = joints
        self.encoder_to_predictor = nn.Linear(encoder_dim, encoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.time_pos = nn.Parameter(torch.randn(segments, encoder_dim) * 0.02)
        self.joint_pos = nn.Parameter(torch.randn(joints, encoder_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim,
            nhead=heads,
            dim_feedforward=encoder_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(encoder_dim)
        self.output = nn.Linear(encoder_dim, encoder_dim)

    def forward(
        self,
        context_features: torch.Tensor,
        target_mask: torch.Tensor,
        valid_patch: torch.Tensor,
    ) -> torch.Tensor:
        batch = len(context_features)
        flat_target = target_mask.reshape(batch, -1)
        flat_valid = valid_patch.reshape(batch, -1)
        full = self.encoder_to_predictor(context_features)
        full = torch.where(flat_target[..., None], self.mask_token.expand_as(full), full)
        positions = (self.time_pos[:, None] + self.joint_pos[None]).reshape(
            1, self.segments * self.joints, -1
        )
        full = full + positions
        predicted = self.output(
            self.norm(self.blocks(full, src_key_padding_mask=~flat_valid))
        )
        return predicted[flat_target].reshape(batch, -1, predicted.shape[-1])


class SJEPAGait(nn.Module):
    def __init__(
        self,
        frames: int = 64,
        joints: int = 33,
        coordinate_dim: int = 3,
        segment_length: int = 4,
        embed_dim: int = 64,
        encoder_depth: int = 2,
        predictor_depth: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.view_encoder = SkeletonPatchEncoder(
            frames,
            joints,
            coordinate_dim,
            segment_length,
            embed_dim,
            encoder_depth,
            heads,
        )
        self.target_encoder = copy.deepcopy(self.view_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)
        self.predictor = SkeletonPredictor(
            self.view_encoder.segments,
            joints,
            embed_dim,
            predictor_depth,
            heads,
        )
        self.register_buffer("target_center", torch.zeros(embed_dim))

    def forward(
        self,
        view: torch.Tensor,
        target: torch.Tensor,
        valid_patch: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (target_mask & ~valid_patch).any():
            raise ValueError("Target mask selected an invalid patch")
        context = self.view_encoder(view, valid_patch, hide_mask=target_mask)
        predicted = self.predictor(context, target_mask, valid_patch)
        with torch.no_grad():
            target_tokens = self.target_encoder(target, valid_patch)
            flat_mask = target_mask.reshape(len(target), -1)
            selected = target_tokens[flat_mask].reshape(
                len(target), -1, target_tokens.shape[-1]
            )
        return predicted, selected

    @torch.no_grad()
    def update_target(self, momentum: float) -> None:
        for target, view in zip(
            self.target_encoder.parameters(), self.view_encoder.parameters()
        ):
            target.mul_(momentum).add_(view, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(self, targets: torch.Tensor, beta: float = 0.9) -> None:
        self.target_center.mul_(beta).add_(
            targets.mean(dim=(0, 1)), alpha=1.0 - beta
        )


class VICRegProjector(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dimension, dimension), nn.GELU(), nn.Linear(dimension, dimension)
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


def valid_patches(valid: torch.Tensor, segment_length: int) -> torch.Tensor:
    batch, frames, joints = valid.shape
    if frames % segment_length:
        raise ValueError("valid frame count must be divisible by segment_length")
    return valid.reshape(batch, frames // segment_length, segment_length, joints).all(dim=2)


def uniform_authorized_mask(
    valid_patch: np.ndarray,
    authorized_joints: Sequence[int],
    mask_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    valid_array = np.asarray(valid_patch, dtype=bool)
    eligible_joints = np.zeros(valid_array.shape[-1], dtype=bool)
    eligible_joints[np.asarray(authorized_joints, dtype=int)] = True
    eligible = valid_array & eligible_joints[None, None, :]
    counts = eligible.reshape(len(eligible), -1).sum(axis=1)
    if np.any(counts < 2):
        raise ValueError("Every sample needs at least two valid authorized patches")
    masked_count = min(max(1, int(np.floor(counts.min() * mask_fraction))), int(counts.min()) - 1)
    output = np.zeros_like(eligible)
    for row in range(len(output)):
        candidates = np.flatnonzero(eligible[row].reshape(-1))
        chosen = rng.choice(candidates, size=masked_count, replace=False)
        output[row].reshape(-1)[chosen] = True
    return output


def anatomical_reflect_tensor(
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    rows: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    reflected = coordinates.clone()
    reflected_valid = valid.clone()
    selected = torch.ones(len(coordinates), dtype=torch.bool, device=coordinates.device) if rows is None else rows
    if selected.any():
        selected_coordinates = reflected[selected].clone()
        selected_validity = reflected_valid[selected].clone()
        selected_coordinates[..., 0] *= -1
        for left, right in FULL_MIRROR_PAIRS:
            left_coordinates = selected_coordinates[:, :, left].clone()
            selected_coordinates[:, :, left] = selected_coordinates[:, :, right]
            selected_coordinates[:, :, right] = left_coordinates
            left_validity = selected_validity[:, :, left].clone()
            selected_validity[:, :, left] = selected_validity[:, :, right]
            selected_validity[:, :, right] = left_validity
        reflected[selected] = selected_coordinates
        reflected_valid[selected] = selected_validity
    return reflected, reflected_valid


def geometric_view(
    coordinates: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_degrees: float = 8.0,
    translate: float = 0.03,
) -> torch.Tensor:
    view = coordinates.clone()
    batch = len(view)
    angles = (torch.rand(batch, device=view.device) * 2.0 - 1.0) * math.radians(max_degrees)
    cosine, sine = torch.cos(angles), torch.sin(angles)
    old_x = view[..., 0].clone()
    old_z = view[..., 2].clone()
    view[..., 0] = cosine[:, None, None] * old_x + sine[:, None, None] * old_z
    view[..., 2] = -sine[:, None, None] * old_x + cosine[:, None, None] * old_z
    offsets = (torch.rand(batch, 1, 1, 2, device=view.device) * 2.0 - 1.0) * translate
    view[..., :2] += offsets
    return view.masked_fill(~valid[..., None], 0.0)


def authorized_pool(
    tokens: torch.Tensor,
    valid_patch: torch.Tensor,
    authorized_joints: Sequence[int],
) -> torch.Tensor:
    batch, segments, _, dimension = tokens.shape
    selected = tokens[:, :, authorized_joints].reshape(batch, -1, dimension)
    weights = valid_patch[:, :, authorized_joints].reshape(batch, -1).to(tokens.dtype)
    denominator = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
    return (selected * weights[..., None]).sum(dim=1) / denominator


def sjepa_cross_entropy(
    predicted: torch.Tensor,
    targets: torch.Tensor,
    center: torch.Tensor,
    predictor_temperature: float = 0.10,
    target_temperature: float = 0.06,
) -> torch.Tensor:
    target_probability = torch.softmax(
        (targets - center[None, None]) / target_temperature, dim=-1
    ).detach()
    prediction_log_probability = torch.log_softmax(
        predicted / predictor_temperature, dim=-1
    )
    return -(target_probability * prediction_log_probability).sum(dim=-1).mean()


def _off_diagonal(matrix: torch.Tensor) -> torch.Tensor:
    size, other = matrix.shape
    if size != other:
        raise ValueError("Covariance matrix must be square")
    return matrix.flatten()[:-1].view(size - 1, size + 1)[:, 1:].flatten()


def vicreg_loss(first: torch.Tensor, second: torch.Tensor, epsilon: float = 1e-4) -> torch.Tensor:
    invariance = F.mse_loss(first, second)
    first_std = torch.sqrt(first.var(dim=0, unbiased=False) + epsilon)
    second_std = torch.sqrt(second.var(dim=0, unbiased=False) + epsilon)
    variance = 0.5 * (
        F.relu(1.0 - first_std).mean() + F.relu(1.0 - second_std).mean()
    )
    first_centered = first - first.mean(dim=0)
    second_centered = second - second.mean(dim=0)
    denominator = max(len(first) - 1, 1)
    first_covariance = first_centered.T @ first_centered / denominator
    second_covariance = second_centered.T @ second_centered / denominator
    covariance = (
        _off_diagonal(first_covariance).square().sum()
        + _off_diagonal(second_covariance).square().sum()
    ) / (2.0 * first.shape[1])
    return 25.0 * invariance + 25.0 * variance + covariance


def cosine_ema(step: int, total_steps: int, start: float, end: float = 1.0) -> float:
    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0
