"""Compact, explicitly adapted skeleton JEPA with optional stochastic variants.

The default objective follows skeleton S-JEPA's centered, sharpened categorical
latent targets. This teaching architecture uses final-layer targets and joint-time
rather than multi-frame patch tokens. It is not an official reproduction.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .masking import JOINT_IDS


@dataclass(frozen=True)
class ModelConfig:
    frames: int = 16
    joint_ids: tuple[int, ...] = JOINT_IDS
    dim: int = 48
    heads: int = 4
    encoder_layers: int = 2
    predictor_layers: int = 1
    target_temperature: float = 0.06
    prediction_temperature: float = 0.1
    ema: float = 0.99
    center_ema: float = 0.9
    position_jitter: float = 0.0
    latent_dim: int = 0

    def __post_init__(self):
        if tuple(self.joint_ids) != JOINT_IDS:
            raise ValueError("Only the exact, deduplicated 16-joint contract is supported")
        if self.frames < 2 or self.dim < 4 or self.dim % self.heads:
            raise ValueError("Invalid frames, embedding dimension or attention heads")
        if self.dim % 2 or self.latent_dim < 0 or self.position_jitter < 0:
            raise ValueError("Need even dim, nonnegative jitter and latent dimension")
        if not 0 <= self.ema < 1 or not 0 <= self.center_ema < 1:
            raise ValueError("EMA rates must be in [0,1)")
        if self.prediction_temperature <= 0 or self.target_temperature <= 0:
            raise ValueError("Temperatures must be positive")


def _transformer(dim, heads, layers):
    layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout=0.0,
                                       activation="gelu", batch_first=True,
                                       norm_first=True)
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(dim),
                                 enable_nested_tensor=False)


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input = nn.Linear(2, config.dim)
        self.transformer = _transformer(config.dim, config.heads, config.encoder_layers)

    def forward(self, x, position, visible):
        # Zero before projection: a NaN or very large hidden coordinate cannot leak.
        safe_x = torch.where(visible[..., None], x, torch.zeros_like(x))
        h = self.input(safe_x.flatten(1, 2)) + position
        h = self.transformer(h, src_key_padding_mask=~visible.flatten(1))
        return h.masked_fill(~visible.flatten(1)[..., None], 0.0)


class SkeletonJEPA(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config
        self.online = Encoder(c)
        self.teacher = deepcopy(self.online).requires_grad_(False)
        self.joint_embedding = nn.Parameter(torch.randn(len(JOINT_IDS), c.dim) * 0.02)
        # Teacher uses its own EMA joint embeddings as part of its encoder.
        self.register_buffer("teacher_joint_embedding", self.joint_embedding.detach().clone())
        self.mask_token = nn.Parameter(torch.zeros(1, 1, c.dim))
        self.predictor = _transformer(c.dim, c.heads, c.predictor_layers)
        self.latent_head = nn.Linear(c.dim, c.dim)
        self.coordinate_head = nn.Linear(c.dim, 2)
        self.register_buffer("center", torch.zeros(c.dim))
        if c.latent_dim:
            self.prior = nn.Linear(c.dim, 2 * c.latent_dim)
            self.posterior = nn.Linear(2 * c.dim, 2 * c.latent_dim)
            self.z_projection = nn.Linear(c.latent_dim, c.dim)
        self.run_metadata = {}

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def _validate(self, x, valid, target=None):
        if x.ndim != 4 or x.shape[1:] != (self.config.frames, len(JOINT_IDS), 2):
            raise ValueError("x must have shape [B,config.frames,16,2]")
        if valid.shape != x.shape[:-1] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean with shape [B,T,16]")
        if not torch.isfinite(x[valid]).all():
            raise ValueError("Observed coordinates must be finite")
        if not valid.flatten(1).any(1).all():
            raise ValueError("Each sequence needs at least one observed token")
        if target is not None:
            if target.shape != valid.shape or target.dtype != torch.bool or (target & ~valid).any():
                raise ValueError("Targets must be boolean observed tokens")
            if not target.flatten(1).any(1).all() or not (valid & ~target).flatten(1).any(1).all():
                raise ValueError("Every sequence needs both targets and context")

    def positions(self, batch, device, dtype):
        c = self.config
        time = torch.arange(c.frames, device=device, dtype=dtype).expand(batch, -1)
        if self.training and c.position_jitter:
            # Proposed shared temporal jitter, not a reproduction of learned StoP.
            time = time + torch.randn_like(time) * c.position_jitter
        frequencies = torch.exp(torch.arange(0, c.dim, 2, device=device, dtype=dtype)
                                * (-math.log(10000.0) / c.dim))
        angles = time[..., None] * frequencies
        temporal = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(-2)
        temporal = temporal[:, :, None, :].expand(-1, -1, len(JOINT_IDS), -1).flatten(1, 2)
        online = temporal + self.joint_embedding.repeat(c.frames, 1)
        teacher = temporal + self.teacher_joint_embedding.repeat(c.frames, 1)
        return online, teacher

    @staticmethod
    def _pool(h, valid):
        weights = valid.flatten(1).to(h.dtype)[..., None]
        return (h * weights).sum(1) / weights.sum(1).clamp_min(1)

    def forward(self, x, valid, target, *, posterior=False, sample_latent=False):
        """Predict masked targets. Posterior conditioning is training-only.

        Inference predictions use visible x only. Teacher outputs are included for
        scoring the observed outcome, never as inputs to the deployment predictor.
        """
        self._validate(x, valid, target)
        if posterior and not self.training:
            raise ValueError("Target-conditioned posterior is forbidden at inference")
        visible = valid & ~target
        position, teacher_position = self.positions(len(x), x.device, x.dtype)
        context = self.online(x, position, visible)
        with torch.no_grad():
            teacher = self.teacher(x, teacher_position, valid)
        pooled = self._pool(context, visible)
        inp = torch.where(visible.flatten(1)[..., None], context,
                          self.mask_token.expand(len(x), context.shape[1], -1)) + position
        kl = x.new_zeros(())
        if self.config.latent_dim:
            pm, pl = self.prior(pooled).chunk(2, -1)
            pl = pl.clamp(-8, 6)
            mean, logvar = pm, pl
            if posterior:
                outcome = self._pool(teacher, target)
                qm, ql = self.posterior(torch.cat([pooled, outcome], -1)).chunk(2, -1)
                ql = ql.clamp(-8, 6)
                mean, logvar = qm, ql
                kl = 0.5 * (pl - ql + (ql.exp() + (qm - pm).square()) / pl.exp() - 1).mean()
            z = mean + (0.5 * logvar).exp() * torch.randn_like(mean) if sample_latent else mean
            inp = inp + self.z_projection(z)[:, None]
        # All unobserved positions remain mask queries. Future visibility is
        # not an input to the predictor, avoiding acquisition-pattern leakage.
        predicted = self.predictor(inp)
        shape = (*valid.shape, -1)
        return {"latent": self.latent_head(predicted).reshape(shape),
                "coordinates": self.coordinate_head(predicted).reshape(*valid.shape, 2),
                "teacher": teacher.reshape(shape), "representation": pooled,
                "online_tokens": context.reshape(shape), "kl": kl}

    @torch.no_grad()
    def update_teacher(self):
        c = self.config
        for teacher, online in zip(self.teacher.parameters(), self.online.parameters(), strict=True):
            teacher.lerp_(online, 1 - c.ema)
        self.teacher_joint_embedding.lerp_(self.joint_embedding, 1 - c.ema)

    @torch.no_grad()
    def encode(self, x, valid, encoder="teacher"):
        self._validate(x, valid)
        old = self.training
        self.eval()
        p, tp = self.positions(len(x), x.device, x.dtype)
        if encoder not in {"teacher", "online"}:
            raise ValueError("encoder must be teacher or online")
        h = self.teacher(x, tp, valid) if encoder == "teacher" else self.online(x, p, valid)
        result = self._pool(h, valid)
        self.train(old)
        return result


def variance_covariance(z):
    """VICReg's variance and covariance terms. No class labels are involved."""
    if z.shape[0] < 2:
        raise ValueError("VICReg needs at least two independent examples in a batch")
    centered = z - z.mean(0)
    std = (centered.var(0, unbiased=False) + 1e-4).sqrt()
    variance = F.relu(1 - std).mean()
    cov = centered.T @ centered / (len(z) - 1)
    offdiag = cov - torch.diag_embed(cov.diagonal())
    covariance = offdiag.square().sum() / z.shape[1]
    return variance, covariance


def loss_terms(model, output, x, valid, target, *, variance_weight=1.0,
               covariance_weight=0.04, coordinate_weight=0.1,
               latent_mse_weight=0.0, kl_weight=0.01):
    c = model.config
    teacher_logits = (output["teacher"].detach()[target] - model.center) / c.target_temperature
    probabilities = teacher_logits.softmax(-1)
    prediction_logprob = (output["latent"][target] / c.prediction_temperature).log_softmax(-1)
    # Equal sequence weighting prevents high-visibility windows dominating loss.
    token_ce = -(probabilities * prediction_logprob).sum(-1)
    token_coordinate = (output["coordinates"][target] - x[target]).square().mean(-1)
    counts = target.flatten(1).sum(1).tolist()
    ce = torch.stack([v.mean() for v in token_ce.split(counts)]).mean()
    coordinates = torch.stack([v.mean() for v in token_coordinate.split(counts)]).mean()
    variance, covariance = variance_covariance(output["representation"])
    latent_mse = F.mse_loss(output["latent"][target], output["teacher"].detach()[target])
    total = ce + variance_weight * variance + covariance_weight * covariance + coordinate_weight * coordinates
    total = total + latent_mse_weight * latent_mse
    if c.latent_dim:
        total = total + kl_weight * output["kl"]
    if model.training:
        with torch.no_grad():
            model.center.lerp_(output["teacher"].detach()[valid].mean(0), 1 - c.center_ema)
    return {"loss": total, "latent_ce": ce, "coordinate_mse": coordinates,
            "variance": variance, "covariance": covariance,
            "latent_mse": latent_mse, "kl": output["kl"]}
