"""Offline body-12 restoration; model inputs never include reference support.

This is a local temporal-restoration implementation inspired by the existing
temporal-gait encoder. Its past-only forecasting API is deliberately unchanged.
Every patch/joint has a query, including joints the extractor did not observe.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

INPUT_KEYS = frozenset({"xy", "confidence", "observed", "timestamps"})
ARMS = ("smoothnet", "direct", "static", "initialized", "coordinate",
        "ordinary_jepa", "paired_jepa", "shuffled_jepa")
LATENT_ARMS = frozenset({"ordinary_jepa", "paired_jepa", "shuffled_jepa"})
JOINTS = 12


@dataclass(frozen=True)
class ModelConfig:
    width: int = 96
    encoder_layers: int = 4
    predictor_layers: int = 2
    heads: int = 4
    patch_size: int = 4
    window_size: int = 64

    def validate(self):
        if min(self.width, self.encoder_layers, self.predictor_layers, self.heads,
               self.patch_size, self.window_size) < 1:
            raise ValueError("All model dimensions must be positive")
        if self.width % self.heads or self.window_size % self.patch_size:
            raise ValueError("Width must divide heads and window must divide patch size")
        return self


def model_config(value):
    return (value if isinstance(value, ModelConfig) else ModelConfig(**value)).validate()


def validate_inputs(inputs, window_size=None):
    if set(inputs) != INPUT_KEYS:
        raise ValueError(f"Inference accepts only {sorted(INPUT_KEYS)}")
    xy, observed = inputs["xy"], inputs["observed"]
    if xy.ndim != 4 or xy.shape[-2:] != (JOINTS, 2) or len(xy) < 1:
        raise ValueError("Expected nonempty xy [batch,time,12,2]")
    if window_size is not None and xy.shape[1] != window_size:
        raise ValueError("Input window differs from fixed model query grid")
    if observed.dtype != torch.bool or observed.shape != xy.shape[:-1]:
        raise ValueError("Observed must be boolean [batch,time,12]")
    confidence, times = inputs["confidence"], inputs["timestamps"]
    if confidence.shape != observed.shape or times.shape != xy.shape[:2]:
        raise ValueError("Confidence/timestamp shape mismatch")
    if not torch.isfinite(confidence).all():
        raise ValueError("Native estimator scores must be finite after the missing-score adapter")
    if (observed & (confidence <= 0)).any():
        raise ValueError("Observed joints require positive native estimator scores")
    if not torch.isfinite(times).all() or (times.diff(dim=1) <= 0).any():
        raise ValueError("Timestamps must be finite and strictly increasing physical seconds")
    if (observed & ~torch.isfinite(xy).all(-1)).any():
        raise ValueError("Observed coordinates must be finite")


def frame_hidden(hidden, cfg, batch, device):
    shape = (batch, cfg.window_size // cfg.patch_size, JOINTS)
    if hidden is None:
        return torch.zeros((batch, cfg.window_size, JOINTS), device=device, dtype=torch.bool)
    if hidden.dtype != torch.bool or hidden.shape != shape:
        raise ValueError("Artificial mask must be bool [batch,patch,12], independent of target support")
    return hidden.repeat_interleave(cfg.patch_size, dim=1)


def _blocks(cfg, layers):
    layer = nn.TransformerEncoderLayer(cfg.width, cfg.heads, cfg.width * 4,
        dropout=0., activation="gelu", batch_first=True, norm_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)


def _positions(length, width, device, dtype):
    # Fixed query positions supplement, rather than replace, physical timestamps.
    pos = torch.arange(length, device=device).float()[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device).float()
                          * (-math.log(10000.) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = (pos * frequency).sin()
    result[:, 1::2] = (pos * frequency[:width // 2]).cos()
    return result.to(dtype)


class Body12Encoder(nn.Module):
    """Patch tokens retain a fixed grid, even when all input joints are absent."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = model_config(cfg)
        self.projection = nn.Linear(self.cfg.patch_size * 5, self.cfg.width)
        self.joints = nn.Embedding(JOINTS, self.cfg.width)
        self.missing_query = nn.Parameter(torch.zeros(1, 1, self.cfg.width))
        self.blocks = _blocks(self.cfg, self.cfg.encoder_layers)
        self.norm = nn.LayerNorm(self.cfg.width)

    def forward(self, inputs, hidden=None):
        validate_inputs(inputs, self.cfg.window_size)
        xy, observed = inputs["xy"], inputs["observed"]
        b, t, j, _ = xy.shape
        artificial = frame_hidden(hidden, self.cfg, b, xy.device)
        usable = observed & ~artificial
        safe = torch.where(usable[..., None], xy, 0)
        confidence = torch.where(~artificial, inputs["confidence"], 0)
        seconds = inputs["timestamps"] - inputs["timestamps"][:, :1]
        channels = torch.cat((safe, confidence[..., None], usable[..., None].to(xy.dtype),
                              seconds[:, :, None, None].expand(-1, -1, j, -1)), dim=-1)
        p, s = self.cfg.patch_size, t // self.cfg.patch_size
        patches = channels.reshape(b, s, p, j, 5).permute(0, 1, 3, 2, 4).reshape(b, s * j, p * 5)
        tokens = self.projection(patches)
        positions = _positions(s, self.cfg.width, xy.device, tokens.dtype)[:, None]
        positions = positions + self.joints.weight[None]
        support = usable.reshape(b, s, p, j).any(2).reshape(b, s * j)
        tokens = tokens + positions.reshape(1, s * j, -1)
        tokens = tokens + (~support)[..., None] * self.missing_query
        # No target-derived key padding. Missing positions are valid OUTPUT queries.
        return self.norm(self.blocks(tokens))


class CoordinateReadout(nn.Module):
    """Identical nonlinear patch readout, fitted separately for each representation."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width),
                                     nn.GELU(), nn.Linear(cfg.width, cfg.patch_size * 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, tokens):
        b = len(tokens)
        return self.network(tokens).reshape(b, self.cfg.window_size // self.cfg.patch_size,
            JOINTS, self.cfg.patch_size, 2).permute(0, 1, 3, 2, 4).reshape(b, self.cfg.window_size, JOINTS, 2)


class SmoothNetStyle(nn.Module):
    """Source-trained temporal MLP, a 2D body-12 adaptation, not author weights.

    The same MLP is applied to each joint; full time histories contain coordinates,
    confidence, support and seconds once. Output is a residual/absolute correction.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.network = nn.Sequential(nn.Linear(cfg.window_size * 5, cfg.width * 2), nn.GELU(),
            nn.Linear(cfg.width * 2, cfg.width * 2), nn.GELU(),
            nn.Linear(cfg.width * 2, cfg.window_size * 2))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs):
        validate_inputs(inputs, self.cfg.window_size)
        xy, observed = inputs["xy"], inputs["observed"]
        safe = torch.where(observed[..., None], xy, 0)
        seconds = inputs["timestamps"] - inputs["timestamps"][:, :1]
        features = torch.cat((safe, inputs["confidence"][..., None], observed[..., None].to(xy.dtype),
                              seconds[:, :, None, None].expand(-1, -1, JOINTS, -1)), -1)
        residual = self.network(features.permute(0, 2, 1, 3).flatten(2))
        residual = residual.reshape(len(xy), JOINTS, self.cfg.window_size, 2).transpose(1, 2)
        return residual + safe


class StaticCoordinates(nn.Module):
    """Only current-frame coordinates; the SAME full-window auxiliary information.

    This control is auxiliary-matched, not parameter-matched. Confidence, support,
    and physical time are each included once in the shared auxiliary vector.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.aux = nn.Linear(cfg.window_size * (2 * JOINTS + 1), cfg.width)
        self.query = nn.Embedding(cfg.window_size, cfg.width)
        self.network = nn.Sequential(nn.Linear(2 * JOINTS + cfg.width, cfg.width), nn.GELU(),
            nn.Linear(cfg.width, cfg.width), nn.GELU(), nn.Linear(cfg.width, 2 * JOINTS))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs):
        validate_inputs(inputs, self.cfg.window_size)
        xy, observed = inputs["xy"], inputs["observed"]
        safe = torch.where(observed[..., None], xy, 0)
        times = inputs["timestamps"] - inputs["timestamps"][:, :1]
        auxiliary = torch.cat((inputs["confidence"].flatten(1), observed.to(xy.dtype).flatten(1), times), 1)
        context = self.aux(auxiliary)[:, None] + self.query.weight[None]
        correction = self.network(torch.cat((safe.flatten(2), context), -1)).reshape_as(xy)
        return correction + safe


class RestorationModel(nn.Module):
    def __init__(self, arm, cfg):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}")
        self.arm, self.cfg = arm, model_config(cfg)
        if arm in {"smoothnet", "static"}:
            self.practical = (SmoothNetStyle if arm == "smoothnet" else StaticCoordinates)(self.cfg)
        else:
            self.encoder = Body12Encoder(self.cfg)
            self.readout = CoordinateReadout(self.cfg)
            self.initialized_encoder = deepcopy(self.encoder).requires_grad_(False)
            self.initialized_readout = deepcopy(self.readout).requires_grad_(False)
            if arm in LATENT_ARMS:
                self.teacher = deepcopy(self.encoder).requires_grad_(False)
                self.predictor = nn.Sequential(_blocks(self.cfg, self.cfg.predictor_layers),
                                               nn.LayerNorm(self.cfg.width), nn.Linear(self.cfg.width, self.cfg.width))
                self.projector = nn.Sequential(nn.Linear(self.cfg.width, self.cfg.width), nn.GELU(),
                                               nn.Linear(self.cfg.width, self.cfg.width))
                self.register_buffer("center", torch.zeros(self.cfg.width))
        self.training_report = {}

    def train(self, mode=True):
        super().train(mode)
        for name in ("initialized_encoder", "initialized_readout", "teacher"):
            if hasattr(self, name):
                getattr(self, name).eval()
        return self

    def forward(self, inputs, hidden=None):
        if hasattr(self, "practical"):
            if hidden is not None:
                raise ValueError("Practical denoisers do not accept pretraining masks")
            return self.practical(inputs)
        tokens = self.encoder(inputs, hidden)
        residual = self.readout(tokens)
        observed = inputs["observed"] & ~frame_hidden(hidden, self.cfg, len(residual), residual.device)
        # A missing coordinate receives an absolute prediction, never NaN+residual.
        return residual + torch.where(observed[..., None], inputs["xy"], 0)

    @torch.no_grad()
    def update_teacher(self, momentum):
        if not 0 <= momentum < 1 or not hasattr(self, "teacher"):
            raise ValueError("A JEPA teacher and momentum in [0,1) are required")
        for target, online in zip(self.teacher.parameters(), self.encoder.parameters()):
            target.mul_(momentum).add_(online, alpha=1 - momentum)
        for target, online in zip(self.teacher.buffers(), self.encoder.buffers()):
            target.copy_(online)

    def configuration(self):
        return asdict(self.cfg)
