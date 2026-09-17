"""Small skeleton JEPA with explicit past-only inference and teacher spaces.

Default 96-wide four-layer encoder, two-layer predictor, four heads and patch4
are an S-JEPA-inspired adaptation. They are not a published-model reproduction.
Public inference methods accept prepared context tensors, never a dataset or
future labels. Every future teacher interval is independently encoded.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from .masking import patch_validity


CONTEXT_KEYS = ("context", "context_valid", "context_times", "context_age", "query_times")


def _position(length: int, width: int, device, dtype):
    index = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2, device=device).float()
                          * (-math.log(10000.) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2] = (index * frequency).sin()
    result[:, 1::2] = (index * frequency[:width // 2]).cos()
    return result.to(dtype)


def _context_only(context: dict) -> None:
    unknown = set(context) - set(CONTEXT_KEYS)
    if unknown or "context" not in context or "context_valid" not in context:
        raise ValueError(f"Context-only API rejects target/unknown keys: {sorted(unknown)}")


def _past_boundary(context: dict) -> None:
    _context_only(context)
    if set(CONTEXT_KEYS) - set(context):
        raise ValueError("Public inference requires the full context-only clock/validity contract")
    observed = context["context_valid"].bool()
    if (observed & (context["context_times"] >= 0)).any() or (context["query_times"] >= 0).any():
        raise ValueError("Past-only context contains a timestamp at/after the issue boundary")


class SkeletonEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.width = int(cfg.hidden_dim)
        self.patch = int(cfg.patch_size)
        self.clocks = bool(cfg.clock_channels)
        self.projection = nn.Linear(self.patch * (3 + 3 * self.clocks), self.width)
        self.joints = nn.Embedding(33, self.width)
        self.sentinel = nn.Parameter(torch.zeros(1, 1, self.width))
        layer = nn.TransformerEncoderLayer(self.width, int(cfg.heads), 4 * self.width,
                                           dropout=0., activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, int(cfg.encoder_depth),
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.width)

    def forward(self, context: dict, hidden: torch.Tensor | None = None):
        _context_only(context)
        x, observed = context["context"], context["context_valid"].bool()
        if x.ndim != 4 or x.shape[-2:] != (33, 2) or observed.shape != x.shape[:-1]:
            raise ValueError("Expected context [B,T,33,2] and matching observation flags")
        if (observed & ~torch.isfinite(x).all(-1)).any():
            raise ValueError("Observed coordinates must be finite")
        b, t, j, _ = x.shape
        valid = patch_validity(observed, self.patch)
        if hidden is None:
            hidden = torch.zeros_like(valid)
        if hidden.shape != valid.shape or (hidden & ~valid).any():
            raise ValueError("Hidden mask must select valid prepared-feature tokens")
        frame_hidden = hidden.repeat_interleave(self.patch, 1)[:, :t]
        visible_frames = observed & ~frame_hidden
        # Clear hidden coordinates AND validity/time channels BEFORE projection.
        safe = torch.where(visible_frames[..., None], x, torch.zeros_like(x))
        channels = [safe, visible_frames[..., None].to(x.dtype)]
        if self.clocks:
            for key in ("context_times", "context_age"):
                if key not in context or context[key].shape != observed.shape:
                    raise ValueError(f"Clock mode requires {key} with observation shape")
                if (visible_frames & ~torch.isfinite(context[key])).any():
                    raise ValueError(f"Nonfinite observed clock {key}")
                channels.append(torch.where(visible_frames, context[key], 0)[..., None])
            queries = context["query_times"]
            if queries.shape != (b, t) or not torch.isfinite(queries).all():
                raise ValueError("Fixed query clocks must have finite [B,T] shape")
            channels.append(torch.where(visible_frames, queries[:, :, None], 0)[..., None])
        prepared = torch.cat(channels, -1)
        prepared = F.pad(prepared, (0, 0, 0, 0, 0, (-t) % self.patch))
        segments = prepared.shape[1] // self.patch
        patches = prepared.reshape(b, segments, self.patch, j, -1).permute(0, 1, 3, 2, 4)
        patches = patches.reshape(b, segments * j, -1)
        h = self.projection(patches)
        position = _position(segments, self.width, x.device, h.dtype)[:, None]
        position = position + self.joints.weight[None]
        h = h + position.reshape(1, segments * j, self.width)
        visible = (valid & ~hidden).reshape(b, -1)
        # The learned nondata sentinel makes all-invalid batches numerically safe.
        h = torch.cat([self.sentinel.expand(b, -1, -1), h], 1)
        attention_valid = torch.cat([torch.ones(b, 1, dtype=torch.bool, device=x.device), visible], 1)
        tokens = self.norm(self.blocks(h, src_key_padding_mask=~attention_valid))[:, 1:]
        tokens = torch.where(visible[..., None], tokens, 0)
        return tokens, valid.reshape(b, -1), visible

    @staticmethod
    def pool(tokens, valid):
        return (tokens * valid[..., None]).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)


class Predictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = int(cfg.hidden_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
        self.horizon = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, int(cfg.heads), 4 * width,
                                           dropout=0., activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, int(cfg.predictor_depth),
                                             enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, width)

    def masked(self, tokens, valid, hidden):
        b, n, d = tokens.shape
        hidden = hidden.reshape(b, n)
        values = torch.where(hidden[..., None], self.mask_token.expand(b, n, d), tokens)
        values = values + _position(n, d, values.device, values.dtype)[None]
        sentinel = self.mask_token.expand(b, 1, d)
        values = torch.cat([sentinel, values], 1)
        usable = torch.cat([torch.ones(b, 1, device=valid.device, dtype=torch.bool), valid], 1)
        return self.output(self.norm(self.blocks(values, src_key_padding_mask=~usable)))[:, 1:]

    def future(self, tokens, valid, horizons):
        b, n, d = tokens.shape
        queries = self.horizon(horizons[:, None]).to(tokens.dtype)[None].expand(b, -1, -1)
        values = torch.cat([self.mask_token.expand(b, 1, d), tokens, queries], 1)
        usable = torch.cat([torch.ones(b, 1, device=valid.device, dtype=torch.bool), valid,
                            torch.ones(b, len(horizons), device=valid.device, dtype=torch.bool)], 1)
        return self.output(self.norm(self.blocks(values, src_key_padding_mask=~usable)))[:, n + 1:]


class JEPA(nn.Module):
    """Pooled encoder and fixed-horizon predictor APIs.

    `encode` is the validity-weighted mean of final LayerNorm token features.
    `predict_future(...,'teacher')` uses the EMA encoder and online predictor;
    it is not a separate EMA-predictor model. Initialized state includes both
    untouched encoder and predictor. No prediction API accepts target masks.
    """
    def __init__(self, cfg):
        super().__init__()
        self.online = SkeletonEncoder(cfg)
        self.teacher = copy.deepcopy(self.online)
        self.initialized = copy.deepcopy(self.online)
        self.predictor = Predictor(cfg)
        self.initialized_predictor = copy.deepcopy(self.predictor)
        self.projector = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.GELU(),
                                       nn.Linear(cfg.hidden_dim, cfg.hidden_dim))
        self.register_buffer("center_masked", torch.zeros(cfg.hidden_dim))
        self.register_buffer("center_future", torch.zeros(cfg.hidden_dim))
        self.register_buffer("horizons", torch.tensor(cfg.horizons, dtype=torch.float32))
        self.grid_hz = float(cfg.grid_hz)
        for module in (self.teacher, self.initialized, self.initialized_predictor):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
            module.eval()

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.initialized.eval()
        self.initialized_predictor.eval()
        return self

    def _encoder(self, state):
        if state not in {"online", "teacher", "initialized"}:
            raise ValueError(f"Unknown encoder state {state!r}")
        return getattr(self, state)

    def encode(self, context: dict, state="online"):
        _past_boundary(context)
        tokens, valid, _ = self._encoder(state)(context)
        return SkeletonEncoder.pool(tokens, valid)

    def predict_future(self, context: dict, state="online"):
        _past_boundary(context)
        tokens, valid, _ = self._encoder(state)(context)
        predictor = self.initialized_predictor if state == "initialized" else self.predictor
        return predictor.future(tokens, valid, self.horizons)

    def encode_future_targets(self, future, future_valid, state="teacher"):
        """Privileged teacher/diagnostic only; each [2,33,2] interval is separate.

        Structural padding to patch4 does not stretch the 80ms interval. The
        caller prepared targets in the PREFIX reference frame; no future-fitted
        normalization is performed here. Do not call this during forecasting.
        """
        if future.ndim != 5 or future.shape[1:4] != (len(self.horizons), 2, 33):
            raise ValueError("Expected independently sampled [B,H,2,33,2] future intervals")
        if future_valid.shape != future.shape[:-1]:
            raise ValueError("Future validity shape mismatch")
        b, h, t, j, c = future.shape
        times = torch.stack([self.horizons - 1 / self.grid_hz, self.horizons], 1)
        queries = times[None].expand(b, -1, -1).reshape(b * h, t)
        target_context = {"context": future.reshape(b * h, t, j, c),
                          "context_valid": future_valid.reshape(b * h, t, j),
                          "query_times": queries,
                          "context_times": queries[:, :, None].expand(-1, -1, j),
                          "context_age": torch.zeros(b * h, t, j, device=future.device)}
        tokens, valid, _ = self._encoder(state)(target_context)
        return SkeletonEncoder.pool(tokens, valid).reshape(b, h, -1)

    def masked_prediction(self, context, hidden):
        _past_boundary(context)
        tokens, valid, _ = self.online(context, hidden)
        predicted = self.predictor.masked(tokens, valid, hidden)
        with torch.no_grad():
            targets, _, _ = self.teacher(context)
        return predicted, targets, hidden.reshape(len(tokens), -1)

    @torch.no_grad()
    def update_teacher(self, momentum):
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be in [0,1]")
        for teacher, online in zip(self.teacher.parameters(), self.online.parameters()):
            teacher.mul_(momentum).add_(online, alpha=1 - momentum)

    @torch.no_grad()
    def update_center(self, target, valid, name, momentum=.9):
        center = getattr(self, "center_" + name)
        counts = valid.reshape(len(target), -1).sum(1)
        means = (target * valid[..., None]).reshape(len(target), -1, target.shape[-1]).sum(1)
        supported = counts > 0
        if supported.any():
            value = (means / counts[:, None].clamp_min(1))[supported].mean(0)
            center.mul_(momentum).add_(value, alpha=1 - momentum)
