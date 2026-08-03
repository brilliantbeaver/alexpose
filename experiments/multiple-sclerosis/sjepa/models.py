"""The S-JEPA networks: a transformer encoder, a predictor, and an EMA wrapper.

We keep the transformer plain (pre-norm multi-head self-attention plus an MLP),
exactly the "vanilla transformer" the paper uses, just smaller. Three of these
pieces make the full model:

* ``SJEPAEncoder`` = tokenizer + transformer. Used for both the view encoder and
  the target encoder. They share the architecture but are updated differently.
* ``Predictor`` = a shallow transformer that takes the visible context features,
  inserts learnable mask tokens at the hidden positions, and predicts the target
  latents there.
* ``EMA`` = a helper that keeps the target encoder as a slow moving, stop-gradient
  copy of the view encoder. Removing this is what makes S-JEPA collapse, so it is
  the load bearing anti-collapse mechanism.

``SJEPA`` ties them together and exposes the two-lane forward pass used in
training and the simple ``embed`` call used for downstream classification.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .config import SJEPAConfig
from .tokenizer import SkeletonTokenizer


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``key_padding_mask`` is (B, N) with True where a token must be IGNORED."""
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False,
                                key_padding_mask=key_padding_mask)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class SJEPAEncoder(nn.Module):
    """Tokenizer plus transformer. Outputs per-token features (B, N, dim)."""

    def __init__(self, cfg: SJEPAConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = SkeletonTokenizer(
            num_joints=cfg.num_joints,
            in_channels=cfg.in_channels,
            frame_group=cfg.frame_group,
            num_time_tokens=cfg.num_time_tokens,
            dim=cfg.encoder_dim,
        )
        self.transformer = Transformer(
            dim=cfg.encoder_dim, depth=cfg.encoder_depth, heads=cfg.encoder_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(self.tokenizer(x))

    def forward_context(self, x: torch.Tensor, context_mask: torch.Tensor,
                        mask_token: torch.Tensor) -> torch.Tensor:
        """Encode only the visible context, then place mask tokens at hidden slots.

        ``context_mask`` is a (N,) boolean tensor, True for visible tokens. We run
        the transformer over just those tokens (the view encoder never sees the
        masked joints), then scatter the encoded visible tokens back and fill the
        masked positions with the shared learnable ``mask_token``. The result is a
        full length (B, N, dim) tensor for the predictor.
        """
        tokens = self.tokenizer(x)                      # (B, N, dim)
        B, N, D = tokens.shape
        ctx_idx = torch.nonzero(context_mask, as_tuple=False).squeeze(1)
        visible = tokens[:, ctx_idx, :]                 # (B, Nc, dim)
        encoded_visible = self.transformer(visible)     # (B, Nc, dim)

        full = mask_token.expand(B, N, D).clone()       # (B, N, dim)
        full[:, ctx_idx, :] = encoded_visible
        return full

    def forward_context_per_example(
        self, x: torch.Tensor, context_mask: torch.Tensor,
        mask_token: torch.Tensor,
    ) -> torch.Tensor:
        """Repaired context encoding with PER-EXAMPLE masks (D2/D7 fix).

        ``context_mask`` is (B, N) boolean, True where a token is VISIBLE. Because
        each row can keep a different set of visible tokens, we cannot gather a
        common variable-length subset. Instead we tokenize the full sequence and
        run the transformer with a key-padding mask so each row only *attends to*
        its own visible tokens; the hidden positions are then overwritten with the
        shared ``mask_token`` before the predictor sees them. This keeps every
        joint able to be context on some steps, so no joint-position row is
        starved of gradient.
        """
        tokens = self.tokenizer(x)                       # (B, N, dim)
        B, N, D = tokens.shape
        key_padding = ~context_mask                      # True where IGNORED (hidden)
        encoded = self.transformer(tokens, key_padding_mask=key_padding)
        hidden = key_padding.unsqueeze(-1)               # (B, N, 1)
        full = torch.where(hidden, mask_token.expand(B, N, D), encoded)
        return full


class Predictor(nn.Module):
    """Maps view features (B, N, enc_dim) to predicted target latents (B, N, enc_dim).

    Legacy predictor kept for the E0 reference path. It has NO target-position
    identity (the D1 defect): all hidden slots receive the same mask token and no
    positional information, so their predictions are identical. Use
    :class:`PredictorV2` for the repaired model.
    """

    def __init__(self, cfg: SJEPAConfig):
        super().__init__()
        self.in_proj = nn.Linear(cfg.encoder_dim, cfg.predictor_dim)
        self.transformer = Transformer(
            dim=cfg.predictor_dim, depth=cfg.predictor_depth, heads=cfg.encoder_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        )
        self.out_proj = nn.Linear(cfg.predictor_dim, cfg.encoder_dim)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(view_features)
        h = self.transformer(h)
        return self.out_proj(h)


class PredictorV2(nn.Module):
    """Repaired predictor with factorized joint/time position identity (D1 fix).

    This is the I-JEPA pattern: every predictor token is the projected context
    feature (or the shared mask token at hidden slots) PLUS a learned positional
    embedding that says exactly which joint and which time block this slot is.
    With the tag present, a hidden left-ankle-at-t3 slot and a hidden
    right-shoulder-at-t0 slot are no longer identical inputs, so the predictor can
    produce different latents for them. Without it, they are provably identical.

    Position identity is factorized into a per-joint (spatial) and a per-time
    (temporal) embedding added in predictor space, matching the tokenizer's
    ``token index = t * V + v`` layout.
    """

    def __init__(self, cfg: SJEPAConfig):
        super().__init__()
        self.num_joints = cfg.num_joints
        self.num_time_tokens = cfg.num_time_tokens
        self.in_proj = nn.Linear(cfg.encoder_dim, cfg.predictor_dim)
        self.spatial_pos = nn.Parameter(torch.zeros(1, 1, cfg.num_joints, cfg.predictor_dim))
        self.temporal_pos = nn.Parameter(torch.zeros(1, cfg.num_time_tokens, 1, cfg.predictor_dim))
        nn.init.trunc_normal_(self.spatial_pos, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        self.transformer = Transformer(
            dim=cfg.predictor_dim, depth=cfg.predictor_depth, heads=cfg.encoder_heads,
            mlp_ratio=cfg.mlp_ratio, dropout=cfg.dropout,
        )
        self.out_proj = nn.Linear(cfg.predictor_dim, cfg.encoder_dim)

    def _add_positions(self, h: torch.Tensor) -> torch.Tensor:
        B, N, D = h.shape
        Tt, V = self.num_time_tokens, self.num_joints
        h = h.view(B, Tt, V, D) + self.spatial_pos + self.temporal_pos
        return h.reshape(B, N, D)

    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(view_features)
        h = self._add_positions(h)               # <-- the repair: joint/time identity
        h = self.transformer(h)
        return self.out_proj(h)


class EMA:
    """Keeps a stop-gradient exponential moving average copy of a module."""

    def __init__(self, source: nn.Module):
        self.target = copy.deepcopy(source)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.target.eval()

    @torch.no_grad()
    def update(self, source: nn.Module, tau: float) -> None:
        for tp, sp in zip(self.target.parameters(), source.parameters()):
            tp.mul_(tau).add_(sp.detach(), alpha=1.0 - tau)
        for tb, sb in zip(self.target.buffers(), source.buffers()):
            tb.copy_(sb)

    def to(self, device):
        self.target.to(device)
        return self


def ema_tau_schedule(step: int, total_steps: int, start: float, end: float) -> float:
    """Cosine schedule for the EMA momentum, rising from ``start`` toward ``end``."""
    if total_steps <= 1:
        return end
    progress = min(1.0, step / float(total_steps))
    return end - (end - start) * (math.cos(math.pi * progress) + 1.0) / 2.0


class SJEPA(nn.Module):
    """The full model: view encoder + EMA target encoder + predictor.

    The target encoder lives outside ``nn.Module``'s parameter tree (it is an EMA
    copy with no gradients), so only the view encoder and predictor are optimised.
    """

    def __init__(self, cfg: SJEPAConfig, repaired: bool = False):
        super().__init__()
        self.cfg = cfg
        self.repaired = repaired
        self.view_encoder = SJEPAEncoder(cfg)
        self.predictor = PredictorV2(cfg) if repaired else Predictor(cfg)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.encoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.ema = EMA(self.view_encoder)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Keep the EMA target on the same device.
        device = next(self.parameters()).device
        self.ema.to(device)
        return self

    @property
    def target_encoder(self) -> SJEPAEncoder:
        return self.ema.target

    def update_target(self, tau: float) -> None:
        self.ema.update(self.view_encoder, tau)

    def forward(self, x_view: torch.Tensor, x_full: torch.Tensor,
                context_mask: torch.Tensor):
        """Two-lane forward pass.

        * View lane: encode the visible context of a transformed view, insert mask
          tokens, run the predictor. Returns predicted latents at all tokens.
        * Target lane: encode the complete (untransformed) skeleton with the EMA
          encoder, no gradient. Masking is applied to its *output* by the caller.

        Returns ``(predicted (B, N, dim), target (B, N, dim))``. The caller slices
        both to the masked positions before the loss.
        """
        view_features = self.view_encoder.forward_context(x_view, context_mask, self.mask_token)
        predicted = self.predictor(view_features)
        with torch.no_grad():
            target = self.target_encoder(x_full)
        return predicted, target

    def forward_repaired(self, x_view: torch.Tensor, x_full: torch.Tensor,
                         context_mask: torch.Tensor):
        """Repaired two-lane forward with PER-EXAMPLE (B, N) context masks.

        Uses :meth:`SJEPAEncoder.forward_context_per_example` (key-padding
        attention so every joint can be context) and :class:`PredictorV2` (target
        joint/time position identity). The target lane still masks the target
        encoder's OUTPUT by the caller, which the paper shows is essential.
        Returns ``(predicted (B, N, dim), target (B, N, dim))``.
        """
        view_features = self.view_encoder.forward_context_per_example(
            x_view, context_mask, self.mask_token)
        predicted = self.predictor(view_features)
        with torch.no_grad():
            target = self.target_encoder(x_full)
        return predicted, target

    @torch.no_grad()
    def embed(self, x_full: torch.Tensor, target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Downstream representation: masked mean-pool of target-encoder tokens.

        By default we pool over the masked (neurologically relevant) tokens, since
        those are the joints the model was trained to reason about. Pass a
        different mask to pool over other tokens.
        """
        self.target_encoder.eval()
        feats = self.target_encoder(x_full)             # (B, N, dim)
        if target_mask is None:
            return feats.mean(dim=1)
        idx = torch.nonzero(target_mask, as_tuple=False).squeeze(1)
        return feats[:, idx, :].mean(dim=1)


def build_model(cfg: SJEPAConfig, device: Optional[str] = None,
                repaired: bool = False) -> SJEPA:
    """Construct the model, seed it, and move it to the chosen device.

    ``repaired=True`` builds the corrected model (PredictorV2 with target-position
    identity + per-example mask support). ``repaired=False`` keeps the E0 model.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = SJEPA(cfg, repaired=repaired)
    if device is None:
        device = pick_device()
    return model.to(device)


def pick_device() -> str:
    """Prefer CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
