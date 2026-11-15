"""Small, inspectable skeleton JEPA for the ASTAD tutorials.

This is an adaptation of Skeleton-JEPA (ECCV 2024), not its benchmark
implementation. It uses its EMA target and centered, sharpened latent
cross-entropy, adds VICReg, and replaces motion-aware masking with uniform
sampling. Only the explicitly listed 16 BlazePose joints enter this network.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

JOINT_INDICES = (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32)
Mode = Literal["masked", "forecast"]


@contextmanager
def _evaluation(module: nn.Module):
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


@dataclass(frozen=True)
class ModelConfig:
    frames: int = 32
    patch_frames: int = 4
    joints: int = 16
    channels: int = 2
    dim: int = 64
    depth: int = 2
    predictor_depth: int = 1
    heads: int = 4
    mask_ratio: float = 0.5
    forecast_context: float = 0.5
    student_temperature: float = 0.1
    target_temperature: float = 0.06
    center_momentum: float = 0.9
    vicreg_weight: float = 0.1
    spe: bool = False
    spe_sigma: float = 0.25

    def __post_init__(self):
        if self.patch_frames < 1 or self.frames < 2 or self.frames % self.patch_frames:
            raise ValueError("frames must be a positive multiple of patch_frames")
        if self.joints != len(JOINT_INDICES) or self.channels != 2:
            raise ValueError("Use exactly the 16 whitelisted joints and XY channels")
        if self.heads < 1 or self.dim < 4 or self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if self.depth < 1 or self.predictor_depth < 1:
            raise ValueError("Both transformer depths must be positive")
        if not 0 < self.mask_ratio < 1:
            raise ValueError("mask_ratio must lie strictly between 0 and 1")
        if not 0.5 <= self.forecast_context < 1:
            raise ValueError("forecast_context must be at least half, matching prefix-only scale normalization")
        if min(self.student_temperature, self.target_temperature) <= 0:
            raise ValueError("Temperatures must be positive")
        if self.vicreg_weight < 0 or self.spe_sigma < 0:
            raise ValueError("Loss and noise strengths cannot be negative")
        if not 0 <= self.center_momentum < 1:
            raise ValueError("center_momentum must lie in [0,1)")

    @property
    def time_tokens(self) -> int:
        return self.frames // self.patch_frames

    @property
    def tokens(self) -> int:
        return self.time_tokens * self.joints


def uniform_mask(batch: int, tokens: int, ratio: float = 0.5, *,
                 generator: torch.Generator | None = None,
                 device: torch.device | str = "cpu") -> torch.Tensor:
    """True means hidden. Every token has exactly the same marginal probability.

    Sampling is without replacement and independently permuted per example.
    The function accepts no motion coordinates, intensities, or labels.
    Randomness is generated on CPU for identical masks across CPU/MPS/CUDA.
    """
    if tokens < 2 or batch < 1 or not 0 < ratio < 1:
        raise ValueError("Need a nonempty batch, at least two tokens, and 0 < ratio < 1")
    hidden = min(tokens - 1, max(1, round(tokens * ratio)))
    order = torch.rand(batch, tokens, generator=generator).argsort(dim=1)
    mask = torch.zeros(batch, tokens, dtype=torch.bool)
    mask.scatter_(1, order[:, :hidden], True)
    return mask.to(device)


def forecast_mask(batch: int, cfg: ModelConfig, *, device="cpu") -> torch.Tensor:
    """Hide every joint in a contiguous future suffix at a patch boundary."""
    if cfg.time_tokens < 2:
        raise ValueError("Forecasting needs at least two temporal patches")
    prefix = min(cfg.time_tokens - 1, max(1, math.ceil(cfg.time_tokens * cfg.forecast_context)))
    mask = torch.arange(cfg.tokens, device=device) >= prefix * cfg.joints
    return mask.expand(batch, -1).clone()


def _gather(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return tokens.gather(1, indices[..., None].expand(-1, -1, tokens.shape[-1]))


def _visible_indices(mask: torch.Tensor) -> torch.Tensor:
    counts = (~mask).sum(dim=1)
    if not torch.all(counts == counts[0]) or counts[0] < 1 or counts[0] == mask.shape[1]:
        raise ValueError("Every row must have the same nonzero number of visible and hidden tokens")
    return (~mask).nonzero(as_tuple=False)[:, 1].reshape(mask.shape[0], -1)


def token_validity(x: torch.Tensor, cfg: ModelConfig) -> torch.Tensor:
    """A joint patch is observed only if every XY coordinate in it is finite.

    Validity is a measurement-quality mask, separate from uniform learning
    masks. It never changes which positions the random sampler selects.
    """
    if x.ndim != 4 or tuple(x.shape[1:]) != (cfg.frames, cfg.joints, cfg.channels) or len(x) == 0:
        raise ValueError(f"Expected nonempty [B,{cfg.frames},16,2], got {tuple(x.shape)}")
    patches = x.reshape(len(x), cfg.time_tokens, cfg.patch_frames, cfg.joints, cfg.channels)
    return torch.isfinite(patches).all(dim=2).all(dim=-1).reshape(len(x), cfg.tokens)


def _valid_pool(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if (valid.sum(dim=1) == 0).any():
        raise ValueError("Each example needs at least one valid token to pool")
    return (tokens * valid[..., None]).sum(dim=1) / valid.sum(dim=1, keepdim=True)


class Encoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Linear(cfg.patch_frames * cfg.channels, cfg.dim)
        self.joint_position = nn.Parameter(torch.randn(1, 1, cfg.joints, cfg.dim) * 0.02)
        self.time_position = nn.Parameter(torch.randn(1, cfg.time_tokens, 1, cfg.dim) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 4,
                                           dropout=0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg.depth,
                                                  norm=nn.LayerNorm(cfg.dim),
                                                  enable_nested_tensor=False)

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        valid = token_validity(x, cfg)  # Validate shape before reshaping.
        # Zero replacement is a numerical placeholder, never an observation.
        # Invalid patches are excluded from attention keys and downstream pools.
        clean = torch.nan_to_num(x, nan=0., posinf=0., neginf=0.)
        patches = clean.reshape(len(x), cfg.time_tokens, cfg.patch_frames, cfg.joints, 2)
        patches = patches.permute(0, 1, 3, 2, 4).reshape(len(x), cfg.time_tokens, cfg.joints, -1)
        # Discard the entire unusable patch before projection. This also stops
        # extreme finite remnants of a missing patch from overflowing attention.
        patches = patches.masked_fill(~valid.reshape(len(x), cfg.time_tokens, cfg.joints, 1), 0.)
        tokens = self.projection(patches) + self.joint_position + self.time_position
        return tokens.reshape(len(x), cfg.tokens, cfg.dim)

    def forward(self, x: torch.Tensor, visible: torch.Tensor | None = None) -> torch.Tensor:
        valid = token_validity(x, self.cfg)
        tokens = self.tokenize(x)
        if visible is not None:
            # Hidden coordinates cannot participate in any attention operation.
            tokens = _gather(tokens, visible)
            valid = valid.gather(1, visible)
        if (valid.sum(dim=1) == 0).any():
            raise ValueError("Each example needs at least one valid context token")
        encoded = self.transformer(tokens, src_key_padding_mask=~valid)
        # Query positions can still produce outputs when their keys are padded.
        # Explicitly clear these outputs so accidental unweighted use is visible.
        return encoded.masked_fill(~valid[..., None], 0.)


def vicreg(a: torch.Tensor, b: torch.Tensor) -> dict[str, torch.Tensor]:
    """VICReg on unnormalized projections of two views of the same examples.

    This discourages constant dimensions and redundancy. It does not repel
    clinical classes, and no class labels enter this function.
    """
    if a.shape != b.shape or a.ndim != 2 or len(a) < 2:
        raise ValueError("VICReg needs two matching [B,D] views with B >= 2")
    invariance = F.mse_loss(a, b)
    std_a = (a.var(dim=0, unbiased=False) + 1e-4).sqrt()
    std_b = (b.var(dim=0, unbiased=False) + 1e-4).sqrt()
    variance = (F.relu(1 - std_a).mean() + F.relu(1 - std_b).mean()) / 2
    cov_loss = a.new_zeros(())
    for z in (a, b):
        centered = z - z.mean(dim=0)
        cov = centered.T @ centered / (len(z) - 1)
        off_diagonal = cov - torch.diag(torch.diag(cov))
        cov_loss = cov_loss + off_diagonal.square().sum() / z.shape[1] / 2
    return {"vicreg": 25 * invariance + 25 * variance + cov_loss,
            "invariance": invariance, "variance": variance, "covariance": cov_loss,
            "projection_std": (std_a.mean() + std_b.mean()) / 2}


class SJEPA(nn.Module):
    """Visible-token online encoder, full-input EMA target, and latent predictor."""

    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg
        self.view_encoder = Encoder(cfg)
        self.target_encoder = copy.deepcopy(self.view_encoder).requires_grad_(False)
        self.target_encoder.eval()
        self.context_projection = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.predictor_position = nn.Parameter(torch.randn(1, cfg.tokens, cfg.dim) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 4,
                                           dropout=0, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.predictor = nn.TransformerEncoder(layer, cfg.predictor_depth,
                                                norm=nn.LayerNorm(cfg.dim),
                                                enable_nested_tensor=False)
        self.predictor_output = nn.Linear(cfg.dim, cfg.dim)
        self.vicreg_projector = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.GELU(),
                                            nn.Linear(cfg.dim, cfg.dim))
        self.register_buffer("target_center", torch.zeros(1, 1, cfg.dim))

    def train(self, mode: bool = True):
        super().train(mode)
        self.target_encoder.eval()
        return self

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *,
                stochastic_positions: bool = False,
                generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        if mask.dtype != torch.bool or mask.shape != (len(x), self.cfg.tokens):
            raise ValueError("mask must be boolean [B, tokens]")
        valid = token_validity(x, self.cfg)
        if ((valid & ~mask).sum(dim=1) == 0).any():
            raise ValueError("Each example needs at least one valid context token")
        if ((valid & mask).sum(dim=1) == 0).any():
            raise ValueError("Each example needs at least one valid target token")
        visible = _visible_indices(mask)
        context_valid = valid.gather(1, visible)
        context = self.view_encoder(x, visible)
        encoded = self.context_projection(context)
        predictor_input = self.mask_token.expand(len(x), self.cfg.tokens, -1).clone()
        predictor_input.scatter_(1, visible[..., None].expand_as(encoded), encoded)
        predictor_input = predictor_input + self.predictor_position
        if stochastic_positions and self.cfg.spe:
            # StoP-inspired adaptation: noise is projected through the SAME
            # trainable linear map as context. Only target positional queries
            # receive noise. No skeletal coordinate is jittered here.
            noise = torch.randn(len(x), self.cfg.tokens, self.cfg.dim,
                                generator=generator, dtype=x.dtype).to(x.device)
            predictor_input = predictor_input + self.cfg.spe_sigma * self.context_projection(noise) * mask[..., None]
        # Invalid visible slots contain no observation. Hidden query slots stay
        # available even when their target is missing: future missingness must
        # never alter the predictor's view of the observed prefix.
        predictor_padding = (~valid) & (~mask)
        predicted = self.predictor_output(self.predictor(predictor_input, src_key_padding_mask=predictor_padding))
        with torch.no_grad():
            target = self.target_encoder(x)
        return {"predicted": predicted, "target": target, "context": context,
                "mask": mask, "valid": valid, "context_valid": context_valid,
                "target_mask": mask & valid}

    def _token_kl(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """KL(target || prediction), an entropy-adjusted latent surprise score."""
        teacher_log = F.log_softmax((target.detach() - self.target_center) / self.cfg.target_temperature, dim=-1)
        student_log = F.log_softmax(predicted / self.cfg.student_temperature, dim=-1)
        return (teacher_log.exp() * (teacher_log - student_log)).sum(dim=-1).clamp_min(0)

    def loss(self, x: torch.Tensor, mask: torch.Tensor | None = None, *,
             generator: torch.Generator | None = None, mode: Mode = "masked",
             positional_generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        if mode not in ("masked", "forecast"):
            raise ValueError("mode must be masked or forecast")
        if mask is None:
            mask = (uniform_mask(len(x), self.cfg.tokens, self.cfg.mask_ratio, generator=generator, device=x.device)
                    if mode == "masked" else forecast_mask(len(x), self.cfg, device=x.device))
        if mode == "forecast" and not torch.equal(mask, forecast_mask(len(x), self.cfg, device=x.device)):
            raise ValueError("Forecasting must hide the entire configured future suffix")
        # SPE uses a separate stream so toggling it cannot change later masks.
        out = self(x, mask, stochastic_positions=self.training, generator=positional_generator)
        teacher_prob = F.softmax((out["target"] - self.target_center) / self.cfg.target_temperature, dim=-1)
        student_log = F.log_softmax(out["predicted"] / self.cfg.student_temperature, dim=-1)
        token_ce = -(teacher_prob * student_log).sum(dim=-1)
        target_mask = out["target_mask"]
        prediction = ((token_ce * target_mask).sum(dim=1) / target_mask.sum(dim=1)).mean()
        pooled_context = _valid_pool(out["context"], out["context_valid"])
        auxiliary = {key: x.new_zeros(()) for key in ("vicreg", "invariance", "variance", "covariance", "projection_std")}
        if self.cfg.vicreg_weight:
            second_mask = (uniform_mask(len(x), self.cfg.tokens, self.cfg.mask_ratio, generator=generator, device=x.device)
                           if mode == "masked" else mask)
            # Mild coordinate noise gives a second view even in forecast mode.
            # It never changes which positions the forecaster may observe.
            noise = torch.randn(x.shape, generator=generator, dtype=x.dtype).to(x.device) * 0.005
            second_visible = _visible_indices(second_mask)
            second_valid = out["valid"].gather(1, second_visible)
            second_context = self.view_encoder(x + noise, second_visible)
            a = self.vicreg_projector(pooled_context)
            b = self.vicreg_projector(_valid_pool(second_context, second_valid))
            auxiliary = vicreg(a, b)
        return {"loss": prediction + self.cfg.vicreg_weight * auxiliary["vicreg"],
                "prediction": prediction, **auxiliary,
                "embedding_std": pooled_context.std(dim=0, unbiased=False).mean(),
                "target_mean": out["target"][target_mask].mean(dim=0).detach(),
                "observed_token_fraction": out["valid"].float().mean(),
                "supervised_token_fraction": target_mask.float().mean(),
                "mask": mask, "target_mask": target_mask}

    @torch.no_grad()
    def update_target(self, momentum: float = 0.996, target_mean: torch.Tensor | None = None):
        """Call exactly once AFTER a successful optimizer step."""
        if not 0 <= momentum <= 1:
            raise ValueError("EMA momentum must be between zero and one")
        for target, source in zip(self.target_encoder.parameters(), self.view_encoder.parameters(), strict=True):
            target.lerp_(source.detach(), 1 - momentum)
        if target_mean is not None:
            self.target_center.lerp_(target_mean.reshape_as(self.target_center), 1 - self.cfg.center_momentum)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Mean-pool full-window online features for the frozen downstream probe.

        No noise or random masking is used. This is recognition of an observed
        window, and must not be described as a future-only prediction.
        """
        with _evaluation(self):
            return _valid_pool(self.view_encoder(x), token_validity(x, self.cfg))

    @torch.no_grad()
    def latent_residual(self, x: torch.Tensor, *, masks: int = 8, seed: int = 17,
                        mode: Mode = "masked") -> dict[str, torch.Tensor]:
        """Deterministic mean masked KL and token attribution from a fixed mask bank.

        Attribution means prediction error, not a causal clinical explanation.
        Unobserved tokens have NaN attributions and zero coverage. The fixed
        bank is repeated independently for every window, so changing the batch
        size or order does not change a window's score.
        """
        if masks < 1 or mode not in ("masked", "forecast"):
            raise ValueError("Need masks >= 1 and a recognized mode")
        generator = torch.Generator().manual_seed(seed)
        total = x.new_zeros((len(x), self.cfg.tokens))
        counts = x.new_zeros(total.shape)
        with _evaluation(self):
            for _ in range(masks if mode == "masked" else 1):
                one = (uniform_mask(1, self.cfg.tokens, self.cfg.mask_ratio, generator=generator, device=x.device)
                       if mode == "masked" else forecast_mask(1, self.cfg, device=x.device))
                mask = one.expand(len(x), -1)
                out = self(x, mask, stochastic_positions=False)
                error = self._token_kl(out["predicted"], out["target"])
                observed_targets = out["target_mask"]
                total += error * observed_targets
                counts += observed_targets
        token_scores = torch.where(counts > 0, total / counts.clamp_min(1), torch.nan)
        return {"score": total.sum(dim=1) / counts.sum(dim=1),
                "per_token": token_scores.reshape(len(x), self.cfg.time_tokens, self.cfg.joints),
                "coverage": counts.reshape(len(x), self.cfg.time_tokens, self.cfg.joints)}


def save_checkpoint(model: SJEPA, path: str | Path, *, manifest_hash: str,
                    split_hash: str, stage: str, history: list | None = None,
                    overwrite: bool = False) -> Path:
    """Save model AND EMA target/center with mandatory dataset and split identity.

    Anchors are protected by the default refusal to overwrite an existing file.
    These are inference/restart checkpoints, not exact optimizer-resume files.
    """
    path = Path(path)
    if not manifest_hash or not split_hash or not stage:
        raise ValueError("manifest_hash, split_hash and stage are required")
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"config": asdict(model.cfg), "state_dict": model.state_dict(),
               "joint_indices": list(JOINT_INDICES), "manifest_hash": manifest_hash,
               "split_hash": split_hash, "stage": stage, "history": history or []}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    return path


def load_checkpoint(path: str | Path, *, manifest_hash: str, split_hash: str,
                    device: str = "cpu") -> tuple[SJEPA, dict]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["manifest_hash"] != manifest_hash or payload["split_hash"] != split_hash:
        raise ValueError("Checkpoint provenance does not match current manifest and split")
    if tuple(payload["joint_indices"]) != JOINT_INDICES:
        raise ValueError("Checkpoint joint whitelist differs")
    model = SJEPA(ModelConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}
