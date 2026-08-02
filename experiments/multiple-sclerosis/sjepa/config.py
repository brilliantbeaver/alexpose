"""Configuration for the S-JEPA gait tutorials.

This module defines a single :class:`SJEPAConfig` dataclass and two ready made
profiles that control the size and speed of the model. A learner picks a profile
with the ``SJEPA_PROFILE`` environment variable, usually set in the repository
root ``.env`` file:

* ``laptop`` (default): a tiny model that trains in minutes on a CPU, an Apple
  Silicon MPS device, or a free Colab T4 GPU. Use this to learn the ideas.
* ``gpu``: a larger model, closer to the settings in the S-JEPA paper. Use this
  when you have a real GPU and want stronger representations.

A third switch, ``SJEPA_SMOKE=1``, shrinks whichever profile you picked down to a
near instant size. The notebooks and the test suite use it to check that every
step runs end to end without waiting for real training.

Nothing here imports torch, so the config can be inspected cheaply from any
notebook cell before the heavy libraries load.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import List


# The twelve neurologically relevant BlazePose-33 landmarks that we mask. These
# are both shoulders plus both complete legs. See masking.py for the full story.
ANATOMICAL_MASK_IDX: List[int] = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

# The three health conditions we classify. The order fixes the label integers.
CLASS_NAMES: List[str] = ["normal", "ms", "pd"]


@dataclass
class SJEPAConfig:
    """All knobs for building and training the S-JEPA model.

    The defaults describe the ``laptop`` profile. Call :func:`get_config` to load
    a named profile and apply any environment overrides.
    """

    # Identity
    profile: str = "laptop"

    # Skeleton and windowing
    num_joints: int = 33            # BlazePose-33
    in_channels: int = 3            # x, y, visibility
    window_frames: int = 32         # frames per training window
    frame_group: int = 4            # l: adjacent frames grouped into one token
    target_fps: int = 15            # sampling rate used during pose extraction
    window_stride: int = 16         # hop between consecutive windows

    # Transformer sizes
    encoder_dim: int = 96
    encoder_depth: int = 3
    encoder_heads: int = 4
    predictor_dim: int = 96
    predictor_depth: int = 2
    mlp_ratio: float = 2.0
    dropout: float = 0.0

    # Optimisation
    batch_size: int = 32
    lr: float = 1.0e-3
    weight_decay: float = 0.05
    pretrain_epochs: int = 40
    finetune_epochs: int = 30
    warmup_epochs: int = 3

    # EMA target encoder schedule (cosine from start to end across training)
    ema_start: float = 0.996
    ema_end: float = 1.0

    # Centering and sharpening for the latent cross-entropy loss
    center_beta: float = 0.9
    tau_pred: float = 0.1
    tau_target: float = 0.06

    # VICReg (added as an extension, not part of the original S-JEPA)
    vicreg_weight: float = 0.5
    vicreg_sim: float = 25.0
    vicreg_var: float = 25.0
    vicreg_cov: float = 1.0
    vicreg_gamma: float = 1.0       # variance floor per dimension

    # Reproducibility
    seed: int = 42

    # Derived helpers -----------------------------------------------------

    @property
    def num_time_tokens(self) -> int:
        """How many temporal tokens each joint contributes."""
        return self.window_frames // self.frame_group

    @property
    def num_tokens(self) -> int:
        """Total token count N = time tokens x joints."""
        return self.num_time_tokens * self.num_joints

    @property
    def token_in_dim(self) -> int:
        """Flattened size of one token before projection: l x C."""
        return self.frame_group * self.in_channels

    def to_dict(self) -> dict:
        return asdict(self)


# Profile presets. Each dict lists only the fields that differ from the laptop
# defaults above, so the intent of every profile stays readable.
_PROFILES = {
    "laptop": {},  # the dataclass defaults already are the laptop profile
    "gpu": dict(
        profile="gpu",
        window_frames=64,
        window_stride=24,
        encoder_dim=256,
        encoder_depth=8,
        encoder_heads=8,
        predictor_dim=256,
        predictor_depth=5,
        mlp_ratio=4.0,
        batch_size=128,
        pretrain_epochs=200,
        finetune_epochs=120,
        warmup_epochs=10,
        ema_start=0.999,
    ),
}

# The smoke override collapses any profile to something that runs in seconds.
_SMOKE_OVERRIDE = dict(
    window_frames=16,
    window_stride=16,
    encoder_dim=32,
    encoder_depth=1,
    encoder_heads=2,
    predictor_dim=32,
    predictor_depth=1,
    mlp_ratio=2.0,
    batch_size=8,
    pretrain_epochs=2,
    finetune_epochs=2,
    warmup_epochs=0,
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def get_config(profile: str | None = None, smoke: bool | None = None) -> SJEPAConfig:
    """Build a config for a named profile with environment overrides applied.

    Resolution order:

    1. ``profile`` argument if given, else ``SJEPA_PROFILE`` env var, else
       ``laptop``.
    2. If ``smoke`` is True (argument) or ``SJEPA_SMOKE`` is set, shrink to the
       smoke size.
    """
    if profile is None:
        profile = os.environ.get("SJEPA_PROFILE", "laptop").strip().lower()
    if profile not in _PROFILES:
        raise ValueError(
            f"Unknown SJEPA profile '{profile}'. Choose one of {sorted(_PROFILES)}."
        )

    cfg = SJEPAConfig(**{**{"profile": profile}, **_PROFILES[profile]})

    # Optional environment overrides for a couple of common knobs, so the sample
    # .env values actually take effect.
    if os.environ.get("SJEPA_TARGET_FPS"):
        cfg.target_fps = int(os.environ["SJEPA_TARGET_FPS"])
    if os.environ.get("SJEPA_SEED"):
        cfg.seed = int(os.environ["SJEPA_SEED"])

    use_smoke = smoke if smoke is not None else _env_flag("SJEPA_SMOKE")
    if use_smoke:
        for key, value in _SMOKE_OVERRIDE.items():
            setattr(cfg, key, value)
        cfg.profile = f"{profile}+smoke"

    # window_frames must be a whole multiple of frame_group so tokenisation is exact
    if cfg.window_frames % cfg.frame_group != 0:
        raise ValueError(
            f"window_frames ({cfg.window_frames}) must be divisible by "
            f"frame_group ({cfg.frame_group})."
        )
    return cfg


def describe(cfg: SJEPAConfig) -> str:
    """A short human readable summary, handy for a notebook print cell."""
    return (
        f"profile={cfg.profile} | window={cfg.window_frames}f @ {cfg.target_fps}fps | "
        f"tokens N={cfg.num_tokens} ({cfg.num_time_tokens} time x {cfg.num_joints} joints) | "
        f"encoder {cfg.encoder_depth}L x {cfg.encoder_dim}d x {cfg.encoder_heads}h | "
        f"predictor {cfg.predictor_depth}L x {cfg.predictor_dim}d | "
        f"pretrain {cfg.pretrain_epochs}ep, finetune {cfg.finetune_epochs}ep"
    )
