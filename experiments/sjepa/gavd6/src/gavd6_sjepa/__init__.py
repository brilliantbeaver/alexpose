"""Training components for the GAVD6 S-JEPA experiments."""

from .gait_parity_jepa import TrainConfig, VARIANTS, build_model

__all__ = ["TrainConfig", "VARIANTS", "build_model"]
