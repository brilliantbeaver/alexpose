"""S-JEPA gait tutorials: a compact, teachable implementation.

This package is the single source of truth for the model, data, and training
code used across the seven tutorial notebooks. The notebooks import from here so
there is no copy-pasted logic to drift out of sync. For Google Colab, each
notebook's first cells make this package importable (clone or pip install), so the
same code runs locally and in the cloud.

The heavy pieces (torch models, training) are imported lazily by the functions
that need them, so lightweight modules like ``config`` and ``masking`` can be used
even before torch is installed.
"""

from .config import (
    SJEPAConfig,
    get_config,
    describe,
    ANATOMICAL_MASK_IDX,
    CLASS_NAMES,
)
from .masking import (
    AnatomicalMaskSampler,
    joint_token_mask,
    masked_joint_names,
    MASKED_JOINTS,
)

__all__ = [
    "SJEPAConfig",
    "get_config",
    "describe",
    "ANATOMICAL_MASK_IDX",
    "CLASS_NAMES",
    "AnatomicalMaskSampler",
    "joint_token_mask",
    "masked_joint_names",
    "MASKED_JOINTS",
]

__version__ = "0.1.0"
