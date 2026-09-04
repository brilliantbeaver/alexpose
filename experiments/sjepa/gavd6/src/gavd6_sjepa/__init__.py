"""GAVD6 S-JEPA research package.

The public model objects are loaded lazily so ``gavd6 --help`` does not import
Torch, OpenCV, or the body-model stack.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .research_directions.reflection_equivariance.jepa_model_architecture import TrainConfig

__all__ = ["TrainConfig", "VARIANTS", "build_model"]


def __getattr__(name: str):
    if name in __all__:
        from .research_directions.reflection_equivariance import jepa_model_architecture

        return getattr(jepa_model_architecture, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
