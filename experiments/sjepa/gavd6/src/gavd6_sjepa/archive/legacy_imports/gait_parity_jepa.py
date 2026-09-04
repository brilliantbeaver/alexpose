"""Compatibility imports for the renamed JEPA implementation.

New code should import the reflection-equivariance implementation package.
"""

from gavd6_sjepa.research_directions.reflection_equivariance import jepa_model_architecture as _implementation
from gavd6_sjepa.research_directions.reflection_equivariance.jepa_model_architecture import *


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
