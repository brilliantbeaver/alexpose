"""Compatibility imports for the renamed AMASS Core11 training pipeline.

New code should import the reflection-equivariance implementation package.
"""

from gavd6_sjepa.research_directions.reflection_equivariance import amass_training as _implementation
from gavd6_sjepa.research_directions.reflection_equivariance.amass_training import *


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
