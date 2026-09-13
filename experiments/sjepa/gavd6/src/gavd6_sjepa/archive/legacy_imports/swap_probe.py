"""Compatibility imports for the renamed swap-probe evaluation pipeline."""

from gavd6_sjepa.research_directions.reflection_equivariance import swap_probe as _implementation
from gavd6_sjepa.research_directions.reflection_equivariance.swap_probe import *


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
