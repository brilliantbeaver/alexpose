"""Compatibility imports for the renamed laterality benchmark module."""

from gavd6_sjepa.research_directions.latent_laterality import benchmark as _implementation
from gavd6_sjepa.research_directions.latent_laterality.benchmark import *


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
