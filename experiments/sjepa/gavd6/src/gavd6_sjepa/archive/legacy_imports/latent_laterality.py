"""Compatibility imports for the renamed laterality inference module."""

from gavd6_sjepa.research_directions.latent_laterality import laterality_corruption_inference as _implementation
from gavd6_sjepa.research_directions.latent_laterality.laterality_corruption_inference import *


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
