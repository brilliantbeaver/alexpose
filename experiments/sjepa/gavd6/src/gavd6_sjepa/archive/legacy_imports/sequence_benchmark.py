"""Compatibility imports for the renamed laterality benchmark module."""

from gavd6_sjepa.research_directions.latent_laterality import laterality_sequence_benchmarking as _implementation
from gavd6_sjepa.research_directions.latent_laterality.laterality_sequence_benchmarking import *


def __getattr__(name: str):
    return getattr(_implementation, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(dir(_implementation)))
