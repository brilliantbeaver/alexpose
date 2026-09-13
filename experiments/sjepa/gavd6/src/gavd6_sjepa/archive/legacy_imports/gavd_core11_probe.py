"""Compatibility imports for the renamed GAVD Core11 probe evaluator."""

from gavd6_sjepa.research_directions.reflection_equivariance import gavd_probe as _implementation
from gavd6_sjepa.research_directions.reflection_equivariance.gavd_probe import *


def __getattr__(name: str):
    return getattr(_implementation, name)
