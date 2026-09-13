"""Compatibility imports for the renamed laterality gauge trainer."""

from gavd6_sjepa.research_directions.latent_laterality import training as _implementation
from gavd6_sjepa.research_directions.latent_laterality.training import *


def __getattr__(name: str):
    return getattr(_implementation, name)
