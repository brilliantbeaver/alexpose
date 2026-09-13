"""Compatibility imports for the renamed laterality gauge evaluator."""

from gavd6_sjepa.research_directions.latent_laterality import evaluation as _implementation
from gavd6_sjepa.research_directions.latent_laterality.evaluation import *


def __getattr__(name: str):
    return getattr(_implementation, name)
