"""Compatibility imports for the renamed laterality gauge evaluator."""

from gavd6_sjepa.research_directions.latent_laterality import laterality_gauge_evaluation_pipeline as _implementation
from gavd6_sjepa.research_directions.latent_laterality.laterality_gauge_evaluation_pipeline import *


def __getattr__(name: str):
    return getattr(_implementation, name)
