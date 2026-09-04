#!/usr/bin/env python3
"""Compatibility launcher for ``gavd6 laterality build-manifest``."""

from gavd6_sjepa.research_directions.latent_laterality import laterality_manifest_construction as _implementation
from gavd6_sjepa.research_directions.latent_laterality.laterality_manifest_construction import *


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
