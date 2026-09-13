#!/usr/bin/env python3
"""Compatibility launcher for ``gavd6 amass convert``."""

from gavd6_sjepa.data_foundations import amass_conversion as _implementation
from gavd6_sjepa.data_foundations.amass_conversion import *


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
