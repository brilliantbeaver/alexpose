#!/usr/bin/env python3
"""Compatibility launcher for ``gavd6 gavd download``."""

from gavd6_sjepa.data_foundations import video_download as _implementation
from gavd6_sjepa.data_foundations.video_download import *


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
