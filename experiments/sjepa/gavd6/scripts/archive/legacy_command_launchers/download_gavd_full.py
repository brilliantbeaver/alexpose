#!/usr/bin/env python3
"""Compatibility launcher for ``gavd6 gavd download``."""

from gavd6_sjepa.data_foundations import gavd_video_download_pipeline as _implementation
from gavd6_sjepa.data_foundations.gavd_video_download_pipeline import *


def __getattr__(name: str):
    return getattr(_implementation, name)


if __name__ == "__main__":
    raise SystemExit(_implementation.main())
