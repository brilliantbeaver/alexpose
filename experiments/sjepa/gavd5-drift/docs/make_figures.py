#!/usr/bin/env python
"""Regenerate every figure referenced by the current top-level documents."""

from make_brainbody_figures import main as make_brainbody
from make_downstream_probe_figure import main as make_downstream


if __name__ == "__main__":
    make_brainbody()
    make_downstream()
