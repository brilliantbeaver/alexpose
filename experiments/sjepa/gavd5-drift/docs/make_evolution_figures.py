#!/usr/bin/env python
"""Compatibility entry point for the retired evolution figure generator.

The old plots mixed the 159-row augmented run with current claims. This entry
point now regenerates only the current, fingerprint-checked figures.
"""

from make_figures import make_brainbody, make_downstream


if __name__ == "__main__":
    make_brainbody()
    make_downstream()
