"""Compatibility entry point for the complete, consistently styled figure set.

Uses the Pillow/CairoSVG environment documented in build_figures.py.
HTML galleries remain owned by build_paper.py.
"""
from build_figures import main

if __name__ == '__main__':
    main()
