"""One shared connector vocabulary for every generated diagram in docs/.

Before this module existed each generator carried its own arrow settings, and the set shipped
four incompatible ones: mutation scales of 9, 10, 11, and 12.5, line widths from 1.0 to 1.5,
three different greys, and no clear space reserved at either end anywhere. The diagram design
system in notes/09_diagram_design_system.md asks for a single shared arrow marker with clear
space before the arrowhead, so the geometry lives here and nowhere else.

This module deliberately holds no artifact dependencies, so the two self-contained loss figures
can share the vocabulary without importing the artifact-bound generators.
"""

from __future__ import annotations

# mutation_scale and linewidth are in points, so a connector drawn from this dict is the same
# physical size in every vector output regardless of the figure's pixel density.
#
# shrinkA and shrinkB reserve the clear space at the tail and at the arrowhead. They assume the
# endpoints handed to the arrow are a card's *nominal* rectangle edges, as passed to a
# FancyBboxPatch, so each one also has to absorb that patch's own pad before any clear space is
# left over. A figure whose endpoints are already the drawn edge should ask for smaller values
# through ``arrow_style``.
ARROW_STYLE: dict[str, object] = {
    "arrowstyle": "-|>",
    "mutation_scale": 11,
    "linewidth": 1.5,
    "color": "#20364E",
    "shrinkA": 6,
    "shrinkB": 10,
    "connectionstyle": "arc3,rad=0",
    "joinstyle": "round",
    "capstyle": "round",
}

# A secondary, proposed, or EMA edge keeps the same geometry, weight, and head, and differs only
# by dashing, so it stays distinguishable without introducing a second arrow style.
ARROW_DASHES = (0, (7, 6))


def arrow_style(**overrides) -> dict[str, object]:
    """Return the shared style, with per-figure layout overrides applied.

    Only the clear-space and colour keys are meant to be overridden. Overriding the head, weight,
    or joins would reintroduce exactly the inconsistency this module exists to remove.
    """
    style = dict(ARROW_STYLE)
    style.update(overrides)
    return style
