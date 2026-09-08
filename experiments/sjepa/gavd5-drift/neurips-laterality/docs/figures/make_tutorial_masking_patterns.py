"""Draw four illustrative masking patterns with the same hidden-token count.

Run from the repository root:
    .venv/bin/python -B neurips-laterality/docs/figures/make_tutorial_masking_patterns.py

These small grids explain masking choices. They are not training results or
the landmark set and masking fraction used in the completed experiments.
The script reads no research data and writes only its SVG and PDF figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


HERE = Path(__file__).resolve().parent
ROW_LABELS = (
    "Left hip", "Left knee", "Left ankle",
    "Right hip", "Right knee", "Right ankle",
)
HIDDEN = "#2563A6"
VISIBLE = "#EDF2F7"
VISIBLE_EDGE = "#CDD7E1"
TEXT = "#263445"


def illustrative_patterns() -> tuple[tuple[str, np.ndarray], ...]:
    """Specify equally sized examples without sampling or selecting results."""
    scattered = np.zeros((6, 6), dtype=bool)
    for row, columns in enumerate(((0, 3), (1, 5), (2, 4), (0, 4), (2, 5), (1, 3))):
        scattered[row, columns] = True

    trajectories = np.zeros((6, 6), dtype=bool)
    trajectories[[2, 5], :] = True

    connected_limb = np.zeros((6, 6), dtype=bool)
    connected_limb[:3, 1:5] = True

    temporal_gap = np.zeros((6, 6), dtype=bool)
    temporal_gap[:, 2:4] = True

    patterns = (
        ("A  Scattered tokens", scattered),
        ("B  Whole-joint trajectories", trajectories),
        ("C  Connected-limb interval", connected_limb),
        ("D  Temporal gap", temporal_gap),
    )
    for _, mask in patterns:
        assert mask.shape == (6, 6)
        assert mask.dtype == np.dtype(bool)
        assert int(mask.sum()) == 12
    return patterns


def make_figure() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(6.8, 5.9), facecolor="white")
    positions = (
        (0.158, 0.555, 0.27, 0.30),
        (0.655, 0.555, 0.27, 0.30),
        (0.158, 0.110, 0.27, 0.30),
        (0.655, 0.110, 0.27, 0.30),
    )
    heading_positions = ((0.031, 0.890), (0.528, 0.890), (0.031, 0.445), (0.528, 0.445))
    for (heading, mask), position, heading_position in zip(
        illustrative_patterns(), positions, heading_positions
    ):
        ax = fig.add_axes(position)
        for row in range(6):
            for column in range(6):
                hidden = bool(mask[row, column])
                ax.add_patch(Rectangle(
                    (column + 0.06, row + 0.06), 0.88, 0.88,
                    facecolor=HIDDEN if hidden else VISIBLE,
                    edgecolor=HIDDEN if hidden else VISIBLE_EDGE,
                    linewidth=0.55,
                ))
        ax.set_xlim(0, 6)
        ax.set_ylim(6, 0)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(6) + 0.5, [str(block) for block in range(1, 7)])
        ax.set_yticks(np.arange(6) + 0.5, ROW_LABELS)
        ax.tick_params(axis="both", which="both", length=0, pad=5, labelsize=9)
        ax.set_xlabel("Time block", fontsize=9, labelpad=5)
        for spine in ax.spines.values():
            spine.set_visible(False)
        fig.text(*heading_position, heading, fontsize=10.6, weight="bold", va="bottom")

    fig.legend(
        handles=[
            Patch(facecolor=HIDDEN, edgecolor=HIDDEN, label="Hidden target"),
            Patch(facecolor=VISIBLE, edgecolor=VISIBLE_EDGE, label="Visible"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.985),
        frameon=False, ncol=2, fontsize=10,
        handlelength=1.0, handleheight=1.0, columnspacing=2.3,
    )
    for suffix in ("svg", "pdf"):
        destination = HERE / f"tutorial_masking_patterns.{suffix}"
        fig.savefig(destination, facecolor="white")
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    make_figure()
