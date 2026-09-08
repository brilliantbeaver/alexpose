"""Small vector figures that distinguish deliberate masks from missing data."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

from .comparative_masks import LANDMARK_NAMES, MaskResult


def plot_mask_examples(
    examples: Sequence[tuple[str, MaskResult]], token_valid: np.ndarray,
    *, title: str = "Synthetic illustration of missing-information tasks",
):
    """Draw actual masks with vector cells and one shared, literal legend.

All 33 landmarks remain visible in the figure. Horizontal position represents
anatomical identity and vertical position represents the time block.
"""
    if not examples:
        raise ValueError("Provide at least one mask example")
    valid = np.asarray(token_valid, bool)
    columns = min(2, len(examples))
    rows = (len(examples) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, figsize=(13, 3.8 * rows), squeeze=False)
    colors = ListedColormap(["#d8dadd", "#f0f5f8", "#21679b"])
    labels = [name.replace("left ", "L ").replace("right ", "R ") for name in LANDMARK_NAMES]
    for index, (label, result) in enumerate(examples):
        ax = axes.flat[index]
        if result.mask.shape != valid.shape:
            raise ValueError("Every plotted mask must match the shared validity grid")
        values = np.where(valid, 1, 0)
        values[result.mask] = 2
        ax.pcolormesh(np.arange(34), np.arange(len(valid) + 1), values,
                      cmap=colors, vmin=0, vmax=2, edgecolors="white", linewidth=0.6,
                      rasterized=False)
        ax.set_xticks(np.arange(33) + 0.5, labels, rotation=90, fontsize=6.6)
        ax.set_yticks(np.arange(len(valid)) + 0.5, np.arange(1, len(valid) + 1), fontsize=8)
        ax.set_ylabel("Four-step time block", fontsize=9)
        ax.set_ylim(len(valid), 0)
        ax.tick_params(length=0)
        ax.set_title(f"{label}\n{result.coverage['hidden_tokens']} hidden · {result.context_count} observed context",
                     loc="left", fontsize=10.5, pad=8)
        for boundary in (11, 23):
            ax.axvline(boundary, color="#536b7c", linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in axes.flat[len(examples):]:
        ax.set_visible(False)
    legend = [Patch(facecolor=colors(2), label="Deliberately hidden target"),
              Patch(facecolor=colors(1), edgecolor="#9fb0bc", label="Observed context"),
              Patch(facecolor=colors(0), label="Naturally missing; no target")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.945),
               ncol=3, frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=12, y=0.985)
    fig.subplots_adjust(top=0.86, bottom=0.20 if rows == 1 else 0.11,
                        hspace=1.05, wspace=0.18)
    return fig
