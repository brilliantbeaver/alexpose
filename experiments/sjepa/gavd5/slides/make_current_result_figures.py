#!/usr/bin/env python3
"""Rebuild the tutorial's downstream-result figures from verified artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import PercentFormatter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "docs"))

from make_downstream_probe_figure import (  # noqa: E402
    COLORS,
    load_probe_scores,
    plot_probe_scores,
)


DISPLAY_NAMES = {
    "cerebralpalsy": "Cerebral palsy",
    "myopathic": "Myopathic gait",
    "normal": "Typical gait",
    "parkinsons": "Parkinson's disease",
    "stroke": "Stroke",
}


def save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg"):
        figure.savefig(output_dir / f"{stem}.{suffix}", bbox_inches="tight")
    figure.savefig(output_dir / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(figure)


def label_bars(axis: plt.Axes, containers: list, offset: float = 0.022) -> None:
    for container in containers:
        for bar in container:
            value = bar.get_height()
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.1%}",
                ha="center",
                fontsize=7,
                color=COLORS["ink"],
            )


def plot_video_separation(scores: pd.DataFrame) -> plt.Figure:
    selected = scores.iloc[[0, 3]].reset_index(drop=True)
    labels = [
        "Examples divided at random\n(source videos can cross sides)",
        "Source videos kept separate\n(all predictions combined)",
    ]
    x = np.arange(2)
    width = 0.32
    figure, axis = plt.subplots(figsize=(7.1, 3.15))
    accuracy = axis.bar(
        x - width / 2,
        selected["accuracy"],
        width,
        color=COLORS["blue"],
        label="Accuracy (correct identifications)",
    )
    f1 = axis.bar(
        x + width / 2,
        selected["macro_f1"],
        width,
        color=COLORS["teal"],
        label="F1 score (balances mistakes and misses)",
    )
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_ylabel("Performance (higher is better)")
    axis.set_title("The estimate becomes more cautious when source videos cannot cross sides")
    axis.legend(loc="upper right", frameon=False)
    axis.grid(axis="y", alpha=0.24)
    axis.set_axisbelow(True)
    label_bars(axis, [accuracy, f1])
    figure.suptitle(
        "Keeping source videos separate changes the result",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.text(
        0.5,
        0.04,
        "This is still not a fully unseen-video test because every video helped shape the learned movement features earlier.",
        ha="center",
        fontsize=6.6,
        color=COLORS["red"],
    )
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.25, top=0.78)
    return figure


def plot_condition_f1(report: pd.DataFrame) -> plt.Figure:
    rows = report.loc[report.index.isin(DISPLAY_NAMES)].copy()
    rows = rows.loc[list(DISPLAY_NAMES)]
    values = rows["f1-score"].astype(float).to_numpy()
    support = rows["support"].astype(int).to_numpy()
    labels = [DISPLAY_NAMES[name] for name in rows.index]

    figure, axis = plt.subplots(figsize=(7.1, 3.15))
    x = np.arange(len(labels))
    bars = axis.bar(x, values, width=0.62, color=COLORS["teal"])
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    axis.set_ylabel("F1 score (higher is better)")
    axis.set_title("Matched comparison: 47 examples for training and 21 for testing")
    axis.grid(axis="y", alpha=0.24)
    axis.set_axisbelow(True)
    label_bars(axis, [bars], offset=0.025)
    for index, count in enumerate(support):
        axis.text(
            index,
            0.035,
            f"{count} test examples",
            ha="center",
            fontsize=6.2,
            color=COLORS["ink"],
        )
    figure.suptitle(
        "Performance varies by walking condition",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.text(
        0.5,
        0.035,
        "The smallest conditions have only three test examples, so individual percentages are unstable.",
        ha="center",
        fontsize=6.6,
        color=COLORS["red"],
    )
    figure.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.79)
    return figure


def plot_confusion_matrix(matrix: pd.DataFrame) -> plt.Figure:
    order = list(DISPLAY_NAMES)
    matrix = matrix.loc[order, order]
    labels = [DISPLAY_NAMES[name] for name in order]
    values = matrix.to_numpy(dtype=int)
    figure, axis = plt.subplots(figsize=(6.4, 4.4))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=max(1, values.max()))
    axis.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Predicted walking condition")
    axis.set_ylabel("Actual walking condition")
    axis.set_title("Matched 21-example test: correct predictions lie on the diagonal")
    threshold = values.max() / 2
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else COLORS["ink"],
                fontsize=9,
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Number of examples")
    figure.suptitle(
        "Where did the matched comparison make mistakes?",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.subplots_adjust(left=0.22, right=0.90, bottom=0.25, top=0.83)
    return figure


def plot_representation_geometry(
    matrix: pd.DataFrame, summary: pd.Series
) -> plt.Figure:
    order = list(DISPLAY_NAMES)
    matrix = matrix.loc[order, order]
    labels = [DISPLAY_NAMES[name] for name in order]
    values = matrix.to_numpy(dtype=float)

    figure, axis = plt.subplots(figsize=(7.1, 4.25))
    image = axis.imshow(
        values,
        cmap="YlGnBu",
        vmin=0,
        vmax=max(0.7, float(values.max())),
    )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=24, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels)
    axis.set_xlabel("Average movement pattern")
    axis.set_ylabel("Average movement pattern")
    axis.set_title(
        "Larger values mean the average movement patterns are more different"
    )
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.38 else COLORS["ink"],
            )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Difference between condition averages")

    closest = float(summary["minimum_between_centroid_cosine_distance"])
    within = float(summary["mean_within_condition_cosine_distance"])
    figure.suptitle(
        "The five walking conditions do not form cleanly separated groups",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.text(
        0.5,
        0.025,
        f"The closest condition averages are only {closest:.3f} apart, while examples within one condition vary by {within:.3f} on average. "
        "The groups therefore overlap substantially.",
        ha="center",
        fontsize=6.7,
        color=COLORS["red"],
    )
    figure.subplots_adjust(left=0.20, right=0.88, bottom=0.25, top=0.82)
    return figure


def add_card(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    color: str,
) -> None:
    axis.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            facecolor="#FFFFFF",
            edgecolor=color,
            linewidth=1.6,
        )
    )
    axis.text(
        x + 0.03,
        y + height - 0.08,
        title,
        ha="left",
        va="top",
        fontsize=8.8,
        fontweight="bold",
        color=color,
    )
    axis.text(
        x + 0.03,
        y + height - 0.29,
        body,
        ha="left",
        va="top",
        fontsize=7.2,
        color=COLORS["ink"],
        linespacing=1.35,
    )


def plot_evaluation_approaches() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.1, 3.15))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    add_card(
        axis,
        0.02,
        0.22,
        0.29,
        0.60,
        "All 96 examples\ndivided at random",
        "Can the features recover labels\ninside the known set?\n\nSource videos can appear\non both sides.",
        COLORS["blue"],
    )
    add_card(
        axis,
        0.355,
        0.22,
        0.29,
        0.60,
        "Matched comparison\n47 training, 21 testing",
        "Can we repeat the historical\ncomparison?\n\nSource videos can appear\non both sides.",
        COLORS["teal"],
    )
    add_card(
        axis,
        0.69,
        0.22,
        0.29,
        0.60,
        "Source videos\nkept separate",
        "What changes when the final\nclassifier cannot share videos?\n\nEarlier feature learning still\nused every video.",
        COLORS["gold"],
    )
    axis.text(
        0.5,
        0.08,
        "Each approach answers a different question; none estimates a completely unseen video from start to finish.",
        ha="center",
        fontsize=7,
        color=COLORS["red"],
    )
    figure.suptitle(
        "Three evaluation approaches, three limited questions",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.86)
    return figure


def plot_interpretation_boundary() -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7.1, 3.15))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    add_card(
        axis,
        0.04,
        0.27,
        0.42,
        0.56,
        "What the current results support",
        "The movement features contain information\nrelated to the five walking conditions.\n\nKeeping source videos separate makes the\nclassification result more cautious.",
        COLORS["teal"],
    )
    add_card(
        axis,
        0.54,
        0.27,
        0.42,
        0.56,
        "What they do not yet support",
        "They do not estimate performance for a new\nperson, clinic, camera, or source video.\n\nEvery example helped shape the movement\nfeatures before the final classifier test.",
        COLORS["red"],
    )
    axis.text(
        0.5,
        0.10,
        "Next valid test: separate complete source videos before any feature learning begins.",
        ha="center",
        fontsize=7.5,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.suptitle(
        "A clear boundary around the current evidence",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    figure.subplots_adjust(left=0.03, right=0.97, bottom=0.04, top=0.86)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ROOT / "work" / "artifacts" / "real",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "slides" / "figures",
    )
    args = parser.parse_args()
    artifacts = args.artifact_dir.resolve()
    output_dir = args.output_dir.resolve()

    scores, contract, checkpoint_sha = load_probe_scores(artifacts)
    report = pd.read_csv(
        artifacts / "five_class_classification_report.csv", index_col=0
    )
    confusion = pd.read_csv(
        artifacts / "five_class_confusion_matrix.csv", index_col=0
    )
    geometry_matrix = pd.read_csv(
        artifacts / "curriculum_centroid_distances.csv", index_col=0
    )
    geometry_summary = pd.read_csv(
        artifacts / "curriculum_representation_geometry.csv"
    ).iloc[0]
    if str(geometry_summary["checkpoint_fingerprint"]) != str(
        contract["checkpoint_fingerprint"]
    ):
        raise RuntimeError("Representation geometry does not match the verified run")

    save_figure(
        plot_probe_scores(scores, contract, checkpoint_sha),
        output_dir,
        "current_downstream_scores",
    )
    save_figure(plot_video_separation(scores), output_dir, "current_video_separation")
    save_figure(plot_condition_f1(report), output_dir, "current_condition_f1")
    save_figure(plot_confusion_matrix(confusion), output_dir, "current_confusion_matrix")
    save_figure(
        plot_representation_geometry(geometry_matrix, geometry_summary),
        output_dir,
        "current_representation_geometry",
    )
    save_figure(plot_evaluation_approaches(), output_dir, "current_evaluation_approaches")
    save_figure(plot_interpretation_boundary(), output_dir, "current_interpretation_boundary")

    print(scores.to_string(index=False))
    print(f"checkpoint_sha256={checkpoint_sha}")
    print(f"wrote figures to {output_dir}")


if __name__ == "__main__":
    main()
