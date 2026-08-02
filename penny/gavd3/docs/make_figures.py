"""Generate publication figures from the locked real-run artifacts."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "cache" / "artifacts" / "real"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 180,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.01, 0.60, 0.16, 0.24, "Public RGB video\nGAVD annotations", "#E8F1FA"),
        (0.21, 0.60, 0.16, 0.24, "MediaPipe\n33-landmark pose", "#E8F1FA"),
        (0.41, 0.60, 0.18, 0.24, "Normal-only S-JEPA\n12 sequences, 1 video", "#FFF2CC"),
        (0.63, 0.60, 0.16, 0.24, "Frozen 384-D\nsequence embedding", "#E2F0D9"),
        (0.83, 0.60, 0.16, 0.24, "Five-class\nRandom Forest", "#E2F0D9"),
        (0.41, 0.12, 0.18, 0.24, "Uniform mask on 10\nliterature-linked joints", "#FCE4D6"),
        (0.63, 0.12, 0.16, 0.24, "82 handcrafted\nfeature reference", "#EDEDED"),
        (0.83, 0.12, 0.16, 0.24, "Leakage and\nmissingness audits", "#EDEDED"),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=0.8,
                edgecolor="#404040",
                facecolor=color,
            )
        )
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=6.5,
            linespacing=1.15,
        )

    def arrow(a: tuple[float, float], b: tuple[float, float]) -> None:
        ax.add_patch(
            FancyArrowPatch(
                a,
                b,
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#505050",
            )
        )

    arrow((0.17, 0.72), (0.21, 0.72))
    arrow((0.37, 0.72), (0.41, 0.72))
    arrow((0.59, 0.72), (0.63, 0.72))
    arrow((0.79, 0.72), (0.83, 0.72))
    arrow((0.50, 0.36), (0.50, 0.60))
    arrow((0.79, 0.24), (0.83, 0.24))
    arrow((0.72, 0.36), (0.90, 0.60))
    ax.text(
        0.72,
        0.49,
        "system-level comparison",
        ha="center",
        va="center",
        fontsize=7,
        color="#555555",
    )
    save(fig, "pipeline")


def make_result_summary() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    labels = ["82-feature RF", "Frozen S-JEPA", "Missingness", "Majority"]
    values = [0.7619047619, 0.6190476190, 0.3333333333, 0.2941176471]
    colors = ["#4C78A8", "#F58518", "#BAB0AC", "#9D755D"]
    bars = ax.bar(labels, values, color=colors, width=0.66)
    ax.axhline(0.20, color="#555555", linestyle="--", linewidth=1, label="Five-class chance")
    ax.set_ylim(0, 0.86)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Exact 47/21 sequence split")
    ax.tick_params(axis="x", rotation=24, labelsize=6.5)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save(fig, "exact_split_results")


def make_training_health() -> None:
    history = pd.read_csv(ARTIFACTS / "training_history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.35))

    axes[0].plot(history["epoch"], history["loss"], color="#4C78A8", linewidth=1.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training cross-entropy")
    axes[0].set_title("Prediction objective")
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        history["epoch"],
        history["feature_std"],
        color="#54A24B",
        linewidth=1.2,
        label="Feature standard deviation",
    )
    axes[1].plot(
        history["epoch"],
        history["mean_pair_cosine"],
        color="#E45756",
        linewidth=1.2,
        label="Mean pair cosine",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Representation health")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    save(fig, "training_health")


if __name__ == "__main__":
    make_pipeline()
    make_result_summary()
    make_training_health()
