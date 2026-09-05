"""Paper-quality, evidence-aware figures for the BrainBodyFM notebooks.

The notebook is the analysis surface and ``docs/figures`` is the paper surface.
Every renderer therefore produces the same figure in both places: an inline SVG
for Jupyter, plus SVG/PNG/PDF files generated from the same figure object.

The module deliberately distinguishes three states:

* current protocol-v2 evidence is plotted from hash-bound fold artifacts;
* archived/transductive evidence is visibly labelled as such; and
* missing prerequisites produce a status diagram, never an empty performance
  plot or an invented value.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd


# Okabe-Ito-derived colors: distinguishable under common color-vision deficits.
BLUE = "#0072B2"
SKY = "#56B4E9"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILION = "#D55E00"
PURPLE = "#CC79A7"
TEXT = "#172033"
MUTED = "#526176"
GRID = "#D9E0EA"
LIGHT = "#EEF3F8"
WHITE = "#FFFFFF"

CONDITIONS = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
CONDITION_LABELS = {
    "normal": "Normal",
    "parkinsons": "Parkinson's",
    "stroke": "Stroke",
    "myopathic": "Myopathic",
    "cerebralpalsy": "Cerebral palsy",
}
CONDITION_COLORS = dict(zip(CONDITIONS, [BLUE, ORANGE, VERMILION, PURPLE, GREEN]))
ROLE_COLORS = {"train": BLUE, "validation": ORANGE, "test": GREEN}


@dataclass(frozen=True)
class FigureResult:
    """Paths emitted for one notebook figure."""

    stem: str
    notebook_svg: Path
    paper_svg: Path
    paper_png: Path
    paper_pdf: Path


def _configure() -> None:
    """Apply a restrained print/inline style and request vector notebook output."""

    try:
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        set_matplotlib_formats("svg")
    except Exception:
        # A non-IPython renderer still receives the exported SVG/PDF files.
        pass
    mpl.rcParams.update(
        {
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 9.5,
            "axes.labelcolor": TEXT,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": TEXT,
            "grid.color": GRID,
            "grid.alpha": 0.72,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def _clean_axis(ax: mpl.axes.Axes, *, grid_axis: str | None = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def _panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="semibold",
        va="top",
        ha="left",
    )


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _save_and_display(
    fig: mpl.figure.Figure,
    *,
    stem: str,
    title: str,
    experiment_dir: Path,
    artifact_dir: Path,
) -> FigureResult:
    """Export one figure to notebook and paper locations, then display its SVG."""

    experiment_dir = Path(experiment_dir).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    notebook_dir = artifact_dir / "figures"
    paper_dir = experiment_dir / "docs" / "figures"
    notebook_dir.mkdir(parents=True, exist_ok=True)
    paper_dir.mkdir(parents=True, exist_ok=True)
    if getattr(fig, "_brainbody_manual_layout", False):
        fig.subplots_adjust(left=0.11, right=0.95, bottom=0.10, top=0.88, hspace=0.42, wspace=0.38)
    else:
        fig.set_constrained_layout(True)

    notebook_svg = notebook_dir / f"{stem}.svg"
    paper_svg = paper_dir / f"{stem}.svg"
    paper_png = paper_dir / f"{stem}.png"
    paper_pdf = paper_dir / f"{stem}.pdf"
    common = {"bbox_inches": "tight", "pad_inches": 0.08, "facecolor": WHITE}
    fig.savefig(notebook_svg, format="svg", metadata={"Title": title}, **common)
    fig.savefig(paper_svg, format="svg", metadata={"Title": title}, **common)
    fig.savefig(paper_png, format="png", dpi=220, metadata={"Title": title}, **common)
    fig.savefig(paper_pdf, format="pdf", metadata={"Title": title}, **common)

    try:
        from IPython.display import SVG, display

        display(SVG(filename=str(notebook_svg)))
    except Exception:
        plt.show()
    finally:
        plt.close(fig)
    return FigureResult(stem, notebook_svg, paper_svg, paper_png, paper_pdf)


def _status_figure(
    *,
    heading: str,
    states: Sequence[tuple[str, str, str]],
    note: str,
) -> mpl.figure.Figure:
    """Render a truthful prerequisite/evidence state instead of a blank chart."""

    fig, ax = plt.subplots(figsize=(8.2, 2.55))
    ax.set_xlim(-0.5, len(states) - 0.5)
    ax.set_ylim(-0.52, 1.0)
    ax.axis("off")
    ax.set_title(heading, loc="left", pad=10)
    for index in range(len(states) - 1):
        ax.plot([index + 0.16, index + 0.84], [0.42, 0.42], color=GRID, lw=2, zorder=1)
    color_by_state = {"ready": GREEN, "blocked": VERMILION, "sealed": BLUE, "archived": ORANGE}
    glyph_by_state = {"ready": "OK", "blocked": "!", "sealed": "LOCK", "archived": "OLD"}
    for index, (label, state, detail) in enumerate(states):
        color = color_by_state[state]
        ax.scatter(index, 0.42, s=620, color=WHITE, edgecolor=color, linewidth=2.5, zorder=2)
        ax.text(index, 0.42, glyph_by_state[state], ha="center", va="center", color=color, fontsize=8, fontweight="bold")
        ax.text(index, 0.05, label, ha="center", va="top", fontsize=9, fontweight="semibold")
        ax.text(index, -0.16, detail, ha="center", va="top", fontsize=8, color=MUTED, wrap=True)
    fig.text(0.5, 0.015, note, ha="center", va="bottom", fontsize=8.5, color=MUTED)
    return fig


def plot_method_map() -> mpl.figure.Figure:
    fig, ax = plt.subplots(figsize=(8.4, 2.8))
    ax.axis("off")
    nodes = [
        ("Source-grouped\nvideo", "independent unit"),
        ("Pose + validity", "sensor representation"),
        ("Masked context", "visible tokens"),
        ("S-JEPA predictor", "latent targets"),
        ("Frozen evaluation", "source-level metrics"),
    ]
    x = np.arange(len(nodes))
    for index in range(len(nodes) - 1):
        ax.annotate("", (x[index + 1] - 0.34, 0.5), (x[index] + 0.34, 0.5), arrowprops={"arrowstyle": "->", "color": MUTED, "lw": 1.5})
    for index, (label, detail) in enumerate(nodes):
        color = BLUE if index < 4 else GREEN
        circle = plt.Circle((x[index], 0.5), 0.31, facecolor=LIGHT, edgecolor=color, linewidth=2)
        ax.add_patch(circle)
        ax.text(x[index], 0.52, label, ha="center", va="center", fontsize=9, fontweight="semibold")
        ax.text(x[index], 0.05, detail, ha="center", va="top", fontsize=8, color=MUTED)
    ax.text(1.5, 0.92, "fit on train sources", ha="center", color=BLUE, fontsize=8.5)
    ax.text(3.5, 0.92, "select on validation; test remains sealed", ha="center", color=GREEN, fontsize=8.5)
    ax.set_xlim(-0.55, len(nodes) - 0.45)
    ax.set_ylim(-0.3, 1.08)
    ax.set_title("From public movement video to an auditable representation test", loc="left")
    return fig


def plot_cohort_funnel(protocol_dir: Path) -> mpl.figure.Figure:
    protocol_dir = Path(protocol_dir)
    raw = pd.read_csv(protocol_dir / "raw_sequence_manifest.csv")
    public = pd.read_csv(protocol_dir / "metadata_public_sequence_manifest.csv")
    decoded = pd.read_csv(protocol_dir / "eligible_sequence_manifest.csv")
    qc = pd.read_csv(protocol_dir / "pose_qc_eligibility_outer_fold_0.csv")
    decoded = decoded.loc[_as_bool(decoded["decoded_frame_eligible"])]
    qc = qc.loc[_as_bool(qc["in_locked_manifest"]) & _as_bool(qc["pose_qc_eligible"])]
    stages = [
        ("Raw annotations", len(raw), raw["video_id"].nunique()),
        ("Metadata public", len(public), public["video_id"].nunique()),
        ("Decoded span", len(decoded), decoded["video_id"].nunique()),
        ("Pose-QC eligible", len(qc), qc["video_id"].nunique()),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25))
    labels = [row[0] for row in stages][::-1]
    for panel, field, xlabel, color in [(0, 1, "Sequences", BLUE), (1, 2, "Independent source videos", GREEN)]:
        values = [row[field] for row in stages][::-1]
        axes[panel].barh(labels, values, color=color, alpha=0.86, height=0.58, zorder=2)
        for y, value in enumerate(values):
            axes[panel].text(value, y, f"  {value:,}", va="center", ha="left", fontweight="semibold")
        axes[panel].set_xlabel(xlabel)
        axes[panel].set_xlim(0, max(values) * 1.18)
        axes[panel].set_title("Corpus attrition" if panel == 0 else "Evaluation units retained", loc="left")
        _clean_axis(axes[panel], grid_axis="x")
        _panel_label(axes[panel], chr(ord("A") + panel))
    fig.suptitle("Each gate is reported without redrawing the frozen source split", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_pose_qc(protocol_dir: Path, outer_fold: int) -> mpl.figure.Figure:
    qc = pd.read_csv(Path(protocol_dir) / f"pose_qc_eligibility_outer_fold_{outer_fold}.csv")
    qc = qc.loc[_as_bool(qc["in_locked_manifest"])].copy()
    qc["pose_qc_eligible"] = _as_bool(qc["pose_qc_eligible"])
    eligible = qc.loc[qc["pose_qc_eligible"]].copy()

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.55))
    roles = ["train", "validation", "test"]
    left = eligible.groupby(["split_role", "condition"]).size().unstack(fill_value=0).reindex(roles, fill_value=0)
    bottom = np.zeros(len(left))
    for condition in CONDITIONS:
        values = left.get(condition, pd.Series(0, index=left.index)).to_numpy()
        axes[0].bar(left.index, values, bottom=bottom, color=CONDITION_COLORS[condition], label=CONDITION_LABELS[condition], width=0.68)
        bottom += values
    for index, total in enumerate(bottom.astype(int)):
        axes[0].text(index, total, f"{total}", ha="center", va="bottom", fontweight="semibold")
    axes[0].set_ylabel("Pose-QC-eligible sequences")
    axes[0].set_title(f"Outer fold {outer_fold} composition", loc="left")
    axes[0].legend(ncol=2, loc="upper right")
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    coverage = pd.to_numeric(qc["neurologic_observed_fraction"], errors="coerce").dropna().sort_values().to_numpy()
    y = np.arange(1, len(coverage) + 1) / len(coverage)
    axes[1].plot(coverage, y, color=BLUE)
    threshold = 0.50
    axes[1].axvline(threshold, color=VERMILION, ls="--", lw=1.5, label="Predeclared QC threshold")
    axes[1].fill_between(coverage, 0, y, where=coverage >= threshold, color=BLUE, alpha=0.10)
    axes[1].set(xlabel="Observed fraction of neurologic joints", ylabel="Cumulative fraction", xlim=(0, 1.01), ylim=(0, 1.02))
    axes[1].set_title("Pose visibility distribution", loc="left")
    axes[1].legend(loc="lower right")
    _clean_axis(axes[1])
    _panel_label(axes[1], "B")
    fig.suptitle("Pose quality is audited after source assignment", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_mask_layout(mask_fraction: float = 0.60) -> mpl.figure.Figure:
    neurologic = np.array([11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
    patches = 8
    state = np.zeros((patches, 33), dtype=int)
    state[:, neurologic] = 1
    candidates = [(time, joint) for time in range(patches) for joint in neurologic]
    rng = np.random.default_rng(42)
    count = int(math.floor(len(candidates) * mask_fraction))
    for selected in rng.choice(len(candidates), size=count, replace=False):
        time, joint = candidates[int(selected)]
        state[time, joint] = 2
    cmap = ListedColormap([LIGHT, GREEN, BLUE])
    fig, ax = plt.subplots(figsize=(9.3, 3.05))
    ax.imshow(state, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=2)
    ax.set(xlabel="BlazePose joint index", ylabel="Four-frame patch", yticks=np.arange(patches), yticklabels=np.arange(1, patches + 1))
    ax.set_xticks(np.arange(0, 33, 2))
    ax.set_title("One deterministic illustration of a stochastic neurologic target mask", loc="left")
    handles = [mpl.patches.Patch(color=color, label=label) for color, label in [(LIGHT, "Context-only joint"), (GREEN, "Eligible, visible"), (BLUE, "Prediction target")]]
    ax.legend(handles=handles, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    ax.text(1.0, 1.08, f"{count}/{len(candidates)} eligible tokens masked ({100 * count / len(candidates):.1f}%)", transform=ax.transAxes, ha="right", color=MUTED)
    return fig


def plot_training_history(path: Path) -> mpl.figure.Figure:
    history = pd.read_csv(path).sort_values(["stage", "epoch"]).reset_index(drop=True)
    history["step"] = np.arange(len(history))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65))
    axes[0].plot(history["step"], history["train_loss"], color=BLUE, label="Train")
    axes[0].plot(history["step"], history["validation_loss"], color=ORANGE, label="Validation")
    best = history.loc[history.groupby("stage")["validation_loss"].idxmin()]
    axes[0].scatter(best["step"], best["validation_loss"], color=ORANGE, edgecolor=WHITE, linewidth=0.8, zorder=5, label="Stage selection")
    axes[0].set(xlabel="Epoch across curriculum", ylabel="Objective", title="Optimization and validation selection")
    axes[0].legend()
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    components = [
        ("validation_jepa", 1.0, BLUE, "JEPA"),
        ("validation_variance", 0.1, GREEN, "0.1 x variance"),
        ("validation_covariance", 0.01, PURPLE, "0.01 x covariance"),
    ]
    for column, weight, color, label in components:
        axes[1].plot(history["step"], history[column] * weight, color=color, label=label)
    axes[1].set(xlabel="Epoch across curriculum", ylabel="Weighted validation contribution", title="What drives validation loss")
    axes[1].legend()
    _clean_axis(axes[1])
    _panel_label(axes[1], "B")

    for ax in axes:
        boundaries = history.groupby("stage")["step"].agg(["min", "max"])
        for stage, row in boundaries.iterrows():
            if stage % 2:
                ax.axvspan(row["min"] - 0.5, row["max"] + 0.5, color=LIGHT, zorder=-1)
        for value in boundaries["max"].iloc[:-1]:
            ax.axvline(value + 0.5, color=GRID, lw=0.8)
    labels = history.groupby("stage").agg(x=("step", "mean"), label=("stage_name", "first"))
    short = [str(value).replace("add_", "+").replace("_only", "") for value in labels["label"]]
    axes[0].set_xticks(labels["x"], short, rotation=24, ha="right")
    axes[1].set_xticks(labels["x"], short, rotation=24, ha="right")
    fig.suptitle("Fold-local training dynamics; selection uses validation sources only", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_latent_audit(path: Path) -> mpl.figure.Figure:
    audit = json.loads(Path(path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(audit["diagnostics"])
    labels = frame["role"].str.replace("test_final", "test (final)").tolist()
    x = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(8.7, 3.35))
    bars = axes[0].bar(x, frame["mean_train_normal_distance"], color=[ROLE_COLORS.get(role.split("_")[0], BLUE) for role in frame["role"]], width=0.62)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Euclidean distance", title="Distance from train-normal reference")
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    values = frame["condition_silhouette"].to_numpy()
    axes[1].axhline(0, color=MUTED, lw=1)
    axes[1].scatter(x, values, s=75, color=[ROLE_COLORS.get(role.split("_")[0], BLUE) for role in frame["role"]], zorder=3)
    for index, value in enumerate(values):
        axes[1].text(index, value, f" {value:+.2f}", va="center", ha="left", fontsize=8.5)
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Silhouette coefficient", title="Folder-label separation diagnostic")
    axes[1].set_ylim(min(-0.35, values.min() - 0.06), max(0.15, values.max() + 0.06))
    _clean_axis(axes[1])
    _panel_label(axes[1], "B")
    fig.suptitle("Representation geometry is descriptive, not a clinical endpoint", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_readout(metrics_path: Path, contract_path: Path) -> mpl.figure.Figure:
    metrics = pd.read_csv(metrics_path)
    contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    validation = {key: value["validation_macro_f1"] for key, value in contract["selected_hyperparameters"].items()}
    lanes = metrics["lane"].tolist()
    labels = [lane.replace("_", " ") for lane in lanes]
    x = np.arange(len(lanes))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.7, 3.55))
    val = [validation[lane] for lane in lanes]
    test = metrics["test_source_macro_f1"].to_numpy()
    validation_bars = axes[0].bar(x - width / 2, val, width, color=ORANGE, label="Validation selection")
    test_bars = axes[0].bar(x + width / 2, test, width, color=GREEN, label="Outer test")
    axes[0].bar_label(validation_bars, fmt="%.2f", padding=2, fontsize=8)
    axes[0].bar_label(test_bars, fmt="%.2f", padding=2, fontsize=8)
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Macro-F1", ylim=(0, 1.05), title="Validation-to-test transfer")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].legend()
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    balanced = metrics["test_source_balanced_accuracy"].to_numpy()
    bars = axes[1].barh(labels[::-1], balanced[::-1], color=BLUE, height=0.55)
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    axes[1].set(xlabel="Balanced accuracy", xlim=(0, 1.05), title="Source-equal outer-test score")
    _clean_axis(axes[1], grid_axis="x")
    _panel_label(axes[1], "B")
    fig.suptitle("Readout evaluation includes representation and sensor baselines", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_temporal_readout(path: Path) -> mpl.figure.Figure:
    frame = pd.read_csv(path)
    targets = list(dict.fromkeys(frame["target"]))
    lanes = list(dict.fromkeys(frame["lane"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(3.2 * len(targets), 3.65), squeeze=False)
    axes = axes[0]
    for panel, target in enumerate(targets):
        sub = frame.loc[frame["target"].eq(target)].set_index("lane").reindex(lanes)
        values = sub["source_level_r2"].to_numpy()
        colors = [BLUE if value >= 0 else VERMILION for value in values]
        y = np.arange(len(lanes))
        axes[panel].axvline(0, color=MUTED, lw=1)
        bars = axes[panel].barh(y, values, color=colors, height=0.55)
        axes[panel].bar_label(
            bars,
            labels=[f"{value:+.2f}" for value in values],
            padding=3,
            fontsize=8,
        )
        axes[panel].set_yticks(y, [lane.replace("_", " ") for lane in lanes] if panel == 0 else [])
        axes[panel].set_xlabel("Outer-test source-level $R^2$")
        axes[panel].set_title(target.replace("_", " ").title(), loc="left")
        axes[panel].margins(x=0.18)
        _clean_axis(axes[panel], grid_axis="x")
        _panel_label(axes[panel], chr(ord("A") + panel))
    fig.suptitle("Temporal probes are tuned on validation sources and scored once on test sources", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_anchor_drift(path: Path) -> mpl.figure.Figure:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    frame = pd.DataFrame(report["development_drift"])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.45))
    for role, color in [("train", BLUE), ("validation", ORANGE)]:
        sub = frame.loc[frame["role"].eq(role)].sort_values("stage")
        axes[0].plot(sub["stage"], sub["source_equal_anchor_cosine"], marker="o", color=color, label=role.title())
    names = frame.drop_duplicates("stage").sort_values("stage")
    axes[0].set(xticks=names["stage"], xticklabels=[name.replace("add_", "+").replace("_only", "") for name in names["stage_name"]], ylabel="Cosine to Stage-0 normal anchor", title="Development-only retention curve")
    axes[0].tick_params(axis="x", rotation=24)
    axes[0].legend()
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    final = report["final_test"]
    values = [final["validation_source_equal_anchor_cosine"], final["test_source_equal_anchor_cosine"]]
    bars = axes[1].bar(["Validation\n(selection)", "Test\n(final once)"], values, color=[ORANGE, GREEN], width=0.58)
    axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8.5)
    axes[1].set(ylim=(0, 1.05), ylabel="Source-equal anchor cosine", title=f"Selected: {final['selected_objective']}")
    _clean_axis(axes[1])
    _panel_label(axes[1], "B")
    fig.suptitle("Consolidation selection and the final normal-source retention estimate", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_predictive_surprise(path: Path) -> mpl.figure.Figure:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    horizon = pd.DataFrame(report["horizon_selection"])
    auroc = pd.DataFrame(report["auroc"])
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    axes[0].plot(horizon["horizon"], horizon["validation_future_cosine"], marker="o", color=BLUE, label="Future predictor")
    axes[0].plot(horizon["horizon"], horizon["validation_copy_last_cosine"], marker="s", color=ORANGE, label="Copy-last baseline")
    selected = int(report["selected_horizon"])
    axes[0].axvline(selected, color=GREEN, ls="--", lw=1.5, label=f"Selected h={selected}")
    axes[0].set(xlabel="Future horizon (patches)", ylabel="Validation cosine", title="Validation-only horizon selection")
    axes[0].legend()
    _clean_axis(axes[0])
    _panel_label(axes[0], "A")

    labels = [CONDITION_LABELS.get(value, value) for value in auroc["condition"]]
    bars = axes[1].barh(labels[::-1], auroc["source_level_auroc"].to_numpy()[::-1], color=BLUE, height=0.55)
    axes[1].axvline(0.5, color=MUTED, ls="--", lw=1, label="Chance")
    axes[1].bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    axes[1].set(xlabel="Outer-test source AUROC", xlim=(0, 1.02), title="Predictive-surprise discrimination")
    axes[1].legend(loc="lower right")
    _clean_axis(axes[1], grid_axis="x")
    _panel_label(axes[1], "B")
    fig.suptitle("Only a separately future-mask-trained predictor supports this evaluation", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def plot_protocol_execution_overview(
    *,
    history_path: Path,
    readout_metrics_path: Path,
    drift_report_path: Path,
    temporal_path: Path,
) -> mpl.figure.Figure:
    """Condense the worked fold into one page-efficient paper figure."""

    history = pd.read_csv(history_path).sort_values(["stage", "epoch"]).reset_index(drop=True)
    history["step"] = np.arange(len(history))
    readout = pd.read_csv(readout_metrics_path)
    drift = json.loads(Path(drift_report_path).read_text(encoding="utf-8"))
    drift_frame = pd.DataFrame(drift["development_drift"])
    temporal = pd.read_csv(temporal_path)

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.25))
    ax = axes[0, 0]
    ax.plot(history["step"], history["train_loss"], color=BLUE, label="Train")
    ax.plot(history["step"], history["validation_loss"], color=ORANGE, label="Validation")
    best = history.loc[history.groupby("stage")["validation_loss"].idxmin()]
    ax.scatter(best["step"], best["validation_loss"], color=ORANGE, edgecolor=WHITE, linewidth=0.8, zorder=4)
    for value in history.groupby("stage")["step"].max().iloc[:-1]:
        ax.axvline(value + 0.5, color=GRID, lw=0.8)
    ax.set(xlabel="Epoch across curriculum", ylabel="Objective", title="Validation-selected curriculum")
    ax.legend()
    _clean_axis(ax)
    _panel_label(ax, "A")

    ax = axes[0, 1]
    lanes = readout["lane"].str.replace("_", " ").tolist()
    values = readout["test_source_macro_f1"].to_numpy()
    bars = ax.barh(lanes[::-1], values[::-1], color=[BLUE, ORANGE, GREEN][::-1], height=0.56)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set(xlabel="Outer-test source macro-F1", xlim=(0, max(0.55, values.max() + 0.1)), title="Readout and sensor controls")
    _clean_axis(ax, grid_axis="x")
    _panel_label(ax, "B")

    ax = axes[1, 0]
    for role, color in [("train", BLUE), ("validation", ORANGE)]:
        sub = drift_frame.loc[drift_frame["role"].eq(role)].sort_values("stage")
        ax.plot(sub["stage"], sub["source_equal_anchor_cosine"], marker="o", color=color, label=role.title())
    final = drift["final_test"]
    last_stage = int(drift_frame["stage"].max())
    ax.scatter(last_stage, final["test_source_equal_anchor_cosine"], marker="*", s=120, color=GREEN, label="Test normal (final once)", zorder=4)
    ax.set(xticks=sorted(drift_frame["stage"].unique()), xlabel="Curriculum stage", ylabel="Cosine to Stage-0 anchor", ylim=(0.65, 1.02), title="Normal-anchor retention")
    ax.legend(loc="lower left")
    _clean_axis(ax)
    _panel_label(ax, "C")

    ax = axes[1, 1]
    matrix = temporal.pivot(index="lane", columns="target", values="source_level_r2")
    lane_order = [lane for lane in ["A_mean_std", "B_signed_moment", "C_time_bins"] if lane in matrix.index]
    target_order = [target for target in ["peak_phase", "energy_ratio", "phase_lag"] if target in matrix.columns]
    matrix = matrix.reindex(index=lane_order, columns=target_order)
    image = ax.imshow(matrix.to_numpy(), cmap="RdBu", vmin=-0.35, vmax=0.35, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix.iat[row, column]
            ax.text(column, row, f"{value:+.2f}", ha="center", va="center", color=TEXT, fontsize=8.5, fontweight="bold")
    ax.set_xticks(np.arange(len(target_order)), [value.replace("_", " ") for value in target_order], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(lane_order)), [value.replace("_", " ") for value in lane_order])
    ax.set_title("Outer-test temporal-probe $R^2$", loc="left")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    colorbar.set_label("Source-level $R^2$")
    _panel_label(ax, "D")

    fig.suptitle(
        "Worked protocol execution: outer fold 0, seed 42 (not a cross-fold estimate)",
        x=0.01,
        ha="left",
        fontsize=12,
        fontweight="semibold",
    )
    # Matplotlib 3.11 can divide by zero when constrained layout is enabled
    # after adding a colorbar to a 2x2 grid. This explicit paper layout is
    # deterministic and is visually verified after export.
    fig._brainbody_manual_layout = True
    return fig


def render_protocol_execution_overview(
    *,
    experiment_dir: str | Path,
    artifact_dir: str | Path,
    outer_fold: int = 0,
    model_seed: int = 42,
    objective: str = "jepa_vicreg",
) -> FigureResult:
    """Export the compact paper overview from current protocol-v2 artifacts."""

    _configure()
    experiment_dir = Path(experiment_dir).resolve()
    artifact_dir = Path(artifact_dir).resolve()
    fold_dir = artifact_dir / "fold_evaluation" / f"outer_fold_{outer_fold}"
    fig = plot_protocol_execution_overview(
        history_path=artifact_dir / f"sjepa_outer_fold_{outer_fold}_seed_{model_seed}_{objective}_history.csv",
        readout_metrics_path=artifact_dir / f"readout_outer_fold_{outer_fold}_seed_{model_seed}_{objective}_metrics.csv",
        drift_report_path=fold_dir / f"normal_anchor_drift_seed_{model_seed}.json",
        temporal_path=fold_dir / f"temporal_readout_seed_{model_seed}.csv",
    )
    return _save_and_display(
        fig,
        stem="bbfm_protocol_execution",
        title="Worked protocol-v2 execution overview",
        experiment_dir=experiment_dir,
        artifact_dir=artifact_dir,
    )


def plot_archived_laterality(path: Path, variant: str) -> mpl.figure.Figure:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    cohort = report["primary_cohort"]
    lanes = cohort["lanes"]
    names = list(lanes)
    r2 = [lanes[name]["r2"] for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.55))
    labels = [name.replace("_", " ") for name in names]
    bars = axes[0].barh(labels[::-1], r2[::-1], color=ORANGE, height=0.55)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)
    axes[0].set(xlabel="$R^2$", title="Archived transductive decodability")
    _clean_axis(axes[0], grid_axis="x")
    _panel_label(axes[0], "A")
    slopes = cohort.get("mirror_slopes", {})
    if not slopes and "mirror" in cohort:
        slopes = {names[0]: cohort["mirror"].get("slope", np.nan)}
    slope_names = list(slopes)
    slope_values = [slopes[name] for name in slope_names]
    axes[1].axvline(-1, color=GREEN, ls="--", lw=1.4, label="Exact antisymmetry")
    axes[1].barh([name.replace("_", " ") for name in slope_names][::-1], slope_values[::-1], color=PURPLE, height=0.55)
    axes[1].set(xlabel="Mirror slope", title="Reflection behavior")
    axes[1].legend(loc="lower right")
    _clean_axis(axes[1], grid_axis="x")
    _panel_label(axes[1], "B")
    fig.suptitle(f"{variant}: archived diagnostic, not protocol-v2 paper evidence", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    return fig


def render_notebook_summary(
    notebook_id: str,
    *,
    experiment_dir: str | Path,
    artifact_dir: str | Path | None = None,
    outer_fold: int = 0,
    model_seed: int = 42,
    objective: str = "jepa_vicreg",
) -> FigureResult:
    """Render and export the evidence-aware summary assigned to a notebook."""

    _configure()
    experiment_dir = Path(experiment_dir).resolve()
    if artifact_dir is None:
        mode = os.getenv("GAVD_MODE", "real").strip().lower()
        artifact_dir = experiment_dir / "work" / "artifacts" / mode
    artifact_dir = Path(artifact_dir).resolve()
    protocol_dir = artifact_dir / "evaluation_protocol"
    notebook_id = str(notebook_id).lower()

    stem: str
    title: str
    if notebook_id == "00":
        fig, stem, title = plot_method_map(), "bbfm_method_map", "Auditable S-JEPA method map"
    elif notebook_id == "01":
        fig, stem, title = plot_cohort_funnel(protocol_dir), "bbfm_data_funnel", "GAVD data-validity funnel"
    elif notebook_id == "02":
        fig, stem, title = plot_pose_qc(protocol_dir, outer_fold), "bbfm_pose_qc", "Pose quality and split composition"
    elif notebook_id == "03":
        fig, stem, title = plot_mask_layout(), "bbfm_mask_geometry", "Neurologic masking geometry"
    elif notebook_id == "04":
        path = artifact_dir / f"sjepa_outer_fold_{outer_fold}_seed_{model_seed}_{objective}_history.csv"
        if path.is_file():
            fig = plot_training_history(path)
        else:
            fig = _status_figure(
                heading="Training telemetry unavailable",
                states=[("Protocol", "ready", "source split frozen"), ("Fold training", "blocked", "history file missing"), ("Outer test", "sealed", "never opened for training")],
                note="Run notebook 04 for this fold, seed, and objective to create learning curves.",
            )
        stem, title = "bbfm_training_dynamics", "Fold-local S-JEPA training dynamics"
    elif notebook_id == "05":
        path = artifact_dir / f"latent_audit_outer_fold_{outer_fold}_seed_{model_seed}_{objective}.json"
        if path.is_file():
            fig = plot_latent_audit(path)
        else:
            fig = _status_figure(
                heading="Latent audit unavailable",
                states=[("Checkpoint", "ready", "fold-local encoder"), ("Latent audit", "blocked", "audit file missing"), ("Claim", "sealed", "no geometry result")],
                note="Run notebook 05 after notebook 04; a missing audit is never plotted as zero.",
            )
        stem, title = "bbfm_latent_geometry", "Fold-local latent geometry diagnostic"
    elif notebook_id == "06":
        base = f"readout_outer_fold_{outer_fold}_seed_{model_seed}_{objective}"
        metrics_path = artifact_dir / f"{base}_metrics.csv"
        contract_path = artifact_dir / f"{base}_contract.json"
        if metrics_path.is_file() and contract_path.is_file():
            fig = plot_readout(metrics_path, contract_path)
        else:
            fig = _status_figure(
                heading="Readout evaluation unavailable",
                states=[("Encoder", "ready", "fold-local checkpoint"), ("Validation", "blocked", "readout contract missing"), ("Outer test", "sealed", "no score claimed")],
                note="Run notebook 06 to tune on validation sources and score test sources once.",
            )
        stem, title = "bbfm_readout_evaluation", "Source-level readout evaluation"
    elif notebook_id == "07":
        path = artifact_dir / "fold_evaluation" / f"outer_fold_{outer_fold}" / f"temporal_readout_seed_{model_seed}.csv"
        if path.is_file():
            fig = plot_temporal_readout(path)
        else:
            fig = _status_figure(
                heading="Temporal diagnostic unavailable",
                states=[("Checkpoint", "ready", "fit without test"), ("Temporal probes", "blocked", "result file missing"), ("Outer test", "sealed", "no temporal claim")],
                note="The chart appears only after validation tuning and one source-level test pass.",
            )
        stem, title = "bbfm_temporal_readout_v2", "Protocol-v2 temporal readout diagnostic"
    elif notebook_id == "08":
        path = artifact_dir / "fold_evaluation" / f"outer_fold_{outer_fold}" / f"normal_anchor_drift_seed_{model_seed}.json"
        if path.is_file():
            fig = plot_anchor_drift(path)
        else:
            fig = _status_figure(
                heading="Retention evaluation unavailable",
                states=[("Stage lineage", "ready", "hash-checked checkpoints"), ("Validation selection", "blocked", "drift report missing"), ("Test normal", "sealed", "not opened")],
                note="No retention curve is inferred from archived transductive AnchorGuard artifacts.",
            )
        stem, title = "bbfm_anchor_drift_v2", "Fold-local normal-anchor retention"
    elif notebook_id == "09":
        path = artifact_dir / "fold_evaluation" / f"outer_fold_{outer_fold}" / f"predictive_surprise_seed_{model_seed}.json"
        if path.is_file():
            fig = plot_predictive_surprise(path)
        else:
            future = artifact_dir / "checkpoints" / f"sjepa_outer_fold_{outer_fold}_seed_{model_seed}_jepa_vicreg_future_mask.pt"
            spatial = artifact_dir / "checkpoints" / f"sjepa_outer_fold_{outer_fold}_seed_{model_seed}_{objective}.pt"
            fig = _status_figure(
                heading="Predictive-surprise experiment is correctly blocked",
                states=[("Spatial checkpoint", "ready" if spatial.is_file() else "blocked", "spatial infilling only"), ("Future-mask checkpoint", "ready" if future.is_file() else "blocked", "separate training required"), ("Outer test", "sealed", "no forecasting score")],
                note="A spatial-infilling predictor is not relabelled as a future predictor.",
            )
        stem, title = "bbfm_predictive_surprise_v2", "Predictive-surprise evidence status"
    elif notebook_id == "05a":
        fig = plot_archived_laterality(artifact_dir / "idea5_signed_laterality_result_hardened.json", "Signed laterality probe")
        stem, title = "bbfm_archived_laterality", "Archived signed-laterality diagnostic"
    elif notebook_id == "05c":
        fig = plot_archived_laterality(artifact_dir / "idea9_equivariant_readout_result.json", "Equivariant readout")
        stem, title = "bbfm_archived_equivariant_readout", "Archived equivariant-readout diagnostic"
    elif notebook_id == "05d":
        fig = plot_archived_laterality(artifact_dir / "idea9_equivariant_encoder_result.json", "Equivariant encoder")
        stem, title = "bbfm_archived_equivariant_encoder", "Archived equivariant-encoder diagnostic"
    else:
        raise ValueError(f"No scientific visual is registered for notebook {notebook_id!r}")

    return _save_and_display(
        fig,
        stem=stem,
        title=title,
        experiment_dir=experiment_dir,
        artifact_dir=artifact_dir,
    )


__all__ = [
    "FigureResult",
    "render_notebook_summary",
    "plot_anchor_drift",
    "plot_cohort_funnel",
    "plot_latent_audit",
    "plot_mask_layout",
    "plot_method_map",
    "plot_pose_qc",
    "plot_predictive_surprise",
    "plot_protocol_execution_overview",
    "plot_readout",
    "plot_temporal_readout",
    "plot_training_history",
    "render_protocol_execution_overview",
]
