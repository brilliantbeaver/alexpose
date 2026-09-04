#!/usr/bin/env python
"""Generate the three current figures used by the BrainBodyFM draft.

Only fingerprint-matched, non-augmented artifacts are accepted. Historical
AnchorGuard, Lane C, and predictive-surprise artifacts are intentionally not
read by this script.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "work" / "artifacts" / "real"
OUT = ROOT / "docs" / "figures"
EXPECTED_FP = "7d13841aceac9eda843d43ca8434193e294d2fa10a48b6c6d21f6413a6e457e2"
EXPECTED_SHA = "64008d77689cefa4beb51a0dcf5ed6cae743454134c163e9087f66510af4e7ad"
EXPECTED_FILES = {
    "classifier_contract.json": "bd7ee48accb7f31e51b2fffe6e8789a20fcf487ab68db18b45f21615bb8f45e0",
    "classifier_pose_coverage.csv": "873c94c5372f764d5d51e67a21dea8b384324014d071c4b170302938b33aec8e",
    "curriculum_stage_summary.csv": "4578efd4555d25e4d7bace40b57f7193637bf2ac51aadce1e842430b8f8cc63d",
    "temporal_readout_results.json": "1759f97d754656c7f0880237e0ea5ed3e17686b4f204014645805d791337c9b1",
}

NAVY = "#17324D"
BLUE = "#3977A8"
TEAL = "#3D8B7D"
ORANGE = "#D97745"
RED = "#B94E48"
PURPLE = "#7562A8"
INK = "#25384A"
MUTED = "#64788A"
PAPER = "#F7F5EF"
GRID = "#D9E0E6"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Figure contract failed: {message}")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{suffix}", bbox_inches="tight", dpi=220)
    plt.close(fig)
    print(f"wrote {name}.svg/.pdf/.png")


def load_current() -> tuple[pd.DataFrame, dict]:
    for name, expected in EXPECTED_FILES.items():
        require(file_sha256(ART / name) == expected, f"{name} content hash changed")
    require(file_sha256(ART / "sjepa_curriculum_final.pt") == EXPECTED_SHA,
            "checkpoint file SHA mismatch")
    contract = json.loads((ART / "classifier_contract.json").read_text())
    require(contract["checkpoint_fingerprint"] == EXPECTED_FP, "wrong checkpoint fingerprint")
    require(contract.get("include_augmented_normal") is False, "added-normal data is enabled")
    require(contract.get("encoder_checkpoint") == "sjepa_curriculum_final.pt", "wrong checkpoint")
    require(contract.get("feature_count") == 384, "wrong frozen-feature width")
    require(contract.get("conditions_seen") == ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"],
            "wrong condition order")
    coverage = pd.read_csv(ART / "classifier_pose_coverage.csv")
    require(len(coverage) == 626 and coverage["sequence_id"].nunique() == 626,
            "coverage cohort is not 626 unique rows")
    require(coverage["video_id"].nunique() == 93, "coverage cohort is not 93 videos")
    require(coverage["condition"].value_counts().to_dict() == {
        "normal": 270, "parkinsons": 41, "stroke": 74,
        "myopathic": 183, "cerebralpalsy": 58,
    }, "coverage condition counts changed")
    temporal = json.loads((ART / "temporal_readout_results.json").read_text())
    require(temporal["experiment_fingerprint"] == EXPECTED_FP, "temporal probe fingerprint mismatch")
    require(temporal["checkpoint_file_sha256"] == EXPECTED_SHA, "temporal probe SHA mismatch")
    require(temporal["corpus"]["total_sequences"] == 626, "temporal probe is not the 626-row run")
    summary = pd.read_csv(ART / "curriculum_stage_summary.csv")
    require(summary["stage"].tolist() == [1, 2, 3, 4], "unexpected curriculum stages")
    require(summary["checkpoint"].tolist() == [
        "sjepa_stage_01_parkinsons.pt", "sjepa_stage_02_stroke.pt",
        "sjepa_stage_03_myopathic.pt", "sjepa_stage_04_cerebralpalsy.pt",
    ], "unexpected stage checkpoint names")
    expected = np.array([0.7001505494, 0.5021127462, 0.3962131739, 0.2966379523])
    require(np.allclose(summary["normal_anchor_cosine"], expected, atol=1e-9, rtol=0),
            "anchor curve changed")
    return summary, temporal


def box(ax, x: float, y: float, w: float, h: float, text: str, color: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.15",
            facecolor="white", edgecolor=color, linewidth=1.5,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)


def arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(
        FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                        color=MUTED, linewidth=1.3)
    )


def fig_overview() -> None:
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 38)
    ax.axis("off")
    ax.text(50, 35.5, "Current GAVD-only experiment and claim boundary", ha="center",
            fontsize=12, weight="bold", color=NAVY)

    box(ax, 2, 22, 20, 8, "GAVD available\n642 sequences\n94 videos", BLUE)
    box(ax, 27, 22, 20, 8, "Pose coverage ≥ 0.50\n626 sequences\n93 videos", TEAL)
    box(ax, 52, 22, 20, 8, "S-JEPA-inspired variant\n270 normal + 4 stages", PURPLE)
    box(ax, 77, 22, 21, 8, "Raw matched anchor\n0.700 → 0.297", ORANGE)
    for left in (22, 47, 72):
        arrow(ax, left, 26, left + 5, 26)

    box(ax, 8, 6, 24, 8, "Supported\nraw coordinate drift\nfor seed 42", TEAL)
    box(ax, 38, 6, 24, 8, "Descriptive only\nin-corpus geometry\nand readouts", ORANGE)
    box(ax, 68, 6, 24, 8, "Not established\nforgetting, generalization,\nor clinical value", RED)
    arrow(ax, 87.5, 22, 80, 14)
    ax.text(50, 1.7, "The optional added-normal dataset is off. Later stages use a label-aware group loss.",
            ha="center", fontsize=8, color=MUTED)
    save(fig, "bbfm_overview")


def fig_drift(summary: pd.DataFrame) -> None:
    stages = np.arange(5)
    anchor = np.r_[1.0, summary["normal_anchor_cosine"].to_numpy()]
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.plot(stages, anchor, color=ORANGE, marker="o", linewidth=2.3, markersize=6)
    for x, value in zip(stages, anchor):
        ax.annotate(f"{value:.3f}", (x, value), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.set_xticks(stages, ["Stage 0\nnormal", "+ Parkinson's", "+ stroke", "+ myopathic", "+ CP"])
    ax.set_ylim(0.2, 1.08)
    ax.set_ylabel("Mean matched normal-sequence cosine")
    ax.set_title("Raw normal coordinates move across the curriculum", weight="bold", color=NAVY)
    ax.grid(axis="y", color=GRID, alpha=0.8)
    ax.text(0.02, 0.03, "One seed; encoder-exposed rows; not a forgetting score",
            transform=ax.transAxes, fontsize=8, color=RED)
    save(fig, "bbfm_drift_curve")


def fig_readout(temporal: dict) -> None:
    labels = ["Peak phase", "Phase lag", "Energy ratio"]
    keys = ["peak_phase", "phase_lag", "energy_ratio"]
    improvement = np.array([temporal["decision"][key]["relative_pooled_improvement"] for key in keys]) * 100
    consistency = np.array([temporal["decision"][key]["sign_consistent_source_fraction"] for key in keys]) * 100
    require(not any(temporal["decision"][key]["passes_10pct_and_75pct"] for key in keys),
            "paper verdict must be updated because a target now passes")

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    x = np.arange(3)
    axes[0].bar(x, improvement, color=[TEAL if value >= 0 else RED for value in improvement])
    axes[0].axhline(10, color=ORANGE, linestyle="--", linewidth=1.4)
    axes[0].set_xticks(x, labels, rotation=12, ha="right")
    axes[0].set_ylabel("Relative MAE improvement (%)")
    axes[0].set_title("Pooled error")
    axes[0].set_ylim(-3.2, 11.2)
    axes[0].text(0.02, 10.25, "10% gate", color=ORANGE, fontsize=8)
    for i, value in enumerate(improvement):
        axes[0].text(i, value + (0.45 if value >= 0 else -0.35), f"{value:.1f}%",
                     ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    axes[1].bar(x, consistency, color=BLUE)
    axes[1].axhline(75, color=ORANGE, linestyle="--", linewidth=1.4)
    axes[1].set_xticks(x, labels, rotation=12, ha="right")
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Sources improved (%)")
    axes[1].set_title("Source-level consistency")
    axes[1].text(0.02, 77, "75% gate", color=ORANGE, fontsize=8)
    for i, value in enumerate(consistency):
        axes[1].text(i, value + 2, f"{value:.1f}%", ha="center", fontsize=8)

    fig.suptitle("The temporal-moment readout does not clear either decision gate",
                 fontsize=11, weight="bold", color=NAVY)
    for ax in axes:
        ax.grid(axis="y", color=GRID, alpha=0.7)
        ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "bbfm_readout_sweep")


def main() -> None:
    summary, temporal = load_current()
    fig_overview()
    fig_drift(summary)
    fig_readout(temporal)


if __name__ == "__main__":
    main()
