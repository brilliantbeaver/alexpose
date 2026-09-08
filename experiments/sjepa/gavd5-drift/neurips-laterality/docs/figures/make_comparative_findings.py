"""Recompute the Notebook 12 findings and draw a vector summary, without training.

Run from the repository root with the project's Python environment. Only
aggregate evidence enters the public assets; per-clip predictions stay private.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SUITE = HERE.parents[1]
ARMS = ("gait_targets", "all_landmark_targets")
SEEDS = tuple(range(42, 47))
FOLDS = tuple(range(5))
REPRESENTATIONS = (
    "pretrained_teacher", "pretrained_online", "initial_online",
    "direct_pose", "training_mean",
)


def verified_tables():
    """Require a single complete saved grid and reproduce its intact-input scores."""
    candidates = sorted((SUITE / "artifacts/comparative_masking/summaries").glob("*/manifest.json"))
    if len(candidates) != 1:
        raise ValueError("Select the intended result explicitly when multiple summaries exist")
    path = candidates[0]
    manifest = json.loads(path.read_text())
    if manifest.get("complete") is not True:
        raise ValueError("The saved summary is incomplete")
    for name, expected in manifest["files"].items():
        with (path.parent / name).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Summary integrity check failed for {name}")
    predictions = pd.read_csv(path.parent / "predictions.csv")
    saved = pd.read_csv(path.parent / "summary.csv")
    interval = pd.read_csv(path.parent / "paired_intervals.csv")
    if predictions.synthetic.any() or len(predictions) != 125_000:
        raise ValueError("Expected the complete retained real-data comparison")
    keys = ["condition", "representation", "observation", "seed", "sequence_id"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate predictions")
    if predictions.groupby("source_id").fold.nunique().max() != 1:
        raise ValueError("A source crosses outer test folds")
    intact = predictions.loc[predictions.observation == "unaltered"].copy()
    if (not intact.available.all()
            or intact.groupby(["condition", "representation"]).checkpoint.nunique().max() != 1):
        raise ValueError("Intact predictions have missing cases or mixed checkpoints")
    rows = []
    cohort = None
    for (condition, representation, seed), group in intact.groupby(
        ["condition", "representation", "seed"]
    ):
        if (condition not in ARMS or representation not in REPRESENTATIONS
                or seed not in SEEDS or len(group) != 625
                or group.source_id.nunique() != 93 or set(group.fold) != set(FOLDS)):
            raise ValueError("Unexpected comparison coverage")
        observed = group[["sequence_id", "source_id", "fold", "target"]].sort_values("sequence_id").reset_index(drop=True)
        if cohort is None:
            cohort = observed
        else:
            pd.testing.assert_frame_equal(cohort, observed)
        counts = group.groupby("source_id").source_id.transform("size")
        weights = 1.0 / counts.to_numpy()
        target = group.target.to_numpy()
        error = target - group.prediction.to_numpy()
        mean = np.average(target, weights=weights)
        variance = np.average((target - mean) ** 2, weights=weights)
        rows.append({"condition": condition, "representation": representation,
                     "seed": int(seed),
                     "r2": float(1 - np.average(error ** 2, weights=weights) / variance),
                     "mae": float(np.average(abs(error), weights=weights))})
    scores = pd.DataFrame(rows)
    if len(scores) != len(ARMS) * len(REPRESENTATIONS) * len(SEEDS):
        raise ValueError("Missing arm, representation, or seed")
    for (condition, representation), group in scores.groupby(["condition", "representation"]):
        reference = saved.loc[(saved.condition == condition)
                              & (saved.representation == representation)
                              & (saved.observation == "unaltered")].iloc[0]
        np.testing.assert_allclose([group.r2.mean(), group.mae.mean()],
                                   [reference.mean_r2, reference.mean_mae], atol=1e-12, rtol=0)
    for representation in ("initial_online", "direct_pose", "training_mean"):
        controls = [intact.loc[(intact.condition == arm) & (intact.representation == representation),
                              ["seed", "sequence_id", "prediction"]].sort_values(
                                  ["seed", "sequence_id"]).reset_index(drop=True) for arm in ARMS]
        pd.testing.assert_frame_equal(*controls)
    if len(interval) != 1 or interval.iloc[0].subtraction != "all_landmark_targets minus gait_targets":
        raise ValueError("Unexpected interval comparison")
    teacher = scores.loc[scores.representation == "pretrained_teacher"]
    difference = teacher.loc[teacher.condition == ARMS[1], "r2"].mean() - teacher.loc[teacher.condition == ARMS[0], "r2"].mean()
    np.testing.assert_allclose(difference, interval.iloc[0].difference, atol=1e-12, rtol=0)
    return scores, saved, interval, path


def draw(scores, interval):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(8.0, 5.2))
    ax = fig.add_axes([0.35, 0.47, 0.60, 0.39])
    rows = [
        ("gait_targets", "pretrained_teacher", "Gait-focused masking targets", "#28769c"),
        ("all_landmark_targets", "pretrained_teacher", "All-landmark masking targets", "#25816e"),
        ("gait_targets", "initial_online", "Encoder without pretraining", "#6a7180"),
        ("gait_targets", "direct_pose", "Direct pose summaries", "#8b5b39"),
        ("gait_targets", "training_mean", "Training-video mean", "#8c8c8c"),
    ]
    for y, (arm, representation, label, color) in zip(range(4, -1, -1), rows):
        group = scores.loc[(scores.condition == arm) & (scores.representation == representation)].sort_values("seed")
        if representation not in ("direct_pose", "training_mean"):
            ax.scatter(group.r2, y + np.linspace(0.07, 0.19, 5), s=27,
                       color=color, alpha=.8, linewidths=.4, edgecolors="white", zorder=3)
        ax.scatter(group.r2.mean(), y - .13, marker="D", s=48, color=color,
                   edgecolors="white", linewidths=.5, zorder=4)
        ax.text(.157, y - .13, f"{group.r2.mean():.3f}".replace("-", "−"),
                ha="right", va="center", fontsize=10, color="#263445")
    ax.axvline(0, color="#9aa2ab", lw=.9)
    ax.set(yticks=range(4, -1, -1), yticklabels=[row[2] for row in rows],
           xlim=(-.092, .166), ylim=(-.6, 4.55), xticks=[-.05, 0, .05, .10, .15],
           xlabel="Movement-prediction R²  ·  Higher values are better")
    ax.tick_params(axis="y", length=0, pad=13)
    ax.grid(axis="x", alpha=.12)
    ax.spines[["top", "left", "right"]].set_visible(False)
    ax.spines["bottom"].set_color("#bcc3cb")
    fig.text(.04, .955, "What did masking pretraining improve?", fontsize=15, weight="bold", color="#243243")
    fig.text(.04, .905, "625 clips · 93 source videos · source-held-out evaluation · five seeds", fontsize=10, color="#526170")
    fig.text(.04, .335, "Neither trained teacher improves on the initial encoder.", fontsize=11, color="#243243")
    fig.text(.04, .29, "Diamonds show means; circles show individual training seeds.", fontsize=9, color="#526170")
    bx = fig.add_axes([.45, .10, .49, .075])
    entry = interval.iloc[0]
    bx.axvline(0, color="#929ca8", lw=.9)
    bx.errorbar(entry.difference, 0, xerr=[[entry.difference - entry.lower_95],
                                        [entry.upper_95 - entry.difference]],
                fmt="D", color="#25816e", capsize=4, ms=5, lw=1.7)
    bx.set(xlim=(-.04, .07), ylim=(-1, 1), yticks=[], xticks=[-.02, 0, .02, .04, .06])
    bx.tick_params(axis="x", labelsize=9)
    bx.spines[["top", "right", "left"]].set_visible(False)
    bx.spines["bottom"].set_color("#bcc3cb")
    fig.text(.04, .177, "Effect of broader targets", fontsize=10, weight="bold", color="#243243")
    fig.text(.04, .130, "All-landmark R² minus gait-focused R²", fontsize=9, color="#526170")
    fig.text(.04, .071, "The 95% source interval includes zero.", fontsize=9, color="#526170")
    fig.text(.45, .025, "Paired difference, conditional on the fitted models", fontsize=9, color="#526170")
    for extension in ("svg", "pdf"):
        fig.savefig(HERE / f"tutorial_comparative_masking_results.{extension}", facecolor="white")
    plt.close(fig)


def main():
    scores, saved, interval, path = verified_tables()
    summary = {
        "evidence": "Completed Notebook 12 real-data comparison, reviewed 8 September 2026",
        "coverage": {"clips": 625, "sources": 93, "folds": list(FOLDS), "seeds": list(SEEDS),
                     "trained_encoders": 50, "updates_per_encoder": 1200},
        "aggregation": "Pool held-out folds within each seed, weight source videos equally, then average seed scores",
        "per_seed": scores.to_dict(orient="records"),
        "summary": saved.to_dict(orient="records"),
        "teacher_mask_contrast": interval.to_dict(orient="records")[0],
        "saved_summary": str(path.parent.relative_to(SUITE)),
        "interpretation": "No established masking advantage or learned-over-initial improvement for this endpoint",
    }
    (HERE / "tutorial_comparative_masking_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    draw(scores, interval)
    print("Verified the complete saved comparison; generated vector figure and aggregate evidence.")


if __name__ == "__main__":
    main()
