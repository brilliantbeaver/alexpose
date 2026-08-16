"""Build publication figures from the completed, fingerprinted real run.

Every measured figure reads the same classifier contract and checkpoint variant.
The script refuses incomplete or legacy artifacts so old results cannot be
silently mixed with the current five-stage curriculum.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "cache" / "artifacts" / "real"
FIGURES = Path(__file__).resolve().parent / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "navy": "#17324D",
    "blue": "#3977A8",
    "teal": "#3D8B7D",
    "green": "#72A66A",
    "gold": "#D9A441",
    "orange": "#D97745",
    "red": "#B94E48",
    "purple": "#7562A8",
    "ink": "#25384A",
    "muted": "#64788A",
    "paper": "#F7F5EF",
    "grid": "#D9E0E6",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "legend.fontsize": 7,
        "figure.dpi": 180,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)


def load_contract() -> dict:
    path = ARTIFACTS / "classifier_contract.json"
    if not path.is_file():
        raise FileNotFoundError("Run notebooks 04 through 06 before building figures")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not contract.get("curriculum_complete"):
        raise RuntimeError("Classifier contract does not name a complete curriculum")
    expected = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
    if contract.get("conditions_seen") != expected:
        raise RuntimeError(f"Unexpected curriculum order: {contract.get('conditions_seen')}")
    return contract


CONTRACT = load_contract()
CHECKPOINT_NAME = CONTRACT["encoder_checkpoint"]
VARIANT = "_augmented" if CHECKPOINT_NAME.endswith("_augmented.pt") else ""
FINGERPRINT = CONTRACT["checkpoint_fingerprint"]


def validate_current_artifacts() -> None:
    """Reject mixed-run inputs before any publication figure is written."""

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"Figure input contract failed: {message}")

    require((ARTIFACTS / CHECKPOINT_NAME).is_file(), f"missing {CHECKPOINT_NAME}")

    canonical = pd.read_parquet(ARTIFACTS / "sequence_embeddings.parquet")
    augmented = pd.read_parquet(ARTIFACTS / "augmented_normal_embeddings.parquet")
    require(len(canonical) == 96, f"expected 96 canonical embeddings, found {len(canonical)}")
    require(len(augmented) == 63, f"expected 63 added-normal embeddings, found {len(augmented)}")
    require(augmented["video_id"].nunique() == 17, "added-normal video count is not 17")
    for name, frame in (("canonical", canonical), ("added-normal", augmented)):
        fingerprints = set(frame["checkpoint_fingerprint"].astype(str))
        require(fingerprints == {FINGERPRINT}, f"{name} embeddings have the wrong fingerprint")

    expected_counts = {
        "normal": 12,
        "parkinsons": 9,
        "stroke": 12,
        "myopathic": 47,
        "cerebralpalsy": 16,
    }
    require(canonical["condition"].value_counts().to_dict() == expected_counts, "canonical class counts changed")

    history_path = ARTIFACTS / f"curriculum_training_history{VARIANT}.csv"
    history = pd.read_csv(history_path)
    expected_stage_epochs = {
        int(stage["stage"]): int(stage["epochs"]) for stage in CONTRACT["completed_stages"]
    }
    observed_stage_epochs = history["stage"].astype(int).value_counts().sort_index().to_dict()
    require(observed_stage_epochs == expected_stage_epochs, "training history does not match checkpoint stages")

    geometry = pd.read_csv(ARTIFACTS / "curriculum_representation_geometry.csv")
    require(
        set(geometry["checkpoint_fingerprint"].astype(str)) == {FINGERPRINT},
        "geometry summary has the wrong fingerprint",
    )

    metrics = pd.read_csv(ARTIFACTS / "classifier_metrics.csv")
    task_splits = dict(zip(metrics["task"], metrics["split"]))
    require(
        task_splits.get("five_class_all_sequences") == CONTRACT["splits"]["all_sequences"],
        "A1 classifier split does not match the contract",
    )
    require(
        task_splits.get("five_class_exp5_exact") == CONTRACT["splits"]["exp5_exact"],
        "A2 classifier split does not match the contract",
    )

    leakage = pd.read_csv(ARTIFACTS / "leakage_audit.csv")
    leakage_by_split = leakage.set_index("split")
    require(
        int(leakage_by_split.loc[CONTRACT["splits"]["all_sequences"], "test_sequences"]) == 29,
        "A1 leakage audit does not contain 29 test rows",
    )
    require(
        int(leakage_by_split.loc[CONTRACT["splits"]["exp5_exact"], "test_sequences"]) == 21,
        "A2 leakage audit does not contain 21 test rows",
    )

    lane_c = pd.read_csv(ARTIFACTS / "lane_c_video_disjoint_metrics.csv")
    require(set(lane_c["encoder"].astype(str)) == {CHECKPOINT_NAME}, "Lane C names a different encoder")
    require(set(lane_c["n_sequences"].astype(int)) == {159}, "Lane C sequence count is not 159")
    require(
        set(lane_c["test_seqs_seen_in_representation_training"].astype(int)) == {159},
        "Lane C encoder-exposure count is not 159",
    )

    result_history = pd.read_csv(Path(__file__).with_name("result_history.csv"))
    current_exp5 = result_history.loc[
        result_history["comparison"].eq("exact_exp5")
        & result_history["version"].eq("current_five_stage")
    ].iloc[0]
    corrected_lane_c = result_history.loc[
        result_history["version"].eq("corrected_two_fold_mean")
    ].iloc[0]
    a2 = metrics.loc[metrics["task"].eq("five_class_exp5_exact")].iloc[0]
    five_class = lane_c.loc[
        lane_c["task"].eq("five_class_classifier_video_disjoint_encoder_transductive")
    ].iloc[0]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        require(
            np.isclose(float(current_exp5[metric]), float(a2[metric])),
            f"result ledger disagrees with current A2 {metric}",
        )
        lane_column = f"{metric}_mean"
        require(
            np.isclose(float(corrected_lane_c[metric]), float(five_class[lane_column])),
            f"result ledger disagrees with corrected Lane C {metric}",
        )


validate_current_artifacts()


def artifact(stem: str, suffix: str) -> Path:
    path = ARTIFACTS / f"{stem}{VARIANT}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Missing current-run artifact: {path}")
    return path


def save(fig: plt.Figure, stem: str, aliases=()) -> None:
    for output_stem in (stem, *aliases):
        fig.savefig(FIGURES / f"{output_stem}.pdf", bbox_inches="tight")
        fig.savefig(FIGURES / f"{output_stem}.svg", bbox_inches="tight")
        fig.savefig(FIGURES / f"{output_stem}.png", bbox_inches="tight", dpi=240)
    plt.close(fig)


def rounded_box(ax, xywh, text, facecolor, fontsize=7, textcolor=None) -> None:
    x, y, w, h = xywh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=0.9,
            edgecolor=COLORS["navy"],
            facecolor=facecolor,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=textcolor or COLORS["ink"],
        linespacing=1.25,
    )


def arrow(ax, start, end, color=None, width=1.0) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=width,
            color=color or COLORS["navy"],
        )
    )


def make_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["paper"])

    boxes = [
        (0.01, 0.58, 0.13, 0.25, "GAVD video\nand frame rows", "#E6F0F7"),
        (0.17, 0.58, 0.13, 0.25, "33-landmark\npose sequence", "#E6F0F7"),
        (0.33, 0.58, 0.15, 0.25, "12 landmark IDs\neligible for masking", "#F2EAF7"),
        (0.51, 0.58, 0.17, 0.25, "Normal first, then\n4 cumulative stages", "#FFF0D6"),
        (0.71, 0.58, 0.13, 0.25, "Frozen 384-D\nreadout vector", "#E4F2E8"),
        (0.87, 0.58, 0.12, 0.25, "RF probes\nand audits", "#E4F2E8"),
        (0.33, 0.12, 0.15, 0.22, "Uniform target draw\nno motion score", "#F8E4DA"),
        (0.51, 0.12, 0.17, 0.22, "JEPA + VICReg\n+ group after Stage 0", "#F8E4DA"),
        (
            0.71,
            0.12,
            0.28,
            0.22,
            "Report source overlap,\nencoder exposure, and\nmissingness-only controls",
            "#EDF0F2",
        ),
    ]
    for x, y, w, h, text, color in boxes:
        rounded_box(ax, (x, y, w, h), text, color, fontsize=6.4)
    for left, right in zip(boxes[:5], boxes[1:6]):
        arrow(ax, (left[0] + left[2], 0.705), (right[0], 0.705))
    arrow(ax, (0.405, 0.34), (0.405, 0.58))
    arrow(ax, (0.595, 0.34), (0.595, 0.58))
    arrow(ax, (0.85, 0.23), (0.93, 0.58))
    ax.text(
        0.01,
        0.94,
        "From public video to an auditable descriptive readout",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    save(fig, "pipeline")


def make_cohort_curriculum() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [0.9, 1.4]})
    canonical = pd.Series(
        {"Normal": 12, "Parkinson's": 9, "Stroke": 12, "Myopathic": 47, "Cerebral palsy": 16}
    )
    colors = [COLORS["green"], COLORS["purple"], COLORS["blue"], COLORS["orange"], COLORS["gold"]]
    axes[0].barh(canonical.index[::-1], canonical.values[::-1], color=colors[::-1])
    axes[0].barh(["Normal"], [63], left=[12], color="#B7D6B0", hatch="///", edgecolor=COLORS["green"])
    axes[0].set_xlim(0, 82)
    axes[0].set_xlabel("Sequences")
    axes[0].set_title("Training corpus: 159 sequences")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].text(6, 4, "12", va="center", ha="center", fontsize=6.5, color="white", fontweight="bold")
    axes[0].text(43.5, 4, "+ 63 project-labeled normal", va="center", ha="center", fontsize=6.1)

    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    stages = [
        ("0", "Normal", "300 epochs", COLORS["green"]),
        ("1", "Parkinson's", "75 epochs", COLORS["purple"]),
        ("2", "Stroke", "75 epochs", COLORS["blue"]),
        ("3", "Myopathic", "75 epochs", COLORS["orange"]),
        ("4", "Cerebral palsy", "75 epochs", COLORS["gold"]),
    ]
    xs = np.linspace(0.02, 0.82, len(stages))
    for index, (stage, label, epochs, color) in enumerate(stages):
        rounded_box(ax, (xs[index], 0.42, 0.16, 0.30), f"Stage {stage}\n{label}\n{epochs}", color, fontsize=5.8)
        if index < len(stages) - 1:
            arrow(ax, (xs[index] + 0.16, 0.57), (xs[index + 1], 0.57))
    ax.text(0.02, 0.86, "One continuing model, 600 curriculum epochs", fontsize=9, fontweight="bold", color=COLORS["navy"])
    ax.text(0.02, 0.25, "Earlier groups stay in each later stage through condition-balanced replay.", fontsize=6.8, color=COLORS["ink"])
    ax.text(0.02, 0.12, "Total optimizer updates: 11,400. Final fingerprint: " + FINGERPRINT[:12] + "...", fontsize=6.4, color=COLORS["muted"])
    fig.text(
        0.02,
        0.025,
        "35 source videos: 18 normal and 17 non-normal\n"
        "Added labels were created in project and not independently reviewed.",
        fontsize=5.9,
        color=COLORS["muted"],
        linespacing=1.35,
    )
    fig.subplots_adjust(bottom=0.31, wspace=0.26)
    save(fig, "cohort_curriculum")


def make_training_health() -> None:
    history = pd.read_csv(artifact("curriculum_training_history", ".csv")).reset_index(drop=True)
    history["curriculum_epoch"] = np.arange(1, len(history) + 1)
    required = {
        "jepa_loss",
        "vicreg_loss",
        "feature_std",
        "normal_anchor_cosine",
        "minimum_centroid_distance",
        "group_separation",
        "stage",
    }
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Curriculum history is missing {sorted(missing)}")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.2), sharex=True)
    x = history["curriculum_epoch"]
    axes[0, 0].plot(x, history["jepa_loss"], color=COLORS["blue"], linewidth=1.0)
    axes[0, 0].set_title("Latent prediction loss")
    axes[0, 0].set_ylabel("JEPA loss")

    axes[0, 1].plot(x, history["vicreg_loss"], color=COLORS["purple"], linewidth=1.0)
    axes[0, 1].set_title("Anti-collapse regularization")
    axes[0, 1].set_ylabel("VICReg loss")

    axes[1, 0].plot(x, history["feature_std"], color=COLORS["teal"], linewidth=1.1, label="feature std")
    axes[1, 0].plot(x, history["normal_anchor_cosine"], color=COLORS["orange"], linewidth=1.1, label="normal anchor cosine")
    axes[1, 0].set_title("Spread stays nonzero, normal anchor drifts")
    axes[1, 0].set_ylabel("Recorded value")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(x, history["minimum_centroid_distance"], color=COLORS["gold"], linewidth=1.1, label="minimum centroid distance")
    axes[1, 1].plot(x, history["group_separation"], color=COLORS["red"], linewidth=1.1, label="margin penalty")
    axes[1, 1].set_title("Training-corpus group geometry")
    axes[1, 1].set_ylabel("Recorded value")
    axes[1, 1].legend(frameon=False)

    boundaries = history.groupby("stage")["curriculum_epoch"].min().iloc[1:]
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.set_xlabel("Curriculum epoch")
        for boundary in boundaries:
            axis.axvline(boundary - 0.5, color=COLORS["muted"], linestyle="--", linewidth=0.65)
    fig.suptitle("Recommended augmented-normal run: 300 + 4 x 75 epochs", fontsize=10, fontweight="bold", color=COLORS["navy"])
    fig.tight_layout()
    save(fig, "training_health")


def make_representation_geometry() -> None:
    matrix = pd.read_csv(ARTIFACTS / "curriculum_centroid_distances.csv", index_col=0)
    summary = pd.read_csv(ARTIFACTS / "curriculum_representation_geometry.csv").iloc[0]
    labels = ["Normal", "Parkinson's", "Stroke", "Myopathic", "Cerebral palsy"]
    values = matrix.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.45, 3.55))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=max(0.9, values.max()))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row, column]:.3f}", ha="center", va="center", fontsize=7, color="white" if values[row, column] > 0.48 else COLORS["ink"])
    ax.set_xticks(range(5), labels, rotation=28, ha="right")
    ax.set_yticks(range(5), labels)
    ax.set_title("Canonical 96: cosine distance between condition centroids")
    fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04, label="Cosine distance")
    note = (
        f"Minimum between centroids {summary['minimum_between_centroid_cosine_distance']:.3f}  |  "
        f"mean within-condition {summary['mean_within_condition_cosine_distance']:.3f}  |  "
        f"silhouette {summary['cosine_silhouette']:.3f}"
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=6.7, color=COLORS["muted"])
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "representation_geometry")


def make_readout_results() -> None:
    metrics = pd.read_csv(ARTIFACTS / "classifier_metrics.csv")
    missingness = pd.read_csv(ARTIFACTS / "missingness_only_classifier_metrics.csv")
    lane_c = pd.read_csv(ARTIFACTS / "lane_c_video_disjoint_metrics.csv")
    comparison = pd.read_csv(ARTIFACTS / "exp5_comparison.csv")
    a1 = metrics.loc[metrics["task"].eq("five_class_all_sequences")].iloc[0]
    a2 = metrics.loc[metrics["task"].eq("five_class_exp5_exact")].iloc[0]
    miss_a1 = missingness.loc[missingness["lane"].eq("all_sequences")].iloc[0]
    miss_a2 = missingness.loc[missingness["lane"].eq("exp5_exact")].iloc[0]
    historical = comparison.loc[
        comparison["system"].str.contains("handcrafted 82-feature", regex=False)
    ].iloc[0]
    binary = lane_c.loc[lane_c["task"].eq("normal_vs_abnormal_video_disjoint")].iloc[0]
    five = lane_c.loc[
        lane_c["task"].eq("five_class_classifier_video_disjoint_encoder_transductive")
    ].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    left_x = np.arange(2)
    left_width = 0.24
    left_groups = [
        axes[0].bar(
            left_x - left_width,
            [float(miss_a1["accuracy"]), float(miss_a2["accuracy"])],
            left_width,
            color=COLORS["muted"],
            label="Missingness only",
        ),
        axes[0].bar(
            left_x,
            [float(a1["accuracy"]), float(a2["accuracy"])],
            left_width,
            color=COLORS["teal"],
            label="S-JEPA",
        ),
        axes[0].bar(
            left_x + left_width,
            [np.nan, float(historical["accuracy"])],
            left_width,
            color=COLORS["blue"],
            label="Historical 82-feature",
        ),
    ]
    axes[0].set_xticks(left_x, ["A1\nall 96", "A2\nexact split"])
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Sequence-split accuracy and controls")
    axes[0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=6.3,
        handlelength=1.3,
        columnspacing=0.8,
    )
    for group in left_groups:
        for bar in group:
            value = bar.get_height()
            if np.isfinite(value):
                axes[0].text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.025,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=6.7,
                )

    right_labels = ["Normal vs abnormal\n5 folds", "Five class\n2 folds"]
    right_values = [float(binary["accuracy_mean"]), float(five["accuracy_mean"])]
    lower = right_values[0] - float(binary["accuracy_ci_low"])
    upper = float(binary["accuracy_ci_high"]) - right_values[0]
    bars = axes[1].bar(
        range(2),
        right_values,
        color=[COLORS["green"], COLORS["gold"]],
        width=0.58,
    )
    axes[1].errorbar(
        [0],
        [right_values[0]],
        yerr=np.array([[lower], [upper]]),
        fmt="none",
        color="black",
        capsize=4,
        linewidth=1.0,
    )
    axes[1].set_xticks(range(2), right_labels)
    axes[1].set_ylim(0, 1.0)
    axes[1].set_ylabel("Mean accuracy across grouped folds")
    axes[1].set_title("Lane C groups the Random Forest by video")
    for bar, value in zip(bars, right_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value - 0.09, f"{value:.3f}", ha="center", fontsize=7, color="white", fontweight="bold")

    for axis in axes:
        axis.grid(axis="y", alpha=0.24)
    fig.suptitle("Descriptive results, not independent clinical performance", fontsize=10, fontweight="bold", color=COLORS["navy"])
    fig.text(
        0.27,
        0.025,
        "A1 and A2 share videos across train and test.\n"
        "The historical reference uses a different pose and feature pipeline.",
        ha="center",
        fontsize=6.2,
        color=COLORS["red"],
        linespacing=1.25,
    )
    fig.text(
        0.75,
        0.025,
        "Different tasks; do not compare bar heights directly.\n"
        "Black interval: percentile bootstrap over five fold scores.",
        ha="center",
        fontsize=6.2,
        color=COLORS["red"],
        linespacing=1.25,
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.31, top=0.77, wspace=0.20)
    # Keep the old filename as a current compatibility alias. This prevents an
    # obsolete 0.619 legacy plot from surviving beside the corrected figures.
    save(fig, "readout_results", aliases=("exact_split_results",))


def make_result_changes() -> None:
    """Show which numerical changes came from a new model or a repaired audit."""
    metrics = pd.read_csv(ARTIFACTS / "classifier_metrics.csv")
    lane_c = pd.read_csv(ARTIFACTS / "lane_c_video_disjoint_metrics.csv")
    a2 = metrics.loc[metrics["task"].eq("five_class_exp5_exact")].iloc[0]
    five = lane_c.loc[
        lane_c["task"].eq("five_class_classifier_video_disjoint_encoder_transductive")
    ].iloc[0]

    result_history = pd.read_csv(Path(__file__).with_name("result_history.csv"))
    legacy = result_history.loc[
        result_history["comparison"].eq("exact_exp5")
        & result_history["version"].eq("legacy_normal_only")
    ].iloc[0]
    superseded = result_history.loc[
        result_history["version"].eq("superseded_five_fold_mean")
    ].iloc[0]
    legacy_exp5 = [float(legacy["accuracy"]), float(legacy["macro_f1"])]
    superseded_lane_c = [
        float(superseded["accuracy"]),
        float(superseded["balanced_accuracy"]),
        float(superseded["macro_f1"]),
    ]
    current_exp5 = [float(a2["accuracy"]), float(a2["macro_f1"])]
    corrected_lane_c = [
        float(five["accuracy_mean"]),
        float(five["balanced_accuracy_mean"]),
        float(five["macro_f1_mean"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.15))
    width = 0.34

    left_x = np.arange(2)
    left_old = axes[0].bar(
        left_x - width / 2,
        legacy_exp5,
        width,
        color=COLORS["muted"],
        label="Legacy normal-only",
    )
    left_new = axes[0].bar(
        left_x + width / 2,
        current_exp5,
        width,
        color=COLORS["teal"],
        label="Current five-stage",
    )
    axes[0].set_xticks(left_x, ["Accuracy", "Macro-F1"])
    axes[0].set_title("Model revision on the exact exp5 split")
    axes[0].legend(loc="upper left", frameon=False)

    right_x = np.arange(3)
    right_old = axes[1].bar(
        right_x - width / 2,
        superseded_lane_c,
        width,
        color=COLORS["muted"],
        label="Superseded 5-fold audit",
    )
    right_new = axes[1].bar(
        right_x + width / 2,
        corrected_lane_c,
        width,
        color=COLORS["gold"],
        label="Corrected 2-fold audit",
    )
    axes[1].set_xticks(right_x, ["Accuracy", "Balanced\naccuracy", "Macro-F1"])
    axes[1].set_title("Evaluation repair with the same checkpoint")
    axes[1].legend(loc="upper left", frameon=False)

    for axis, groups in ((axes[0], (left_old, left_new)), (axes[1], (right_old, right_new))):
        axis.set_ylim(0, 1.0)
        axis.set_ylabel("Score")
        axis.grid(axis="y", alpha=0.24)
        for group in groups:
            for bar in group:
                value = bar.get_height()
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.022,
                    f"{value:.3f}",
                    ha="center",
                    fontsize=6.8,
                )

    fig.suptitle(
        "Previous and current results must be compared at the right layer",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    fig.text(
        0.27,
        0.035,
        "The model, data, and objective changed; the split stayed confounded.",
        ha="center",
        fontsize=6.4,
        color=COLORS["red"],
    )
    fig.text(
        0.75,
        0.035,
        "The model stayed fixed; only the downstream evaluation changed.",
        ha="center",
        fontsize=6.4,
        color=COLORS["red"],
    )
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.78, wspace=0.22)
    save(fig, "result_changes")


def make_evidence_ladder() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.91, "Three questions need three different claims", fontsize=10, fontweight="bold", color=COLORS["navy"])
    cards = [
        (0.02, "A1: all 96", "67 / 29 stratified\n16 shared videos\n29 test rows seen by encoder", "Descriptive", "#F7E5D7", COLORS["orange"]),
        (0.35, "A2: exp5 match", "47 / 21 legacy split\n9 shared videos\n21 test rows seen by encoder", "Split match only", "#E4EEF7", COLORS["blue"]),
        (
            0.68,
            "Lane C: grouped RF",
            "159 rows, 35 videos\nbinary: 5; five class: 2 folds\nall rows seen by encoder",
            "Useful but not independent",
            "#E5F1E8",
            COLORS["green"],
        ),
    ]
    for x, title, body, badge, face, badge_color in cards:
        rounded_box(ax, (x, 0.20, 0.29, 0.58), "", face)
        ax.text(x + 0.018, 0.70, title, fontsize=8.5, fontweight="bold", color=COLORS["navy"])
        ax.text(x + 0.018, 0.58, body, fontsize=7, va="top", color=COLORS["ink"], linespacing=1.35)
        ax.add_patch(FancyBboxPatch((x + 0.018, 0.25), 0.254, 0.10, boxstyle="round,pad=0.008,rounding_size=0.015", linewidth=0, facecolor=badge_color))
        ax.text(x + 0.145, 0.30, badge, ha="center", va="center", fontsize=6.8, color="white", fontweight="bold")
    ax.text(0.02, 0.07, "Independent performance still requires fold-local preprocessing choices and fold-local retraining of all five representation stages.", fontsize=6.8, color=COLORS["red"])
    save(fig, "evidence_ladder")


if __name__ == "__main__":
    make_pipeline()
    make_cohort_curriculum()
    make_training_health()
    make_representation_geometry()
    make_readout_results()
    make_result_changes()
    make_evidence_ladder()
    print(f"wrote figures for {CHECKPOINT_NAME} ({FINGERPRINT[:12]}...)")
