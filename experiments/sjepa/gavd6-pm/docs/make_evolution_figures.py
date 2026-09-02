"""Build vector figures for the S-JEPA evolution tutorials.

The measured values come from the same validated real-run contract used by
``make_figures.py``. Concept diagrams distinguish completed work from proposed
next experiments so an idea is never presented as an achieved result.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

from make_figures import (
    ARROW_DASHES,
    ARROW_STYLE,
    ARTIFACTS,
    CHECKPOINT_NAME,
    COLORS,
    CONTRACT,
    FINGERPRINT,
    FIGURES,
    save,
)


LEDGER = pd.read_csv(Path(__file__).with_name("result_history.csv"))
STAGES = pd.read_csv(ARTIFACTS / "curriculum_stage_summary_augmented.csv")
GEOMETRY = pd.read_csv(ARTIFACTS / "curriculum_representation_geometry.csv").iloc[0]
HISTORY = pd.read_csv(ARTIFACTS / "curriculum_training_history_augmented.csv")
A1_REPORT = pd.read_csv(ARTIFACTS / "all_sequences_classification_report.csv", index_col=0)
A2_REPORT = pd.read_csv(ARTIFACTS / "five_class_classification_report.csv", index_col=0)
CLASSIFIER_METRICS = pd.read_csv(ARTIFACTS / "classifier_metrics.csv")
MISSINGNESS = pd.read_csv(ARTIFACTS / "missingness_only_classifier_metrics.csv").set_index("lane")
ELIGIBLE_BY_STAGE = HISTORY.groupby("stage")["eligible_mask_fraction"].mean()

CHECKPOINT_FILES = [
    "sjepa_normal_augmented.pt",
    "sjepa_stage_01_parkinsons_augmented.pt",
    "sjepa_stage_02_stroke_augmented.pt",
    "sjepa_stage_03_myopathic_augmented.pt",
    "sjepa_stage_04_cerebralpalsy_augmented.pt",
]
CHECKPOINT_METADATA = [
    torch.load(ARTIFACTS / name, map_location="cpu", weights_only=False)
    for name in CHECKPOINT_FILES
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Evolution figure contract failed: {message}")


require(CHECKPOINT_NAME == "sjepa_curriculum_final_augmented.pt", "unexpected final checkpoint")
require(CONTRACT["feature_count"] == 384, "expected a 384-dimensional readout")
require(len(CONTRACT["mask_keypoints"]) == 12, "expected 12 eligible landmark identities")
require(len(CONTRACT["completed_stages"]) == 5, "expected five completed curriculum stages")
require(
    [len(item["sequence_ids"]) for item in CHECKPOINT_METADATA] == [75, 84, 96, 143, 159],
    "checkpoint lineage has unexpected sequence counts",
)
require(
    CHECKPOINT_METADATA[-1]["dataset_fingerprint"] == FINGERPRINT,
    "Stage 4 fingerprint differs from the classifier contract",
)

CURRENT_LEDGER_TASKS = {
    "all_96": ("five_class_all_sequences", "all"),
    "one_vs_normal_parkinsons": ("one_vs_normal", "parkinsons"),
    "one_vs_normal_stroke": ("one_vs_normal", "stroke"),
    "one_vs_normal_myopathic": ("one_vs_normal", "myopathic"),
    "one_vs_normal_cerebralpalsy": ("one_vs_normal", "cerebralpalsy"),
}
for comparison, (task, condition) in CURRENT_LEDGER_TASKS.items():
    ledger_row = LEDGER.loc[
        LEDGER["comparison"].eq(comparison)
        & LEDGER["version"].eq("current_five_stage")
    ].iloc[0]
    metric_row = CLASSIFIER_METRICS.loc[
        CLASSIFIER_METRICS["task"].eq(task)
        & CLASSIFIER_METRICS["condition"].eq(condition)
    ].iloc[0]
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        require(
            np.isclose(float(ledger_row[metric]), float(metric_row[metric])),
            f"current ledger {comparison} disagrees on {metric}",
        )


# A card's drawn edge sits this far outside the rectangle passed to ``card``, because that is the
# FancyBboxPatch pad. Layout code has to budget for it on both sides of every gap: it is
# subtracted from the connector clear space that shrinkA and shrinkB reserve. This matches the
# pad make_figures.rounded_box uses, so a connector clears a card identically in both modules.
CARD_PAD = 0.008
# The horizontal gap between two cards in a connected row, in axes units. It has to hold the
# card pad twice plus the connector's own clear space at both ends and still leave a shaft that
# reads as a line rather than a stub. At the width these figures render this is about 71 pixels.
ROW_GAP = 0.056


def canvas(figsize=(7.1, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    # These are pure diagrams with no visible axes, so the drawing area is the whole canvas.
    # On matplotlib's default subplot margins a quarter of the width was unusable margin, and
    # that missing width is what starved the connectors between cards.
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.01, top=0.99)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(COLORS["paper"])
    return fig, ax


def row_positions(count: int, x0: float = 0.006, x1: float = 0.994, gap: float = ROW_GAP):
    """Return (left edges, card width) for ``count`` equal cards in a connected row.

    Positions are derived rather than written out so a card cannot be nudged into its neighbour,
    and so the gap that the connectors live in is guaranteed rather than hoped for.
    """
    width = (x1 - x0 - gap * (count - 1)) / count
    return [x0 + index * (width + gap) for index in range(count)], width


def title(ax, text: str, subtitle: str | None = None) -> None:
    ax.text(
        0.02,
        0.95,
        text,
        va="top",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
    )
    if subtitle:
        ax.text(0.02, 0.875, subtitle, va="top", fontsize=6.4, color=COLORS["muted"])


def card(
    ax,
    xywh,
    heading: str,
    body: str,
    face: str = "#FFFFFF",
    heading_color: str | None = None,
    body_color: str | None = None,
    heading_size: float = 7.2,
    body_size: float = 5.9,
    center: bool = False,
):
    x, y, w, h = xywh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad={CARD_PAD},rounding_size=0.018",
            linewidth=0.9,
            edgecolor=COLORS["navy"],
            facecolor=face,
        )
    )
    ha = "center" if center else "left"
    tx = x + w / 2 if center else x + 0.014
    heading_top = y + h - 0.045
    ax.text(
        tx,
        heading_top,
        heading,
        ha=ha,
        va="top",
        fontsize=heading_size,
        fontweight="bold",
        color=heading_color or COLORS["navy"],
    )
    # The body starts below however many lines the heading actually occupies. A fixed 0.125 drop
    # collided with the second line of every two-line heading. Line height is converted from
    # points to axes units through the figure height so it holds at any figsize.
    figure_height_points = ax.figure.get_figheight() * 72.0
    heading_lines = heading.count("\n") + 1
    heading_height = heading_lines * heading_size * 1.30 / figure_height_points
    ax.text(
        tx,
        heading_top - heading_height - 0.022,
        body,
        ha=ha,
        va="top",
        fontsize=body_size,
        color=body_color or COLORS["ink"],
        linespacing=1.32,
    )


def pill(ax, x, y, text_value, color, width=0.12, text_size=5.7):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            0.055,
            boxstyle="round,pad=0.005,rounding_size=0.018",
            linewidth=0,
            facecolor=color,
        )
    )
    ax.text(
        x + width / 2,
        y + 0.0275,
        text_value,
        ha="center",
        va="center",
        fontsize=text_size,
        fontweight="bold",
        color="white",
    )


def connector(ax, start, end, color=None, dashed=False):
    """Draw one connector in the shared style used by every figure in this set.

    ``start`` and ``end`` are the nominal card rectangle edges. The shared style's shrinkA and
    shrinkB pull the drawn line back from both of them, which is what produces the clear space
    the diagram design system asks for in front of an arrowhead.
    """
    style = dict(ARROW_STYLE)
    if color is not None:
        style["color"] = color
    if dashed:
        style["linestyle"] = ARROW_DASHES
    ax.add_patch(FancyArrowPatch(start, end, **style))


def make_timeline() -> None:
    legacy = LEDGER.loc[
        LEDGER["comparison"].eq("exact_exp5")
        & LEDGER["version"].eq("legacy_normal_only")
    ].iloc[0]
    current = LEDGER.loc[
        LEDGER["comparison"].eq("exact_exp5")
        & LEDGER["version"].eq("current_five_stage")
    ].iloc[0]
    fig, ax = canvas((7.1, 3.45))
    title(
        ax,
        "The project evolved in layers, not through one isolated model change",
        "Completed changes are green or blue. The final card is required work, not a reported result.",
    )
    cards = [
        (
            "1. Legacy",
            "12 normal rows\n1 source video\n10 target IDs\nJEPA only\nA2: "
            f"{legacy.accuracy:.3f} accuracy\n{legacy.macro_f1:.3f} macro-F1",
            "#F8E4DA",
            "SUPERSEDED",
            COLORS["red"],
        ),
        (
            "2. Failure audit",
            "video overlap found\nvisibility shortcut: "
            f"{float(MISSINGNESS.loc['all_sequences', 'accuracy']):.3f}"
            "\norder lost in pooling\nnormal had one video\nouter test blocked",
            "#FFF0D6",
            "COMPLETED",
            COLORS["gold"],
        ),
        (
            "3. Current model",
            "+63 normal rows\n12 target IDs\nthree-part loss\n5 stages, 159 rows\nA2: "
            f"{current.accuracy:.3f} accuracy\n{current.macro_f1:.3f} macro-F1",
            "#E4F2E8",
            "CURRENT",
            COLORS["green"],
        ),
        (
            "4. Eval repair",
            "A1 and A2 show overlap\nLane C groups the RF\nfive-class uses 2 folds\nsame checkpoint\nencoder saw 159/159",
            "#E4EEF7",
            "CURRENT",
            COLORS["blue"],
        ),
        (
            "5. Independent",
            "split videos first\nfit rules on train only\nretrain all 5 stages\nfreeze before test\nreport controls and ranges",
            "#F2EAF7",
            "NOT RUN",
            COLORS["purple"],
        ),
    ]
    xs, width = row_positions(len(cards))
    for i, (heading, body, face, status, status_color) in enumerate(cards):
        card(ax, (xs[i], 0.20, width, 0.60), heading, body, face, heading_size=6.0, body_size=4.95)
        pill(ax, xs[i] + 0.014, 0.22, status, status_color, width=0.105, text_size=5.0)
        if i < len(cards) - 1:
            connector(ax, (xs[i] + width, 0.50), (xs[i + 1], 0.50))
    ax.text(
        0.02,
        0.08,
        "Key lesson: a larger in-corpus score, a repaired fold design, and a truly independent estimate are three different kinds of progress.",
        fontsize=6.2,
        color=COLORS["red"],
    )
    save(fig, "evolution_timeline")


def make_layer_matrix() -> None:
    fig, ax = canvas((7.1, 4.3))
    title(ax, "Legacy and current methods differ at six layers")
    # Column widths follow the longest line each column has to hold. The legacy column used to be
    # 0.19 wide, which was narrower than "Loose filename-based reuse" at its own font size.
    columns = [
        (0.006, 0.105, "Layer"),
        (0.117, 0.225, "Legacy prototype"),
        (0.348, 0.320, "Current completed run"),
        (0.674, 0.320, "Impact and remaining limit"),
    ]
    for x, w, label in columns:
        ax.add_patch(Rectangle((x, 0.82), w, 0.075, facecolor=COLORS["navy"], edgecolor="none"))
        ax.text(x + 0.012, 0.857, label, va="center", fontsize=6.7, fontweight="bold", color="white")

    rows = [
        (
            "Data",
            "12 normal sequences\n1 source video",
            "Stage 0: 75 sequences / 18 videos\nStage 4: 159 sequences / 35 videos",
            "Broader support, but added-normal\nprovenance can still be a shortcut",
        ),
        (
            "Targets",
            "10 eligible\nlandmark identities",
            "12 identities from shoulders through feet\nuniform batch-safe draw; no motion score",
            "Heels restored; rule is explicit\nrealized fraction is not fixed at 0.60",
        ),
        (
            "Objective",
            "Centered, sharpened\nlatent cross-entropy",
            "JEPA + 0.05 VICReg + 0.25 group\ngroup term starts after Stage 0",
            "Active anti-collapse pressure added\nlater stages are label-informed",
        ),
        (
            "Training",
            "Normal-only checkpoint",
            "Normal first, then four condition stages\nwith condition-balanced replay",
            "One continuing 600-epoch model\n"
            f"normal anchor still drifted to {float(STAGES.iloc[-1]['normal_anchor_cosine']):.3f}",
        ),
        (
            "Artifacts",
            "Loose filename-based reuse",
            "Lineage, cohort, configuration, and\nfingerprint are recorded",
            "Reject incomplete, stale, wrong-mode,\nor mixed-fingerprint artifacts",
        ),
        (
            "Evaluation",
            "Single video-confounded\n47/21 comparison",
            "A1, A2, grouped-RF Lane C, plus\nmissingness and exposure audits",
            "Claims are narrower; no lane yet retrains\nthe encoder outside test videos",
        ),
    ]
    row_height = 0.115
    y = 0.80 - row_height
    faces = ["#F7F5EF", "#FFFFFF"]
    for row_index, row in enumerate(rows):
        for col_index, (x, w, _) in enumerate(columns):
            face = faces[row_index % 2]
            ax.add_patch(Rectangle((x, y), w, row_height, facecolor=face, edgecolor=COLORS["grid"], linewidth=0.7))
            ax.text(
                x + 0.012,
                y + row_height / 2,
                row[col_index],
                va="center",
                fontsize=5.2 if col_index else 6.0,
                fontweight="bold" if col_index == 0 else "normal",
                color=COLORS["navy"] if col_index == 0 else COLORS["ink"],
                linespacing=1.25,
            )
        y -= row_height
    ax.text(
        0.02,
        0.045,
        "The rows must be interpreted together. The accuracy change cannot be assigned to any single row without a controlled ablation.",
        fontsize=6.1,
        color=COLORS["red"],
    )
    save(fig, "evolution_layer_matrix")


def make_masking_math() -> None:
    fig, ax = canvas((7.1, 3.35))
    title(
        ax,
        "The current masking rule is constrained, uniform, and batch-safe",
        "The numbers in the middle card are a teaching example. Saved run values appear in the final card.",
    )
    xs, width = row_positions(3)
    card(
        ax,
        (xs[0], 0.22, width, 0.58),
        "1. Build joint-time tokens",
        "64 normalized frames\n÷ 4 frames per patch\n= 16 time patches\n\n16 x 33 joints\n= 528 possible tokens\n\nAll 33 identities may provide context.\nOnly 12 may become targets.",
        "#E4EEF7",
        heading_size=7.3,
        body_size=6.0,
    )
    card(
        ax,
        (xs[1], 0.22, width, 0.58),
        "2. Worked batch example",
        "Valid eligible counts:\n180, 160, 150, 175\n\ncommon target count\n= floor(0.60 x 150)\n= 90 tokens per sample\n\nrealized fractions:\n0.500, 0.562, 0.600, 0.514",
        "#FFF0D6",
        heading_size=7.3,
        body_size=6.0,
    )
    card(
        ax,
        (xs[2], 0.22, width, 0.58),
        "3. What the run recorded",
        f"Stage 0 endpoint mean: {float(ELIGIBLE_BY_STAGE.loc[0]):.3f}\n"
        f"Stage 4 endpoint mean: {float(ELIGIBLE_BY_STAGE.loc[4]):.3f}\n\nA sample with more eligible tokens\ngets a lower realized fraction.\n\nTargets are drawn uniformly.\nNo velocity, displacement, or\nlearned motion score is consulted.",
        "#E4F2E8",
        heading_size=7.3,
        body_size=6.0,
    )
    for left, right in zip(xs[:-1], xs[1:]):
        connector(ax, (left + width, 0.51), (right, 0.51))
    ax.text(
        0.02,
        0.08,
        "This corrects the misleading shortcut phrase '60% of each sample.' The setting is applied once to the smallest eligible count in the batch.",
        fontsize=6.1,
        color=COLORS["red"],
    )
    save(fig, "evolution_masking_math")


def make_data_scale() -> None:
    fig, ax = canvas((7.1, 3.7))
    title(
        ax,
        "Training breadth increased, while provenance became a new confounder",
        "Bar lengths encode sequence counts. Video counts are written beside each stage.",
    )
    stages = [
        ("Legacy normal-only", 12, "1 video", COLORS["red"]),
        ("Current Stage 0 normal", 75, "18 videos", COLORS["green"]),
        ("Current Stage 4 total", 159, "35 videos", COLORS["blue"]),
    ]
    y_positions = [0.70, 0.50, 0.30]
    max_width = 0.62
    for (label, count, videos, color), y in zip(stages, y_positions):
        ax.text(0.02, y + 0.045, label, fontsize=6.7, fontweight="bold", color=COLORS["navy"])
        width = max_width * count / 159
        ax.add_patch(
            FancyBboxPatch(
                (0.24, y),
                width,
                0.10,
                boxstyle="round,pad=0.004,rounding_size=0.015",
                facecolor=color,
                edgecolor="none",
            )
        )
        if width < 0.12:
            ax.text(
                0.24 + width + 0.012,
                y + 0.05,
                f"{count} sequences | {videos}",
                va="center",
                fontsize=6.0,
                color=COLORS["ink"],
            )
        else:
            ax.text(0.25, y + 0.05, f"{count} sequences", va="center", fontsize=6.2, color="white", fontweight="bold")
            ax.text(0.24 + width + 0.012, y + 0.05, videos, va="center", fontsize=6.1, color=COLORS["ink"])

    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.035),
            0.96,
            0.15,
            boxstyle="round,pad=0.008,rounding_size=0.016",
            facecolor="#F8E4DA",
            edgecolor=COLORS["red"],
            linewidth=0.9,
        )
    )
    ax.text(
        0.04,
        0.110,
        "Provenance warning: 63 of 75 normal rows use added extraction; all 84 abnormal rows use canonical extraction.\n"
        "A classifier may learn camera, crop, bounding-box, or detector differences.\n"
        "Added labels were not independently clinically reviewed.",
        va="center",
        fontsize=5.25,
        color=COLORS["red"],
        linespacing=1.30,
    )
    save(fig, "evolution_data_scale")


def make_augmentation_pipeline() -> None:
    fig, ax = canvas((7.1, 4.1))
    title(
        ax,
        "Added-normal diversity came from a separate, auditable extraction path",
        "The windows overlap within a video. The scientific gain is 17 additional source videos, not 63 independent people.",
    )
    xs, width = row_positions(4)
    top = [
        (
            xs[0],
            "1. Added videos",
            "17 YouTube videos\nproject-labeled normal\nkept outside canonical GAVD",
            "#E4EEF7",
        ),
        (
            xs[1],
            "2. Select walker",
            "up to 3 pose candidates\nvisibility + area +\ntemporal continuity",
            "#E4EEF7",
        ),
        (
            xs[2],
            "3. Track a box",
            "pad, fill short gaps,\nsmooth, then clamp\nto the video frame",
            "#FFF0D6",
        ),
        (
            xs[3],
            "4. Estimate cycle",
            "ankle-separation signal\nuses autocorrelation\nto estimate stride timing",
            "#FFF0D6",
        ),
    ]
    bottom = [
        (
            xs[3],
            "5. Cut windows",
            "about two cycles\n50 percent overlap\n24-frame minimum; max 8",
            "#F2EAF7",
        ),
        (
            xs[2],
            "6. Extract poses",
            "MediaPipe landmarks\nseparate provenance and\nposes_augmented storage",
            "#F2EAF7",
        ),
        (
            xs[1],
            "7. Apply gate",
            "64 candidates\nneurologic coverage\nmust be at least 0.45",
            "#E4F2E8",
        ),
        (
            xs[0],
            "8. Freeze cohort",
            "63 accepted\n1 rejected at 0.027\n17 videos retained",
            "#E4F2E8",
        ),
    ]
    for x, heading, body, face in top:
        card(ax, (x, 0.52, width, 0.28), heading, body, face, heading_size=6.4, body_size=5.25)
    for x, heading, body, face in bottom:
        card(ax, (x, 0.14, width, 0.28), heading, body, face, heading_size=6.4, body_size=5.25)
    # The row runs left to right on top, drops down the right edge, then runs right to left.
    for left, right in zip(top[:-1], top[1:]):
        connector(ax, (left[0] + width, 0.66), (right[0], 0.66))
    turn_x = xs[3] + width / 2
    connector(ax, (turn_x, 0.52), (turn_x, 0.42))
    for right, left in zip(bottom[:-1], bottom[1:]):
        connector(ax, (right[0], 0.28), (left[0] + width, 0.28))
    ax.text(
        0.02,
        0.045,
        "Selection improves reproducibility, but it cannot remove the acquisition-path difference between added normal and canonical abnormal rows.",
        fontsize=6.0,
        color=COLORS["red"],
    )
    save(fig, "evolution_augmentation_pipeline")


def make_objective() -> None:
    final_stage = STAGES.loc[STAGES["stage"].eq(4)].iloc[0]
    fig, ax = canvas((7.1, 3.75))
    title(ax, "The objective evolved from prediction alone to three explicit jobs")
    card(
        ax,
        (0.006, 0.46, 0.29, 0.36),
        "Legacy objective",
        "L = L_JEPA\n\nCentered and sharpened latent\ncross-entropy trained prediction.\nCollapse was monitored,\nnot actively regularized.",
        "#F8E4DA",
        heading_size=7.4,
        body_size=5.35,
    )
    connector(ax, (0.296, 0.64), (0.352, 0.64))
    card(
        ax,
        (0.352, 0.46, 0.642, 0.36),
        "Current objective",
        "L = L_JEPA + 0.05 L_VICReg + 0.25 L_group\n\nJEPA: predict authorized hidden teacher features\nVICReg: keep variation and reduce redundancy\nGroup: compact labels and penalize close centroids\nGroup is off during Stage 0",
        "#E4F2E8",
        heading_size=7.4,
        body_size=5.2,
    )
    observed = [
        ("Feature std", float(final_stage["feature_std"]), "evidence against total collapse", COLORS["green"]),
        ("Normal anchor", float(final_stage["normal_anchor_cosine"]), "substantial drift", COLORS["orange"]),
        ("Margin penalty", float(final_stage["group_separation"]), "requested margin not fully met", COLORS["red"]),
        ("Silhouette", float(GEOMETRY["cosine_silhouette"]), "weak canonical geometry", COLORS["purple"]),
    ]
    x_positions, metric_width = row_positions(len(observed), gap=0.020)
    for x, (label, value, meaning, color) in zip(x_positions, observed):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.12), metric_width, 0.22,
                boxstyle="round,pad=0.008,rounding_size=0.015",
                facecolor="#FFFFFF", edgecolor=color, linewidth=1.0,
            )
        )
        ax.text(x + 0.012, 0.295, label, va="top", fontsize=6.5, fontweight="bold", color=COLORS["navy"])
        ax.text(x + 0.012, 0.235, f"{value:.3f}", va="top", fontsize=9.0, fontweight="bold", color=color)
        ax.text(x + 0.012, 0.165, meaning, va="top", fontsize=5.4, color=COLORS["ink"])
    ax.text(
        0.02,
        0.035,
        "The group term reads condition labels. Only Stage 0 is label-free; Stages 1 through 4 are label-informed representation fine-tuning.",
        fontsize=6.0,
        color=COLORS["red"],
    )
    save(fig, "evolution_objective")


def make_artifact_contract() -> None:
    fig, ax = canvas((7.1, 3.2))
    title(
        ax,
        "The artifact contract turns a notebook chain into an auditable experiment",
        f"Current fingerprint: {FINGERPRINT[:16]}...",
    )
    nodes = [
        ("Notebook 04\ncheckpoint", "model + teacher\n5 completed stages", "#FFF0D6"),
        ("Lineage\ncontract", "cohort + mask + loss\nparent fingerprint", "#F2EAF7"),
        ("Notebook 05\nembeddings", "96 canonical + 63 added\n384 values each", "#E4F2E8"),
        ("Notebook 06\nreadouts", "metrics + overlap\nmissingness controls", "#E4EEF7"),
        ("Figures\nand docs", "ledger values checked\nagainst artifacts", "#EDF0F2"),
    ]
    # Every heading is wrapped to two lines so the row of cards shares one baseline grid.
    xs, width = row_positions(len(nodes))
    for i, (heading, body, face) in enumerate(nodes):
        card(ax, (xs[i], 0.44, width, 0.34), heading, body, face, heading_size=6.7, body_size=5.5, center=True)
        if i < len(nodes) - 1:
            connector(ax, (xs[i] + width, 0.61), (xs[i + 1], 0.61))
    checks = [
        "complete curriculum",
        "correct stage order",
        "12-target whitelist",
        "matching fingerprint",
        "expected row counts",
    ]
    pill_positions, pill_width = row_positions(len(checks), gap=0.014)
    for x, label in zip(pill_positions, checks):
        pill(ax, x, 0.24, label, COLORS["green"], width=pill_width, text_size=4.8)
    ax.text(
        0.02,
        0.11,
        "A missing filename is no longer silently replaced by a different checkpoint. Consumers stop when the lineage or cohort does not match.",
        fontsize=6.2,
        color=COLORS["red"],
    )
    save(fig, "evolution_artifact_contract")


def make_checkpoint_lineage() -> None:
    fig, ax = canvas((7.1, 3.65))
    title(
        ax,
        "Five fingerprints bind one continuing model lineage",
        "Model and teacher states continue. A fresh optimizer, scheduler, warmup, and EMA schedule position start at each stage.",
    )
    names = ["Normal", "Parkinson's", "Stroke", "Myopathic", "Cerebral palsy"]
    endpoints = HISTORY.sort_values(["stage", "epoch_in_stage"]).groupby("stage").tail(1).set_index("stage")
    xs, width = row_positions(5)
    for stage, (x, metadata, condition) in enumerate(zip(xs, CHECKPOINT_METADATA, names)):
        completed = metadata["completed_stages"][-1]
        n_sequences = len(metadata["sequence_ids"])
        n_videos = len(set(metadata["video_ids"]))
        fingerprint = metadata["dataset_fingerprint"][:8]
        anchor = "reference" if stage == 0 else f"{float(endpoints.loc[stage, 'normal_anchor_cosine']):.3f}"
        body = (
            f"add {condition}\n"
            f"{n_sequences} sequences / {n_videos} videos\n"
            f"{completed['epochs']} epochs\n"
            f"{completed['optimizer_updates']:,} updates\n"
            f"fingerprint {fingerprint}...\n"
            f"normal anchor {anchor}"
        )
        face = ["#E4F2E8", "#F2EAF7", "#E4EEF7", "#FFF0D6", "#F8E4DA"][stage]
        card(
            ax,
            (x, 0.25, width, 0.55),
            f"Stage {stage}",
            body,
            face,
            heading_size=7.0,
            body_size=5.15,
        )
        if stage < 4:
            connector(ax, (x + width, 0.52), (xs[stage + 1], 0.52))
    ax.text(
        0.02,
        0.115,
        "Total: 600 curriculum epochs and 11,400 optimizer updates. The Stage 4 state is also saved under the final checkpoint alias.",
        fontsize=6.1,
        color=COLORS["ink"],
    )
    ax.text(
        0.02,
        0.055,
        "The experiment fingerprint identifies data and configuration lineage. It is not a byte checksum of the checkpoint file.",
        fontsize=6.0,
        color=COLORS["red"],
    )
    save(fig, "evolution_checkpoint_lineage")


def make_classwise_results() -> None:
    classes = ["cerebralpalsy", "myopathic", "normal", "parkinsons", "stroke"]
    labels = ["Cerebral\npalsy", "Myopathic", "Normal", "Parkinson's", "Stroke"]
    a1 = A1_REPORT.loc[classes, "f1-score"].to_numpy(dtype=float)
    a2 = A2_REPORT.loc[classes, "f1-score"].to_numpy(dtype=float)
    support_a1 = A1_REPORT.loc[classes, "support"].astype(int).to_numpy()
    support_a2 = A2_REPORT.loc[classes, "support"].astype(int).to_numpy()
    x = np.arange(len(classes))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.1, 3.45))
    first = ax.bar(x - width / 2, a1, width, color=COLORS["blue"], label="A1: all 96")
    second = ax.bar(x + width / 2, a2, width, color=COLORS["teal"], label="A2: exact exp5")
    ax.set_ylim(0, 1.13)
    ax.set_ylabel("Class F1")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{label}\nn={n1}/{n2}" for label, n1, n2 in zip(labels, support_a1, support_a2)]
    )
    ax.grid(axis="y", alpha=0.24)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.set_title(
        "Aggregate scores hide very different class behavior",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
        pad=12,
    )
    for bars in (first, second):
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.3f}",
                ha="center",
                fontsize=5.7,
            )
    fig.text(
        0.5,
        0.055,
        # Name whichever class is currently weakest rather than assuming it stays stroke.
        "Support below each class is A1 / A2. A2 macro-F1 is "
        f"{float(A2_REPORT.loc['macro avg', 'f1-score']):.3f}, yet "
        f"{A2_REPORT.drop(index=['accuracy', 'macro avg', 'weighted avg'])['f1-score'].idxmin()}"
        f" F1 is only {A2_REPORT.drop(index=['accuracy', 'macro avg', 'weighted avg'])['f1-score'].min():.3f}.",
        ha="center",
        fontsize=6.2,
        color=COLORS["red"],
    )
    fig.text(
        0.5,
        0.018,
        "Both lanes are sequence-level, video-confounded, and encoder-exposed descriptive readouts.",
        ha="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    fig.subplots_adjust(bottom=0.25, top=0.82, left=0.09, right=0.98)
    save(fig, "evolution_class_f1")


def make_lane_c_repair() -> None:
    old = LEDGER.loc[LEDGER["version"].eq("superseded_five_fold_mean")].iloc[0]
    new = LEDGER.loc[LEDGER["version"].eq("corrected_two_fold_mean")].iloc[0]
    fig, ax = canvas((7.1, 3.75))
    title(
        ax,
        "Lane C improved because its evaluation definition was repaired",
        "The S-JEPA checkpoint and all 159 saved embeddings stayed fixed.",
    )
    card(
        ax,
        (0.006, 0.32, 0.40, 0.50),
        "Superseded: 5 ordinary group folds",
        f"accuracy {old.accuracy:.3f}\n"
        f"balanced accuracy {old.balanced_accuracy:.3f}\n"
        f"macro-F1 {old.macro_f1:.3f}\n\n"
        "One training fold had no cerebral-palsy rows.\n"
        "Macro-F1 label sets varied by fold.",
        "#F8E4DA",
        heading_size=7.3,
        body_size=5.9,
    )
    connector(ax, (0.406, 0.55), (0.594, 0.55), color=COLORS["blue"])
    # Both words sit above the shaft as a two-line block. The lower line used to be drawn at
    # y=0.585 with the shaft at y=0.57, so "new folds" was struck through by its own connector.
    ax.text(
        0.50,
        0.585,
        "same encoder\nnew folds",
        ha="center",
        va="bottom",
        fontsize=5.6,
        color=COLORS["blue"],
        linespacing=1.45,
    )
    card(
        ax,
        (0.594, 0.32, 0.40, 0.50),
        "Corrected: 2 stratified group folds",
        f"accuracy {new.accuracy:.3f}   (change {new.accuracy - old.accuracy:+.3f})\n"
        f"balanced accuracy {new.balanced_accuracy:.3f}   (change {new.balanced_accuracy - old.balanced_accuracy:+.3f})\n"
        f"macro-F1 {new.macro_f1:.3f}   (change {new.macro_f1 - old.macro_f1:+.3f})\n\n"
        "Every train and test fold contains all five labels.\n"
        "Macro-F1 always uses one fixed label order.",
        "#E4F2E8",
        heading_size=7.3,
        body_size=5.75,
    )
    # Both notes are wrapped. On one line each ran past the right edge of the canvas and only
    # survived because save() crops to a tight bounding box.
    ax.text(
        0.006,
        0.24,
        "Why only two folds? Parkinson's and cerebral palsy have two source videos each.\n"
        "Two is the largest fold count that keeps every class on both sides.",
        va="top",
        fontsize=6.0,
        color=COLORS["ink"],
        linespacing=1.5,
    )
    ax.text(
        0.006,
        0.11,
        "What did not improve: the encoder was trained once on all 159 rows.\n"
        "Classifier-level video grouping does not create an independent encoder test.",
        va="top",
        fontsize=6.0,
        color=COLORS["red"],
        linespacing=1.5,
    )
    save(fig, "evolution_lane_c_repair")


def make_readout_evolution() -> None:
    fig, ax = canvas((7.1, 3.7))
    title(
        ax,
        "The current 384-D readout is simple, but it compresses away temporal order",
        "The right side is a proposed controlled ablation. It has not produced a saved result in this repository.",
    )
    card(
        ax,
        (0.006, 0.45, 0.22, 0.36),
        "Encoded token grid",
        "16 time patches\nx 33 joints\nx 96 latent values\n\nvalid tokens only",
        "#E4EEF7",
        heading_size=7.0,
        body_size=6.0,
        center=True,
    )
    connector(ax, (0.226, 0.63), (0.282, 0.63))
    card(
        ax,
        (0.282, 0.45, 0.28, 0.36),
        "Current pooling",
        "global mean: 96\nglobal std: 96\n12-landmark mean: 96\n12-landmark std: 96\n\nTotal = 384 values",
        "#E4F2E8",
        heading_size=7.0,
        body_size=5.8,
        center=True,
    )
    connector(ax, (0.562, 0.63), (0.618, 0.63), color=COLORS["purple"], dashed=True)
    card(
        ax,
        (0.618, 0.45, 0.376, 0.36),
        "Planned dynamics ablation",
        "per-segment profiles\nleft-right trajectories\nraw duration and frame rate\ncadence or autocorrelation\nfit inside each outer fold",
        "#F2EAF7",
        heading_size=7.0,
        body_size=5.8,
        center=True,
    )
    # Clear of the card rather than straddling its bottom border, which the badge used to hide.
    pill(ax, 0.731, 0.372, "PROPOSED", COLORS["purple"], width=0.15, text_size=5.2)
    card(
        ax,
        (0.006, 0.07, 0.48, 0.25),
        "Why the current design is useful",
        "No trainable sequence head.\nLow variance at tiny sample size.\nEasy-to-audit frozen feature probe.",
        "#FFFFFF",
        heading_size=6.4,
        body_size=5.2,
    )
    card(
        ax,
        (0.514, 0.07, 0.48, 0.25),
        "What it cannot retain",
        "Order of the 16 time patches.\nAbsolute walking rate after resizing.\nWhen left-right asymmetry occurs.",
        "#F8E4DA",
        heading_size=6.4,
        body_size=5.2,
    )
    save(fig, "evolution_readout")


def make_status_map() -> None:
    fig, ax = canvas((7.1, 4.0))
    title(ax, "What is implemented, proposed, or explicitly rejected")
    group_xs, group_width = row_positions(3, gap=0.020)
    groups = [
        (
            group_xs[0],
            "Implemented and measured",
            [
                "12-landmark fixed\ntarget whitelist",
                "uniform batch-safe masking\nwith validity gates",
                "63 accepted added-normal\npose sequences",
                "five stages with replay,\nVICReg, and group loss",
                "fingerprinted checkpoints,\nembeddings, and result ledger",
                "A1, A2, and encoder-\ntransductive Lane C audits",
            ],
            "#E4F2E8",
            COLORS["green"],
        ),
        (
            group_xs[1],
            "Proposed next experiments",
            [
                "outer source-video split\nbefore all fitting",
                "fold-local five-stage\nrepresentation training",
                "source-balanced acquisition\nand provenance controls",
                "dynamics and raw-rate\nreadout ablations",
                "same-pose handcrafted,\nembedding, and fused comparison",
                "uncertainty from genuinely\nindependent outer folds",
            ],
            "#F2EAF7",
            COLORS["purple"],
        ),
        (
            group_xs[2],
            "Rejected interpretations",
            [
                "Lane C is an unseen-video\nencoder estimate",
                "every sample masks\nexactly 60 percent",
                "added normal labels are\ncanonical GAVD labels",
                "VICReg guarantees clean\nclinical clusters",
                "binary and five-class\nbar heights are comparable",
                "a fixed Random Forest can\nundo encoder exposure",
            ],
            "#F8E4DA",
            COLORS["red"],
        ),
    ]
    for x, heading, items, face, accent in groups:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.15), group_width, 0.68,
                boxstyle=f"round,pad={CARD_PAD},rounding_size=0.018",
                linewidth=1.0, edgecolor=accent, facecolor=face,
            )
        )
        ax.text(x + 0.016, 0.77, heading, va="top", fontsize=7.2, fontweight="bold", color=COLORS["navy"])
        y = 0.68
        for item in items:
            ax.text(x + 0.022, y, "•", va="top", fontsize=7.0, color=accent)
            ax.text(
                x + 0.045,
                y,
                item,
                va="top",
                fontsize=4.85,
                color=COLORS["ink"],
                linespacing=1.22,
            )
            y -= 0.092
    ax.text(
        0.02,
        0.055,
        "The archived improvement plan remains useful for hypotheses, but only the green column describes the completed checkpoint.",
        fontsize=6.1,
        color=COLORS["red"],
    )
    save(fig, "evolution_status_map")


if __name__ == "__main__":
    make_timeline()
    make_layer_matrix()
    make_masking_math()
    make_data_scale()
    make_augmentation_pipeline()
    make_objective()
    make_artifact_contract()
    make_checkpoint_lineage()
    make_classwise_results()
    make_lane_c_repair()
    make_readout_evolution()
    make_status_map()
    print(f"wrote 12 evolution figure sets for {CHECKPOINT_NAME} ({FINGERPRINT[:12]}...)")
