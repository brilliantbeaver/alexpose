"""Create the representation-methodology review's two accessible SVG diagrams.

Run from the experiment folder:
    .venv/bin/python scripts/scripts_make_representation_review_diagrams.py \
        --preview-dir /tmp/representation-review

The roles diagram describes implemented code separately from a proposed stage.
The weighting diagram reads the frozen fold-0 registry, rather than assuming
that the example's clip and source counts will remain unchanged. Percentages
describe assigned loss weights, not measured gradient contributions or accuracy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from matplotlib.patches import FancyBboxPatch, Rectangle
import matplotlib.pyplot as plt

from scripts_make_notebook03_diagrams import (
    AMBER, AMBER_BG, BLUE, BLUE_BG, Diagram, INK, LINE, MUTED, ROOT,
    TEAL, TEAL_BG,
)


plt.rcParams["svg.hashsalt"] = "representation-methodology-review"


def dashed_card(d, x, y, width, height):
    d.ax.add_patch(FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0,rounding_size=16",
        linewidth=1.4, linestyle=(0, (6, 4)),
        facecolor=AMBER_BG, edgecolor=AMBER,
    ))


def representation_roles(preview_dir):
    d = Diagram(
        1230,
        "The JEPA predictor and the condition classifier have different jobs",
        "Implemented: a student encoder and JEPA predictor learn hidden feature "
        "distributions from a stop-gradient, slow-moving teacher. The feature "
        "loss updates the student and predictor, while the teacher follows the "
        "student by an exponential moving average. Separately, the frozen teacher "
        "encodes full windows; token and window means yield one clip vector. "
        "Training-only StandardScaler and class-balanced logistic regression "
        "produce Normal, MS and PD predictions. Condition labels fit the classifier, "
        "not the encoder. A separate dashed amber strip marks supervised encoder "
        "adaptation as proposed, not implemented.",
    )
    d.header(
        "NOTEBOOKS 03–06  /  TWO DIFFERENT PREDICTION TASKS",
        "Learn movement features, then learn condition labels",
        "The JEPA predictor guesses hidden features. A separate classifier predicts Normal, MS or PD.",
    )

    # Lane 1: only feature prediction trains the implemented encoder.
    d.card(36, 145, 1128, 425)
    d.text(58, 166, "01  Learn movement features", 18, INK, "bold")
    d.text(1142, 171, "03–04  ·  LABEL-FREE", 11.5, BLUE, "bold", ha="right")

    d.card(58, 231, 208, 258)
    d.text(77, 249, "Training window", 14, INK, "bold")
    d.lines(77, 283, ["All three conditions", "Labels withheld"], size=12.5, color=MUTED)
    d.lines(77, 348, ["Masked +", "transformed view"], size=12.5, color=BLUE, step=23)
    d.lines(77, 426, ["Complete original", "window"], size=12.5, color=TEAL, step=23)

    d.card(302, 231, 237, 119, BLUE_BG, "#C8D6F4")
    d.text(321, 250, "Student encoder", 16, BLUE, "bold")
    d.lines(321, 286, ["Reads visible tokens", "Updated by gradients"],
            size=12, color=MUTED, step=26)

    d.card(579, 231, 237, 119, BLUE_BG, "#C8D6F4")
    d.text(598, 250, "JEPA predictor", 16, BLUE, "bold")
    d.lines(598, 286, ["Guesses hidden features", "96 features per token*"],
            size=12, color=MUTED, step=26)

    d.card(302, 404, 237, 105, TEAL_BG, "#BBDEDA")
    d.text(321, 422, "Slow teacher", 16, TEAL, "bold")
    d.lines(321, 454, ["Reads the full window", "No gradient from loss"],
            size=12, color=MUTED, step=24)

    d.card(856, 231, 286, 278, AMBER_BG, "#E5CFB9")
    d.text(877, 252, "Feature loss", 19, AMBER, "bold")
    d.lines(877, 294, ["Compare predictions with", "teacher targets at the", "hidden joint–time slots."],
            size=13, color=MUTED, step=28)
    d.lines(877, 406, ["Cross-entropy over", "feature dimensions;", "no condition labels."],
            size=13, color=AMBER, step=28)

    d.arrow((266, 364), (296, 323), BLUE)
    d.arrow((539, 291), (573, 291), BLUE)
    d.arrow((816, 291), (850, 291), BLUE)
    d.arrow((266, 456), (296, 456), TEAL)
    d.arrow((421, 350), (421, 398), TEAL)
    d.text(438, 371, "slow weight copy (EMA)", 11, TEAL)
    d.arrow((539, 456), (850, 456), TEAL)
    d.text(697, 426, "Target features", 13, TEAL, "bold", ha="center")
    d.text(58, 535, "The feature loss updates the student and JEPA predictor. It does not teach three class labels.",
           12.5, MUTED)

    # Lane 2: the deployed feature representation comes from the EMA teacher.
    d.card(36, 595, 1128, 327)
    d.text(58, 616, "02  Train the condition classifier", 18, INK, "bold")
    d.text(1142, 621, "04 + 06  ·  SUPERVISED HEAD", 11.5, TEAL, "bold", ha="right")
    d.text(58, 654, "Reuse the learned teacher. Fit the scaler and classifier on training clips only.", 13, MUTED)

    boxes = [
        (58, 262, TEAL_BG, "#BBDEDA"),
        (357, 180, "white", LINE),
        (574, 282, BLUE_BG, "#C8D6F4"),
        (893, 249, "white", LINE),
    ]
    for x, w, fill, edge in boxes:
        d.card(x, 704, w, 125, fill, edge)
    d.text(78, 723, "Frozen teacher", 16, TEAL, "bold")
    d.lines(78, 760, ["Full windows → pooled", "96-number clip vector*"], size=12.5, color=MUTED)
    d.text(447, 723, "StandardScaler", 13.5, INK, "bold", ha="center")
    d.lines(377, 760, ["Put features on", "comparable scales"], size=12, color=MUTED)
    d.text(594, 723, "Logistic regression", 16, BLUE, "bold")
    d.lines(594, 760, ["Learns the class boundaries", "Balanced class weights"], size=12.5, color=MUTED)
    d.text(1017, 723, "Condition prediction", 15, INK, "bold", ha="center")
    d.text(1017, 767, "Normal · MS · PD", 17, TEAL, "bold", ha="center")
    for a, b in [(320, 351), (537, 568), (856, 887)]:
        d.arrow((a, 766), (b, 766))
    d.arrow((716, 881), (716, 836), BLUE)
    d.text(737, 851, "Training condition labels", 13, BLUE, "bold")
    d.text(78, 859, "Labels fit the classifier only.", 13, INK, "bold")
    d.text(78, 886, "The encoder stays frozen in this step.", 12.5, MUTED)

    # This proposal is visually isolated, not a new arrow in the current pipeline.
    dashed_card(d, 36, 947, 1128, 225)
    d.text(58, 968, "03  Proposed: let labels refine the encoder", 17, AMBER, "bold")
    d.text(1142, 973, "NOT IMPLEMENTED", 11.5, AMBER, "bold", ha="right")
    d.card(58, 1020, 414, 89, "white", "#E5CFB9")
    d.text(78, 1036, "Trainable encoder + class head", 14.5, INK, "bold")
    d.text(78, 1072, "Adapt features using training labels", 12.5, MUTED)
    d.card(748, 1020, 394, 89, "white", "#E5CFB9")
    d.text(768, 1036, "Supervised classification loss", 14.5, AMBER, "bold")
    d.text(768, 1072, "Compare class predictions with labels", 12.5, MUTED)
    d.arrow((740, 1066), (480, 1066), AMBER, dashed=True)
    d.text(610, 1036, "label gradients", 12.5, AMBER, "bold", ha="center")
    d.text(58, 1133, "This would directly train class-relevant features; improvement must be tested on unseen sources.",
           12.5, MUTED)

    d.text(36, 1194, "*Laptop configuration. Diagram 01 shows learning; diagram 02 shows fitting and using a frozen-encoder classifier.",
           10.7, MUTED)
    d.save("representation_review_roles", preview_dir)


def source_counts():
    registry_path = ROOT / "artifacts/eval/full-v1/fold_registry.json"
    registry = json.loads(registry_path.read_text())
    fold = registry["folds"][0]
    selected = set(fold["train_clips"])
    ms = [r for r in registry["inventory"]["cache"]
          if r["clip"] in selected and r["label"] == "ms"]
    counts = Counter(r["source_id"] for r in ms)
    if not counts:
        raise ValueError("Fold 0 has no MS training sources")
    return len(ms), len(counts), max(counts.values()), len(fold["train_sources"]), registry["registry_sha256"]


def representation_source_weights(preview_dir):
    n_clips, n_sources, largest, all_sources, digest = source_counts()
    current_share, proposed_share = largest / n_clips, 1 / n_sources
    d = Diagram(
        902,
        "Class balance does not guarantee source balance",
        f"Frozen registry {digest}. Fold 0 has {n_sources} MS training sources "
        f"and {n_clips} MS clips; one source provides {largest} clips. Current "
        f"class-balanced logistic regression gives every MS clip the same loss "
        f"weight, so this source accounts for {current_share:.1%} of the MS part "
        f"of the assigned data-loss weights. Proposed source-and-class weighting "
        f"would give each MS source {proposed_share:.1%}, divided across its clips. "
        f"The self-supervised encoder already samples all {all_sources} training "
        "sources uniformly. These weights do not measure accuracy or actual "
        "gradient influence. The probe change is not implemented.",
    )
    d.header(
        "NOTEBOOKS 04 + 06  /  WHO GETS A VOTE IN CLASSIFIER TRAINING?",
        "Class balance does not guarantee source balance",
        f"Fold 0 training data · MS: {n_sources} sources, {n_clips} clips · one source contributes {largest} clips.",
    )

    d.card(36, 154, 552, 530)
    dashed_card(d, 612, 154, 552, 530)
    d.text(58, 177, "CURRENT CLASSIFIER", 11.5, BLUE, "bold")
    d.text(634, 177, "PROPOSED CLASSIFIER CHANGE", 11.5, AMBER, "bold")
    d.text(58, 210, "One equal weight per MS clip", 18, INK, "bold")
    d.text(634, 210, "Equal total per MS source", 16.5, INK, "bold")
    d.lines(58, 256, ["Class balancing scales all MS clips equally.",
                      f"A source with {largest} clips keeps {largest} equal votes."],
            size=13, color=MUTED, step=28)
    d.lines(634, 256, ["Keep the condition total fixed.",
                       "Share it equally among its sources."],
            size=12.5, color=MUTED, step=28)

    for x, share, formula, color in [
        (58, current_share, f"{largest} ÷ {n_clips}", BLUE),
        (634, proposed_share, f"1 ÷ {n_sources}", AMBER),
    ]:
        d.text(x, 324, f"{share:.1%}", 42, color, "bold")
        d.text(x + 266, 346, formula, 20, color)
        d.text(x, 391, "of assigned MS loss weight", 12.5, MUTED)

    # Equal-width bars use the same 0–100% scale; their blue/amber lengths are exact.
    bx, by, bw, bh = 58, 436, 508, 47
    d.ax.add_patch(Rectangle((bx, by), bw, bh, facecolor="#E0E8F2", edgecolor="none"))
    d.ax.add_patch(Rectangle((bx, by), bw * current_share, bh, facecolor=BLUE, edgecolor="none"))
    d.text(bx + bw * current_share / 2, by + 14, "one source", 12, "white", "bold", ha="center")
    d.text(bx + bw * (1 + current_share) / 2, by + 14, f"other {n_sources - 1} sources", 12,
           MUTED, ha="center")

    bx = 634
    segment = bw / n_sources
    for i in range(n_sources):
        d.ax.add_patch(Rectangle((bx + i * segment, by), segment, bh,
                                facecolor=AMBER if i == 0 else "#EAD9C6",
                                edgecolor="white", linewidth=1.5))
    d.text(bx + segment / 2, by + 14, "1", 12, "white", "bold", ha="center")
    d.text(bx + segment + (bw - segment) / 2, by + 14, f"{n_sources - 1} others", 11,
           MUTED, ha="center")

    d.text(58, 511, f"{n_clips} clips: each square has the same weight", 12.5, INK, "bold")
    slot = 508 / n_clips
    for i in range(n_clips):
        d.ax.add_patch(Rectangle((58 + i * slot, 549), slot - 3, 26,
                                facecolor=BLUE if i < largest else "#CED9E7", edgecolor="none"))
    d.lines(58, 602, [f"Blue: {largest} clips from the same source.",
                      "More related clips mean more assigned weight."],
            size=12.5, color=MUTED, step=27)

    d.text(634, 511, f"{n_sources} sources: equal total per source", 12.5, INK, "bold")
    d.lines(634, 551, ["Divide source total across clips.",
                       "More clips do not add weight."],
            size=12.5, color=MUTED, step=27)
    d.text(634, 627, "Test this rule using training + validation.", 11.5, AMBER)

    d.card(36, 710, 1128, 128, TEAL_BG, "#BBDEDA")
    d.text(58, 730, "Already implemented for self-supervised encoder training", 17, TEAL, "bold")
    d.text(58, 772, "SSL samples source videos uniformly, regardless of how many windows they provide.", 13.5, MUTED)
    d.text(58, 801, f"In fold 0, each source has probability 1/{all_sources} per draw. The classifier is fitted in a separate step.",
           13, MUTED)
    d.text(36, 862, "Percentages compare assigned loss weights within MS. They are not accuracy or measured influence on fitted coefficients.",
           10.6, MUTED)
    d.save("representation_review_source_weights", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    representation_roles(args.preview_dir)
    representation_source_weights(args.preview_dir)
