"""Generate the eight tutorial diagrams as clean SVG files.

We draw with matplotlib rather than hand-writing SVG so text is measured and
placed automatically, spacing stays consistent, and nothing overlaps. Each figure
uses one hue per class (normal blue, ms orange, pd green), generous whitespace,
and short labels. Run:  python scripts_make_diagrams.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle

OUT = Path(__file__).resolve().parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREEN = "#2b6cb0", "#dd6b20", "#38a169"
BLUE_BG, ORANGE_BG, GREEN_BG = "#ebf4ff", "#fffaf0", "#f0fff4"
INK, MUTE, LINE = "#1a202c", "#718096", "#a0aec0"

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def _fig(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100 * h / w)
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, fc, ec, fs=9, tc=None, weight="normal", sub=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15,rounding_size=1.2",
                                fc=fc, ec=ec, lw=1.4))
    tc = tc or ec
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fs,
                color=tc, weight=weight)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=fs - 1.5,
                color=MUTE)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=LINE, style="-|>", lw=1.6, rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, color=color,
                                 lw=lw, mutation_scale=12,
                                 connectionstyle=f"arc3,rad={rad}"))


def save(fig, name):
    fig.savefig(OUT / name, format="svg", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------
# 1. Pipeline flowchart
# ---------------------------------------------------------------------------
def pipeline_flowchart():
    fig, ax = _fig(11, 6.6)
    H = 100 * 6.6 / 11
    ax.text(50, H - 3, "From walking video to a health-condition label",
            ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(44, H - 7.5, "Both branches share one pose front-end and the same test set",
            ha="center", fontsize=9.5, color=MUTE)

    # Shared front-end, centered vertically, well below the title.
    y = H - 28
    box(ax, 4, y, 18, 11, "Walking video", BLUE_BG, BLUE, 9.5, weight="bold", sub="49 mp4 clips")
    box(ax, 27, y, 18, 11, "MediaPipe pose", BLUE_BG, BLUE, 9.5, weight="bold", sub="33 landmarks")
    box(ax, 50, y, 18, 11, "Normalize", BLUE_BG, BLUE, 9.5, weight="bold", sub="pelvis + torso")
    arrow(ax, 22, y + 5.5, 27, y + 5.5, BLUE)
    arrow(ax, 45, y + 5.5, 50, y + 5.5, BLUE)

    # A single split node, then two branches stacked on the right.
    split_x = 71
    arrow(ax, 68, y + 5.5, split_x, y + 5.5, BLUE)
    ya = y + 10
    yb = y - 18
    box(ax, 74, ya, 23, 12, "Branch A\nS-JEPA", ORANGE_BG, ORANGE, 9, weight="bold",
        sub="encoder + probe")
    box(ax, 74, yb, 23, 12, "Branch B\nRandom Forest", GREEN_BG, GREEN, 9, weight="bold",
        sub="82 gait features")
    arrow(ax, split_x, y + 5.5, 74, ya + 6, ORANGE, rad=0.3)
    arrow(ax, split_x, y + 5.5, 74, yb + 6, GREEN, rad=-0.3)

    # Converge into a comparison node on the LEFT-bottom, clear of the branches.
    yc = y - 18
    box(ax, 20, yc, 26, 12, "Compare fairly", "#edf2f7", "#4a5568", 9.5, weight="bold",
        sub="same test videos, same metrics")
    arrow(ax, 74, ya + 3, 46, yc + 9, LINE, rad=0.25)
    arrow(ax, 74, yb + 6, 46, yc + 4, LINE, rad=0.15)
    save(fig, "pipeline_flowchart.svg")


# ---------------------------------------------------------------------------
# 2. S-JEPA two-lane training
# ---------------------------------------------------------------------------
def sjepa_two_lane():
    fig, ax = _fig(11, 5.6)
    H = 100 * 5.6 / 11
    ax.text(50, H - 3, "How S-JEPA learns: predict hidden joints in feature space",
            ha="center", fontsize=14, weight="bold", color=INK)

    # Prediction lane (top)
    yt = H - 24
    ax.text(4, yt + 14, "Prediction lane", fontsize=10, weight="bold", color=ORANGE)
    box(ax, 3, yt, 15, 10, "Masked view", "#fff", ORANGE, 8.5, tc=INK, sub="visible joints")
    box(ax, 22, yt, 16, 10, "View encoder", ORANGE_BG, ORANGE, 8.5, weight="bold")
    box(ax, 42, yt, 16, 10, "Predictor", ORANGE_BG, ORANGE, 8.5, weight="bold")
    box(ax, 62, yt, 17, 10, "Predicted\nfeatures", "#fff", ORANGE, 8.5, tc=INK)
    arrow(ax, 18, yt + 5, 22, yt + 5, ORANGE)
    arrow(ax, 38, yt + 5, 42, yt + 5, ORANGE)
    arrow(ax, 58, yt + 5, 62, yt + 5, ORANGE)

    # Target lane (bottom)
    yb = 8
    ax.text(4, yb - 3.5, "Target lane (slow teacher, no gradient)", fontsize=10, weight="bold", color=BLUE)
    box(ax, 3, yb, 15, 10, "Full skeleton", "#fff", BLUE, 8.5, tc=INK, sub="all joints")
    box(ax, 22, yb, 16, 10, "Target encoder", BLUE_BG, BLUE, 8.5, weight="bold", sub="EMA copy")
    box(ax, 42, yb, 16, 10, "Mask output", BLUE_BG, BLUE, 8.5, weight="bold")
    box(ax, 62, yb, 17, 10, "Target\nfeatures", "#fff", BLUE, 8.5, tc=INK, sub="center + sharpen")
    arrow(ax, 18, yb + 5, 22, yb + 5, BLUE)
    arrow(ax, 38, yb + 5, 42, yb + 5, BLUE)
    arrow(ax, 58, yb + 5, 62, yb + 5, BLUE)

    # loss node on the right connecting both lanes
    box(ax, 84, (yt + yb) / 2 + 1, 13, 12, "Latent\ncross-entropy", "#edf2f7", "#4a5568", 8.5, weight="bold")
    arrow(ax, 79, yt + 5, 87, (yt + yb) / 2 + 9, LINE, rad=0.2)
    arrow(ax, 79, yb + 5, 87, (yt + yb) / 2 + 4, LINE, rad=-0.2)

    # EMA feedback arrow (view -> target), dashed
    ax.add_patch(FancyArrowPatch((30, yt), (30, yb + 10), arrowstyle="-|>",
                 color=MUTE, lw=1.3, ls=(0, (4, 3)), mutation_scale=11,
                 connectionstyle="arc3,rad=0.35"))
    ax.text(37, (yt + yb) / 2 + 5, "EMA update\n(slow copy)", fontsize=8, color=MUTE, ha="left")
    save(fig, "sjepa_two_lane.svg")


# ---------------------------------------------------------------------------
# 3. Anatomical mask on the skeleton
# ---------------------------------------------------------------------------
def anatomical_mask():
    # BlazePose-33 2D layout (upright figure). Arms hang down and outward so the
    # elbow, wrist, and hand joints separate cleanly. y grows upward here.
    coords = {
        0: (50, 84),                       # nose
        7: (45, 82), 8: (55, 82),          # ears
        9: (48, 80), 10: (52, 80),         # mouth
        11: (40, 72), 12: (60, 72),        # shoulders
        13: (33, 60), 14: (67, 60),        # elbows
        15: (27, 49), 16: (73, 49),        # wrists
        17: (24, 45), 18: (76, 45),        # pinky
        19: (27, 44), 20: (73, 44),        # index
        21: (30, 47), 22: (70, 47),        # thumb
        23: (44, 52), 24: (56, 52),        # hips
        25: (42, 34), 26: (58, 34),        # knees
        27: (41, 17), 28: (59, 17),        # ankles
        29: (38, 12), 30: (62, 12),        # heels
        31: (45, 9), 32: (55, 9),          # foot index
    }
    conns = [(11,12),(11,23),(12,24),(23,24),
             (11,13),(13,15),(15,17),(15,19),(15,21),(12,14),(14,16),(16,18),(16,20),(16,22),
             (23,25),(25,27),(27,29),(27,31),(29,31),(24,26),(26,28),(28,30),(28,32),(30,32),
             (0,9),(0,10),(9,10)]
    masked = {11,12,23,24,25,26,27,28,29,30,31,32}

    fig, ax = _fig(8, 8.0)
    H = 100 * 8.0 / 8
    ax.set_ylim(0, H)
    ax.text(50, H - 3, "The anatomical mask", ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(50, H - 7.5, "We hide 12 fixed joints, both shoulders and both legs, and predict them",
            ha="center", fontsize=9.2, color=MUTE)

    yoff = 4  # lift figure clear of the legend row; canvas top clears the head
    for a, b in conns:
        if a in coords and b in coords:
            xa, ya = coords[a]; xb, yb = coords[b]
            ax.plot([xa, xb], [ya + yoff, yb + yoff], color="#cbd5e0", lw=2, zorder=1)
    for i, (x, yv) in coords.items():
        if i in masked:
            ax.add_patch(Circle((x, yv + yoff), 1.7, fc=ORANGE, ec="white", lw=0.8, zorder=3))
        else:
            ax.add_patch(Circle((x, yv + yoff), 1.1, fc="#4a5568", ec="white", lw=0.6, zorder=2))

    # legend along the bottom, well below the feet (feet are at y ~ 9 + yoff = 13)
    ax.add_patch(Circle((20, 6), 1.7, fc=ORANGE, ec="white", lw=0.8))
    ax.text(24, 6, "masked target joints (12)", va="center", fontsize=9, color=INK)
    ax.add_patch(Circle((20, 2.5), 1.1, fc="#4a5568", ec="white", lw=0.6))
    ax.text(24, 2.5, "visible context joints (face and arms)", va="center", fontsize=9, color=INK)
    save(fig, "anatomical_mask.svg")


# ---------------------------------------------------------------------------
# 4. Tokenization
# ---------------------------------------------------------------------------
def tokenization():
    fig, ax = _fig(10, 5.0)
    H = 100 * 5.0 / 10
    ax.text(50, H - 3, "Tokenizing a skeleton window", ha="center", fontsize=14, weight="bold", color=INK)
    ax.text(50, H - 7.5, "Group l = 4 adjacent frames of one joint into a single token",
            ha="center", fontsize=9.5, color=MUTE)

    # left: frame x joint grid
    gx, gy, cell = 6, 8, 3.2
    nframes, njoints = 8, 6
    ax.text(gx + nframes*cell/2, gy + njoints*cell + 3.5, "frames x joints", ha="center",
            fontsize=9, color=MUTE)
    for f in range(nframes):
        for j in range(njoints):
            grp = f // 4
            fc = "#bee3f8" if grp == 0 else "#c6f6d5"
            ax.add_patch(Rectangle((gx + f*cell, gy + j*cell), cell*0.9, cell*0.9,
                                   fc=fc, ec="#a0aec0", lw=0.6))
    # brackets showing 4-frame groups
    ax.plot([gx, gx + 4*cell*0.97], [gy - 1.4, gy - 1.4], color=BLUE, lw=1.6)
    ax.plot([gx + 4*cell, gx + 8*cell*0.97], [gy - 1.4, gy - 1.4], color=GREEN, lw=1.6)
    ax.text(gx + 2*cell, gy - 3.6, "block 1", ha="center", fontsize=8, color=BLUE)
    ax.text(gx + 6*cell, gy - 3.6, "block 2", ha="center", fontsize=8, color=GREEN)

    # arrow to tokens
    arrow(ax, gx + nframes*cell + 2, gy + njoints*cell/2, 60, gy + njoints*cell/2, MUTE)

    # right: token column with pos-emb bars
    tx = 63
    for k in range(6):
        yv = gy + k*4
        ax.add_patch(FancyBboxPatch((tx, yv), 20, 3, boxstyle="round,pad=0.05,rounding_size=0.4",
                     fc="#edf2f7", ec="#a0aec0", lw=0.8))
    ax.text(tx + 10, gy + 6*4 + 2.5, "tokens (dim d)", ha="center", fontsize=9, color=MUTE)
    # pos emb bars, spaced so their labels do not touch
    bar1_x = tx + 22.5
    bar2_x = bar1_x + 5.5
    ax.add_patch(Rectangle((bar1_x, gy), 2.6, 6*4 - 1, fc="#fbd38d", ec=ORANGE, lw=0.8))
    ax.text(bar1_x + 1.3, gy - 3.0, "spatial", ha="center", fontsize=7.5, color="#9c4221")
    ax.add_patch(Rectangle((bar2_x, gy), 2.6, 6*4 - 1, fc="#9ae6b4", ec=GREEN, lw=0.8))
    ax.text(bar2_x + 1.3, gy - 3.0, "temporal", ha="center", fontsize=7.5, color="#276749")
    ax.text((bar1_x + bar2_x) / 2 + 1.3, gy + 6*4 + 2.5, "+ position", ha="center",
            fontsize=8.5, color=MUTE)
    save(fig, "tokenization.svg")


# ---------------------------------------------------------------------------
# 5. Progressive training timeline
# ---------------------------------------------------------------------------
def progressive_timeline():
    fig, ax = _fig(10, 3.6)
    H = 100 * 3.6 / 10
    ax.text(50, H - 3, "Progressive training", ha="center", fontsize=14, weight="bold", color=INK)

    y = 12
    ax.plot([6, 94], [y, y], color=LINE, lw=2)
    stages = [
        (20, "Phase 1", "Pretrain on\nnormal gait", BLUE_BG, BLUE),
        (50, "Phase 2", "Add ms and pd\nsequences", ORANGE_BG, ORANGE),
        (80, "Phase 3", "Add VICReg to\nseparate classes", GREEN_BG, GREEN),
    ]
    for x, tag, desc, fc, ec in stages:
        ax.add_patch(Circle((x, y), 2.0, fc=ec, ec="white", lw=1, zorder=3))
        box(ax, x - 13, y + 6, 26, 13, desc, fc, ec, 9, tc=INK, weight="normal")
        ax.text(x, y - 5, tag, ha="center", fontsize=9, weight="bold", color=ec)
    for x in (35, 65):
        arrow(ax, x - 8, y, x + 8, y, LINE)
    ax.text(50, 2.5, "capacity and robustness grow left to right", ha="center", fontsize=8.5, color=MUTE)
    save(fig, "progressive_timeline.svg")


# ---------------------------------------------------------------------------
# 6. VICReg cluster separation
# ---------------------------------------------------------------------------
def vicreg_clusters():
    import numpy as np
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    # left: tangled
    for c, col in zip(range(3), [BLUE, ORANGE, GREEN]):
        pts = rng.normal(5, 1.6, size=(20, 2))
        axes[0].scatter(pts[:, 0], pts[:, 1], s=28, color=col, alpha=0.7, edgecolors="white", lw=0.4)
    axes[0].set_title("Without VICReg\nclasses overlap", fontsize=11, color=INK)

    # right: separated
    centers = [(3, 3.4), (7, 3.2), (5, 7.4)]
    for (cx, cy), col in zip(centers, [BLUE, ORANGE, GREEN]):
        pts = rng.normal([cx, cy], 0.7, size=(20, 2))
        axes[1].scatter(pts[:, 0], pts[:, 1], s=28, color=col, alpha=0.85, edgecolors="white", lw=0.4)
    axes[1].set_title("With VICReg\nclasses pull apart", fontsize=11, color=INK)

    # three-term note under the right panel
    axes[1].text(5, 0.6, "variance keeps spread  |  invariance matches views  |  covariance decorrelates",
                 ha="center", fontsize=7.6, color=MUTE)
    # shared legend
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l, markeredgecolor="white")
               for c, l in [(BLUE, "normal"), (ORANGE, "ms"), (GREEN, "pd")]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("VICReg keeps the three condition clusters apart", fontsize=13.5,
                 weight="bold", color=INK, y=1.02)
    fig.savefig(OUT / "vicreg_clusters.svg", format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote vicreg_clusters.svg")


# ---------------------------------------------------------------------------
# 7. Grouped split (leakage-safe)
# ---------------------------------------------------------------------------
def grouped_split():
    fig, ax = _fig(10, 4.2)
    H = 100 * 4.2 / 10
    ax.text(50, H - 3, "Leakage-safe splitting by source video", ha="center",
            fontsize=14, weight="bold", color=INK)
    ax.text(50, H - 7.5, "All clips from one YouTube source stay on the same side of the split",
            ha="center", fontsize=9.5, color=MUTE)

    # source A: 3 clips -> train ; source B: 2 clips -> test
    def clip_row(x0, y0, n, col, bg):
        for i in range(n):
            ax.add_patch(FancyBboxPatch((x0 + i*7.5, y0), 6.5, 5,
                         boxstyle="round,pad=0.1,rounding_size=0.6", fc=bg, ec=col, lw=1.1))
            ax.text(x0 + i*7.5 + 3.2, y0 + 2.5, f"clip {i+1}", ha="center", va="center",
                    fontsize=7.5, color=col)

    ax.text(8, 24, "Source A", fontsize=9.5, weight="bold", color=BLUE)
    clip_row(8, 17, 3, BLUE, BLUE_BG)
    ax.text(8, 12, "Source B", fontsize=9.5, weight="bold", color=ORANGE)
    clip_row(8, 5, 2, ORANGE, ORANGE_BG)

    # train / test bins
    box(ax, 62, 16, 15, 8, "TRAIN", "#e6fffa", "#2c7a7b", 10, weight="bold")
    box(ax, 62, 4, 15, 8, "TEST", "#fef5e7", "#b7791f", 10, weight="bold")
    arrow(ax, 32, 19.5, 62, 20, BLUE, rad=0.05)
    arrow(ax, 24, 7.5, 62, 8, ORANGE, rad=-0.05)
    ax.text(82, 20, "whole\nsource A", fontsize=8, color="#2c7a7b", va="center")
    ax.text(82, 8, "whole\nsource B", fontsize=8, color="#b7791f", va="center")
    save(fig, "grouped_split.svg")


# ---------------------------------------------------------------------------
# 8. RF vs S-JEPA comparison schematic
# ---------------------------------------------------------------------------
def rf_vs_sjepa():
    fig, ax = _fig(10, 4.6)
    H = 100 * 4.6 / 10
    ax.text(50, H - 3, "A fair head-to-head comparison", ha="center", fontsize=14,
            weight="bold", color=INK)

    box(ax, 5, 26, 30, 12, "S-JEPA linear probe", ORANGE_BG, ORANGE, 10, weight="bold",
        sub="frozen learned features")
    box(ax, 5, 8, 30, 12, "Random Forest", GREEN_BG, GREEN, 10, weight="bold",
        sub="82 hand-made features")
    box(ax, 46, 17, 22, 12, "Same test\nvideos", "#edf2f7", "#4a5568", 10, weight="bold")
    box(ax, 76, 17, 20, 12, "Accuracy,\nmacro F1", "#edf2f7", "#4a5568", 9.5, weight="bold",
        sub="mean +/- std")
    arrow(ax, 35, 32, 46, 25, ORANGE, rad=-0.15)
    arrow(ax, 35, 14, 46, 21, GREEN, rad=0.15)
    arrow(ax, 68, 23, 76, 23, LINE)
    ax.text(50, 4, "grouped k-fold over source videos, identical folds for both models",
            ha="center", fontsize=8.5, color=MUTE)
    save(fig, "rf_vs_sjepa.svg")


# ---------------------------------------------------------------------------
# 9. Why it matters (clinical motivation) - for the progress doc
# ---------------------------------------------------------------------------
def why_it_matters():
    fig, ax = _fig(10, 4.4)
    H = 100 * 4.4 / 10
    ax.text(50, H - 3, "Why learn gait from video", ha="center", fontsize=14, weight="bold", color=INK)

    steps = [
        (16, "A phone video", "no wearable sensors,\nno lab, no markers", BLUE_BG, BLUE),
        (50, "Skeleton motion", "how the joints move\nover time", ORANGE_BG, ORANGE),
        (84, "Early, cheap signal", "screening and progress\ntracking for MS and PD", GREEN_BG, GREEN),
    ]
    y = 18
    for x, title, desc, fc, ec in steps:
        box(ax, x - 14, y, 28, 15, title, fc, ec, 10.5, weight="bold", sub=desc)
    for x in (33, 67):
        arrow(ax, x - 5, y + 7.5, x + 5, y + 7.5, LINE)
    ax.text(50, 8, "Gait changes are among the earliest signs of MS and PD.",
            ha="center", fontsize=9.5, color=INK)
    ax.text(50, 3.5, "Making that signal readable from ordinary video could widen access to monitoring.",
            ha="center", fontsize=9, color=MUTE)
    save(fig, "why_it_matters.svg")


# ---------------------------------------------------------------------------
# 10. Project status (what is done)
# ---------------------------------------------------------------------------
def project_status():
    fig, ax = _fig(10, 5.2)
    H = 100 * 5.2 / 10
    ax.text(50, H - 3, "What is built and verified", ha="center", fontsize=14, weight="bold", color=INK)

    items = [
        "Raw-video pose pipeline (cached)",
        "S-JEPA: encoder, teacher, predictor",
        "Fixed clinical mask of 12 joints",
        "Centering + sharpening CE loss",
        "VICReg extension for separation",
        "Progressive normal then MS + PD",
        "Two profiles, both tested",
        "Random Forest baseline (exp5)",
        "Leakage-safe grouped k-fold",
        "Seven notebooks + slides + docs",
    ]
    y0 = H - 12
    for i, it in enumerate(items):
        col = 0 if i < 5 else 1
        row = i % 5
        x = 6 + col * 50
        yv = y0 - row * 7.5
        # green check disc with a tick
        ax.add_patch(Circle((x + 1.5, yv + 1.5), 1.7, fc=GREEN, ec="white", lw=0.6, zorder=2))
        ax.plot([x + 0.7, x + 1.35, x + 2.5], [yv + 1.6, yv + 0.7, yv + 2.5],
                color="white", lw=1.5, zorder=3, solid_capstyle="round")
        ax.text(x + 5, yv + 1.5, it, va="center", fontsize=9, color=INK)
    save(fig, "project_status.svg")


# ---------------------------------------------------------------------------
# 11. Results bar chart (RF vs S-JEPA, mean +/- std)
# ---------------------------------------------------------------------------
def results_bars():
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    metrics = ["accuracy", "macro F1"]
    rf_mean, rf_std = [0.66, 0.67], [0.09, 0.10]
    sj_mean, sj_std = [0.57, 0.57], [0.10, 0.11]
    x = np.arange(len(metrics)); w = 0.34
    ax.bar(x - w/2, rf_mean, w, yerr=rf_std, capsize=5, color=GREEN, label="Random Forest",
           edgecolor="white")
    ax.bar(x + w/2, sj_mean, w, yerr=sj_std, capsize=5, color=ORANGE, label="S-JEPA probe",
           edgecolor="white")
    ax.axhline(1/3, ls="--", lw=1, color=MUTE)
    ax.text(1.35, 1/3 + 0.015, "chance (3 classes)", fontsize=8, color=MUTE, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Grouped k-fold results (laptop profile)", fontsize=13, weight="bold", color=INK)
    ax.legend(frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.02, "Small dataset: treat as a methodology demo, not a clinical result.",
             ha="center", fontsize=8.5, color=MUTE)
    fig.savefig(OUT / "results_bars.svg", format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote results_bars.svg")


# ---------------------------------------------------------------------------
# 12. Roadmap (what is next)
# ---------------------------------------------------------------------------
def roadmap():
    fig, ax = _fig(10, 4.6)
    H = 100 * 4.6 / 10
    ax.text(50, H - 3, "What comes next", ha="center", fontsize=14, weight="bold", color=INK)

    lanes = [
        ("More data", "more clips per class and\nmore independent sources", BLUE_BG, BLUE),
        ("Bigger model", "run the gpu profile and\nlonger pretraining", ORANGE_BG, ORANGE),
        ("Transfer", "pretrain on large action\ndatasets, fine-tune on gait", GREEN_BG, GREEN),
        ("Clinical validity", "compare masks, add gait\nphases, expert review", "#faf5ff", "#805ad5"),
    ]
    y = 16
    wbox = 22
    for i, (title, desc, fc, ec) in enumerate(lanes):
        x = 3 + i * 24
        box(ax, x, y, wbox, 16, title, fc, ec, 10, weight="bold", sub=desc)
        ax.text(x + wbox/2, y - 3.5, f"{i+1}", ha="center", fontsize=10, weight="bold", color=ec)
    ax.annotate("", xy=(96, 8), xytext=(4, 8),
                arrowprops=dict(arrowstyle="-|>", color=LINE, lw=1.6))
    ax.text(50, 4, "near term  to  longer term", ha="center", fontsize=9, color=MUTE)
    save(fig, "roadmap.svg")


if __name__ == "__main__":
    pipeline_flowchart()
    sjepa_two_lane()
    anatomical_mask()
    tokenization()
    progressive_timeline()
    vicreg_clusters()
    grouped_split()
    rf_vs_sjepa()
    why_it_matters()
    project_status()
    results_bars()
    roadmap()
    print("\nAll diagrams written to", OUT)
