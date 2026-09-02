#!/usr/bin/env python
"""Generate the vector figures for docs/neurips-brain-body.md.

Data-driven figures read the real artifacts under work/artifacts/real and
degrade gracefully (canonical-only) when the AnchorGuard retrain is absent.
Every figure is exported as SVG, PDF and PNG into docs/figures/.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "work" / "artifacts" / "real"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": "#6b7280",
    "axes.labelcolor": "#1f2937",
    "text.color": "#1f2937",
    "xtick.color": "#4b5563",
    "ytick.color": "#4b5563",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.bbox": "tight",
})

NAVY = "#0b5394"
BLUE = "#3d85c6"
TEAL = "#1a8a7c"
RED = "#990000"
ORANGE = "#b45f06"
PURPLE = "#674ea7"
GRAY = "#6b7280"
LIGHT = "#eef3f8"
LIGHT2 = "#f6f8fa"
COND_COLORS = {
    "normal": NAVY,
    "parkinsons": TEAL,
    "stroke": ORANGE,
    "myopathic": PURPLE,
    "cerebralpalsy": RED,
}
COND_ORDER = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]


def save(fig, name):
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=150)
    plt.close(fig)
    print(f"wrote {name} (.svg/.pdf/.png)")


def box(ax, x, y, w, h, text, fc=LIGHT, ec=GRAY, fs=9, weight="normal", tc=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fs, color=tc or "#1f2937", weight=weight, zorder=3, wrap=True,
    )


def arrow(ax, x0, y0, x1, y1, color=GRAY, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
            linewidth=lw, color=color, linestyle=ls, zorder=1,
        )
    )


# ---------------------------------------------------------------------------
# 1. Overview: from markerless video to audited claim
# ---------------------------------------------------------------------------
def fig_overview():
    fig, ax = plt.subplots(figsize=(10.6, 5.0))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 50)
    ax.axis("off")

    ax.text(53, 47.5, "S-JEPA gait pipeline and its claim ladder",
            ha="center", fontsize=12, weight="bold")

    # Row 1: data
    box(ax, 2, 36, 20, 8, "Source videos\nGAVD (18) + added (17)", fc=LIGHT2)
    box(ax, 26, 36, 24, 8,
        "MediaPipe pose cache\n33 landmarks + visibility\nfps, frame numbers",
        fc=LIGHT2)
    box(ax, 54, 36, 24, 8,
        "Preprocessing contract\nvalidity 0.45, gaps<=4,\npelvis center, 64 frames",
        fc=LIGHT2)
    box(ax, 82, 36, 22, 8,
        "159 sequences / 35 videos\n96 canonical + 63 added normal",
        fc="#fff4e5", ec=ORANGE)
    arrow(ax, 22, 40, 26, 40)
    arrow(ax, 50, 40, 54, 40)
    arrow(ax, 78, 40, 82, 40)

    # Row 2: model
    box(ax, 2, 24, 30, 8,
        "S-JEPA\nview enc + EMA target enc\n+ predictor (528 tokens)",
        fc=LIGHT)
    box(ax, 36, 24, 30, 8,
        "Objective\nJEPA + 0.05 VICReg\n+ 0.25 group (stages 1-4)",
        fc=LIGHT)
    box(ax, 70, 24, 34, 8,
        "Normal-first curriculum\nStage 0 label-free (300 ep)\n+4 stages (75 ep), replay",
        fc=LIGHT)
    arrow(ax, 32, 28, 36, 28)
    arrow(ax, 66, 28, 70, 28)

    # Row 3: measured
    box(ax, 2, 13, 22, 7.5, "No total collapse\nstd 0.414", fc="#eef7f5", ec=TEAL)
    box(ax, 27, 13, 22, 7.5, "Normal-anchor drift\n0.954 -> 0.594", fc="#fdf0ee", ec=RED)
    box(ax, 52, 13, 24, 7.5, "Weak 5-group geometry\nsilhouette 0.009", fc="#fdf0ee", ec=RED)
    box(ax, 79, 13, 25, 7.5,
        "Transductive readouts only\n0.793 all-96 / 0.653 Lane C",
        fc="#fff4e5", ec=ORANGE)

    # Row 4: the ladder
    ladder = [
        ("L0", "Frozen probes, source-grouped\n(new notebooks 07-09)", LIGHT2),
        ("L1", "Fold-local retrain\n(AnchorGuard, 08)", LIGHT2),
        ("L2", "Nested video-disjoint\nfull pipeline retrain", LIGHT2),
        ("L3", "Multi-site, multi-camera\nclinical claims", "#f4f0fa"),
    ]
    x = 2
    for label, desc, fc in ladder:
        box(ax, x, 1.5, 24.5, 8, f"{label}\n{desc}", fc=fc, ec=PURPLE if fc == "#f4f0fa" else GRAY)
        x += 26
    arrow(ax, 53, 24, 53, 20.5, color=NAVY, lw=1.8)

    ax.text(53, 22.2, "what can we honestly claim?",
            ha="center", fontsize=8.5, color=NAVY, style="italic")
    save(fig, "bbfm_overview")


# ---------------------------------------------------------------------------
# 2. Normal-anchor drift (canonical + AnchorGuard overlay)
# ---------------------------------------------------------------------------
def fig_drift():
    summary = pd.read_csv(ART / "curriculum_stage_summary_augmented.csv")
    stages = [1, 2, 3, 4]
    canonical = summary["normal_anchor_cosine"].to_numpy()
    anchor_curve = None
    ag_path = ART / "anchor_guard_results.json"
    if ag_path.exists():
        ag = json.loads(ag_path.read_text())
        raw = ag["anchor_guard"].get("stage_end_anchor_cosines") or []
        # drop the leading None + the stage-0 NaN slot, keep the last 4 stages
        clean = [v for v in raw[1:] if v is not None and v == v]
        if len(clean) >= 4:
            anchor_curve = clean[-4:]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(stages, canonical, marker="o", ms=7, color=RED, lw=2,
            label="canonical curriculum (reproduced from checkpoints)")
    ax.annotate("0.954", (1, 0.954), textcoords="offset points",
                xytext=(10, 6), fontsize=8.5, color=RED)
    ax.annotate("0.839", (2, 0.839), textcoords="offset points",
                xytext=(10, 4), fontsize=8.5, color=RED)
    ax.annotate("0.707", (3, 0.707), textcoords="offset points",
                xytext=(10, -2), fontsize=8.5, color=RED)
    ax.annotate("0.594", (4, 0.594), textcoords="offset points",
                xytext=(-52, -14), fontsize=8.5, color=RED, weight="bold")
    if anchor_curve:
        ax.plot(stages, anchor_curve, marker="s", ms=7, color=NAVY, lw=2,
                label="AnchorGuard (anchor distillation, λ=0.5)")
    else:
        ax.plot(stages, [0.95, 0.92, 0.90, 0.88], marker="s", ms=7,
                color=NAVY, lw=2, linestyle="--", alpha=0.7,
                label="AnchorGuard target (pre-registered ≥ 0.85)")
    ax.axhline(0.85, color=TEAL, linestyle="--", lw=1.2)
    ax.text(3.06, 0.852, "retention gate", color=TEAL, fontsize=8.5)
    ax.set_xticks(stages)
    ax.set_xticklabels(["+ Parkinson's", "+ stroke", "+ myopathic", "+ cerebral palsy"],
                       rotation=12, ha="right")
    ax.set_ylim(0.5, 1.04)
    ax.set_ylabel("normal-anchor cosine")
    ax.set_xlabel("curriculum stage (one model continues)")
    ax.set_title("A quantified forgetting curve in a continual skeleton JEPA")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "bbfm_drift_curve")


# ---------------------------------------------------------------------------
# 3. Readout diagnostic results (notebook 07)
# ---------------------------------------------------------------------------
def fig_readout():
    res = pd.read_csv(ART / "temporal_readout_results.csv")
    targets = ["cadence", "stride_time", "peak_phase", "phase_lag", "energy_ratio"]
    res = res[res["target"].isin(targets)]
    piv = res.pivot(index="lane", columns="target", values="mae")
    lane_a = piv.loc["A_pooled_384"]
    lanes = ["B_moment_384", "C_bins_384", "D_attn_384"]
    lane_labels = ["B: + temporal moment", "C: 4 time bins", "D: attention pool"]
    colors = [NAVY, TEAL, ORANGE]

    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    x = np.arange(len(targets))
    width = 0.24
    for k, lane in enumerate(lanes):
        if lane not in piv.index:
            continue
        rel = (lane_a - piv.loc[lane]) / lane_a
        ax.bar(x + (k - 1) * width, rel, width, label=lane_labels[k],
               color=colors[k])
    ax.axhline(0.10, color=RED, linestyle="--", lw=1.3)
    ax.text(4.52, 0.105, "pre-registered +10% gate", color=RED, fontsize=8.5)
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["cadence\n(canary)", "stride time\n(canary)",
                        "peak phase", "phase lag\n(signed)", "energy\nratio"])
    ax.set_ylabel("relative MAE improvement vs deployed pooling")
    ax.set_title("Same frozen tokens, different readouts (source-grouped ridge)")
    ax.set_ylim(-0.45, 0.42)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.005, 0.02,
            "all three order-sensitive targets fail the gate -> NO EVIDENCE "
            "of recoverable timing; canary cadence R2≈0 in every lane "
            "(native rate was discarded by preprocessing)",
            transform=ax.transAxes, fontsize=8, color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=GRAY))
    save(fig, "bbfm_readout_sweep")


# ---------------------------------------------------------------------------
# 4. Predictive surprise (notebook 09)
# ---------------------------------------------------------------------------
def fig_surprise():
    path = ART / "predictive_surprise_results.json"
    if not path.exists():
        print("skip bbfm_surprise (artifact missing)")
        return
    r = json.loads(path.read_text())
    hs = r["horizon_summary"]
    horizons = ["2", "4", "8"]
    copy_last = hs["canonical_mean_copy_last_by_condition"]
    ceiling = hs["canonical_mean_spatial_ceiling"]
    vl = r["video_level"]
    auroc = {a["comparison"].replace("normal_vs_", ""): a
             for a in vl["auroc_normal_vs_condition"]}
    kw = vl["kruskal_wallis"]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    ax = axes[0]
    for cond in COND_ORDER:
        vals = [hs["canonical_mean_future_cosine_by_condition"][cond][h]
                for h in horizons]
        ax.plot([0, 1, 2], vals, marker="o", color=COND_COLORS[cond],
                label=cond, lw=1.8)
    ax.plot([0, 1, 2],
            [np.mean([copy_last[c] for c in COND_ORDER])] * 3,
            marker="s", linestyle="--", color="#555555",
            label="copy-last-patch baseline")
    ax.axhline(ceiling, color=NAVY, linestyle="-.", alpha=0.7,
               label="spatial infilling ceiling (0.547)")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["2 patches\n(0.5 s)", "4 patches\n(1.0 s)", "8 patches\n(2.0 s)"])
    ax.set_ylabel("mean latent cosine at future tokens")
    ax.set_title("Infilling does not equal forecasting")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    order = COND_ORDER
    names = ["normal\n(n=18)", "parkinsons\n(n=2)", "stroke\n(n=3)",
             "myopathic\n(n=10)", "cerebral palsy\n(n=2)"]
    data = []
    for cond in order:
        # video-level medians from the JSON
        pass
    # Reconstruct per-video medians is not stored; plot AUROC bars instead.
    conds = ["parkinsons", "stroke", "myopathic", "cerebralpalsy"]
    labels = ["vs Parkinson's", "vs stroke", "vs myopathic", "vs cerebral palsy"]
    vals = [auroc[c]["auroc_video_level"] for c in conds]
    lows = [auroc[c]["ci_low"] for c in conds]
    highs = [auroc[c]["ci_high"] for c in conds]
    err_lo = [v - l for v, l in zip(vals, lows)]
    err_hi = [h - v for v, h in zip(vals, highs)]
    x = np.arange(len(conds))
    ax.bar(x, vals, yerr=[err_lo, err_hi], capsize=4, color=[TEAL, ORANGE, PURPLE, RED],
           width=0.55)
    ax.axhline(0.5, color=GRAY, linestyle="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUROC, video-level surprise\n(normal vs condition)")
    ax.set_title(f"Surprise ranks CP/stroke above normal (KW p={kw['p']:.2f})")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.03,
            "surprise correlates with detector missingness (rho=0.50);\n"
            "residualized AUROC: CP 0.94, stroke 0.82, myo 0.80, PD 0.47",
            transform=ax.transAxes, fontsize=7.8, color="#374151",
            bbox=dict(boxstyle="round,pad=0.4", fc=LIGHT, ec=GRAY))
    fig.tight_layout()
    save(fig, "bbfm_surprise")


# ---------------------------------------------------------------------------
# 5. World-model concept: infilling vs forecasting
# ---------------------------------------------------------------------------
def fig_worldmodel():
    fig, ax = plt.subplots(figsize=(10.2, 5.8))
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 46)
    ax.axis("off")
    ax.text(51, 43.5, "One frozen predictor, three mask geometries",
            ha="center", fontsize=11.5, weight="bold")

    # sequence strip helper
    def strip(x0, y0, h=1.9, w=14, label="16 time patches", future=False):
        for i in range(16):
            c = "#d9e2ec" if not future else "#f6c8c0"
            ax.add_patch(plt.Rectangle((x0 + i * (w / 16), y0), w / 16 - 0.15, h,
                                       facecolor=c, edgecolor=GRAY, lw=0.5))
        ax.text(x0 + w / 2, y0 - 1.6, label, ha="center", fontsize=7.5, color="#4b5563")

    # (a) training: spatial infilling
    ax.text(2, 41, "(a) training: spatial infilling", fontsize=9.5, weight="bold", color=NAVY)
    strip(2, 34.5)
    for i in [3, 5, 9, 12]:
        ax.add_patch(plt.Rectangle((2 + i * (14 / 16), 34.5), 14 / 16 - 0.15, 1.9,
                                   facecolor="#f6c8c0", edgecolor=RED, lw=0.8))
    ax.text(2, 31.4, "hide random joints (12-landmark\nwhitelist), full time context",
            fontsize=8, color="#374151", va="top")

    # (b) world-model probe: future masking
    ax.text(2, 24.5, "(b) world-model probe: future masking", fontsize=9.5,
            weight="bold", color=TEAL)
    strip(2, 18)
    for i in range(14, 16):
        ax.add_patch(plt.Rectangle((2 + i * (14 / 16), 18), 14 / 16 - 0.15, 1.9,
                                   facecolor="#f6c8c0", edgecolor=RED, lw=0.8))
    ax.text(2, 14.9, "hide ALL joints of the last h patches:\npredict the future in latent space",
            fontsize=8, color="#374151", va="top")

    # (c) comparison panel
    ax.text(42, 40, "(c) what the frozen model achieves", fontsize=9.5, weight="bold")
    rows = [
        ("spatial infilling (native task)", "0.547", NAVY),
        ("future, h=2 (Parkinson's, best)", "0.442", TEAL),
        ("future, h=2 (cerebral palsy, worst)", "0.233", RED),
        ("copy-last-patch baseline", "0.88-0.95", "#555555"),
    ]
    y = 33.5
    for name, val, c in rows:
        box(ax, 40, y, 40, 4.2, name, fc=LIGHT2, ec=GRAY, fs=8)
        ax.text(82.5, y + 2.1, val, ha="center", va="center", fontsize=10,
                weight="bold", color=c)
        y -= 5.4

    # (d) rollout note
    ax.text(42, 7.6, "(d) 2-step latent rollout (one normal clip)", fontsize=9.5,
            weight="bold")
    box(ax, 40, 1.5, 42, 4.6,
        "direct h=2 cosine 0.571  ->  chained-rollout cosine 0.608\n"
        "(imagined patch feeds the next prediction; error did not explode in 2 steps)",
        fc=LIGHT2, ec=GRAY, fs=7.8)

    ax.text(51, -3.5, "verdict: the infilling predictor does NOT forecast better than a memoryless baseline - infilling != forecasting",
            ha="center", fontsize=9, color=RED, style="italic", weight="bold")
    save(fig, "bbfm_worldmodel_concept")


# ---------------------------------------------------------------------------
# 6. Consolidation concept: three intervention families
# ---------------------------------------------------------------------------
def fig_consolidation():
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    ax.set_xlim(0, 102)
    ax.set_ylim(0, 48)
    ax.axis("off")
    ax.text(51, 45.5, "Repairing normal-anchor drift: three consolidation families",
            ha="center", fontsize=11.5, weight="bold")

    box(ax, 2, 36, 30, 6,
        "Stage 0: normal-only S-JEPA\n(anchor c0 frozen, label-free)",
        fc=LIGHT, ec=NAVY)
    box(ax, 38, 36, 26, 6,
        "Stages 1-4: new pathology data\narrive one group at a time",
        fc="#fff4e5", ec=ORANGE)
    box(ax, 70, 36, 30, 6,
        "Drift: running normal representation\nmoves away from c0 (0.954 -> 0.594)",
        fc="#fdf0ee", ec=RED)
    arrow(ax, 32, 39, 38, 39)
    arrow(ax, 64, 39, 70, 39)

    families = [
        ("representation space", "Anchor distillation (AnchorGuard)\n"
         "L = lambda * (1 - mean cos(z_n, c0))",
         TEAL, "implemented in notebook 08"),
        ("parameter space", "Anchor-directional EWC\n"
         "penalize the top-k directions that move c0",
         BLUE, "documented extension"),
        ("data space", "Weighted replay\n"
         "upweight normal batches (balanced replay already helps)",
         PURPLE, "documented extension"),
    ]
    x = 2
    for title, desc, color, status in families:
        box(ax, x, 14, 31, 14, f"{title}\n{desc}", fc=LIGHT2, ec=color, fs=8.2)
        box(ax, x, 10.5, 31, 3.2, status, fc="white", ec=color, fs=7.5, tc=color)
        x += 33.5

    box(ax, 2, 1.5, 99, 7,
        "Pre-registered success (notebook 08): stage-4 anchor >= 0.85, feature std >= 0.35 (no collapse),\n"
        "source-grouped probe macro-F1 within 0.05 of the canonical baseline - retention must not buy plasticity loss.",
        fc="#eef7f5", ec=TEAL, fs=8.5)
    save(fig, "bbfm_consolidation")


# ---------------------------------------------------------------------------
# 7. 72-hour roadmap with decision gates
# ---------------------------------------------------------------------------
def fig_roadmap():
    fig, ax = plt.subplots(figsize=(10.6, 4.4))
    ax.set_xlim(0, 106)
    ax.set_ylim(0, 44)
    ax.axis("off")
    ax.text(53, 41.5, "72-hour plan to the Sept 5, 2026 AoE deadline",
            ha="center", fontsize=12, weight="bold")

    days = [
        ("Day 1\n(Sep 2)", NAVY, [
            ("verify venue + anonymize repo copy", "done in this doc"),
            ("reproduce drift + baselines (nb 08, frozen)", "DONE: 0.954->0.594 exact"),
            ("run readout diagnostic (nb 07)", "DONE: informative null"),
            ("run surprise probe (nb 09)", "DONE: CP/stroke > normal"),
            ("freeze results + pre-registration", "write before any retrain"),
        ]),
        ("Day 2\n(Sep 3)", TEAL, [
            ("margin-ablation retrains (nb 08)", "2 x ~4 min"),
            ("AnchorGuard full retrain (nb 08)", "~25 min, launch overnight/AM"),
            ("draft paper: results + methods", "5 pages, figures wired"),
            ("GATE: anchor >= 0.85?", "no -> report trade-off honestly"),
        ]),
        ("Day 3\n(Sep 4)", ORANGE, [
            ("downstream probes on new checkpoint", "source-grouped RF"),
            ("sensitivity: per-source, controls", "missingness/provenance"),
            ("finalize 5-page PDF + OpenReview", "anonymized, references"),
            ("submission by Sept 5 AoE", "with reproducibility appendix"),
        ]),
    ]
    x = 2
    for title, color, items in days:
        box(ax, x, 33, 33, 6, title, fc=color, ec=color, tc="white", weight="bold")
        y = 30.5
        for text, status in items:
            box(ax, x, y - 4.5, 33, 4.2, f"{text}\n({status})",
                fc=LIGHT2, ec=GRAY, fs=6.1)
            y -= 4.5
        x += 34.5

    box(ax, 2, 1.2, 101, 6.4,
        "Decision rule: keep the claim inside what the evidence supports. Primary story = quantified forgetting\n"
        "+ mechanism + repair (nb 08); garnish = predictive surprise (nb 09) and readout diagnostic (nb 07).\n"
        "If any gate fails, ship the informative negative - it is still a workshop-worthy result.",
        fc="#fff4e5", ec=ORANGE, fs=8)
    save(fig, "bbfm_roadmap_72h")


# ---------------------------------------------------------------------------
# 8. Direction portfolio map
# ---------------------------------------------------------------------------
def fig_directions():
    fig, ax = plt.subplots(figsize=(10.4, 5.4))
    ax.set_xlim(0, 104)
    ax.set_ylim(-4, 50)
    ax.axis("off")
    ax.text(52, 47, "Direction portfolio for the BrainBodyFM 2026 5-pager",
            ha="center", fontsize=12, weight="bold")

    ax.text(52, 43.5,
            "x = days of work to a decisive result      y = strength of the claim the result supports",
            ha="center", fontsize=8.5, color="#4b5563", style="italic")

    ax.annotate("", xy=(100, -2), xytext=(6, -2),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
    ax.annotate("", xy=(6, 40), xytext=(6, 0),
                arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=1.4))
    ax.text(53, -3.6, "effort (days) ->", fontsize=8.5, color="#4b5563", ha="center")
    ax.text(-1.5, 20, "claim strength ->", fontsize=8.5, color="#4b5563",
            rotation=90, va="center")

    def point(x, y, label, sub, color, size=9):
        ax.scatter([x], [y], s=size * 40, color=color, zorder=3, alpha=0.9)
        ax.text(x + 2.5, y, label, fontsize=9, weight="bold", color=color, va="center")
        ax.text(x + 2.5, y - 3.4, sub, fontsize=7.2, color="#374151", va="center")

    point(8, 36, "A. Anchor drift + AnchorGuard repair  [PRIMARY]",
          "measured forgetting + mechanism + intervention;\nnb 08; strongest novelty gap (no competitor reports a drift curve)",
          RED)
    point(20, 27, "D. Objective engineering  [long-term]",
          "motion-field targets + causal masks + LOSO retrain;\nneeds multi-day retraining, not this deadline",
          ORANGE)
    point(6, 18, "C. Temporal readout diagnostic  [backup]",
          "nb 07; honest informative null already in hand;\nwins as an evaluation-methodology lesson",
          NAVY)
    point(12, 9.5, "B. World-model surprise probe  [garnish]",
          "frozen future-masking; nb 09; motor-loop framing;\nsurprise confounded by missingness (rho=0.50) - control it",
          TEAL)
    point(4, 4, "E. Clinical scalar retention  [supporting]",
          "five gait scalars + laterality informative null;\ninterpretability garnish, no diagnostic claims",
          PURPLE)

    box(ax, 2, 40.5, 0.1, 0.1, "", fc="white", ec="white")
    save(fig, "bbfm_directions_map")


if __name__ == "__main__":
    fig_overview()
    fig_drift()
    fig_readout()
    fig_surprise()
    fig_worldmodel()
    fig_consolidation()
    fig_roadmap()
    fig_directions()
    print("all brain-body figures written to", OUT)
