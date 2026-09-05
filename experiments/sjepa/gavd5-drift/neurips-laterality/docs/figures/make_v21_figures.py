"""Generate laterality v2.1 paper figures from the completed held-out report.

All quantities are read directly from the registered report artifacts under
protocol_6f7baefbda07 so the figures cannot drift from the verified numbers.
Run with the project venv:  ../../../.venv/bin/python make_v21_figures.py
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.abspath(os.path.join(
    HERE, "..", "..", "artifacts", "paper", "protocol_6f7baefbda07", "report"))
COHORT = os.path.abspath(os.path.join(
    HERE, "..", "..", "artifacts", "paper", "protocol_6f7baefbda07", "cohort", "cohort.npz"))

# Okabe-Ito colourblind-safe palette.
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERM = "#D55E00"
GREY = "#5A5A5A"
LGREY = "#BBBBBB"

plt.rcParams.update({
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(os.path.join(HERE, f"{stem}.{ext}"))
    plt.close(fig)


def ckpt():
    return pd.read_csv(os.path.join(REPORT, "checkpoint_source_bootstrap.csv"))


def load_row(df, **kw):
    m = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        m &= df[k].astype(str) == str(v)
    sub = df[m]
    assert len(sub) == 1, f"expected 1 row for {kw}, got {len(sub)}"
    return sub.iloc[0]


NUMBERS = {}

# ---------------------------------------------------------------------------
# Figure 1: the group action on a real GAVD-derived pose.
# ---------------------------------------------------------------------------
BLAZE_EDGES = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]
SWAP = {1: 4, 2: 5, 3: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16, 17: 18,
        19: 20, 21: 22, 23: 24, 25: 26, 27: 28, 29: 30, 31: 32}
PERM = np.arange(33)
for a, b in SWAP.items():
    PERM[a], PERM[b] = b, a
TARGET_JOINTS = [11, 12, 25, 26, 27, 28, 29, 30, 31, 32]


def draw_pose(ax, pts, valid, title, sign):
    for a, b in BLAZE_EDGES:
        if valid[a] and valid[b]:
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color=GREY, lw=1.6, zorder=1)
    left = [j for j in TARGET_JOINTS if j % 2 == 1]
    right = [j for j in TARGET_JOINTS if j % 2 == 0]
    ax.scatter(pts[left, 0], pts[left, 1], c=BLUE, s=34, zorder=3, label="left landmarks")
    ax.scatter(pts[right, 0], pts[right, 1], c=ORANGE, s=34, zorder=3, label="right landmarks")
    ax.text(0.5, -0.02, title, transform=ax.transAxes, ha="center", va="top",
            fontsize=11, weight="bold")
    ax.text(0.5, -0.1, sign, transform=ax.transAxes, ha="center", va="top",
            fontsize=10.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def figure_group_action():
    z = np.load(COHORT, allow_pickle=True)
    idx = 327
    xyz = z["model_xyz"][idx]        # (64, 33, 3)
    val = z["model_valid"][idx]      # (64, 33)
    y = float(z["pair_contrasts"][idx].mean())
    NUMBERS["concept_pose_y"] = round(y, 3)
    # mean pose over valid frames, per joint
    pts = np.full((33, 2), np.nan)
    joint_valid = np.zeros(33, dtype=bool)
    for j in range(33):
        fv = val[:, j]
        if fv.any():
            pts[j] = xyz[fv, j, :2].mean(axis=0)
            joint_valid[j] = True
    # image y grows downward; flip for display
    disp = pts.copy()
    disp[:, 1] = -disp[:, 1]
    # mirror: negate x, permute left/right
    mir = disp.copy()
    mir[:, 0] = -mir[:, 0]
    mir = mir[PERM]
    mir_valid = joint_valid[PERM]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.1))
    fig.subplots_adjust(top=0.80, bottom=0.14)
    draw_pose(axes[0], disp, joint_valid, "Pose $x$",
              fr"$y = {y:+.2f}$  (more left-side motion)")
    draw_pose(axes[1], mir, mir_valid, "Mirrored pose $Mx$",
              fr"$y \to {-y:+.2f}$  (sign reverses)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.86), handletextpad=0.3,
               columnspacing=1.8)
    fig.suptitle("Bilateral reflection is a group action: flip the horizontal axis,\n"
                 "swap left/right landmarks", fontsize=10.5, weight="bold", y=0.99)
    save(fig, "v21_group_action")


# ---------------------------------------------------------------------------
# Figure 2: what training changed (learned minus untrained baseline).
# ---------------------------------------------------------------------------
def figure_learning_effect():
    df = ckpt()
    # predictive utility deltas: learned - random, per construction (vanilla)
    rows = [
        ("free read-out", "primary_training_content"),
        ("odd wrapper (free origin)", "odd_feature_training_content"),
        ("odd wrapper (zero origin)", "constructed_training_content"),
        ("even wrapper", "even_feature_training_content"),
    ]
    labels, est, lo, hi = [], [], [], []
    for lab, comp in rows:
        r = load_row(df, comparison_type=comp)
        labels.append(lab)
        est.append(float(r.estimate)); lo.append(float(r.ci95_low)); hi.append(float(r.ci95_high))
        NUMBERS[comp] = [round(float(r.estimate), 3), round(float(r.ci95_low), 3), round(float(r.ci95_high), 3)]
    # reflection augmentation effect on the primary lane
    r = load_row(df, comparison_type="reflection_minus_vanilla_primary")
    NUMBERS["reflection_minus_vanilla_primary"] = [round(float(r.estimate), 3), round(float(r.ci95_low), 3), round(float(r.ci95_high), 3)]

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(wspace=0.55)

    ax = axes[0]
    ypos = np.arange(len(labels))[::-1]
    for yv, e, l, h in zip(ypos, est, lo, hi):
        col = VERM if h < 0 else (GREEN if l > 0 else GREY)
        ax.plot([l, h], [yv, yv], color=col, lw=2.4, zorder=2)
        ax.plot(e, yv, "o", color=col, ms=6, zorder=3)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"learned $-$ untrained encoder, source-balanced $R^2$")
    ax.set_title("(a) Predictive utility")
    ax.text(0.03, 0.97, "left of 0: training hurts", transform=ax.transAxes,
            fontsize=8, color=GREY, ha="left", va="top")

    ax = axes[1]
    # equivariance training effect comes from the equivariance bootstrap file
    eq = pd.read_csv(os.path.join(REPORT, "strict_representation_equivariance_source_bootstrap.csv"))
    def eqrow(comp, va):
        m = (eq.comparison_type == comp) & (eq.variant_a == va)
        s = eq[m]; assert len(s) == 1; return s.iloc[0]
    van = eqrow("learned_minus_initial_strict_equivariance", "vanilla")
    ref = eqrow("learned_minus_initial_strict_equivariance", "reflection_augmented")
    NUMBERS["equiv_learned_minus_initial_vanilla"] = [round(float(van.estimate), 3), round(float(van.ci95_low), 3), round(float(van.ci95_high), 3)]
    NUMBERS["equiv_learned_minus_initial_reflection"] = [round(float(ref.estimate), 3), round(float(ref.ci95_low), 3), round(float(ref.ci95_high), 3)]
    labs2 = ["vanilla", "reflection\naugmented"]
    yy = np.array([0.7, 0.0])
    for yv, r in zip(yy, [van, ref]):
        e, l, h = float(r.estimate), float(r.ci95_low), float(r.ci95_high)
        col = VERM if l > 0 else GREY
        ax.plot([l, h], [yv, yv], color=col, lw=2.4)
        ax.plot(e, yv, "o", color=col, ms=6)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_ylim(-0.4, 1.1)
    ax.set_xlim(-0.006, 0.052)
    ax.set_yticks(yy); ax.set_yticklabels(labs2)
    ax.set_xlabel("learned $-$ untrained,\nstrict token error $q$")
    ax.set_title("(b) Reflection equivariance")
    ax.text(0.97, 0.97, "right of 0: training hurts", transform=ax.transAxes,
            fontsize=8, color=GREY, ha="right", va="top")
    fig.suptitle("Self-supervised training adds no laterality signal and degrades the "
                 "initialization's approximate equivariance",
                 fontsize=10, weight="bold", y=1.03)
    save(fig, "v21_learning_effect")


# ---------------------------------------------------------------------------
# Figure 3: absolute levels against the frozen decision thresholds.
# ---------------------------------------------------------------------------
def figure_absolute_levels():
    df = ckpt()
    metrics = pd.read_csv(os.path.join(REPORT, "metrics.csv"))
    def ens(variant, lane, col="source_balanced_r2"):
        m = (metrics.analysis_set == "all_qc_eligible") & (metrics.variant == variant) & (metrics.lane == lane)
        s = metrics[m]; assert len(s) == 1, (variant, lane, len(s)); return float(s.iloc[0][col])

    prim = load_row(df, comparison_type="absolute_primary")
    cons = load_row(df, comparison_type="absolute_constructed")
    NUMBERS["absolute_primary"] = [round(float(prim.estimate), 3), round(float(prim.ci95_low), 3), round(float(prim.ci95_high), 3)]
    NUMBERS["absolute_constructed"] = [round(float(cons.estimate), 3), round(float(cons.ci95_low), 3), round(float(cons.ci95_high), 3)]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), gridspec_kw={"width_ratios": [1, 1]})

    # Panel (a): source-balanced R^2 with CI where available, threshold 0.
    ax = axes[0]
    entries = [
        ("oracle", ens("vanilla", "oracle_target_components"), None, None, GREY),
        ("learned\nfree", float(prim.estimate), float(prim.ci95_low), float(prim.ci95_high), BLUE),
        ("constructed\nodd", float(cons.estimate), float(cons.ci95_low), float(cons.ci95_high), ORANGE),
        ("nuisances", ens("vanilla", "nuisance_all_free"), None, None, LGREY),
    ]
    xs = np.arange(len(entries))
    for x, (lab, e, l, h, c) in zip(xs, entries):
        ax.bar(x, e, width=0.6, color=c, zorder=2)
        if l is not None:
            ax.plot([x, x], [l, h], color="k", lw=1.4, zorder=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([e[0] for e in entries], fontsize=8.5)
    ax.set_ylabel(r"source-balanced $R^2$")
    ax.set_title("(a) Predictive utility")
    ax.annotate(r"$\approx 1.00$", (0, 1.0), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=8)
    ax.text(0.98, 0.5, "gate: lower CI > 0\n(not met)", transform=ax.transAxes,
            fontsize=8, color=VERM, ha="right", va="center")

    # Panel (b): geometric fidelity errors with margins.
    ax = axes[1]
    eq = pd.read_csv(os.path.join(REPORT, "strict_representation_equivariance_source_bootstrap.csv"))
    def eqabs(comp, va):
        m = (eq.comparison_type == comp) & (eq.variant_a == va)
        s = eq[m]; assert len(s) == 1; return s.iloc[0]
    ql = eqabs("absolute_learned_strict_equivariance", "vanilla")
    qi = eqabs("absolute_initial_strict_equivariance", "vanilla")
    nat = pd.read_csv(os.path.join(REPORT, "native_output_symmetry_source_bootstrap.csv"))
    def natabs(va):
        m = (nat.comparison_type == "absolute_native_learned_symmetry") & (nat.variant_a == va)
        s = nat[m]; assert len(s) == 1; return s.iloc[0]
    nl = natabs("vanilla")
    NUMBERS["strict_q_learned_vanilla"] = [round(float(ql.estimate), 3), round(float(ql.ci95_low), 3), round(float(ql.ci95_high), 3)]
    NUMBERS["strict_q_initial_vanilla"] = [round(float(qi.estimate), 3), round(float(qi.ci95_low), 3), round(float(qi.ci95_high), 3)]
    NUMBERS["native_antisym_learned_vanilla"] = [round(float(nl.estimate), 3), round(float(nl.ci95_low), 3), round(float(nl.ci95_high), 3)]

    bars = [
        ("$q$ untrained", float(qi.estimate), float(qi.ci95_low), float(qi.ci95_high), LGREY),
        ("$q$ learned", float(ql.estimate), float(ql.ci95_low), float(ql.ci95_high), BLUE),
        ("native\noutput", float(nl.estimate), float(nl.ci95_low), float(nl.ci95_high), VERM),
        ("constructed\noutput", 0.0, None, None, ORANGE),
    ]
    xs = np.arange(len(bars))
    for x, (lab, e, l, h, c) in zip(xs, bars):
        ax.bar(x, e, width=0.6, color=c, zorder=2)
        if l is not None:
            ax.plot([x, x], [l, h], color="k", lw=1.4, zorder=3)
    ax.axhline(0.10, color=GREEN, lw=1.2, ls="--", zorder=1)
    ax.text(3.45, 0.108, "margin 0.10", color=GREEN, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("normalized error (0 = exact)")
    ax.set_title("(b) Geometric fidelity")
    ax.annotate("= 0 by\nconstruction", (3, 0.0), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=7.5)
    fig.suptitle("Learned representation misses both gates; only the built-in construction "
                 "reaches exact symmetry", fontsize=9.5, weight="bold", y=1.02)
    save(fig, "v21_absolute_levels")


if __name__ == "__main__":
    figure_group_action()
    figure_learning_effect()
    figure_absolute_levels()
    with open(os.path.join(HERE, "v21_figure_numbers.json"), "w") as f:
        json.dump(NUMBERS, f, indent=2, sort_keys=True)
    print(json.dumps(NUMBERS, indent=2, sort_keys=True))
    print("wrote v21_group_action, v21_learning_effect, v21_absolute_levels (pdf+svg)")
