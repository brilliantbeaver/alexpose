"""Render publication figures from the independently verified current scores."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "review-2026-10-15/evidence/results-verification.json"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "savefig.facecolor": "white",
})
if SOURCE.exists():
    verified = json.loads(SOURCE.read_text())["runs"]["capstone-20261009T060955236580Z"]
    payload = {name: {
        "fold_macro_f1": verified["metrics"][name]["fold_source_macro_f1"],
        "source_weighted_confusion": verified["metrics"][name]["source_weighted"]["confusion"],
    } for name in ("sjepa", "rf", "mean_pose")}
    (HERE / "figure-values.json").write_text(json.dumps(payload, indent=2) + "\n")
else:
    payload = json.loads((HERE / "figure-values.json").read_text())

def save(fig, stem):
    fig.savefig(HERE / (stem + ".pdf"), bbox_inches="tight", pad_inches=.03)
    fig.savefig(HERE / (stem + ".png"), dpi=250, bbox_inches="tight", pad_inches=.03)
    plt.close(fig)

# The plot contains one mark per fold and method, not a time series.
fig = plt.figure(figsize=(7.0, 2.42))
gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 1, 1, 1],
                     left=.065, right=.99, bottom=.23, top=.78, wspace=.43)
ax = fig.add_subplot(gs[0])
methods = [("sjepa", "S-JEPA", "#176B89", "o", -.16),
           ("rf", "Corrected RF", "#B34F20", "s", 0),
           ("mean_pose", "Mean pose", "#49545D", "^", .16)]
for key, label, color, marker, offset in methods:
    ax.scatter(np.arange(5) + offset, payload[key]["fold_macro_f1"],
               color=color, marker=marker, s=22, label=label, zorder=3)
ax.set(xticks=range(5), xlabel="Held-out fold", ylabel="Source-weighted macro-F1",
       ylim=(0, 1), xlim=(-.55, 4.55), yticks=[0, .25, .5, .75, 1])
ax.grid(axis="y", color="#DFE4E7", lw=.55)
ax.set_title("(a) Variation across folds", loc="left", fontsize=8, pad=8)
fig.legend(*ax.get_legend_handles_labels(), loc="upper left",
           bbox_to_anchor=(.04, 1.015), ncol=3, frameon=False,
           columnspacing=1.3, handletextpad=.4, fontsize=7.7)
for j, (key, title) in enumerate([("sjepa", "(b) S-JEPA"),
                                 ("rf", "(c) Corrected RF"),
                                 ("mean_pose", "(d) Mean pose")], 1):
    a = fig.add_subplot(gs[j])
    cm = np.array(payload[key]["source_weighted_confusion"])
    pct = 100 * cm / cm.sum(axis=1, keepdims=True)
    a.imshow(pct, vmin=0, vmax=100, cmap="Blues", aspect="equal")
    for row in range(3):
        for col in range(3):
            # Remove representation noise at exact decimal half-way values.
            a.text(col, row, f"{round(float(pct[row, col]), 10):.1f}", ha="center", va="center",
                   fontsize=7, color="white" if pct[row, col] > 55 else "#15212A")
    a.set(xticks=range(3), xticklabels=["N", "MS", "PD"],
          yticks=range(3), yticklabels=["N", "MS", "PD"] if j == 1 else [],
          xlabel="Predicted label")
    if j == 1:
        a.set_ylabel("Collection label", labelpad=3)
    a.set_title(title, fontsize=8, pad=8)
    a.tick_params(length=0, labelsize=7.5)
    for spine in a.spines.values():
        spine.set_visible(False)
fig.text(.67, .035, "N: Normal     Cell values: row percentage (%)",
         ha="center", fontsize=7.3)
save(fig, "results")

# Six numbered stages preserve the source-partition-before-windowing order.
fig, ax = plt.subplots(figsize=(7, 2.05))
ax.set(xlim=(0, 10), ylim=(0, 3))
ax.set_axis_off()
boxes = [
    (.05, 1.82, "1  Gait clips",
     "91 clips / 41 source videos\nNormal, MS and PD labels"),
    (3.43, 1.82, "2  Pose extraction",
     "33 landmarks: x, y, visibility\n88 usable clips after cleaning"),
    (6.81, 1.82, "3  Partition by source",
     "Five held-out test folds\nSeparate training and validation"),
    (.05, .12, "4  Windows and tokens",
     "32 frames; stride 16\n264 landmark–time tokens"),
    (3.43, .12, "5  Feature learning",
     "800 + restarted 400 updates\nSelect stage using validation"),
    (6.81, .12, "6  Held-out classification",
     "Frozen encoder + linear classifier\nCompare RF and simple summaries"),
]
for x, y, title, body in boxes:
    ax.add_patch(FancyBboxPatch((x, y), 3.12, 1.02,
                               boxstyle="round,pad=0.015,rounding_size=.035",
                               fc="#F1F6F8", ec="#567783", lw=.65))
    ax.text(x+.12, y+.77, title, fontsize=8.8, weight="bold", va="center")
    ax.text(x+.12, y+.36, body, fontsize=7.9, va="center", linespacing=1.5)
for x in [3.20, 6.58]:
    for y in [2.33, .63]:
        ax.annotate("", (x+.20, y), (x-.03, y),
                    arrowprops={"arrowstyle": "->", "lw": .7, "color": "#567783"})
# The elbow connector explicitly links stage 3 to stage 4.
ax.plot([8.37, 8.37, 1.61], [1.80, 1.48, 1.48], color="#567783", lw=.7)
ax.annotate("", (1.61, 1.16), (1.61, 1.48),
            arrowprops={"arrowstyle": "->", "lw": .7, "color": "#567783"})
ax.text(5, 1.30, "Encoder learns without condition labels; the classifier uses them.",
        ha="center", va="center", fontsize=7.5, color="#294958")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
save(fig, "pipeline")

# Schematic coordinates, not participant measurements.
xy = np.array([
    [0, 2.7], [-.08, 2.77], [-.15, 2.78], [-.22, 2.76],
    [.08, 2.77], [.15, 2.78], [.22, 2.76],
    [-.29, 2.66], [.29, 2.66], [-.10, 2.55], [.10, 2.55],
    [-.43, 2.22], [.43, 2.22], [-.66, 1.75], [.66, 1.75],
    [-.85, 1.32], [.85, 1.32], [-.99, 1.20], [.99, 1.20],
    [-.94, 1.12], [.94, 1.12], [-.77, 1.21], [.77, 1.21],
    [-.31, 1.25], [.31, 1.25], [-.34, .68], [.34, .68],
    [-.37, .12], [.37, .12], [-.29, .01], [.29, .01],
    [-.55, -.06], [.55, -.06],
])
edges = [(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),
         (11,12),(11,23),(12,24),(23,24),
         (11,13),(13,15),(15,17),(15,19),(15,21),(17,19),
         (12,14),(14,16),(16,18),(16,20),(16,22),(18,20),
         (23,25),(25,27),(27,29),(27,31),(29,31),
         (24,26),(26,28),(28,30),(28,32),(30,32)]
selected = [11,12,23,24,25,26,27,28,29,30,31,32]
other = [i for i in range(33) if i not in selected]
fig, ax = plt.subplots(figsize=(3.3, 2.48))
for a, b in edges:
    ax.plot(xy[[a,b], 0], xy[[a,b], 1], color="#A2ADB3", lw=1.1, zorder=1)
ax.scatter(xy[other, 0], xy[other, 1], color="#929DA5", s=13, zorder=2)
ax.scatter(xy[selected, 0], xy[selected, 1], color="#176B89", s=26, zorder=3)
ax.set_aspect("equal")
ax.set(xlim=(-1.28, 2.42), ylim=(-.20, 2.95))
ax.set_axis_off()
ax.text(1.12, 2.02, "Selected landmarks", fontsize=8.1, color="#176B89", weight="bold")
ax.text(1.12, 1.73, "Shoulders\nHips\nKnees\nAnkles\nHeels\nFoot indices",
        va="top", fontsize=8, linespacing=1.65)
ax.legend(handles=[
    Line2D([0],[0], marker="o", linestyle="", color="#176B89", markersize=5,
           label="12 selected"),
    Line2D([0],[0], marker="o", linestyle="", color="#929DA5", markersize=4,
           label="21 remaining"),
], loc="lower left", bbox_to_anchor=(-.02, -.12), ncol=2, frameon=False,
   fontsize=7.5, handletextpad=.3, columnspacing=.8)
fig.subplots_adjust(left=.01, right=.99, bottom=.12, top=.99)
save(fig, "landmarks")
print("Rendered figures and saved exact numerical plot values.")
