"""Create a descriptive naming-condition figure from audited person/seed rows.

This script performs no new fitting or outcome selection. The three methods and
all naming conditions are explicit below. No uncertainty is synthesized.
"""
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/iclr/figures"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "docs/iclr/qa/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                     "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "pdf.fonttype": 42, "svg.fonttype": "none",
                     "axes.spines.top": False, "axes.spines.right": False})
source = ROOT / "docs/iclr/evidence/laterality-condition-person.csv"
upstream = ROOT / "outputs/iclr/jepa-response/evaluation/response-by-condition-person.csv"
all_rows = pd.read_csv(source)
methods = [
    ("P-direct-none-base", "Direct / base", "#0072B2", "o", "-"),
    ("F-response-jepa_delta_v1-graph_time-paired_change", "Delta JEPA / change", "#B96812", "^", "--"),
    ("F-response-jepa_endpoint_v1-graph_time-paired_change", "Endpoint JEPA / change", "#65727B", "s", ":"),
]
conditions = ["correct", "global_swap", "temporary_swap"]
selected = all_rows[all_rows.factor.eq("naming") & all_rows.method.isin([m[0] for m in methods])].copy()
assert not selected.duplicated(["method", "seed", "canonical_person_id", "condition"]).any()
assert selected.groupby(["method", "condition"]).size().eq(42).all()
assert set(selected.seed) == {17, 29, 43}
assert selected.canonical_person_id.nunique() == 14
person = selected.groupby(["method", "condition", "canonical_person_id"]).assignment_failure_rate.mean()
means = person.groupby(["method", "condition"]).mean()

fig, ax = plt.subplots(figsize=(5.5, 1.9))
fig.subplots_adjust(left=.115, right=.99, bottom=.22, top=.78)
x = np.arange(3)
plot_rows = []
for method, label, color, marker, style in methods:
    y = 100 * means.loc[method].reindex(conditions).to_numpy()
    ax.plot(x, y, label=label, color=color, marker=marker, linestyle=style,
            linewidth=1.15, markersize=4.5, markeredgewidth=.7)
    plot_rows.extend({"method": method, "label": label, "naming": condition,
                      "assignment_failure_percent": float(value), "people": 14, "seeds": 3}
                     for condition, value in zip(conditions, y))
ax.set(xlim=(-.28, 2.28), ylim=(0, 100), ylabel="Geometric assignment\nfailure (%)",
       xticks=x, xticklabels=["Correct names", "Global swap", "Temporary swap"],
       yticks=[0, 25, 50, 75, 100])
ax.grid(axis="y", alpha=.18, linewidth=.5)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.38), ncol=3, frameon=False,
          columnspacing=1.1, handlelength=1.55, handletextpad=.45)
for extension in ["pdf", "svg", "png"]:
    fig.savefig(OUT / f"laterality-naming.{extension}", dpi=220)
plt.close(fig)
pd.DataFrame(plot_rows).to_csv(OUT / "laterality-naming-means.csv", index=False)
provenance = {
    "question": "Does the scalar response point advantage imply better geometric naming under input label corruption?",
    "input_files": [{"path": str(p.relative_to(ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    for p in [source, upstream]],
    "aggregation": "Audited endpoint strata use nonheld:held weight 4:1, then mean across three seeds within each person and equal mean across 14 people.",
    "conditions": conditions, "means": plot_rows,
    "inference": "Post hoc descriptive development means, selected after completed experiments; no error bars or confidence intervals. No calibrated chance line.",
    "metric": "Reference-valid hip/knee/ankle pairs separated by at least 4 pixels; swapped sum-of-distance better by >2 pixels is wrong, absolute named-swapped difference <=2 pixels is ambiguous, nonfinite predictions are missing; all fail.",
    "scope": "Projected geometric naming diagnostic, not anatomical affected-side accuracy. Direct trains encoder and readout jointly; JEPA readouts use frozen encoders.",
    "figure_size_inches": [5.5, 1.9], "font_points": 8,
    "source_script": "docs/iclr/scripts/build_laterality_figure.py",
}
(OUT / "laterality-naming-provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
print(pd.DataFrame(plot_rows)[["label", "naming", "assignment_failure_percent"]].to_string(index=False))
