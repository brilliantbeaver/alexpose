"""Draw the saved descriptive completion comparisons without recomputing models."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "docs/draft03/advisory/completion_audit_v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    table = pd.read_csv(OUT / "completion_summary.csv")
    shown = table.loc[table.outer_fold.eq("all")].copy()
    if list(zip(shown.dataset, shown.bank)) != [("external_angles", "U"), ("GAVD", "U"), ("GAVD", "E_leg")]:
        raise ValueError("Unexpected completed comparisons")
    labels = ["External angles: both-leg vs own-leg\n128 healthy participants",
              "GAVD: A training vs U training, random masks\n28 normal recording sources",
              "GAVD: A training vs U training, leg occlusion\n28 normal recording sources"]
    colors = ["#196F88", "#707780", "#307D5A"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(10.8, 5.0))
    fig.subplots_adjust(left=.44, right=.96, top=.79, bottom=.28)
    y = np.arange(3)[::-1]
    points = shown.relative_mse_reduction.to_numpy() * 100
    low = shown.relative_ci_low.to_numpy() * 100
    high = shown.relative_ci_high.to_numpy() * 100
    for i, position in enumerate(y):
        ax.errorbar(points[i], position, xerr=[[points[i]-low[i]], [high[i]-points[i]]],
                    fmt="o", markersize=7, capsize=4, color=colors[i], linewidth=2)
        ax.text(points[i], position+.19, f"{points[i]:.2f}%  [{low[i]:.2f}, {high[i]:.2f}]",
                color=colors[i], ha="center", va="bottom", fontsize=9.2)
    ax.axvline(0, color="#8D9298", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0, pad=14)
    ax.set_ylim(-.42, 2.55)
    ax.set_xlim(-2, 11)
    ax.set_xticks(np.arange(-2, 11, 2))
    ax.set_xlabel("Relative reduction in mean squared error (%)", labelpad=10)
    ax.grid(axis="x", color="#E4E8EB", linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#BCC3C9")
    fig.text(.04, .94, "Paired changes in normal completion error", fontsize=17, weight="bold", va="top")
    fig.text(.04, .865, "Positive values indicate lower error. Each comparison uses its own frozen task and units.", fontsize=10, color="#414B54")
    fig.text(.04, .06, "Post-outcome descriptive audit. Conditional 95% paired bootstrap intervals, 1,000 draws.\nAll four folds are retained; no model, cohort, mask or anomaly score was changed.", fontsize=9, color="#414B54")
    for suffix in ("svg", "png"):
        fig.savefig(OUT / f"completion_gains.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), producer_sha256=sha(__file__),
                    input_sha256=sha(OUT / "completion_summary.csv"),
                    interpretation="Three distinct relative-error comparisons, not a pooled analysis or anomaly-performance claim.",
                    outputs={name: sha(OUT/name) for name in ("completion_gains.svg", "completion_gains.png")})
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
