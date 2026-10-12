"""Reproduce the author's rounded means; no inferred confidence intervals.

Run from the repository root with .venv/bin/python. This is a descriptive
manuscript preview, not a reconstruction from the remote prediction receipts.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ["response_error", "nuisance_error", "synthetic_all_nle", "waveform_error", "direction_accuracy"]
# For each family, base then paired_change. Values transcribed from the user.
DATA = {
    "Direct": [[7.5442, 10.4929, .0298, 12.0733, .6609], [10.7543, 16.6262, .0518, 19.2808, .5638]],
    "Coordinate": [[10.2473, 11.1250, .0487, 15.8800, .5019], [11.1743, 12.4444, .0561, 21.4048, .4904]],
    "Paired JEPA": [[10.6914, 14.1674, .0626, 17.4532, .4932], [10.0662, 12.7601, .0700, 22.5533, .4894]],
    "Initialized": [[9.4331, 13.9304, .0687, 16.9685, .4730], [10.1591, 14.3239, .0718, 21.9566, .4705]],
    "Shuffled JEPA": [[9.3450, 13.4056, .0676, 16.7352, .4797], [11.7461, 14.8242, .0718, 21.3673, .4842]],
    "Coordinate delta": [[10.1773, 10.5416, .0550, 16.2209, .5028], [14.5198, 15.0029, .0617, 21.0849, .4865]],
    "JEPA endpoint": [[10.2327, 13.5610, .0632, 16.8988, .4753], [10.3611, 14.1406, .0704, 22.0311, .4784]],
    "JEPA delta": [[10.9448, 14.3844, .0628, 17.3770, .4931], [9.9880, 12.2297, .0699, 21.7648, .4844]],
}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.4),
                             gridspec_kw={"width_ratios": [1.15, 1.15, .9]})
    y = np.arange(len(DATA))
    values = np.array(list(DATA.values()))
    colors = ["#187c87", "#c16c27"]
    for panel, metric_index, title, label in (
        (axes[0], 0, "A  Response fidelity", "Response error (degrees; lower is better)"),
        (axes[1], 3, "B  Trajectory fidelity", "Waveform error (degrees; lower is better)"),
    ):
        for i in y:
            panel.plot(values[i, :, metric_index], [i, i], color="#9eabb4", lw=1.6, zorder=1)
        panel.scatter(values[:, 0, metric_index], y, c=colors[0], marker="o", s=37, label="Base", zorder=3)
        panel.scatter(values[:, 1, metric_index], y, c=colors[1], marker="^", s=43, label="Paired change", zorder=3)
        panel.set_yticks(y)
        panel.set_yticklabels(list(DATA) if panel is axes[0] else [])
        panel.set_ylim(len(DATA) - .55, -.65)
        panel.set_xlabel(label, fontsize=8)
        panel.set_title(title, loc="left", fontsize=10, weight="bold", pad=13)
        panel.grid(axis="x", alpha=.18)
        panel.axhline(4.5, color="#c9cfd4", linestyle=":", lw=1)
    axes[0].set_xlim(7, 15.3)
    axes[1].set_xlim(11, 24)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.127, .925), ncol=2,
               frameon=False, fontsize=8, handletextpad=.4, columnspacing=1.3)

    endpoint = np.array(DATA["JEPA endpoint"])
    delta = np.array(DATA["JEPA delta"])
    advantage = endpoint[:, 0] - delta[:, 0]
    axes[2].axhline(0, color="#7a8288", linestyle="--", lw=1)
    axes[2].plot([0, 1], advantage, color="#354e6a", marker="o", markersize=6, lw=1.8)
    for x, val in enumerate(advantage):
        axes[2].annotate(f"{val:+.4f}°", (x, val), xytext=(0, 12 if val > 0 else -18),
                         textcoords="offset points", ha="center", fontsize=9, weight="bold")
    axes[2].set_xticks([0, 1], ["Base", "Paired\nchange"])
    axes[2].set_xlim(-.4, 1.4)
    axes[2].set_ylim(-1.1, .8)
    axes[2].set_ylabel("Endpoint − delta response error (degrees)", fontsize=8)
    axes[2].set_title("C  Readout dependence", loc="left", fontsize=10, weight="bold", pad=13)
    axes[2].text(.5, .69, "Positive favors delta", ha="center", fontsize=8, color="#46596c")
    axes[2].text(.5, -1.00, f"Interaction: {advantage[1]-advantage[0]:+.4f}°", ha="center", fontsize=8)
    axes[2].grid(axis="y", alpha=.15)
    fig.subplots_adjust(left=.13, right=.985, top=.80, bottom=.21, wspace=.36)
    fig.text(.13, .96, "Movement fidelity depends on the readout objective", fontsize=13, weight="bold")
    fig.text(.13, .055, "Descriptive preview from author-supplied rounded means. No uncertainty intervals shown; all methods share development data.",
             fontsize=8, color="#50575d")
    for extension in ("pdf", "png"):
        fig.savefig(ROOT / f"descriptive-results.{extension}", dpi=190)
    plt.close(fig)
    result = dict(provenance="Author-supplied rounded follow-up table; not independently reconstructed", metrics=METRICS,
                  objective_order=["base", "paired_change"], family_means=DATA,
                  paired_change_minus_base={name: dict(zip(METRICS, np.round(np.subtract(v[1], v[0]), 4).tolist()))
                                             for name, v in DATA.items()},
                  response_endpoint_minus_delta=dict(base=round(float(advantage[0]), 4),
                      paired_change=round(float(advantage[1]), 4), interaction=round(float(advantage[1]-advantage[0]), 4)))
    (ROOT / "descriptive-results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["response_endpoint_minus_delta"], indent=2))


if __name__ == "__main__":
    main()
