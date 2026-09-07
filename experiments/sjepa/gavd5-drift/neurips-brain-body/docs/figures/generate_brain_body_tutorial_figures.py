#!/usr/bin/env python3
"""Build tutorial vector figures from the preserved notebook evidence.

Run from the repository root with:
  .venv/bin/python neurips-brain-body/docs/figures/generate_brain_body_tutorial_figures.py

The source notebooks are read without execution. Exact notebook SHA-256 guards
prevent a later execution from silently replacing the evidence used here. Data
are parsed from saved plain-text tables; numerical expectations provide a second
check against table-layout changes. A changed notebook requires a fresh audit
before its expected digest should be updated. Checkpoint bundles and fold-v2
JSON reports were unavailable in this checkout when these figures were written.
The provenance file therefore identifies notebook evidence, not a local rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np


OUT = Path(__file__).resolve().parent
NOTEBOOKS = OUT.parents[1]
EXPECTED_NOTEBOOK_SHA256 = {
    "01_gavd_manifest_and_youtube.ipynb": "4dbda1d1c28a20a341bb96515b7a9e893bbb434304877931a554b44dbecff038",
    "02_extract_and_watch_skeletons.ipynb": "7408cd21490c65040f6c177a73fbe17cb28b723ebd5d1d25b78238ec08004f89",
    "04_pretrain_sjepa_on_normal.ipynb": "919d77e4f93f3146b71e4c3d0b8f1868d5bb1064c0d40d0f7ef90479b847e1c1",
    "06_capstone_health_condition_classifiers.ipynb": "a9c3abf9d2759952e3fc6137b6982bbd53693ed8d421dea35bf84ec3570fc610",
    "07_temporal_readout_diagnostic.ipynb": "80b6152a0f7a2792d5abf1f690ac396655d3b7338c3d430f85305d7f1bed6524",
    "08_normal_anchor_drift_and_consolidation.ipynb": "84f0dffd34e9f74a66a32aec80090b45f0dfdb86eaa434940bb861d950f01b29",
    "09_predictive_surprise_world_model.ipynb": "0172e1f3c203f1431d241915843b3481ac501ed304d066a07c19f08772b8b955",
}

INK = "#183247"
MUTED = "#526575"
GRID = "#dce4e9"
BLUE = "#0072b2"
TEAL = "#007d70"
ORANGE = "#ce6b16"
GREY = "#7a8792"
PALE_BLUE = "#e8f3fa"
PALE_TEAL = "#e8f4f0"
PALE_ORANGE = "#fceede"
COLORS = [BLUE, GREY, TEAL]
RUN_NOTE = "Saved notebook execution; fold 0, seed 42; no interval estimates."

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.8,
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": GRID,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "svg.fonttype": "none",  # Keep labels editable and searchable.
    "pdf.fonttype": 42,
    "savefig.facecolor": "white",
    "svg.hashsalt": "brain-body-tutorial-20260905",
})


def sha(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def text_outputs(cell: dict) -> str:
    chunks = []
    for output in cell.get("outputs", []):
        value = output.get("text", output.get("data", {}).get("text/plain", ""))
        chunks.append("".join(value) if isinstance(value, list) else value)
    return "\n".join(chunks)


def plain_tables(cell: dict) -> str:
    return "\n".join("".join(item.get("data", {}).get("text/plain", []))
                     for item in cell.get("outputs", []))


def load_evidence() -> tuple[dict, dict]:
    notebooks = {}
    for name, expected in EXPECTED_NOTEBOOK_SHA256.items():
        data = (NOTEBOOKS / name).read_bytes()
        actual = sha(data)
        if actual != expected:
            raise RuntimeError(f"Notebook evidence changed: {name}; expected {expected}, got {actual}. Audit before rebuilding.")
        notebooks[name[:2]] = json.loads(data)

    def source(nb: str, cell: int) -> str:
        return "".join(notebooks[nb]["cells"][cell]["source"])

    def output(nb: str, cell: int) -> str:
        return text_outputs(notebooks[nb]["cells"][cell])

    raw = re.search(r"raw: (\d+) sequences / (\d+) sources", output("01", 10))
    public = re.search(r"metadata-public: (\d+) sequences / (\d+) sources", output("01", 10))
    decoded = re.search(r"decoded-frame eligible: (\d+) sequences / (\d+) sources", output("01", 14))
    pose = re.search(r"pose-QC eligible: (\d+) sequences / (\d+) sources", output("02", 12))
    attrition = np.array([[int(x) for x in match.groups()] for match in (raw, public, decoded, pose)])
    np.testing.assert_array_equal(attrition, [[666, 103], [657, 100], [655, 98], [639, 97]])

    split = np.array([[int(x) for x in re.search(rf"^{role}\s+(\d+)\s+(\d+)\s*$", output("07", 10), re.M).groups()]
                      for role in ("train", "validation", "test")])
    np.testing.assert_array_equal(split, [[60, 392], [20, 134], [20, 131]])
    # Pandas prints the role once per block; sum its five condition rows.
    post_qc = {}
    role = None
    for line in output("04", 8).splitlines():
        match = re.match(r"^(?:(train|validation|test)\s+)?\s*(cerebralpalsy|myopathic|normal|parkinsons|stroke)\s+(\d+)\s+(\d+)\s*$", line)
        if match:
            role = match[1] or role
            post_qc.setdefault(role, [0, 0])
            post_qc[role][0] += int(match[3])
            post_qc[role][1] += int(match[4])
    assert post_qc == {"train": [377, 59], "validation": [131, 18], "test": [131, 20]}, post_qc

    readout_text = plain_tables(notebooks["06"]["cells"][13])
    block = readout_text.split("test_source_balanced_accuracy  test_source_macro_f1  test_source_sources", 1)[1].split("test_sequence_accuracy", 1)[0]
    readout = np.array([[float(x) for x in match.groups()]
                        for match in re.finditer(r"^\s*\d+\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$", block, re.M)])
    np.testing.assert_allclose(readout, [[.257143, .292424, 20], [.247619, .251111, 20], [.442857, .440513, 20]], atol=0, rtol=0)

    temporal_text = plain_tables(notebooks["07"]["cells"][19])
    block = temporal_text.split("source_equal_mae  source_level_r2  test_sources", 1)[1]
    temporal = np.array([[float(x) for x in match.groups()]
                         for match in re.finditer(r"^\s*\d+\s+([\d.]+)\s+([-\d.]+)\s+(\d+)\s*$", block, re.M)])
    expected_r2 = [.173011, .052344, .317745, .105126, .053575, .176175, -.070535, -.053989, -.052171]
    np.testing.assert_allclose(temporal[:, 1], expected_r2, atol=0, rtol=0)
    np.testing.assert_array_equal(temporal[:, 2], np.repeat(20, 9))

    drift_text = plain_tables(notebooks["08"]["cells"][15])
    drift = {role: [] for role in ("train", "validation")}
    for match in re.finditer(r"^\s*\d+\s+(\d+)\s+(\w+)\s+(train|validation)\s+([\d.]+)\s+(\d+)\s*$", drift_text, re.M):
        drift[match[3]].append([int(match[1]), float(match[4]), int(match[5])])
    np.testing.assert_allclose(np.array(drift["train"])[:, 1], [1, .995692, .984225, .892483, .867689], atol=0, rtol=0)
    np.testing.assert_allclose(np.array(drift["validation"])[:, 1], [1, .992833, .969039, .737187, .701058], atol=0, rtol=0)
    test_text = plain_tables(notebooks["08"]["cells"][17])
    match = re.search(r"^0\s+(0\.701058)\s+(0\.849632)\s*$", test_text, re.M)
    assert match, "Final validation/test anchor result changed."
    final_test = float(match[2])
    assert re.search(r"test_normal_sources\s*\n0\s+7", test_text)

    assert "mean(dim=2)" in source("04", 5)
    assert "return self.net(context[:, None, None, :] + self.position[None])" in source("04", 5)
    assert '"64"' in source("04", 10) and '"segment_length": 4' in source("04", 10)
    assert 'MASK_FRACTION = float(os.getenv("SJEPA_MASK_FRACTION", "0.60"))' in source("04", 10)
    assert 'EMA_DECAY = float(os.getenv("SJEPA_EMA_DECAY", "0.996"))' in source("04", 10)
    assert '"vicreg_variance_weight": 0.10' in source("04", 10)
    assert '"vicreg_covariance_weight": 0.01' in source("04", 10)
    assert "MASK_KEYPOINTS = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]" in source("04", 4)
    assert "mask[:, SEGMENTS - horizon:, MASK_KEYPOINTS] = True" in source("09", 15)
    assert "predicted_all = self.predictor(online, ~target_mask)" in source("09", 11)
    assert "SKIPPED: no future-mask scores were computed." in output("09", 15)
    assert "SKIPPED: outer test was not opened." in output("09", 17)

    data = {
        "attrition_sequences_sources": attrition.tolist(),
        "metadata_split_sources_sequences": split.tolist(),
        "post_qc_split_sequences_sources": post_qc,
        "readout_lanes": ["sjepa_latent", "missingness_only", "raw_kinematics"],
        "readout_columns": ["balanced_accuracy", "macro_f1", "test_sources"],
        "readout": readout.tolist(),
        "temporal_target_order": ["peak_phase", "energy_ratio", "phase_lag"],
        "temporal_lane_order": ["A_mean_std", "B_signed_moment", "C_time_bins"],
        "temporal_columns": ["source_equal_mae", "source_level_r2", "test_sources"],
        "temporal": temporal.tolist(),
        "drift_columns": ["stage", "source_equal_anchor_cosine", "normal_sources"],
        "drift": drift,
        "final_test_anchor_cosine": final_test,
        "final_test_normal_sources": 7,
        "model_schematic": {"frames": 64, "joints": 33, "frames_per_patch": 4, "segments": 16,
                            "all_tokens": 528, "max_eligible_tokens": 192,
                            "mask_fraction_of_batch_minimum_valid_eligible": .6,
                            "fully_valid_target_count": 115, "fully_valid_fraction_of_all_tokens": 115 / 528,
                            "ema_decay": .996, "variance_weight": .10, "covariance_weight": .01},
        "forecast_schematic": {"illustrative_horizon_segments": 4,
                                "current_suffix_target_joints": 12, "current_visible_suffix_joints": 21,
                                "prefix_only_future_context": "required change; not executed",
                                "numerical_result_available": False},
    }
    used_cells = {"01": [10, 14], "02": [12], "04": [4, 5, 8, 10, 11],
                  "06": [7, 13], "07": [10, 14, 17, 19], "08": [15, 17], "09": [11, 12, 15, 17]}
    evidence_cells = []
    for nb, indexes in used_cells.items():
        name = next(name for name in EXPECTED_NOTEBOOK_SHA256 if name.startswith(nb))
        for index in indexes:
            cell = notebooks[nb]["cells"][index]
            evidence_cells.append({
                "notebook": f"../../{name}", "cell_index_zero_based": index,
                "execution_count": cell.get("execution_count"),
                "source_sha256": sha("".join(cell["source"])),
                "text_output_sha256": sha(text_outputs(cell)),
                "execution_metadata": cell.get("metadata", {}).get("execution", {}),
            })
    provenance = {
        "evidence_status": "Saved notebook output and inspected code; not locally rerun.",
        "scope": "Protocol-v2 fold 0 seed 42; no interval or cross-fold estimates.",
        "audit_date": "2026-09-05",
        "limitation": "The corresponding current fold-v2 checkpoint and JSON report bundles were absent locally at the time of writing.",
        "extractor": Path(__file__).name,
        "notebook_sha256": EXPECTED_NOTEBOOK_SHA256,
        "cells": evidence_cells,
        "selected_data": data,
        "privacy": "No individual trajectories, frames, sequence identifiers, or video identifiers are copied into these figure assets.",
    }
    return data, provenance


def canvas(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, text, color=PALE_BLUE, fontsize=9.8, edge=GRID):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.07",
                               linewidth=1, edgecolor=edge, facecolor=color))
    return ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.5)


def arrow(ax, start, end, color=MUTED, dashed=False):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.4, color=color, linestyle="--" if dashed else "-"))


def footer(fig, note=RUN_NOTE):
    fig.text(.025, .015, note, ha="left", va="bottom", color=MUTED, fontsize=8.5, linespacing=1.4)


def save(fig, name, preview_dir):
    for extension in ("svg", "pdf"):
        metadata = {"Creator": Path(__file__).name, "Date": None} if extension == "svg" else {
            "Creator": Path(__file__).name, "CreationDate": None, "ModDate": None}
        fig.savefig(OUT / f"{name}.{extension}", bbox_inches="tight", pad_inches=.10, metadata=metadata)
    if preview_dir:
        fig.savefig(preview_dir / f"{name}.png", dpi=125, bbox_inches="tight", pad_inches=.10)
    plt.close(fig)


def data_figure(data, preview_dir):
    fig, ax = canvas(6.7, 4.3)
    fig.subplots_adjust(left=.025, right=.985, bottom=.11, top=.96)
    ax.text(0, 6.85, "A. Count availability at each measurement gate", fontsize=11.5, weight="bold")
    labels = ["Raw annotations", "Public metadata", "Decoded span", "Pose QC passed"]
    for i, (label, counts) in enumerate(zip(labels, data["attrition_sequences_sources"])):
        x = i * 3.05
        box(ax, x, 5.10, 2.6, 1.18, f"{label}\n{counts[0]} sequences\n{counts[1]} sources",
            color=PALE_TEAL if i == 3 else PALE_BLUE)
        if i < 3:
            arrow(ax, (x + 2.64, 5.68), (x + 3.00, 5.68))
    ax.text(4.35, 4.48, "Freeze source roles here", fontsize=9.8, ha="center", color=BLUE)
    arrow(ax, (4.35, 4.74), (4.35, 5.07), color=BLUE)
    ax.text(0, 3.82, "B. Retain source roles through later QC", fontsize=11.5, weight="bold")
    for i, role in enumerate(("train", "validation", "test")):
        x = .5 + i * 4.0
        initial, _ = data["metadata_split_sources_sequences"][i]
        sequences, sources = data["post_qc_split_sequences_sources"][role]
        ax.text(x + 1.4, 3.23, role.capitalize(), ha="center", fontsize=10.5, weight="bold")
        box(ax, x, 2.35, 2.8, .55, f"{initial} public sources")
        arrow(ax, (x + 1.4, 2.29), (x + 1.4, 1.75))
        box(ax, x, .89, 2.8, .8, f"QC: {sources} sources\n{sequences} sequences", color=PALE_TEAL)
    ax.text(6.0, .18, "Uploads define groups; person-level independence is unknown.",
            ha="center", fontsize=9.8, color=MUTED)
    footer(fig, "Saved September 2026 outputs; source split shown for fold 0.")
    save(fig, "tutorial_01_data_and_source_split", preview_dir)


def model_figure(data, preview_dir):
    fig, ax = canvas(6.7, 5.55)
    fig.subplots_adjust(left=.025, right=.985, bottom=.105, top=.97)
    ax.text(0, 6.84, "The current fold-local model", fontsize=12, weight="bold")
    box(ax, .1, 5.58, 3.25, .88, "64 frames × 33 joints\nNormalized xyz\nRetain validity mask")
    box(ax, 4.35, 5.58, 3.25, .88, "Average each 4 frames\nWithin-patch order\nis removed")
    box(ax, 8.6, 5.58, 3.25, .88, "16 × 33 = 528 tokens\nProject xyz; add\ntime and joint codes")
    arrow(ax, (3.42, 6.04), (4.28, 6.04))
    arrow(ax, (7.67, 6.04), (8.53, 6.04))
    box(ax, .1, 4.27, 11.75, .91,
        "At most 12 eligible joints × 16 segments = 192 target positions.\n"
        "Mask 60% of the batch-minimum valid-eligible count.\n"
        "Fully valid case: 115 targets / 528 tokens = 21.8%.",
        color=PALE_ORANGE)
    ax.text(6, 3.86, "Both encoders receive the same prepared sequence.", fontsize=9.8, ha="center", color=MUTED)
    box(ax, .1, 2.60, 2.70, .96, "Online encoder\nReplace targets\nwith mask tokens")
    box(ax, 3.43, 2.60, 2.5, .96, "Average tokens\nOnly valid-visible\npositions")
    box(ax, 6.57, 2.60, 2.35, .96, "Target position\n+ pooled context\n→ MLP predictor")
    box(ax, .1, .76, 2.70, .96, "Target encoder\nFull input\nNo gradients", color=PALE_TEAL)
    box(ax, 3.43, .76, 2.5, .96, "Target features\nSelect masked\npositions", color=PALE_TEAL)
    box(ax, 9.56, 1.66, 2.29, .96, "Smooth-L1\nPredicted vs.\ntarget features", color=PALE_ORANGE)
    arrow(ax, (2.86, 3.05), (3.36, 3.05))
    arrow(ax, (5.99, 3.05), (6.50, 3.05))
    arrow(ax, (8.98, 2.94), (9.51, 2.30))
    arrow(ax, (2.86, 1.24), (3.36, 1.24))
    arrow(ax, (5.99, 1.24), (9.51, 1.92))
    arrow(ax, (1.44, 2.54), (1.44, 1.78), color=TEAL, dashed=True)
    ax.text(1.66, 2.15, "EMA update\ndecay 0.996", fontsize=9.8, va="center", color=TEAL)
    ax.text(6, .23, "Loss = Smooth-L1 + 0.10 variance + 0.01 covariance",
            fontsize=9.8, ha="center")
    footer(fig, "Notebook 04 code; real-mode defaults. Folder labels determine the curriculum.")
    save(fig, "tutorial_02_model_and_masking", preview_dir)


def readout_figure(data, preview_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.7, 3.0), sharey=True)
    fig.subplots_adjust(left=.205, right=.98, bottom=.255, top=.765, wspace=.28)
    values = np.asarray(data["readout"])
    labels = ["Learned latent", "Missingness only", "Raw kinematics"]
    for ax, column, title in zip(axes, (1, 0), ("Macro-F1", "Balanced accuracy")):
        for row, (value, color) in enumerate(zip(values[:, column], COLORS)):
            ax.barh(2 - row, value, color=color, height=.50)
            ax.text(value + .015, 2 - row, f"{value:.3f}", va="center", fontsize=10.1, weight="bold")
        ax.set_xlim(0, .60)
        ax.set_xticks([0, .2, .4, .6])
        ax.set_yticks([2, 1, 0], labels)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.set_title(title, loc="left", fontsize=10.5, pad=11)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=9.8)
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("Source-level score")
    fig.suptitle("Readout scores on 20 held-out sources", x=.025, y=.99, ha="left", fontsize=12, weight="bold")
    footer(fig, RUN_NOTE)
    save(fig, "tutorial_03_source_readouts", preview_dir)


def temporal_figure(data, preview_dir):
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 5.55), sharex=True)
    fig.subplots_adjust(left=.255, right=.98, bottom=.235, top=.855, hspace=.50)
    values = np.asarray(data["temporal"])[:, 1].reshape(3, 3)
    labels = ["A: mean / SD", "B: signed moment", "C: four time bins"]
    colors = [GREY, BLUE, TEAL]
    for ax, scores, title in zip(axes, values, ("Peak position (clip-relative)", "Late/early motion ratio", "Bilateral lag (pose proxy)")):
        ax.axvline(0, color=INK, linewidth=1.2)
        for lane, (value, color) in enumerate(zip(scores, colors)):
            y = 2 - lane
            ax.hlines(y, min(value, 0), max(value, 0), color=color, linewidth=3)
            ax.scatter(value, y, s=35, color=color, zorder=3)
            ax.annotate(f"{value:.3f}", (value, y), xytext=(7 if value >= 0 else -7, 0), textcoords="offset points",
                        ha="left" if value >= 0 else "right", va="center", fontsize=9.8, color=color, weight="bold")
        ax.set(xlim=(-.16, .43), ylim=(-.5, 2.5))
        ax.set_xticks([-.1, 0, .1, .2, .3, .4])
        ax.set_yticks([2, 1, 0], labels)
        ax.set_title(title, loc="left", fontsize=10.5, pad=8)
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=9, labelsize=9.8)
    axes[-1].set_xlabel("Source-level R²", labelpad=7)
    fig.suptitle("Temporal readouts of pose-derived targets", x=.025, y=.99, ha="left", fontsize=12, weight="bold")
    fig.text(.025, .104, "Negative R²: error exceeds the test-source mean reference.", color=MUTED, fontsize=9.8)
    footer(fig, RUN_NOTE + "\nAll probes use the same 20 held-out sources.")
    save(fig, "tutorial_04_temporal_probes", preview_dir)


def drift_figure(data, preview_dir):
    fig, (ax, test_ax) = plt.subplots(1, 2, figsize=(6.7, 3.85), gridspec_kw={"width_ratios": [4.8, 1.0]}, sharey=True)
    fig.subplots_adjust(left=.095, right=.98, bottom=.31, top=.74, wspace=.25)
    x = np.arange(5)
    for role, color, label in (("train", BLUE, "Train: 18 normal sources"),
                               ("validation", ORANGE, "Validation: 5 normal sources")):
        scores = np.asarray(data["drift"][role])[:, 1]
        ax.plot(x, scores, marker="o", color=color, label=label, linewidth=1.8, markersize=4.5)
        for i, value in enumerate(scores):
            if i == 0:
                continue
            offset = 11 if role == "train" else -15
            ax.annotate(f"{value:.3f}", (i, value), xytext=(0, offset), textcoords="offset points",
                        ha="center", fontsize=9.8, color=color)
    ax.annotate("1.000", (0, 1), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9.8)
    ax.set_xticks(x, ["Normal", "+ Parkinson's", "+ Stroke", "+ Myopathic", "+ Cerebral\npalsy"])
    ax.tick_params(axis="x", labelsize=9.8)
    ax.set(xlim=(-.35, 4.35), ylim=(.58, 1.09), ylabel="Normal-anchor cosine")
    ax.set_yticks([.6, .7, .8, .9, 1])
    ax.grid(axis="y", color=GRID, linewidth=.7)
    ax.legend(loc="lower left", frameon=False, fontsize=9.8, borderpad=.1, handlelength=1.7)
    ax.set_title("Cumulative curriculum", loc="left", fontsize=10.5, pad=13)
    final = data["final_test_anchor_cosine"]
    test_ax.scatter([0], [final], marker="D", color=TEAL, s=55, zorder=3)
    test_ax.annotate(f"{final:.3f}", (0, final), xytext=(0, 13), textcoords="offset points",
                     ha="center", fontsize=10.5, weight="bold", color=TEAL)
    test_ax.set(xticks=[0], xticklabels=["Final\nstage"], xlim=(-.7, .7))
    test_ax.set_title("Test normals\n7 sources", fontsize=10.5, pad=13)
    test_ax.grid(axis="y", color=GRID, linewidth=.7)
    test_ax.tick_params(axis="y", left=False, labelleft=False)
    test_ax.spines["left"].set_visible(False)
    test_ax.tick_params(axis="x", labelsize=9.8)
    fig.suptitle("Representations change during cumulative training", x=.025, y=.99, ha="left", fontsize=12, weight="bold")
    fig.text(.025, .153, "Cosine to stage 0 measures geometric change, not a retention fraction.", fontsize=9.8, color=MUTED)
    footer(fig, RUN_NOTE + "\nThe final test point measures different sources from the validation curve.")
    save(fig, "tutorial_05_normal_anchor_drift", preview_dir)


def forecast_figure(data, preview_dir):
    fig, axes = plt.subplots(2, 1, figsize=(6.7, 5.1))
    fig.subplots_adjust(left=.195, right=.98, bottom=.205, top=.775, hspace=1.10)
    for index, ax in enumerate(axes):
        ax.set(xlim=(0, 16), ylim=(0, 2))
        for row in range(2):
            for col in range(16):
                if col < 12:
                    color = PALE_BLUE
                elif row == 1:
                    color = PALE_ORANGE
                else:
                    color = PALE_BLUE if index == 0 else "#e4e7ea"
                rect = Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=1.2)
                ax.add_patch(rect)
        ax.axvline(12, color=INK, linewidth=1.5)
        ax.set_xticks([0, 4, 8, 12, 16], ["0", "4", "8", "12", "16"])
        ax.set_yticks([.5, 1.5], ["Other 21\nlandmarks", "Whitelist 12\nlandmarks"])
        ax.tick_params(axis="y", length=0, pad=8, labelsize=10)
        ax.set_xlabel("Temporal segment (4 averaged frames)", labelpad=7, fontsize=9.8)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.text(6, 1.5, "Visible", ha="center", va="center", color=BLUE, weight="bold", fontsize=9.8)
        ax.text(6, .5, "Visible", ha="center", va="center", color=BLUE, weight="bold", fontsize=9.8)
        ax.text(14, 1.5, "Hidden targets", ha="center", va="center", fontsize=9.8, color=ORANGE, weight="bold")
        ax.text(14, .5, "Visible" if index == 0 else "Hidden", ha="center", va="center",
                fontsize=9.8, color=BLUE if index == 0 else MUTED, weight="bold")
        ax.text(6, 2.16, "Observed prefix", ha="center", fontsize=9.8, color=MUTED)
        ax.text(14, 2.16, "Future suffix", ha="center", fontsize=9.8, color=MUTED)
    axes[0].set_title("A. Current mask: 21 future landmarks remain visible", loc="left", fontsize=10.5, pad=28)
    axes[1].set_title("B. Required prefix-only context — not executed", loc="left", fontsize=10.5, pad=28)
    fig.suptitle("Forecasting depends on the allowed context", x=.025, y=.99, ha="left", fontsize=12, weight="bold")
    fig.text(.025, .907, "Select evaluation targets separately from the context mask.", fontsize=9.8, color=MUTED)
    footer(fig, "Code audit; illustrative horizon: 4 of 16 segments.\nNotebook 09 computed no forecast scores and left the outer test unopened.")
    save(fig, "tutorial_06_forecast_context_audit", preview_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional temporary directory for raster inspection only.")
    args = parser.parse_args()
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
    data, provenance = load_evidence()
    for draw in (data_figure, model_figure, readout_figure, temporal_figure, drift_figure, forecast_figure):
        draw(data, args.preview_dir)
    provenance["generator_sha256"] = sha(Path(__file__).read_bytes())
    provenance["figures"] = [
        {"filename": path.name, "sha256": sha(path.read_bytes())}
        for path in sorted(OUT.glob("tutorial_0*")) if path.suffix in (".svg", ".pdf")
    ]
    (OUT / "tutorial_figure_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(EXPECTED_NOTEBOOK_SHA256)} notebook digests; wrote 6 editable SVGs, 6 vector PDFs, and provenance to {OUT}")
    if args.preview_dir:
        print(f"Raster inspection copies: {args.preview_dir}")


if __name__ == "__main__":
    main()
