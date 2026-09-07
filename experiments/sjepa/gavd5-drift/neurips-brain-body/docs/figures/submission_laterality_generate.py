#!/usr/bin/env python3
"""Render aggregate laterality contrasts for the BrainBody submission.

Run from the repository root:
  .venv/bin/python neurips-brain-body/docs/figures/submission_laterality_generate.py

Reads the registered paper report directly. Predictive results are means of
per-seed metrics, not metrics of seed-averaged ensemble predictions. The source
bootstrap conditions on the fitted cross-fitted models and does not refit them.
No individual pose, sequence identifier, or source identifier is exported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
RUN = ROOT / "neurips-laterality/artifacts/paper/protocol_6f7baefbda07"
STEM = "submission_laterality_effects"
EXPECTED_HASHES = {
    "report/checkpoint_source_bootstrap.csv": "0a0f7e1ed439f95a2f76c01aeb6c7e9691548f484647c5cfe775f1fa93e9acd1",
    "report/strict_representation_equivariance_source_bootstrap.csv": "9e02963be9f9259c11af1ea2d93f1f26d1d79d6b45406a8b447506f9a43859ef",
    "report/summary.json": "b864f35d6da52660a49ecd25811ce739788b33e61203912c063e8e1b02bb6bd6",
    "protocol_snapshot.json": "92f12fe7210cd2f3d1a4ff84d0e3adb472e78af35adb722b3cd9cd594e78ec6a",
}
BLUE = "#0072b2"
TEAL = "#007d70"
ORANGE = "#bf6518"
INK = "#183247"
MUTED = "#526575"
GRID = "#dce4e9"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_inputs():
    data = {}
    for relative, expected in EXPECTED_HASHES.items():
        payload = (RUN / relative).read_bytes()
        observed = digest(payload)
        if observed != expected:
            raise RuntimeError(f"Registered report changed: {relative}; expected {expected}, got {observed}. Reaudit before rebuilding.")
        data[relative] = (list(csv.DictReader(io.StringIO(payload.decode("utf-8"))))
                          if relative.endswith(".csv") else json.loads(payload))
    summary = data["report/summary.json"]
    snapshot = data["protocol_snapshot.json"]
    assert summary["paper_run_complete"] and not summary["synthetic_evidence"]
    assert summary["profile"] == "paper"
    assert summary["selected_folds"] == list(range(5))
    assert summary["selected_seeds"] == list(range(42, 47))
    assert summary["protocol_digest"] == snapshot["protocol_digest"]
    assert summary["context_digest"] == snapshot["context_digest"]
    prediction = data["report/checkpoint_source_bootstrap.csv"]
    symmetry = data["report/strict_representation_equivariance_source_bootstrap.csv"]

    def select(rows, **criteria):
        found = [row for row in rows if all(row[key] == value for key, value in criteria.items())]
        assert len(found) == 1, (criteria, len(found))
        row = found[0]
        assert int(row["bootstrap_repetitions"]) == 2000
        assert int(row["registered_seed_count"]) == 5
        assert row["uncertainty_scope"].startswith("source_resampling_conditional_on_fixed_cross_fitted_checkpoints")
        return row

    predictive = [
        select(prediction, comparison_type="primary_training_content", variant_a="vanilla"),
        select(prediction, comparison_type="constructed_training_content", variant_a="vanilla"),
        select(prediction, comparison_type="reflection_minus_vanilla_primary"),
    ]
    tokens = [
        select(symmetry, comparison_type="learned_minus_initial_strict_equivariance", variant_a="vanilla"),
        select(symmetry, comparison_type="learned_minus_initial_strict_equivariance", variant_a="reflection_augmented"),
        select(symmetry, comparison_type="reflection_minus_vanilla_strict_equivariance"),
    ]
    expected_predictive = [
        [-0.017983928580114723, -0.038513529861257965, 0.002481630927645069],
        [-0.058743975750874755, -0.09549126061617483, -0.017398157750384884],
        [0.004082790379340208, -0.0055599426818007825, 0.012772985390520403],
    ]
    expected_tokens = [
        [0.03055140816298557, 0.015764135487161903, 0.04762529731917507],
        [0.022125287174293115, 0.008439162619211868, 0.039342331014064445],
        [-0.008426120988692457, -0.010195763057535874, -0.006871700241680472],
    ]
    for rows, expected in ((predictive, expected_predictive), (tokens, expected_tokens)):
        values = [[float(row[key]) for key in ("estimate", "ci95_low", "ci95_high")] for row in rows]
        np.testing.assert_allclose(values, expected, rtol=0, atol=0)
    return predictive, tokens, summary


def render(predictive, tokens, preview_dir=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.7, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "brainbody-laterality-6f7baefbda07", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.43))
    fig.subplots_adjust(left=.218, right=.98, bottom=.34, top=.825, wspace=.95)
    labels = [
        ["Native:\nlearned − init.", "Constructed:\nlearned − init.", "Augmented −\nvanilla (native)"],
        ["Vanilla:\nlearned − init.", "Augmented:\nlearned − init.", "Augmented −\nvanilla"],
    ]
    for ax, rows, lane_labels in zip(axes, (predictive, tokens), labels):
        ax.axvline(0, color=INK, linewidth=1.05, zorder=1)
        for y, row, color in zip((2, 1, 0), rows, (BLUE, ORANGE, TEAL)):
            estimate, low, high = (float(row[key]) for key in ("estimate", "ci95_low", "ci95_high"))
            ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]],
                        fmt="o", markersize=4.1, color=color, elinewidth=1.8,
                        capsize=3.0, capthick=1.05, zorder=3)
            ax.annotate(f"{estimate:+.3f}", (estimate, y), xytext=(0, 10),
                        textcoords="offset points", ha="center", fontsize=9.7,
                        color=color, weight="bold")
        ax.set_yticks([2, 1, 0], lane_labels)
        ax.set_ylim(-.38, 2.65)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=9.7)
        ax.tick_params(axis="x", labelsize=9.7)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_xlim(-.115, .035)
    axes[0].set_xticks([-.10, -.05, 0], ["−.10", "−.05", "0"])
    axes[0].set_title("A. Predictive value", loc="left", fontsize=10.3, weight="bold", pad=9)
    axes[0].set_xlabel("ΔR² (higher is better)", fontsize=9.7, labelpad=5)
    axes[1].set_xlim(-.025, .065)
    axes[1].set_xticks([-.02, 0, .02, .04, .06], ["−.02", "0", ".02", ".04", ".06"])
    axes[1].set_title("B. Token symmetry", loc="left", fontsize=10.3, weight="bold", pad=9)
    axes[1].set_xlabel("Δq (lower is better)", fontsize=9.7, labelpad=5)
    fig.text(.02, .088, "5 folds × 5 seeds; 95% pointwise source-bootstrap CIs.",
             fontsize=9.3, color=MUTED)
    fig.text(.02, .021, "Conditional on fixed fits; init. = paired initialization.",
             fontsize=9.3, color=MUTED)
    # Fix the exported physical dimensions to the official 5.5-inch text width.
    # Center the complete content inside this box rather than scaling its text.
    fig.canvas.draw()
    content = fig.get_tightbbox(fig.canvas.get_renderer())
    print_width, print_height = 5.5, 2.3
    if content.width > print_width or content.height > print_height:
        raise RuntimeError(f"Figure content no longer fits the {print_width} × {print_height}-inch publication box: {content.width:.3f} × {content.height:.3f}.")
    publication_box = Bbox.from_bounds(
        content.x0 - (print_width - content.width) / 2,
        content.y0 - (print_height - content.height) / 2,
        print_width, print_height,
    )
    for extension in ("svg", "pdf"):
        metadata = ({"Creator": Path(__file__).name, "Date": None} if extension == "svg" else
                    {"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None})
        fig.savefig(OUT / f"{STEM}.{extension}", bbox_inches=publication_box, pad_inches=0, metadata=metadata)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{STEM}.png", dpi=150, bbox_inches=publication_box, pad_inches=0)
    plt.close(fig)


def provenance(predictive, tokens, summary):
    payload = {
        "schema": "brainbody_aggregate_laterality_figure/v1",
        "generator": Path(__file__).name,
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "report_directory": str(RUN.relative_to(ROOT)),
        "profile": summary["profile"],
        "protocol_digest": summary["protocol_digest"],
        "context_digest": summary["context_digest"],
        "cohort_digest": summary["cohort_digest"],
        "selected_folds": summary["selected_folds"],
        "selected_seeds": summary["selected_seeds"],
        "selected_variants": summary["selected_variants"],
        "source_sha256": EXPECTED_HASHES,
        "producer_code_sha256": {
            relative: digest((ROOT / relative).read_bytes()) for relative in (
                "neurips-laterality/laterality/reporting.py",
                "neurips-laterality/laterality/evaluation.py",
            )
        },
        "panel_a": {"source": "report/checkpoint_source_bootstrap.csv", "rows": predictive},
        "panel_b": {"source": "report/strict_representation_equivariance_source_bootstrap.csv", "rows": tokens},
        "uncertainty": "95% pointwise source-bootstrap intervals from 2000 draws, conditional on fixed cross-fitted models; no refitting or multiplicity adjustment.",
        "publication_geometry": {"intended_print_width_inches": 5.5,
                                 "source_figure_width_inches": 5.5,
                                 "source_figure_height_inches": 2.43,
                                 "exported_width_inches": 5.5,
                                 "exported_height_inches": 2.3,
                                 "minimum_source_label_size_points": 9.7,
                                 "minimum_source_evidence_note_size_points": 9.3,
                                 "note": "PDF/SVG dimensions are fixed at 5.5 × 2.3 inches without scaling text; rendering fails if the complete content exceeds that box."},
        "scope": "Aggregate results only. No clinical, person-held-out, or population-level guarantee is established.",
        "privacy": "No individual poses, sequence identifiers, or source identifiers are exported.",
        "assets": {f"{STEM}.{extension}": digest((OUT / f"{STEM}.{extension}").read_bytes())
                   for extension in ("svg", "pdf")},
    }
    (OUT / "submission_laterality_provenance.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional temporary PNG directory for visual review.")
    args = parser.parse_args()
    predictive, tokens, summary = read_inputs()
    render(predictive, tokens, args.preview_dir)
    provenance(predictive, tokens, summary)
    print(f"Wrote {STEM}.svg/.pdf and submission_laterality_provenance.json; six exact report contrasts verified.")


if __name__ == "__main__":
    main()
