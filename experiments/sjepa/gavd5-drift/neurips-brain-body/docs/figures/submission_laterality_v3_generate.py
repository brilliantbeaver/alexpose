#!/usr/bin/env python3
"""Render the V3-only, plain-language laterality figure from verified reports.

Run from the repository root:
  .venv/bin/python neurips-brain-body/docs/figures/submission_laterality_v3_generate.py \
      --preview-dir /absolute/path/to/a/temporary/directory

The shared generator supplies only its read-only input verification. Its
rendering and provenance functions are never called, and its files are checked
for changes before and after this V3 build. No experiments are rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

# Importing the shared reader must not create files alongside shared assets.
sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import Bbox


OUT = Path(__file__).resolve().parent
STEM = "submission_laterality_effects_v3"
PROVENANCE_NAME = "submission_laterality_v3_provenance.json"
SHARED_GENERATOR = OUT / "submission_laterality_generate.py"
PROTECTED = (
    SHARED_GENERATOR,
    OUT / "submission_laterality_effects.svg",
    OUT / "submission_laterality_effects.pdf",
    OUT / "submission_laterality_provenance.json",
)

WIDTH = 5.5
HEIGHT = 6.3
FONT = 9.7
TITLE_FONT = 11.0
PLOT_LEFT = 2.94
PLOT_RIGHT = 5.35
INK = "#183247"
NEUTRAL = "#536875"
TEAL = "#007d70"
GRID = "#e1e7ec"
MUTED = "#526575"
AGREEMENT_TITLE = "Feature agreement between original and mirrored clips"
TRAINING_GROUP = "Effect of encoder training"
MIRRORING_GROUP = "Effect of adding mirrored training clips"
TRAINING_REFERENCE = "Reference: encoder at its random starting weights"
MIRRORING_REFERENCE = "Reference: encoder trained without mirrored clips"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_inputs():
    spec = importlib.util.spec_from_file_location("laterality_shared_reader", SHARED_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the shared read-only input verifier.")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    # read_inputs checks the pinned source-file hashes, exact six estimates and
    # confidence intervals, completed report status, five folds, and five seeds.
    predictive, mismatch, summary = shared.read_inputs()
    return predictive, mismatch, summary, shared


def disagreement_reduction_rows(source_rows):
    """Reverse the displayed error contrast without changing source evidence.

    The reports store error(tested condition) minus error(labeled reference).
    A reduction is error(labeled reference) minus error(tested condition), so
    negating the estimate also requires reversing and negating the interval
    endpoints. Returning fresh, explicitly labeled records keeps these display
    values distinct from the original reported rows in the provenance file.
    """
    before = json.dumps(source_rows, sort_keys=True)
    displayed_rows = []
    for index, source in enumerate(source_rows):
        estimate, low, high = (float(source[key])
                               for key in ("estimate", "ci95_low", "ci95_high"))
        assert low <= estimate <= high
        displayed = {
            "source_row_index": index,
            "quantity": "Reduction in normalized feature disagreement",
            "estimate": -estimate,
            "ci95_low": -high,
            "ci95_high": -low,
        }
        assert displayed["ci95_low"] <= displayed["estimate"] <= displayed["ci95_high"]
        # An involution: the same conversion must recover all three source
        # values exactly, with no rounding or new uncertainty calculation.
        recovered = (-displayed["estimate"],
                     -displayed["ci95_high"], -displayed["ci95_low"])
        assert recovered == (estimate, low, high)
        assert displayed["ci95_high"] - displayed["ci95_low"] == high - low
        assert (displayed["ci95_low"] <= 0 <= displayed["ci95_high"]) == (low <= 0 <= high)
        displayed_rows.append(displayed)
    assert json.dumps(source_rows, sort_keys=True) == before
    assert [f"{row['estimate']:+.3f}" for row in displayed_rows] == ["-0.031", "-0.022", "+0.008"]
    return displayed_rows


def validate_layout(fig, row_labels):
    """Check exported bounds, label/plot separation, and text collisions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [artist for artist in fig.findobj(Text)
             if artist.get_visible() and artist.get_text().strip()]
    canvas = Bbox.from_bounds(0, 0, WIDTH * fig.dpi, HEIGHT * fig.dpi)
    boxes = []
    for artist in texts:
        box = artist.get_window_extent(renderer)
        if (box.x0 < canvas.x0 or box.y0 < canvas.y0
                or box.x1 > canvas.x1 or box.y1 > canvas.y1):
            raise RuntimeError(f"Text exceeds the fixed publication box: {artist.get_text()!r}")
        if artist.get_fontsize() < FONT:
            raise RuntimeError(f"Text is smaller than {FONT} pt: {artist.get_text()!r}")
        boxes.append((artist, box))
    for artist in row_labels:
        if artist.get_window_extent(renderer).x1 > (PLOT_LEFT - .10) * fig.dpi:
            raise RuntimeError(f"Row label intrudes on the forest plot: {artist.get_text()!r}")
    for index, (left_artist, left_box) in enumerate(boxes):
        for right_artist, right_box in boxes[index + 1:]:
            overlap = Bbox.intersection(left_box, right_box)
            if overlap is not None and overlap.width > .5 and overlap.height > .5:
                raise RuntimeError(
                    f"Text collision: {left_artist.get_text()!r} / {right_artist.get_text()!r}")
    return {"visible_text_items": len(texts), "bounds_check": "passed",
            "text_collision_check": "passed", "row_label_plot_separation_check": "passed"}


def render(predictive, reduction, preview_dir):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT,
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": INK,
        "axes.edgecolor": GRID, "axes.linewidth": .6,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "brainbody-laterality-v3-plain-comparisons",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=150)
    row_labels = []

    def label(x, y, value, **kwargs):
        return fig.text(x / WIDTH, y / HEIGHT, value, fontsize=FONT,
                        ha="left", va="center", **kwargs)

    # Physical coordinates give the references their own lines. Conditions
    # shared by the first two prediction rows are stated once; only the
    # prediction-rule setting changes between those two row labels.
    panels = [
        {"title_y": 6.13, "title": "A. Laterality prediction", "rows": predictive,
         "group_y": (5.87, 4.67), "reference_y": (5.68, 4.48),
         "shared_note": "Training excludes mirrored clips", "shared_note_y": 5.49,
         "row_y": (5.28, 5.00, 4.25), "separator_y": 4.81,
         "axis_bottom": 4.04, "axis_top": 5.60,
         "contexts": (
             "Sign reversal not enforced",
             "Sign reversal enforced\nfor both encoders",
             "Sign reversal not enforced",
         ),
         "limits": (-.115, .035),
         "ticks": (-.10, -.05, 0, .02), "tick_labels": ("−0.10", "−0.05", "0", "+0.02"),
         "axis_label": "Difference in R²\nPositive = better prediction"},
        {"title_y": 3.23, "title": f"B. {AGREEMENT_TITLE}", "rows": reduction,
         "subtitle": "The encoder's features for the original and mirrored clip\nare compared after matching left/right body landmarks.",
         "subtitle_y": 2.93,
         "direct_comparison_note": "Features compared directly; no prediction rule.",
         "direct_comparison_note_y": 2.67,
         "group_y": (2.43, 1.36), "reference_y": (2.24, 1.17),
         "row_y": (2.00, 1.68, .94), "separator_y": 1.50,
         "axis_bottom": .70, "axis_top": 2.33,
         "contexts": (
             "Training without mirrored clips",
             "Training with mirrored clips",
             "Both encoders trained",
         ),
         "limits": (-.065, .025),
         "ticks": (-.06, -.04, -.02, 0, .02),
         "tick_labels": ("−0.06", "−0.04", "−0.02", "0", "+0.02"),
         "axis_label": "Reduction in disagreement\nPositive = closer agreement"},
    ]
    for panel in panels:
        axis_bottom = panel["axis_bottom"]
        axis_height = panel["axis_top"] - axis_bottom
        ax = fig.add_axes((PLOT_LEFT / WIDTH, axis_bottom / HEIGHT,
                           (PLOT_RIGHT - PLOT_LEFT) / WIDTH, axis_height / HEIGHT))
        ax.set_xlim(*panel["limits"])
        ax.set_ylim(0, axis_height)
        ax.set_yticks([])
        ax.set_xticks(panel["ticks"], panel["tick_labels"])
        ax.tick_params(axis="x", labelsize=FONT, pad=4, length=3, width=.6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.axvline(0, color=INK, linewidth=1.0, zorder=1)
        ax.grid(axis="x", color=GRID, linewidth=.55)
        ax.set_axisbelow(True)
        ax.set_xlabel(panel["axis_label"], fontsize=FONT, labelpad=6)

        fig.text(.10 / WIDTH, panel["title_y"] / HEIGHT, panel["title"], fontsize=TITLE_FONT,
                 ha="left", va="center", weight="bold", color=INK)
        if "subtitle" in panel:
            label(.10, panel["subtitle_y"], panel["subtitle"], linespacing=1.12, color=MUTED)
        if "direct_comparison_note" in panel:
            label(.10, panel["direct_comparison_note_y"], panel["direct_comparison_note"], color=MUTED)
        label(.10, panel["group_y"][0], TRAINING_GROUP, weight="bold")
        label(.10, panel["group_y"][1], MIRRORING_GROUP, weight="bold", color=TEAL)
        for y, reference in zip(panel["reference_y"], (TRAINING_REFERENCE, MIRRORING_REFERENCE)):
            label(.10, y, reference, color=MUTED,
                  bbox={"facecolor": "white", "edgecolor": "none", "pad": .3})
        if "shared_note" in panel:
            label(.10, panel["shared_note_y"], panel["shared_note"])
        # A gap and fine rule distinguish comparisons against initialization
        # from the comparison between two pretrained encoders.
        separator = panel["separator_y"]
        fig.add_artist(Line2D([.10 / WIDTH, PLOT_RIGHT / WIDTH],
                              [separator / HEIGHT, separator / HEIGHT],
                              transform=fig.transFigure, color=GRID,
                              linewidth=.7, zorder=0))

        for index, (row_y, row, context) in enumerate(zip(
                panel["row_y"], panel["rows"], panel["contexts"])):
            estimate, low, high = (float(row[key]) for key in ("estimate", "ci95_low", "ci95_high"))
            if not (panel["limits"][0] < low <= estimate <= high < panel["limits"][1]):
                raise RuntimeError("A confidence interval would be clipped by the plot limits.")
            color, marker = (TEAL, "D") if index == 2 else (NEUTRAL, "o")
            y = row_y - axis_bottom
            ax.errorbar(estimate, y, xerr=[[estimate - low], [high - estimate]],
                        fmt=marker, markersize=4.8, color=color, elinewidth=1.65,
                        capsize=3.0, capthick=1.0, zorder=3)
            ax.annotate(f"{estimate:+.3f}", (estimate, y), xytext=(0, 8.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=FONT, color=color, weight="bold")
            row_labels.append(label(.10, row_y, context, linespacing=1.12))

    # The paper caption defines estimates and intervals, leaving the plotting
    # area available for full comparison labels and readable axis directions.
    layout = validate_layout(fig, row_labels)
    publication_box = Bbox.from_bounds(0, 0, WIDTH, HEIGHT)
    for extension in ("svg", "pdf"):
        metadata = ({"Creator": Path(__file__).name, "Date": None}
                    if extension == "svg" else
                    {"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None})
        fig.savefig(OUT / f"{STEM}.{extension}", bbox_inches=publication_box,
                    pad_inches=0, metadata=metadata)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{STEM}.png", dpi=180,
                    bbox_inches=publication_box, pad_inches=0)
    plt.close(fig)
    # Confirm that the editable SVG has text and contains no raster image nodes.
    svg = ET.parse(OUT / f"{STEM}.svg").getroot()
    svg_namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert svg.findall(".//svg:text", svg_namespace)
    assert not svg.findall(".//svg:image", svg_namespace)
    layout["svg_editable_text_check"] = "passed"
    layout["svg_no_raster_check"] = "passed"
    return layout


def write_provenance(predictive, mismatch, reduction, summary, shared, layout, protected_hashes):
    result = {
        "schema": "brainbody_aggregate_laterality_figure/v3",
        "generator": Path(__file__).name,
        "generator_sha256": digest(Path(__file__)),
        "read_only_input_verifier": SHARED_GENERATOR.name,
        "read_only_input_verifier_sha256": digest(SHARED_GENERATOR),
        "report_directory": str(shared.RUN.relative_to(shared.ROOT)),
        "protocol_digest": summary["protocol_digest"],
        "context_digest": summary["context_digest"],
        "cohort_digest": summary["cohort_digest"],
        "source_sha256": shared.EXPECTED_HASHES,
        "selected_folds": summary["selected_folds"],
        "selected_seeds": summary["selected_seeds"],
        "selected_variants": summary["selected_variants"],
        "producer_code_sha256": {
            relative: digest(shared.ROOT / relative) for relative in (
                "neurips-laterality/laterality/reporting.py",
                "neurips-laterality/laterality/evaluation.py",
            )
        },
        "panel_a": {"title": "Laterality prediction",
                    "source": "report/checkpoint_source_bootstrap.csv", "rows": predictive},
        "panel_b": {
            "title": AGREEMENT_TITLE,
            "source": "report/strict_representation_equivariance_source_bootstrap.csv",
            "source_rows": mismatch,
            "displayed_rows": reduction,
            "display_transformation": {
                "estimate": "negative of the source estimate",
                "ci95_low": "negative of the source upper confidence bound",
                "ci95_high": "negative of the source lower confidence bound",
                "reason": "Display reduction in disagreement, so a positive value favors the tested condition relative to its labeled reference in both panels.",
                "source_rows_immutable": True,
                "exact_round_trip_check": "passed",
                "interval_order_width_and_zero_inclusion_checks": "passed",
            },
        },
        "uncertainty": "95% pointwise source-bootstrap intervals from 2000 draws, conditional on fixed cross-fitted models; no refitting or multiplicity adjustment.",
        "point_estimates": "Per-seed metrics pooled across five source-held-out folds, then averaged over five seeds; not metrics of seed-averaged predictions.",
        "publication_geometry": {
            "exported_width_inches": WIDTH, "exported_height_inches": HEIGHT,
            "minimum_text_size_points": FONT, "title_size_points": TITLE_FONT,
            "fixed_size_without_text_scaling": True,
        },
        "style": {
            "pretraining_contrasts": "Neutral circles; effect of encoder training, referenced to the same encoder at its random starting weights.",
            "mirroring_contrasts": "Teal diamonds; effect of adding mirrored training clips, referenced to an encoder trained without mirrored clips.",
            "displayed_group_headings": [TRAINING_GROUP, MIRRORING_GROUP],
            "displayed_group_references": [TRAINING_REFERENCE, MIRRORING_REFERENCE],
            "panel_a_shared_training_condition": "Training excludes mirrored clips; applies to the first two rows only.",
            "panel_a_row_contexts": ["Sign reversal not enforced", "Sign reversal enforced for both encoders", "Sign reversal not enforced"],
            "panel_b_row_contexts": ["Training without mirrored clips", "Training with mirrored clips", "Both encoders trained"],
            "difference_definition": {
                "panel_a": "Prediction R² for the tested condition minus R² for the labeled reference; unchanged from the source report. The sign rule is held fixed within each comparison.",
                "panel_b": "Normalized disagreement for the labeled reference minus disagreement for the tested condition; a displayed error reduction with the original source values retained above. Features are compared directly without a prediction rule.",
            },
            "favorable_directions": {"panel_a": "Positive = better prediction",
                                      "panel_b": "Positive = closer agreement"},
            "displayed_estimate_decimals": 3,
        },
        "validation": layout,
        "shared_files_verified_unchanged": protected_hashes,
        "scope": "Aggregate results only; same six source contrasts as the shared figure. Panel B reverses the displayed sign to express error reduction. No new model fits or clinical claims.",
        "privacy": "No individual poses, sequence identifiers, or source identifiers are exported.",
        "assets": {f"{STEM}.{extension}": digest(OUT / f"{STEM}.{extension}")
                   for extension in ("svg", "pdf")},
    }
    (OUT / PROVENANCE_NAME).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional temporary directory for a PNG preview.")
    args = parser.parse_args()
    protected_hashes = {path.name: digest(path) for path in PROTECTED}
    predictive, mismatch, summary, shared = load_verified_inputs()
    reduction = disagreement_reduction_rows(mismatch)
    layout = render(predictive, reduction, args.preview_dir)
    layout["panel_b_sign_conversion_and_source_immutability_checks"] = "passed"
    after = {path.name: digest(path) for path in PROTECTED}
    if after != protected_hashes:
        raise RuntimeError("A shared figure file changed during this V3-only build.")
    write_provenance(predictive, mismatch, reduction, summary, shared, layout, protected_hashes)
    print(f"Wrote {STEM}.svg/.pdf and {PROVENANCE_NAME}; shared assets unchanged.")
    print(f"Validated all six original contrasts and a {WIDTH} × {HEIGHT}-inch layout with text ≥ {FONT} pt.")
    if args.preview_dir is not None:
        print(f"Preview: {args.preview_dir / (STEM + '.png')}")


if __name__ == "__main__":
    main()
