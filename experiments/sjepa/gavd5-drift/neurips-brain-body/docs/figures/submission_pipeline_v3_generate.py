#!/usr/bin/env python3
"""Draw the current laterality method, without executing any experiment.

Run from the repository root with the existing environment, optionally adding
``--preview-dir`` for a temporary, near-print-scale PNG. Only the new pipeline
SVG, PDF, and internal provenance file are written beside this script. The
shared laterality reader verifies the frozen protocol and reports read-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon
from matplotlib.path import Path as DrawingPath
from matplotlib.text import Text
from matplotlib.transforms import Bbox


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
STEM = "submission_pipeline_v3"
SHARED_READER = OUT / "submission_laterality_generate.py"
PROTECTED = (
    SHARED_READER,
    OUT / "submission_laterality_v3_generate.py",
    OUT / "submission_laterality_effects.svg",
    OUT / "submission_laterality_effects.pdf",
    OUT / "submission_laterality_effects_v3.svg",
    OUT / "submission_laterality_effects_v3.pdf",
)
WIDTH, HEIGHT = 5.5, 4.8
FONT, TITLE_FONT = 10.2, 10.2
INK = "#19374a"
MUTED = "#526575"
EDGE = "#9eafba"
TEAL = "#087b70"
TARGET = "#865625"
PALE = "#f0f5f8"
TEACHER_PALE = "#eef7f4"
TARGET_PALE = "#fbf5ed"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verified_method():
    """Check the displayed configuration against the completed paper run.

    Source checks below make the easily confused relationships explicit. They
    do not claim to rerun the models or to replace the manuscript's code audit.
    """
    spec = importlib.util.spec_from_file_location("pipeline_read_only_reader", SHARED_READER)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the shared read-only protocol verifier.")
    shared = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shared)
    _, _, summary = shared.read_inputs()
    snapshot_path = shared.RUN / "protocol_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    protocol = snapshot["protocol"]
    data, model = protocol["data"], protocol["model"]
    assert (data["frames"], data["joints"], data["segment_length"]) == (64, 33, 4)
    assert (model["encoder_depth"], model["embed_dim"], model["predictor_depth"]) == (4, 96, 2)
    assert len(model["authorized_target_joints"]) == 12
    assert model["mask_fraction"] == .6
    assert data["max_interpolation_gap"] == 4
    assert data["extraction_provenance"]["pose_model"] == "pose_landmarker_lite.task"

    paths = {name: ROOT / "neurips-laterality/laterality" / f"{name}.py"
             for name in ("model", "training", "evaluation", "geometry")}
    sources = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    required = {
        "model": (
            "tokens = tokens.masked_fill(hide_mask[..., None], 0.0)",
            "tokens = tokens + self.time_pos[None, :, None, :] + self.joint_pos[None, None, :, :]",
            "target_tokens = self.target_encoder(target, valid_patch)",
            "target.mul_(momentum).add_(view, alpha=1.0 - momentum)",
        ),
        "training": (
            "if set(drawn_sources) & test_sources:",
            "view_a, coordinates, patch_valid, target_mask",
            "tokens_a = model.view_encoder(view_a, patch_valid)",
            "tokens_b = model.view_encoder(view_b, patch_valid)",
            "regularizer = vicreg_loss(projector(pooled_a), projector(pooled_b))",
        ),
        "evaluation": (
            'floor.target_encoder.load_state_dict(checkpoint["initial_target_state"])',
            "learned.target_encoder,",
            "floor.target_encoder,",
        ),
        "geometry": (
            "target_xyz, target_valid, target_scale = pelvis_normalize(",
            "filled_xyz, filled_valid = interpolate_short_gaps(",
            "model_xyz = temporal_resize(",
        ),
    }
    for name, fragments in required.items():
        for fragment in fragments:
            if fragment not in sources[name]:
                raise RuntimeError(f"Method code changed; re-audit the schematic: {name}: {fragment}")
    training = sources["training"]
    assert training.index("coordinates, valid = anatomical_reflect_tensor(") < training.index(
        "view_a = geometric_view(") < training.index("predicted, targets = model(")
    geometry = sources["geometry"]
    assert geometry.index("target_xyz, target_valid, target_scale = pelvis_normalize(") < geometry.index(
        "filled_xyz, filled_valid = interpolate_short_gaps(")
    return {
        "snapshot": str(snapshot_path.relative_to(ROOT)),
        "snapshot_sha256": digest(snapshot_path),
        "protocol_digest": summary["protocol_digest"],
        "source_sha256": {str(path.relative_to(ROOT)): digest(path) for path in paths.values()},
        "configuration": {
            "frames": 64, "landmarks": 33, "frames_per_patch": 4,
            "mask_eligible_landmarks": 12, "eligible_mask_fraction": .6,
            "encoder_layers": 4, "feature_dimension": 96, "predictor_layers": 2,
        },
        "scope": "Adapted laterality model only; not the exploratory classifier or an exact reproduction of the original S-JEPA architecture.",
        "checks": "Frozen protocol verified; source markers for masking, target update, view construction, target preparation, and evaluation encoder passed.",
    }


def validate_layout(fig, box_texts, connectors):
    """Require publication-size text, no clipping, and no text/line collisions."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = Bbox.from_bounds(0, 0, WIDTH * fig.dpi, HEIGHT * fig.dpi)
    texts = [artist for artist in fig.findobj(Text)
             if artist.get_visible() and artist.get_text().strip()]
    boxes = []
    for artist in texts:
        box = artist.get_window_extent(renderer)
        if (box.x0 < 0 or box.y0 < 0 or box.x1 > canvas.x1 or box.y1 > canvas.y1):
            raise RuntimeError(f"Text exceeds publication bounds: {artist.get_text()!r}")
        if artist.get_fontsize() < FONT:
            raise RuntimeError(f"Text below {FONT} pt: {artist.get_text()!r}")
        boxes.append((artist, box))
    for index, (left, left_box) in enumerate(boxes):
        for right, right_box in boxes[index + 1:]:
            overlap = Bbox.intersection(left_box, right_box)
            if overlap is not None and overlap.width > .5 and overlap.height > .5:
                raise RuntimeError(f"Text collision: {left.get_text()!r} / {right.get_text()!r}")
    for artist, bounds in box_texts:
        text_box = artist.get_window_extent(renderer)
        x, y, width, height = bounds
        padding = .032 * fig.dpi
        expected = Bbox.from_bounds(x * fig.dpi + padding, y * fig.dpi + padding,
                                   width * fig.dpi - 2 * padding,
                                   height * fig.dpi - 2 * padding)
        if (text_box.x0 < expected.x0 or text_box.y0 < expected.y0
                or text_box.x1 > expected.x1 or text_box.y1 > expected.y1):
            raise RuntimeError(f"Text is too close to its box edge: {artist.get_text()!r}")
    for points in connectors:
        path = DrawingPath([(x * fig.dpi, y * fig.dpi) for x, y in points])
        for artist, text_box in boxes:
            padded = text_box.padded(1.0)
            if path.intersects_bbox(padded, filled=False):
                raise RuntimeError(f"Connector crosses text: {artist.get_text()!r}")
    word_count = sum(len(re.findall(r"\b[\w’]+\b", artist.get_text())) for artist in texts)
    if word_count > 80:
        raise RuntimeError(f"The visual overview has become text-heavy: {word_count} words.")
    return {
        "visible_text_items": len(texts), "text_bounds": "passed",
        "text_collisions": "passed", "box_padding": "passed",
        "connector_text_collisions": "passed",
        "displayed_word_count": word_count, "maximum_displayed_words": 80,
    }


def render(preview_dir: Path | None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": FONT,
        "text.color": INK, "svg.fonttype": "none", "pdf.fonttype": 42,
        "svg.hashsalt": "brainbody-current-laterality-method-v3",
        "savefig.facecolor": "white", "figure.facecolor": "white",
    })
    fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=160)
    ax = fig.add_axes((0, 0, 1, 1), xlim=(0, WIDTH), ylim=(0, HEIGHT))
    ax.set_axis_off()
    box_texts, connectors = [], []

    def text(x, y, value, **kwargs):
        return ax.text(x, y, value, ha=kwargs.pop("ha", "left"), va="center",
                       fontsize=kwargs.pop("fontsize", FONT), linespacing=1.10,
                       zorder=4, **kwargs)

    def arrow(points, *, color=MUTED, dashed=False, head=True, both=False):
        connectors.append(points)
        path = DrawingPath(points, [DrawingPath.MOVETO] + [DrawingPath.LINETO] * (len(points) - 1))
        style = "<->" if both else ("-|>" if head else "-")
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle=style,
                                    mutation_scale=8.0, linewidth=1.0, color=color,
                                    linestyle=(0, (3, 2)) if dashed else "solid",
                                    capstyle="round", joinstyle="round", zorder=2))

    def round_rect(x, y, width, height, *, color=INK, fill=PALE, linewidth=.9):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
                                   boxstyle="round,pad=0,rounding_size=0.025",
                                   edgecolor=color, facecolor=fill,
                                   linewidth=linewidth, zorder=3))

    def stage(number, y, label):
        ax.add_patch(Circle((.35, y), .145, facecolor=INK, edgecolor="none", zorder=3))
        text(.35, y, str(number), ha="center", color="white", weight="bold")
        text(.35, y - .39, label, ha="center", weight="bold")

    def video(x, y):
        round_rect(x - .25, y - .18, .50, .36)
        ax.add_patch(Polygon([(x - .045, y - .085), (x - .045, y + .085), (x + .09, y)],
                             facecolor=INK, edgecolor="none", zorder=4))

    def pose(x, y):
        # Generic landmark glyph: hand-drawn coordinates, never a person's pose.
        points = [(0, .25), (0, .14), (-.13, .10), (.13, .10), (-.18, -.025),
                  (.19, -.02), (-.05, -.09), (.06, -.09), (-.12, -.24), (.14, -.23)]
        for a, b in ((0, 1), (1, 2), (1, 3), (2, 4), (3, 5), (1, 6), (1, 7), (6, 8), (7, 9)):
            ax.plot([x + points[a][0], x + points[b][0]],
                    [y + points[a][1], y + points[b][1]], color=INK, linewidth=1.1, zorder=3)
        for dx, dy in points:
            ax.add_patch(Circle((x + dx, y + dy), .024, facecolor="white", edgecolor=INK, linewidth=.8, zorder=4))

    def patches(x, y, masked=False):
        # Empty cells retain the same outlines and positions. Cell counts are
        # illustrative, not the actual sequence dimensions or masking rate.
        hidden = {(0, 1), (1, 0), (1, 2), (2, 3)} if masked else set()
        for row in range(3):
            for col in range(4):
                fill = "white" if (row, col) in hidden else "#c3d5e1"
                round_rect(x - .27 + col * .14, y - .165 + row * .12,
                           .115, .095, fill=fill, color=EDGE, linewidth=.65)

    def encoder(x, y, *, color=INK, layers=4, locked=False):
        fill = TEACHER_PALE if color == TEAL else PALE
        for index in reversed(range(layers)):
            round_rect(x - .23 + .035 * index, y - .17 + .035 * index,
                       .38, .30, color=color, fill=fill)
        ax.plot([x - .13, x + .045], [y - .04, y - .04], color=color, linewidth=.9, zorder=4)
        ax.plot([x - .13, x + .045], [y + .04, y + .04], color=color, linewidth=.9, zorder=4)
        if locked:
            # A lock, rather than a text footnote, marks the frozen weights.
            ax.add_patch(Circle((x + .17, y - .15), .10, facecolor="white", edgecolor="none", zorder=5))
            ax.add_patch(Circle((x + .17, y - .125), .045, fill=False, edgecolor=color, linewidth=.9, zorder=6))
            ax.add_patch(FancyBboxPatch((x + .105, y - .22), .13, .09,
                                       boxstyle="round,pad=0,rounding_size=0.015",
                                       edgecolor=color, facecolor="white", linewidth=.9, zorder=7))

    def features(x, y, *, color=INK):
        # Equal-size feature cells, with no axes or magnitude encoding: these
        # are feature-vector symbols, never an empirical bar chart.
        for index in range(4):
            round_rect(x - .20 + index * .105, y - .15, .075, .30,
                       color=color, fill=TEACHER_PALE if color == TEAL else PALE, linewidth=.8)

    def regression(x, y):
        # Two tiny operator glyphs indicate independent fitted readouts.
        round_rect(x - .22, y - .15, .19, .30, color=TEAL, fill=TEACHER_PALE)
        round_rect(x + .055, y - .15, .19, .30, color=MUTED)
        for shift, color in ((-.125, TEAL), (.15, MUTED)):
            ax.plot([x + shift - .045, x + shift + .045], [y - .04, y + .04], color=color, linewidth=1.0, zorder=4)

    def reflection(x, y):
        # These columns are original/mirrored features within each encoder,
        # not the teal/gray trained-versus-starting-weight model comparison.
        for shift in (-.15, .15):
            for row in range(3):
                round_rect(x + shift - .055, y - .13 + .10 * row, .11, .065,
                           color=MUTED, fill=PALE, linewidth=.7)
        ax.plot([x, x], [y - .19, y + .19], color=EDGE, linewidth=.8, linestyle=(0, (2, 2)), zorder=3)

    # Sparse, numbered bands establish the overview; implementation details
    # belong in Appendix C rather than in the diagram's node labels.
    for y in (3.59, 1.58):
        ax.plot([.10, 5.22], [y, y], color="#e3eaee", linewidth=.7, zorder=0)
    stage(1, 4.47, "Prepare\npose")
    stage(2, 3.28, "Train\nencoder")
    stage(3, 1.34, "Freeze\nand test")

    # 1. Observed pose branches before encoder preparation. Only the gold
    # laterality value bypasses pretraining and reaches the regression tests.
    video(1.20, 4.28)
    text(1.20, 3.96, "GAVD video", ha="center")
    pose(2.65, 4.28)
    text(2.65, 3.90, "Estimated pose", ha="center")
    text(1.88, 4.56, "Fixed estimator", ha="center", color=MUTED)
    arrow([(1.49, 4.28), (2.35, 4.28)])
    patches(3.87, 4.28)
    text(3.87, 3.96, "Prepared pose", ha="center")
    arrow([(2.94, 4.28), (3.55, 4.28)])
    ax.add_patch(Circle((5.00, 4.28), .145, facecolor=TARGET_PALE, edgecolor=TARGET, linewidth=.9, zorder=3))
    text(5.00, 4.28, "y", ha="center", color=TARGET, style="italic", fontsize=13)
    text(5.00, 3.96, "Laterality\nvalue", ha="center", color=TARGET)
    arrow([(2.65, 4.57), (2.65, 4.68), (5.00, 4.68), (5.00, 4.45)], color=TARGET)
    arrow([(4.18, 4.28), (4.43, 4.28), (4.43, 3.69), (1.30, 3.69), (1.30, 3.34)])
    # The unmasked target branch splits before masking, not after it.
    arrow([(1.30, 3.69), (.74, 3.69), (.74, 2.10), (.98, 2.10)])

    # 2. Masked student prediction and the unmasked target are separate rows.
    # Reflection and the independent unmasked VICReg passes remain in prose.
    patches(1.30, 3.10, masked=True)
    text(1.30, 2.80, "Masked input", ha="center")
    encoder(2.65, 3.10)
    text(2.65, 3.48, "Encoder", ha="center")
    encoder(3.80, 3.10, layers=2)
    text(3.80, 3.48, "Predictor", ha="center")
    features(4.75, 3.10)
    arrow([(1.61, 3.10), (2.37, 3.10)])
    arrow([(2.99, 3.10), (3.51, 3.10)])
    arrow([(4.06, 3.10), (4.50, 3.10)])

    patches(1.30, 2.10)
    text(1.30, 1.79, "Unmasked input", ha="center")
    encoder(2.65, 2.10, color=TEAL)
    text(2.65, 1.79, "Target encoder", ha="center", color=TEAL)
    features(4.75, 2.10, color=TEAL)
    arrow([(1.61, 2.10), (2.37, 2.10)])
    arrow([(2.99, 2.10), (4.50, 2.10)], color=TEAL)
    text(3.18, 2.65, "Average\nweights", ha="center", color=TEAL)
    arrow([(2.65, 2.91), (2.65, 2.37)], color=TEAL, dashed=True)
    text(4.17, 2.50, "Match\nfeatures", ha="center")
    arrow([(4.75, 2.90), (4.75, 2.30)], color=TEAL, both=True)

    # Freeze the trained TARGET encoder, not the student or the predictor.
    arrow([(2.85, 2.00), (3.28, 2.00), (3.28, 1.49), (1.30, 1.49), (1.30, 1.26)], color=TEAL)

    # 3. The state comparison is evaluated twice, with independently fitted
    # regressions. It is not a merged feature ensemble. Feature agreement uses
    # original/reflected inputs directly and never consumes the laterality y.
    encoder(1.30, 1.00, color=TEAL, locked=True)
    encoder(2.42, 1.00, color=MUTED, locked=True)
    text(1.30, .64, "Trained", ha="center", color=TEAL)
    text(2.42, .64, "Starting weights", ha="center", color=MUTED)
    text(1.87, 1.00, "vs.", ha="center", color=MUTED)
    text(1.88, .30, "Each encoder separately", ha="center", color=MUTED)
    arrow([(2.77, 1.00), (3.15, 1.00), (3.15, 1.18), (3.30, 1.18)])
    arrow([(3.15, 1.00), (3.15, .48), (3.30, .48)])
    text(3.90, 1.18, "Predict\nlaterality", ha="center")
    regression(4.86, 1.18)
    text(3.90, .48, "Feature\nagreement", ha="center")
    reflection(4.86, .48)
    text(4.32, .14, "Original / mirrored", ha="center", color=MUTED)
    arrow([(5.19, 4.28), (5.38, 4.28), (5.38, 1.18), (5.14, 1.18)], color=TARGET)

    layout = validate_layout(fig, box_texts, connectors)
    publication_box = Bbox.from_bounds(0, 0, WIDTH, HEIGHT)
    for extension in ("svg", "pdf"):
        metadata = ({"Creator": Path(__file__).name, "Date": None}
                    if extension == "svg" else
                    {"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None})
        fig.savefig(OUT / f"{STEM}.{extension}", bbox_inches=publication_box,
                    pad_inches=0, metadata=metadata)
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{STEM}.png", dpi=160,
                    bbox_inches=publication_box, pad_inches=0)
    plt.close(fig)
    svg = ET.parse(OUT / f"{STEM}.svg").getroot()
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert svg.findall(".//svg:text", namespace)
    assert not svg.findall(".//svg:image", namespace)
    layout.update({"svg_editable_text": "passed", "svg_no_raster_images": "passed"})
    return layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, help="Optional temporary directory for a PNG preview.")
    args = parser.parse_args()
    before = {path.name: digest(path) for path in PROTECTED}
    method = verified_method()
    layout = render(args.preview_dir)
    after = {path.name: digest(path) for path in PROTECTED}
    if before != after:
        raise RuntimeError("A protected shared or existing Figure 1 asset changed.")
    provenance = {
        "schema": "brainbody_current_laterality_pipeline/v3",
        "generator": Path(__file__).name, "generator_sha256": digest(Path(__file__)),
        "method": method,
        "publication_geometry": {"width_inches": WIDTH, "height_inches": HEIGHT,
                                 "minimum_font_points": FONT, "title_font_points": TITLE_FONT},
        "validation": layout,
        "protected_files_unchanged": before,
        "interpretation": {
            "gold_connector": "Laterality target goes only to regression evaluation, never into the pretraining loss.",
            "slow_weight_update": "Exponential moving average of the trainable encoder; target encoder receives no gradients.",
            "feature_regularization": "Separate two-view unmasked trainable-encoder passes; not predicted hidden features or a mirrored feature pair.",
            "evaluation": "Frozen trained target encoder compared with its own saved random starting weights; readouts refitted independently using training videos.",
            "omissions": "Architecture settings, mask/context rules, reflection conditions, and separate unmasked-view regularization remain in Appendix C and its cross-references. No classifier, full autodiff wiring, or tuning graph is drawn.",
            "visual_symbols": "Hand-drawn pose, patch, encoder, and feature-vector glyphs are schematic, with no person-level pose data or empirical magnitude encoding. Hollow patch cells retain positions. Lock symbols mark frozen encoder states.",
            "evaluation_grouping": "The two displayed encoder states are evaluated separately, not combined into an ensemble. The two regression operator glyphs denote independently fitted readouts.",
            "agreement_glyph_colors": "Both feature-agreement columns use the same neutral color: they depict original and mirrored clips within each encoder, not a comparison across encoder states.",
        },
        "privacy": "No participant images, individual pose traces, or source identifiers.",
        "assets": {f"{STEM}.{extension}": digest(OUT / f"{STEM}.{extension}")
                   for extension in ("svg", "pdf")},
    }
    (OUT / f"{STEM}_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {STEM}.svg/.pdf at {WIDTH} × {HEIGHT} inches; all text ≥ {FONT} pt.")
    print("Verified frozen protocol, text/connector layout, editable vector SVG, and protected files unchanged.")
    if args.preview_dir is not None:
        print(f"Preview: {args.preview_dir / (STEM + '.png')}")


if __name__ == "__main__":
    main()
