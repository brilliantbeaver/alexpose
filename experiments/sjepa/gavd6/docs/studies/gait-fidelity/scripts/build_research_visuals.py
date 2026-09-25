"""Build the revised proposal's vector method and completed-core figures.

Run from the repository root::

    .venv/bin/python docs/studies/gait-fidelity/scripts/build_research_visuals.py --preview

The SVGs use live text and have a white background. Results are read directly
from the checked core-analysis CSV; no result values are transcribed here.
Method diagrams use a small SVG builder; results use Matplotlib through
build_results_figure.py. Text bounds are checked before writing. --preview
requires CairoSVG for the method diagrams; Matplotlib also saves result PNGs.
On macOS, CairoSVG may need DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib.
"""

from argparse import ArgumentParser
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
INK, MUTED, LINE = "#183247", "#536575", "#dbe4e9"
TEAL, BLUE, GRAY = "#087e78", "#265f9e", "#687681"
TEAL_LIGHT, BLUE_LIGHT, NEUTRAL = "#edf8f5", "#eff5fc", "#f6f8fa"


class SVG:
    """Small SVG builder with bounds checks for all live text."""

    def __init__(self, width, height, title, description):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{escape(title)}</title>',
            f'<desc id="desc">{escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" '
            'orient="auto" markerUnits="userSpaceOnUse">'
            f'<path d="M0,0 L9,4.5 L0,9 Z" fill="{MUTED}"/></marker></defs>',
            f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        ]
        self.text_items = []

    def text(self, x, y, label, size=20, color=INK, bold=False, anchor="start"):
        assert size >= 20, "Keep figure labels readable at a 7.3-inch page width."
        self.parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{700 if bold else 400}" '
            f'fill="{color}" text-anchor="{anchor}">{escape(label)}</text>'
        )
        self.text_items.append((x, y, label, size, bold, anchor))

    def rect(self, x, y, width, height, fill=NEUTRAL, stroke=LINE, radius=9, line_width=1.4):
        assert 0 <= x <= x + width <= self.width
        assert 0 <= y <= y + height <= self.height
        self.parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{line_width}"/>'
        )

    def path(self, points, arrow=False, color=MUTED, width=2, dashed=False):
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"'
            + (' stroke-dasharray="5 5"' if dashed else "")
            + (' marker-end="url(#arrow)"' if arrow else "") + '/>'
        )

    def circle(self, x, y, radius, color, fill=None):
        self.parts.append(
            f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill or color}" '
            f'stroke="{color}" stroke-width="2.3"/>'
        )

    def save(self, name):
        # Pillow checks actual glyph widths rather than character-count estimates.
        try:
            from PIL import ImageFont
        except ImportError:
            ImageFont = None
        font_root = Path("/System/Library/Fonts/Supplemental")
        if ImageFont is not None and (font_root / "Arial.ttf").exists():
            for x, y, label, size, bold, anchor in self.text_items:
                font = ImageFont.truetype(str(font_root / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)
                length = font.getlength(label)
                left = x - (length / 2 if anchor == "middle" else length if anchor == "end" else 0)
                assert left >= 0 and left + length <= self.width, (name, label, left, length)
                assert y - size >= 0 and y + size * 0.25 <= self.height, (name, label, y)
        content = "\n".join(self.parts + ["</svg>"]) + "\n"
        ET.fromstring(content)
        path = IMAGES / name
        path.write_text(content, encoding="utf-8")
        return path


def method():
    s = SVG(
        1000, 762,
        "Learn paired movement features, then test a frozen encoder",
        "Proposed follow-up design, not a result. Each original and altered walking state "
        "has estimated poses and a clean reference. A masked student encoder and predictor "
        "predict the clean-reference features of an exponential-moving-average teacher. "
        "Compare feature-difference matching with independent-state regression while keeping "
        "the original JEPA objective. Freeze each resulting encoder and fit a coordinate-only "
        "readout and a separate change-supervised readout. Both receive the same frozen "
        "checkpoint. Evaluate movement-response error and the restored trajectories. "
        "Deployment uses a single observed sequence and no reference or paired state."
    )
    s.text(26, 36, "Can predictive features preserve a movement change?", 28, bold=True)
    s.text(26, 71, "The follow-up places change supervision before the encoder is frozen.", 21, MUTED)

    s.rect(20, 91, 960, 252, "#ffffff")
    s.text(38, 123, "1  Pretrain on both states: original and altered walking", 23, bold=True)
    # Teacher targets come from clean reference poses; only student input is hidden.
    boxes = [
        (40, 147, 228, 75, BLUE_LIGHT, "Estimated poses", "Some tokens hidden"),
        (350, 147, 276, 75, BLUE_LIGHT, "Student encoder", "+ predictor"),
        (708, 147, 252, 75, BLUE_LIGHT, "Predicted features", "At masked positions"),
        (40, 249, 228, 75, TEAL_LIGHT, "Reference poses", "Training only"),
        (350, 249, 276, 75, TEAL_LIGHT, "Teacher encoder", "Moving-average weights"),
        (708, 249, 252, 75, TEAL_LIGHT, "Reference features", "No target gradients"),
    ]
    for x, y, w, h, fill, title, subtitle in boxes:
        s.rect(x, y, w, h, fill)
        s.text(x + w / 2, y + 30, title, 22, bold=True, anchor="middle")
        s.text(x + w / 2, y + 58, subtitle, 20, MUTED, anchor="middle")
    for y in (184.5, 286.5):
        s.path([(270, y), (340, y)], arrow=True)
        s.path([(628, y), (698, y)], arrow=True)

    s.rect(20, 363, 960, 163, "#ffffff")
    s.text(38, 395, "2  Compare added losses at matched joint–time positions", 23, bold=True)
    s.rect(40, 411, 445, 73, TEAL_LIGHT, TEAL)
    s.text(60, 440, "Feature-difference matching", 22, TEAL, True)
    s.text(60, 469, "Predicted change ≈ reference change", 20)
    s.rect(515, 411, 445, 73, BLUE_LIGHT, BLUE)
    s.text(535, 440, "Independent-state regression", 22, BLUE, True)
    s.text(535, 469, "Predict each reference state separately", 20)
    s.text(500, 510, "Both keep the original JEPA objective and feature regularization.", 20, MUTED, anchor="middle")

    s.rect(20, 546, 960, 156, "#ffffff")
    s.text(38, 578, "3  Test what each frozen encoder supports", 23, bold=True)
    s.rect(40, 598, 221, 85, NEUTRAL)
    s.text(150.5, 630, "Frozen encoder", 23, bold=True, anchor="middle")
    s.text(150.5, 659, "Predictor discarded", 20, MUTED, anchor="middle")
    s.path([(263, 640.5), (302, 640.5)])
    s.path([(302, 618), (302, 662)])
    s.path([(302, 618), (334, 618)], arrow=True)
    s.path([(302, 662), (334, 662)], arrow=True)
    s.rect(344, 598, 285, 39, BLUE_LIGHT)
    s.rect(344, 644, 285, 39, TEAL_LIGHT)
    s.text(486.5, 625, "Coordinate-only readout", 21, BLUE, True, "middle")
    s.text(486.5, 671, "Change-supervised readout", 21, TEAL, True, "middle")
    s.path([(631, 618), (662, 618), (662, 641)])
    s.path([(631, 664), (662, 664), (662, 641)])
    s.path([(662, 641), (698, 641)], arrow=True)
    s.rect(708, 598, 252, 85, NEUTRAL)
    s.text(834, 628, "Evaluate each readout", 22, bold=True, anchor="middle")
    s.text(834, 657, "Response + trajectories", 20, MUTED, anchor="middle")
    s.text(500, 740, "Deployment: one observed sequence; no partner or clean reference is required.", 21, MUTED, anchor="middle")
    return s.save("research-method.svg")


def method_compact():
    s = SVG(
        1000, 225,
        "Paired pretraining and a controlled frozen-encoder comparison",
        "Proposed follow-up. Matched original and altered walking sequences provide noisy "
        "observations and clean references. Three pretraining objectives produce encoders: "
        "feature-difference JEPA, independent-state JEPA, and a coordinate-difference control. "
        "Each encoder is frozen, then evaluated with separately trained coordinate-only "
        "and change-supervised readouts. Three variants at three seeds with two readouts "
        "give eighteen final models. Assess movement response, trajectory accuracy and failures."
    )
    s.text(18, 28, "Learn from paired movement states; test both downstream objectives", 24, bold=True)
    stages = [
        (18, 210, "Paired walking", ["Original and altered", "Noisy observations", "+ clean references"], BLUE_LIGHT),
        (258, 243, "JEPA and control", ["Feature difference", "vs independent states", "Coordinate control"], TEAL_LIGHT),
        (531, 217, "Freeze encoder", ["Train two readouts:", "coordinates only", "or + change loss"], BLUE_LIGHT),
        (778, 204, "Evaluate", ["Response error", "Trajectory error", "Failures + side errors"], NEUTRAL),
    ]
    for index, (x, width, title, lines, fill) in enumerate(stages):
        s.rect(x, 48, width, 132, fill)
        s.text(x + width / 2, 76, title, 20, bold=True, anchor="middle")
        for i, label in enumerate(lines):
            s.text(x + width / 2, 107 + i * 27, label, 20, MUTED, anchor="middle")
        if index < 3:
            s.path([(x + width + 3, 114), (stages[index + 1][0] - 8, 114)], arrow=True)
    s.text(500, 211, "3 pretraining variants × 3 seeds × 2 readouts = 18 final models", 21, MUTED, anchor="middle")
    return s.save("research-method-compact.svg")


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true", help="Also render PNG previews with CairoSVG.")
    args = parser.parse_args()
    from build_results_figure import build_results
    paths = [method(), method_compact(), *build_results()]
    for path in paths:
        print(path)
    if args.preview:
        import cairosvg
        previews = IMAGES / "previews"
        previews.mkdir(exist_ok=True)
        for path in paths:
            if path.stem.startswith("research-core-results"):
                continue  # Preserve the direct Matplotlib previews.
            target = previews / (path.stem + ".png")
            cairosvg.svg2png(url=str(path), write_to=str(target), output_width=1600)
            print(target)


if __name__ == "__main__":
    main()
