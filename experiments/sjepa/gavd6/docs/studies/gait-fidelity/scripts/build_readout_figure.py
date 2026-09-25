"""Build only the amended proposal's readout-design SVG.

    python docs/studies/gait-fidelity/scripts/build_readout_figure.py

The SVG builder uses the standard library. Add --preview to render a 1200-pixel
PNG with CairoSVG; on macOS set DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib.
"""
from argparse import ArgumentParser
from html import escape
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 1200, 744
INK, MUTED, LINE = "#183247", "#526675", "#d8e3e8"
TEAL, BLUE, AMBER = "#087e78", "#265f9e", "#805b25"


def build():
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" '
        'role="img" aria-labelledby="title desc">',
        '<title id="title">Test each frozen encoder with two readout objectives</title>',
        '<desc id="desc">Three pretraining variants are each trained at three seeds, '
        'giving nine encoders. Every frozen encoder checkpoint feeds two separately '
        'trained readouts: coordinate-only and change-supervised. The primary '
        'comparison is feature-difference versus independent-state JEPA with '
        'change-supervised readouts. The coordinate-only comparison and interaction '
        'with readout objective are secondary analyses. Nine pretraining fits plus '
        'eighteen readout fits yield eighteen final models through twenty-seven '
        'optimization phases. This diagram describes the design, not a result.</desc>',
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" '
        'refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
        f'<path d="M0,0 L8,4 L0,8 Z" fill="{MUTED}"/></marker></defs>',
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
    ]

    def text(x, y, label, size=22, color=INK, weight="normal", anchor="start"):
        parts.append(
            f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}">{escape(label)}</text>'
        )

    def rect(x, y, width, height, fill, stroke=LINE, stroke_width=1.5):
        parts.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="10" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
        )

    def path(points, arrow=True, color=MUTED, width=2):
        coords = " ".join(f"{x},{y}" for x, y in points)
        parts.append(
            f'<polyline points="{coords}" fill="none" stroke="{color}" '
            f'stroke-width="{width}" stroke-linejoin="round" '
            f'stroke-linecap="round"'
            + (' marker-end="url(#arrow)"' if arrow else "") + '/>'
        )

    text(32, 49, "Test each encoder with both readout objectives", 32, weight="bold")
    text(32, 85, "3 pretraining variants × 3 seeds = 9 encoders; every checkpoint is tested twice.", 23, MUTED)
    text(32, 135, "PRETRAINING VARIANT", 19, MUTED, "bold")
    text(550, 135, "FREEZE THE ENCODER", 19, MUTED, "bold", "middle")
    text(974, 135, "TRAIN TWO READOUTS", 19, MUTED, "bold", "middle")

    variants = [
        (164, "Feature-difference JEPA", "Coupled feature residuals", TEAL, "#edf8f5"),
        (296, "Independent-state JEPA", "Separate feature residuals", BLUE, "#eff5fc"),
        (428, "Coordinate-difference model", "Common-scale coordinate errors", AMBER, "#fcf7ec"),
    ]
    for index, (y, title, subtitle, accent, fill) in enumerate(variants):
        rect(32, y, 350, 108, fill)
        parts.append(f'<rect x="32" y="{y + 15}" width="4" height="78" rx="2" fill="{accent}"/>')
        text(49, y + 42, title, 23, accent, "bold")
        text(49, y + 77, subtitle, 21, MUTED)

        path([(384, y + 54), (420, y + 54)])
        rect(424, y, 252, 108, "#f7f9fb")
        text(550, y + 43, "Frozen encoder", 25, weight="bold", anchor="middle")
        text(550, y + 78, "One checkpoint per seed", 20, MUTED, anchor="middle")

        # Branch once from the saved checkpoint; neither readout trains another encoder.
        path([(678, y + 54), (730, y + 54)], arrow=False)
        path([(730, y + 22.5), (730, y + 85.5)], arrow=False)
        path([(730, y + 22.5), (784, y + 22.5)])
        path([(730, y + 85.5), (784, y + 85.5)])
        parts.append(f'<circle cx="730" cy="{y + 54}" r="3.5" fill="{MUTED}"/>')

        rect(788, y, 380, 45, "#f5f8fa")
        text(978, y + 30, "Coordinate-only", 24, weight="bold", anchor="middle")
        primary = index < 2
        rect(788, y + 63, 380, 45, "#edf8f5" if primary else "#f5f8fa",
             TEAL if primary else LINE, 2.5 if primary else 1.5)
        text(978, y + 93, "Change-supervised", 24,
             TEAL if primary else INK, "bold", "middle")

    rect(32, 573, 668, 93, "#edf8f5", TEAL, 2)
    text(49, 601, "Primary comparison", 23, TEAL, "bold")
    text(49, 629, "Feature-difference vs independent-state JEPA", 23)
    text(49, 654, "Compare their change-supervised readouts.", 21, MUTED)
    rect(724, 573, 444, 93, "#f5f8fa")
    text(743, 601, "Secondary analyses", 23, weight="bold")
    text(743, 629, "Coordinate-only comparison", 22)
    text(743, 654, "and pretraining × readout interaction", 21, MUTED)

    path([(32, 688), (1168, 688)], arrow=False, color=LINE, width=1)
    text(600, 720, "9 pretraining fits + 18 readout fits = 27 optimization phases · 18 final models",
         23, INK, "bold", "middle")
    parts.append("</svg>")
    svg = "\n".join(parts) + "\n"
    ET.fromstring(svg)
    output = ROOT / "images" / "proposal-readout-design.svg"
    output.write_text(svg, encoding="utf-8")
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    output = build()
    print(output)
    if args.preview:
        import cairosvg

        preview = output.parent / "previews" / "proposal-readout-design.png"
        preview.parent.mkdir(parents=True, exist_ok=True)
        cairosvg.svg2png(url=str(output), write_to=str(preview), output_width=1200)
        print(preview)


if __name__ == "__main__":
    main()
