"""Render the submission weighting figure from surviving numerical summaries.

This script reads two small, independently stored copies of the same source-level
table and requires them to agree. It does not load an encoder, open checkpoints,
run inference, or modify existing figures.
"""

from __future__ import annotations

import csv
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


DOCS = Path(__file__).resolve().parents[1]
INPUTS = (
    DOCS / "numerical_supplement/normal_validation_weighting.csv",
    DOCS / "evidence/validation_normal_source_weighting.csv",
)
OUTPUT = DOCS / "figures"
STEM = "weighting_comparison"
SIZE_INCHES = (5.5, 2.75)


def read_verified_values() -> dict:
    """Recompute both averaging rules from five recorded source summaries."""
    copies = []
    for path in INPUTS:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            assert reader.fieldnames == ["source", "sequences", "mean_cosine"]
            copies.append(list(reader))
    assert copies[0] == copies[1], "The two stored numerical tables disagree."
    rows = copies[0]
    assert [row["source"] for row in rows] == list("ABCDE")
    counts = [int(row["sequences"]) for row in rows]
    means = [Decimal(row["mean_cosine"]) for row in rows]
    assert counts == [60, 1, 1, 1, 1]
    assert all(Decimal(-1) <= value <= Decimal(1) for value in means)
    total = sum(counts)
    equal_clip = sum(n * value for n, value in zip(counts, means)) / total
    equal_video = sum(means) / len(means)
    assert abs(equal_clip - Decimal("0.88906143")) < Decimal("0.00000001")
    assert abs(equal_video - Decimal("0.70105767")) < Decimal("0.00000001")
    return {
        "rows": rows,
        "total_clips": total,
        "total_videos": len(rows),
        "equal_clip": equal_clip,
        "equal_video": equal_video,
        "weights": ((Decimal(60) / 64, Decimal(4) / 64),
                    (Decimal(1) / 5, Decimal(4) / 5)),
    }


def render(values: dict) -> dict:
    """Use physical figure coordinates so publication text is never shrunk."""
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "genai4health-weighting-comparison",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    figure = plt.figure(figsize=SIZE_INCHES, dpi=200, facecolor="white")
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set(xlim=(0, SIZE_INCHES[0]), ylim=(0, SIZE_INCHES[1]))
    axes.set_axis_off()

    ink = "#202E3B"
    muted = "#4C5C69"
    dominant = "#CF7F34"
    others = "#CCD5DC"
    texts = []

    def label(x: float, y: float, text: str, *, size: float = 10,
              color: str = ink, weight: str = "normal", ha: str = "left"):
        result = axes.text(x, y, text, fontsize=size, color=color,
                           fontweight=weight, ha=ha, va="center", linespacing=1.25)
        texts.append(result)
        return result

    label(0.16, 2.53, "Same clips, different averages", size=13, weight="bold")

    # Legend names both contributions directly; color is a redundant cue.
    for x, color, text in (
        (0.16, dominant, "One video · 60 clips"),
        (2.78, others, "Other four · 1 clip each"),
    ):
        axes.add_patch(Rectangle((x, 2.13), 0.105, 0.105,
                                 facecolor=color, edgecolor="none"))
        label(x + 0.16, 2.184, text)

    bar_left, bar_width, bar_height = 1.72, 2.50, 0.235
    label(bar_left + bar_width / 2, 1.86, "Contribution to average", ha="center",
          color=muted)
    label(4.91, 1.86, "Average\ncosine", ha="center", color=muted)
    axes.plot([4.47, 4.47], [0.50, 1.66], color="#DDE3E8", linewidth=0.7)

    for y, title, weights, value in (
        (1.35, "Equal clip\nweight", values["weights"][0], values["equal_clip"]),
        (0.72, "Equal video\nweight", values["weights"][1], values["equal_video"]),
    ):
        label(0.16, y, title, size=10.5)
        left = bar_left
        for weight, color in zip(weights, (dominant, others)):
            width = float(weight) * bar_width
            axes.add_patch(Rectangle((left, y - bar_height / 2), width, bar_height,
                                     facecolor=color, edgecolor="none"))
            # All percentages sit above the bars, including the narrow 6% segment.
            # Segment geometry uses exact weights; labels use whole percentages.
            label(left + width / 2, y + 0.25, f"{float(weight):.0%}", ha="center")
            left += width
        label(4.91, y, f"{value:.2f}", size=17, ha="center", weight="bold")

    label(0.16, 0.20, "Cosine measures feature similarity between two encoder states.",
          color=muted)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    canvas = figure.bbox
    extents = [text.get_window_extent(renderer).expanded(1.015, 1.02) for text in texts]
    for text, extent in zip(texts, extents):
        assert text.get_fontsize() >= 10
        assert canvas.contains(extent.x0, extent.y0) and canvas.contains(extent.x1, extent.y1), (
            "Text falls outside the canvas", text.get_text(), tuple(extent.extents)
        )
    for index, first in enumerate(extents):
        for other_index in range(index + 1, len(extents)):
            assert not first.overlaps(extents[other_index]), (
                "Text labels overlap", texts[index].get_text(), texts[other_index].get_text()
            )

    OUTPUT.mkdir(exist_ok=True)
    figure.savefig(OUTPUT / f"{STEM}.svg", metadata={
        "Date": None, "Creator": "Verified numerical-summary renderer",
        "Title": "Same clips, different averages",
        "Description": "The same 64 validation clips from five videos receive equal clip or equal video weight.",
    })
    figure.savefig(OUTPUT / f"{STEM}.pdf", metadata={
        "CreationDate": None, "ModDate": None,
        "Creator": "Verified numerical-summary renderer",
        "Title": "Same clips, different averages",
    })
    figure.savefig(OUTPUT / f"{STEM}.png", dpi=240,
                   metadata={"Software": "Verified numerical-summary renderer"})
    visible_text = [text.get_text() for text in texts]
    plt.close(figure)

    svg = ET.parse(OUTPUT / f"{STEM}.svg").getroot()
    assert not svg.findall(".//{http://www.w3.org/2000/svg}image"), "Unexpected raster in SVG."
    assert len(svg.findall(".//{http://www.w3.org/2000/svg}text")) >= len(texts)
    return {
        "size_inches": list(SIZE_INCHES),
        "minimum_font_points": min(text.get_fontsize() for text in texts),
        "visible_word_count": len(re.findall(r"\S+", " ".join(visible_text))),
        "visible_text": visible_text,
        "vector_svg": True,
        "labels_in_canvas": True,
        "no_text_overlap": True,
    }


def main() -> None:
    values = read_verified_values()
    design = render(values)
    output_files = [OUTPUT / f"{STEM}.{extension}" for extension in ("svg", "pdf", "png")]
    provenance = {
        "purpose": "Visualize how two averaging rules weight the same observed clips.",
        "scope": "Arithmetic from stored source-level summaries only; no model or checkpoint execution.",
        "inputs": {
            str(path.relative_to(DOCS)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in INPUTS
        },
        "source_summaries": values["rows"],
        "total_clips": values["total_clips"],
        "total_videos": values["total_videos"],
        "equal_clip_average": str(values["equal_clip"]),
        "equal_video_average": str(values["equal_video"]),
        "equal_clip_formula": "sum(clips_in_video * video_mean_cosine) / total_clips",
        "equal_video_formula": "sum(video_mean_cosine) / total_videos",
        "display_rounding": "Cosines: two decimal places. Weights: whole percentages; bar lengths use exact weights.",
        "dominant_video_weight_equal_clip": str(values["weights"][0][0]),
        "dominant_video_weight_equal_video": str(values["weights"][1][0]),
        "interpretation_boundary": "Cosine is feature similarity; this figure does not measure retained clinical function or clinical prediction quality.",
        "design_checks": design,
        "outputs": {
            str(path.relative_to(DOCS)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output_files
        },
    }
    (OUTPUT / f"{STEM}_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "equal_clip_average": provenance["equal_clip_average"],
        "equal_video_average": provenance["equal_video_average"],
        "design_checks": design,
        "outputs": [str(path.relative_to(DOCS)) for path in output_files],
    }, indent=2))


if __name__ == "__main__":
    main()
