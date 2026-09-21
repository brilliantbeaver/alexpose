"""Create editable SVG teaching diagrams for notebook 03, with optional PNG QA.

Run: .venv/bin/python scripts/scripts_make_notebook03_diagrams.py
Add --preview-dir /tmp/notebook03-previews to render PNGs for visual inspection.
These are conceptual diagrams of the laptop configuration, not measured plots.
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parent.parent
INK = "#19334B"
MUTED = "#53677A"
LINE = "#D6E1E9"
BG = "#F6F8FB"
BLUE = "#2F5CC9"
BLUE_BG = "#ECF2FF"
TEAL = "#08777B"
TEAL_BG = "#E9F6F3"
AMBER = "#A75219"
AMBER_BG = "#FFF4E7"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "none",
    "svg.hashsalt": "notebook03-tutorial",
})


class Diagram:
    def __init__(self, height, title, description):
        self.height = height
        self.title = title
        self.description = description
        self.fig = plt.figure(figsize=(12, height / 100), dpi=100, facecolor=BG)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1200), ylim=(height, 0))
        self.ax.axis("off")
        self.texts = []

    def text(self, x, y, value, size=14, color=INK, weight="normal", **kwargs):
        item = self.ax.text(x, y, value, fontsize=size, color=color,
                            fontweight=weight, va="top", **kwargs)
        self.texts.append(item)

    def lines(self, x, y, values, step=27, **kwargs):
        for i, value in enumerate(values):
            self.text(x, y + i * step, value, **kwargs)

    def card(self, x, y, width, height, fill="white", edge=LINE):
        self.ax.add_patch(FancyBboxPatch(
            (x, y), width, height, boxstyle="round,pad=0,rounding_size=16",
            linewidth=1.0, facecolor=fill, edgecolor=edge,
        ))

    def arrow(self, start, end, color=MUTED, dashed=False, bend=None):
        self.ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=14, linewidth=1.8,
            color=color, linestyle=(0, (5, 4)) if dashed else "-",
            connectionstyle=bend or "arc3", shrinkA=0, shrinkB=0,
        ))

    def header(self, eyebrow, title, subtitle):
        self.text(36, 24, eyebrow, 11, TEAL, "bold")
        self.text(36, 54, title, 24, INK, "bold")
        self.text(36, 98, subtitle, 13, MUTED)

    def save(self, name, preview_dir=None):
        # Check text stays on the canvas before saving; visual QA checks the cards.
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for item in self.texts:
            bounds = item.get_window_extent(renderer)
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > 1201 or bounds.y1 > self.height + 1:
                raise ValueError(f"Text outside diagram: {item.get_text()!r}")
        path = ROOT / "images" / f"{name}.svg"
        self.fig.savefig(path, format="svg", metadata={"Date": None, "Title": self.title,
                                                       "Description": self.description})
        content = path.read_text()
        content = content.replace('<svg ', '<svg role="img" aria-labelledby="diagram-title diagram-description" ', 1)
        start = content.index(">", content.index("<svg ")) + 1
        content = (content[:start] + f'\n<title id="diagram-title">{escape(self.title)}</title>'
                   + f'\n<desc id="diagram-description">{escape(self.description)}</desc>' + content[start:])
        # Matplotlib splits long SVG path data across indented lines with a
        # trailing space. Strip it here so generated diagrams pass git's
        # whitespace check without changing the SVG's rendered geometry.
        path.write_text("\n".join(line.rstrip() for line in content.splitlines()) + "\n")
        if preview_dir:
            preview_dir.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(preview_dir / f"{name}.png", dpi=100)
        plt.close(self.fig)
        print(path.relative_to(ROOT))


def data_to_tokens(preview_dir):
    d = Diagram(680, "One walking window becomes 264 tokens",
                "Laptop profile: 32 frames, 33 landmarks and three channels. Group four frames "
                "of one landmark into 12 numbers, project to 96 features, then add joint and time "
                "information. The grid has eight time blocks and 33 landmarks: 264 tokens.")
    d.header("NOTEBOOK 03  /  REPRESENTING MOVEMENT", "One walking window becomes 264 tokens",
             "Laptop configuration · Each token represents one landmark over four consecutive frames.")

    for x, color, fill, number, title in [
        (36, TEAL, TEAL_BG, "01", "Take a short window"),
        (432, BLUE, BLUE_BG, "02", "Group four frames"),
        (828, AMBER, AMBER_BG, "03", "Learn its features"),
    ]:
        d.card(x, 152, 336, 210, fill)
        d.text(x + 20, 170, number, 11, color, "bold")
        d.text(x + 20, 198, title, 16, INK, "bold")
    d.text(56, 241, "32 × 33 × 3", 26, TEAL, "bold")
    d.lines(56, 294, ["frames × landmarks × channels", "Channels: x, y, visibility"], size=12.5, color=MUTED)

    for i in range(4):
        x = 452 + 65 * i
        d.card(x, 242, 54, 42, "white", "#C4D2F1")
        d.text(x + 27, 253, f"f{i + 1}", 13, BLUE, "bold", ha="center")
    d.lines(452, 294, ["One landmark, four frames", "4 × 3 = 12 input numbers"], size=12.5, color=MUTED)
    d.text(848, 241, "12 → 96", 26, AMBER, "bold")
    d.lines(848, 294, ["A learned linear mapping", "+ joint and time information"], size=12.5, color=MUTED)
    d.arrow((384, 255), (420, 255))
    d.arrow((780, 255), (816, 255))

    d.card(36, 390, 1128, 233)
    d.text(58, 411, "The complete token grid", 16, INK, "bold")
    d.text(242, 447, "33 landmarks →", 12, MUTED)
    d.lines(57, 485, ["8 time", "blocks", "↓"], size=13, color=MUTED, step=24)
    gx, gy, cw, ch = 175, 481, 20, 14
    for t in range(8):
        for j in range(33):
            selected = t == 3 and j == 25
            d.ax.add_patch(Rectangle((gx + j * cw, gy + t * ch), cw - 3, ch - 3,
                                    facecolor=BLUE if selected else "#DCE6F0", edgecolor="none"))
    d.lines(872, 457, ["8 × 33 = 264 tokens", "Each has 96 features."], size=14, weight="bold", step=31)
    d.lines(872, 538, ["Blue square: one token", "at one joint–time slot."], size=12.5, color=BLUE, step=25)
    d.text(36, 642, "Teaching diagram · The grid illustrates token positions; it is not a sampled training mask.", 11, MUTED)
    d.save("notebook03_data_to_tokens", preview_dir)


def training_update(preview_dir):
    d = Diagram(926, "One training update, then repeat",
                "Draw 32 windows with equal probability per training source. The student sees "
                "a transformed, masked view and predicts hidden features. The slow teacher sees "
                "the full original window. Compare only hidden slots, update the student and "
                "predictor using the loss, and update the teacher by a slow moving average.")
    d.header("NOTEBOOK 03  /  HOW LEARNING HAPPENS", "One training update, then repeat",
             "Laptop run · 800 updates × 32 windows = 25,600 draws, including repeated windows.")

    d.card(36, 148, 1128, 92)
    d.text(58, 169, "1  Draw 32 training windows", 17, INK, "bold")
    d.text(58, 204, "Each of the 24 sources has probability 1/24 per draw. Use fresh masks and transformed views.", 13, MUTED)
    d.arrow((230, 240), (230, 279), BLUE)
    d.arrow((806, 240), (806, 279), TEAL)

    d.card(36, 282, 552, 301, BLUE_BG, "#C8D6F4")
    d.card(612, 282, 552, 301, TEAL_BG, "#BBDEDA")
    d.text(58, 302, "2A  STUDENT + PREDICTOR", 12, BLUE, "bold")
    d.text(634, 302, "2B  SLOW TEACHER", 12, TEAL, "bold")
    d.text(58, 337, "Transformed view + hidden slots", 17, INK, "bold")
    d.text(634, 337, "Complete original window", 17, INK, "bold")
    d.lines(58, 383, ["Use visible tokens as clues.", "Fill hidden slots with a placeholder.", "Use joint and time addresses to predict."],
            size=13.5, color=MUTED, step=30)
    d.lines(634, 383, ["Encode the whole training window.", "Keep the features at the hidden slots.", "Center and sharpen these targets."],
            size=13.5, color=MUTED, step=30)
    d.card(58, 495, 508, 65, "white", "#C8D6F4")
    d.card(634, 495, 508, 65, "white", "#BBDEDA")
    d.text(80, 516, "Predicted feature distributions", 14.5, BLUE, "bold")
    d.text(656, 516, "Target feature distributions", 14.5, TEAL, "bold")
    d.arrow((312, 584), (439, 628), BLUE)
    d.arrow((888, 584), (761, 628), TEAL)

    d.card(276, 630, 648, 90, AMBER_BG, "#E5CFB9")
    d.text(600, 647, "3  Compare only the hidden slots", 17, AMBER, "bold", ha="center")
    d.text(600, 683, "Cross-entropy loss measures prediction disagreement.", 13, MUTED, ha="center")

    d.arrow((435, 721), (313, 762), BLUE)
    d.card(36, 766, 520, 116, "white", "#C8D6F4")
    d.card(644, 766, 520, 116, "white", "#BBDEDA")
    d.text(58, 783, "4  Update student + predictor", 16, BLUE, "bold")
    d.lines(58, 818, ["AdamW uses gradients from the loss.", "The teacher receives no loss gradient."], size=13, color=MUTED)
    d.text(666, 783, "5  Slowly copy the student", 16, TEAL, "bold")
    d.lines(666, 810, ["At first: 99.6% old teacher", "+ 0.4% student.",
                       "The teacher changes", "more slowly over time."],
            size=13, color=MUTED, step=19)
    d.arrow((562, 841), (638, 841), TEAL, dashed=True)
    d.text(600, 813, "EMA", 11, TEAL, "bold", ha="center")
    d.text(36, 899, "Solid: data or gradient update. Dashed: copy encoder weights by EMA, with no gradient to the teacher.", 10.5, MUTED)
    d.save("notebook03_training_update", preview_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path)
    args = parser.parse_args()
    data_to_tokens(args.preview_dir)
    training_update(args.preview_dir)
