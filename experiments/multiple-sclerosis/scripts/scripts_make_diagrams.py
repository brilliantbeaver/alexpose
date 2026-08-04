"""Generate the tutorial diagrams as clean, modern SVG files.

Design system (applied to every figure):
  * One accent hue per class: normal = blue, ms = orange, pd = green; a neutral
    slate for shared/structural nodes.
  * Rounded "cards" with a soft fill, a 1.5px border in the accent colour, and a
    bold title over a muted subtitle.
  * Orthogonal (right-angle) connectors routed through clear channels so a line
    never crosses a box and two lines never overlap.
  * Generous whitespace, a single title, and short labels. No text sits on top of
    another line or box.

We emit SVG directly (not via matplotlib) so we control connector routing exactly
and guarantee nothing crosses. A tiny set of helpers keeps the code readable.

Run:  python scripts_make_diagrams.py
"""

from __future__ import annotations

import html
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
INK = "#1a2733"        # near-black text
MUTE = "#64748b"       # slate-500 muted text
HAIR = "#e2e8f0"       # hairline / soft divider
LINE = "#94a3b8"       # connector grey (slate-400)

BLUE, BLUE_BG, BLUE_LN = "#2563eb", "#eff6ff", "#bfdbfe"
ORANGE, ORANGE_BG, ORANGE_LN = "#ea7317", "#fff7ed", "#fed7aa"
GREEN, GREEN_BG, GREEN_LN = "#16a34a", "#f0fdf4", "#bbf7d0"
SLATE, SLATE_BG, SLATE_LN = "#475569", "#f8fafc", "#e2e8f0"
PURPLE, PURPLE_BG, PURPLE_LN = "#7c3aed", "#f5f3ff", "#ddd6fe"

FONT = ('font-family="Inter, -apple-system, BlinkMacSystemFont, \'Segoe UI\', '
        'Helvetica, Arial, sans-serif"')


# ---------------------------------------------------------------------------
# SVG primitives
# ---------------------------------------------------------------------------
class SVG:
    def __init__(self, w, h):
        self.w, self.h = w, h
        self.body = []

    def add(self, s):
        self.body.append(s)

    def rect(self, x, y, w, h, r=14, fill="#fff", stroke=None, sw=1.5, shadow=False):
        if shadow:
            self.add(f'<rect x="{x+1.5}" y="{y+2.5}" width="{w}" height="{h}" rx="{r}" '
                     f'fill="#0f172a" opacity="0.06"/>')
        s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"'
        if stroke:
            s += f' stroke="{stroke}" stroke-width="{sw}"'
        self.add(s + "/>")

    def text(self, x, y, s, size=14, fill=INK, weight="400", anchor="middle",
             italic=False, spacing=None):
        style = ""
        if italic:
            style += "font-style:italic;"
        if spacing:
            style += f"letter-spacing:{spacing}px;"
        st = f' style="{style}"' if style else ""
        self.add(f'<text x="{x}" y="{y}" {FONT} font-size="{size}" fill="{fill}" '
                 f'font-weight="{weight}" text-anchor="{anchor}"{st}>{html.escape(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke=LINE, w=2, dash=None, cap="round"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
                 f'stroke-width="{w}" stroke-linecap="{cap}"{d}/>')

    def path(self, d, stroke=LINE, w=2, arrow="arrow", dash=None):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        mk = f' marker-end="url(#{arrow})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="{w}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{da}{mk}/>')

    def circle(self, cx, cy, r, fill, stroke="#fff", sw=2):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="{sw}"/>')

    def dot(self, cx, cy, r, fill):
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>')

    def marker_defs(self):
        defs = ['<defs>']
        for name, col in [("arrow", LINE), ("arrowBlue", BLUE), ("arrowOrange", ORANGE),
                          ("arrowGreen", GREEN), ("arrowSlate", SLATE),
                          ("arrowMute", MUTE), ("arrowPurple", PURPLE)]:
            defs.append(
                f'<marker id="{name}" markerWidth="9" markerHeight="9" refX="6.5" '
                f'refY="4" orient="auto"><path d="M0.5,0.5 L8,4 L0.5,7.5 z" '
                f'fill="{col}"/></marker>')
        defs.append('</defs>')
        return "\n".join(defs)

    def render(self):
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" '
                f'height="{self.h}" viewBox="0 0 {self.w} {self.h}">\n'
                f'<rect width="{self.w}" height="{self.h}" fill="#ffffff"/>\n'
                + self.marker_defs() + "\n"
                + "\n".join(self.body) + "\n</svg>\n")


def card(svg, x, y, w, h, title, sub=None, accent=BLUE, bg=BLUE_BG, ln=BLUE_LN,
         title_size=14.5, r=14):
    """A rounded card with a bold accent title and optional muted subtitle."""
    svg.rect(x, y, w, h, r=r, fill=bg, stroke=ln, sw=1.5, shadow=True)
    if sub:
        svg.text(x + w / 2, y + h / 2 - 4, title, size=title_size, fill=accent, weight="700")
        svg.text(x + w / 2, y + h / 2 + 15, sub, size=12, fill=MUTE, weight="400")
    else:
        svg.text(x + w / 2, y + h / 2 + 5, title, size=title_size, fill=accent, weight="700")


def title(svg, t, subtitle=None):
    svg.text(svg.w / 2, 34, t, size=21, fill=INK, weight="700")
    if subtitle:
        svg.text(svg.w / 2, 57, subtitle, size=13.5, fill=MUTE, weight="400")


def save(svg, name):
    (OUT / name).write_text(svg.render())
    print("wrote", name)


def elbow(x1, y1, x2, y2, mid=None):
    """Orthogonal path: horizontal then vertical (or via a mid x)."""
    if mid is None:
        mid = (x1 + x2) / 2
    return f"M {x1} {y1} H {mid} V {y2} H {x2}"


# ---------------------------------------------------------------------------
# 1. Pipeline flowchart  (clean two-row layout, orthogonal connectors)
# ---------------------------------------------------------------------------
def pipeline_flowchart():
    svg = SVG(980, 470)
    title(svg, "From a walking video to a health-condition label",
          "Both branches share one pose front-end, then meet again for a fair test")

    # Row 1: shared front-end (3 blue cards), left to right.
    y1 = 96
    cw, ch, gap = 172, 82, 40
    x0 = 40
    fe = []
    for i, (t, s) in enumerate([("Walking video", "49 mp4 clips"),
                                ("MediaPipe pose", "33 landmarks"),
                                ("Normalize", "pelvis + torso")]):
        x = x0 + i * (cw + gap)
        card(svg, x, y1, cw, ch, t, s, BLUE, BLUE_BG, BLUE_LN)
        fe.append((x, x + cw))
    # arrows between front-end cards
    for i in range(2):
        xa = fe[i][1]
        xb = fe[i + 1][0]
        svg.path(f"M {xa} {y1+ch/2} H {xb-2}", stroke=BLUE, arrow="arrowBlue", w=2.4)

    # Split node after Normalize.
    nx = fe[2][1]
    ny = y1 + ch / 2
    split_x = nx + 26
    svg.path(f"M {nx} {ny} H {split_x}", stroke=SLATE, arrow=None, w=2.4)
    svg.dot(split_x, ny, 4, SLATE)

    # Two branch cards on the right, stacked with a clear vertical gap between them.
    bx = split_x + 30
    bw, bh = 210, 72
    ay = 94          # branch A (top), aligned with the front-end row
    by = 214         # branch B (bottom)
    card(svg, bx, ay, bw, bh, "Branch A: S-JEPA", "self-supervised encoder + probe",
         ORANGE, ORANGE_BG, ORANGE_LN, title_size=13.5)
    card(svg, bx, by, bw, bh, "Branch B: Random Forest", "82 hand-made gait features",
         GREEN, GREEN_BG, GREEN_LN, title_size=13.5)

    # split -> Branch A: straight into A's left edge (A sits on the front-end row).
    svg.path(f"M {split_x} {ny} H {bx-2}", stroke=ORANGE, arrow="arrowOrange", w=2.2)
    # split -> Branch B: drop down a channel that stays LEFT of both cards (between
    # the split dot and the cards), so it never touches Branch A.
    gap_x = (split_x + bx) / 2          # channel sits in the gap before the cards
    svg.path(f"M {split_x} {ny} V {by+bh/2} H {bx-2}", stroke=GREEN,
             arrow="arrowGreen", w=2.2)

    # Compare card, bottom-centre.
    comp_w, comp_h = 250, 76
    comp_x = 300
    comp_y = 364
    card(svg, comp_x, comp_y, comp_w, comp_h, "Compare fairly",
         "same test videos, same metrics", SLATE, SLATE_BG, SLATE_LN)

    # Both branches exit their RIGHT edges into a shared vertical channel to the
    # right of the cards, drop to two separate horizontal channels below, then run
    # left into the Compare card top. Nothing crosses a card, lines never overlap.
    right_ch = bx + bw + 36             # vertical channel right of the cards
    chOrange = comp_y - 40              # lower entry channel (A)
    chGreen = comp_y - 20               # nearer entry channel (B)
    entryA = comp_x + comp_w * 0.4
    entryB = comp_x + comp_w * 0.6
    # Branch A: right edge -> right channel -> down -> left along chOrange -> into top.
    svg.path(f"M {bx+bw} {ay+bh/2} H {right_ch} V {chOrange} H {entryA} V {comp_y-2}",
             stroke=ORANGE, arrow="arrowOrange", w=2.0)
    # Branch B: right edge -> a slightly inner channel -> down -> left along chGreen.
    right_ch_b = right_ch - 16
    svg.path(f"M {bx+bw} {by+bh/2} H {right_ch_b} V {chGreen} H {entryB} V {comp_y-2}",
             stroke=GREEN, arrow="arrowGreen", w=2.0)

    # phase labels under the front-end
    svg.text(fe[0][0], y1 + ch + 26, "notebooks 00 - 01", size=11, fill=MUTE, anchor="start")
    save(svg, "pipeline_flowchart.svg")


# ---------------------------------------------------------------------------
# 2. S-JEPA two-lane training
# ---------------------------------------------------------------------------
def sjepa_two_lane():
    svg = SVG(940, 470)
    title(svg, "How S-JEPA learns: predict hidden joints in feature space")

    cw, ch, gap = 150, 66, 26
    x0 = 40
    # top lane (prediction) y, bottom lane (target) y
    yt = 120
    yb = 320

    svg.text(x0, yt - 20, "Prediction lane", size=13, fill=ORANGE, weight="700", anchor="start")
    svg.text(x0, yb - 20, "Target lane  (slow teacher, no gradient)", size=13, fill=BLUE,
             weight="700", anchor="start")

    top = [("Masked view", "visible joints"), ("View encoder", None),
           ("Predictor", None), ("Predicted", "features")]
    bot = [("Full skeleton", "all joints"), ("Target encoder", "EMA copy"),
           ("Mask output", None), ("Target", "center + sharpen")]

    tx, bx = [], []
    for i, (t, s) in enumerate(top):
        x = x0 + i * (cw + gap)
        tx.append(x)
        card(svg, x, yt, cw, ch, t, s, ORANGE, ORANGE_BG if i in (1, 2) else "#ffffff",
             ORANGE_LN, title_size=13)
    for i, (t, s) in enumerate(bot):
        x = x0 + i * (cw + gap)
        bx.append(x)
        card(svg, x, yb, cw, ch, t, s, BLUE, BLUE_BG if i in (1, 2) else "#ffffff",
             BLUE_LN, title_size=13)

    for i in range(3):
        svg.path(f"M {tx[i]+cw} {yt+ch/2} H {tx[i+1]-2}", stroke=ORANGE, arrow="arrowOrange", w=2)
        svg.path(f"M {bx[i]+cw} {yb+ch/2} H {bx[i+1]-2}", stroke=BLUE, arrow="arrowBlue", w=2)

    # Loss node on the right, both lanes flow into it (orthogonal, no cross).
    lx = tx[3] + cw + 30
    lw, lh = 120, 90
    ly = (yt + yb) / 2 + ch / 2 - lh / 2
    card(svg, lx, ly, lw, lh, "Latent", "cross-entropy", SLATE, SLATE_BG, SLATE_LN)
    svg.path(f"M {tx[3]+cw} {yt+ch/2} H {lx-14} V {ly+lh/2-8} H {lx-2}",
             stroke=MUTE, arrow="arrowMute", w=1.8)
    svg.path(f"M {bx[3]+cw} {yb+ch/2} H {lx-14} V {ly+lh/2+8} H {lx-2}",
             stroke=MUTE, arrow="arrowMute", w=1.8)

    # EMA feedback: view encoder -> target encoder (dashed, left channel, no cross).
    ema_x = x0 + cw + gap / 2 - 4
    ema_mid = (yt + yb) / 2
    svg.path(f"M {tx[1]+cw/2} {yt+ch} V {ema_mid} H {ema_x} V {yb-2}",
             stroke=MUTE, arrow="arrowMute", w=1.6, dash="5 4")
    svg.text(tx[1] + cw / 2 + 12, ema_mid - 10, "EMA update (slow copy)", size=11,
             fill=MUTE, anchor="start")
    save(svg, "sjepa_two_lane.svg")


# ---------------------------------------------------------------------------
# 3. Anatomical mask
# ---------------------------------------------------------------------------
def anatomical_mask():
    svg = SVG(720, 620)
    title(svg, "The anatomical mask",
          "We hide 12 fixed joints, both shoulders and both legs, and predict them")

    # upright skeleton, y grows downward in SVG so use screen coords directly
    C = {
        0: (360, 118), 7: (338, 112), 8: (382, 112), 9: (348, 128), 10: (372, 128),
        11: (300, 205), 12: (420, 205),
        13: (250, 285), 14: (470, 285),
        15: (215, 360), 16: (505, 360),
        17: (198, 388), 18: (522, 388), 19: (214, 392), 20: (506, 392),
        21: (232, 372), 22: (488, 372),
        23: (325, 320), 24: (395, 320),
        25: (315, 425), 26: (405, 425),
        27: (308, 520), 28: (412, 520),
        29: (292, 552), 30: (428, 552), 31: (330, 566), 32: (390, 566),
    }
    conns = [(11, 12), (11, 23), (12, 24), (23, 24),
             (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
             (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
             (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
             (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
             (0, 9), (0, 10), (9, 10)]
    masked = {11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32}

    for a, b in conns:
        if a in C and b in C:
            svg.line(*C[a], *C[b], stroke=HAIR, w=3)
    for i, (x, y) in C.items():
        if i in masked:
            svg.circle(x, y, 9, ORANGE)
        else:
            svg.circle(x, y, 6, SLATE, sw=1.5)

    # legend
    ly = 588
    svg.circle(120, ly, 9, ORANGE)
    svg.text(138, ly + 4, "masked target joints (12)", size=13, fill=INK, anchor="start")
    svg.circle(400, ly, 6, SLATE, sw=1.5)
    svg.text(414, ly + 4, "visible context joints (face and arms)", size=13, fill=INK,
             anchor="start")
    save(svg, "anatomical_mask.svg")


# ---------------------------------------------------------------------------
# 4. Tokenization
# ---------------------------------------------------------------------------
def tokenization():
    svg = SVG(900, 460)
    title(svg, "Tokenizing a skeleton window",
          "Group 4 adjacent frames of one joint into a single token")

    # left grid: frames (x) by joints (y)
    gx, gy, cell = 60, 130, 30
    nf, nj = 8, 6
    svg.text(gx + nf * cell / 2, gy - 16, "frames × joints", size=13, fill=MUTE)
    for f in range(nf):
        for j in range(nj):
            fill = BLUE_BG if f < 4 else GREEN_BG
            stroke = BLUE_LN if f < 4 else GREEN_LN
            svg.rect(gx + f * cell, gy + j * cell, cell - 3, cell - 3, r=4,
                     fill=fill, stroke=stroke, sw=1)
    # group brackets
    svg.line(gx, gy + nj * cell + 8, gx + 4 * cell - 3, gy + nj * cell + 8, stroke=BLUE, w=2.5)
    svg.line(gx + 4 * cell, gy + nj * cell + 8, gx + nf * cell - 3, gy + nj * cell + 8,
             stroke=GREEN, w=2.5)
    svg.text(gx + 2 * cell, gy + nj * cell + 28, "block 1", size=12, fill=BLUE)
    svg.text(gx + 6 * cell, gy + nj * cell + 28, "block 2", size=12, fill=GREEN)

    # arrow to tokens
    ax1 = gx + nf * cell + 20
    ax2 = 560
    midy = gy + nj * cell / 2
    svg.path(f"M {ax1} {midy} H {ax2}", stroke=MUTE, arrow="arrowMute", w=2.4)

    # token column
    tx, tw = 580, 190
    svg.text(tx + tw / 2, gy - 16, "tokens (dim d)", size=13, fill=MUTE)
    for k in range(6):
        yv = gy + k * 27
        svg.rect(tx, yv, tw, 20, r=6, fill=SLATE_BG, stroke=SLATE_LN, sw=1)
    # position-embedding bars, clearly separated with labels below
    bar_top = gy
    bar_h = 6 * 27 - 7
    b1 = tx + tw + 20
    b2 = b1 + 60
    svg.rect(b1, bar_top, 22, bar_h, r=4, fill="#fde3c4", stroke=ORANGE, sw=1)
    svg.rect(b2, bar_top, 22, bar_h, r=4, fill="#c8f2d4", stroke=GREEN, sw=1)
    svg.text(b1 + 11, bar_top + bar_h + 22, "spatial", size=11, fill=ORANGE)
    svg.text(b2 + 11, bar_top + bar_h + 22, "temporal", size=11, fill=GREEN)
    svg.text((b1 + b2) / 2 + 11, gy - 16, "+ position", size=12, fill=MUTE)
    save(svg, "tokenization.svg")


# ---------------------------------------------------------------------------
# 5. Progressive timeline
# ---------------------------------------------------------------------------
def progressive_timeline():
    svg = SVG(900, 340)
    title(svg, "Progressive training")

    y = 150
    svg.line(60, y, 840, y, stroke=HAIR, w=3)
    stages = [(180, "Phase 1", "Pretrain on normal gait", BLUE, BLUE_BG, BLUE_LN),
              (450, "Phase 2", "Add ms and pd sequences", ORANGE, ORANGE_BG, ORANGE_LN),
              (720, "Phase 3", "Add VICReg to separate", GREEN, GREEN_BG, GREEN_LN)]
    cw, ch = 220, 66
    for x, tag, desc, ac, bg, ln in stages:
        card(svg, x - cw / 2, y - ch - 34, cw, ch, desc, None, ac, bg, ln, title_size=13.5)
        svg.circle(x, y, 8, ac)
        svg.text(x, y + 30, tag, size=13, fill=ac, weight="700")
    for x in (315, 585):
        svg.path(f"M {x-40} {y} H {x+40}", stroke=LINE, arrow="arrow", w=2)
    svg.text(svg.w / 2, 312, "capacity and robustness grow left to right", size=12, fill=MUTE)
    save(svg, "progressive_timeline.svg")


# ---------------------------------------------------------------------------
# 6. VICReg clusters
# ---------------------------------------------------------------------------
def vicreg_clusters():
    import math
    svg = SVG(900, 440)
    title(svg, "VICReg keeps the three condition clusters apart")

    def blob(cx, cy, col, spread, seed):
        # deterministic pseudo-random scatter
        pts = []
        s = seed
        for _ in range(16):
            s = (s * 1103515245 + 12345) & 0x7fffffff
            a = (s / 0x7fffffff) * 2 * math.pi
            s = (s * 1103515245 + 12345) & 0x7fffffff
            r = (s / 0x7fffffff) ** 0.5 * spread
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
        for (x, y) in pts:
            svg.dot(x, y, 4.5, col)

    # left: overlapping
    lx, ly = 250, 260
    svg.text(lx, 110, "Without VICReg", size=14, fill=INK, weight="600")
    svg.text(lx, 130, "clusters overlap", size=12, fill=MUTE)
    blob(lx, ly, BLUE, 55, 7)
    blob(lx + 15, ly + 10, ORANGE, 55, 23)
    blob(lx - 10, ly - 8, GREEN, 55, 99)

    # right: separated
    rx, ry = 660, 250
    svg.text(rx, 110, "With VICReg", size=14, fill=INK, weight="600")
    svg.text(rx, 130, "clusters pull apart", size=12, fill=MUTE)
    blob(rx - 55, ry + 40, BLUE, 30, 5)
    blob(rx + 60, ry + 45, ORANGE, 30, 31)
    blob(rx, ry - 55, GREEN, 30, 71)

    # three-term note
    svg.text(svg.w / 2, 400,
             "variance keeps spread    •    invariance matches views    •    "
             "covariance decorrelates", size=12, fill=MUTE)
    # legend
    for i, (c, lab) in enumerate([(BLUE, "normal"), (ORANGE, "ms"), (GREEN, "pd")]):
        x = 360 + i * 90
        svg.dot(x, 424, 5, c)
        svg.text(x + 10, 428, lab, size=12, fill=INK, anchor="start")
    save(svg, "vicreg_clusters.svg")


# ---------------------------------------------------------------------------
# 7. Grouped split
# ---------------------------------------------------------------------------
def grouped_split():
    svg = SVG(900, 400)
    title(svg, "Leakage-safe splitting by source video",
          "All clips from one source stay on the same side of the split")

    def clips(x0, y0, n, ac, bg, ln):
        for i in range(n):
            svg.rect(x0 + i * 74, y0, 66, 46, r=8, fill=bg, stroke=ln, sw=1.4)
            svg.text(x0 + i * 74 + 33, y0 + 28, f"clip {i+1}", size=11, fill=ac)

    svg.text(70, 150, "Source A", size=14, fill=BLUE, weight="700", anchor="start")
    clips(70, 160, 3, BLUE, BLUE_BG, BLUE_LN)
    svg.text(70, 270, "Source B", size=14, fill=ORANGE, weight="700", anchor="start")
    clips(70, 280, 2, ORANGE, ORANGE_BG, ORANGE_LN)

    # bins
    card(svg, 640, 150, 150, 66, "TRAIN", None, "#0e7490", "#ecfeff", "#a5f3fc")
    card(svg, 640, 268, 150, 66, "TEST", None, "#b45309", "#fffbeb", "#fde68a")
    svg.path(f"M 300 183 H 638", stroke=BLUE, arrow="arrowBlue", w=2.2)
    svg.path(f"M 226 303 H 638", stroke=ORANGE, arrow="arrowOrange", w=2.2)
    svg.text(806, 187, "whole source A", size=11, fill="#0e7490", anchor="start")
    svg.text(806, 305, "whole source B", size=11, fill="#b45309", anchor="start")
    save(svg, "grouped_split.svg")


# ---------------------------------------------------------------------------
# 8. RF vs S-JEPA
# ---------------------------------------------------------------------------
def rf_vs_sjepa():
    svg = SVG(900, 400)
    title(svg, "A fair head-to-head comparison")

    card(svg, 50, 110, 250, 74, "S-JEPA linear probe", "frozen learned features",
         ORANGE, ORANGE_BG, ORANGE_LN, title_size=14)
    card(svg, 50, 236, 250, 74, "Random Forest", "82 hand-made features",
         GREEN, GREEN_BG, GREEN_LN, title_size=14)
    card(svg, 400, 173, 170, 74, "Same test videos", None, SLATE, SLATE_BG, SLATE_LN,
         title_size=14)
    card(svg, 640, 173, 200, 74, "Accuracy, macro F1", "mean ± std", SLATE, SLATE_BG,
         SLATE_LN, title_size=13.5)

    svg.path("M 300 147 H 360 V 200 H 398", stroke=ORANGE, arrow="arrowOrange", w=2.2)
    svg.path("M 300 273 H 360 V 220 H 398", stroke=GREEN, arrow="arrowGreen", w=2.2)
    svg.path("M 570 210 H 638", stroke=SLATE, arrow="arrowSlate", w=2.2)
    svg.text(svg.w / 2, 360,
             "grouped k-fold over source videos, identical folds for both models",
             size=12, fill=MUTE)
    save(svg, "rf_vs_sjepa.svg")


# ---------------------------------------------------------------------------
# 9. Why it matters
# ---------------------------------------------------------------------------
def why_it_matters():
    svg = SVG(900, 380)
    title(svg, "Why learn gait from video")

    cw, ch, gap = 230, 90, 60
    x0 = 40
    y = 120
    steps = [("A phone video", "no wearables, no lab", BLUE, BLUE_BG, BLUE_LN),
             ("Skeleton motion", "how the joints move", ORANGE, ORANGE_BG, ORANGE_LN),
             ("Early, cheap signal", "screening for MS and PD", GREEN, GREEN_BG, GREEN_LN)]
    xs = []
    for i, (t, s, ac, bg, ln) in enumerate(steps):
        x = x0 + i * (cw + gap)
        xs.append(x)
        card(svg, x, y, cw, ch, t, s, ac, bg, ln, title_size=15)
    for i in range(2):
        svg.path(f"M {xs[i]+cw} {y+ch/2} H {xs[i+1]-2}", stroke=LINE, arrow="arrow", w=2.2)
    svg.text(svg.w / 2, 280, "Gait changes are among the earliest signs of MS and PD.",
             size=13.5, fill=INK)
    svg.text(svg.w / 2, 306,
             "Reading that signal from ordinary video could widen access to monitoring.",
             size=12.5, fill=MUTE)
    save(svg, "why_it_matters.svg")


# ---------------------------------------------------------------------------
# 10. Project status
# ---------------------------------------------------------------------------
def project_status():
    svg = SVG(900, 430)
    title(svg, "What is built and verified")

    items = [
        "Raw-video pose pipeline (cached)", "Progressive: normal then MS + PD",
        "S-JEPA encoder, teacher, predictor", "Two profiles, both tested",
        "Fixed clinical mask of 12 joints", "Random Forest baseline (exp5)",
        "Centering + sharpening CE loss", "Leakage-safe grouped k-fold",
        "VICReg extension for separation", "Notebooks, slides, and docs",
    ]
    col_x = [70, 480]
    y0 = 110
    row_h = 56
    for i, it in enumerate(items):
        col = i % 2
        row = i // 2
        x = col_x[col]
        y = y0 + row * row_h
        # green check disc
        svg.circle(x, y, 12, GREEN)
        svg.path(f"M {x-5} {y+1} L {x-1} {y+5} L {x+6} {y-5}", stroke="#fff", w=2.4, arrow=None)
        svg.text(x + 24, y + 5, it, size=13.5, fill=INK, anchor="start")
    save(svg, "project_status.svg")


# ---------------------------------------------------------------------------
# 11. Roadmap
# ---------------------------------------------------------------------------
def roadmap():
    svg = SVG(900, 380)
    title(svg, "What comes next")

    lanes = [("More data", "more clips and sources", BLUE, BLUE_BG, BLUE_LN),
             ("Bigger model", "gpu profile, longer runs", ORANGE, ORANGE_BG, ORANGE_LN),
             ("Transfer", "pretrain then fine-tune", GREEN, GREEN_BG, GREEN_LN),
             ("Clinical validity", "gait phases, expert review", PURPLE, PURPLE_BG, PURPLE_LN)]
    cw, ch, gap = 190, 100, 24
    x0 = 40
    y = 120
    xs = []
    for i, (t, s, ac, bg, ln) in enumerate(lanes):
        x = x0 + i * (cw + gap)
        xs.append(x + cw / 2)
        svg.text(x + cw / 2, y - 12, str(i + 1), size=14, fill=ac, weight="700")
        card(svg, x, y, cw, ch, t, s, ac, bg, ln, title_size=14.5)
    # single timeline arrow beneath
    svg.path(f"M {x0} 300 H {x0 + 4*cw + 3*gap - 10}", stroke=LINE, arrow="arrow", w=2.2)
    svg.text(svg.w / 2, 330, "near term  →  longer term", size=12, fill=MUTE)
    save(svg, "roadmap.svg")


# ---------------------------------------------------------------------------
# 12. Results bars  (kept in matplotlib for accurate error bars)
# ---------------------------------------------------------------------------
def results_bars():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    metrics = ["accuracy", "macro F1"]
    rf_mean, rf_std = [0.66, 0.67], [0.09, 0.10]
    sj_mean, sj_std = [0.57, 0.57], [0.10, 0.11]
    x = np.arange(len(metrics)); w = 0.34
    ax.bar(x - w / 2, rf_mean, w, yerr=rf_std, capsize=6, color=GREEN, label="Random Forest",
           edgecolor="white", linewidth=1)
    ax.bar(x + w / 2, sj_mean, w, yerr=sj_std, capsize=6, color=ORANGE, label="S-JEPA probe",
           edgecolor="white", linewidth=1)
    ax.axhline(1 / 3, ls="--", lw=1, color=MUTE)
    ax.text(1.36, 1 / 3 + 0.015, "chance (3 classes)", fontsize=8.5, color=MUTE, ha="right")
    ax.set_xticks(x); ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1); ax.set_ylabel("score", fontsize=11)
    ax.set_title("Grouped k-fold results (laptop profile)", fontsize=14, weight="bold", color=INK)
    ax.legend(frameon=False, fontsize=10.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(0.5, -0.01, "Small dataset: treat as a methodology demo, not a clinical result.",
             ha="center", fontsize=9, color=MUTE)
    fig.savefig(OUT / "results_bars.svg", format="svg", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote results_bars.svg")


# ---------------------------------------------------------------------------
# 13. Steps we followed  (vertical numbered flowchart, beginner friendly)
# ---------------------------------------------------------------------------
def steps_we_followed():
    steps = [
        ("Collect walking videos", "49 short clips: normal, MS, and PD", BLUE, BLUE_BG, BLUE_LN),
        ("Find the skeleton", "run a pose detector, 33 body points per frame",
         BLUE, BLUE_BG, BLUE_LN),
        ("Tidy and line up", "fill small gaps, center on the hips, scale by body size",
         ORANGE, ORANGE_BG, ORANGE_LN),
        ("Cut into tokens", "group 4 frames of a joint into one token",
         ORANGE, ORANGE_BG, ORANGE_LN),
        ("Teach S-JEPA", "hide 12 joints, guess them as features, learn with a slow teacher",
         GREEN, GREEN_BG, GREEN_LN),
        ("Add MS, PD, and VICReg", "keep training on all three groups, spread the clusters",
         GREEN, GREEN_BG, GREEN_LN),
        ("Compare against Random Forest", "same videos, same fair splits, same scores",
         PURPLE, PURPLE_BG, PURPLE_LN),
    ]
    n = len(steps)
    row_h = 74
    top = 80
    svg = SVG(760, top + n * row_h + 30)
    title(svg, "The steps we followed")

    cx = 70          # centre of the number circles
    cw = 520         # card width
    card_x = 130
    for i, (t, s, ac, bg, ln) in enumerate(steps):
        y = top + i * row_h
        cy = y + 24
        # connector to next step
        if i < n - 1:
            svg.line(cx, cy + 18, cx, y + row_h + 6, stroke=HAIR, w=3)
        card(svg, card_x, y, cw, 50, t, s, ac, bg, ln, title_size=13.5, r=12)
        # number badge on top
        svg.circle(cx, cy, 18, ac)
        svg.text(cx, cy + 5, str(i + 1), size=15, fill="#ffffff", weight="700")
    save(svg, "steps_we_followed.svg")


# ---------------------------------------------------------------------------
# 14. What we tried  (attempt -> outcome log)
# ---------------------------------------------------------------------------
def experiments_tried():
    rows = [
        ("Pose on raw video, no lab setup",
         "Worked: 47 of 49 clips gave clean skeletons", GREEN),
        ("Hide 12 clinical joints instead of the busiest ones",
         "Kept: better fit for gait, model still trains", GREEN),
        ("Pretrain on normal, then add MS and PD",
         "Kept: staged training runs end to end", GREEN),
        ("Plain training with no slow teacher",
         "Failed: features collapsed, so we kept the teacher", ORANGE),
        ("Tiny/short training runs",
         "Fixed: added gradient clipping to stop NaN blow-ups", ORANGE),
        ("Add VICReg to separate the three groups",
         "Partial: clusters not clean yet on this little data", ORANGE),
        ("Random Forest on 82 hand-made features",
         "Worked: strong baseline, our fair yardstick", GREEN),
    ]
    n = len(rows)
    row_h = 52
    top = 84
    svg = SVG(900, top + n * row_h + 30)
    title(svg, "What we tried, and what happened")

    x0 = 40
    wtry, wout = 400, 400
    gap = 20
    # header
    svg.text(x0 + wtry / 2, top - 12, "What we tried", size=12.5, fill=MUTE, weight="600")
    svg.text(x0 + wtry + gap + wout / 2, top - 12, "What happened", size=12.5,
             fill=MUTE, weight="600")
    for i, (attempt, outcome, col) in enumerate(rows):
        y = top + i * row_h
        svg.rect(x0, y, wtry, 42, r=10, fill="#ffffff", stroke=HAIR, sw=1.4)
        svg.text(x0 + 16, y + 26, attempt, size=12.5, fill=INK, anchor="start")
        obg = GREEN_BG if col == GREEN else ORANGE_BG
        oln = GREEN_LN if col == GREEN else ORANGE_LN
        svg.rect(x0 + wtry + gap, y, wout, 42, r=10, fill=obg, stroke=oln, sw=1.4)
        # status dot
        svg.dot(x0 + wtry + gap + 16, y + 21, 5, col)
        svg.text(x0 + wtry + gap + 30, y + 26, outcome, size=12.5, fill=INK, anchor="start")
    save(svg, "experiments_tried.svg")


# ---------------------------------------------------------------------------
# 15. How to read the results
# ---------------------------------------------------------------------------
def results_readout():
    svg = SVG(900, 400)
    title(svg, "How to read the results")

    # left: the headline numbers as two stat cards
    card(svg, 50, 100, 360, 90, "Random Forest", "accuracy 0.66   •   macro F1 0.67",
         GREEN, GREEN_BG, GREEN_LN, title_size=16)
    card(svg, 50, 210, 360, 90, "S-JEPA probe", "accuracy 0.57   •   macro F1 0.57",
         ORANGE, ORANGE_BG, ORANGE_LN, title_size=16)

    # right: three plain-language takeaways
    notes = [
        (BLUE, "The test is fair", "same videos, same splits, no leakage"),
        (SLATE, "Numbers wobble", "only ~35 sources, so a few swap and scores shift"),
        (PURPLE, "S-JEPA has room to grow", "its edge shows with more data and less labels"),
    ]
    nx = 470
    for i, (col, head, body) in enumerate(notes):
        y = 105 + i * 70
        svg.rect(nx, y, 12, 46, r=4, fill=col)          # accent bar
        svg.text(nx + 26, y + 18, head, size=14, fill=INK, weight="700", anchor="start")
        svg.text(nx + 26, y + 38, body, size=12, fill=MUTE, anchor="start")
    svg.text(svg.w / 2, 360,
             "On a few dozen labeled videos a classical model is hard to beat. This is a "
             "method demo, not a clinical result.", size=12, fill=MUTE)
    save(svg, "results_readout.svg")


if __name__ == "__main__":
    pipeline_flowchart()
    sjepa_two_lane()
    anatomical_mask()
    tokenization()
    progressive_timeline()
    vicreg_clusters()
    grouped_split()
    rf_vs_sjepa()
    why_it_matters()
    project_status()
    roadmap()
    results_bars()
    steps_we_followed()
    experiments_tried()
    results_readout()
    print("\nAll diagrams written to", OUT)
