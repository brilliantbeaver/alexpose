"""Generate the diagrams for docs/03-0803-FIXES.md.

These illustrate the correctness repairs, the evaluation firewall, and the R1
plan. We reuse the clean SVG design system in ``scripts_make_diagrams.py`` (one
accent hue per role, rounded cards, orthogonal connectors routed through clear
channels so no line crosses a box and no two lines overlap) rather than
duplicating it, so the whole image set stays visually consistent.

Run:  python scripts_make_fixes_diagrams.py
"""

from __future__ import annotations

from scripts_make_diagrams import (
    SVG, card, title, save, elbow,
    INK, MUTE, HAIR, LINE,
    BLUE, BLUE_BG, BLUE_LN,
    ORANGE, ORANGE_BG, ORANGE_LN,
    GREEN, GREEN_BG, GREEN_LN,
    SLATE, SLATE_BG, SLATE_LN,
    PURPLE, PURPLE_BG, PURPLE_LN,
)

# A red accent for "broken / defect" callouts (kept out of the class palette).
RED, RED_BG, RED_LN = "#dc2626", "#fef2f2", "#fecaca"


def _red_marker(svg: SVG) -> None:
    """Add a red arrowhead marker (the shared defs only ship the palette hues)."""
    svg.add(
        '<marker id="arrowRed" markerWidth="9" markerHeight="9" refX="6.5" '
        'refY="4" orient="auto"><path d="M0.5,0.5 L8,4 L0.5,7.5 z" '
        f'fill="{RED}"/></marker>'
    )


# ---------------------------------------------------------------------------
# 1. Fixes overview: 7 defects -> 7 repairs
# ---------------------------------------------------------------------------
def fixes_overview():
    svg = SVG(980, 620)
    _red_marker(svg)
    title(svg, "Seven repairs, from broken mechanics to a trustworthy baseline",
          "Left: what is wrong today.  Right: the fix.  Each row is independent.")

    rows = [
        ("D1", "Predictor has no sense of WHICH joint/time it predicts",
         "Give every hidden slot a joint + time position tag"),
        ("D2", "The 12 clinical joints are hidden on every step",
         "Hide different joint groups each step (stochastic masks)"),
        ("D3", "“VICReg” secretly uses the diagnosis labels",
         "Remove labels from self-supervised training"),
        ("D4", "Training restarts its optimizer state every stage",
         "Save and restore the full training state"),
        ("D5", "All MS clips are 60fps square: an easy cheat",
         "Score fair splits + measure the shortcut controls"),
        ("D6", "Per-frame centering erases walking speed",
         "Keep a validity mask; log what normalization removes"),
        ("D7", "One mask is reused for the whole batch",
         "Per-example masks, padding handled in attention"),
    ]
    top = 84
    row_h = 74
    left_x, right_x = 70, 545
    cw = 360
    for i, (tag, bad, good) in enumerate(rows):
        y = top + i * row_h
        # broken card (left)
        svg.rect(left_x, y, cw, 54, r=12, fill=RED_BG, stroke=RED_LN, sw=1.4, shadow=True)
        svg.circle(left_x + 26, y + 27, 15, RED)
        svg.text(left_x + 26, y + 32, tag, size=12.5, fill="#ffffff", weight="700")
        svg.text(left_x + 52, y + 31, bad, size=12, fill=INK, anchor="start")
        # repaired card (right)
        svg.rect(right_x, y, cw, 54, r=12, fill=GREEN_BG, stroke=GREEN_LN, sw=1.4, shadow=True)
        svg.circle(right_x + 26, y + 27, 15, GREEN)
        # check glyph
        cx0, cy0 = right_x + 26, y + 27
        svg.path(f"M {cx0-6} {cy0+1} L {cx0-1} {cy0+6} L {cx0+7} {cy0-6}",
                 stroke="#fff", w=2.6, arrow=None)
        svg.text(right_x + 52, y + 31, good, size=12, fill=INK, anchor="start")
        # arrow broken -> fixed
        svg.path(f"M {left_x+cw+8} {y+27} H {right_x-8}", stroke=SLATE,
                 arrow="arrowSlate", w=2.0)
    save(svg, "fixes_overview.svg")


# ---------------------------------------------------------------------------
# 2. D1 predictor position identity
# ---------------------------------------------------------------------------
def defect_predictor_positions():
    svg = SVG(960, 470)
    _red_marker(svg)
    title(svg, "Repair 1: the predictor must know which joint it is guessing",
          "Without a position tag, every hidden slot gets the identical input and identical guess")

    # ---- BEFORE (top half) ----
    svg.text(60, 92, "Before", size=14, fill=RED, weight="700", anchor="start")
    svg.text(60, 110, "same mask token everywhere", size=12, fill=MUTE, anchor="start")
    bx = 60
    by = 122
    # three identical hidden tokens
    for i in range(3):
        x = bx + i * 95
        svg.rect(x, by, 78, 40, r=9, fill=RED_BG, stroke=RED_LN, sw=1.4)
        svg.text(x + 39, by + 25, "mask", size=12, fill=RED)
    svg.text(bx + 3 * 95 + 6, by + 25, "→  predictor", size=12.5, fill=MUTE, anchor="start")
    # predictor box
    pbx = bx + 3 * 95 + 108
    svg.rect(pbx, by - 4, 96, 48, r=10, fill=SLATE_BG, stroke=SLATE_LN, sw=1.5)
    svg.text(pbx + 48, by + 24, "predictor", size=12.5, fill=SLATE, weight="700")
    # identical outputs
    obx = pbx + 130
    for i in range(3):
        x = obx + i * 60
        svg.rect(x, by, 50, 40, r=9, fill="#ffffff", stroke=RED_LN, sw=1.4)
        svg.text(x + 25, by + 25, "=", size=15, fill=RED, weight="700")
    svg.path(f"M {pbx+96} {by+20} H {obx-4}", stroke=RED, arrow="arrowRed", w=2)
    svg.text(obx + 3 * 60 + 8, by + 25, "std ≈ 0", size=12.5, fill=RED,
             weight="700", anchor="start")

    # divider
    svg.line(60, 232, 900, 232, stroke=HAIR, w=1.5)

    # ---- AFTER (bottom half) ----
    svg.text(60, 268, "After", size=14, fill=GREEN, weight="700", anchor="start")
    svg.text(60, 286, "mask token + joint tag + time tag", size=12, fill=MUTE, anchor="start")
    ay = 300
    tags = [("mask+J25,T3", ORANGE), ("mask+J11,T0", BLUE), ("mask+J28,T6", GREEN)]
    for i, (lab, col) in enumerate(tags):
        x = bx + i * 95
        svg.rect(x, ay, 78, 40, r=9, fill="#ffffff", stroke=col, sw=1.6)
        svg.text(x + 39, ay + 24, lab, size=9.5, fill=col, weight="700")
    svg.text(bx + 3 * 95 + 6, ay + 25, "→  predictor", size=12.5, fill=MUTE, anchor="start")
    svg.rect(pbx, ay - 4, 96, 48, r=10, fill=SLATE_BG, stroke=SLATE_LN, sw=1.5)
    svg.text(pbx + 48, ay + 24, "predictor", size=12.5, fill=SLATE, weight="700")
    outs = [ORANGE, BLUE, GREEN]
    for i, col in enumerate(outs):
        x = obx + i * 60
        svg.rect(x, ay, 50, 40, r=9, fill="#ffffff", stroke=col, sw=1.6)
        svg.text(x + 25, ay + 25, "≠", size=15, fill=col, weight="700")
    svg.path(f"M {pbx+96} {ay+20} H {obx-4}", stroke=GREEN, arrow="arrowGreen", w=2)
    svg.text(obx + 3 * 60 + 8, ay + 25, "distinct", size=12.5, fill=GREEN,
             weight="700", anchor="start")

    svg.text(svg.w / 2, 448,
             "Test: permuting the position tags must change the matching predictions.",
             size=12, fill=MUTE)
    save(svg, "defect_predictor_positions.svg")


# ---------------------------------------------------------------------------
# 3. D2 mask starvation -> stochastic graph-time masks
# ---------------------------------------------------------------------------
def defect_mask_starvation():
    svg = SVG(960, 500)
    title(svg, "Repair 2: rotate which joints are hidden",
          "A joint the encoder never sees never learns. Every joint must be context sometimes.")

    joints_left = [11, 12, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    def joint_grid(x0, y0, hidden_set, label, sub, always_hidden=False):
        svg.text(x0 + 150, y0 - 14, label, size=13.5, fill=INK, weight="700")
        svg.text(x0 + 150, y0 + 4, sub, size=11.5, fill=MUTE)
        cols = 6
        cell = 42
        for k in range(18):  # show 18 of the 33 joints for compactness
            r, c = divmod(k, cols)
            x = x0 + c * cell
            y = y0 + 22 + r * cell
            jid = k
            is_hidden = jid in hidden_set
            if is_hidden:
                svg.circle(x + 16, y + 16, 11, ORANGE)
                svg.text(x + 16, y + 20, str(jid), size=9, fill="#fff", weight="700")
            else:
                svg.circle(x + 16, y + 16, 8, SLATE, sw=1.4)
                svg.text(x + 16, y + 20, str(jid), size=8.5, fill=MUTE)

    # BEFORE: joints 11,12 (of the shown 0..17) always hidden -> map to the clinical set
    # For the illustration we mark a fixed subset every panel to convey "same every time".
    fixed = {11, 12, 13, 14, 15, 16, 17}
    joint_grid(70, 110, fixed, "Before: fixed mask",
               "same joints hidden on every single step")
    # three little "step" panels showing identical mask
    for s in range(3):
        svg.text(70 + 150, 300 + s * 8, "", size=10)
    svg.text(70 + 150, 300, "step 1 = step 2 = step 3  (identical)", size=11.5,
             fill=RED, weight="600")
    svg.text(70 + 150, 320, "hidden joints get zero context gradient", size=11.5, fill=RED)

    # divider
    svg.line(490, 96, 490, 440, stroke=HAIR, w=1.5)

    # AFTER: three different masks across steps
    svg.text(560 + 150, 96, "After: stochastic graph-time masks", size=13.5,
             fill=INK, weight="700")
    svg.text(560 + 150, 114, "a different connected group each step", size=11.5, fill=MUTE)
    step_sets = [{2, 3, 8, 9, 14, 15}, {5, 6, 11, 12, 17}, {0, 1, 7, 13}]
    for s, hs in enumerate(step_sets):
        y0 = 140 + s * 100
        svg.text(560, y0 + 16, f"step {s+1}", size=11.5, fill=SLATE, weight="700", anchor="start")
        cols = 6
        cell = 34
        for k in range(12):
            r, c = divmod(k, cols)
            x = 620 + c * cell
            y = y0 + r * cell
            if k in hs:
                svg.circle(x + 14, y + 14, 9, ORANGE)
            else:
                svg.circle(x + 14, y + 14, 6, SLATE, sw=1.3)
    svg.text(560 + 150, 452, "clinical joints simply targeted a bit more often (~1.5x)",
             size=11.5, fill=GREEN, weight="600")
    # legend
    svg.circle(120, 470, 9, ORANGE)
    svg.text(136, 474, "hidden / target", size=12, fill=INK, anchor="start")
    svg.circle(300, 470, 6, SLATE, sw=1.4)
    svg.text(314, 474, "visible context", size=12, fill=INK, anchor="start")
    save(svg, "defect_mask_starvation.svg")


# ---------------------------------------------------------------------------
# 4. D5 domain leakage
# ---------------------------------------------------------------------------
def defect_domain_leakage():
    svg = SVG(960, 470)
    _red_marker(svg)
    title(svg, "Repair 5: the acquisition shortcut we must not reward",
          "Every MS clip is 60fps and a square 1080x1080 frame. No normal or PD clip is.")

    # three class rows with fps/resolution chips
    rows = [
        ("normal", "24-30 fps  •  many shapes (portrait, wide, small)", BLUE, BLUE_BG, BLUE_LN, "16 sources"),
        ("MS", "ALL 60 fps  •  ALL square 1080x1080", ORANGE, ORANGE_BG, ORANGE_LN, "11 sources"),
        ("PD", "24-30 fps  •  many shapes", GREEN, GREEN_BG, GREEN_LN, "8 sources"),
    ]
    y0 = 110
    rh = 78
    for i, (cls, desc, ac, bg, ln, src) in enumerate(rows):
        y = y0 + i * rh
        svg.rect(60, y, 620, 60, r=12, fill=bg, stroke=ln, sw=1.5, shadow=True)
        svg.text(96, y + 26, cls, size=16, fill=ac, weight="700", anchor="start")
        svg.text(96, y + 46, src, size=11, fill=MUTE, anchor="start")
        svg.text(230, y + 36, desc, size=12.5, fill=INK, anchor="start")
        if cls == "MS":
            svg.rect(230, y + 10, 12, 40, r=3, fill=RED)  # highlight bar

    # the danger callout
    svg.rect(710, 110, 210, 204, r=14, fill=RED_BG, stroke=RED_LN, sw=1.6, shadow=True)
    svg.text(815, 140, "The trap", size=14, fill=RED, weight="700")
    for k, ln in enumerate([
        "“Is it 60fps &",
        "square?” alone",
        "labels MS at",
        "~100% accuracy.",
        "",
        "That is domain,",
        "not gait.",
    ]):
        svg.text(815, 168 + k * 20, ln, size=12, fill=INK)

    # what we do about it
    svg.rect(60, 350, 860, 74, r=14, fill=SLATE_BG, stroke=SLATE_LN, sw=1.5, shadow=True)
    svg.text(90, 380, "What we do:", size=13, fill=SLATE, weight="700", anchor="start")
    svg.text(90, 404,
             "Report shortcut controls (fps / resolution / static-pose / visibility) next to every "
             "headline score,",
             size=12, fill=INK, anchor="start")
    svg.text(90, 420,
             "so a result that a nuisance control can fully explain cannot claim to read gait.",
             size=12, fill=INK, anchor="start")
    save(svg, "defect_domain_leakage.svg")


# ---------------------------------------------------------------------------
# 5. Repair dependency DAG (phase order)
# ---------------------------------------------------------------------------
def repair_dependency_dag():
    svg = SVG(980, 440)
    title(svg, "The order we fix things",
          "Mechanics first, then state, then evaluation, then one bounded training run")

    nodes = [
        ("P0", "Freeze data\n& reproduce E0", 70, 150, SLATE, SLATE_BG, SLATE_LN),
        ("P1", "Predictor positions\n+ stochastic masks", 285, 150, ORANGE, ORANGE_BG, ORANGE_LN),
        ("P2", "Remove label leak,\nsource-uniform, resume", 285, 280, BLUE, BLUE_BG, BLUE_LN),
        ("F", "Evaluation firewall\n(nested CV, OOF)", 540, 215, PURPLE, PURPLE_BG, PURPLE_LN),
        ("R1", "Bounded R1 run\n+ RF + controls", 780, 215, GREEN, GREEN_BG, GREEN_LN),
    ]
    box = {}
    cw, ch = 170, 74
    for tag, label, x, y, ac, bg, ln in nodes:
        lines = label.split("\n")
        svg.rect(x, y, cw, ch, r=13, fill=bg, stroke=ln, sw=1.6, shadow=True)
        svg.circle(x + 22, y + 20, 14, ac)
        svg.text(x + 22, y + 25, tag, size=11.5, fill="#fff", weight="700")
        svg.text(x + cw / 2 + 12, y + 30, lines[0], size=11.5, fill=ac, weight="700")
        if len(lines) > 1:
            svg.text(x + cw / 2 + 12, y + 48, lines[1], size=10.5, fill=MUTE)
        box[tag] = (x, y, cw, ch)

    def right(tag): x, y, w, h = box[tag]; return (x + w, y + h / 2)
    def left(tag): x, y, w, h = box[tag]; return (x, y + h / 2)

    # P0 -> P1 and P0 -> P2 (fan out)
    x0, y0 = right("P0")
    svg.path(f"M {x0} {y0} H {left('P1')[0]-24} V {left('P1')[1]} H {left('P1')[0]-2}",
             stroke=SLATE, arrow="arrowSlate", w=2)
    svg.path(f"M {x0} {y0} H {left('P2')[0]-24} V {left('P2')[1]} H {left('P2')[0]-2}",
             stroke=SLATE, arrow="arrowSlate", w=2)
    # P1 -> F and P2 -> F (fan in)
    svg.path(f"M {right('P1')[0]} {right('P1')[1]} H {left('F')[0]-24} "
             f"V {left('F')[1]-8} H {left('F')[0]-2}", stroke=ORANGE, arrow="arrowOrange", w=2)
    svg.path(f"M {right('P2')[0]} {right('P2')[1]} H {left('F')[0]-24} "
             f"V {left('F')[1]+8} H {left('F')[0]-2}", stroke=BLUE, arrow="arrowBlue", w=2)
    # F -> R1
    svg.path(f"M {right('F')[0]} {right('F')[1]} H {left('R1')[0]-2}",
             stroke=PURPLE, arrow="arrowPurple", w=2.2)

    svg.text(svg.w / 2, 400,
             "Freeze the R1 recipe on inner folds BEFORE the one development evaluation.",
             size=12, fill=MUTE)
    save(svg, "repair_dependency_dag.svg")


# ---------------------------------------------------------------------------
# 6. Evaluation firewall
# ---------------------------------------------------------------------------
def eval_firewall():
    svg = SVG(980, 500)
    title(svg, "The evaluation firewall",
          "Choose everything on inner folds; touch each outer fold exactly once")

    # outer fold strip
    svg.text(70, 100, "Outer folds (grouped by source)", size=12.5, fill=INK,
             weight="700", anchor="start")
    fold_w = 150
    for k in range(5):
        x = 70 + k * (fold_w + 12)
        is_test = (k == 2)
        bg = ORANGE_BG if is_test else BLUE_BG
        ln = ORANGE_LN if is_test else BLUE_LN
        ac = ORANGE if is_test else BLUE
        svg.rect(x, 112, fold_w, 48, r=10, fill=bg, stroke=ln, sw=1.4)
        svg.text(x + fold_w / 2, 134, f"fold {k}", size=12, fill=ac, weight="700")
        svg.text(x + fold_w / 2, 151, "TEST (once)" if is_test else "train", size=10.5, fill=MUTE)

    # inner selection box under the train folds
    svg.rect(70, 200, 470, 150, r=14, fill=PURPLE_BG, stroke=PURPLE_LN, sw=1.6, shadow=True)
    svg.text(305, 226, "Inner folds: pick everything here", size=13, fill=PURPLE, weight="700")
    for i, t in enumerate([
        "mask ratio, learning rate, update budget",
        "pooling, probe C, PCA dims",
        "checkpoint selection, EMA schedule",
        "SSL sees only inner-training sources (no labels)",
    ]):
        svg.dot(96, 250 + i * 22, 3.5, PURPLE)
        svg.text(108, 254 + i * 22, t, size=11.5, fill=INK, anchor="start")

    # refit + eval box
    svg.rect(580, 200, 330, 150, r=14, fill=GREEN_BG, stroke=GREEN_LN, sw=1.6, shadow=True)
    svg.text(745, 226, "Then, once per fold", size=13, fill=GREEN, weight="700")
    for i, t in enumerate([
        "refit frozen recipe on all outer-train",
        "predict the held-out fold -> save OOF probs",
        "one probability vector per source",
        "RF uses the SAME folds + aggregation",
    ]):
        svg.dot(606, 250 + i * 22, 3.5, GREEN)
        svg.text(618, 254 + i * 22, t, size=11.5, fill=INK, anchor="start")

    svg.path(f"M 540 275 H 578", stroke=SLATE, arrow="arrowSlate", w=2.2)

    # pooled OOF -> metrics
    svg.rect(300, 396, 380, 60, r=13, fill=SLATE_BG, stroke=SLATE_LN, sw=1.5, shadow=True)
    svg.text(490, 420, "Pool all OOF rows → macro-F1, PD recall,", size=12.5,
             fill=SLATE, weight="700")
    svg.text(490, 440, "paired source-bootstrap vs RF (development estimate)", size=11.5, fill=MUTE)
    svg.path("M 745 350 V 372 H 490 V 394", stroke=GREEN, arrow="arrowGreen", w=2)
    save(svg, "eval_firewall.svg")


# ---------------------------------------------------------------------------
# 7. R1 frozen recipe
# ---------------------------------------------------------------------------
def r1_config():
    svg = SVG(940, 470)
    title(svg, "The R1_repaired32 recipe (frozen before evaluation)",
          "Change as little as possible beyond the correctness fixes")

    groups = [
        ("Data", BLUE, BLUE_BG, BLUE_LN, [
            "legacy cache (cache_v0)", "32 frames, [x, y, visibility]",
            "all training sources, no labels"]),
        ("Model", ORANGE, ORANGE_BG, ORANGE_LN, [
            "encoder 3 layers x 96", "predictor + joint/time positions",
            "stochastic graph-time masks 0.60"]),
        ("Optim", GREEN, GREEN_BG, GREEN_LN, [
            "AdamW lr 3e-4, wd 0.04", "source-uniform sampler",
            "EMA 0.990 -> 0.9995 (by steps)"]),
        ("Readout", PURPLE, PURPLE_BG, PURPLE_LN, [
            "frozen encoder", "mean pooling (probe changes later)",
            "balanced logistic regression"]),
    ]
    cw, ch = 210, 200
    gap = 24
    x0 = 40
    y = 96
    for i, (t, ac, bg, ln, items) in enumerate(groups):
        x = x0 + i * (cw + gap)
        svg.rect(x, y, cw, ch, r=14, fill=bg, stroke=ln, sw=1.6, shadow=True)
        svg.text(x + cw / 2, y + 30, t, size=15, fill=ac, weight="700")
        svg.line(x + 18, y + 42, x + cw - 18, y + 42, stroke=ln, w=1.4)
        for k, it in enumerate(items):
            yy = y + 68 + k * 44
            svg.dot(x + 22, yy - 4, 3.5, ac)
            # wrap long items to two lines
            words = it.split()
            if len(it) > 24 and len(words) > 3:
                mid = len(words) // 2
                svg.text(x + 34, yy, " ".join(words[:mid]), size=11.5, fill=INK, anchor="start")
                svg.text(x + 34, yy + 15, " ".join(words[mid:]), size=11.5, fill=INK, anchor="start")
            else:
                svg.text(x + 34, yy, it, size=11.5, fill=INK, anchor="start")

    svg.rect(40, 330, 860, 0, r=0, fill="#fff")  # spacer no-op
    svg.text(svg.w / 2, 366,
             "Learning curve at 300 / 1,000 / 3,000 updates, chosen on inner folds only.",
             size=12.5, fill=INK)
    svg.text(svg.w / 2, 392,
             "If the repaired baseline does not beat E0 on inner folds, we stop scaling and "
             "report the negative result.",
             size=12, fill=MUTE)
    svg.text(svg.w / 2, 430,
             "This measures the repaired BUNDLE. It cannot attribute a gain to any single fix.",
             size=12, fill=RED, weight="600")
    save(svg, "r1_config.svg")


# ---------------------------------------------------------------------------
# 8. R1 results readout (the honest development estimate)
# ---------------------------------------------------------------------------
def r1_results():
    svg = SVG(940, 460)
    title(svg, "R1 result: an honest, lower number is progress",
          "Repaired S-JEPA scores below RF and below nuisance controls on this tiny cache")

    # horizontal bars, pooled macro-F1 on the g1 folds
    bars = [
        ("Random Forest (paired)", 0.667, GREEN),
        ("nuisance: pose mean+std", 0.694, SLATE),
        ("nuisance: visibility only", 0.602, SLATE),
        ("old broken S-JEPA", 0.570, MUTE),
        ("repaired S-JEPA (R1)", 0.438, ORANGE),
        ("chance (3 classes)", 0.333, HAIR),
    ]
    x0, y0 = 300, 96
    bw_max = 520
    bh = 34
    gap = 20
    for i, (label, val, col) in enumerate(bars):
        y = y0 + i * (bh + gap)
        svg.text(x0 - 16, y + bh / 2 + 4, label, size=12.5, fill=INK, anchor="end")
        svg.rect(x0, y, bw_max, bh, r=8, fill="#f8fafc", stroke=HAIR, sw=1)
        w = bw_max * val
        svg.rect(x0, y, w, bh, r=8, fill=col, stroke=None)
        svg.text(x0 + w + 10, y + bh / 2 + 4, f"{val:.3f}", size=12.5, fill=INK,
                 weight="700", anchor="start")

    svg.text(svg.w / 2, 430,
             "Pooled macro-F1 on the locked g1 folds (source-grouped, development estimate). "
             "PD recall: RF 0.59 vs S-JEPA 0.24.",
             size=11.5, fill=MUTE)
    save(svg, "r1_results.svg")


if __name__ == "__main__":
    fixes_overview()
    defect_predictor_positions()
    defect_mask_starvation()
    defect_domain_leakage()
    repair_dependency_dag()
    eval_firewall()
    r1_config()
    r1_results()
    print("\nAll fixes diagrams written.")
