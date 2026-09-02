"""Generate the self-contained vector training-flow diagram for the loss tutorial."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_style import ARROW_DASHES, arrow_style  # noqa: E402


OUTPUT = Path(__file__).resolve().parent / "figures" / "sjepa_loss_flow.svg"

INK = "#10233f"
MUTED = "#526575"
EMA = "#9a6700"
# The box outline follows the shared connector colour, so an edge and the line leaving it read as
# one drawing rather than two greys.
LINE = arrow_style()["color"]
# The visual edge of a box sits this far outside the rectangle it is given.
BOX_PAD = 0.008
# Connectors here are handed the drawn box edge rather than the nominal rectangle, so they need
# less shrink than the shared default, which budgets for a card pad it would otherwise absorb.
CONNECTOR = arrow_style(shrinkA=3, shrinkB=6, zorder=3)
# Clear space for the routed elbows, which cannot use shrink.
TAIL_CLEARANCE = 0.010
TIP_CLEARANCE = 0.016


def rounded_box(ax, box, color, linewidth=1.4):
    x, y, width, height = box
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad={BOX_PAD},rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=LINE,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label_box(ax, box, title, lines, title_size=11.5, body_size=8.8):
    """Draw a card title with its body lines as one block below it.

    The body used to be one Text artist per line at hardcoded fractions of the box height, which
    put adjacent baselines within half a pixel of each other in the three-line cards. Drawing the
    block as a single multi-line Text hands the leading to matplotlib, which cannot self-collide.
    """
    x, y, width, height = box
    center_x = x + width / 2
    # Points to axes units on the vertical, so the stack holds its proportions at any figure size.
    axes_height_points = ax.figure.get_figheight() * 72.0 * ax.get_position().height
    title_block = title_size * 1.30 / axes_height_points
    gap = title_size * 0.72 / axes_height_points
    body_block = len(lines) * body_size * 1.55 / axes_height_points
    # The title and body are centred as one stack, so a card with three body lines stays inside
    # its box and a card with one stays visually balanced, without per-case magic fractions.
    stack_top = y + height / 2 + (title_block + gap + body_block) / 2
    ax.text(
        center_x,
        stack_top,
        title,
        ha="center",
        va="top",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    ax.text(
        center_x,
        stack_top - title_block - gap,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=body_size,
        color=MUTED,
        linespacing=1.55,
        zorder=3,
    )


def arrow(ax, start, end, color=None, dashed=False):
    """Connect two boxes, given the facing points on their drawn edges."""
    style = dict(CONNECTOR)
    if color is not None:
        style["color"] = color
    if dashed:
        style["linestyle"] = ARROW_DASHES
    ax.add_patch(FancyArrowPatch(start, end, **style))


def rounded_route(ax, vertices, color=LINE, dashed=False):
    """Draw a rounded orthogonal route ending in one consistent arrowhead."""
    start, corner1, corner2, end = vertices
    sx, sy = start
    c1x, c1y = corner1
    c2x, c2y = corner2
    ex, ey = end
    # Keep the corner radius smaller than the short final segment into Loss;
    # otherwise the last path segment can reverse and flip its arrowhead.
    radius = 0.006

    # These routes are horizontal, then vertical, then horizontal.
    first_sign = 1 if c1x > sx else -1
    vertical_sign = 1 if c2y > c1y else -1
    final_sign = 1 if ex > c2x else -1
    path_vertices = [
        (sx, sy),
        (c1x - first_sign * radius, sy),
        (c1x, sy),
        (c1x, sy + vertical_sign * radius),
        (c2x, c2y - vertical_sign * radius),
        (c2x, c2y),
        (c2x + final_sign * radius, c2y),
        (ex, ey),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
    ]
    # A path-routed arrow cannot use shrink, so the caller supplies endpoints that already hold
    # the clear space. Everything else is the shared vocabulary.
    style = {key: value for key, value in CONNECTOR.items() if not key.startswith("shrink")}
    style["color"] = color
    if dashed:
        style["linestyle"] = ARROW_DASHES
    ax.add_patch(FancyArrowPatch(path=MplPath(path_vertices, codes), **style))


def make_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "path",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 6.8))
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "One S-JEPA training step",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )

    ax.text(
        0.26,
        0.875,
        "STUDENT PATH  ·  transformed view with hidden targets",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#6941a5",
    )
    input_box = (0.03, 0.415, 0.14, 0.18)
    loss_box = (0.882, 0.415, 0.108, 0.18)
    # Taller lane boxes. At 0.16 the three-line student and teacher view cards did not fit their
    # own boxes, and every card's title sat within two pixels of its top edge.
    student_y = 0.660
    teacher_y = 0.160
    flow_h = 0.185
    # The row is pulled leftward and the output column narrowed to open a clear vertical channel
    # between the output boxes and the loss box. There used to be 8 px of space there, so the two
    # elbow routes into the loss box ran along the right border of the boxes they left.
    view_x, encoder_x, output_x = 0.25, 0.479, 0.6645
    # The view boxes are the widest copy in the figure, so they get the widest box.
    view_w, encoder_w, output_w = 0.185, 0.1415, 0.1415
    # Where the two elbow routes turn, in the middle of that channel.
    channel_x = 0.845

    student_view = (view_x, student_y, view_w, flow_h)
    student_encoder = (encoder_x, student_y, encoder_w, flow_h)
    predictor = (output_x, student_y, output_w, flow_h)
    teacher_view = (view_x, teacher_y, view_w, flow_h)
    target_encoder = (encoder_x, teacher_y, encoder_w, flow_h)
    teacher_target = (output_x, teacher_y, output_w, flow_h)

    rounded_box(ax, input_box, "#e8f1fa")
    rounded_box(ax, student_view, "#fff2cc")
    rounded_box(ax, student_encoder, "#fce4d6")
    rounded_box(ax, predictor, "#eadcf8")
    rounded_box(ax, teacher_view, "#e2f0d9")
    rounded_box(ax, target_encoder, "#d9ead3")
    rounded_box(ax, teacher_target, "#d9ead3")
    rounded_box(ax, loss_box, "#fde9d9")

    label_box(ax, input_box, "Gait sequence", ["64 frames × 33 joints", "528 tokens"])
    label_box(
        ax,
        student_view,
        "Student view",
        [
            "Small rotation + translation",
            "Shared target count: 60% of",
            "the batch's smallest eligible set",
        ],
        # The longest line in the figure. At 8.3 it was wider than its own box.
        body_size=7.9,
    )
    label_box(
        ax,
        student_encoder,
        "View encoder",
        ["Encodes visible context", "Updated by gradients"],
        body_size=8.0,
    )
    label_box(
        ax,
        predictor,
        "Predictor",
        ["Guesses hidden tokens", r"$\mathbf{p}$ · 96 values"],
        body_size=8.0,
    )
    label_box(
        ax,
        teacher_view,
        "Teacher view",
        ["Original complete sequence", "Nothing is hidden", "No gradient"],
        body_size=8.0,
    )
    label_box(
        ax,
        target_encoder,
        "Target encoder",
        ["Encodes full context", "Updated slowly by EMA"],
        body_size=8.0,
    )
    label_box(
        ax,
        teacher_target,
        "Teacher target",
        ["Hidden-token feature", r"$\mathbf{z}_t$ · 96 values"],
        body_size=8.0,
    )
    label_box(
        ax,
        loss_box,
        "JEPA sub-loss",
        ["Latent cross-entropy", "Teacher vs. guess"],
        # Sized to the narrowest box in the figure: at 10.5 the heading was wider than it.
        title_size=9.5,
        body_size=7.5,
    )

    # One tidy junction splits the input into the two parallel paths.
    branch_x = 0.215
    input_center_y = input_box[1] + input_box[3] / 2
    student_center_y = student_y + flow_h / 2
    teacher_center_y = teacher_y + flow_h / 2
    ax.plot(
        [input_box[0] + input_box[2], branch_x],
        [input_center_y, input_center_y],
        color=LINE,
        linewidth=1.5,
        solid_capstyle="round",
        zorder=1,
    )
    ax.plot(
        [branch_x, branch_x],
        [teacher_center_y, student_center_y],
        color=LINE,
        linewidth=1.5,
        solid_capstyle="round",
        zorder=1,
    )
    ax.add_patch(Circle((branch_x, input_center_y), 0.0045, color=LINE, zorder=2))
    # Every endpoint is a drawn box edge, which is BOX_PAD outside the nominal rectangle. The
    # previous hand-tuned nudges of 0.004 and 0.010 were smaller than that pad, so the arrowheads
    # were landing just inside the box borders they were pointing at.
    arrow(ax, (branch_x, student_center_y), (view_x - BOX_PAD, student_center_y))
    arrow(ax, (branch_x, teacher_center_y), (view_x - BOX_PAD, teacher_center_y))

    # Straight arrows keep each lane easy to scan.
    for lane_y in (student_center_y, teacher_center_y):
        arrow(ax, (view_x + view_w + BOX_PAD, lane_y), (encoder_x - BOX_PAD, lane_y))
        arrow(ax, (encoder_x + encoder_w + BOX_PAD, lane_y), (output_x - BOX_PAD, lane_y))

    # Student and teacher enter distinct points on the loss box.
    for lane_y, entry_y in ((student_center_y, 0.55), (teacher_center_y, 0.46)):
        rounded_route(
            ax,
            [
                (output_x + output_w + BOX_PAD + TAIL_CLEARANCE, lane_y),
                (channel_x, lane_y),
                (channel_x, entry_y),
                (loss_box[0] - BOX_PAD - TIP_CLEARANCE, entry_y),
            ],
        )

    # A distinct dashed arrow communicates parameter update, not data flow.
    encoder_center_x = encoder_x + encoder_w / 2
    arrow(
        ax,
        (encoder_center_x, student_y - BOX_PAD),
        (encoder_center_x, teacher_y + flow_h + BOX_PAD),
        color=EMA,
        dashed=True,
    )
    ax.text(
        encoder_center_x + 0.014,
        0.505,
        "EMA update",
        ha="left",
        va="center",
        fontsize=8.8,
        fontweight="bold",
        color=EMA,
    )

    ax.text(
        0.26,
        0.105,
        "TEACHER PATH  ·  complete sequence with no gradient",
        ha="left",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color="#3d7a45",
    )

    footer = (0.07, 0.020, 0.86, 0.050)
    rounded_box(ax, footer, "#eef3f8", linewidth=0.8)
    ax.text(
        0.5,
        footer[1] + footer[3] / 2,
        "Full training also adds VICReg; post-normal stages add a label-aware group-margin term.",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
        zorder=3,
    )

    fig.savefig(
        OUTPUT,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={
            "Title": "S-JEPA training flow",
            "Description": (
                "A gait sequence splits at one junction into a masked student path and "
                "a complete teacher path. Both paths feed the JEPA latent cross-entropy sub-loss."
            ),
        },
    )
    plt.close(fig)
    svg = OUTPUT.read_text(encoding="utf-8")
    OUTPUT.write_text(
        "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    make_figure()
