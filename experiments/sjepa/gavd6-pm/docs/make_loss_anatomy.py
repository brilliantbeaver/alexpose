"""Generate the self-contained vector anatomy diagram for the loss tutorial."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagram_style import arrow_style  # noqa: E402


OUTPUT = Path(__file__).resolve().parent / "figures" / "sjepa_loss_anatomy.svg"

INK = "#10233f"
MUTED = "#526575"
# The box outline follows the shared connector colour, so an edge and the line leaving it read as
# one drawing rather than two greys.
LINE = arrow_style()["color"]
# The visual edge of a box sits this far outside the rectangle it is given.
BOX_PAD = 0.008
# Connectors here are handed the drawn box edge rather than the nominal rectangle, so they need
# less shrink than the shared default, which budgets for a card pad it would otherwise absorb.
CONNECTOR = arrow_style(shrinkA=3, shrinkB=6, zorder=1)
# Clear space for the routed elbows, which cannot use shrink. In axes units on the vertical,
# where this figure's axes is 585 px tall, these are about 6 px and 10 px.
TAIL_CLEARANCE = 0.010
TIP_CLEARANCE = 0.017


def rounded_box(ax, x, y, width, height, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad={BOX_PAD},rounding_size=0.018",
        linewidth=1.4,
        edgecolor=LINE,
        facecolor=color,
    )
    ax.add_patch(patch)
    return patch


def horizontal_arrow(ax, x1, x2, y):
    """Connect two boxes whose facing nominal edges are at x1 and x2."""
    ax.add_patch(FancyArrowPatch((x1 + BOX_PAD, y), (x2 - BOX_PAD, y), **CONNECTOR))


def routed_input(ax, start, bend_y, end):
    """Draw one clean, rounded elbow path into the loss panel."""
    sx, sy = start
    ex, ey = end
    radius = 0.012
    vertical_sign = 1 if bend_y > sy else -1
    horizontal_sign = 1 if ex > sx else -1
    final_sign = 1 if ey > bend_y else -1
    vertices = [
        (sx, sy),
        (sx, bend_y - vertical_sign * radius),
        (sx, bend_y),
        (sx + horizontal_sign * radius, bend_y),
        (ex - horizontal_sign * radius, bend_y),
        (ex, bend_y),
        (ex, bend_y + final_sign * radius),
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
    routed = {key: value for key, value in CONNECTOR.items() if not key.startswith("shrink")}
    ax.add_patch(FancyArrowPatch(path=MplPath(vertices, codes), **routed))


def label_box(
    ax,
    box,
    heading,
    detail,
    math_heading=False,
    heading_size=None,
    detail_size=9.0,
):
    x, y, width, height = box
    # The heading sits above the box centre and the detail below it, so the pair stays balanced
    # inside the box at any of the heading sizes these panels use.
    ax.text(
        x + width / 2,
        y + height * 0.68,
        heading,
        ha="center",
        va="center",
        fontsize=heading_size or (15 if math_heading else 12.5),
        fontweight=None if math_heading else "bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.24,
        detail,
        ha="center",
        va="center",
        fontsize=detail_size,
        color=MUTED,
    )


def make_figure():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            # Convert math and labels to outlines so the SVG is self-contained
            # and displays identically without depending on installed fonts.
            "svg.fonttype": "path",
        }
    )
    fig, ax = plt.subplots(figsize=(12, 7.6))
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.955,
        "Anatomy of the JEPA sub-loss for one hidden token",
        ha="center",
        va="center",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )

    # Student flow: top row.
    ax.text(
        0.055,
        0.875,
        "1. Student makes the predicted distribution",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )
    student_y, flow_h = 0.735, 0.105
    student_boxes = [
        (0.04, student_y, 0.11, flow_h),
        (0.20, student_y, 0.12, flow_h),
        (0.37, student_y, 0.13, flow_h),
        (0.56, student_y, 0.40, flow_h),
    ]
    for box, color in zip(
        student_boxes, ["#eadcf8", "#fce4d6", "#e8f1fa", "#eadcf8"]
    ):
        rounded_box(ax, *box, color)
    label_box(
        ax,
        student_boxes[0],
        r"$\mathbf{p}$",
        "predicted feature",
        True,
        detail_size=8.6,
    )
    label_box(ax, student_boxes[1], r"$\div\ \tau_p$", r"$\tau_p=0.10$", True)
    label_box(ax, student_boxes[2], "softmax", "positive; sums to 1")
    label_box(
        ax,
        student_boxes[3],
        r"$\mathbf{r}=\operatorname{softmax}(\mathbf{p}/\tau_p)$",
        "student emphasis across 96 dimensions",
        True,
        heading_size=14.5,
        detail_size=8.8,
    )
    horizontal_arrow(ax, 0.15, 0.20, student_y + flow_h / 2)
    horizontal_arrow(ax, 0.32, 0.37, student_y + flow_h / 2)
    horizontal_arrow(ax, 0.50, 0.56, student_y + flow_h / 2)

    # Main cross-entropy formula: middle row.
    # Slightly taller than it was, because the formula's summation limits reach above the line
    # box matplotlib reports for a 18 point glyph run and were touching the box edge.
    loss_box = (0.25, 0.412, 0.50, 0.205)
    rounded_box(ax, *loss_box, "#fde9d9")
    ax.text(
        0.5,
        0.556,
        r"$\mathcal{L}_{\mathrm{JEPA}}=-\sum_{d=1}^{D}q_d\log r_d$",
        ha="center",
        va="center",
        fontsize=18,
        color=INK,
    )
    ax.text(
        0.5,
        0.476,
        "Teacher weights × student log-probabilities",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    # Dropped from 0.445. At the earlier spacing these two lines were 4 px apart, which is less
    # than the gap between the words inside either of them.
    ax.text(
        0.5,
        0.437,
        r"$D=96$; mean over hidden targets and batch items",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )

    # The student enters through the top of the formula box. A routed path cannot shrink itself,
    # so both ends stand off the drawn box edge by the box pad plus the clear space.
    routed_input(
        ax,
        (0.82, student_y - BOX_PAD - TAIL_CLEARANCE),
        0.665,
        (0.68, loss_box[1] + loss_box[3] + BOX_PAD + TIP_CLEARANCE),
    )

    # Teacher flow: below the formula, as requested.
    teacher_y = 0.19
    teacher_boxes = [
        (0.04, teacher_y, 0.11, flow_h),
        (0.20, teacher_y, 0.12, flow_h),
        (0.37, teacher_y, 0.13, flow_h),
        (0.56, teacher_y, 0.40, flow_h),
    ]
    for box, color in zip(
        teacher_boxes, ["#d9ead3", "#fff2cc", "#fce4d6", "#d9ead3"]
    ):
        rounded_box(ax, *box, color)
    label_box(ax, teacher_boxes[0], r"$\mathbf{z}_t$", "teacher feature", True)
    label_box(ax, teacher_boxes[1], r"$-\mathbf{c}$", "running center", True)
    label_box(ax, teacher_boxes[2], r"$\div\ \tau_t$", r"$\tau_t=0.06$", True)
    label_box(
        ax,
        teacher_boxes[3],
        r"$\mathbf{q}=\operatorname{softmax}((\mathbf{z}_t-\mathbf{c})/\tau_t)$",
        "teacher importance across 96 dimensions",
        True,
        heading_size=13.5,
        detail_size=8.8,
    )
    horizontal_arrow(ax, 0.15, 0.20, teacher_y + flow_h / 2)
    horizontal_arrow(ax, 0.32, 0.37, teacher_y + flow_h / 2)
    horizontal_arrow(ax, 0.50, 0.56, teacher_y + flow_h / 2)

    # The teacher enters through the bottom of the formula box.
    routed_input(
        ax,
        (0.82, teacher_y + flow_h + BOX_PAD + TAIL_CLEARANCE),
        0.35,
        (0.68, loss_box[1] - BOX_PAD - TIP_CLEARANCE),
    )

    # Put the teacher caption beneath its flow so the row itself stays uncluttered.
    ax.text(
        0.055,
        0.105,
        "2. Teacher makes the centered, sharpened target distribution",
        ha="left",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=INK,
    )

    fig.savefig(
        OUTPUT,
        format="svg",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={
            "Title": "Anatomy of the JEPA latent cross-entropy sub-loss",
            "Description": (
                "The student prediction flow is above the JEPA cross-entropy sub-loss. "
                "The centered and sharpened teacher flow is below it."
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
