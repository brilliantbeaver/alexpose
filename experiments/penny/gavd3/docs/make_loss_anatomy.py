"""Generate the self-contained vector anatomy diagram for the loss tutorial."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent / "figures" / "urtc_loss_anatomy.svg"

INK = "#10233f"
LINE = "#34495e"
MUTED = "#526575"


def rounded_box(ax, x, y, width, height, color):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=LINE,
        facecolor=color,
    )
    ax.add_patch(patch)
    return patch


def horizontal_arrow(ax, x1, x2, y):
    arrow = FancyArrowPatch(
        (x1 + 0.004, y),
        (x2 - 0.004, y),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.5,
        color=LINE,
        shrinkA=0,
        shrinkB=0,
        zorder=1,
    )
    arrow.set_capstyle("round")
    arrow.set_joinstyle("round")
    ax.add_patch(arrow)


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
    arrow = FancyArrowPatch(
        path=MplPath(vertices, codes),
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=1.5,
        color=LINE,
        zorder=1,
    )
    arrow.set_capstyle("round")
    arrow.set_joinstyle("round")
    ax.add_patch(arrow)


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
    ax.text(
        x + width / 2,
        y + height * 0.64,
        heading,
        ha="center",
        va="center",
        fontsize=heading_size or (15 if math_heading else 12.5),
        fontweight=None if math_heading else "bold",
        color=INK,
    )
    ax.text(
        x + width / 2,
        y + height * 0.27,
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
        "Anatomy of the loss for one hidden token",
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
    label_box(ax, student_boxes[0], r"$\mathbf{p}$", "predicted feature", True)
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
    loss_box = (0.25, 0.42, 0.50, 0.19)
    rounded_box(ax, *loss_box, "#fde9d9")
    ax.text(
        0.5,
        0.555,
        r"$\mathcal{L}=-\sum_{d=1}^{D}q_d\log r_d$",
        ha="center",
        va="center",
        fontsize=18,
        color=INK,
    )
    ax.text(
        0.5,
        0.475,
        "Teacher weights × student log-probabilities",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK,
    )
    ax.text(
        0.5,
        0.445,
        r"$D=96$; mean over hidden targets and batch items",
        ha="center",
        va="center",
        fontsize=9.0,
        color=MUTED,
    )

    # The student enters through the top of the formula box.
    routed_input(ax, (0.82, student_y), 0.665, (0.68, loss_box[1] + loss_box[3]))

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
    routed_input(ax, (0.82, teacher_y + flow_h), 0.35, (0.68, loss_box[1]))

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
            "Title": "Anatomy of the URTC latent cross-entropy loss",
            "Description": (
                "The student prediction flow is above the cross-entropy formula. "
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
