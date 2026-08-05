"""Generate the self-contained vector training-flow diagram for the loss tutorial."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


OUTPUT = Path(__file__).resolve().parent / "figures" / "urtc_loss_flow.svg"

INK = "#10233f"
LINE = "#40566d"
MUTED = "#526575"
EMA = "#9a6700"


def rounded_box(ax, box, color, linewidth=1.4):
    x, y, width, height = box
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=linewidth,
        edgecolor=LINE,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def label_box(ax, box, title, lines, title_size=12.5, body_size=8.8):
    x, y, width, height = box
    center_x = x + width / 2
    ax.text(
        center_x,
        y + height * 0.72,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=INK,
        zorder=3,
    )
    if len(lines) == 1:
        positions = [0.34]
    elif len(lines) == 2:
        positions = [0.43, 0.23]
    else:
        positions = [0.49, 0.31, 0.15]
    for position, line in zip(positions, lines):
        ax.text(
            center_x,
            y + height * position,
            line,
            ha="center",
            va="center",
            fontsize=body_size,
            color=MUTED,
            zorder=3,
        )


def arrow(ax, start, end, color=LINE, dashed=False):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12.5,
        linewidth=1.5,
        linestyle=(0, (4, 4)) if dashed else "solid",
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=3,
    )
    patch.set_capstyle("round")
    patch.set_joinstyle("round")
    ax.add_patch(patch)


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
    patch = FancyArrowPatch(
        path=MplPath(path_vertices, codes),
        arrowstyle="-|>",
        mutation_scale=12.5,
        linewidth=1.5,
        linestyle=(0, (4, 4)) if dashed else "solid",
        color=color,
        zorder=3,
    )
    patch.set_capstyle("round")
    patch.set_joinstyle("round")
    ax.add_patch(patch)


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
    loss_box = (0.89, 0.415, 0.09, 0.18)
    student_y = 0.675
    teacher_y = 0.155
    flow_h = 0.16
    view_x, encoder_x, output_x = 0.26, 0.49, 0.69
    view_w, encoder_w, output_w = 0.18, 0.15, 0.16

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
        ["Small rotation + translation", "60% eligible targets hidden", "Visible tokens only"],
        body_size=8.3,
    )
    label_box(
        ax,
        student_encoder,
        "View encoder",
        ["Encodes visible context", "Updated by gradients"],
        body_size=8.3,
    )
    label_box(
        ax,
        predictor,
        "Predictor",
        ["Guesses hidden tokens", r"$\mathbf{p}$ · 96 values"],
        body_size=8.3,
    )
    label_box(
        ax,
        teacher_view,
        "Teacher view",
        ["Original complete sequence", "Nothing is hidden", "No gradient"],
        body_size=8.3,
    )
    label_box(
        ax,
        target_encoder,
        "Target encoder",
        ["Encodes full context", "Updated slowly by EMA"],
        body_size=8.3,
    )
    label_box(
        ax,
        teacher_target,
        "Teacher target",
        ["Hidden-token feature", r"$\mathbf{z}_t$ · 96 values"],
        body_size=8.3,
    )
    label_box(ax, loss_box, "Loss", ["Cross-entropy", "Teacher vs. guess"], body_size=8.0)

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
    arrow(ax, (branch_x, student_center_y), (view_x - 0.010, student_center_y))
    arrow(ax, (branch_x, teacher_center_y), (view_x - 0.010, teacher_center_y))

    # Straight arrows keep each lane easy to scan.
    arrow(ax, (view_x + view_w + 0.004, student_center_y), (encoder_x - 0.010, student_center_y))
    arrow(ax, (encoder_x + encoder_w + 0.004, student_center_y), (output_x - 0.010, student_center_y))
    arrow(ax, (view_x + view_w + 0.004, teacher_center_y), (encoder_x - 0.010, teacher_center_y))
    arrow(ax, (encoder_x + encoder_w + 0.004, teacher_center_y), (output_x - 0.010, teacher_center_y))

    # Student and teacher enter distinct points on the loss box.
    rounded_route(
        ax,
        [
            (output_x + output_w, student_center_y),
            (0.87, student_center_y),
            (0.87, 0.55),
            (loss_box[0] - 0.010, 0.55),
        ],
    )
    rounded_route(
        ax,
        [
            (output_x + output_w, teacher_center_y),
            (0.87, teacher_center_y),
            (0.87, 0.46),
            (loss_box[0] - 0.010, 0.46),
        ],
    )

    # A distinct dashed arrow communicates parameter update, not data flow.
    encoder_center_x = encoder_x + encoder_w / 2
    arrow(
        ax,
        (encoder_center_x, student_y - 0.004),
        (encoder_center_x, teacher_y + flow_h + 0.004),
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
        "Goal: infer a useful description of hidden motion—not reconstruct exact coordinates.",
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
            "Title": "URTC S-JEPA training flow",
            "Description": (
                "A gait sequence splits at one junction into a masked student path and "
                "a complete teacher path. Both paths feed a latent cross-entropy loss."
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
