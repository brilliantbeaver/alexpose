"""Build vector figures for the reflection-symmetry investigations, ideas 5 and 9.

Idea 5 asks whether the frozen representation carries signed left-minus-right gait
information. Idea 9 asks the same question through an antisymmetric readout head
(arm 1) and then whether training with a reflection-equivariance term changes the
encoder at all (arm 2).

Both idea 5 and idea 9 arm 1 returned negative verdicts, so these figures are built
to make a negative result legible rather than to dress it up. Every lane, control,
and gate is drawn, because with a negative result the controls are the finding.

Importing make_figures runs its strict artifact validation, which guarantees that
everything drawn here comes from one coherent checkpoint lineage.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from make_figures import ARTIFACTS, COLORS, FINGERPRINT, save

ARM2 = ARTIFACTS / "idea9_arm2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"Symmetry figure contract failed: {message}")


def load(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def one(matches) -> dict:
    """Return the single match, so a renamed bundle field fails loudly instead of silently."""
    found = list(matches)
    require(len(found) == 1, f"expected exactly one matching row, found {len(found)}")
    return found[0]


IDEA5 = load(ARTIFACTS / "idea5_signed_laterality_result.json")
IDEA9A = load(ARTIFACTS / "idea9_antisymmetric_readout_result.json")
MECHANISM = load(ARM2 / "idea9_arm2_mechanism_validation.json")

for name, bundle in (("idea 5", IDEA5), ("idea 9 arm 1", IDEA9A)):
    require(bundle["mode"] == "real", f"{name} bundle is not a real-mode result")
    require(
        bundle["fingerprint"] == FINGERPRINT,
        f"{name} bundle describes a different checkpoint than the current contract",
    )

# The mechanism notebook runs on synthetic fixtures on purpose. Label it that way so its
# large rho values are never mistaken for gait measurements.
require(
    MECHANISM["mode"].startswith("smoke"),
    "mechanism validation should be a synthetic-fixture result",
)


def load_real_ladder() -> dict[str, list[dict]]:
    """Count completed real rungs per arm, so the caption can state the seed coverage.

    A rung that did not finish the full curriculum must not be counted with ones that did,
    because a partial rung averaged into a control mean produces a confident wrong answer.
    """
    rungs: dict[str, list[dict]] = {}
    for path in sorted(ARM2.glob("idea9_arm2_[DE][01]_seed*.json")):
        rung = json.loads(path.read_text(encoding="utf-8"))
        if rung.get("mode") != "real":
            continue
        updates = [stage.get("optimizer_updates") for stage in rung.get("completed_stages", [])]
        if updates != [5700, 1425, 1425, 1425, 1425]:
            continue
        rungs.setdefault(rung["rung"], []).append(rung)
    return rungs


LADDER = load_real_ladder()


def lane_panel(ax, lanes: list[tuple[str, float, str]], reference: float, reference_label: str) -> None:
    names = [name for name, _, _ in lanes]
    values = [value for _, value, _ in lanes]
    colors = [color for _, _, color in lanes]
    y = np.arange(len(lanes))[::-1]
    ax.barh(y, values, color=colors, height=0.62, edgecolor=COLORS["navy"], linewidth=0.5)
    ax.axvline(0, color=COLORS["ink"], linewidth=0.8)
    ax.axvline(reference, color=COLORS["red"], linestyle="--", linewidth=1.0)

    low = min(values) - 0.30
    high = max(values) + 0.26
    ax.set_xlim(low, high)
    # Keep a blank strip above the top bar so the reference line's label has somewhere
    # to sit that is neither on a bar, on the frame, nor on top of the lane names.
    ax.set_ylim(-0.62, len(lanes) - 0.02)

    # Anchor the reference label on whichever side of its line has more room, so a long
    # label near the right edge folds inward instead of running off the panel.
    span = high - low
    opens_right = (high - reference) > (reference - low)
    ax.text(
        reference + (0.015 if opens_right else -0.015) * span,
        len(lanes) - 0.36,
        f"{reference_label} {reference:+.3f}",
        color=COLORS["red"],
        fontsize=5.9,
        ha="left" if opens_right else "right",
        va="center",
        bbox={"boxstyle": "round,pad=0.20", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
    )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.grid(axis="x", alpha=0.24)
    for yi, value in zip(y, values):
        # Negative bars grow toward the lane names, so their labels go on the empty
        # right-hand side of zero instead of on top of the tick text.
        ax.text(
            value + 0.02 * span if value >= 0 else 0.02 * span,
            yi,
            f"{value:+.3f}",
            va="center",
            ha="left",
            fontsize=6.2,
            color=COLORS["ink"],
        )


def make_lane_ladder() -> None:
    """Draw both probes' lanes so the controls sit beside the treatment lanes."""
    # Taller than it was: the three-line right-hand title, the shared axis label, and the
    # two-line footer each need their own band rather than sharing one. The extra height also
    # opens the baseline gap between the right panel's six lane names.
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.9))

    five = IDEA5["lanes"]
    lane_panel(
        axes[0],
        [
            ("B raw null\n(sanity: coordinates)", five["B_raw_null"]["r2"], COLORS["green"]),
            ("A learned\n(frozen S-JEPA)", five["A_learned"]["r2"], COLORS["blue"]),
            ("C floor\n(no side information)", five["C_floor"]["r2"], COLORS["muted"]),
            ("D pooled control", five["D_pooled"]["r2"], COLORS["gold"]),
        ],
        five["C_floor"]["r2"],
        "floor",
    )
    axes[0].set_title(
        f"Idea 5 signed laterality: {IDEA5['verdict']['PRIMARY_VERDICT']}",
        fontsize=8.2,
        color=COLORS["navy"],
        fontweight="bold",
    )

    nine = IDEA9A["lanes"]
    lane_panel(
        axes[1],
        [
            ("B raw null\n(sanity: coordinates)", nine["B_raw_null"]["r2"], COLORS["green"]),
            ("A' antisymmetric head", nine["A_prime"]["r2"], COLORS["purple"]),
            ("Ac capacity matched", nine["Ac_capacity_matched"]["r2"], COLORS["teal"]),
            ("C floor", nine["C_floor"]["r2"], COLORS["muted"]),
            ("D standard readout", nine["D_standard"]["r2"], COLORS["blue"]),
            ("E side-agnostic control", nine["E_pooled"]["r2"], COLORS["orange"]),
        ],
        IDEA9A["binding_bar_max_D_C"],
        "binding bar max(D, C)",
    )
    # Wrapped to three short lines. On two lines this title was wider than the whole figure and
    # only stayed visible because save() crops to a tight bounding box.
    axes[1].set_title(
        "Idea 9 arm 1: ARTIFACT\n(a side-agnostic control beat\nthe antisymmetric lane)",
        fontsize=8.2,
        color=COLORS["navy"],
        fontweight="bold",
    )

    fig.suptitle(
        "Reflection-symmetry probes: every treatment lane lost to a control",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["navy"],
    )
    # Explicit y. Left to itself the shared label lands in the same band as the note below it.
    fig.supxlabel(
        "source-disjoint $R^2$ (higher is better; 0 means no better than predicting the fold mean)",
        y=0.105,
        fontsize=6.8,
        color=COLORS["ink"],
    )
    # Drawn at a positive figure coordinate inside a reserved band. This note used to sit at
    # y=-0.10, outside the canvas, and survived only because save() crops to a tight box.
    fig.text(
        0.5,
        0.018,
        "Lane B recovers the target from raw coordinates almost perfectly, so the target and the\n"
        "pipeline are sound. Both learned lanes still fall below their own floors.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["muted"],
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.16, 1, 0.94])
    save(fig, "symmetry_lane_ladder")


def make_gate_table() -> None:
    """Show every preregistered gate and how it resolved."""
    verdict5 = IDEA5["verdict"]
    verdict9 = IDEA9A["verdict"]
    rows = [
        ("Idea 5", "beats floor by 0.05", verdict5["beats_floor_by_0.05"]),
        ("Idea 5", "reaches 80% of raw null", verdict5["reaches_80pct_of_null"]),
        ("Idea 5", "correct sign on 75% of held-out sources", verdict5["sign_consistent_75pct"]),
        ("Idea 5", "pooled control behaves", verdict5["D_control_ok"]),
        ("Idea 9", "beats binding bar by 0.05", verdict9["beats_binding_bar_by_0.05"]),
        ("Idea 9", "beats capacity-matched by 0.05", verdict9["beats_capacity_matched_by_0.05"]),
        ("Idea 9", "better than permuted labels", verdict9["A_prime_permutation_ok"]),
        ("Idea 9", "side-agnostic lane stays weak", verdict9["E_control_ok_abs"]),
        ("Idea 9", "target varies between sources", verdict9["y_quality_gate_ok"]),
        ("Idea 9", "head wiring verified", verdict9["wiring_identity_ok"]),
    ]
    passed = sum(1 for _, _, ok in rows if ok)
    fig, ax = plt.subplots(figsize=(7.1, 2.9))
    fig.subplots_adjust(left=0.008, right=0.992, bottom=0.10, top=0.99)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.02,
        0.985,
        "Preregistered gates, resolved",
        fontsize=10,
        fontweight="bold",
        color=COLORS["navy"],
        va="top",
    )
    # Wrapped to two lines and dropped clear of the heading. On one line it was wider than the
    # axes, and at y=0.92 its bounding box abutted the heading's with no baseline gap at all.
    ax.text(
        0.02,
        0.885,
        f"{passed} of {len(rows)} gates passed, and every one of those is a wiring or control check\n"
        "rather than a result. That is what makes the negative verdicts trustworthy rather than a bug.",
        fontsize=6.4,
        color=COLORS["muted"],
        va="top",
        linespacing=1.5,
    )
    # Rows are spaced from the row count and the band they have to fill, so adding or removing a
    # gate cannot silently push the last row off the axes.
    band_top, band_bottom = 0.73, 0.05
    step = (band_top - band_bottom) / (len(rows) - 1)
    for index, (probe, label, ok) in enumerate(rows):
        y = band_top - index * step
        ax.text(0.03, y, probe, fontsize=6.6, color=COLORS["ink"], va="center")
        ax.text(0.13, y, label, fontsize=6.6, color=COLORS["ink"], va="center")
        ax.text(
            0.86,
            y,
            "PASSED" if ok else "not met",
            fontsize=6.6,
            fontweight="bold",
            color=COLORS["green"] if ok else COLORS["red"],
            va="center",
            ha="center",
        )
    fig.text(
        0.5,
        0.012,
        f"Only {IDEA9A['y_variance']['between_source_fraction'] * 100:.1f}% of the target's variance lies "
        f"between source videos against a {IDEA9A['y_variance']['threshold'] * 100:.0f}% requirement,\n"
        "so source-disjoint folds have very little shared signal to find.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["red"],
        linespacing=1.5,
    )
    save(fig, "symmetry_gate_table")


def make_mechanism() -> None:
    """Contrast the degenerate loss form with the scale-invariant replacement."""
    bakeoff = MECHANISM["check_4_variant_bakeoff"]
    trajectories = MECHANISM["trajectories"]
    treatments = [row for row in bakeoff["rows"] if not row["rung"].startswith("D0")]
    absolute = one(row for row in treatments if row["variant"] == "absolute")
    selected = one(row for row in treatments if row["variant"] == bakeoff["selected_variant"])
    require(
        all(row["beats_gate"] for row in treatments if row["variant"] != "absolute"),
        "the caption says both scale-invariant forms clear the gate; the bundle disagrees",
    )
    # Extra width and height: the two panels' bottom tick labels used to meet in the middle, and
    # the three-line note under the figure needs a band of its own. The gutter is widened after
    # tight_layout, because passing wspace through gridspec_kw makes tight_layout refuse to run.
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))

    for label, series in trajectories.items():
        if "absolute" in label:
            color, width = COLORS["red"], 1.6
        elif "parameter_free" in label:
            color, width = COLORS["green"], 1.6
        elif "normalized" in label:
            color, width = COLORS["teal"], 1.2
        else:
            color, width = COLORS["muted"], 1.2
        axes[0].plot(series["epoch"], series["head_signal_scale"], color=color, linewidth=width, label=label)
        axes[1].plot(series["epoch"], series["rho_view"], color=color, linewidth=width, label=label)

    axes[0].set_title("The head's own output scale", fontsize=8.2, color=COLORS["navy"], fontweight="bold")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("head signal scale")
    axes[1].set_title(
        "Mirror residual rho (the endpoint that matters)",
        fontsize=8.2,
        color=COLORS["navy"],
        fontweight="bold",
    )
    axes[1].set_xlabel("epoch")
    # Short y label. At its full length this rotated label ran the height of the panel and fought
    # the tick labels; the rho scale it used to carry is stated in the note under the figure.
    axes[1].set_ylabel(r"mirror residual $\rho$")
    for axis in axes:
        axis.grid(alpha=0.24)
        # Headroom above the top data point, so the topmost y tick label is not pressed against
        # the panel frame directly under the suptitle.
        axis.margins(y=0.12)
    axes[1].legend(frameon=False, fontsize=5.4, loc="lower left")

    fig.suptitle(
        "Why the original equivariance term had to be replaced",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["navy"],
    )
    # Three balanced lines at a positive figure coordinate. The rho scale is stated here because
    # the panel's y label no longer carries it.
    fig.text(
        0.5,
        0.015,
        f"Synthetic fixtures, not gait. Rho is 0 when mirror equivariant and 4 when mirror blind. "
        f"The red absolute form cuts its own loss\n{absolute['term_fold_reduction']:.0f}-fold by shrinking the head "
        f"{absolute['head_scale_fold_shrink']:.1f}-fold (left), yet leaves the endpoint within "
        f"{absolute['improvement_vs_control']:.3f} of the control (right).\nBoth scale-invariant forms clear the gate, but "
        f"{selected['variant']} moves rho by {selected['improvement_vs_control']:.2f}, so it is the one carried forward.",
        ha="center",
        va="bottom",
        fontsize=6.3,
        color=COLORS["muted"],
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.17, 1, 0.93])
    fig.subplots_adjust(wspace=0.26)
    save(fig, "symmetry_mechanism")


def make_real_verdict() -> None:
    """Draw the preregistered verdict: primary endpoint, paired interval, and guardrails.

    Driven entirely by the evaluation bundle so the figure cannot disagree with the notebook
    that applied the rule. Skipped when the evaluation has not run yet.
    """
    path = ARM2 / "idea9_arm2_evaluation_result.json"
    if not path.is_file():
        print("skipping symmetry_real_verdict: new_nb_09_03 has not written its bundle yet")
        return
    require(
        len(LADDER.get("D0", [])) > 1 and len(LADDER.get("E1", [])) > 1,
        "the verdict figure quotes a seed spread, so it needs more than one completed rung per arm",
    )
    result = load(path)
    require(
        all("measurable" in row for row in result.get("guardrails", [])),
        f"{path.name} predates the guardrail-measurability fields; re-run new_nb_09_03",
    )
    primary = result["primary"]
    bootstrap = primary["paired_bootstrap"]
    conditions = result["credit_rule"]
    verdict = result["PRIMARY_VERDICT"]

    # Taller and wider than before: three three-line panel titles, a two-line x label with a
    # legend beneath it, and a two-line note all need their own bands. The gutter is widened after
    # tight_layout, because passing wspace through gridspec_kw makes tight_layout refuse to run.
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.6))

    per_rung = result["per_rung"]
    x = np.arange(2)
    means = [primary["D0_mean"], primary["E1_mean"]]
    axes[0].bar(
        x,
        means,
        color=[COLORS["muted"], COLORS["green"]],
        edgecolor=COLORS["navy"],
        linewidth=0.6,
        width=0.58,
    )
    for index, rung in enumerate(("D0", "E1")):
        values = [row["rho_target"] for row in per_rung if row["rung"] == rung]
        axes[0].scatter(
            [index] * len(values),
            values,
            s=16,
            color="white",
            edgecolor=COLORS["ink"],
            linewidth=0.7,
            zorder=3,
        )
    # Labels clear the tallest thing in each column, which is either the bar or the highest
    # per-seed dot, and the limit is derived from that same maximum. Both were hardcoded before,
    # which would have put the label under a dot as soon as rho left the unit interval.
    per_rung_values = {
        rung: [row["rho_target"] for row in per_rung if row["rung"] == rung] for rung in ("D0", "E1")
    }
    tallest = max(means + [value for values in per_rung_values.values() for value in values])
    axes[0].set_ylim(0, tallest * 1.30)
    for xi, rung, value in zip(x, ("D0", "E1"), means):
        label_y = max([value] + per_rung_values[rung]) + tallest * 0.05
        axes[0].text(xi, label_y, f"{value:.3f}", ha="center", va="bottom", fontsize=6.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["D0 control\nterm off", "E1 treatment\nterm on"], fontsize=6.3)
    axes[0].set_ylabel(r"$\rho$ on the target encoder")
    # Three short lines plus a generous pad. The second line of the two-line version was wider
    # than the panel and its bounding box landed on the topmost y tick label.
    axes[0].set_title(
        f"Primary endpoint improves\nby {primary['improvement']:.3f}, control\n"
        f"spread {primary['D0_seed_spread']:.3f}",
        fontsize=7.3,
        color=COLORS["navy"],
        fontweight="bold",
        pad=9,
    )

    centre = bootstrap["mean"]
    axes[1].errorbar(
        [centre],
        [0],
        xerr=[[centre - bootstrap["ci_low"]], [bootstrap["ci_high"] - centre]],
        fmt="o",
        color=COLORS["blue"],
        capsize=4,
        markersize=5,
    )
    axes[1].axvline(0, color=COLORS["red"], linestyle="--", linewidth=1.0)
    axes[1].set_yticks([])
    axes[1].set_ylim(-1, 1)
    # Each video contributes its own ratio here, whereas the left panel is a single ratio of
    # summed terms, so the two improvements are on deliberately different scales.
    axes[1].set_xlabel(
        "mean per-source improvement in $\\rho$\n(one ratio per video, so not the left panel's scale)",
        fontsize=6.6,
    )
    axes[1].set_title(
        f"Paired over {bootstrap['sources']} source videos\n"
        f"95% interval [{bootstrap['ci_low']:.2f}, {bootstrap['ci_high']:.2f}]",
        fontsize=7.3,
        color=COLORS["navy"],
        fontweight="bold",
        pad=9,
    )

    measurable = [row for row in result["guardrails"] if row["measurable"]]
    y = np.arange(len(measurable))[::-1]
    axes[2].barh(
        y,
        [row["regression"] for row in measurable],
        color=[COLORS["red"] if not row["within_control_spread"] else COLORS["green"] for row in measurable],
        edgecolor=COLORS["navy"],
        linewidth=0.5,
        height=0.55,
    )
    axes[2].scatter(
        [row["D0_seed_spread"] for row in measurable],
        y,
        marker="|",
        s=150,
        color=COLORS["ink"],
        label="control seed spread",
    )
    axes[2].axvline(0, color=COLORS["ink"], linewidth=0.8)
    pretty = {
        "feature_std": "feature spread",
        "mean_pair_cosine": "mean pair cosine",
        "leaky_probe_balanced_accuracy": "condition probe\n(leaky stand-in)",
        "minimum_centroid_distance": "minimum centroid\ndistance",
    }
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(
        [pretty.get(row["guardrail"], row["guardrail"].replace("_", " ")) for row in measurable],
        fontsize=6.0,
    )
    axes[2].set_xlabel("regression (positive is worse)", fontsize=6.6)
    # A blank strip above the top bar gives the legend somewhere to sit that is neither on a bar
    # nor outside the axes. Anchored outside, this legend was counted in the axes tight bounding
    # box and tight_layout shrank all three panels to 74 pixels tall to make room for it.
    axes[2].set_ylim(-0.55, len(measurable) - 0.25)
    axes[2].legend(
        frameon=True,
        framealpha=0.85,
        edgecolor="none",
        fontsize=5.6,
        loc="upper right",
        handletextpad=0.4,
        borderpad=0.3,
    )
    axes[2].set_title(
        "Guardrails against the\ncontrol's own seed spread",
        fontsize=7.3,
        color=COLORS["navy"],
        fontweight="bold",
        pad=9,
    )

    for axis in axes:
        axis.grid(alpha=0.22)

    fig.suptitle(
        f"Real equivariance ladder, preregistered verdict: {verdict}",
        fontsize=9.6,
        fontweight="bold",
        color=COLORS["navy"] if verdict == "CREDIT" else COLORS["red"],
    )
    passed = [name for name, ok in conditions.items() if name.startswith("condition") and ok]
    failed = [name for name, ok in conditions.items() if name.startswith("condition") and not ok]
    scale = result["degenerate_solution_check"]
    # Two lines at a positive figure coordinate inside a reserved band. This note used to be drawn
    # at y=-0.10 and was 1441 px wide on a 1314 px canvas.
    fig.text(
        0.5,
        0.015,
        f"{len(passed)} of {len(passed) + len(failed)} credit conditions met, and all three are required. "
        f"Failed: " + (", ".join(name.replace("_", " ") for name in failed) or "none") + ".\n"
        f"The head's output scale rose from {scale['D0_head_scale']:.2f} to {scale['E1_head_scale']:.2f}, "
        "so the endpoint did not improve by the readout collapsing.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color=COLORS["muted"],
        linespacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.14, 1, 0.92])
    fig.subplots_adjust(wspace=0.32)
    save(fig, "symmetry_real_verdict")


def main() -> None:
    make_lane_ladder()
    make_gate_table()
    make_mechanism()
    make_real_verdict()
    rung_note = ", ".join(f"{rung}={len(runs)}" for rung, runs in sorted(LADDER.items())) or "none complete"
    print(f"wrote symmetry figures for {FINGERPRINT[:12]}... (real ladder rungs: {rung_note})")


if __name__ == "__main__":
    main()
