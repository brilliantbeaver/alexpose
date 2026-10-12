#!/usr/bin/env python3
"""Check and compare the rounded, downloaded seed-17 Markdown tables.

This does not reconstruct predictions, aggregate people, or verify HAIC execution.
The source hash prevents a later report from silently replacing this evidence.
Run with Python 3; --plot additionally requires matplotlib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE = ROOT / "outputs/synthetic-training-v2/updates-2000-seed-17/report.md"
if not SOURCE.exists():
    SOURCE = ROOT / "outputs/synthetic-training-v2/report.md"
SOURCE_SHA256 = "20256edaa8f04c0e909a980cf65a258c4f8f49e8ad39d09eb21bdc51927c8f55"
EXTRACTORS = ("hrnet_w32", "rtmpose_m", "vitpose_base")
METHODS = {"coordinate", "direct", "initialized", "joint_affine", "joint_offset", "paired_jepa", "unchanged"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tables(markdown):
    result, active = [], False
    for line in markdown.splitlines():
        if line.startswith("|"):
            if not active:
                result.append([])
            result[-1].append([cell.strip() for cell in line.strip("|").split("|")])
            active = True
        else:
            active = False
    return result


def fraction(value):
    return tuple(int(v.strip()) for v in value.split("/"))


def metric_rows(table, metric):
    result = []
    for cells in table[2:]:
        method, extractor, seed, value = cells[:4]
        require(method in METHODS and extractor in EXTRACTORS and seed == "17", "Unexpected metric identity")
        require(float(value) > 0, "Expected positive reported errors")
        result.append(dict(method=method, extractor=extractor, seed=int(seed), metric=metric, value=float(value)))
    require(len(result) == 21, "Expected 21 method/extractor rows per metric")
    require(len({(r["method"], r["extractor"]) for r in result}) == 21, "Duplicate metric identity")
    return result


def plot(metric_values):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    selected = (
        ("joint_offset", "Joint offset", "#838b93"),
        ("joint_affine", "Joint affine", "#a76524"),
        ("initialized", "Initialized encoder", "#5d537d"),
        ("coordinate", "Coordinate pretraining", "#57928f"),
        ("paired_jepa", "Paired JEPA", "#2762a4"),
        ("direct", "Direct supervision", "#bd4041"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    width = .125
    for ax, metric, title in zip(axes, ("visible_nle", "displacement_nle"),
                                 ("Visible position error", "0.20-second displacement error")):
        values = metric_values[metric]
        for index, (method, label, color) in enumerate(selected):
            x = [position + (index - 2.5) * width for position in range(3)]
            changes = [100 * (values["unchanged", e] - values[method, e]) / values["unchanged", e]
                       for e in EXTRACTORS]
            ax.bar(x, changes, width=width * .94, color=color, label=label)
        ax.axhline(0, color="#333333", linewidth=.8)
        ax.set_xticks(range(3), ("HRNet-W32", "RTMPose-M", "ViTPose-Base"))
        ax.set_title(title, fontsize=12)
        ax.set_ylabel("Error reduction relative to unchanged tracks")
        ax.yaxis.set_major_formatter(PercentFormatter())
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylim(-1, 51)
    axes[1].set_ylim(-2, 11)
    fig.suptitle("Expanded synthetic development panel: seed 17, 8 people", fontsize=14, y=.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, .055), frameon=False)
    fig.text(.5, .025, "Rounded report values; 2,000 updates per phase confirmed by user. No uncertainty estimates available.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.08, right=.985, bottom=.28, top=.87, wspace=.29)
    for extension in ("png", "svg"):
        fig.savefig(HERE / f"coordinate-and-displacement.{extension}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    raw = SOURCE.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == SOURCE_SHA256, "Source report changed; preserve this analysis and use a new analysis directory")
    parsed = tables(raw.decode())
    require(len(parsed) == 4, "Expected four Markdown result tables")
    require(parsed[0][0][3:] == ["Visible normalized error", "People"], "Unexpected position table")
    require(parsed[1][0][3:] == ["0.20 s displacement error", "Amplitude error", "Amplitude ratio"],
            "Unexpected movement table")
    require(all(r[4] == "8" for r in parsed[0][2:]), "Expected eight development people")
    require(all(r[4:] == ["unsupported", "unsupported"] for r in parsed[1][2:]), "Amplitude support changed")
    metrics = metric_rows(parsed[0], "visible_nle") + metric_rows(parsed[1], "displacement_nle")
    values = {name: {(r["method"], r["extractor"]): r["value"] for r in metrics if r["metric"] == name}
              for name in ("visible_nle", "displacement_nle")}
    contrasts = []
    comparisons = (("joint_offset", "unchanged"), ("joint_affine", "unchanged"),
                   ("direct", "unchanged"), ("direct", "joint_offset"), ("direct", "joint_affine"),
                   ("direct", "paired_jepa"), ("direct", "coordinate"), ("direct", "initialized"),
                   ("paired_jepa", "unchanged"), ("paired_jepa", "coordinate"),
                   ("paired_jepa", "initialized"))
    rankings = []
    for metric, lookup in values.items():
        for extractor in EXTRACTORS:
            rankings.append(dict(metric=metric, extractor=extractor,
                                 methods=sorted(METHODS, key=lambda method: lookup[method, extractor])))
            for candidate, comparator in comparisons:
                a, b = lookup[candidate, extractor], lookup[comparator, extractor]
                contrasts.append(dict(metric=metric, extractor=extractor, candidate=candidate, comparator=comparator,
                                      candidate_value=a, comparator_value=b, absolute_error_reduction=b-a,
                                      relative_error_reduction_percent=100*(b-a)/b))
    eligibility = {}
    for extractor, seed, ratio, ineligible in parsed[2][2:]:
        eligible, total = fraction(ratio)
        require(extractor in EXTRACTORS and seed == "17", "Unexpected oracle identity")
        require((eligible, total, int(ineligible)) == (32, 128, 96), "Unexpected oracle counts")
        eligibility[extractor] = dict(eligible_records=eligible, all_records=total,
                                      reference_ineligible=int(ineligible), eligible_fraction=eligible/total)
    require(set(eligibility) == set(EXTRACTORS), "Incomplete oracle table")
    timings = []
    for cells in parsed[3][2:]:
        method, extractor, seed, support, incomplete, mismatch, missed, extra, matches, mae = cells
        require(method in METHODS and extractor in EXTRACTORS and seed == "17", "Unexpected timing identity")
        supported, eligible = fraction(support)
        matched, reference = fraction(matches)
        missed, extra = int(float(missed)), int(float(extra))
        require(matched + missed == reference == 158, "Timing reference count inconsistency")
        require(supported + int(mismatch) == eligible == 32, "Original-support count inconsistency")
        require(int(incomplete) == 0, "Precision needs separate treatment of incomplete predictions")
        require(0 <= float(mae) <= .12, "Matched timing error exceeds the configured tolerance")
        timings.append(dict(method=method, extractor=extractor, seed=int(seed),
                            original_supported=supported, eligible_records=eligible, all_records=128,
                            incomplete_predictions=int(incomplete), count_mismatch_records=int(mismatch),
                            reference_events=reference, matched_events=matched, missed_events=missed,
                            unmatched_predicted_events=extra, predicted_events=matched+extra,
                            event_recall=matched/reference, event_precision=matched/(matched+extra),
                            conditional_timing_mae_s=float(mae)))
    require(len(timings) == 21 and len({(r["method"], r["extractor"]) for r in timings}) == 21,
            "Incomplete/duplicate timing table")
    require(all(r["methods"][0] == "direct" for r in rankings), "Reported best method changed")
    require(all(values["displacement_nle"]["joint_offset", e] == values["displacement_nle"]["unchanged", e]
                for e in EXTRACTORS), "Offset displacement invariant differs at reported precision")
    code = ["scripts/research_directions/synthetic_training_v2/postrun_checks.py",
            "scripts/research_directions/synthetic_training_v2/diagnostics/calibration.py",
            "scripts/research_directions/synthetic_training_v2/diagnostics/timing.py",
            "scripts/research_directions/synthetic_training_v2/expansion_results.py",
            "src/gavd6_sjepa/research_directions/synthetic_training_v2/evaluation.py",
            "src/gavd6_sjepa/research_directions/synthetic_training_v2/training.py",
            "src/gavd6_sjepa/research_directions/synthetic_training_v2/models.py"]
    output = dict(
        scope="Descriptive arithmetic and consistency checks on rounded Markdown tables only",
        source=str(SOURCE.relative_to(ROOT)), source_sha256=digest, seed=17, development_people=8,
        training_budget=dict(updates_per_phase=2000, evidence="User confirmed source folder updates-2000-seed-17",
                             independently_verified_from_run_artifacts=False),
        limitations=["No full-run per-person/per-window data or run provenance available locally",
                     "No raw-prediction reconstruction, training-completion verification or confidence intervals",
                     "Seven displayed methods are a subset of the full experiment",
                     "Timing counts are correlated records/events, not independent people"],
        local_code_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in code},
        local_code_provenance="Reviewed current local code; matching HAIC code hashes are unavailable",
        metric_rows=metrics, comparisons=contrasts, rankings=rankings,
        timing_eligibility=eligibility, timing=timings,
        checks=dict(metric_cells=42, timing_rows=21, oracle_rows=3, unique_complete_method_rosters=True,
                    timing_denominators_consistent=True, offset_displacement_matches_report=True,
                    primary_amplitude_aggregates_unsupported=21, source_unchanged=True))
    if args.plot:
        plot(values)
    require(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest, "Input report changed during analysis")
    (HERE / "calculations.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps(output["checks"], indent=2))


if __name__ == "__main__":
    main()
