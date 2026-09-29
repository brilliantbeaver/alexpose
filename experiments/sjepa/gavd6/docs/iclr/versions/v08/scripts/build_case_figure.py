"""Participant-level success/failure illustrations from completed response exports.

Recalculate and plot in the repository:
  .venv/bin/python docs/iclr/versions/v08/scripts/build_case_figure.py
Plot the bound analysis-ready rows in a portable source bundle:
  python scripts/build_case_figure.py --from-audit

All 14 people and all three fitted seeds are shown. No raw pose reconstruction,
new experiment, confidence interval, or clinical status is inferred.
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from paper_style import HERE, DIRECT, GRAY, LIGHT, INK, plt, axis_style, panel, save
from matplotlib.lines import Line2D


EVIDENCE = HERE / "evidence"
ROOT = next((p for p in HERE.parents if (p / "outputs/iclr").is_dir()), HERE / "inputs")
DIRECT_METHOD = "P-direct-none-base"
SEEDS = [17, 29, 43]
PANELS = [
    ("pooled_noisy", "A  Pooled observations", "Unchanged − direct", (-5, 25), [0, 10, 20]),
    ("clear_zero", "B  Clear images", "Zero − direct", (-.2, 2.8), [0, 1, 2]),
    ("occluded_zero", "C  Occluded images", "Zero − direct", (-22, 2), [-20, -10, 0]),
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derive():
    paths = {
        "core": ROOT / "outputs/iclr/walking-core/evaluation/per-person.csv",
        "conditions": ROOT / "outputs/iclr/jepa-response/evaluation/response-by-condition-person.csv",
        "audited_conditions": EVIDENCE / "condition_person.csv",
    }
    core = pd.read_csv(paths["core"])
    condition = pd.read_csv(paths["conditions"])
    audited = pd.read_csv(paths["audited_conditions"])
    base = core[core.method.eq(DIRECT_METHOD)].set_index(["canonical_person_id", "seed"])
    noisy = core[core.method.eq("unchanged")].set_index(["canonical_person_id", "seed"])
    assert base.index.is_unique and noisy.index.is_unique
    assert len(base) == len(noisy) == 42
    assert set(base.index) == set(noisy.index)
    people = sorted(base.index.get_level_values("canonical_person_id").unique())
    assert len(people) == 14
    assert sorted(base.index.get_level_values("seed").unique()) == SEEDS
    person_codes = {p: f"P{i:02d}" for i, p in enumerate(people, 1)}
    rows = []
    for person, seed in base.index:
        candidate = base.loc[(person, seed)]
        comparator = noisy.loc[(person, seed)]
        rows.append(dict(panel="pooled_noisy", person_code=person_codes[person],
                         canonical_person_id=person, seed=int(seed), observation="pooled",
                         candidate=DIRECT_METHOD, comparator="unchanged",
                         candidate_response_error=float(candidate.response_error),
                         comparator_response_error=float(comparator.response_error)))

    # Each exported condition cell has already averaged windows/motions within
    # person. Nonzero response uses seen5/10 : held15 weights2:1. Conditions and
    # their crossings are complete in this executed panel.
    condition = condition[condition.method.eq(DIRECT_METHOD)].copy()
    response_metrics = ["response_error", "zero_response_error", "response_failure_rate"]
    condition["weight"] = np.where(condition.held_intervention, 1., 2.)
    for metric in response_metrics:
        condition[metric + "_n"] = condition[metric] * condition.weight
    fields = [metric + "_n" for metric in response_metrics] + ["weight"]
    groups = condition.groupby(["canonical_person_id", "seed", "observation"])[fields].sum()
    for metric in response_metrics:
        groups[metric] = groups[metric + "_n"] / groups.weight
    assert len(groups) == 84
    verification = audited[audited.method.eq(DIRECT_METHOD) & audited.stratum_type.eq("observation")]
    verification = verification.set_index(["canonical_person_id", "seed", "observation"]).sort_index()
    for metric in response_metrics:
        assert np.allclose(groups.sort_index()[metric], verification[metric], rtol=0, atol=1e-10)
    assert np.allclose(groups.groupby(["canonical_person_id", "seed"]).response_error.mean().sort_index(),
                       base.response_error.sort_index(), rtol=0, atol=1e-10)
    for (person, seed, observation), row in groups.iterrows():
        assert observation in ["clear", "occluded"]
        rows.append(dict(panel=observation + "_zero", person_code=person_codes[person],
                         canonical_person_id=person, seed=int(seed), observation=observation,
                         candidate=DIRECT_METHOD, comparator="zero_response",
                         candidate_response_error=float(row.response_error),
                         comparator_response_error=float(row.zero_response_error)))
    seed_rows = pd.DataFrame(rows)
    seed_rows["gain_deg"] = seed_rows.comparator_response_error - seed_rows.candidate_response_error
    seed_rows["response_failure_penalty_deg"] = 720
    seed_rows = seed_rows.sort_values(["panel", "person_code", "seed"]).reset_index(drop=True)
    source_hashes = {
        "outputs/iclr/walking-core/evaluation/per-person.csv": sha256(paths["core"]),
        "outputs/iclr/jepa-response/evaluation/response-by-condition-person.csv": sha256(paths["conditions"]),
        "evidence/condition_person.csv": sha256(paths["audited_conditions"]),
    }
    return seed_rows, source_hashes


def summarize(seed_rows):
    assert len(seed_rows) == 126 and seed_rows.gain_deg.notna().all()
    assert not seed_rows.duplicated(["panel", "canonical_person_id", "seed"]).any()
    assert seed_rows.groupby(["panel", "canonical_person_id"]).size().eq(3).all()
    assert seed_rows.groupby("panel").canonical_person_id.nunique().eq(14).all()
    fields = ["panel", "person_code", "canonical_person_id", "observation", "candidate", "comparator"]
    out = seed_rows.groupby(fields, as_index=False).agg(
        candidate_response_error=("candidate_response_error", "mean"),
        comparator_response_error=("comparator_response_error", "mean"),
        gain_deg=("gain_deg", "mean"), seed_min_gain_deg=("gain_deg", "min"),
        seed_max_gain_deg=("gain_deg", "max"),
        fitted_seeds_favoring_direct=("gain_deg", lambda x: int((x > 0).sum())))
    out["interpretation"] = np.where(out.gain_deg > 0, "lower mean response error for direct",
                                     np.where(out.gain_deg < 0, "higher mean response error for direct", "equal mean response error"))
    return out.sort_values(["panel", "person_code"]).reset_index(drop=True)


def plot(seed_rows, person_rows):
    codes = sorted(person_rows.person_code.unique())
    assert codes == [f"P{i:02d}" for i in range(1, 15)]
    fig = plt.figure(figsize=(5.5, 4.0))
    gs = fig.add_gridspec(1, 3, left=.10, right=.98, bottom=.17, top=.80, wspace=.32)
    ys = {code: 13-i for i, code in enumerate(codes)}
    for i, (key, title, subtitle, limits, ticks) in enumerate(PANELS):
        ax = fig.add_subplot(gs[0, i])
        current = person_rows[person_rows.panel.eq(key)].set_index("person_code")
        for code in codes:
            y = ys[code]
            row = current.loc[code]
            fitted = seed_rows[seed_rows.panel.eq(key) & seed_rows.person_code.eq(code)].sort_values("seed")
            ax.plot([row.seed_min_gain_deg, row.seed_max_gain_deg], [y, y], color=GRAY, lw=.7, zorder=2)
            ax.scatter(fitted.gain_deg, y + np.array([-.11, 0., .11]), s=7, c=GRAY, marker="o", lw=0, zorder=3)
            ax.scatter([row.gain_deg], [y], s=19, c=DIRECT, marker="D", edgecolors="white", linewidths=.35, zorder=4)
        ax.axvline(0, color=INK, lw=.85, zorder=1)
        ax.set(xlim=limits, xticks=ticks, ylim=(-.65, 13.65), yticks=list(ys.values()),
               yticklabels=codes if i == 0 else [], xlabel="Response gain (°)")
        axis_style(ax)
        ax.set_title(title, loc="left", fontsize=9.0, fontweight="bold", pad=26)
        ax.text(0, 1.035, subtitle, transform=ax.transAxes, fontsize=8.5, va="bottom")
        wins = int((current.gain_deg > 0).sum())
        ax.text(.5, -.165, f"{wins}/14 means favor direct", transform=ax.transAxes,
                ha="center", va="top", fontsize=8)
        assert seed_rows[seed_rows.panel.eq(key)].gain_deg.between(*limits).all()
    handles = [Line2D([], [], color=GRAY, marker="o", markersize=2.6, lw=.7, label="Three fitted seeds (range)"),
               Line2D([], [], color=DIRECT, marker="D", markersize=4.2, lw=0, label="Mean across seeds")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, 1.0), ncol=2,
               frameon=False, columnspacing=1.1, handletextpad=.5, handlelength=1.4)
    save(fig, "cases")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-audit", action="store_true", help="Plot retained audited per-seed rows without source exports")
    args = parser.parse_args()
    seed_path = EVIDENCE / "case-figure-seeds.csv"
    meta_path = EVIDENCE / "case-figure-provenance.json"
    if args.from_audit:
        seed_rows = pd.read_csv(seed_path)
        old = json.loads(meta_path.read_text())
        assert sha256(seed_path) == old["outputs_sha256"]["evidence/case-figure-seeds.csv"]
        source_hashes = old["source_sha256"]
    else:
        seed_rows, source_hashes = derive()
        seed_rows.to_csv(seed_path, index=False)
    person_rows = summarize(seed_rows)
    person_path = EVIDENCE / "case-figure-person.csv"
    person_rows.to_csv(person_path, index=False)
    plot(seed_rows, person_rows)
    totals = {}
    for key, _, _, limits, _ in PANELS:
        subset = person_rows[person_rows.panel.eq(key)]
        totals[key] = dict(people=14, fitted_seeds=3, mean_gain_deg=float(subset.gain_deg.mean()),
                           person_means_favoring_direct=int((subset.gain_deg > 0).sum()),
                           person_means_favoring_comparator=int((subset.gain_deg < 0).sum()),
                           x_axis_limits=list(limits))
    metadata = dict(
        scope="Exploratory participant-level illustration of saved development response errors; all 14 people shown, no selection.",
        source_sha256=source_hashes,
        definitions={
            "selection": "Every development person, ordered lexicographically by canonical person ID; display codes P01–P14 preserve this fixed order across panels.",
            "unit": "One mean per person across seeds17/29/43. Small marks and horizontal ranges show those three fitted values, not confidence intervals or additional independent participants.",
            "gain": "Comparator response MAE minus jointly fitted direct-coordinate response MAE; positive favors direct. Success/failure here means relative error improvement/deterioration, not a binary valid/invalid prediction event or clinical outcome.",
            "aggregation": "Original conditions/windows/source motions/person hierarchy retained. Clear and occluded response conditions use nonheld:held2:1 weights. All three pose estimators, cameras, naming conditions and physical orientations are pooled.",
            "cost": "All response scores retain the original720-degree cost for eligible measurement failures. Zero response produces no restored coordinates.",
            "panels": "A compares unchanged observations with direct after pooling clear/occluded; B and C compare zero response with direct within clear and occluded observations. Comparators and x-axis ranges differ and are explicitly labeled.",
            "clinical_scope": "Recorded source identifiers and synthetic development results do not establish disease, clinical affected side, healthy/diseased pairing, treatment effect, or biomechanical-world-model validity.",
        },
        panels=totals,
        checks={"all_people_retained": True, "paired_seeds_retained": True,
                "observation_reaggregation_matches_core_direct": True,
                "condition_rows_match_previous_independent_audit": True,
                "no_confidence_intervals_invented": True},
        script_sha256=sha256(Path(__file__)),
        shared_style_sha256=sha256(Path(__file__).with_name("paper_style.py")),
        outputs_sha256={"evidence/" + p.name: sha256(p) for p in [seed_path, person_path]},
    )
    metadata["outputs_sha256"].update({"figures/cases." + ext: sha256(HERE / "figures" / ("cases." + ext))
                                       for ext in ["pdf", "svg", "png"]})
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
