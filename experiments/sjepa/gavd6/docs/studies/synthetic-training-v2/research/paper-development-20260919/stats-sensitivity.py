#!/usr/bin/env python3
"""Planning sensitivity from inspected development people, never a new experiment.

Requires numpy, pandas and scipy. Source CSVs are read-only. Outputs begin stats-.
The noncentral-t calculations assume independent, normally distributed paired
person differences conditional on the fitted models. They omit unknown seed and
dataset-shift variance and are therefore not a power guarantee for the new study.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats


def paired_power(n: int, dz: float, alpha: float = 0.05) -> float:
    critical = stats.t.ppf(1 - alpha / 2, n - 1)
    ncp = abs(dz) * math.sqrt(n)
    # Reflection avoids cancellation/NaNs in the tiny lower tail for large ncp.
    return float(stats.nct.sf(critical, n - 1, ncp)
                 + stats.nct.sf(critical, n - 1, -ncp))


def required_people(dz: float, target: float, alpha: float = 0.05) -> int:
    if dz <= 0:
        raise ValueError("A positive planned standardized effect is required")
    low, high = 2, 4
    while paired_power(high, dz, alpha) < target:
        high *= 2
        if high > 2_000_000:
            raise ValueError("Sample size exceeds planning bound")
    while low < high:
        middle = (low + high) // 2
        if paired_power(middle, dz, alpha) >= target:
            high = middle
        else:
            low = middle + 1
    return low


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path(__file__).resolve().parents[2]
                        / "results/seed17-complete-analysis-20260919")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = [source / name for name in (
        "person-effects.csv", "per-person-metrics.csv", "paired-comparisons.csv")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    effects = pd.read_csv(paths[0])
    people = pd.read_csv(paths[1])
    comparisons = pd.read_csv(paths[2])
    summaries = []
    for key, group in effects.groupby(["metric", "extractor", "candidate", "comparator"]):
        metric, extractor, candidate, comparator = key
        if group.person_id.duplicated().any() or len(group) != 8:
            raise ValueError(f"Unexpected independent-person coverage: {key}")
        values = group.improvement.to_numpy()
        mean, sd = float(values.mean()), float(values.std(ddof=1))
        base = float(group.comparator_value.mean())
        retained = comparisons.loc[
            comparisons.metric.eq(metric) & comparisons.extractor.eq(extractor)
            & comparisons.candidate.eq(candidate) & comparisons.comparator.eq(comparator)]
        if len(retained) != 1 or not np.isclose(mean, retained.improvement.iloc[0], atol=1e-15):
            raise ValueError(f"Paired aggregate mismatch: {key}")
        # Independent check against balanced per-person primary metric table.
        subset = people.loc[
            people.endpoint.eq("visible") & people.scale_policy.eq("frame_reference")
            & people.condition.eq("all_conditions") & people.extractor.eq(extractor)
            & people.method.isin([candidate, comparator])]
        wide = subset.pivot(index="person_id", columns="method", values=metric)
        independently = wide[comparator] - wide[candidate]
        ordered = group.set_index("person_id").improvement.sort_index()
        if not np.allclose(independently.sort_index(), ordered, atol=1e-15, rtol=0):
            raise ValueError(f"Person-level reconstruction mismatch: {key}")
        summaries.append(dict(metric=metric, extractor=extractor, candidate=candidate,
                              comparator=comparator, people=8, seed=17,
                              mean_difference=mean, sd_difference=sd,
                              comparator_mean=base,
                              ratio_of_means_improvement_percent=100 * mean / base,
                              sd_as_percent_of_comparator_mean=100 * sd / base,
                              observed_dz=mean / sd))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "stats-pilot-variation.csv", index=False)

    standardized = []
    for alpha in [0.05, 0.05 / 3]:
        for dz in [0.2, 0.25, 0.35, 0.5, 0.75]:
            standardized.append(dict(alpha=alpha, planned_dz=dz,
                                     people_for_80_percent=required_people(dz, 0.8, alpha),
                                     people_for_90_percent=required_people(dz, 0.9, alpha)))
    pd.DataFrame(standardized).to_csv(output / "stats-standardized-sample-size.csv", index=False)

    sensitivity = []
    for n in [8, 20, 30, 40, 60, 80, 100, 120]:
        for target in [0.8, 0.9]:
            detectable = optimize.brentq(lambda dz: paired_power(n, dz) - target, 0.0001, 10)
            sensitivity.append(dict(people=n, target_power=target, alpha=0.05,
                                    detectable_dz=float(detectable)))
    pd.DataFrame(sensitivity).to_csv(output / "stats-fixed-sample-sensitivity.csv", index=False)

    # This deliberately fixes the desired effect at 5%, instead of using the
    # observed mean as an expected future effect. Comparator means are pilot
    # scale anchors only; candidate-vs-direct future variance is unknown.
    proxies = summary.loc[summary.candidate.eq("direct") & summary.comparator.eq("static")]
    scenarios = []
    for row in proxies.itertuples(index=False):
        for multiplier in [1.0, 1.5, 2.0]:
            for reduction in [2.0, 5.0]:
                sd_percent = row.sd_as_percent_of_comparator_mean * multiplier
                dz = reduction / sd_percent
                scenarios.append(dict(
                    variance_proxy_contrast="direct_minus_static", metric=row.metric,
                    extractor=row.extractor, sd_multiplier=multiplier,
                    assumed_paired_sd_percent=sd_percent,
                    planned_true_reduction_percent=reduction,
                    null_reduction_percent=0.0, planned_dz=dz,
                    people_for_80_percent=required_people(dz, 0.8),
                    people_for_90_percent=required_people(dz, 0.9),
                    person_only_power_at_60=paired_power(60, dz)))
    pd.DataFrame(scenarios).to_csv(output / "stats-pilot-scale-scenarios.csv", index=False)

    # Programmatic numerical checks for the calculation rather than experiment
    # tests. At zero effect, rejection probability must equal alpha.
    assert abs(paired_power(40, 0) - 0.05) < 1e-10
    assert required_people(0.5, 0.8) == 34
    for record in standardized:
        for target, field in [(0.8, "people_for_80_percent"), (0.9, "people_for_90_percent")]:
            n = record[field]
            assert paired_power(n, record["planned_dz"], record["alpha"]) >= target
            assert n == 2 or paired_power(n - 1, record["planned_dz"], record["alpha"]) < target
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[p.name] for p in paths)
    verification = dict(
        status="PLANNING_SENSITIVITY_COMPLETE", source_hashes=hashes,
        reconstructed_contrasts=len(summary), independent_people=8, fitted_seeds=[17],
        alpha=0.05, calculation="two-sided paired t noncentral distribution",
        population="inspected normal-treadmill development people; synthetic imagery",
        sd_95_percent_interval_multipliers_normal_assumption=[
            math.sqrt(7 / stats.chi2.ppf(0.975, 7)),
            math.sqrt(7 / stats.chi2.ppf(0.025, 7))],
        limitations=[
            "Pilot differences are not confirmatory evidence or future true effects.",
            "Person-only normal model omits training-seed and new-dataset uncertainty.",
            "No real-data access, usable-subject count, or 5% practical margin is validated.",
            "A 5% true effect powered against zero does not power proof of an effect exceeding 5%.",
            "No training or inference experiment was launched."])
    (output / "stats-verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
