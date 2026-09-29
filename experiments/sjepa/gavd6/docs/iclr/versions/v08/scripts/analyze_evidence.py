"""Reconstruct v08 estimates from the immutable completed-study exports.

Run: .venv/bin/python docs/iclr/versions/v08/scripts/analyze_evidence.py
No training, new participant selection, raw-data inference, or source mutation.
All new strata, penalty sensitivities, and plot intervals are exploratory.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

HERE = Path(__file__).resolve()
VERSION = HERE.parents[1]
# A portable source bundle can supply the identical outputs/iclr tree under
# version-local inputs. The repository layout remains the default otherwise.
ROOT = VERSION / "inputs" if (VERSION / "inputs").is_dir() else HERE.parents[5]
OUT = VERSION / "evidence"
assert OUT.is_dir(), "Create the version-owned evidence directory first"
DATA = ROOT / "outputs/iclr"
INPUTS = {}
SEEDS = [17, 29, 43]
DRAWS = 2000
BOOTSTRAP_SEED = 731
PENALTIES = [0., 180., 360., 720.]
DIRECT = "P-direct-none-base"
DELTA = "F-response-jepa_delta_v1-graph_time-paired_change"
ENDPOINT = "F-response-jepa_endpoint_v1-graph_time-paired_change"


def read(rel, csv=True):
    p = ROOT / rel
    INPUTS[rel] = hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p) if csv else json.loads(p.read_text())


core = read("outputs/iclr/walking-core/evaluation/per-person.csv")
response = read("outputs/iclr/jepa-response/evaluation/per-person.csv")
conditions = read("outputs/iclr/jepa-response/evaluation/response-by-condition-person.csv")
coverage = read("outputs/iclr/jepa-response/evaluation/coverage-per-person.csv")
curves = read("outputs/iclr/jepa-response/evaluation/response-curves-person.csv")
repair = read("outputs/iclr/readout-repair/development/evaluation/per-person-by-extractor.csv")
vit = repair[repair.extractor.eq("vitpose_base")].copy()
people = sorted(response.canonical_person_id.unique())
assert len(people) == 14
for frame in [core, response, vit]:
    assert set(frame.canonical_person_id) == set(people)
    assert not frame.duplicated(["method", "canonical_person_id", "seed"]).any()
    assert frame.groupby("method").size().eq(42).all()
    assert sorted(frame.seed.unique()) == SEEDS

# Generate the same interleaved person/seed draws as the historical evaluator.
rng = np.random.default_rng(BOOTSTRAP_SEED)
draw_people, draw_seeds = [], []
for _ in range(DRAWS):
    draw_people.append(rng.integers(14, size=14))
    draw_seeds.append(rng.integers(3, size=3))
draw_people, draw_seeds = np.asarray(draw_people), np.asarray(draw_seeds)


def matrix(frame, metric):
    x = frame.pivot(index="canonical_person_id", columns="seed", values=metric).reindex(index=people, columns=SEEDS)
    assert x.shape == (14, 3) and np.isfinite(x.to_numpy()).all(), metric
    return x.to_numpy()


def summarize(x):
    draws = x[draw_people[:, :, None], draw_seeds[:, None, :]].mean(axis=(1, 2))
    per_person = x.mean(axis=1)
    half = t.ppf(.975, 13) * per_person.std(ddof=1) / np.sqrt(14)
    lo, hi = np.quantile(draws, [.025, .975])
    return dict(estimate=float(x.mean()), ci_low=float(lo), ci_high=float(hi),
                person_t_low=float(per_person.mean()-half), person_t_high=float(per_person.mean()+half),
                people=14, seeds=3, interval_type="descriptive crossed person/seed percentile bootstrap, 2000 draws, RNG731")


def contrast(frame, candidate, comparator, metric):
    a = matrix(frame[frame.method.eq(candidate)], metric)
    b = matrix(frame[frame.method.eq(comparator)], metric)
    d = b-a
    return dict(candidate=candidate, comparator=comparator, metric=metric,
                effect_definition="comparator minus candidate", **summarize(d),
                people_favoring_candidate=int((d.mean(1)>0).sum()),
                seed17=float(d[:, 0].mean()), seed29=float(d[:, 1].mean()), seed43=float(d[:, 2].mean()))


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


units = {"response_error": "degrees", "waveform_error": "degrees", "A_error": "degrees",
         "synthetic_all_nle": "dimensionless NLE", "assignment_failure_rate": "fraction",
         "response_failure_rate": "fraction", "response_failure_contribution": "degrees",
         "response_success_contribution": "degrees", "zero_response_error": "degrees",
         "response_success_conditional_error": "degrees", "direction_accuracy": "fraction",
         "left_excursion_error": "degrees", "right_excursion_error": "degrees"}
metrics = list(units)

# All available response families/readouts: paired person/seed means and effects.
mean_rows = []
for method, group in response.groupby("method"):
    for metric in metrics:
        mean_rows.append(dict(method=method, metric=metric, unit=units[metric],
                              trainable_encoder=method.startswith("P-direct"), **summarize(matrix(group, metric))))
save(pd.DataFrame(mean_rows), "restoration_means.csv")
families = ["P-direct-none", "M-coordinate-graph_time", "M-paired_jepa-graph_time",
            "I-initialized-none", "I-shuffled_jepa-graph_time", "F-response-coordinate_delta_v1-graph_time",
            "F-response-jepa_endpoint_v1-graph_time", "F-response-jepa_delta_v1-graph_time"]
effects = []
for family in families:
    for metric in ["response_error", "waveform_error", "synthetic_all_nle", "assignment_failure_rate"]:
        effects.append(dict(family=family, contrast="original change minus coordinate-only", unit=units[metric],
                            **contrast(response, family+"-base", family+"-paired_change", metric)))
save(pd.DataFrame(effects), "restoration_effects.csv")

# Show all retained repair arms, including both original base readouts.
repair_means, repair_effects = [], []
for method, group in vit.groupby("method"):
    for metric in ["response_error", "waveform_error", "synthetic_all_nle", "response_failure_rate"]:
        repair_means.append(dict(method=method, extractor="vitpose_base", metric=metric, unit=units[metric],
                                 **summarize(matrix(group, metric))))
for variant in ["jepa_delta_v1", "jepa_endpoint_v1"]:
    original = f"F-response-{variant}-graph_time-paired_change"
    base = f"F-response-{variant}-graph_time-base"
    low = f"R-repair-{variant}-scalar_low"
    dense = f"R-repair-{variant}-dense_change"
    for name, candidate, comparator in [("dense versus low scalar", dense, low),
                                       ("low scalar versus original", low, original),
                                       ("dense versus original", dense, original),
                                       ("dense versus coordinate readout", dense, base),
                                       ("dense versus direct coordinate", dense, DIRECT)]:
        for metric in ["waveform_error", "response_error"]:
            repair_effects.append(dict(variant=variant, contrast=name, extractor="vitpose_base",
                                       declared_primary=(variant=="jepa_delta_v1" and name=="dense versus low scalar" and metric=="waveform_error"),
                                       **contrast(vit, candidate, comparator, metric)))
save(pd.DataFrame(repair_means), "repair_means.csv")
save(pd.DataFrame(repair_effects), "repair_effects.csv")

# Reliability: retain unconditional terms and identify two conditional estimands.
means = response.groupby("method")[metrics].mean()
reliability = means.reset_index().copy()
reliability["failure_percent"] = 100*reliability.response_failure_rate
reliability["conditional_error_under_original_weights"] = reliability.response_success_contribution / (1-reliability.response_failure_rate)
reliability.rename(columns={"response_success_conditional_error": "exported_nested_success_conditional_error"}, inplace=True)
counts = coverage.groupby("method")[["response_pairs", "response_reference_eligible", "response_successful"]].sum().reset_index()
counts["raw_failed_pairs_across_seeds"] = counts.response_reference_eligible-counts.response_successful
counts["raw_pair_failure_percent"] = 100*counts.raw_failed_pairs_across_seeds/counts.response_reference_eligible
reliability = reliability.merge(counts, on="method", validate="one_to_one")
save(reliability, "method_summary.csv")
assert np.allclose(response.response_error, response.response_success_contribution+720*response.response_failure_rate)
assert np.allclose(response.response_failure_contribution, 720*response.response_failure_rate)

# Define strata before calculating their method rankings; retain every group.
endpoint_metrics = ["A_error", "waveform_error", "synthetic_all_nle", "assignment_failure_rate",
                    "left_excursion_error", "right_excursion_error"]
response_metrics = ["response_error", "zero_response_error", "response_failure_rate",
                    "response_failure_contribution", "response_success_contribution"]
stratum_metrics = endpoint_metrics + response_metrics
configurations = ["observation", "held_intervention", "observation_by_intervention", "observation_by_extractor", "naming"]
group_fields = [["observation"], ["held_intervention"], ["observation", "held_intervention"], ["observation", "extractor"], ["naming"]]
condition_frames = []
for kind, fields in zip(configurations, group_fields):
    x = conditions.copy()
    for metric in stratum_metrics:
        weight = np.where(x.held_intervention, 1., 4. if metric in endpoint_metrics else 2.)
        x[metric+"_n"] = x[metric]*weight
        x[metric+"_w"] = weight
    cols = [m+s for m in stratum_metrics for s in ["_n", "_w"]]
    keys = ["method", "canonical_person_id", "seed"]+fields
    sums = x.groupby(keys, dropna=False)[cols].sum()
    for metric in stratum_metrics:
        sums[metric] = sums[metric+"_n"]/sums[metric+"_w"]
    out = sums[stratum_metrics].reset_index()
    out["stratum_type"] = kind
    out["stratum"] = out[fields].astype(str).agg(" | ".join, axis=1)
    condition_frames.append(out)
condition_person = pd.concat(condition_frames, ignore_index=True)
save(condition_person, "condition_person.csv")
condition_summary, condition_effects = [], []
for (kind, stratum, method), group in condition_person.groupby(["stratum_type", "stratum", "method"]):
    for metric in stratum_metrics:
        condition_summary.append(dict(stratum_type=kind, stratum=stratum, method=method,
                                       metric=metric, unit=units[metric], **summarize(matrix(group, metric))))
for (kind, stratum), group in condition_person.groupby(["stratum_type", "stratum"]):
    for metric in ["response_error", "response_failure_rate", "response_success_contribution", "assignment_failure_rate"]:
        condition_effects.append(dict(stratum_type=kind, stratum=stratum, contrast="delta versus endpoint",
                                      **contrast(group, DELTA, ENDPOINT, metric)))
    for method in sorted(group.method.unique()):
        zero = group[group.method.eq(method)].copy()
        zero["method"] = "zero_response"
        zero["response_error"] = zero.zero_response_error
        condition_effects.append(dict(stratum_type=kind, stratum=stratum, contrast="method versus zero response",
                                      **contrast(pd.concat([group[group.method.eq(method)], zero]), method, "zero_response", "response_error")))
save(pd.DataFrame(condition_summary), "condition_summary.csv")
save(pd.DataFrame(condition_effects), "condition_effects.csv")
for kind in ["observation", "naming", "observation_by_extractor"]:
    recovered = condition_person[condition_person.stratum_type.eq(kind)].groupby("method")[stratum_metrics].mean()
    assert np.allclose(recovered, means[stratum_metrics], rtol=0, atol=1e-10)

# Penalty sensitivity preserves all failures and the original hierarchy.
penalty_person, penalty_summary, penalty_effects = [], [], []
for penalty in PENALTIES:
    x = response[["method", "canonical_person_id", "seed", "response_failure_rate", "response_success_contribution", "zero_response_error"]].copy()
    x["failure_penalty_deg"] = penalty
    x["response_error"] = x.response_success_contribution+penalty*x.response_failure_rate
    x["failure_contribution"] = penalty*x.response_failure_rate
    penalty_person.append(x)
    for method, group in x.groupby("method"):
        penalty_summary.append(dict(failure_penalty_deg=penalty, method=method, **summarize(matrix(group, "response_error")),
                                    zero_response_mean=float(group.zero_response_error.mean()),
                                    success_contribution=float(group.response_success_contribution.mean()),
                                    failure_contribution=float(group.failure_contribution.mean()),
                                    failure_rate=float(group.response_failure_rate.mean())))
        z = group.copy(); z["method"] = "zero_response"; z["response_error"] = z.zero_response_error
        penalty_effects.append(dict(failure_penalty_deg=penalty, contrast="method versus zero response",
                                    **contrast(pd.concat([group, z]), method, "zero_response", "response_error")))
    penalty_effects.append(dict(failure_penalty_deg=penalty, contrast="delta versus endpoint",
                                **contrast(x, DELTA, ENDPOINT, "response_error")))
save(pd.concat(penalty_person, ignore_index=True), "penalty_sensitivity_person.csv")
save(pd.DataFrame(penalty_summary), "penalty_sensitivity.csv")
save(pd.DataFrame(penalty_effects), "penalty_effects.csv")

# Reproduce saved primary calculations, including their distinct interval rules.
saved_checks = []
for run, frame in [("walking-core", core), ("jepa-response", response)]:
    saved = read(f"outputs/iclr/{run}/evaluation/comparisons.json", csv=False)
    for metric, entry in saved.items():
        check = contrast(frame, entry["candidate"], entry["comparator"], metric)
        assert np.allclose([check["estimate"], check["ci_low"], check["ci_high"]],
                           [entry["improvement"], *entry["crossed_person_seed_ci95"]], rtol=0, atol=1e-10)
        saved_checks.append(dict(run=run, metric=metric, passed=True))
saved = read("outputs/iclr/readout-repair/development/evaluation/comparisons.json", csv=False)
for metric, entry in saved["primary"].items():
    check = contrast(vit, entry["candidate"], entry["comparator"], metric)
    assert np.allclose([check["estimate"], check["person_t_low"], check["person_t_high"]],
                       [entry["improvement"], *entry["person_averaged_t_ci95"]], rtol=0, atol=1e-10)
    saved_checks.append(dict(run="readout-repair", metric=metric, passed=True))

calibration = read("outputs/iclr/jepa-response/diagnostics/loss-calibration.json", csv=False)
g = calibration["gradients"]; coefficient = calibration["coefficients"]["jepa_delta_v1"]
gradient = {name: float(coefficient*g[name]["rms"]/g["jepa_base"]["rms"])
            for name in ["jepa_delta_v1", "jepa_endpoint_v1"]}
clipping = []
ledger = read("outputs/iclr/jepa-response/ledger.json", csv=False)
def find_clips(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            if "clip" in k.lower() and isinstance(v, (int, float, str, bool)):
                clipping.append(dict(path=path+"/"+k, value=v))
            find_clips(v, path+"/"+k)
    elif isinstance(value, list):
        for i, v in enumerate(value): find_clips(v, path+f"/{i}")
find_clips(ledger.get("completed", {}))

definitions = {
    "canonical_primary_penalty_deg": 720,
    "penalty_derivation": "Interior projected angles are in [0,180], per-leg excursion in [0,180], A in [-180,180], response in [-360,360], and a valid prediction's absolute response error is bounded by 720 degrees.",
    "penalty_choices": {"0": "Accounting lower bound assigning zero error to failures; not a useful operating score and not a conditional-success mean.",
                        "180": "Supplementary quarter-cost stress test; 180 is an angle-error bound, not a response-error bound.",
                        "360": "Half-cost stress test and worst-case zero-response error; not the maximum error of a valid restored response.",
                        "720": "Inherited canonical maximum response-error bound; original declared analysis unchanged."},
    "unconditional_response": "E_w[absolute response error times successful indicator] + penalty * P_w(failure), with reference eligibility and original hierarchy fixed.",
    "conditional_error_under_original_weights": "E_w[absolute error times successful indicator] / P_w(success). Conditioning can change population weights; this is not the independently averaged per-family conditional export.",
    "exported_nested_success_conditional_error": "Existing exporter averages successful errors within each source family before the motion/person hierarchy. Retained separately; it is not additive with the failure contribution.",
    "failure_rate": "Person-balanced, seed-averaged probability under the same hierarchical weights as the unconditional score. Raw failure counts and raw pair rates are separate columns.",
    "condition_weighting": "Nonheld:held 4:1 for endpoint metrics (baseline plus duplicate zero, 5 and 10 versus 15); 2:1 for response metrics (5 and 10 versus 15). No no-change contrast enters primary response.",
    "true_magnitude_bins": "Not reconstructable from the compact exports: signed reference_change in response-curves-person.csv is already averaged across source windows/motions, naming, extractor and seeds. abs(mean reference_change) is not mean(abs(reference_change)). No per-pair magnitude-bin analysis is manufactured.",
    "alternative_scale_analysis": "Report all clear/occluded by seen (5 and 10) or held (15) intervention groups, each with its saved zero-response MAE. Nominal edits are never labeled true response magnitudes.",
    "intervals": "All new crossed intervals are exploratory, unadjusted person/seed percentile bootstrap with 2000 draws and RNG731; contrasts retain pairing. Repair mean and effect CSVs also provide person-t intervals after averaging seeds within each person, conditional on three fits. Only the retained delta dense-versus-low-scalar ViTPose waveform contrast is the repair's declared primary. New mean intervals do not imply new independent samples.",
    "effect_direction": "All contrast CSVs use comparator minus candidate. Positive favors the candidate for error metrics. Restoration effects choose coordinate-only as candidate and original change as comparator, so positive is deterioration from adding the package.",
    "laterality_components": "Wrong/ambiguous/missing counts are not available in the compact condition exports. Only the combined geometric assignment failure rate can be plotted.",
}
metadata = {"scope": "Exploratory reanalysis of completed synthetic development exports; no new training or confirmation.",
            "people": people, "seeds": SEEDS, "input_sha256": INPUTS, "saved_comparisons_reproduced": saved_checks,
            "definitions": definitions, "weighted_initial_auxiliary_to_base_gradient_rms": gradient,
            "clipping_fields_from_saved_ledger": clipping, "all_selected_groups_reported": True,
            "true_magnitude_reanalysis_supported": False,
            "observed_nonnegative_penalty_lower_bounds": {
                method: {"unconditional_success_contribution": float(means.loc[method, "response_success_contribution"]),
                         "zero_response_mean": float(means.loc[method, "zero_response_error"]),
                         "all_nonnegative_cost_scores_exceed_zero_mean": bool(means.loc[method, "response_success_contribution"] > means.loc[method, "zero_response_error"])}
                for method in [DIRECT, DELTA, ENDPOINT]},
            "outputs": sorted(p.name for p in OUT.glob("*.csv"))}
(OUT/"provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
claim_map = [
    {"claim": "Pooled zero-response benchmark and all 16 learned variants", "source": "outputs/iclr/jepa-response/evaluation/per-person.csv", "derived": "method_summary.csv", "fields": ["response_error", "zero_response_error"], "status": "Reproduced completed result"},
    {"claim": "Delta and endpoint failure rates and unconditional score decomposition", "source": "outputs/iclr/jepa-response/evaluation/per-person.csv", "derived": "method_summary.csv", "fields": ["response_failure_rate", "response_success_contribution", "response_failure_contribution", "response_error"], "status": "New descriptive presentation of existing metrics"},
    {"claim": "Sensitivity to all fixed failure costs", "source": "outputs/iclr/jepa-response/evaluation/per-person.csv", "derived": "penalty_sensitivity.csv; penalty_effects.csv", "status": "New exploratory reanalysis; original primary unchanged"},
    {"claim": "Delta and endpoint exceed zero response even at the zero-cost accounting lower bound, so their observed pooled ranking holds for every nonnegative failure cost", "source": "outputs/iclr/jepa-response/evaluation/per-person.csv", "derived": "method_summary.csv; penalty_effects.csv; provenance.json", "status": "New exploratory lower-bound argument about observed means; not a success-only accuracy comparison or general population theorem"},
    {"claim": "All observation/intervention and observation/extractor groups", "source": "outputs/iclr/jepa-response/evaluation/response-by-condition-person.csv", "derived": "condition_summary.csv; condition_effects.csv", "status": "New exploratory strata; true magnitude bins unavailable"},
    {"claim": "Original package effects across eight families", "source": "outputs/iclr/jepa-response/evaluation/per-person.csv", "derived": "restoration_means.csv; restoration_effects.csv", "status": "Existing completed comparisons with new descriptive plot intervals"},
    {"claim": "Original, low-scalar and dense repair on ViTPose", "source": "outputs/iclr/readout-repair/development/evaluation/per-person-by-extractor.csv", "derived": "repair_means.csv; repair_effects.csv", "status": "Original declared primary and secondary contrasts retained; new mean intervals descriptive"},
    {"claim": "Initial feature-auxiliary influence asymmetry", "source": "outputs/iclr/jepa-response/diagnostics/loss-calibration.json", "derived": "provenance.json", "status": "Verified initialization-only calculation"},
]
(OUT/"claim-to-artifact.json").write_text(json.dumps(claim_map, indent=2)+"\n")

selected = reliability[reliability.method.isin([DIRECT, DELTA, ENDPOINT])]
lines = ["# Version 08 evidence audit", "", "This analysis uses the completed 14-person, three-seed development panel. All new penalty sensitivities, condition summaries, and figure intervals are exploratory and unadjusted. Original primary endpoints and the 720-degree scoring rule are retained.", "", "## What the compact exports support", "", definitions["true_magnitude_bins"], "", definitions["alternative_scale_analysis"], "", "## Failure accounting", "", "The person-balanced rates differ from raw pair fractions because the published scores average within windows, motions, and people. Conditional errors are reported separately from unconditional contributions.", "", "| Method | Failure rate (%) | Successful contribution (degrees) | Failure contribution at 720 (degrees) | Total (degrees) | Conditional error under original weights (degrees) |", "|---|---:|---:|---:|---:|---:|"]
for _, r in selected.iterrows():
    lines.append(f"| {r.method} | {r.failure_percent:.6f} | {r.response_success_contribution:.6f} | {r.response_failure_contribution:.6f} | {r.response_error:.6f} | {r.conditional_error_under_original_weights:.6f} |")
lines += ["", definitions["conditional_error_under_original_weights"], "", definitions["exported_nested_success_conditional_error"], "", "## Sensitivity without changing the primary", "", definitions["penalty_derivation"], "", "| Failure cost (degrees) | Delta-versus-endpoint improvement (degrees) | Crossed 95% interval | Learned methods below zero-response mean |", "|---|---:|---|---:|"]
ps = pd.DataFrame(penalty_summary)
for row in penalty_effects:
    if row["contrast"] == "delta versus endpoint":
        z = ps[ps.failure_penalty_deg.eq(row["failure_penalty_deg"])]
        wins = int((z.estimate < z.zero_response_mean).sum())
        lines.append(f"| {row['failure_penalty_deg']:.0f} | {row['estimate']:.6f} | [{row['ci_low']:.6f}, {row['ci_high']:.6f}] | {wins} |")
lines += ["", *[f"- Cost {k}: {v}" for k, v in definitions["penalty_choices"].items()], "", "The same predictions and failure indicators are rescored; no alternative cost is selected as preferable. A zero cost assigns failed cases a free score and must not be interpreted as success-only accuracy. All intervals in the delta/endpoint sensitivity remain descriptive.", "", "## Version 07 accuracy assessment", "", "The main v07 numerical claims reproduce. Its geometric assignment is post hoc, its primary response interval remains unresolved, and its 93% repair ratio and 74% failure fraction are descriptive. The 0.0013% versus 10% initialization gradient influence is verified from the calibration receipt. Important improvements achievable in v08 are to expose rates separately from contributions, make the zero-response benchmark central, show all supported observation strata, and separate jointly trained direct models from frozen readouts. These do not supply independent confirmation.", "", "## Limits requiring new artifacts or experiments", "", "Per-pair true response magnitudes and wrong/ambiguous/missing assignment components are absent from the compact packet. Raw predictions, learning curves, and feature arrays are unavailable locally. Fourteen repeatedly inspected people and three fitted seeds do not become additional evidence through new strata or bootstrap draws. External methods were not fitted; independent population, natural-observation, anatomical, and clinical validation remain future work.", "", "## Reproduction", "", "Run `.venv/bin/python docs/iclr/versions/v08/scripts/analyze_evidence.py`. The script modifies only version-08 evidence files, verifies the complete person/seed panels, checks additive decompositions and exact condition reaggregation, and reconstructs all 16 saved comparison entries. CSV intervals label their estimand and procedure. Source hashes and definitions are in `provenance.json`; the claim-to-artifact map is `claim-to-artifact.json`."]
lines += ["", "## An informative lower-bound distinction", "", "Delta and endpoint have unconditional successful contributions of 6.209596 and 6.305974 degrees, both exceeding the zero-response mean of 5.810830 degrees. Since failure rates are nonnegative, S(C) is at least this successful contribution for every nonnegative cost C. Thus no nonnegative failure cost reverses their observed pooled ranking against zero response. This is an accounting lower-bound argument, not a comparison of conditional successful accuracy. Direct differs: its successful contribution is 5.069067 degrees and its ranking against zero does depend on cost. The argument concerns these development means, not a universal claim about the representation families."]
(OUT/"audit.md").write_text("\n".join(lines)+"\n")
print(json.dumps({"people": 14, "seeds": 3, "saved_comparisons": len(saved_checks),
                  "condition_rows": len(condition_person), "files": metadata["outputs"],
                  "true_magnitude_bins_supported": False}, indent=2))
