"""Independent, read-only reanalysis of the downloaded evidence for seven drafts.

Writes only new ICLR audit artifacts beside this script. The historical evidence
and analyses are never rewritten. Condition means preserve the declared level
weights; response direction is deliberately not reaggregated from conditional
rates because its eligibility denominator differs across conditions.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import t

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
DATA = ROOT / "outputs/iclr"


def js(path):
    return json.loads(path.read_text())


def contrast(frame, candidate, comparator, metric):
    paired = frame[frame.method.isin([candidate, comparator])].pivot(
        index=["canonical_person_id", "seed"], columns="method", values=metric)
    assert paired.notna().all().all()
    d = (paired[comparator] - paired[candidate]).unstack("seed").sort_index()
    x = d.to_numpy()
    rng = np.random.default_rng(731)
    draws, people_draws = [], []
    for _ in range(2000):
        p = rng.integers(0, x.shape[0], x.shape[0])
        s = rng.integers(0, x.shape[1], x.shape[1])
        draws.append(x[p][:, s].mean())
        people_draws.append(x[p].mean())
    person = x.mean(axis=1)
    half = t.ppf(.975, len(person)-1)*person.std(ddof=1)/np.sqrt(len(person))
    return dict(candidate=candidate, comparator=comparator, metric=metric,
                people=len(person), seeds=x.shape[1], improvement=float(x.mean()),
                crossed_person_seed_ci95=np.quantile(draws, [.025, .975]).tolist(),
                person_conditional_ci95=np.quantile(people_draws, [.025, .975]).tolist(),
                person_averaged_t_ci95=[float(person.mean()-half), float(person.mean()+half)],
                people_favoring_candidate=int((person > 0).sum()),
                per_seed_improvement={str(k): float(v) for k, v in d.mean().items()})


paths = {name: DATA/name/("development/evaluation" if name == "readout-repair" else "evaluation")
         for name in ["walking-core", "jepa-response", "readout-repair"]}
frames = {name: pd.read_csv(path/"per-person.csv") for name, path in paths.items()}
checks = []
for name, frame in frames.items():
    assert not frame.duplicated(["method", "seed", "canonical_person_id"]).any()
    assert frame.groupby("method").size().eq(42).all()
    assert frame.canonical_person_id.nunique() == 14
    assert set(frame.seed) == {17, 29, 43}
    checks.append(dict(stage=name, rows=len(frame), methods=frame.method.nunique(), people=14, seeds=3))
assert set(frames["walking-core"].canonical_person_id) == set(frames["jepa-response"].canonical_person_id) == set(frames["readout-repair"].canonical_person_id)

verified = []
for name in ["walking-core", "jepa-response"]:
    for metric, saved in js(paths[name]/"comparisons.json").items():
        result = contrast(frames[name], saved["candidate"], saved["comparator"], metric)
        for k in ["improvement", "crossed_person_seed_ci95", "person_conditional_ci95"]:
            assert np.allclose(result[k], saved[k], rtol=0, atol=1e-10)
        verified.append(dict(stage=name, **result))
repair = pd.read_csv(paths["readout-repair"] / "per-person-by-extractor.csv")
vit = repair[repair.extractor.eq("vitpose_base")]
for metric, saved in js(paths["readout-repair"] / "comparisons.json")["primary"].items():
    result = contrast(vit, saved["candidate"], saved["comparator"], metric)
    for k in ["improvement", "person_averaged_t_ci95"]:
        assert np.allclose(result[k], saved[k], rtol=0, atol=1e-10)
    assert np.allclose(result["crossed_person_seed_ci95"], saved["crossed_bootstrap"]["crossed_person_seed_ci95"], rtol=0, atol=1e-10)
    verified.append(dict(stage="readout-repair", **result))

inventory = js(next(DATA.glob("transfer-inventory-*.json")))
selected = [entry for entry in inventory["files"] if entry["status"] == "selected"]
for entry in selected:
    content = (DATA/entry["destination"]).read_bytes()
    assert len(content) == entry["bytes"]
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]

conditions = pd.read_csv(paths["jepa-response"] / "response-by-condition-person.csv")
endpoint_metrics = ["assignment_failure_rate", "A_error", "waveform_error", "synthetic_all_nle",
                    "left_excursion_error", "right_excursion_error"]
response_metrics = ["response_error", "zero_response_error", "response_failure_contribution", "response_success_contribution"]
metrics = endpoint_metrics + response_metrics
strata = []
for factor in ["naming", "physical_state", "camera_id", "observation", "extractor", "held_intervention"]:
    x = conditions.copy()
    for metric in metrics:
        w = np.where(x.held_intervention, 1., 4. if metric in endpoint_metrics else 2.)
        x[metric+"_n"] = x[metric]*w
        x[metric+"_w"] = w
    cols = [m+s for m in metrics for s in ["_n", "_w"]]
    group = ["method", "seed", "canonical_person_id", factor]
    z = x.groupby(group)[cols].sum()
    for metric in metrics:
        z[metric] = z[metric+"_n"] / z[metric+"_w"]
    z = z[metrics].reset_index().rename(columns={factor: "condition"})
    z["factor"] = factor
    strata.append(z)
strata = pd.concat(strata, ignore_index=True)
strata.to_csv(OUT / "laterality-condition-person.csv", index=False)
condition_means = strata.groupby(["method", "factor", "condition"])[metrics].mean().reset_index()
condition_means.to_csv(OUT / "laterality-condition-means.csv", index=False)
for factor in ["naming", "physical_state", "camera_id", "observation", "extractor"]:
    recovered = strata[strata.factor.eq(factor)].groupby("method")[metrics].mean().sort_index()
    original = frames["jepa-response"].groupby("method")[metrics].mean().sort_index()
    assert np.allclose(recovered, original, rtol=0, atol=1e-10)

direct = "P-direct-none-base"
delta = "F-response-jepa_delta_v1-graph_time-paired_change"
endpoint = "F-response-jepa_endpoint_v1-graph_time-paired_change"
exploratory = []
for candidate, comparator in [(direct, "unchanged"), (direct, "M-paired_jepa-graph_time-base")]:
    exploratory.append(dict(factor="overall_core", condition="all", **contrast(frames["walking-core"], candidate, comparator, "assignment_failure_rate")))
for candidate, comparator in [(direct, delta), (delta, endpoint)]:
    exploratory.append(dict(factor="overall_response", condition="all", **contrast(frames["jepa-response"], candidate, comparator, "assignment_failure_rate")))
    for (factor, condition), group in strata.groupby(["factor", "condition"], dropna=False):
        if factor in ["naming", "physical_state", "observation"]:
            exploratory.append(dict(factor=factor, condition=str(condition), **contrast(group, candidate, comparator, "assignment_failure_rate")))

means = frames["jepa-response"].groupby("method").mean(numeric_only=True)
failure_gain = means.loc[endpoint, "response_failure_contribution"]-means.loc[delta, "response_failure_contribution"]
total_gain = means.loc[endpoint, "response_error"]-means.loc[delta, "response_error"]
summary = dict(protocol="Independent audit; laterality strata and contrasts are post hoc, descriptive and unadjusted.",
               positive_direction="Comparator error minus candidate error; multiply assignment-rate differences by 100 for percentage points.",
               source_packet_files_verified=len(selected), panel_checks=checks,
               saved_primary_and_secondary_checks=verified, exploratory_laterality_contrasts=exploratory,
               delta_response_mean_gain_failure_fraction=float(failure_gain/total_gain),
               zero_response_mean=float(means.zero_response_error.iloc[0]),
               learned_variants_beating_zero=int((means.response_error < means.zero_response_error).sum()),
               condition_reaggregation_max_tolerance=1e-10,
               historical_files_modified=False)
(OUT / "independent-evidence-verification.json").write_text(json.dumps(summary, indent=2)+"\n")
print(json.dumps({"verified_packet_files": len(selected), "reconstructed_saved_comparisons": len(verified),
                  "laterality_person_strata_rows": len(strata), "posthoc_laterality_contrasts": len(exploratory),
                  "zero_response": summary["zero_response_mean"], "learned_variants_beating_zero": summary["learned_variants_beating_zero"]}, indent=2))
