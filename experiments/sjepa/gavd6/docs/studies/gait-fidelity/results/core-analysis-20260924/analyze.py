"""Reconstruct the saved core results and export descriptive secondary analyses.

Run from the repository root with .venv/bin/python <this file>. This reads the
six supplied core artifacts, never modifies them, and does not load predictions.
All newly calculated comparisons are exploratory. Bootstrap draws and ordering
match the saved evaluator: 2,000 draws, seed 731, crossed people and fit seeds.
"""
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "outputs/gait-fidelity"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/gait-fidelity-analysis-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(SOURCE / "evaluation/per-person.csv")
coverage = pd.read_csv(SOURCE / "evaluation/coverage.csv")
saved = json.loads((SOURCE / "evaluation/comparisons.json").read_text())
metrics = list(df.columns[3:])
keys = ["method", "canonical_person_id", "seed"]
assert not df.duplicated(keys).any()
assert np.isfinite(df[metrics].to_numpy()).all()
assert len(df) == 16 * 14 * 3
assert (df.groupby("method").size() == 42).all()
assert (coverage.records == 55800).all()
assert (coverage.reference_eligible == coverage.records).all()
means = df.groupby("method")[metrics].mean()
means.to_csv(OUT / "method-means.csv", float_format="%.12g")
df.groupby(["method", "seed"])[metrics].mean().to_csv(OUT / "seed-means.csv", float_format="%.12g")

def interval(x):
    rng = np.random.default_rng(731)
    conditional, crossed = [], []
    for _ in range(2000):
        p = rng.integers(len(x), size=len(x))
        s = rng.integers(x.shape[1], size=x.shape[1])
        conditional.append(float(x[p].mean()))
        crossed.append(float(x[p][:, s].mean()))
    return np.quantile(conditional, [.025, .975]), np.quantile(crossed, [.025, .975])

def difference(candidate, comparator, metric):
    pivot = df[df.method.isin([candidate, comparator])].pivot(
        index=["canonical_person_id", "seed"], columns="method", values=metric)
    delta = (pivot[comparator] - pivot[candidate]).unstack("seed")
    assert delta.shape == (14, 3) and not delta.isna().any().any()
    return delta

J = "M-paired_jepa-graph_time"
D = "P-direct-none"
C = "M-coordinate-graph_time"
I = "I-initialized-none"
S = "I-shuffled_jepa-graph_time"
pair = "-paired_change"
base = "-base"
comparisons = [
    ("registered", J+pair, D+pair),
    ("direct_vs_input", D+base, "unchanged"),
    ("direct_vs_jepa_base", D+base, J+base),
    ("direct_vs_jepa_change", D+base, J+pair),
    ("jepa_vs_initialized_change", J+pair, I+pair),
    ("jepa_vs_shuffled_change", J+pair, S+pair),
    ("jepa_vs_coordinate_change", J+pair, C+pair),
    ("jepa_vs_initialized_base", J+base, I+base),
    ("jepa_vs_shuffled_base", J+base, S+base),
    ("jepa_vs_coordinate_base", J+base, C+base),
] + [("objective_"+name, prefix+pair, prefix+base)
     for name, prefix in [("jepa", J), ("direct", D), ("coordinate", C), ("initialized", I), ("shuffled", S)]]
contrasts, person_rows, verification = [], [], {}
for name, candidate, comparator in comparisons:
    for metric in metrics:
        delta = difference(candidate, comparator, metric)
        x = delta.to_numpy()
        conditional, crossed = interval(x)
        person = delta.mean(axis=1)
        row = dict(comparison=name, candidate=candidate, comparator=comparator,
                   metric=metric, improvement=float(x.mean()),
                   conditional_low=float(conditional[0]), conditional_high=float(conditional[1]),
                   crossed_low=float(crossed[0]), crossed_high=float(crossed[1]),
                   people_improved=int((person > 0).sum()), people_worsened=int((person < 0).sum()),
                   loo_min=float(min(person.drop(p).mean() for p in person.index)),
                   loo_max=float(max(person.drop(p).mean() for p in person.index)))
        row.update({f"seed_{s}": float(v) for s, v in delta.mean().items()})
        contrasts.append(row)
        for p, v in person.items():
            person_rows.append(dict(comparison=name, metric=metric, canonical_person_id=p, improvement=float(v)))
        if name == "registered" and metric in saved:
            old = saved[metric]
            errors = [abs(old["improvement"]-row["improvement"]),
                      np.max(abs(np.array(old["person_conditional_ci95"])-conditional)),
                      np.max(abs(np.array(old["crossed_person_seed_ci95"])-crossed))]
            errors += [abs(old["per_seed_improvement"][str(s)]-v) for s, v in delta.mean().items()]
            assert max(errors) < 1e-10, (metric, errors)
            verification[metric] = {"max_absolute_reconstruction_difference": float(max(errors))}
contrast_table = pd.DataFrame(contrasts)
contrast_table.to_csv(OUT / "paired-contrasts.csv", index=False, float_format="%.12g")
pd.DataFrame(person_rows).to_csv(OUT / "person-effects.csv", index=False, float_format="%.12g")

# Factorial descriptive interaction: how adding the change objective changes
# JEPA's advantage over direct training. Positive favors the JEPA change effect.
interaction = []
for metric in metrics:
    x = (difference(J+pair, J+base, metric) - difference(D+pair, D+base, metric)).to_numpy()
    conditional, crossed = interval(x)
    interaction.append(dict(metric=metric, interaction=float(x.mean()),
                            crossed_low=float(crossed[0]), crossed_high=float(crossed[1])))
pd.DataFrame(interaction).to_csv(OUT / "objective-interaction.csv", index=False, float_format="%.12g")

coverage["failed"] = coverage.reference_eligible - coverage.successful
coverage["success_percent"] = 100 * coverage.successful / coverage.reference_eligible
coverage.to_csv(OUT / "coverage-by-seed.csv", index=False, float_format="%.12g")
pooled = coverage.groupby("method")[["records", "reference_eligible", "successful", "failed"]].sum()
pooled["success_percent"] = 100*pooled.successful/pooled.reference_eligible
pooled.to_csv(OUT / "coverage-pooled.csv", float_format="%.12g")

source = df.copy()
source["source_collection"] = source.canonical_person_id.str.split("::").str[0]
source.groupby(["method", "source_collection"])[metrics].mean().to_csv(OUT / "source-means.csv", float_format="%.12g")

report_errors = []
for line in (SOURCE / "report.md").read_text().splitlines():
    cells = [s.strip() for s in line.split("|")[1:-1]]
    if len(cells) == 5 and cells[0] in means.index:
        expected = means.loc[cells[0], ["A_error", "response_error", "nuisance_error", "waveform_error"]].to_numpy()
        observed = np.array([float(s) for s in cells[1:]])
        assert np.max(abs(expected-observed)) <= .0000500001
        report_errors.append(float(np.max(abs(expected-observed))))
assert len(report_errors) == 16
assert np.allclose(means.loc["unchanged"], means.loc["filter0"], rtol=0, atol=1e-12)
assert (means.idxmin() == D+base).all()
deterministic = ["unchanged", "joint_offset", "joint_affine", "filter0", "filter1", "filter2"]
for method in deterministic:
    assert (df[df.method.eq(method)].groupby("canonical_person_id")[metrics].nunique() == 1).all().all()

files = ["config.json", "report.md", "evaluation/comparisons.json", "evaluation/summary.json",
         "evaluation/per-person.csv", "evaluation/coverage.csv"]
missing = ["ledger.json", "plan.json", "cohort/manifest.json", "evaluation/per-window.csv",
           "evaluation/responses.csv", "evaluation/nuisance.csv", "evaluation/interaction.csv",
           "evaluation/per-family.csv", "evaluation/by-condition-person.csv", "evaluation/coverage-per-person.csv"]
manifest = dict(
    analysis_date="2026-09-24", inputs={s:hashlib.sha256((SOURCE/s).read_bytes()).hexdigest() for s in files},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    python=sys.version, numpy=np.__version__, pandas=pd.__version__, matplotlib=matplotlib.__version__,
    source_directory=str(SOURCE.relative_to(ROOT)), rows=len(df), methods=len(means), people=14, seeds=[17,29,43],
    bootstrap_draws=2000, bootstrap_seed=731,
    registered_reconstruction=verification, report_rows_verified=len(report_errors),
    maximum_report_rounding_difference=max(report_errors),
    all_metrics_finite=True, complete_person_seed_panel=True, reference_eligible_records_per_method_seed=55800,
    source_people={k:int(v) for k,v in source.drop_duplicates("canonical_person_id").source_collection.value_counts().items()},
    unavailable_local_artifacts=[s for s in missing if not (SOURCE/s).exists()],
    scope="Person-table reconstruction only; no raw-prediction, cohort-admission, or failure-penalty reconstruction.",
    secondary_comparisons="Exploratory; descriptive intervals without multiplicity adjustment; no clinical margin or independent confirmation.",
)
(OUT / "verification.json").write_text(json.dumps(manifest, indent=2)+"\n")

plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10,
                     "axes.spines.top":False, "axes.spines.right":False, "svg.fonttype":"none"})
fig, ax = plt.subplots(figsize=(9.4, 4.1), layout="constrained")
order = ["response_error", "nuisance_error", "A_error", "waveform_error"]
labels = ["Movement-response error (primary)", "Observation-induced error", "Excursion-difference error", "Angle-waveform error"]
for y, metric in enumerate(order):
    r = contrast_table[(contrast_table.comparison == "registered") & (contrast_table.metric == metric)].iloc[0]
    color = "#166a76" if r.improvement > 0 else "#a34831"
    ax.plot([r.crossed_low, r.crossed_high], [y,y], color=color, lw=2)
    ax.scatter(r.improvement, y, color=color, s=40, zorder=3)
    ax.text(6.4, y, f"{r.improvement:+.2f}  [{r.crossed_low:+.2f}, {r.crossed_high:+.2f}]", va="center", fontsize=10)
ax.axvline(0, color="#777777", lw=1, linestyle="--")
ax.set_yticks(range(4), labels)
ax.invert_yaxis()
ax.set_xlim(-8, 13)
ax.set_xticks([-8,-6,-4,-2,0,2,4,6])
ax.set_xlabel("Direct + change error − JEPA + change error (image-plane degrees)")
ax.set_title("Core comparison: a nuisance benefit with uncertain response gain", loc="left", weight="bold", pad=15)
fig.text(.36, -.02, "Positive favors JEPA. Descriptive 95% crossed person/seed intervals; 14 people, 3 seeds.", fontsize=9)
for ext in ["png", "svg"]:
    fig.savefig(OUT / f"primary-contrasts.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)

families = [("Direct",D,"#205d96"),("JEPA",J,"#bc593e"),("Coordinate pretraining",C,"#42846a"),
            ("Initialized features",I,"#8055a0"),("Shuffled JEPA",S,"#8b762f")]
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), layout="constrained")
for ax, metric, title in zip(axes,["response_error","waveform_error"], ["Movement-response error","Angle-waveform error"]):
    for label, prefix, color in families:
        vals=[means.loc[prefix+o,metric] for o in [base,pair]]
        ax.plot([0,1], vals, "o-", color=color, label=label, lw=1.7, ms=5)
    ax.set_xticks([0,1], ["Coordinate objective", "+ Paired-change objective"])
    ax.set_xlim(-.14,1.14)
    ax.set_title(title, loc="left", weight="bold")
    ax.set_ylabel("Image-plane degrees; lower is better")
    ax.grid(axis="y", color="#dddddd", alpha=.7)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, -.015), ncol=3, fontsize=9, frameon=False)
fig.suptitle("The added change objective worsens waveform error in every family", weight="bold", fontsize=12)
fig.text(.02,-.15,"Person- and seed-balanced development means; 14 people, 3 seeds. Lines are descriptive, without uncertainty bars.",fontsize=9)
for ext in ["png","svg"]:
    fig.savefig(OUT / f"objective-tradeoffs.{ext}", dpi=180, bbox_inches="tight")
plt.close(fig)
print(json.dumps({"output":str(OUT),"verified_saved_comparisons":list(verification),"methods":len(means),"people":14,"seeds":3},indent=2))
