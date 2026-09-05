"""Verify saved results and generate manuscript figures without retraining.

Run audit_advanced_artifacts.py first to refresh its independently checked JSON.
All new outputs stay under neurips-genai4health/docs. Upstream files are read only.
"""
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

DOCS = Path(__file__).resolve().parents[1]
ROOT = DOCS.parents[1]
ART = ROOT / "work/artifacts/real"
FIG = DOCS / "figures"
FIG.mkdir(exist_ok=True)
TABLES = DOCS / "evidence"
TABLES.mkdir(exist_ok=True)
inputs = []

def read(path, kind="json"):
    inputs.append(path)
    if kind == "csv":
        return pd.read_csv(path)
    return json.loads(path.read_text(encoding="utf-8"))

prefix = "readout_outer_fold_0_seed_42_jepa_vicreg"
contract = read(ART / f"{prefix}_contract.json")
metrics = read(ART / f"{prefix}_metrics.csv", "csv")
pred = read(ART / f"{prefix}_source_predictions.csv", "csv")
sidecar = read(ART / "checkpoints/sjepa_outer_fold_0_seed_42_jepa_vicreg.json")
checkpoint = ART / "checkpoints/sjepa_outer_fold_0_seed_42_jepa_vicreg.pt"
inputs.append(checkpoint)
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == contract["checkpoint_sha256"] == sidecar["checkpoint_sha256"]
assert sidecar["objective"]["vicreg_variance_weight"] == 0.10
assert sidecar["objective"]["vicreg_covariance_weight"] == 0.01
assert sidecar["objective"]["group_loss_enabled"] is False
roles = {k: set(v) for k, v in sidecar["role_video_ids"].items()}
assert not (roles["train"] & roles["validation"] or roles["train"] & roles["test"] or roles["validation"] & roles["test"])
labels = ["normal", "parkinsons", "stroke", "myopathic", "cerebralpalsy"]
for _, row in metrics.iterrows():
    p = pred.loc[pred.lane.eq(row.lane)]
    assert len(p) == 20 and p.video_id.nunique() == 20
    assert set(p.video_id) == roles["test"]
    for metric, actual in {
        "accuracy": accuracy_score(p.condition, p.prediction),
        "balanced_accuracy": balanced_accuracy_score(p.condition, p.prediction),
        "macro_f1": f1_score(p.condition, p.prediction, labels=labels, average="macro", zero_division=0),
    }.items():
        assert np.isclose(actual, row[f"test_source_{metric}"], atol=1e-12)
    matrix = pd.DataFrame(confusion_matrix(p.condition, p.prediction, labels=labels), index=labels, columns=labels)
    matrix.to_csv(TABLES / f"{row.lane}_source_confusion.csv")
metrics.to_csv(TABLES / "source_readout_metrics.csv", index=False)

census = read(ART / "evaluation_protocol/source_video_census.csv", "csv")
decoded = read(ART / "evaluation_protocol/decoded_frame_census.csv", "csv")
qc = read(ART / "evaluation_protocol/pose_qc_eligibility_outer_fold_0.csv", "csv")
good = qc.loc[qc.pose_qc_eligible.eq(True)].copy()
cohort = []
for condition in labels:
    r = census.loc[(census.condition == condition) & (census.gate == "raw_annotation")].iloc[0]
    p = census.loc[(census.condition == condition) & (census.gate == "metadata_public")].iloc[0]
    d = decoded.loc[decoded.condition == condition].iloc[0]
    g = good.loc[good.condition == condition]
    cohort.append({"condition": condition, "raw_sequences": int(r.sequences), "raw_sources": int(r.unique_videos),
                   "public_sequences": int(p.sequences), "public_sources": int(p.unique_videos),
                   "decoded_sequences": int(d.sequences), "decoded_sources": int(d.unique_videos),
                   "qc_sequences": len(g), "qc_sources": g.video_id.nunique()})
cohort = pd.DataFrame(cohort)
assert cohort.raw_sequences.sum() == 666 and cohort.public_sequences.sum() == 657
assert cohort.decoded_sequences.sum() == 655 and len(good) == 639 and good.video_id.nunique() == 97
cohort.to_csv(TABLES / "cohort_by_annotation.csv", index=False)
role_counts = good.groupby("split_role").agg(sequences=("sequence_id", "size"), sources=("video_id", "nunique"))
role_counts.to_csv(TABLES / "fold0_role_counts.csv")

advanced = read(DOCS / "review/advanced_artifact_checks.json")
details = pd.DataFrame(advanced["final_source_details"]["validation"])
details = details.sort_values(["sequences", "mean_cosine"], ascending=[False, False]).reset_index(drop=True)
clip_average = float(np.average(details.mean_cosine, weights=details.sequences))
source_average = float(details.mean_cosine.mean())
assert details.sequences.sum() == 64 and len(details) == 5 and details.sequences.max() == 60
assert np.isclose(clip_average, 0.8890614510) and np.isclose(source_average, 0.7010577321)
anonymous_details = details.drop(columns="video_id")
anonymous_details.insert(0, "source", ["A", "B", "C", "D", "E"])
anonymous_details.to_csv(TABLES / "validation_normal_source_weighting.csv", index=False)
drift = read(ART / "fold_evaluation/outer_fold_0/normal_anchor_drift_seed_42.json")
pd.DataFrame(drift["development_drift"]).to_csv(TABLES / "normal_anchor_trajectory.csv", index=False)
temporal = read(ART / "fold_evaluation/outer_fold_0/temporal_readout_seed_42.csv", "csv")
temporal.to_csv(TABLES / "supplementary_temporal_metrics.csv", index=False)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.titleweight": "bold", "savefig.dpi": 220})
blue, orange, gray = "#176B87", "#B65B24", "#68737D"
def save(fig, name):
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(5.5, 3.85))
display_names = {"sjepa_latent": "Skeleton JEPA", "missingness_only": "Missingness", "raw_kinematics": "Raw kinematics"}
order = ["sjepa_latent", "missingness_only", "raw_kinematics"]
m = metrics.set_index("lane").loc[order]
ax = axes[0]
for offset, col, marker, color, label in [(0.19, "test_source_macro_f1", "o", blue, "Macro-F1"),
                                       (-0.19, "test_source_balanced_accuracy", "s", orange, "Balanced accuracy")]:
    vals = m[col].to_numpy()
    y = np.arange(3) + offset
    ax.scatter(vals, y, marker=marker, color=color, s=39, label=label, zorder=3)
    for xx, yy in zip(vals, y):
        ax.text(xx + 0.014, yy, f"{xx:.3f}", va="center", fontsize=9, color=color)
ax.set_yticks(np.arange(3), [display_names[s] for s in order]); ax.invert_yaxis()
ax.set_xlim(0, 0.57); ax.set_ylim(2.5, -0.55); ax.set_xlabel("Source-level score")
ax.grid(axis="x", alpha=.15); ax.set_title("A  Readout on 20 test sources", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.02, 1.50), ncol=2)
ax = axes[1]
ys = [1, 0]
values = [clip_average, source_average]
ax.plot(values, ys, color="#CBD4DA", linewidth=2, zorder=1)
ax.scatter(values, ys, c=[orange, blue], s=60, zorder=3)
for xx, yy in zip(values, ys):
    ax.text(xx + .018, yy, f"{xx:.3f}", va="center", fontsize=10)
ax.set_yticks(ys, ["Equal clip weight", "Equal source weight"])
ax.set_xlim(.45, 1.02); ax.set_ylim(-.5, 1.6)
ax.set_xlabel("Cosine to normal-only checkpoint")
ax.set_title("B  Same 64 normal-validation clips", loc="left", fontsize=11)
ax.text(.46, 1.36, "One source supplies 60 of 64 clips", fontsize=9, color=gray)
ax.grid(axis="x", alpha=.15)
fig.subplots_adjust(left=.34, right=.97, top=.87, bottom=.14, hspace=1.0)
save(fig, "audited_findings")

fig, axes = plt.subplots(1, 2, figsize=(9, 2.7))
stage_labels = ["Annotated", "Metadata\npublic", "Decoded", "Pose QC"]
for ax, vals, title, color in [(axes[0], [666, 657, 655, 639], "Sequences", blue),
                               (axes[1], [103, 100, 98, 97], "Source videos", orange)]:
    ax.bar(np.arange(4), vals, color=color, width=.62)
    for x, v in enumerate(vals):
        ax.text(x, v + max(vals)*.025, str(v), ha="center", fontsize=11)
    ax.set_xticks(np.arange(4), stage_labels); ax.set_ylim(0, max(vals)*1.16)
    ax.set_title(title, loc="left", fontsize=11); ax.set_axisbelow(True); ax.grid(axis="y", alpha=.13)
fig.tight_layout(w_pad=3)
save(fig, "cohort_attrition")

fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), gridspec_kw={"width_ratios": [1, 1.25]})
ax = axes[0]
ax.bar(anonymous_details.source, anonymous_details.sequences, color=[orange, blue, blue, blue, blue])
for i, n in enumerate(anonymous_details.sequences): ax.text(i, n+1, str(n), ha="center")
ax.set_ylim(0, 68); ax.set_ylabel("Validation-normal clips"); ax.set_xlabel("Source aliases (local to this figure)")
ax.set_title("A  Concentration", loc="left", fontsize=11)
ax = axes[1]
df = pd.DataFrame(drift["development_drift"])
for role, color, label in [("train", gray, "Training: 18 normal sources"), ("validation", blue, "Validation: 5 normal sources")]:
    d = df.loc[df.role.eq(role)]
    ax.plot(d.stage, d.source_equal_anchor_cosine, "o-", color=color, label=label)
ax.set_xticks(range(5), ["Normal", "+P", "+S", "+M", "+CP"])
ax.set_ylabel("Mean source cosine to stage 0"); ax.set_ylim(.5, 1.035)
ax.set_xlabel("Cumulative annotation-ordered training stage")
ax.set_title("B  Coordinate retention (descriptive)", loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=8, loc="lower left"); ax.grid(alpha=.13)
fig.tight_layout(w_pad=3)
save(fig, "source_weighting_and_drift")

manifest = {
    "audit_date": "2026-09-05", "python": platform.python_version(),
    "checkpoint_sha256": contract["checkpoint_sha256"],
    "manifest_sha256": contract["manifest_sha256"], "split_sha256": contract["split_sha256"],
    "checks": {"readout_metrics_recomputed": True, "checkpoint_hash": True,
               "source_id_roles_disjoint": True, "cohort_counts_recomputed": True,
               "weighting_recomputed_from_checked_source_details": True},
    "derived_values": {"validation_clip_cosine": clip_average, "validation_source_cosine": source_average,
                       "raw_minus_latent_macro_f1": float(m.loc["raw_kinematics", "test_source_macro_f1"] - m.loc["sjepa_latent", "test_source_macro_f1"]),
                       "dominant_validation_source_clip_share": 60/64},
    "inputs": [{"path": str(p.relative_to(ROOT)).replace("\\", "/"),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
    "limitations": ["Recomputes saved evidence; does not rerun training.",
                    "Hashes cannot prove historical test secrecy or person independence.",
                    "One fold and seed; no statistical superiority claim."]}
(TABLES / "verification_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"checks": manifest["checks"], "derived_values": manifest["derived_values"]}, indent=2))
