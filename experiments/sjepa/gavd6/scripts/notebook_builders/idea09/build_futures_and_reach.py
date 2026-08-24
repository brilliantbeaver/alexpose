"""Build the Idea 09 futures-and-reach notebook.

Run: uv run python scripts/notebook_builders/idea09/build_futures_and_reach.py
Creates the Idea 9 counterpart to nb_05b: possible futures, decision logic,
and no-download AMASS / MoVi reach scaffolds.
"""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[3]
NB_PATH = ROOT / "notebooks" / "experiments" / "idea09_reflection_equivariance" / "02_futures_and_reach.ipynb"

_CELL_N = [0]

def _next_id(prefix):
    _CELL_N[0] += 1
    return f"{prefix}{_CELL_N[0]:02d}"

def md(source):
    return {"cell_type": "markdown", "id": _next_id("md"), "metadata": {},
            "source": source.strip("\n").splitlines(keepends=True)}

def code(source):
    return {"cell_type": "code", "id": _next_id("code"), "execution_count": None, "metadata": {},
            "outputs": [], "source": source.strip("\n").splitlines(keepends=True)}

cells = [
md("""# Notebook 09b: Possible futures and reach scaffolds for Idea 9

Notebook 09a proves the paired architecture contract. This notebook does the work that must happen
*before* real results arrive:

1. It turns six plausible futures into one deterministic decision rule, so exact geometry, a healthy
   representation, an architecture gain, an invalid comparison, and a real-camera failure cannot be
   conflated after the fact.
2. It defines auditable, no-download manifests for AMASS generic pretraining and MoVi calibrated-view
   evaluation. The synthetic fixtures exercise the split and metric code only; they are not dataset
   results.

The notebook mirrors Idea 05's `02_futures_and_reach.ipynb`. See the binding protocol at
`notes/research/ideas/09-reflection-equivariant-symmetry-axis/METHODOLOGY.md`.
"""),
md("""## 0. Frozen decision inputs and data-free startup

This follows 05b's startup convention: it needs no pose cache, checkpoint, AMASS download, or MoVi
download. It resolves only the Idea 9 directory, so its futures and split-manifest checks run anywhere
that 05b runs. A real experiment still records a practical signed-target margin before held-out outcomes
are opened; the illustrative normalized margin below is not a clinical threshold.
"""),
code("""from pathlib import Path
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM_SEED = 19
rng = np.random.default_rng(RANDOM_SEED)
MODE = os.getenv("GAVD_MODE", "smoke").strip().lower()
if MODE not in {"smoke", "real"}:
    raise ValueError("GAVD_MODE must be smoke or real")

def project_root():
    '''Resolve the gavd6 root without requiring a GAVD artifact tree.'''
    for base in [Path.cwd(), *Path.cwd().parents]:
        candidate = base / "notes" / "research" / "ideas" / "09-reflection-equivariant-symmetry-axis"
        if candidate.exists():
            return base
    return Path.cwd()

PROJECT_DIR = project_root()
IDEA9_DIR = PROJECT_DIR / "notes" / "research" / "ideas" / "09-reflection-equivariant-symmetry-axis"
RUN_ID = os.getenv("IDEA9_RUN_ID", f"idea9-{MODE}")
OUT_DIR = Path(os.getenv(
    "IDEA9_OUTPUT_DIR", PROJECT_DIR / "artifacts" / "research" / "idea09" / RUN_ID
)).expanduser()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Illustrative normalized error units for the simulator only.
SIMULATED_PRACTICAL_MARGIN = 0.02
CONTRACT = {
    "mode": MODE,
    "run_id": RUN_ID,
    "data_requirement": "none; this is a simulator and manifest scaffold like 05b",
    "simulated_practical_margin": SIMULATED_PRACTICAL_MARGIN,
    "real_rule": ("A real run must load its pre-registered participant-level margin, interval method, "
                  "and matching manifest; simulated values are not empirical thresholds."),
}
print(f"IDEA9_DIR: {IDEA9_DIR}")
print("Data requirement: no checkpoint, pose cache, AMASS download, or MoVi download.")
print(json.dumps(CONTRACT, indent=2))
"""),
md("""## 1. One decision function for every future

The architecture conclusion requires all of these conditions: numerical geometry passes, representation
health passes, the exposure/compute manifest is credible, and the two paired effects both beat the
pre-registered practical margin in the favourable direction. Failure at an earlier gate stops the
claim ladder rather than being averaged into a performance result.
"""),
code("""def score_future(*, geometry, health, matched, delta_vs_odd, delta_vs_pair,
                 practical_margin, movi_view_stable=None, nuisance_clean=True):
    # Negative error deltas favour equivariant_encoder.
    architecture_gain = (
        geometry and health and matched and nuisance_clean
        and delta_vs_odd <= -practical_margin
        and delta_vs_pair <= -practical_margin
    )
    if not geometry or not matched or not nuisance_clean:
        level = "F5_INVALID_EVIDENCE"
        claim = "No scientific comparison: repair contract, matching, leakage, or nuisance failure."
    elif not health:
        level = "F4_EXACT_BUT_COLLAPSED"
        claim = "Geometry may be correct, but the odd representation is not informative."
    elif architecture_gain:
        level = "F1_INTERIOR_ADVANTAGE"
        claim = "Encoder-wide reflection structure helps beyond output repair and paired fusion."
    elif delta_vs_odd <= -practical_margin and delta_vs_pair > -practical_margin:
        level = "F3_GENERIC_PAIR_ADVANTAGE"
        claim = "Linked paired fusion helped; swap-preserving ties did not show an added benefit."
    else:
        level = "F2_GEOMETRY_WITHOUT_UTILITY"
        claim = "The exact, healthy interior did not beat cheap output repair at the stated margin."
    if movi_view_stable is False:
        camera = "F6_REAL_CAMERA_FAILURE: do not claim real-view robustness."
    elif movi_view_stable is True:
        camera = "C5_VIEW_TEST_PASSES: non-clinical real-view stability only."
    else:
        camera = "C5_NOT_TESTED"
    return {
        "future": level,
        "architecture_gain": bool(architecture_gain),
        "camera": camera,
        "licensed_claim": claim,
    }

# Sanity checks for the two outcomes easiest to accidentally conflate.
assert score_future(geometry=True, health=True, matched=True, nuisance_clean=True,
                    delta_vs_odd=-.04, delta_vs_pair=-.03,
                    practical_margin=.02)["future"] == "F1_INTERIOR_ADVANTAGE"
assert score_future(geometry=True, health=True, matched=True, nuisance_clean=True,
                    delta_vs_odd=-.04, delta_vs_pair=-.003,
                    practical_margin=.02)["future"] == "F3_GENERIC_PAIR_ADVANTAGE"
print("Decision-function self-check passes.")
"""),
md("""## 2. Six possible futures

The values below are deliberately synthetic expected shapes. They specify the claim consequences before
a model is selected or a test outcome is inspected; they are not an estimate of what GAVD, AMASS, or
MoVi will show.
"""),
code("""FUTURES = [
    dict(name="F1 interior advantage", geometry=True, health=True, matched=True, nuisance_clean=True,
         delta_vs_odd=-.045, delta_vs_pair=-.031, movi_view_stable=True),
    dict(name="F2 geometry without utility", geometry=True, health=True, matched=True, nuisance_clean=True,
         delta_vs_odd=-.005, delta_vs_pair=-.004, movi_view_stable=None),
    dict(name="F3 generic pair advantage", geometry=True, health=True, matched=True, nuisance_clean=True,
         delta_vs_odd=-.039, delta_vs_pair=-.003, movi_view_stable=None),
    dict(name="F4 exact but collapsed", geometry=True, health=False, matched=True, nuisance_clean=True,
         delta_vs_odd=-.060, delta_vs_pair=-.050, movi_view_stable=None),
    dict(name="F5 invalid evidence", geometry=False, health=True, matched=False, nuisance_clean=False,
         delta_vs_odd=-.070, delta_vs_pair=-.060, movi_view_stable=None),
    dict(name="F6 real-camera failure", geometry=True, health=True, matched=True, nuisance_clean=True,
         delta_vs_odd=-.041, delta_vs_pair=-.025, movi_view_stable=False),
]
rows = []
for f in FUTURES:
    verdict = score_future(**{k: v for k, v in f.items() if k != "name"},
                           practical_margin=SIMULATED_PRACTICAL_MARGIN)
    rows.append({**f, **verdict})
future_table = pd.DataFrame(rows)
future_table[["name", "future", "licensed_claim", "camera"]]
"""),
md("""## 3. Expected-shape panels and decision table

The left panel shows the two co-primary effects. The dashed lines are the illustrative practical margin;
only the lower-left quadrant beyond both lines is an architecture win. The right panel makes clear that
failed geometry or health vetoes performance-looking values.
"""),
code("""import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
colors = {
    "F1_INTERIOR_ADVANTAGE": "#16856a",
    "F2_GEOMETRY_WITHOUT_UTILITY": "#756bb1",
    "F3_GENERIC_PAIR_ADVANTAGE": "#e07a4b",
    "F4_EXACT_BUT_COLLAPSED": "#c43d4b",
    "F5_INVALID_EVIDENCE": "#555555",
}
for _, r in future_table.iterrows():
    c = colors.get(r["future"], "#4b84b4")
    axes[0].scatter(r["delta_vs_odd"], r["delta_vs_pair"], s=95, c=c, zorder=3)
    axes[0].annotate(r["name"].split(" ")[0], (r["delta_vs_odd"], r["delta_vs_pair"]),
                     xytext=(4, 5), textcoords="offset points", fontsize=8)
axes[0].axvline(-SIMULATED_PRACTICAL_MARGIN, ls="--", color="black", lw=1)
axes[0].axhline(-SIMULATED_PRACTICAL_MARGIN, ls="--", color="black", lw=1)
axes[0].axvline(0, color="#d9d9d9", lw=.8); axes[0].axhline(0, color="#d9d9d9", lw=.8)
axes[0].set(xlabel="equivariant MAE − odd_output MAE", ylabel="equivariant MAE − paired_unconstrained MAE",
            title="Simulated co-primary effects (negative favours equivariant)")

gate_frame = future_table[["name", "geometry", "health", "matched", "nuisance_clean"]].set_index("name").astype(int)
axes[1].imshow(gate_frame.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
axes[1].set_xticks(range(len(gate_frame.index)), [x.split(" ")[0] for x in gate_frame.index])
axes[1].set_yticks(range(len(gate_frame.columns)), gate_frame.columns)
axes[1].set_title("Gate matrix: 1 = pass, 0 = veto")
for y in range(gate_frame.shape[1]):
    for x in range(gate_frame.shape[0]):
        axes[1].text(x, y, str(gate_frame.iloc[x, y]), ha="center", va="center", fontsize=9)
fig.tight_layout()
figure_path = OUT_DIR / "idea9_possible_futures.png"
fig.savefig(figure_path, dpi=140)
from IPython.display import Image, display
display(Image(filename=str(figure_path)))
plt.show()

pd.set_option("display.max_colwidth", 100)
print(future_table[["name", "future", "licensed_claim", "camera"]].to_string(index=False))
"""),
md("""## 4. AMASS pretraining split and overlap audit scaffold

AMASS is for broad, non-clinical label-free motion pretraining. Split subjects before cutting windows,
retain source-dataset and subject provenance, and reject any source/subject that overlaps a downstream
cohort. This interface has no download logic and is tested only on a tiny synthetic manifest.
"""),
code("""AMASS_REQUIRED = {"source_dataset", "subject_id", "motion_id", "split"}

def validate_amass_manifest(records, downstream_people=()):
    frame = pd.DataFrame(records).copy()
    missing = AMASS_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"AMASS manifest missing columns: {sorted(missing)}")
    if set(frame["split"]) - {"train", "validation", "test"}:
        raise ValueError("AMASS split values must be train, validation, or test.")
    # The safe identity is source dataset + subject, not a window ID.
    identities = frame["source_dataset"].astype(str) + "::" + frame["subject_id"].astype(str)
    frame = frame.assign(identity=identities)
    split_count = frame.groupby("identity")["split"].nunique()
    if (split_count > 1).any():
        leaked = split_count[split_count > 1].index.tolist()
        raise ValueError(f"Subject split leakage: {leaked[:5]}")
    downstream = {str(x) for x in downstream_people}
    overlap = sorted(set(frame["identity"]) & downstream)
    if overlap:
        raise ValueError(f"Known downstream overlap in AMASS pretraining: {overlap[:5]}")
    return frame

amass_smoke = [
    {"source_dataset": "fixture_A", "subject_id": "s01", "motion_id": "walk_01", "split": "train"},
    {"source_dataset": "fixture_A", "subject_id": "s02", "motion_id": "walk_02", "split": "train"},
    {"source_dataset": "fixture_B", "subject_id": "s03", "motion_id": "walk_03", "split": "validation"},
]
amass_validated = validate_amass_manifest(amass_smoke)
print(amass_validated[["identity", "motion_id", "split"]].to_string(index=False))
print("AMASS smoke manifest passes. Real manifest must be audited before window cutting.")
"""),
md("""## 5. MoVi actor-held-out calibrated-view scaffold

MoVi is a distinct test: a body-frame signed output should stay stable across synchronized real camera
views while anatomical reflection flips it. The actor, not a view or trial, is the independent split
unit. The fixture below only verifies that the grouping and metrics work; it is not a substitute for
MoVi data.
"""),
code("""MOVI_REQUIRED = {"actor_id", "motion_id", "camera_id", "split", "body_frame_score"}

def validate_movi_manifest(records):
    frame = pd.DataFrame(records).copy()
    missing = MOVI_REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"MoVi manifest missing columns: {sorted(missing)}")
    actor_splits = frame.groupby("actor_id")["split"].nunique()
    if (actor_splits > 1).any():
        raise ValueError(f"Actor leakage across splits: {actor_splits[actor_splits > 1].index.tolist()}")
    if frame.groupby(["actor_id", "motion_id"])["camera_id"].nunique().max() < 2:
        raise ValueError("Need at least two calibrated views per actor-motion for a view test.")
    return frame

movi_smoke = []
for actor in range(6):
    latent = 1.0 if actor % 2 == 0 else -1.0
    for camera, perturbation in [("cam_01", .01), ("cam_02", -.012), ("cam_03", .006)]:
        movi_smoke.append({
            "actor_id": f"actor_{actor:02d}", "motion_id": "walk_01", "camera_id": camera,
            "split": "test", "body_frame_score": latent + perturbation,
        })
movi_frame = validate_movi_manifest(movi_smoke)
wide = movi_frame.pivot(index=["actor_id", "motion_id"], columns="camera_id", values="body_frame_score")
pairwise = []
for left, right in [("cam_01", "cam_02"), ("cam_01", "cam_03"), ("cam_02", "cam_03")]:
    pairwise.append({"views": f"{left} / {right}",
                     "pearson_r": float(np.corrcoef(wide[left], wide[right])[0, 1]),
                     "mean_abs_difference": float((wide[left] - wide[right]).abs().mean())})
print(pd.DataFrame(pairwise).to_string(index=False))
print("MoVi fixture passes actor-held-out grouping. A real run additionally records calibration and visibility.")
"""),
md("""## 6. Persist the futures and reach bundle

The output identifies every quantity as simulated or scaffolded. It gives the later real run a stable
decision schema to populate, without allowing synthetic values to masquerade as AMASS, MoVi, or clinical
findings.
"""),
code("""bundle = {
    "notebook": "nb_09b_equivariant_futures_and_reach",
    "contract": CONTRACT,
    "futures": future_table.to_dict(orient="records"),
    "amass": {
        "status": "synthetic manifest scaffold only; no data downloaded",
        "required_columns": sorted(AMASS_REQUIRED),
        "rules": ["split people before windows", "retain source/subject provenance", "exclude downstream overlaps"],
    },
    "movi": {
        "status": "synthetic calibrated-view scaffold only; no data downloaded",
        "required_columns": sorted(MOVI_REQUIRED),
        "rules": ["actor-held-out split", "views are correlated not independent", "test body-frame stability separately from anatomical mirror"],
        "fixture_metrics": pairwise,
    },
    "figure": str(figure_path),
    "scope": "All outcome numbers in this bundle are simulated or synthetic-fixture checks.",
}
bundle_path = OUT_DIR / "idea9_futures_and_reach.json"
bundle_path.write_text(json.dumps(bundle, indent=2))
print(f"Wrote {bundle_path}")
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
NB_PATH.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH} ({len(cells)} cells)")
