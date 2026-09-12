"""Generate the read-only tutorial for the separately versioned source study."""
from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks/future_innovation/23_source_learning_curves.ipynb"
def render():
    nb=nbf.v4.new_notebook()
    md=nbf.v4.new_markdown_cell; code=nbf.v4.new_code_cell
    nb.cells=[
    md('''# 23 — Does more source diversity help repaired Experiment 0?

This notebook inspects the **source-learning-curve-v1** development study. It does
not fit models or encode videos. A learning curve compares the same held-out
recordings while the training set grows by whole recordings. More clips from one
recording are useful examples, but they are not more independent sources.

Read [the protocol](../../docs/studies/future-innovation/source-learning-curve-protocol.md)
and [execution guide](../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md). The repaired
50-clip STOP remains a completed historical result. Missing expanded processing
is an execution state and is not another negative scientific result.'''),
    code('''from pathlib import Path
import json
import os
import pandas as pd
from IPython.display import display
ROOT = Path(os.environ.get("GAVD6_ROOT", Path.cwd())).expanduser().resolve()
while not (ROOT / "src/gavd6_sjepa").is_dir() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
RUN = Path(os.environ.get("FI_RUN_ROOT", ROOT / "outputs/future-innovation-source-curve-dev-20260911-v2")).expanduser()
RUN = (ROOT / RUN).resolve() if not RUN.is_absolute() else RUN.resolve()
if not RUN.is_dir():
    raise FileNotFoundError(f"Study directory does not exist: {RUN}")
print("Study root:", RUN)
print("Protocol: source-learning-curve-v1; development data")
print("Inspection only: saved values and file presence; no fitting or numerical verification.")
def read(relative):
    return json.loads((RUN / relative).read_text())
'''),
    md('''## 1. Establish the denominator before counting examples

The full annotation manifest, the original candidate list, verified pose windows,
and cached teacher arrays count different things. A clip excluded because the
old 50-window cap was reached has not failed pose quality. Unknown participant
identity remains unknown; a video ID is not a participant ID.

Confirmation recordings are reserved before new pose extraction or teacher
encoding. All sequences from those recordings stay outside the development curve.
Prior experiment exposure is recorded conservatively from supplied manifests.'''),
    code('''if (RUN / "reports/cohort-audit.json").exists():
    audit = read("reports/cohort-audit.json")
    display(pd.Series({k: audit[k] for k in ["annotated_sequences", "recordings", "parent_candidates", "verified_eligible_sequences", "explicitly_identified_participants", "total_participants"]}, dtype=object).to_frame("count"))
    display(pd.DataFrame(audit["reservation"]).T)
    print(audit["participant_status"])
else:
    print("No frozen expanded inventory is present locally.")
'''),
    md('''## 2. Compare increasing training subsets on fixed evaluation sources

Five outer folds keep recordings together. Within each outer training pool,
three independently ordered source lists provide nested prefixes near 40, 80,
and 160 sources. All eligible clips from each selected recording are included.
The all-sources endpoint is fitted once per fold. Three labels pointing to that
same endpoint are not repeated optimization evidence.

Each subset has its own three inner folds, input scaling, target scaling, and
penalty selection. The solved objective uses summed weighted loss. Setting
`lambda = number_of_training_windows * rho` keeps the penalty comparable when
the amount of training data changes. The fixed rho grid is
`[0.0025, 0.025, 0.25, 2.5, 25, 250]`.'''),
    code('''if (RUN / "manifests/learning-plan.json").exists():
    plan = read("manifests/learning-plan.json")
    display(pd.DataFrame(plan["entries"]))
    print("Distinct fitted source subsets required:", len(plan["fits"]))
else:
    print("The final subset plan waits for expanded pose eligibility, cache completion, and teacher audits.")
'''),
    md('''## 3. Read improvements against the right controls

Report the RGB reference and four arms: real history, four-frame block shuffle,
different-source mismatch within the current partition, and no skeleton retaining
joint validity. The primary skeleton comparison is real minus no skeleton.
Improvement over RGB by itself can include changes to RGB regularization.

Inputs use frames 0–31. Target frames 38–39 were encoded in the full 64-frame
video; the task predicts **contextual teacher features**. This scaling study
does not change that target or claim a pure coordinate-motion effect.'''),
    code('''stages = ["data/cohort-complete.json", "data/cache-complete.json", "data/audit-complete.json", "manifests/plan-complete.json", "reports/complete.json"]
display(pd.DataFrame({"artifact": stages, "present_locally": [(RUN / p).is_file() for p in stages]}))
print("Presence is an inventory check. It does not assert integrity or numerical reconstruction.")
'''),
    md('''## 4. Interpret the curve and its uncertainty

The declared predictive R² compares errors with the applicable outer-training
mean, never the held-out mean. Its denominator changes with training size, so
the report also includes raw teacher-unit squared error. Paired bootstrap draws
resample whole recording sources and retain multiplicity, aligned across sizes,
arms and subset repetitions. Their intervals condition on saved fitted models;
they do not include retraining or adaptive research choices.

A growing matched skeleton increment can motivate confirmation. Lower overall
error with a flat matched increment motivates a separate target or representation
study. The final decision uses the frozen largest-endpoint criteria; no notebook
launches confirmation or student training.'''),
    code('''report_path = RUN / "reports/learning-curve.json"
if report_path.exists():
    result = read("reports/learning-curve.json")
    print("Saved status (not independently verified here):", result["status"])
    print("Saved measurement_complete:", result["measurement_complete"])
    print("Synthetic fixture:", result["synthetic"])
    display(pd.DataFrame(result["arm_means"]))
    display(pd.DataFrame(result["contrasts"]))
    display(pd.DataFrame(result["per_subset_contrasts"]))
    from IPython.display import SVG, display
    if (RUN / "reports/learning-curve.svg").exists():
        display(SVG(filename=str(RUN / "reports/learning-curve.svg")))
else:
    print("No learning-curve report is present in the selected study; execution is incomplete.")
'''),
    md('''## 5. Verify or resume outside this inspection notebook

Use `bash slurm/future-innovation-scaling/launch/submit.sh status` with the same
`FI_RUN_ROOT` for a frozen-input integrity check and stage inventory. After
reporting, use the launcher’s `verify` command to reconstruct
selected coefficients, predictions and paired scores. That command performs CPU
verification fits but writes no scientific artifacts. Use the execution guide
for preparation, caching, Slurm submission and interrupted-stage recovery.''')]
    nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
    for index, cell in enumerate(nb.cells):
        cell.id = f"source-curve-{index:02d}"
    return nb


if __name__ == "__main__":
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(render(), DESTINATION)
    print(f"Generated {DESTINATION.relative_to(ROOT)}")
