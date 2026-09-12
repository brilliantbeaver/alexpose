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
and [execution guide](../../slurm/future-innovation/SOURCE_LEARNING_CURVE.md). New runs use the
[available-cohort amendment](../../docs/studies/future-innovation/source-learning-curve-available-cohort-protocol.md);
the saved configuration below identifies the policy for this run. The repaired
50-clip STOP remains a completed historical result. Missing expanded processing
is an execution state and is not another negative scientific result.'''),
    code('''from pathlib import Path
import json
import hashlib
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
if (RUN / "config/study.json").exists():
    study = json.loads((RUN / "config/study.json").read_text())
    print("Frozen run ID:", study.get("run_id", "not recorded"))
    print("Synthetic study:", study.get("synthetic", "not recorded"))
    print("Cohort policy:", study.get("cohort_policy", "historical strict development-media policy"))
def read(relative):
    return json.loads((RUN / relative).read_text())
def file_hash(relative):
    path = RUN / relative
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
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
    print("Inventory at reservation time — verified counts here precede expanded processing:")
    display(pd.Series({k: audit[k] for k in ["annotated_sequences", "recordings", "parent_candidates", "verified_eligible_sequences", "explicitly_identified_participants", "total_participants"]}, dtype=object).to_frame("count"))
    display(pd.DataFrame(audit["reservation"]).T)
    print(audit["participant_status"])
else:
    print("No frozen expanded inventory is present locally.")
if (RUN / "config/availability-contract.json").exists():
    availability = read("config/availability-contract.json")
    print("Frozen available-cohort selection (file availability precedes pose eligibility):")
    display(pd.Series(availability["summary"], dtype=object).to_frame("saved value"))
    inventory = pd.read_csv(RUN / "config/media-availability.csv")
    omitted = inventory.loc[inventory.role.eq("development") & ~inventory.included]
    print("Unavailable development recordings excluded before processing:", len(omitted))
    display(omitted[["video_id", "annotated_sequences", "exclusion_reason"]])
    print("Confirmation roles and source folds were assigned before filtering and remain unchanged.")
    print("Only config/processing-sequences.csv and config/processing-videos.csv feed new candidate construction.")
if (RUN / "data/cohort-complete.json").exists():
    prepared = read("data/cohort-complete.json")
    print("Saved completed preparation:")
    display(pd.Series({k: prepared.get(k) for k in ["eligible_windows", "eligible_sources", "failed_pose_windows", "confirmation_processed"]}, dtype=object).to_frame("saved value"))
    windows = pd.read_csv(RUN / "data/manifests/development-windows.csv")
    display(windows.groupby("outer_fold").agg(clips=("window_id", "size"), recordings=("video_id", "nunique")))
    display(windows.groupby("evidence_origin").agg(clips=("window_id", "size"), recordings=("video_id", "nunique")))
else:
    print("Expanded pose eligibility has not been sealed. The initial verified count is not the expanded cohort size.")
if (RUN / "data/cache-complete.json").exists():
    cached = read("data/cache-complete.json")
    display(pd.Series({k: v for k, v in cached.items() if k != "artifacts"}, dtype=object).to_frame("saved cache evidence"))
else:
    print("Expanded teacher encoding has not completed. Existing gate-v2 arrays cover only their original windows.")
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
if (RUN / "manifests/learning-plan.json").exists():
    identities = read("manifests/learning-plan.json")["fits"]
    print("Selected-model completion receipts present:", sum((RUN / f"models/{i}/complete.json").is_file() for i in identities), "/", len(identities))
if (RUN / "data/logs/development-media.csv").exists():
    media = pd.read_csv(RUN / "data/logs/development-media.csv")
    if "role" in media:
        media = media.loc[media.role.eq("development")]
    unavailable = media.loc[~media.available]
    label = "Frozen availability exclusions" if (RUN / "config/availability-contract.json").exists() else "Last preparation media check: unavailable development recordings"
    print(label, "=", len(unavailable))
    if len(unavailable):
        display(unavailable[["video_id", "method", "reason"]])
attempts = [json.loads(p.read_text()) for p in sorted((RUN / "logs/stages").glob("*.json"))]
if attempts:
    display(pd.DataFrame(attempts).reindex(columns=["started_utc", "stage", "fold", "status", "returncode", "elapsed_seconds", "slurm_job_id", "log"]))
    for attempt in attempts:
        if attempt["status"] == "failed":
            print(f"Failed {attempt['stage']} attempt: {attempt['log']}")
            print(attempt.get("error", ""), attempt.get("output_tail", "")[-2000:])
else:
    print("No structured stage logs are present. Older runs may have ordinary Slurm logs only.")
print("A running record without a finish time may reflect an interrupted job; check Slurm. Old failures remain visible after retries.")
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
    if "cohort_availability" in result:
        print("Result population:", result["cohort_policy"])
        display(pd.Series(result["cohort_availability"], dtype=object).to_frame("saved cohort"))
    display(pd.DataFrame(result["arm_means"]))
    display(pd.DataFrame(result["contrasts"]))
    display(pd.DataFrame(result["per_subset_contrasts"]))
    current_binding = {p: file_hash(p) for p in ["config/study.json", "manifests/learning-plan.json", "reports/complete.json", "reports/learning-curve.json"]}
    verified = [a for a in attempts if a.get("stage") == "verify" and a.get("status") == "passed"
                and a.get("returncode") == 0 and all(current_binding.values())
                and a.get("binding_before") == current_binding and a.get("binding_after") == current_binding]
    if verified:
        print("Saved numerical-verification job succeeded for these study/plan/report bytes:", verified[-1]["finished_utc"])
        print("This notebook has not rechecked model or cache integrity. Use verify for a fresh reconstruction.")
    else:
        print("No successful numerical-verification log matches the current study/plan/report bytes.")
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
