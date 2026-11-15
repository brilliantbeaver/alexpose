"""Build six independent real-data tutorials without changing historical notebooks.

Generated notebooks require an explicit completed and verified iteration02 run.
This script only authors notebooks; it neither executes models nor fabricates output.
"""
from pathlib import Path
import hashlib
import json
import textwrap
import nbformat as nbf

BASE = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


NAVIGATION_MARKER = "## Historical tutorial and current study"


def navigation_cell():
    return md('''## Historical tutorial and current study

This notebook preserves the earlier tutorial's code, observations and experimental scope. Its original result identities remain historical; they are not measurements of the new retained-predictor or U/J/A study. Start the current independent tutorial sequence at [Notebook 12](12_observation_population_and_video_context.ipynb), then continue through [Notebook 17](17_failure_analysis_and_manuscript_decision.ipynb). The [current protocol](docs/iteration02/protocol.md) and [current results](docs/iteration02/results.md) identify the verified new run. The [draft01 manuscript](docs/draft01/manuscript.md) remains the record of the completed earlier study.

The current primary question is normal versus the four requested condition families, with all other folders retained in a mandatory broad secondary analysis. It tests opposite-leg information beyond own-leg geometry and extra capacity. Uniform masking remains the reference, while J and A are declared motion-independent alternatives with matched budgets. Read the newer notebooks for their support rules and fixed evaluation banks. Do not combine old and new scores or denominators.''')


def update_historical_navigation():
    records = []
    for path in sorted(BASE.glob("[01][0-9]_*.ipynb")):
        if int(path.name[:2]) >= 12:
            continue
        book = nbf.read(path, as_version=4)
        protected = [(cell.id, cell.source, cell.get("execution_count"), cell.get("outputs", []))
                     for cell in book.cells if cell.cell_type == "code"]
        old_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        book.cells = [cell for cell in book.cells if not (
            cell.cell_type == "markdown" and cell.source.startswith((NAVIGATION_MARKER, "## How this tutorial fits the revised study")))]
        book.cells.insert(1, navigation_cell())
        after = [(cell.id, cell.source, cell.get("execution_count"), cell.get("outputs", []))
                 for cell in book.cells if cell.cell_type == "code"]
        assert after == protected, "Historical scientific cells or outputs changed."
        nbf.validate(book)
        nbf.write(book, path)
        records.append({"notebook": path.name, "before_sha256": old_hash,
                        "after_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "change": "navigation Markdown only", "code_cells_outputs_ids_preserved": True})
    path = BASE / "artifacts/iteration02/preparation/historical_navigation_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2))
    return records


SETUP = '''
from pathlib import Path
import os, sys, json
from dotenv import load_dotenv as load_env, find_dotenv as find_env

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / "ambient").is_dir() and (p / "data/gavd").is_dir())
os.chdir(ROOT)
load_env(find_env())
BASE = ROOT / "experiments/aaai"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from aaai_gait.iteration02_notebook_support import (
    EvidenceRun, package_versions, show_diagram, show_video, show_status,
    optional_evidence, joint_table, source_role_table, plot_counts,
    prediction_example, example_losses, plot_prediction, plot_adequacy,
    plot_timing, plot_source_scores, within_fold_metrics, plot_metric_table,
    fold_json_records, fold_tables, cohort_table, mask_bank_summary,
    measured_window, plot_paired_differences, timing_video_example, external_replication, plot_condition_confusions,
)

run = EvidenceRun.from_env()
display(run.describe().T.rename(columns={0: "value"}))
display(package_versions())
if run.mode not in {"full", "full_study"}:
    display(Markdown("**DEVELOPMENT RUN. This is incomplete evidence and cannot supply final manuscript results.**"))
'''


def start(number, title, question):
    return [md(f'''# {number}. {title}

{question}

## Run this notebook independently

From the repository root, install the shared environment and start Jupyter:

```bash
uv sync --project experiments/aaai
AAAI_ITERATION02_RUN_DIR=/absolute/path/to/experiments/aaai/artifacts/iteration02/PROTOCOL_HASH \\
  uv run --project experiments/aaai jupyter lab experiments/aaai
```

Replace `PROTOCOL_HASH` with the exact completed run you intend to inspect. The setup displays its producer command and verifies its recorded identity before showing results. It never chooses the newest run automatically. Alternatively, put `AAAI_ITERATION02_RUN_DIR` in the repository root `.env`. Do not create another environment file. Python-dotenv calls the real functions `load_dotenv` and `find_dotenv`; the aliases below implement the requested `load_env(find_env())` spelling without printing credentials.

Python 3.12 and the notebook, plotting, data and model dependencies are declared in the existing `pyproject.toml`. Select its kernel. This notebook does not depend on cells from another notebook. A full experiment fits models through the recorded producer command; this tutorial loads those real fitted outputs, checks them and teaches how to interpret them. Missing required outputs stop execution instead of triggering synthetic examples.

A small development run has its own protocol identity. To inspect it deliberately, also set `AAAI_ITERATION02_ALLOW_DEVELOPMENT=1`. Its outputs remain labeled incomplete. Do not use that option to bypass a failed evidence verifier.

The experiment is exploratory because the available sources informed earlier work. A dataset folder label is not a diagnosis independently established here. The focal endpoint is normal versus Parkinsons, stroke, cerebral palsy and myopathic labels. The mandatory broad endpoint uses all non-normal folders. The word “normal” refers to the dataset annotation, not a clinical clearance.'''), code(SETUP)]


def finish(cells, filename):
    stem = Path(filename).stem
    cells += [md('''## Preserve the evidence trail

The run directory is the unit of reproducibility. Keep its input hashes, source roles, fitted models, score references, target predictions, source scores, verification record and failed-comparison ledger together. A successful notebook execution establishes that this tutorial can reproduce its displayed analysis. It does not establish a neurological cause or a clinically valid classifier.

The [iteration02 protocol](docs/iteration02/protocol.md), [results](docs/iteration02/results.md), [limitations](docs/iteration02/limitations.md), and [draft02 readiness review](docs/draft02/README.md) must agree with the same run. The earlier draft01 and notebooks 00–11 remain historical evidence. New figures are written as SVG and PDF masters with PNG previews and an input-hash sidecar under this run's `figures/notebooks` directory.'''), code(f'''
record_path = run.record({stem!r})
print("Recorded the exact real artifacts read by this notebook:", record_path)
print("This notebook did not replace the experiment producer or alter its source roles.")
''')]
    notebook = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (AAAI gait)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
        "iteration02": {"status": "authored; execution requires verified real run", "source_generator": "scripts/build_iteration02_notebooks.py"},
    })
    nbf.validate(notebook)
    nbf.write(notebook, BASE / filename)
    return filename


def build():
    outputs = []
    cells = start("12", "Define the observation population and see the actual movement",
                  "Which real source videos can answer the new question, and what information does each measured window contain? We will connect the fresh annotations, local video, extracted pose, eligibility rule and source split before inspecting any detection score.")
    cells += [md('''## 1. Start with the recorded cohort, then compare it with today's annotations

The experiment reads the current condition CSVs in `data/gavd`. It does not count the `source` distribution cache as another participant. A video may supply several annotated sequences, and each sequence may supply overlapping windows. These are repeated observations of a source, not additional independent people.

The saved manifest fixes the annotations used by this run. We also read the current CSVs and compare sequence, label and annotation hashes. If the dataset has changed, stop and create a new run identity. Do not combine an old score table with a refreshed denominator. Existing poses can be reused only when their extraction provenance matches the annotation and media inputs.'''), code('''
from aaai_gait.study_data import build_study_manifest
saved_manifest = run.table("manifest.csv", columns=("sequence_id", "source_id", "label", "csv_sha256"))
fresh_manifest = build_study_manifest(ROOT, BASE / "artifacts/poses")
keys = ["sequence_id", "source_id", "label", "csv_sha256"]
saved_keys = saved_manifest[keys].sort_values("sequence_id").reset_index(drop=True)
fresh_keys = fresh_manifest[keys].sort_values("sequence_id").reset_index(drop=True)
assert saved_keys.equals(fresh_keys), "Annotations differ from the verified run. Create a new protocol identity."
display(saved_manifest.groupby("label").agg(sequences=("sequence_id", "nunique"), source_memberships=("source_id", "nunique")))
print("Current annotations match the saved run snapshot.")
'''), md('''## 2. Trace one normal sequence back to its source video

The preview below uses a deterministic source and start time from this run. The overlay uses the stored raw image coordinates, the extraction's frame-number base and its actual video frame rate. It draws only confident observed joints. It does not infer a missing ankle from the picture to improve the result.

Compare the person, crop and skeleton. A pose can look numerically plausible while following the wrong person. Check turning, occlusion, walking aids and camera movement. A failed decode remains a visible availability problem. Previews stay under `data/videos-gavd/<source_id>/previews/` and keep the source playback rate.'''), code('''
normal_example = show_video(run, label="normal")
display(pd.DataFrame([normal_example]))
'''), md('''## 3. Inspect an annotated abnormal example without diagnosing it

Stroke can leave weakness, altered selective control and repeated compensations. A repeatable compensation can be easy to predict. Parkinsonian walking may have small excursion while timing remains regular. Cerebral palsy includes varied developmental motor limitations, and myopathic labels concern muscle-related limitations rather than one brain lesion. These are reasons to preserve movement size, posture and relative information separately.

The [neurological evidence review](docs/iteration02/advice_neuro.md) connects these hypotheses to primary studies and explains their limits. The example is selected by source order, not by the model's most dramatic error. Change the label to explore another recorded family. Visible asymmetry is an observation; it does not identify the affected pathway, medication state or muscle force.'''), code('''
abnormal_example = show_video(run, label=os.getenv("AAAI_PREVIEW_LABEL", "stroke"))
'''), md('''## 4. Separate raw availability from the final observation population

The final source registry follows the declared support rules. Invalid coordinates stay missing. A target patch needs four valid frames. Source inclusion additionally requires four distinct physical target patches per lower-leg readout joint. Repeated masks and overlapping windows do not turn the same patch into new support.

Longer clips can help estimate cycles but exclude short videos and reduce the normal sample. The primary 32-sample window spans 31 intervals at 15 Hz, about 2.07 seconds. It does not guarantee two complete gait cycles. The 48-frame timing parent has its own support requirements and is not evidence that every primary source supports the timing analysis.'''), code('''
registry = run.table("source_registry.csv", columns=("source_id", "binary_label"))
windows = run.table("window_metadata.csv", columns=("source_id", "sequence_id", "label"))
assert not registry.source_id.duplicated().any()
registered_windows = windows.loc[windows.source_id.isin(registry.source_id)]
display(registered_windows.groupby("label").agg(sources=("source_id", "nunique"), sequences=("sequence_id", "nunique"), windows=("source_id", "size")))
display(plot_counts(run)); plt.close("all")
display(cohort_table(run, "sequence_ledger.csv"))
display(cohort_table(run, "source_eligibility.csv"))
display(run.json("timing/census/census.json"))
'''), md('''## 5. Keep the masking allowlist separate from the readout

Every training recipe can hide the same 16 approved joints below. The primary score reads the eight lower-leg joints, indices 25 through 32. This readout change is not permission to silently remove shoulders or hips from the masking grid. The shared body reference already contains body measurements, so an own-leg restriction is conditional on that explicitly declared reference.

The numerical order below is deliberate. Left and right joints are distinct, and the duplicate mention of stroke in the original request does not create a second condition class.'''), code('''
display(joint_table())
assert joint_table()["BLAZEPOSE_33 index"].is_unique
'''), md('''## 6. Freeze fitting, reference and evaluation roles by source

The fitting normals determine model parameters and development choices. Separate reserved normals estimate the relational reference after those choices are frozen. Outer-test sources supply the final source scores. Non-test abnormal sources can fit the separately reported supervised classifier; they do not fit the normal-only anomaly model.

The same video cannot cross roles within a fold. Known duplicate recordings must also stay together. Unknown repeated people across different videos remain a limitation. All recovered normal sources were previously explored, so reorganizing their folds does not create independent confirmation.'''), code('''
roles = source_role_table(registry)
display(roles.groupby(["outer_fold", "role", "binary_label"]).source_id.nunique().rename("sources").reset_index())
display(cohort_table(run, "near_duplicate_candidates.csv"))
print("Retained source IDs:", registry.source_id.nunique(), "retained recording groups:", registry.cluster_id.nunique())
show_diagram("execution_gates")
''')]
    outputs.append(finish(cells, "12_observation_population_and_video_context.ipynb"))

    cells = start("13", "Predict from the own leg, then test what the other leg adds",
                  "Does opposite-leg information improve a competent local predictor beyond the benefit of extra model capacity? We will examine the fitted models on the same coordinate target and learn why masking structure must be tested without changing the evaluation question.")
    cells += [md('''## 1. Define the comparison before reading its result

Imagine hiding four positions of the left ankle. The visible left knee, hip and foot can already constrain that ankle through local geometry. A same-joint predictor tests temporal interpolation. An own-leg predictor adds local geometry. Only then does the opposite leg provide the added information of interest.

We denote permitted own-leg observations by O, opposite-leg observations by C, and the hidden coordinate patch by Y. All conditions use the same Y, weights, validity and declared body reference. They cannot receive a cached embedding that has already attended to hidden or opposite-leg coordinates.'''), code('''
show_diagram("residual_predictor")
'''), md('''## 2. Retain the competent baseline and match extra capacity

First fit the normal-only local predictor f(O). Freeze it. The own-leg correction h(O) adds capacity without adding the other leg. The both-leg correction g(O,C) gets the extra context. Both corrections can be zero, preserving the original prediction.

The primary benefit is loss(f+h) minus loss(f+g), not simply loss(f) minus loss(f+g). Otherwise, more parameters could masquerade as cross-leg information. Matching parameter count alone does not equalize effective rank or inductive bias, so the implementation records those limitations. Zero correction is a function-class guarantee, not a guarantee that fitting improves held-out prediction.'''), md('''## 3. Understand the three motion-independent masks

A 32-frame sequence contains eight four-frame time patches. With 16 joints this gives 128 tokens. U hides 64 distinct tokens uniformly. J chooses one joint uniformly, hides its full eight-token trajectory, then uniformly fills 56 other positions. A hides an eight-token anatomical core, then uses the same fill.

A chooses a group with probability equal to its width divided by 16. Four-joint groups cover two consecutive time patches; two-joint groups cover four. Aligned starts partition the eight-patch window without wrapping. This gives every token the same one-half marginal masking probability as U and J. The policies differ in what disappears together, not their total budget or first-order joint frequency.'''), code('''
show_diagram("mask_families")
groups = pd.DataFrame([
    ("Torso", [11, 12, 23, 24], 4, 2, [0, 2, 4, 6]),
    ("Left arm", [13, 15], 2, 4, [0, 4]),
    ("Right arm", [14, 16], 2, 4, [0, 4]),
    ("Left distal leg", [25, 27, 29, 31], 4, 2, [0, 2, 4, 6]),
    ("Right distal leg", [26, 28, 30, 32], 4, 2, [0, 2, 4, 6]),
], columns=["group", "indices", "width", "time_patches", "starts"])
groups["selection_probability"] = groups.width / 16
display(groups)
core_probability = (groups.width / 16) * (groups.time_patches / 8)
assert np.allclose(core_probability, 1 / 16)
assert np.isclose(1 / 16 + (15 / 16) * (56 / 120), .5)
print("Analytical construction check only; this is not a learned gait result.")
display(mask_bank_summary(run))
print("A finite 16-draw bank need not have exact one-half empirical token frequencies.")
'''), md('''## 4. Respect the prior art and isolate the training factor

[MAMP](https://arxiv.org/abs/2308.07092) and the original [skeleton S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) use motion-aware masking. Their stochastic sampling is not uniformly random. [SLiM, version 4](https://arxiv.org/html/2603.10648v4) already uses anatomical tubes to reduce easy interpolation, so this project does not invent body-part masking. Action-recognition results do not determine the best gait mask.

The correction-mask experiment keeps one U-trained f fixed while training h and g under U, J or A. A separate JEPA pretraining-mask experiment, if executed, keeps downstream head masks fixed to U. Changing both at once would mix two explanations. Dedicated random streams keep the new sampler from changing source order, target selection or initialization. Both VICReg views must use the assigned family. VICReg can discourage collapse; it does not force disease categories apart.'''), code('''
display(pd.json_normalize(fold_json_records(run, "selection.json")))
optional_evidence(run, "fixtures.json", "known-dependence, independence, access and zero-correction checks")
show_status(run, "baseline_status.json")
'''), md('''## 5. Inspect one actual predicted normal patch

The next cell loads a deterministic saved normal example from the U policy and reference evaluation bank. It does not select the largest benefit. Each line predicts the same four observed coordinate pairs. The target values come from the real extracted pose; the other lines come from fitted models recorded by the run.

One attractive plot cannot establish accuracy. Look for obvious scale, target-order or coordinate-sign mistakes, then inspect the full normal-development summary. A very accurate local predictor may leave little headroom for any additional context.'''), code('''
example_row, example_arrays = prediction_example(run)
display(pd.DataFrame([example_row]).drop(columns=[name for name in ("Y", "target", "f", "fh", "fg") if name in example_row], errors="ignore"))
display(plot_prediction(run)); plt.close("all")
display(example_losses(run))
if {"source_id", "sequence_id", "start_seconds", "end_seconds"}.issubset(example_row.index):
    show_video(run, example_row)
else:
    display(Markdown("This saved prediction lacks an exact window key. Its sequence is identified above, but no other interval was substituted for its video."))
'''), md('''## 6. Require normal prediction competence before interpreting an anomaly

The fitted both-leg model must recover useful dependence in known-answer fixtures and meet the declared normal-development gate relative to both f and f+h. Inspect source weighting, effective valid-target exposure and residual magnitude. A J tube removes all history of its selected joint, so the old own-joint eligibility rule cannot silently be used to discard those hard targets.

Draw masks before observation filtering. Confidence and motion must not choose easier masks. A failed gate stays in the report and sets the fold's relational component to zero under the frozen fallback. It does not authorize another architecture search on abnormal AUROC. A null result with almost no valid context is a support limitation rather than evidence that human gait lacks coordination.'''), code('''
adequacy = run.table("adequacy.csv")
display(adequacy)
display(plot_adequacy(run)); plt.close("all")
display(pd.DataFrame(run.json("normal_test_benefit.json")))
print("Development adequacy and held-out normal benefit are separate evidence. Read each evaluation bank and its conditional interval.")
display(pd.json_normalize(fold_json_records(run, "fitting_record.json")))
print("Ridge fitting is closed form. The separate JEPA probe records its own optimization trace.")
'''), md('''## 7. State the narrow contribution the experiment could support

Useful evidence would show that a controlled training mask improves added-leg prediction on the unchanged U task and that the resulting score adds value beyond simple movement descriptors. Improvement only on a distal-leg challenge bank is task specialization. It is still informative, but it is a different claim.

These masks remain independent of motion intensity. Stochastic positional embeddings, stochastic latent variables and learned mask selectors are separate research axes. They should not be introduced into this comparison after reading a favorable or unfavorable result.''')]
    outputs.append(finish(cells, "13_own_leg_and_opposite_leg_prediction.ipynb"))

    cells = start("14", "Test a fixed target with matched distal-leg timing controls",
                  "Does the fitted predictor use time-aligned opposite-leg context, and can the evidence survive support and geometry checks? We will keep the answer fixed while studying a narrowly defined change to permitted predictor inputs.")
    cells += [md('''## 1. Freeze evaluation banks before comparing training masks

E_U preserves the original 16 uniform draws with seed 17. Every U-, J- and A-trained model answers that same question. A versus U is the primary masking-policy contrast, with A versus J and J versus U explaining possible differences.

E_leg is a separate diagnostic. It balances two distal-leg sides and starts 0, 2, 4 and 6, twice, giving 16 masks. Each draw hides an eight-token distal core and uniformly fills to 64 with dedicated seed 117. Only valid core targets enter its score. Reference source identities stay fixed, but its relational reference is fitted for that bank. A missing required reference makes the diagnostic unavailable; an outer-test source cannot replace it.'''), code('''
show_diagram("mask_comparison")
display(mask_bank_summary(run))
show_status(run, "bank_status.json")
'''), md('''## 2. Compare within a column, not native task against native task

U/U and A/E_leg change both training and the tested observation task. Subtracting their losses cannot isolate a training-mask effect. Within either bank, all models use identical target keys, support and loss weights. Report each bank's coverage separately rather than intersecting every diagnostic into one tiny cohort.

The fixed-target timing control retains its own uniform parent-grid protocol across training arms. It does not inherit whichever native training mask produced a favorable score.'''), code('''
scores = run.table("source_scores.csv", columns=("source_id", "policy", "bank"))
display(scores.groupby(["bank", "policy"]).source_id.nunique().unstack("policy"))
'''), md('''## 3. Build the no-wrap timing challenge on a common clock

Take a 48-frame parent. Fix the target and own-leg observations to frames 8:40, using zero-based half-open indexing. The aligned opposite-leg interval is also 8:40. Earlier and later donors are 4:36 and 12:44. No sample wraps from the end to the start.

For a permitted distal joint, subtract its permitted donor hip to obtain a hip-relative offset. Attach that offset to the permitted opposite hip at the original time, using the original fixed scale and body reference. Do not recover an unavailable hip from the unmasked pose. The intervention changes distal-leg input timing and geometry; it does not move the complete pelvis trajectory or create a physically possible patient.'''), code('''
show_diagram("timing_controls")
'''), md('''## 4. Keep visibility, target and normalizer identities auditable

The 48-frame parent has 12 time patches and 192 allowed tokens. Draw its uniform mask before constructing any offset. Donor values keep their original visibility flags. Both donor and attachment hips must be observed and permitted.

Within each aligned-versus-shifted pair, restrict the aligned input to the same permitted support as the shifted input. Keep target coordinates, own-leg input, scale, fitted models, gates and reference parameters fixed. Earlier and later effects are separate primary diagnostics. A symmetric average uses only sources eligible for both and is secondary.'''), code('''
show_status(run, "timing/raw_ledger_verification.json")
show_status(run, "timing/independent_raw_pair_verification.json")
timing = run.table("timing_pairs.csv")
display(timing.head(12))
print("Recorded paired timing rows:", len(timing))
'''), md('''## 5. Look at the source interval before looking at its effect

An input edit cannot tell us which neurological mechanism produced the original movement. First inspect the actual walking interval, missing points and direction of travel. A camera turn or person-tracking change may explain an apparent timing sensitivity.

The preview is tied to a recorded paired source when one is available. If the timing support is empty, the notebook reports that absence and does not substitute another intervention or manufacture a curve.'''), code('''
timing_video_example(run)
'''), md('''## 6. Interpret raw prediction loss before calibrated anomaly scores

The primary timing effect is the frozen both-leg prediction loss under shifted context minus its loss under aligned context. Positive values mean that alignment helped this predictor on the same target and support. Changes in R or S are secondary because calibration introduces another transformation.

Report projected bone-length, joint-angle, displacement, speed and missingness changes. A finite crop does not exactly preserve a motion distribution. Even with hip-relative attachment, the body configuration changes. The result is therefore sensitivity to a compound timing-and-geometry input challenge, not an isolated physiological phase effect or a causal disease mechanism.'''), code('''
timing_figure = plot_timing(run)
if timing_figure is not None:
    display(timing_figure); plt.close("all")
show_status(run, "timing_summary.json")
drift_columns = [name for name in timing if any(term in name for term in ("drift", "displacement", "speed", "bone", "angle", "count"))]
display(timing[["source_id", "policy", "variant", "offset", *drift_columns]].head(20))
paired_adequacy = run.table("timing/paired_support_adequacy.csv")
display(paired_adequacy)
print("Passed sparse-input gates:", int(paired_adequacy.paired_support_adequacy_passed.sum()), "of", len(paired_adequacy))
if not paired_adequacy.paired_support_adequacy_passed.all():
    display(Markdown("**The sparse paired-support adequacy gate failed for at least one comparison. Its timing effect is an input-sensitivity result, not evidence of physiological coordination.**"))
'''), md('''## 7. Preserve a useful null result without overinterpreting it

A null effect is informative only if the aligned sparse input still predicts competently and leaves measurable headroom. With almost no permitted opposite-leg tokens, neither timing nor its absence is identified. Do not relax the masking rule after seeing that outcome.

Independent phase normalization of each leg could erase the timing asymmetry being studied. Full-target phase can also leak the hidden answer. The primary model therefore preserves raw clock time, amplitude and asymmetry. A future phase-based analysis requires its own support and leakage audit.''')]
    outputs.append(finish(cells, "14_matched_distal_leg_timing_controls.ipynb"))

    cells = start("15", "Keep local movement and relational anomaly evidence separate",
                  "How do measured coordinate predictions become a source score without erasing movement amplitude or confusing fitting and calibration? We will reconstruct one benefit, inspect normal references and examine the frozen M, R and S channels.")
    cells += [md('''## 1. Preserve two different explanations for unusual walking

A person can make small, regularly timed steps. Another person can retain an ordinary excursion but move the legs in an unusual relation. A third can use a stable compensation that makes one leg more predictable from the other. None of these examples determines a diagnosis.

M describes visible movement, including projected posture, excursion, asymmetry and velocity. R describes departure from a normal pattern of matched predictive benefits. Treat them as complementary evidence. A purely relational model can miss a small but regular movement, while a local-motion model can miss a changed relationship.'''), code('''
show_diagram("rationale")
'''), md('''## 2. Compute one predictive benefit from real arrays

For the saved four-frame patch, square the difference between each coordinate prediction and the observed target, then average over the eight coordinate values. The benefit is the own-capacity error minus the both-leg error. A positive value means the other leg helped that fitted predictor for this target.

This is not a per-patient diagnostic threshold. It is one constituent observation. Its sign can vary by joint, source, support and predictor quality. Stable compensatory movement could increase rather than reduce the benefit, motivating a two-sided relational reference.'''), code('''
row, arrays = prediction_example(run)
target = arrays["Observed target"]
own_error = float(np.mean((arrays["Own-leg correction f+h"] - target) ** 2))
both_error = float(np.mean((arrays["Both-leg correction f+g"] - target) ** 2))
display(pd.DataFrame([{"source_id": row.source_id, "own_capacity_error": own_error,
                       "both_leg_error": both_error, "predictive_benefit": own_error - both_error}]))
display(example_losses(run))
'''), md('''## 3. Fit M using fitting normals, not the relational reference sources

The declared local channel contains 16 descriptors. Each leg contributes knee-angle median and interquartile range, ankle horizontal and vertical excursion, and horizontal and vertical velocity variability. Four additional values describe projected trunk inclination and left-right differences. The raw clock and common scale remain intact.

M excludes local prediction error. Fit descriptor imputation, scaling and covariance on fitting-normal sources only. A projected trunk angle is not a clinical trunk examination. Camera view, depth projection and missing joints can change these descriptors, so preserve support and nuisance summaries alongside the score.'''), code('''
display(run.table("descriptors_M.csv"))
display(pd.json_normalize(fold_json_records(run, "tabular_audit.json", field="M")))
window = measured_window(run, row)
# The cohort uses the sorted 16-joint allowlist, with x/y per frame.
joints = joint_table()["BLAZEPOSE_33 index"].tolist()
hip, knee, ankle = (window[:, joints.index(index), :2] for index in (23, 25, 27))
u, v = hip - knee, ankle - knee
norms = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
valid = np.isfinite(u).all(1) & np.isfinite(v).all(1) & (norms > 1e-8)
angles = np.degrees(np.arccos(np.clip(np.sum(u[valid] * v[valid], axis=1) / norms[valid], -1, 1)))
display(pd.DataFrame([{ "source_id": row.source_id, "window_left_knee_median_degrees":
    float(np.median(angles)) if len(angles) else np.nan, "valid_angle_frames": len(angles)}]))
print("This hand calculation is one real window. The saved M table follows the full source-aggregation rule.")
show_video(run, row)
'''), md('''## 4. Aggregate repeated measurements before estimating R

Repeated mask draws of the same target do not create new physical observations. Follow the frozen hierarchy: average draws for each target, then targets within windows, windows within sequences, and sequences within sources, preserving the declared joint weighting. Keep left and right constituents inspectable.

The separate reserved normal sources fit R after model and mask choices are frozen. The normal reference must match its evaluation bank. Five reference videos provide a fragile estimate, so leave-one-reference-out sensitivity is useful, but it must not select the most favorable reference after testing.'''), code('''
display(pd.json_normalize(fold_json_records(run, "U_U/references.json")))
constituents = run.table("source_scores.csv")
constituent_columns = [name for name in constituents if name.startswith(("left_", "right_"))]
display(constituents.loc[constituents.policy.eq("U") & constituents.bank.eq("U"), ["source_id", *constituent_columns]])
display(fold_tables(run, "U_U/reference_sensitivity.csv"))
'''), md('''## 5. Form the fixed combined score and keep failed folds

The combined score is S = (M + R) / 2. It is fixed before inspecting abnormal outcomes. If the normal adequacy gate fails in a fold, set R to zero for every source in that fold and record the fallback. Then S = M/2 preserves the M ranking within that fold.

Do not exclude difficult sources because the relation failed. A fallback result can still be evaluated, but it does not establish relational learning. The primary detection comparison is S against M; the masking-policy contrasts are separate questions.'''), code('''
scores = run.table("source_scores.csv", columns=("outer_fold", "source_id", "policy", "bank", "M", "R", "S", "fallback"))
primary_rows = scores.bank.isin(["U", "E_U"])
assert np.isfinite(scores.loc[primary_rows, ["M", "R", "S"]].to_numpy(float)).all()
available = np.isfinite(scores[["M", "R", "S"]].to_numpy(float)).all(axis=1)
assert np.allclose(scores.loc[available, "S"], (scores.loc[available, "M"] + scores.loc[available, "R"]) / 2), "Saved scores do not follow the frozen formula."
fallback = scores.fallback.astype(str).str.lower().isin(["true", "1"])
assert np.allclose(scores.loc[fallback & available, "R"], 0)
print("Unavailable diagnostic score rows:", int((~available).sum()))
display(scores.groupby(["policy", "bank", "outer_fold"]).agg(sources=("source_id", "nunique"), fallback_rows=("fallback", "sum")))
'''), md('''## 6. Inspect score distributions without replacing the primary estimator

The empirical curves below describe the saved scores. A pooled plot can look separated because different folds use slightly different score scales. It is not the primary AUROC calculation. The next notebook computes rankings within each fold and combines them using fixed normal-by-abnormal pair counts.

Do not reverse R's direction or retune the M/R mixture after looking at these curves. Inspect sham or nuisance comparators as explanations of a result, not as opportunities to find a favorable score.'''), code('''
primary_bank = "U" if "U" in scores.bank.unique() else "E_U"
display(plot_source_scores(run, policy="U", bank=primary_bank)); plt.close("all")
display(scores.loc[scores.policy.eq("U") & scores.bank.eq(primary_bank), ["source_id", "M", "R", "S", "S_sham", "nuisance_distance", "nuisance_supervised"]])
''')]
    outputs.append(finish(cells, "15_local_and_relational_anomaly_scores.ipynb"))

    cells = start("16", "Compare baselines on the same held-out source videos",
                  "Does the new score add value beyond local movement, extra capacity and inexpensive models? We will reconstruct the declared source-level estimator, inspect paired uncertainty and separate normal-only detection from supervised condition classification.")
    cells += [md('''## 1. Use the same sources and frozen questions

The focal primary endpoint is normal versus the four requested condition families. The broad secondary endpoint uses all current non-normal folders, including activity and style labels. Both use the same frozen source scores. This focal redesign followed historical all-folder results and remains exploratory.

Compare methods only on their declared matched population. A method should not gain a favorable denominator by silently dropping videos it cannot score. Report unsupported required comparisons in the status ledger. A published method name is justified only by a verified reproduction or a clearly described adaptation.'''), code('''
registry = run.table("source_registry.csv")
display(source_role_table(registry).groupby(["outer_fold", "role"]).source_id.nunique().unstack(fill_value=0))
show_status(run, "baseline_status.json")
show_video(run, label="normal")
'''), md('''## 2. Reconstruct the within-fold AUROC by hand

Within one outer fold, AUROC asks whether an abnormal-labeled source receives a higher score than a normal-labeled source. Average the fold AUROCs with fixed weights n_normal multiplied by n_abnormal. These weights count the comparable pairs within each fold.

Do not pool differently calibrated fold scores to obtain the primary estimate. Average precision is reported per fold together with prevalence because its scale depends on the abnormal proportion. Four folds are not four independent clinical studies.'''), code('''
scores = run.table("source_scores.csv", columns=("outer_fold", "source_id", "policy", "bank", "binary_label", "focal_eligible", "M", "R", "S"))
primary_bank = "U" if "U" in scores.bank.unique() else "E_U"
reference = scores.loc[scores.policy.eq("U") & scores.bank.eq(primary_bank)]
recomputed = []
for endpoint in ("focal", "broad"):
    for score_name in ("M", "S"):
        fold_table, value = within_fold_metrics(reference, score_name, endpoint=endpoint)
        display(fold_table.assign(endpoint=endpoint, score=score_name))
        recomputed.append({"endpoint": endpoint, "score": score_name, "pair_weighted_auroc": value})
recomputed = pd.DataFrame(recomputed)
display(recomputed)
saved_metrics = run.table("metrics.csv")
for item in recomputed.itertuples():
    match = saved_metrics.loc[saved_metrics.policy.eq("U") & saved_metrics.bank.eq(primary_bank)
                              & saved_metrics.endpoint.eq(item.endpoint) & saved_metrics.method.eq(item.score)
                              & saved_metrics.metric.eq("auroc")]
    assert len(match) == 1 and np.isclose(match.value.iloc[0], item.pair_weighted_auroc, atol=1e-12)
print("Every reconstructed M/S AUROC matches the saved estimator.")
'''), md('''## 3. Read paired uncertainty, not just an attractive point estimate

The paired bootstrap resamples whole source records within each fold and binary-label stratum. Each sampled source carries every method score together, and the original fold weights stay fixed. Confirmed duplicate-recording groups must be treated as a shared resampling cluster.

The interval is conditional on the fitted models, source split and selected protocol. It does not include training-seed uncertainty, unknown repeated-person dependence or the earlier adaptive redesign. An interval crossing zero does not establish a detection advantage. A narrow software-reproducibility check is not a replacement for independent clinical evidence.'''), code('''
metrics = run.table("metrics.csv", columns=("metric", "method", "endpoint", "value"))
display(metrics)
display(pd.json_normalize(run.json("paired_intervals.json")))
display(plot_paired_differences(run)); plt.close("all")
raw_metrics = run.json("rawmetrics.json")
display(pd.json_normalize(raw_metrics).T if isinstance(raw_metrics, dict) else raw_metrics)
'''), md('''## 4. Report all training masks within each evaluation bank

A versus U on E_U is the primary masking-policy contrast. A versus J helps distinguish anatomical co-masking from simply hiding one joint's entire history. J versus U is a secondary explanation. Keep all arms, including failed adequacy gates.

If an alternative improves only E_leg, describe it as specialization to that missing-context task. Do not replace the original result with whichever bank looks better. A diagnostic with too few supported normal references must remain unavailable.'''), code('''
masking_rows = []
for (bank, policy), group in scores.groupby(["bank", "policy"], sort=True):
    for endpoint in ("focal", "broad"):
        try:
            folds, value = within_fold_metrics(group, "S", endpoint=endpoint)
        except ValueError as error:
            masking_rows.append({"bank": bank, "policy": policy, "endpoint": endpoint,
                                 "status": "unavailable", "reason": str(error)})
        else:
            masking_rows.append({"bank": bank, "policy": policy, "endpoint": endpoint,
                                 "status": "measured", "sources": len(group), "auroc": value})
display(pd.DataFrame(masking_rows))
display(plot_metric_table(run)); plt.close("all")
'''), md('''## 5. Make simple alternatives difficult to dismiss

The own-leg capacity control asks whether flexibility explains the change. Shared ridge and a small temporal model ask whether a complex representation is needed. Local movement and valid phase descriptors ask whether simpler measurements already explain the score. A random encoder with the same coordinate readout tests whether representation learning adds value.

A failed or unimplemented JEPA branch cannot be described as a successful learned world model. The baseline ledger must distinguish a fitted model, a failed training attempt, a deferred extension and an unavailable published-method adaptation. The strongest paper is the one its completed evidence supports, even if a simple predictor wins.'''), code('''
show_status(run, "auxiliary/protocol.json")
display(pd.json_normalize(fold_json_records(run, "tabular_audit.json")))
optional_evidence(run, "seed_sensitivity.csv", "frozen-recipe repetition across training seeds")
'''), md('''## 6. Keep the five-condition classifier in a separate table

The Random Forest and any encoder probe receive condition labels from permitted non-test training sources. This is a supervised five-label task. The normal-only anomaly detector does not receive that supervision, so the two results should not compete in one unlabeled accuracy table.

The exp5 comparison requires the same selected conditions, source splits and input provenance. Report balanced accuracy, macro-F1, class support and confusion matrices. A source with more than one condition membership needs an explicit handling rule; it cannot become several independent people. Tiny disease-family counts remain descriptive.'''), code('''
condition_results = pd.json_normalize(run.json("five_class_metrics.json"))
display(condition_results)
display(condition_results.groupby("method")[["macro_f1", "balanced_accuracy"]].mean())
print("These means describe four held-out source folds; their variability is not a clinical confidence interval.")
display(plot_condition_confusions(run)); plt.close("all")
'''), md('''## 7. State the generalization boundary with the result

GAVD offers varied real web video but has uncontrolled camera view, age, speed, clothing, aids, diagnostic verification and possible repeated identities. Many windows from one video increase measurement detail without increasing independent participants. Excluding poor poses can shift the observed population, including away from severe or unusual movement.

The study can support a source-level result under its recorded acquisition and eligibility rules. It cannot establish clinical diagnosis, new-site performance or neurological mechanism. A bounded independent laboratory replication would strengthen the measurement claim, but it must be reported with its own participants, sensors and fitted references.''')]
    outputs.append(finish(cells, "16_baselines_and_source_level_evaluation.ipynb"))

    cells = start("17", "Audit failures and decide what the manuscript can claim",
                  "Which conclusions survive the completed experiments, and which attractive explanations remain unsupported? We will inspect actual examples, keep failed gates visible and assemble a claim-by-claim evidence trail for draft02.")
    cells += [md('''## 1. Begin with the verifier, not the abstract

The verification record must confirm the source split, normal-only fitting, frozen mask identities, shared target support, score formulas and metric reconstruction. A passing record supports computational consistency. It cannot certify clinical validity or novelty.

Inspect the exact run identity before copying any number into a slide or manuscript. Development runs, historical draft01 runs and this new protocol are different evidence objects. An unavailable branch stays visible even when the core run completes.'''), code('''
show_status(run, "verification.json")
show_status(run, "baseline_status.json")
show_diagram("execution_gates")
'''), md('''## 2. Select examples using a declared rule

Use a normal example first, then a labeled abnormal example chosen independently of the score. These show what the source data look like. Error analyses can additionally select high-scoring normal sources or low-scoring abnormal sources, but that outcome-based selection must be stated.

Do not call one selected example representative. Look for tracking errors, aids, turning, camera movement, truncation and unreliable ankle positions. A model failure can be an observation failure, and a highly predictable compensation can produce a biologically abnormal movement with little relational anomaly.'''), code('''
show_video(run, label="normal")
show_video(run, label=os.getenv("AAAI_PREVIEW_LABEL", "stroke"))
'''), md('''## 3. Inspect outcome-selected failure candidates honestly

The table below deliberately selects the largest normal score and the smallest abnormal score under the U reference. These are diagnostic candidates selected after scoring. They do not define a tuned threshold or prove an error without a threshold rule.

For each candidate, inspect the corresponding source video and quality record. Avoid a clinical story that the visible sequence cannot support. A broad non-normal label can describe an activity or style rather than pathology.'''), code('''
scores = run.table("source_scores.csv", columns=("source_id", "policy", "bank", "binary_label", "S"))
primary_bank = "U" if "U" in scores.bank.unique() else "E_U"
reference = scores.loc[scores.policy.eq("U") & scores.bank.eq(primary_bank)]
candidates = pd.concat([
    reference.loc[reference.binary_label.eq(0)].nlargest(1, "S").assign(selection="Highest-scoring normal"),
    reference.loc[reference.binary_label.eq(1)].nsmallest(1, "S").assign(selection="Lowest-scoring abnormal"),
], ignore_index=True)
display(candidates)
windows = run.table("window_metadata.csv")
for source in candidates.source_id:
    source_windows = windows.loc[windows.source_id.eq(source)].sort_values(["sequence_id", "start_seconds"])
    if len(source_windows):
        show_video(run, source_windows.iloc[0])
display(run.table("descriptors_quality16.csv"))
display(cohort_table(run, "media_audit.csv"))
show_status(run, "view_sensitivity.json")
'''), md('''## 4. Separate a failure to learn from a failure of the hypothesis

An inadequate local predictor cannot establish what extra context contains. A both-leg correction that fails on normal development cannot support a neurological interpretation of its anomaly score. A timing probe with almost no paired support cannot show timing independence. A source-selection bias can make both successful and failed scores misleading.

The frozen fallback retains those folds with R = 0 and preserves the M ranking. Report how much of the final evaluation actually uses a supported relational component. Do not describe fallback performance as a successful relational model.'''), code('''
adequacy = run.table("adequacy.csv")
display(adequacy)
fallback = scores.fallback.astype(str).str.lower().isin(["true", "1"]) if "fallback" in scores else None
if fallback is not None:
    display(scores.assign(fallback_flag=fallback).groupby(["policy", "bank", "outer_fold"]).agg(
        sources=("source_id", "nunique"), fallback_fraction=("fallback_flag", "mean")))
paired_adequacy = run.table("timing/paired_support_adequacy.csv")
display(paired_adequacy)
print("Passed sparse-input gates:", int(paired_adequacy.paired_support_adequacy_passed.sum()), "of", len(paired_adequacy))
if not paired_adequacy.paired_support_adequacy_passed.all():
    display(Markdown("**The sparse paired-support adequacy gate failed for at least one comparison. Its timing effect is an input-sensitivity result, not evidence of physiological coordination.**"))
'''), md('''## 5. Use the novelty boundary as a reviewer would

Normal-only masked gait modeling, clinical coordination descriptors, dependence-based anomaly scores and anatomical masking all have close precedents. A JEPA label or a different mask picture is not enough. The possible contribution is a controlled measurement of added information that survives capacity, support, timing and local-motion alternatives.

If simple ridge performs well, the paper can emphasize the measurement and its controls. If a learned encoder does not improve a common coordinate target over a random encoder, report that limitation. If only the focused diagnostic improves, frame the result as task-specific. The study should explain why its observation matters rather than enlarge the claim to match an intended title.'''), code('''
show_status(run, "historical_common_source_sensitivity.json")
show_status(run, "per_condition_metrics.json")
print("These subgroup and historical checks are descriptive. The draft02 readiness review records the publication claims.")
'''), md('''## 6. Choose the manuscript outcome from the evidence

There are three defensible routes. If added information improves a competent predictor and improves detection beyond M, write a constrained relational-anomaly paper with paired uncertainty. If a reliable relationship exists but detection does not improve, write a measurement study explaining what local and relational views capture differently. If a key estimator or support gate fails, a clearly scoped failure audit may be the honest result, provided the investigation is substantial.

None of these outcomes proves a neural controller, lesion location or causal disease pathway. GAVD lacks the forces, EMG, strength measurements, medication state and clinical examination needed for those claims. A physical world-model program would require additional observations that distinguish competing explanations.'''), code('''
metrics = run.table("metrics.csv")
display(metrics)
display(pd.json_normalize(run.json("paired_intervals.json")))
display(plot_paired_differences(run)); plt.close("all")
external = external_replication(run)
display(pd.DataFrame([external["completion"]]))
display(pd.DataFrame([{"score": name, "weighted_auroc": values["weighted_auroc"]}
                      for name, values in external["metrics"].items()]))
display(pd.json_normalize(external["paired"]))
display(pd.json_normalize(external["age_sex"]))
print("The independent run is separately fitted on simultaneous laboratory angles.")
'''), md('''## 7. Understand what the independent replication adds

The linked public laboratory study evaluates 128 controls and 50 stroke participants, using six simultaneously sampled joint angles and its own normal references. It does not transfer GAVD weights or pool web videos with laboratory participants. The opposite-leg correction improves held-out normal prediction in each fold, but the fixed M/R fusion still separates the groups less well than M alone.

An age and sex sensitivity uses a rule fixed before its outcomes: same-sex pairs within five years, matched without replacement inside each outer fold. Forty-five pairs remain. Whole-pair bootstrap resampling preserves the match. This comparison is a demographic sensitivity analysis, not causal adjustment, and it still favors local movement. It supports a narrow lesson: useful normal predictive dependence does not automatically become useful anomaly evidence.

The article's participant metadata supply age and sex. These values are not guessed from videos or names. The [independent replication report](docs/iteration02/external_replication_results.md) documents the cohort exclusions, synchronized signals and separate fitting. Neither dataset identifies a lesion, muscle force or clinical mechanism.'''), md('''## 8. Finish the publication evidence package systematically

The manuscript should include an attrition table, physiology-to-observation diagram, access diagram, normal-adequacy plot, paired timing plot, matched baseline table and representative failures. Every measured figure needs its run identity and source artifacts. Illustrations remain illustrations, and unavailable analyses remain unavailable.

Check that the README, notebooks, slides, protocol, results and draft02 use the same source counts, endpoints, mask names, score definitions and intervals. Keep the historical draft01 intact. An independently executed notebook record and a rendered manuscript are necessary readiness checks, but submission format and scientific strength are separate judgments. The final readiness report should state both plainly.'''), code('''
figure_records = sorted(run.figure_dir.glob("*.json"))
display(pd.DataFrame([json.loads(path.read_text()) for path in figure_records]))
print("Measured figure provenance records:", len(figure_records))
''')]
    outputs.append(finish(cells, "17_failure_analysis_and_manuscript_decision.ipynb"))
    return outputs


if __name__ == "__main__":
    for name in build():
        print(name)
    update_historical_navigation()
