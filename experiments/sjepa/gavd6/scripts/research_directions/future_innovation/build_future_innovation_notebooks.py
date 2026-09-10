"""Editable sources for five connected Future Innovation tutorials.

Use --only 00 01 to regenerate selected lessons, --check for exact source
notebooks, or --check-sources to permit retained execution outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from textwrap import dedent

import nbformat


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks/experiments/future_innovation"
BUILDER = Path(__file__).relative_to(ROOT).as_posix()
NAMES = {
    "00": "00_question_and_worked_example.ipynb",
    "01": "01_cohort_and_alignment.ipynb",
    "02": "02_teacher_features_and_validity.ipynb",
    "03": "03_matched_predictors_and_controls.ipynb",
    "04": "04_results_and_next_decision.ipynb",
}


def md(source):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def code(source):
    return nbformat.v4.new_code_cell(dedent(source).strip())


STARTUP = r'''
from pathlib import Path
import os
import sys
from time import perf_counter

started = perf_counter()
override = os.environ.get("GAVD6_ROOT")
if override:
    candidates = [Path(override).expanduser().resolve()]
else:
    candidates = []
    for base in (Path.cwd(), *Path.cwd().parents):
        candidates.extend((base, base / "gavd6", base / "experiments/sjepa/gavd6"))
PROJECT_ROOT = next((p for p in candidates if (p / "src/gavd6_sjepa").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Set GAVD6_ROOT to the checkout containing src/gavd6_sjepa.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, display
from matplotlib_inline.backend_inline import set_matplotlib_formats
get_ipython().run_line_magic("matplotlib", "inline")
set_matplotlib_formats("svg", "png")
plt.rcParams.update({"figure.figsize": (8, 3), "axes.spines.top": False,
                    "axes.spines.right": False, "font.size": 11})

from gavd6_sjepa.research_directions.future_innovation.fi_tutorial_inspection import (
    artifact_inventory, inspect_report, read_optional_table,
)

# Use "execute" for real stages or "inspect" for saved artifacts.
# Relative paths resolve from GAVD6_ROOT. Execution requires an explicit run root.
MODE = os.environ.get("FI_TUTORIAL_MODE", "teach")
if MODE not in {"teach", "inspect", "execute"}:
    raise ValueError("FI_TUTORIAL_MODE must be teach, inspect, or execute.")
if MODE == "execute" and not os.environ.get("FI_RUN_ROOT"):
    raise ValueError("Set FI_RUN_ROOT explicitly before executing real experiment stages.")
RUN_ROOT = Path(os.environ.get("FI_RUN_ROOT", "outputs/future-innovation/gate-v1")).expanduser()
if not RUN_ROOT.is_absolute():
    RUN_ROOT = PROJECT_ROOT / RUN_ROOT
RUN_ROOT = RUN_ROOT.resolve()
print("Teaching examples only; no empirical gait findings." if MODE == "teach"
      else f"{MODE.upper()} mode: {RUN_ROOT}")
if MODE == "execute":
    from gavd6_sjepa.research_directions.future_innovation.fi_notebook_workflow import (
        initialize_from_environment, run_stage, build_notebook_report, require_complete_report,
        attempt_stage, require_stage_success,
    )
'''


def execution_cells(number):
    descriptions = {
        "00": "Initialize the immutable protocol and input provenance, or validate the existing run. New runs use the same environment variables and five official annotation partitions as Slurm job 01. Complete the HAIC environment and input setup first.",
        "01": "Build eligible candidates, extract aligned poses, then freeze the cohort and source folds. This can take hours. The existing stages verify and reuse completed work; inspect the resulting exclusions and overlays below.",
        "02": "Cache the frozen teacher features and run causal/target validity audits on the declared checkpoint. This requires the prepared cohort and normally one H100. Failed audits stop execution before fitting. Completed stages are verified and reused without loading the teacher again.",
        "03": "Run the full registered nested comparison: five outer folds, three seeds, all five arms, and the complete frozen inner-selection grid. With FI_NOTEBOOK_FOLD set, run just that outer fold (all seeds and arms); HAIC supplies five CPU array tasks. Without it, run all folds sequentially. No teaching settings enter this branch.",
        "04": "Score the out-of-fold predictions and build the sealed production report. If scoring fails, still attempt a diagnostic STOP report. After displaying evidence, the notebook fails if measurement is incomplete or scoring failed. A completed scientific STOP or INCONCLUSIVE is a successful execution.",
    }
    operations = {
        "00": 'initialize_from_environment(RUN_ROOT)',
        "01": 'stage_error = attempt_stage("build-cohort", RUN_ROOT)\nif stage_error is None:\n    stage_error = attempt_stage("extract-poses", RUN_ROOT)',
        "02": 'device = os.environ.get("FI_NOTEBOOK_DEVICE", "cuda")\nstage_error = attempt_stage("cache-teacher", RUN_ROOT, "--device", device)\nif stage_error is None:\n    stage_error = attempt_stage("audit-teacher", RUN_ROOT, "--device", device)',
        "03": '''fold = os.environ.get("FI_NOTEBOOK_FOLD")
if fold is not None and fold not in {"0", "1", "2", "3", "4"}:
    raise ValueError("FI_NOTEBOOK_FOLD must be 0–4; unset it to run all five folds.")
options = [] if fold is None else ["--outer-fold", fold]
stage_error = attempt_stage("run-gate", RUN_ROOT, "--device", "cpu", *options)''',
        "04": 'scoring_succeeded = build_notebook_report(RUN_ROOT)',
    }
    return [md(f"## Execute this stage\n\n{descriptions[number]}\n\nThis cell runs only in `execute` mode. Each command uses this kernel's Python and the existing production CLI; stage logs are retained alongside the executed notebook."),
            code('if MODE == "execute":\n' + '\n'.join('    ' + line for line in operations[number].splitlines()))]


def opening(number, title, text):
    return [
        md(f"# {number} · {title}\n\n{text}"),
        md("""
        **Run this notebook independently in a fresh kernel.** The default
        `teach` mode uses small generated examples. Set `FI_TUTORIAL_MODE=inspect`
        and `FI_RUN_ROOT` before starting the kernel to read saved artifacts.
        Set `FI_TUTORIAL_MODE=execute` with an explicit `FI_RUN_ROOT` to run the
        production stages below. Execute notebooks **00 → 04** in order for the
        full Experiment 0; each uses a fresh kernel and the same run directory.
        Use the [notebook HAIC launchers](../../../slurm/future-innovation/NOTEBOOKS.md)
        for scheduled execution. Inspection remains read-only. An absent local
        file says nothing about the current state of a remote HAIC job.

        [Study overview](../../../docs/studies/future-innovation/README.md) ·
        [Detailed experiment specification](../../../notes/future-innovation-distillation/experiment-0-guide.md)
        """),
        code(STARTUP),
    ] + execution_cells(number)


def ending(number, text):
    following = int(number) + 1
    link = (f"Continue with [{NAMES[f'{following:02d}']}]({NAMES[f'{following:02d}']})."
            if following < 5 else
            "Return to the [study overview](../../../docs/studies/future-innovation/README.md) to record the next decision.")
    completion = ([code('if MODE == "execute":\n    completed_decision = require_complete_report(RUN_ROOT, scoring_succeeded=scoring_succeeded)\n    print("Complete measurement:", completed_decision["decision"], "— synthetic:", completed_decision.get("synthetic"))')]
                  if number == "04" else [])
    if number in {"01", "02", "03"}:
        completion = [code('if MODE == "execute":\n    require_stage_success(stage_error)')]
    return completion + [md(f"## What this step establishes\n\n{text}\n\n{link}"),
            code('print(f"Notebook elapsed time: {perf_counter() - started:.2f} seconds ({MODE} mode).")')]


def lesson00():
    cells = opening("00", "What can skeleton history add?", """
We want to know whether an explicit skeleton helps a small predictor after it
has already seen the recent video. A skeleton is derived from the video, so
the claim concerns useful representation under these predictors. It does not
claim that pose reveals information absent from RGB.

Follow one walking clip. The predictor sees frames 0–31. A frozen video teacher
supplies a target at frames 38–39. This target is a vector of numbers describing
the person region. We predict those numbers, not a diagnosis or future pixels.
""")
    cells += [md("""
        ## 1. Predict first, then learn what was missed

        Let `x` contain past video features and recording details, `s` contain
        past skeleton coordinates, confidence and validity, and `y` be the future
        target. Fit a baseline using `x`. The residual `y - baseline(x)` is what
        remains unexplained. A small head uses `s` and `x` to predict a correction.
        It never receives `y` when making a prediction.

        For one imaginary feature, an answer of 0.8 and baseline of 0.5 leave
        a residual of 0.3. A predicted correction of 0.2 moves the prediction to
        0.7. The example below makes that arithmetic visible.
        """), code("""
        if MODE == "teach":
            answer, baseline_prediction, correction = 0.8, 0.5, 0.2
            full_prediction = baseline_prediction + correction
            display(pd.DataFrame({"value": [answer, baseline_prediction, full_prediction]},
                                 index=["Answer", "Video baseline", "With correction"]))
            fig, ax = plt.subplots()
            ax.barh(["Video baseline", "With correction"],
                    [baseline_prediction, full_prediction], color=["#2f6f99", "#5f9e7e"])
            ax.axvline(answer, color="#e07a4b", label="Illustrative answer")
            ax.set(xlim=(0, 1), xlabel="One feature value", title="A correction can reduce the remaining error")
            ax.legend(); plt.show()
        """), md("""
        ## 2. Measure the gain against a common reference

        Predictive R² compares squared error with predicting the training mean.
        It can be negative. `delta_r2` is full R² minus baseline R²; 0.05 means
        an absolute increase of 0.05. `f8` is the fraction of the baseline's
        remaining error recovered at the eight-frame horizon.

        Use the production scoring function on already training-centered toy
        targets. Zero therefore represents the training-mean reference. Real
        fits learn this centering and scaling separately inside training sources.
        """), code("""
        if MODE == "teach":
            from gavd6_sjepa.research_directions.future_innovation.fi_metrics import score_arrays
            y = np.array([[-1.0], [1.0], [-0.5], [0.5]])
            scores, *_ = score_arrays(y, 0.4 * y, 0.7 * y, np.ones(4), np.array([True]))
            display(pd.DataFrame([scores])[["r2_baseline", "r2_full", "delta_r2", "f8"]])
        """), md("""
        ## 3. Ask which alternative explanation each control tests

        | Arm | Question |
        | --- | --- |
        | Real skeleton | Does the correctly paired history help? |
        | Time shuffle | Does the original temporal order matter? |
        | Clip mismatch | Does this skeleton need to belong to this video? |
        | Background target | Is the gain concentrated in the person region? |
        | No skeleton | Could capacity, baseline features, or validity explain the gain? |

        The last control retains validity while zeroing coordinates and
        confidence. Background features still attend to the person in the
        teacher. Each control narrows the interpretation; none proves a clinical
        or causal mechanism on its own.
        """), code("""
        if MODE != "teach":
            display(artifact_inventory(RUN_ROOT))
            evidence = inspect_report(RUN_ROOT)
            print(evidence["state"], "—", evidence["explanation"])
        """)]
    return cells + ending("00", "Execution initializes or verifies the run's frozen protocol; teaching mode constructs an illustrative gain. Neither establishes that skeletons help on GAVD. That question requires aligned examples, source-held-out predictions and all five controls. Next we establish the cohort.")


def lesson01():
    cells = opening("01", "Are the examples aligned and independent?", """
Two excerpts from one upload may share a person, background and camera. Treating
them as independent test examples can reward recognition of the recording.
This stage establishes what one example is and keeps each source video in one
outer fold before any fitting. A source-video split does not establish a split
by verified person identity.
""")
    cells += [md("""
        ## 1. Count windows and sources separately

        The real gate freezes 50 eligible windows and permits at most two from
        a source, so at least 25 source videos are needed. Eligibility uses
        alignment and observation quality, not teacher scores or condition labels.
        This generated census uses the production fold assignment and cohort
        validator. All excerpts from one source must receive the same fold.
        """), code("""
        if MODE == "teach":
            from gavd6_sjepa.research_directions.future_innovation.fi_cohort import assign_source_folds, validate_cohort
            sources = [f"generated-source-{i // 2:02d}" for i in range(50)]
            fold_for = assign_source_folds(sources)
            cohort = pd.DataFrame({"window_id": [f"generated-window-{i}" for i in range(50)],
                                   "video_id": sources, "outer_fold": [fold_for[s] for s in sources]})
            validate_cohort(cohort)
        else:
            cohort = read_optional_table(RUN_ROOT, "manifests/gate-windows.csv")
        if cohort is not None:
            census = cohort.groupby("outer_fold").agg(windows=("window_id", "size"),
                                                      sources=("video_id", "nunique"))
            display(census)
            assert cohort.groupby("video_id").outer_fold.nunique().max() == 1
            ax = census.plot.bar(color=["#2f6f99", "#5f9e7e"], rot=0,
                                 title="Generated outer held-out folds" if MODE == "teach" else "Saved outer held-out folds")
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
            ax.set(xlabel="Held-out fold", ylabel="Count")
            plt.tight_layout(); plt.show()
        else:
            print("No frozen cohort manifest in this local copy. No census is inferred from images alone.")
        """), md("""
        ## 2. Put the observation boundary on one timeline

        Frames are zero-based after annotation conversion. Frames 0–31 supply
        the inputs; the target spans 38–39 and ends eight frames after the last
        observed frame. The teacher target uses the full 64-frame clip and can
        therefore include information after frame 39. Its interpretation is a
        full-clip contextual feature at that location.

        The timeline reads the production `FRAME` contract, keeping the displayed
        boundary tied to the implementation. The physical horizon in seconds
        depends on the source frame rate.
        """), code("""
        from gavd6_sjepa.research_directions.future_innovation.fi_contracts import FRAME
        fig, ax = plt.subplots(figsize=(9, 2.6))
        ax.broken_barh([(0, FRAME.context_stop_exclusive)], (1, 0.65), facecolors="#2f6f99")
        ax.broken_barh([(FRAME.target_tubelet_start, FRAME.tubelet_size)], (1, 0.65), facecolors="#e07a4b")
        ax.broken_barh([(0, FRAME.frames_per_clip)], (0, 0.65), facecolors="#e6f2ea")
        ax.set(yticks=[0.325, 1.325], yticklabels=["Teacher target context", "Past input / target location"],
               xticks=[0, 31, 38.5, 63], xticklabels=["0", "31", "38–39", "63"],
               xlabel="Zero-based frame", xlim=(0, 64))
        ax.set_title("Inputs use the past; target encoding uses the full clip")
        plt.tight_layout(); plt.show()
        """), md("""
        ## 3. Inspect exclusions before interpreting model scores

        A usable row needs exact decoding, retained person crops, aligned boxes
        and adequate pose coverage. Natural missingness stays explicit through
        confidence and validity channels. The pipeline saves every exclusion and
        an alignment overlay for each accepted window.

        Exclusion counts can reveal a narrow selected cohort. An overlay can
        reveal a misplaced crop; it does not certify timing, source separation,
        or teacher validity. Read the manifests as well as the pictures.

        ### How to read the common exclusion reasons

        - **Sequence shorter than 64 frames** means the sequence's inclusive
          `first_frame`–`last_frame` span contains fewer than 64 frames. The
          pipeline must choose one contiguous 64-frame window (frames 0–63 in
          the clip contract), so this sequence cannot supply a valid window.
          This is a length check, not a claim that an individual annotation is
          invalid.
        - **Source video not cached** in an older run describes the old
          `youtube/all/<video_id>` lookup, not all storage on HAIC. New runs
          resolve explicit full-source paths in the video manifest, then exact
          source IDs throughout the configured storage directories. Add other
          directories through `FI_VIDEO_ROOTS` before initialization. Ambiguous
          exports require an explicit manifest `video_path`; clips with reset
          frame numbering are not substitutes for full source videos.
        - **After fixed cohort reached 50** and **source cap** mean a candidate
          was not selected. They are not evidence of a missing or invalid video.
          This pilot deliberately selects 50 windows; accepting all available
          sequences into the study would be a different experiment.

        Existing cohorts retain their original candidate/exclusion records on
        resume. To apply new discovery or window-selection rules, initialize a
        new run directory; never silently replace a cohort behind cached features.
        """), code("""
        if MODE != "teach":
            exclusions = read_optional_table(RUN_ROOT, "manifests/exclusions.csv")
            if exclusions is not None:
                summary = exclusions.groupby(["stage", "reason"]).size().rename("sequences").reset_index()
                summary["outcome"] = np.where(summary.stage == "selection", "Not selected", "Eligibility failure")
                display(summary[["outcome", "stage", "reason", "sequences"]])
                failures = exclusions[exclusions.stage != "selection"]
                display(failures.head(10))
                print(f"{len(failures)} eligibility failures; {len(exclusions) - len(failures)} not selected by cohort size/source cap.")
            availability = read_optional_table(RUN_ROOT, "manifests/source-availability.csv")
            if availability is not None:
                display(availability.groupby(["available", "method"]).size().rename("source videos").reset_index())
                display(availability[availability.video_path.fillna("") == ""].head(10))
            else:
                print("No source-discovery inventory in this older run. Its exclusions describe the original lookup.")
            overlays = sorted((RUN_ROOT / "qc/alignment-overlays").glob("*.jpg"))
            print(f"{len(overlays)} local overlays. Their presence alone does not establish a completed cohort.")
            if overlays:
                from IPython.display import Image
                display(Image(filename=str(overlays[0]), width=720))
            display(artifact_inventory(RUN_ROOT).iloc[:3])
        """)]
    return cells + ending("01", "In teaching mode the census is generated; in execution and inspection it describes the saved cohort. Examine exclusions and alignment alongside source counts. Missing manifests or inconsistent source assignments must be resolved before fitting. Next we test the teacher's observation boundary.")


def lesson02():
    cells = opening("02", "Can we trust the teacher inputs and targets?", """
A frozen encoder can still leak future information through attention or
preprocessing. This stage explains where tokens come from, which tokens the
past input may use, and why the target must respond meaningfully to the person.
The teaching examples use token arithmetic, not downloaded teacher weights.
""")
    cells += [md("""
        ## 1. Count tokens before pooling them

        At 384 × 384 pixels, 16-pixel patches form a 24 × 24 grid. Each temporal
        token covers two frames. The full clip has 32 temporal groups; the past
        input has 16. The actual adapter removes future tokens before attention.
        Removing them only after full-clip attention would leave future content
        mixed into the retained past features.
        """), code("""
        from gavd6_sjepa.research_directions.future_innovation.fi_contracts import FRAME
        from gavd6_sjepa.research_directions.future_innovation.fi_token_regions import context_indices, region_masks
        past_ids = context_indices()
        full_count = FRAME.frames_per_clip // FRAME.tubelet_size * FRAME.grid ** 2
        display(pd.DataFrame({"tokens": [len(past_ids), full_count]}, index=["Past before attention", "Full teacher clip"]))
        assert past_ids[-1] < FRAME.context_stop_exclusive // FRAME.tubelet_size * FRAME.grid ** 2
        """), md("""
        ## 2. See which spatial region supplies the target

        The production region helper expands the person rectangle by one patch
        and leaves a further guard band before background patches. Here the box
        is invented to show that geometry. Real boxes come from the aligned
        annotations and declared crop transformation.

        Person and background targets each have their own baseline fit and
        training-variance mask. Teacher attention can mix information across
        regions, so a background target is not isolated background content.
        """), code("""
        if MODE == "teach":
            person, background = region_masks(np.array([0.35, 0.15, 0.65, 0.85]))
            regions = np.ones(FRAME.grid ** 2)
            regions[background] = 0
            regions[person] = 2
            from matplotlib.colors import ListedColormap
            fig, ax = plt.subplots(figsize=(5, 4))
            picture = ax.imshow(regions.reshape(FRAME.grid, FRAME.grid),
                                cmap=ListedColormap(["#e7f0f8", "#f7f5ef", "#5f9e7e"]), vmin=0, vmax=2)
            colorbar = fig.colorbar(picture, ax=ax, ticks=[0, 1, 2])
            colorbar.ax.set_yticklabels(["Background", "Guard band", "Person"])
            ax.set(title="Illustrative spatial target regions", xlabel="Patch column", ylabel="Patch row")
            plt.tight_layout(); plt.show()
        """), md("""
        ## 3. Try to invalidate the measurement before fitting

        | Saved audit | What must hold |
        | --- | --- |
        | Repeated teacher inference | The declared numerical tolerance is met |
        | Randomized future pixels | The past input features remain unchanged |
        | Future person replacement | The target responds more than to background replacement |
        | Crop and frame alignment | Features correspond to the declared observations |

        The numerical thresholds and measurements live in the run artifacts.
        Reading a passed flag is inspection of an earlier audit, not a new test.
        The production audit stage uses the actual frozen checkpoint and saves
        pixel-edit contact sheets. The synthetic pipeline smoke uses fabricated
        features and audit records; it cannot verify real teacher causality.
        """), code("""
        if MODE != "teach":
            from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
            audit_path = RUN_ROOT / "qc/validity-summary.json"
            if audit_path.is_file():
                audit = read_json(audit_path)
                display(pd.DataFrame(list(audit.get("checks", {}).items()), columns=["saved check", "passed"]))
                failed = [name for name, passed in audit.get("checks", {}).items() if passed is not True]
                print("Failed checks:", ", ".join(failed) if failed else "none")
                display({key: audit.get(key) for key in
                         ("motion_to_background_change_ratio", "person_edit_direction_fraction")})
                thresholds = RUN_ROOT / "config/thresholds.json"
                if thresholds.is_file():
                    display(read_json(thresholds))
                for filename in ("teacher-stability.csv", "causal-leakage.csv", "target-sensitivity.csv"):
                    table = read_optional_table(RUN_ROOT, "qc/" + filename)
                    if table is not None:
                        print(filename)
                        display(table)
                print("Saved audit only; inspection does not revalidate it." if MODE == "inspect"
                      else "Production commands were attempted; failed checks remain blocking.")
            else:
                print("No saved validity summary. The measurement has not been verified by this notebook.")
            display(artifact_inventory(RUN_ROOT).iloc[3:5])
        """)]
    return cells + ending("02", "Token arithmetic explains the interface. Teacher behavior requires the cache and audit stages, which execute mode runs or verifies here. Inspect the saved checks and resolve validity failures before comparing predictors. Fabricated smoke audits remain software tests only.")


def lesson03():
    cells = opening("03", "Does correctly paired motion help?", """
We now hold the prediction problem fixed and change only the skeleton control.
The production experiment uses five source-disjoint outer folds and three
initialization seeds. Inner source folds choose hyperparameters. Outer test
sources never choose a ridge penalty, update count, or preferred arm.

The short example below fits one baseline and one two-update head on generated
data. It demonstrates the production APIs, not the complete scientific grid.
""")
    cells += [md("""
        ## 1. Build every control inside its own partition

        A mismatched donor must be from another source in the same training,
        validation or test partition. Moving a donor across partitions changes
        the information boundary. The production helper enforces different
        sources; the nested runner supplies each partition separately.

        Shuffle moves four-frame blocks with coordinates, confidence and
        validity together. No-skeleton retains validity in the same-size head.
        The background arm retains real skeleton history and changes the target
        downstream; it is not a second skeleton transformation.
        """), code("""
        if MODE == "teach":
            from gavd6_sjepa.research_directions.future_innovation.fi_contracts import ARMS
            from gavd6_sjepa.research_directions.future_innovation.fi_controls import controlled_history
            rng = np.random.default_rng(260905)
            skeleton = rng.normal(size=(12, 32, 33, 4)).astype(np.float32)
            skeleton[..., 2] = 0.8  # confidence
            skeleton[..., 3] = 1.0  # validity
            source_ids = np.array([f"generated-source-{i // 2}" for i in range(12)])
            window_ids = np.array([f"generated-window-{i}" for i in range(12)])
            x = rng.normal(size=(12, 4))
            # This example is one training partition, never a mix of train and test.
            controls = {}
            for arm in ARMS:
                controls[arm], donors = controlled_history(arm, skeleton, window_ids, source_ids, x)
                if donors is not None:
                    source_for = dict(zip(window_ids, source_ids))
                    assert all(source_for[w] != source_for[d] for w, d in zip(window_ids, donors["donor_window_ids"]))
            assert np.array_equal(controls["no-skeleton"][..., 3], skeleton[..., 3])
            display(pd.DataFrame({"arm": list(controls), "history shape": [str(a.shape) for a in controls.values()]}))
        """), md("""
        ## 2. Fit the baseline before defining its residual

        Both scalers and the ridge fit see training examples only. Targets passed
        to the residual head use the baseline's training-standardized units.
        A lower training loss merely shows optimization on those examples.
        It cannot establish held-out prediction quality.

        This demonstration reduces width, target dimension and updates explicitly.
        Real configuration remains owned by the initialized run; never transfer
        these teaching settings into the registered comparison.
        """), code("""
        if MODE == "teach":
            import torch
            from gavd6_sjepa.research_directions.future_innovation.fi_contracts import ModelContract
            from gavd6_sjepa.research_directions.future_innovation.fi_residual_models import fit_baseline, train_head, predict_head
            y = x @ rng.normal(size=(4, 3)) + rng.normal(size=(12, 3)) * 0.2
            baseline = fit_baseline(x, y, window_ids, source_ids, alpha=1.0, variance_tolerance=1e-10)
            residual = baseline.y_scaler.transform(y) - baseline.predict(x)
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)  # Local teaching budget; restore before leaving.
                head, history = train_head(skeleton, baseline.x_scaler.transform(x), residual,
                    source_ids, baseline.valid_features, seed=7, weight_decay=0.1,
                    updates=(1, 2), model_contract=ModelContract(width=8, updates=(1, 2)), device="cpu")
                correction = predict_head(head, skeleton, baseline.x_scaler.transform(x), "cpu")
            finally:
                torch.set_num_threads(old_threads)
            assert correction.shape == y.shape and np.isfinite(correction).all()
            display(pd.DataFrame(history))
            print("Two CPU updates on generated training examples; no held-out scientific result.")
        """), md("""
        ## 3. Inspect the real workload and reuse completed stages

        The declared real grid has 5 outer folds × 3 seeds × 5 arms = 75 final
        residual fits, plus inner model selection. Person-target arms reuse a
        baseline within the same fold; background targets have a separate fit.
        Teacher features are cached once. Small heads run in CPU array tasks.

        Inspect saved configuration before submitting anything. Presence of a
        fold receipt is not a fresh validation of its predictions. Existing
        pipeline commands perform their own integrity and resumption checks.
        """), code("""
        if MODE != "teach":
            from gavd6_sjepa.research_directions.future_innovation.fi_contracts import read_json
            config_path = RUN_ROOT / "config/model-contract.json"
            if config_path.is_file():
                display(read_json(config_path))
            else:
                print("No model configuration available locally; inspect the run's config directory on HAIC.")
            display(artifact_inventory(RUN_ROOT))
            audit_path = RUN_ROOT / "qc/validity-summary.json"
            if audit_path.is_file():
                audit = read_json(audit_path)
                failed = [name for name, passed in audit.get("checks", {}).items() if passed is not True]
                if failed or audit.get("passed") is not True:
                    print("FITTING BLOCKED by teacher validity:", ", ".join(failed) or "invalid summary")
                    print("Inspect notebook 02 and its per-window QC. Do not tune thresholds to pass this run.")
        """), md("""
        ## 4. Execute these notebooks on HAIC

        Run these commands in a HAIC terminal only when you intend to execute the
        real experiment. Set the paths to your checkout and existing run. For a
        new run, first complete the environment/input setup in the
        [HAIC guide](../../../slurm/future-innovation/README.md).

        ```bash
        export GAVD6_ROOT=/path/to/gavd6
        export FI_RUN_ROOT=/path/to/existing/future-innovation/run
        cd "$GAVD6_ROOT"
        bash slurm/future-innovation/submit-fi-notebooks.sh all
        ```

        `all` arranges preparation and compute dependencies and reuses verified
        completed artifacts. Use `compute` when preparation has finished. The
        run guide describes recovery; do not delete run contracts to resume.
        The notebook itself never submits jobs. In `execute` mode its stage cell
        launches the production fit grid, using the allocated CPU resources.
        """)]
    return cells + ending("03", "Teaching mode illustrates a small fit; execute mode completes the registered fits for all folds or the selected array fold. Scientific interpretation needs out-of-fold predictions for every arm and seed across all five folds. Next we score that complete comparison; job completion alone does not establish useful motion information.")


def lesson04():
    cells = opening("04", "What does the evidence permit next?", """
The final question is whether the full comparison justifies continuing. Read
completion, validity and effect size separately. A failed job, a failed
causality audit and a valid negative prediction result require different next
actions even when a saved report labels each one STOP.

In `execute` mode this notebook scores and builds the report. `inspect` mode
only reads it. Both use the experiment's existing decision rules.
""")
    cells += [md("""
        ## 1. Understand the decision rule with invented evidence

        The production gate first checks valid finite measurements, then point
        thresholds, then source-bootstrap and seed stability. These invented
        cases show its behavior. An illustrative ADVANCE has no authority over
        a real run. Adapter training is disallowed even when this raw-skeleton
        feasibility gate advances.
        """), code("""
        if MODE == "teach":
            from gavd6_sjepa.research_directions.future_innovation.fi_gate_decision import decide_gate
            metrics = {
                "delta_r2_real": 0.06, "delta_r2_time_shuffle": 0.02,
                "delta_r2_clip_mismatch": 0.005, "delta_r2_background_target": 0.02,
                "delta_r2_no_skeleton": 0.005, "motion_to_background_change_ratio": 3.0,
                "person_edit_direction_fraction": 0.9, "bootstrap_positive_fraction": 0.95,
                "seed_real_gains": [0.06, 0.06, 0.06],
                **{key: True for key in ("data_contract_valid", "evaluation_contract_valid",
                    "controls_complete", "target_audit_complete", "target_variance_valid",
                    "teacher_stable", "causal_leakage_absent")},
            }
            scenarios = {
                "Illustrative stable gain": metrics,
                "Illustrative weak real gain": {**metrics, "delta_r2_real": 0.01, "seed_real_gains": [0.01] * 3},
                "Illustrative uncertainty": {**metrics, "bootstrap_positive_fraction": 0.7},
                "Illustrative invalid input": {**metrics, "causal_leakage_absent": False},
            }
            display(pd.DataFrame([{"invented case": name, "decision": decide_gate(values)["decision"],
                "adapter allowed": decide_gate(values)["allow_adapter_training"]}
                for name, values in scenarios.items()]))
        """), md("""
        ## 2. Establish the status of the saved evidence

        A local copy may contain only overlays or an incomplete STOP report.
        A complete report is sealed with hashes for its decision and narrative.
        Inspection uses the production seal verifier and never rewrites it.
        A seal establishes file integrity; it does not independently rerun the
        original data, causal audits, predictions or statistical analysis.

        | Inspection state | Interpretation |
        | --- | --- |
        | Unavailable | No decision in this copy; remote state is unknown |
        | Incomplete | Execution or measurement still needs repair |
        | Invalid / unverified | Resolve missing or inconsistent report evidence |
        | Synthetic | Software demonstration; no scientific advancement |
        | Complete / STOP | Read failed checks to distinguish validity failure from insufficient effect |
        | Complete / INCONCLUSIVE | Point checks passed; stability was insufficient |
        | Complete / ADVANCE | Proceed only to the measurement steps permitted by this gate |
        """), code("""
        if MODE != "teach":
            evidence = inspect_report(RUN_ROOT)
            print(evidence["state"], "—", evidence["explanation"])
            decision = evidence["decision"]
            if decision is not None:
                print("Saved reason:", decision.get("reason", "No reason recorded"))
                display(pd.DataFrame(list(decision.get("checks", {}).items()), columns=["saved check", "passed"]))
            if evidence["report_text"] is not None:
                display(Markdown(evidence["report_text"]))
        """), md("""
        ## 3. Compare controls before attributing a gain to motion

        Pool out-of-fold predictions within each seed, then average seed scores.
        Do not average fold R² values or treat repeated seeds as new people.
        Source bootstraps keep each source's windows together. The sealed report
        includes the per-seed scores, intervals and paired real-minus-control
        contrasts; use that report for numerical interpretation.

        The no-skeleton arm is required and reported, but the implemented gate
        has no additional numerical cutoff for its paired contrast. If it
        reproduces the gain, an automatic ADVANCE alone does not establish a
        specific motion contribution. State that limitation and investigate it
        without inventing a new post-result threshold.

        Background targets remain contextual and the horizon is located inside
        full-clip teacher features. No result here establishes a clinical
        endpoint, isolated dynamics, or the benefit of a trained S-JEPA student.
        """), md("""
        ## 4. Record one next action

        | Observation | Next action |
        | --- | --- |
        | Missing or incomplete artifacts | Recover the existing run through its normal pipeline |
        | Failed input or teacher validity | Repair the measurement before interpreting prediction scores |
        | Complete, valid gain below the rule | Preserve the negative result; reconsider the representation or endpoint |
        | Gain reproduced by controls | Investigate the remaining shortcut or capacity explanation |
        | Unstable gain | Report uncertainty and define any further measurement before running it |
        | Complete stable advance with interpretable controls | Follow the proposal's next measurement stage; adapter training remains gated |

        Update the study overview with the identified run, exact report path,
        supported conclusion, unresolved explanation, and next action. Keep
        synthetic examples and older prompts visibly separate from that record.
        Preserve the executed notebook as an artifact when adding result-specific
        commentary; edit the builder for changes to the reusable explanation.
        """)]
    return cells + ending("04", "Record the identified run, its measurement status, the supported conclusion and one next action. Execution scores and seals complete evidence; inspection reads the existing seal. A complete negative result is useful evidence, while an incomplete STOP calls for execution or measurement repair.")


LESSONS = {"00": lesson00, "01": lesson01, "02": lesson02, "03": lesson03, "04": lesson04}


def render(number):
    notebook = nbformat.v4.new_notebook(cells=LESSONS[number]())
    notebook.metadata.update({
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "editable_source": BUILDER,
        "evidence_boundary": "Teach is non-evidentiary; inspect is read-only; execute runs the existing Experiment 0 CLI.",
    })
    for index, cell in enumerate(notebook.cells):
        cell.id = hashlib.sha256(f"{number}:{index}:{cell.source}".encode()).hexdigest()[:12]
        if cell.cell_type == "code":
            compile(cell.source, f"{NAMES[number]}:cell-{index}", "exec")
    nbformat.validate(notebook)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=tuple(NAMES), nargs="+", default=list(NAMES))
    checks = parser.add_mutually_exclusive_group()
    checks.add_argument("--check", action="store_true")
    checks.add_argument("--check-sources", action="store_true")
    args = parser.parse_args()
    for number in args.only:
        expected = render(number)
        path = DESTINATION / NAMES[number]
        if args.check or args.check_sources:
            actual = nbformat.read(path, as_version=4)
            if args.check_sources:
                comparable = lambda nb: [(cell.cell_type, cell.source) for cell in nb.cells]
            else:
                comparable = lambda nb: json.loads(nbformat.writes(nb))
            if comparable(actual) != comparable(expected):
                raise SystemExit(f"Notebook differs from its builder: {path}")
            print(f"Checked {path.name}")
        else:
            if path.exists() and any(c.get("outputs") for c in nbformat.read(path, as_version=4).cells):
                raise SystemExit(f"Preserve executed {path.name} outside the source tree before regeneration.")
            path.parent.mkdir(parents=True, exist_ok=True)
            nbformat.write(expected, path)
            print(f"Built {path.name}")


if __name__ == "__main__":
    main()
