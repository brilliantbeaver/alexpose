"""Build the step-by-step synthetic teaching experiment notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks/synthetic_training"


def md(text: str):
    """Make a Markdown explanation cell."""
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    """Make an unexecuted Python cell."""
    return nbf.v4.new_code_cell(dedent(text).strip())


INTRO = """
**Question:** can a short training response tell us which synthetic lesson
will improve an unfamiliar pose estimator on real video?

The source sequence is **00 → 01 → 02 → 03 → 07**. Run 02 once per configured
source student. After the source decision, prepare independent GAVD references
in 04, deploy without reading their labels in 05, and exchange selected lessons
in 08. Explicitly evaluate in 06. These notebooks contain no precomputed
research results or substitute models.

[Proposal](../../notes/research-agenda/proposals/synthetic-training-selection.md)
· [Notebook guide](README.md)
· [HAIC setup and launch commands](../../slurm/synthetic-training/README.md)
"""

SETUP = """
from pathlib import Path
import json
import os
import sys
from time import perf_counter

root_override = os.environ.get("GAVD6_ROOT")
candidates = ([Path(root_override).expanduser()] if root_override else
              [Path.cwd(), *Path.cwd().parents])
PROJECT_ROOT = next((p.resolve() for p in candidates
                     if (p / "src/gavd6_sjepa").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Set GAVD6_ROOT to the gavd6 checkout.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)
os.environ.setdefault("GAVD6_ROOT", str(PROJECT_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Markdown, SVG, display
from gavd6_sjepa.research_directions.synthetic_training.config import RunConfig
from gavd6_sjepa.research_directions.synthetic_training import workflow

cfg = RunConfig.from_env()
RUN_ROOT = cfg.root
get_ipython().run_line_magic("matplotlib", "inline")
plt.rcParams.update({"figure.figsize": (9, 4), "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})

def show_result(result):
    # Display the tables and artifact paths returned by a workflow stage.
    if isinstance(result, pd.DataFrame):
        display(result)
    elif isinstance(result, dict):
        for name, value in result.items():
            display(Markdown(f"### {name.replace('_', ' ')}"))
            if isinstance(value, pd.DataFrame):
                display(value)
            elif isinstance(value, Path) and value.suffix == ".svg" and value.is_file():
                display(SVG(filename=str(value)))
            else:
                print(json.dumps(value, indent=2, default=str) if isinstance(value, (list, dict)) else value)
    else:
        print(result)

print(f"Run: {RUN_ROOT}")
print(f"Context representation: {cfg.context_kind}; device: {cfg.device}")
"""


def make(title: str, cells: list):
    """Give every notebook the same navigation and fresh-kernel setup."""
    notebook = nbf.v4.new_notebook(cells=[md(title), md(INTRO), code(SETUP), *cells])
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return notebook


def render_all() -> dict:
    """Return tutorials whose code cells call the shared study implementation."""
    result = {}
    result["00_question_and_assets.ipynb"] = make(
        """
        # 00 · The question and the assets

        **Goal:** distinguish a potentially useful teaching mechanism from
        ordinary gains caused by additional training.
        """, [
        md("""
        ## 1. Understand the experiment in one example

        A pose estimator predicts twelve visible body landmarks. A **lesson**
        is a small labeled set of rendered AMASS images, such as oblique views
        or partly occluded people. A **probe** is a common short training update.
        A **selector** uses the estimator's response to choose its next lesson.

        We need four findings in sequence:

        1. Synthetic lessons can improve real accuracy beyond equal-budget replay.
        2. Different students or settings benefit from different lessons.
        3. Target-video prediction changes select better lessons than current
           weaknesses and source learning progress alone.
        4. A selector learned on source trials transfers to a held architecture.

        Improved confidence or lower training loss does not establish real
        accuracy. A useful nearest-neighbor selector is enough to test the
        mechanism; a more complicated teacher is not itself the contribution.
        """),
        code("""
        display(pd.DataFrame([
            ("train", "Fit source outcome predictors", "AMASS source references"),
            ("validation", "Choose selector settings and comparators", "Separate source students and references"),
            ("held", "Test transfer to an unseen architecture", "Never enters source fitting or model selection"),
        ], columns=["Student role", "Purpose", "Reference boundary"]))
        display(pd.DataFrame(cfg.students))
        """),
        md("""
        ## 2. Inspect actual asset availability

        This reads the repository's AMASS and GAVD manifests and checks local
        paths. It does not infer availability from a manifest count or claim
        that a checkpoint has successfully loaded. The following stage needs
        full-body AMASS files, licensed body models, compatible texture/UV
        assets, photographic backgrounds, and labeled COCO replay data.

        Model configuration files and their corresponding released weights
        must be present. Use the HAIC guide's isolated MMPose environment when
        the main project's PyTorch version is incompatible with MMCV.
        """),
        code("""
        started = perf_counter()
        inventory = workflow.inventory(cfg)
        show_result(inventory)
        print(f"Inventory took {perf_counter() - started:.1f} seconds.")
        """),
        md("""
        ## 3. Keep the interpretation narrow

        GAVD supplies real video, not existing accurate twelve-joint reference
        coordinates. Those visible landmarks will be independently annotated.
        AMASS projected labels describe the rendered geometry. Neither source
        establishes forces, 3D clinical accuracy, or hidden-joint accuracy.

        A simple-context run can test the teaching pipeline, but cannot support
        a JEPA-specific claim. A JEPA run must actually load the configured
        encoder and compare its features with simpler context representations.

        **Continue when:** assets are available and you can prepare useful RGB
        lessons and independent references. Missing appearance or human labels
        is a real dependency, not a reason to substitute model predictions.

        Next: [01 · Prepare source data](01_prepare_source_data.ipynb).
        """),
    ])
    result["01_prepare_source_data.ipynb"] = make(
        """
        # 01 · Prepare source data with separate roles

        **Goal:** create labeled training lessons and independent data that
        measure what each lesson actually teaches.
        """, [
        md("""
        ## 1. Separate training from measurement

        The common probe, eight lessons, and labeled diagnostic bank use
        different source-training motions. Context clips tell the selector
        which environment a student will face. Reference clips measure the
        resulting accuracy. The context and reference motions are separate;
        changing a camera or texture does not create a new independent motion.

        The existing AMASS subject registry and split assignments determine
        eligibility. Validation contexts and references use excluded people.
        Every rendering of a motion keeps its original role.
        """),
        code("""
        display(pd.DataFrame([
            ("probe", "One shared short update", "Labels train the student"),
            ("lesson", "Candidate training choices", "Labels train the student"),
            ("diagnostic", "Known synthetic weaknesses", "Error summaries are selector inputs"),
            ("context", "Simulated deployment collection", "Reference coordinates are not selector inputs"),
            ("reference", "Measure lesson utility", "Errors supply source teacher targets only"),
            ("COCO replay", "Retain real-image pose competence", "Labels train the student"),
        ], columns=["Data role", "Purpose", "Use of labels"]))
        """),
        md("""
        ## 2. Render the lesson library and prepare replay

        Each rendered RGB image has a fixed person crop, projected landmarks,
        visibility flags, and its motion/person provenance. Texture, background,
        and viewpoint variation should not accidentally identify the answer.
        Full-body AMASS meshes support shoulders, arms, hips, knees, and ankles.

        The renderer requires compatible UV topology and real texture images;
        it does not silently replace these with a colored stick figure. View,
        resolution, blur, and occlusion are controlled appearance conditions,
        not simulated clinical diagnoses.

        COCO replay is a fixed set of labeled real examples. At the default
        mixture, synthetic branches use 90% replay and 10% lesson images.
        Replay-only fills every slot with real images, so it sees more real
        examples at the same total update budget.
        """),
        code("""
        started = perf_counter()
        prepared = workflow.prepare_data(cfg)
        show_result(prepared)
        print(f"Data preparation took {(perf_counter() - started) / 60:.1f} minutes.")
        """),
        code("""
        # Inspect saved assignments before spending GPU time on teaching trials.
        for name in ("synthetic.csv", "replay.csv"):
            path = RUN_ROOT / "data" / name
            if path.is_file():
                table = pd.read_csv(path)
                display(Markdown(f"### {name}: {len(table):,} frames"))
                display(table.head(8))
                groups = [c for c in ("role", "domain_id", "lesson_id") if c in table]
                if groups:
                    display(table.groupby(groups, dropna=False).size().rename("frames").to_frame())
        """),
        md("""
        ## 3. Check the visual task

        Inspect representative images and labels in the prepared data folder.
        Are body landmarks visible at the intended resolution? Do rendered
        shoulders and hips follow the same convention as real COCO landmarks?
        Do viewpoint and appearance changes preserve meaningful pose labels?

        Short pilot budgets measure whether the experiment runs and whether
        choices differ. They are not validated training recipes. Fix any
        convention or rendering problems before collecting the utility table.

        """),
        code("""
        from gavd6_sjepa.research_directions.synthetic_training.data import PoseFrameDataset, load_pose_manifest

        source_index = load_pose_manifest(RUN_ROOT / "data/synthetic.csv")
        examples = source_index.loc[source_index.role.eq("lesson")].groupby("lesson_id", sort=True).head(1)
        gallery = PoseFrameDataset(examples)
        columns = 4
        rows = (len(gallery) + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(12, 3.5 * rows), squeeze=False)
        for position, axis in enumerate(axes.flat):
            axis.axis("off")
            if position >= len(gallery):
                continue
            sample = gallery[position]
            visible = sample["visible"]
            axis.imshow(sample["image"])
            axis.scatter(*sample["keypoints"][visible].T, s=15, c="#f43f5e", edgecolors="white", linewidths=.4)
            axis.set_title(str(sample["metadata"]["lesson_id"]))
        fig.tight_layout()
        plt.show()
        """),
        md("""
        Next: [02 · Measure source trials](02_measure_source_trials.ipynb),
        once per train/validation student.
        """),
    ])
    result["02_measure_source_trials.ipynb"] = make(
        """
        # 02 · Measure what each lesson teaches

        **Goal:** build one source student's outcome table from independent
        adaptation branches, all starting at the same checkpoint.
        """, [
        md("""
        ## 1. Select a source student

        Set `ST_STUDENT_ID` before starting this notebook. The Slurm launcher
        creates one array task per configured train/validation student. Held
        architectures are excluded from this stage.

        The original estimator is $M_0$. Record its target predictions and
        labeled diagnostic errors, apply the common probe, and obtain $M_p$.
        Record predictions again on exactly the same frames. Coordinate changes
        measure the response to training, not whether the change is correct.
        """),
        code("""
        STUDENT_ID = os.environ.get("ST_STUDENT_ID")
        if not STUDENT_ID:
            raise ValueError("Set ST_STUDENT_ID to one configured train/validation student.")
        student = cfg.student(STUDENT_ID)
        if student["role"] == "held":
            raise ValueError("Held students cannot enter source teaching trials.")
        display(pd.DataFrame([student]))
        print(f"Probe updates: {cfg.probe_steps}; remaining budgets: {cfg.adaptation_steps}")
        """),
        md(r"""
        ## 2. Fork each candidate from the same post-probe student

        For every declared remaining budget, independently adapt $M_p$ using
        replay alone or replay plus one lesson. Do not train the lessons in a
        sequence. Give each branch the same optimizer initialization and update
        count. Reuse a trained branch across reference settings instead of
        repeating the training job for each setting.

        The teacher's target is **gain over replay**:

        $$\mathrm{gain}(k) = E_{\mathrm{replay}} - E_k.$$

        Positive gain means lesson $k$ reduced independent reference error more
        than replay. Negative gain means it was worse. Replay has gain zero and
        remains an available choice. Full-budget replay from $M_0$ is also
        retained to test whether the whole probe-and-lesson procedure is useful.
        """),
        code("""
        started = perf_counter()
        trials = workflow.source_trials(cfg, student_id=STUDENT_ID)
        show_result(trials)
        print(f"Student trials took {(perf_counter() - started) / 3600:.2f} hours.")
        """),
        md("""
        ## 3. Look for a selection opportunity

        Different lessons must produce meaningfully different gains. Different
        students or contexts should sometimes favor different choices. Uniform
        gains are ordinary augmentation, not evidence for a personalized teacher.

        Many domain rows reuse the same adapted checkpoint. They are not
        independent training experiments. Count students and independent
        interventions alongside the number of rows in the outcome table.

        Source trials may use labeled synthetic references to measure every
        lesson. Deployment will choose one lesson before accessing real labels.

        After all source tasks complete, continue to
        [03 · Fit and freeze selectors](03_fit_and_freeze_selectors.ipynb).
        """),
    ])
    result["03_fit_and_freeze_selectors.ipynb"] = make(
        """
        # 03 · Fit selectors and freeze the real experiment

        **Goal:** choose the teacher and its comparators using source outcomes
        alone, then save the decisions used for real deployment.
        """, [
        md("""
        ## 1. Give the comparators precisely defined information

        Before-only uses the original snapshot. After-only uses the post-probe
        snapshot. Before-plus-change receives both the original state and the
        transition. All three select training for the same $M_p$ with the same
        remaining budget. These are compact summaries, not complete prediction
        tensors. Paired target-change features retain signed, absolute, and
        RMS displacements. Subtracting two separately pooled snapshots does
        not recover all of those paired statistics.

        The stronger primary comparison controls for **source learning
        progress**. Both selectors receive post-probe target predictions, source
        probe loss history, both synthetic diagnostic snapshots, descriptors,
        context features, and budget. Only the full teacher also sees target
        prediction change. This tests information from unlabeled target video
        beyond information from labeled source training. The strict matched
        comparator uses the full teacher's selected model family and parameter.
        A separately tuned source-progress comparator also tests the strongest
        available alternative under the same validation budget.
        """),
        md("""
        ## 2. Fit on training students; select settings on validation students

        Start with nearest-neighbor lookup and small regularized predictors.
        Choose settings by the reference error after the selected lesson, not
        just by regression fit to the utility numbers. Fit normalization using
        training episodes only. A random row split would leak nearly identical
        trials across the boundary.

        Other controls include fixed/balanced/random lessons, synthetic weakness,
        scalar response magnitude, simple context, and equal-budget replay.
        A simple selector that uses the response can validate the mechanism.
        A neural selector does not have to beat it for the information to help.
        """),
        code("""
        fitted = workflow.fit_selectors(cfg)
        show_result(fitted)
        """),
        md("""
        ## 3. Read the source decision before running real evaluation

        The saved selector configuration specifies the primary method,
        comparators, feature transformations, and source-selected settings.
        No GAVD reference labels or held-architecture lesson outcomes enter this
        decision. A prescribed deployment probe on a held student is allowed;
        using its lesson outcomes to retune the teacher is not.

        A held model's release recipe may be checked for technical compatibility.
        Its learning rate must not be optimized against its held lesson outcomes.
        If source choices do not improve selected utility over simple controls,
        do not describe later real improvements as an established response effect.

        Next inspect [07 · Source mechanism report](07_source_mechanism_report.ipynb).
        Prepare GAVD only after making the source continuation decision.
        """),
    ])
    result["04_prepare_real_evaluation.ipynb"] = make(
        """
        # 04 · Prepare real context and independent reference frames

        **Goal:** define recording-disjoint GAVD collections and prepare a
        human annotation template without creating pose pseudolabels.
        """, [
        md("""
        ## 1. Select recordings using visible scene information

        GAVD context, early evaluation, and confirmation recordings must be
        separate. Existing study reservations remain excluded. Recording
        independence does not establish that every person is unique.

        The first call writes a view/crop template when checked metadata is
        missing. Review each proposed recording's coarse view, source-person
        height (`person_height_px`), and fixed person crop. Use `side`, `oblique`, or `frontal_rear`
        and explicitly mark `view_checked=true`. Full video height is not the
        person's source resolution. Set `ST_GAVD_VIEWS` to the completed CSV,
        restart this notebook, and rerun the preparation stage.
        """),
        code("""
        prepared = workflow.prepare_gavd(cfg)
        show_result(prepared)
        """),
        md("""
        ## 2. Understand the two real-data roles

        Context recordings supply unlabeled images, predictions, and response
        information to the selector. Independent reference recordings supply
        images for prediction and human labels for later scoring. The selector
        never receives those human labels.

        The planned groups cross three coarse views with low/high source-person
        resolution. Per setting, the defaults are three context recordings,
        four early references, and six confirmation references. These are
        requested counts, not guaranteed available data. Do not silently reuse
        a recording if a group is too small.
        """),
        code("""
        for name in ("gavd_context.csv", "gavd_evaluation.csv"):
            path = RUN_ROOT / "data" / name
            if path.is_file():
                table = pd.read_csv(path)
                display(Markdown(f"### {name}"))
                display(table.head(8))
                groups = [c for c in ("role", "domain_id") if c in table]
                if groups and "recording_id" in table:
                    display(table.groupby(groups).recording_id.nunique().rename("recordings").to_frame())
        """),
        md("""
        ## 3. Annotate references independently

        Open `data/annotate-early.html` or `data/annotate-confirmation.html` in
        a local browser after copying the run's `data` directory with its
        `gavd` image tree. Each page displays the exported full-frame PNGs.
        Enter the annotator name, mark the independent reference box with two
        clicks, and mark each visible joint or label it hidden. Save JSON
        progress between sessions, then export the completed CSV.

        The page exports separate early and confirmation annotation CSVs:
        `frame_id`, `landmark`, `x`, `y`, `visible`, `box_x1`, `box_y1`, `box_x2`,
        `box_y2`, `annotator`, and `reviewer`. Coordinates refer to the exported
        full image. Mark twelve rows per frame: left/right shoulders, elbows,
        wrists, hips, knees, and ankles. For hidden or uncertain landmarks set
        `visible=false` and leave coordinates blank. Do not guess them.

        Independently mark the person's reference box on each frame. Its
        diagonal supplies the model-independent error scale. A second reviewer
        should load the saved progress, inspect each frame, and mark it
        reviewed. Confirmation requires a reviewer different from its annotator.
        Never initialize
        the reference coordinates from one of the evaluated models.

        Save completed `gavd-early-annotations.csv` and
        `gavd-confirmation-annotations.csv` in one directory and point
        `ST_GAVD_ANNOTATIONS` to that directory before notebook 06. A path
        pattern containing `{split}` is also accepted. Evaluation opens only
        the requested split; confirmation requires a second reviewer.
        Predictions in notebook 05 do not require either CSV. Templates are
        not valid reference data until a human completes them.

        Next: [05 · Choose and adapt](05_choose_and_adapt.ipynb).
        """),
    ])
    result["05_choose_and_adapt.ipynb"] = make(
        """
        # 05 · Choose lessons from unlabeled GAVD video

        **Goal:** deploy the frozen selectors, adapt each student with its
        selected lessons, and save predictions before opening reference labels.
        """, [
        md("""
        ## 1. Use only information available at deployment

        Set `ST_STUDENT_ID` to a configured estimator. The deployment array can
        include the held architecture. Each collection supplies its context
        clips. Measure the common probe response on those clips and use the
        frozen source teacher to choose a lesson at the declared budget.

        Labeled synthetic diagnostics remain allowed deployment inputs. They
        were fixed before real evaluation and do not contain GAVD references.
        We are testing a supervised adaptation procedure chosen using unlabeled
        target video, not label-free student training.
        """),
        code("""
        STUDENT_ID = os.environ.get("ST_STUDENT_ID")
        if not STUDENT_ID:
            raise ValueError("Set ST_STUDENT_ID to a configured deployment student.")
        display(pd.DataFrame([cfg.student(STUDENT_ID)]))
        """),
        md("""
        ## 2. Run equal-budget branches and save real predictions

        Every mechanism comparison adapts the same post-probe checkpoint for
        the same remaining number of updates. Replay-only, the original
        estimator, probe-only, and full-budget replay preserve the practical
        comparison. Fixed/random/balanced choices test ordinary augmentation.

        Select once for a collection, then evaluate on its separate recordings.
        Do not choose a different lesson after seeing a reference frame's error.
        When selectors choose the same lesson, their shared prediction result
        is expected; it is not evidence that one information source helped.
        """),
        code("""
        started = perf_counter()
        deployment = workflow.deploy(cfg, student_id=STUDENT_ID)
        show_result(deployment)
        print(f"Deployment took {(perf_counter() - started) / 3600:.2f} hours.")
        """),
        md("""
        ## 3. Keep decision and measurement separate

        The saved choices and predictions are the outputs of deployment.
        Notebook 06 reads independent human references. This stage must not
        search all real lesson outcomes to decide which lesson would have won.

        The primary teacher and comparator were chosen on source validation.
        If an early real result prompts a method change, declare the early set
        as labeled development and retain confirmation recordings untouched.

        Once all students have completed deployment, continue to
        [08 · Exchange selected lessons](08_exchange_selected_lessons.ipynb).
        Then explicitly select `early` or `confirmation` in notebook 06.
        """),
    ])
    result["06_measure_real_accuracy.ipynb"] = make(
        """
        # 06 · Measure real accuracy against independent references

        **Goal:** compare frozen decisions on the same visible landmarks and
        recording groups, with practical gain and mechanism evidence separated.
        """, [
        md("""
        ## 1. Choose the evaluation split explicitly

        `ST_EVALUATION_SPLIT=early` is the initial real decision. Set
        `ST_EVALUATION_SPLIT=confirmation` only for the reserved final result.
        The source pipeline never calls this notebook. Reference annotations
        must be completed independently and selected with `ST_GAVD_ANNOTATIONS`.

        Under the strict protocol, early results only determine continue/stop.
        Changes chosen using early outcomes turn that portion into labeled
        development. Report that distinction in any paper.
        """),
        code("""
        EVALUATION_SPLIT = os.environ.get("ST_EVALUATION_SPLIT", "early")
        if EVALUATION_SPLIT not in {"early", "confirmation"}:
            raise ValueError("Choose early or confirmation.")
        print(f"Opening independent {EVALUATION_SPLIT} references.")
        print(f"Annotation CSV: {cfg.gavd_annotations_csv}")
        """),
        md("""
        ## 2. Score the same anatomical observations

        For each visible landmark, measure the Euclidean pixel distance between
        the prediction and the human reference. Divide by the independently
        annotated person's box diagonal. Average landmarks within each frame,
        then frames within each recording, and finally recordings. Lower is
        better. The reference mask and scale are identical across all methods.

        Missing predictions receive the configured failure penalty; dropping
        their frames would reward model failures. Hidden reference joints are
        outside this visible-landmark task. Report annotation coverage and
        failed predictions alongside the error table.

        Paired recording intervals compare methods on the same recordings.
        Many recordings may share one lesson choice, so an interval over
        recordings alone does not establish transfer to many new environments.
        """),
        code("""
        evaluation = workflow.evaluate(cfg, split=EVALUATION_SPLIT)
        show_result(evaluation)
        """),
        md("""
        ## 3. Read the evidence in the right order

        1. **Practical value:** does the complete method improve on the original
           student and full-budget replay?
        2. **Value of selection:** does it improve on fixed, random, or balanced
           synthetic lessons?
        3. **Proposed mechanism:** does the full teacher beat the matched selector
           without target prediction change, and the strongest snapshot control?
        4. **Transfer:** is that advantage present for the architecture excluded
           from source fitting and validation?
        5. **JEPA role:** do frozen video features add value beyond simple context
           and alternative features while response information stays fixed?

        A favorable first comparison alone is an adaptation result. Source
        ranking differences alone are an opportunity, not proof of successful
        real selection. Report effect sizes, uncertainty, and cases where replay
        was selected, without converting them into a paper-acceptance probability.
        """),
    ])
    result["07_source_mechanism_report.ipynb"] = make(
        """
        # 07 · Interpret the source mechanism before real evaluation

        **Goal:** inspect measured lesson utilities and source-held selection
        performance before committing to the real transfer experiment.
        """, [
        md("""
        ## 1. Ask whether there is useful information to learn

        This report uses source trial outcomes and the source-validation
        comparison only. It does not read independent GAVD landmark outcomes.

        Look for differences in which lesson works best, useful achieved gain
        over replay, and a gap between selected gain and the best available
        lesson. The latter gap is **regret**. An oracle uses every lesson's
        reference outcome and is a diagnostic, not a deployment method.
        """),
        code("""
        report = workflow.report(cfg)
        show_result(report)
        for figure in sorted((RUN_ROOT / "reports").glob("*.svg")):
            display(SVG(filename=str(figure)))
        """),
        md("""
        ## 2. Challenge a promising-looking table

        A response can encode the model family, learning rate, or overall update
        size. The primary matched source-progress control asks whether changes
        on the target clips add information beyond those simpler explanations.
        Read per-student results, not only a pooled average across domain rows.

        A useful model-specific teaching effect requires good decisions, not
        accurate prediction of one student's overall gain scale. Compare
        response-informed and response-free selectors with matched capacity,
        source database, budget, and context features.

        If every selector chooses replay, the rejection option worked but
        useful synthetic teaching has not been demonstrated. If one fixed
        lesson wins almost everywhere, prioritize that simple explanation.
        """),
        md("""
        ## 3. Make a bounded continuation decision

        Continue when lessons offer useful varied gains and the frozen teacher
        selects them better than strong simple controls on excluded source
        students. Source success is still not evidence of real GAVD transfer.

        A few checkpoints provide a feasibility result, not a population of
        learning behaviors. Expand independent students and settings only after
        measuring throughput and seeing a usable signal. Keep held-architecture
        outcomes out of source fitting throughout.

        If proceeding, start [04 · Prepare real evaluation](04_prepare_real_evaluation.ipynb).
        The [proposal](../../notes/research-agenda/proposals/synthetic-training-selection.md)
        defines the claims that each comparison can support.
        """),
    ])
    result["08_exchange_selected_lessons.ipynb"] = make(
        """
        # 08 · Exchange the students' selected lessons

        **Goal:** test whether a student's own selected lesson suits it better
        than the lesson selected for another student in the same collection.
        """, [
        md("""
        ## 1. Start from decisions already made without real labels

        Complete notebook 05 for all intended students first. Suppose the
        frozen teacher chose lesson A for one estimator and lesson B for
        another. Evaluate both students after each selected lesson: A/A, A/B,
        B/A, and B/B. Include pairs whose decisions agree; do not keep only
        interesting-looking differences.

        This **crossover** tests whether the selected training is specific to
        the student. It does not by itself show that the probe response caused
        the advantage. Static weaknesses or model-family information might
        explain it, so the matched feature controls remain essential.
        """),
        code("""
        STUDENT_ID = os.environ.get("ST_STUDENT_ID")
        if not STUDENT_ID:
            raise ValueError("Set ST_STUDENT_ID for the student receiving exchanged lessons.")
        display(pd.DataFrame([cfg.student(STUDENT_ID)]))
        """),
        md("""
        ## 2. Train missing exchanged branches at the same budget

        Read the saved full-teacher choices. For each receiving estimator,
        apply the other students' selected lessons from its same post-probe
        checkpoint, at the same remaining budget and recipe. Reuse already
        available branch predictions. This avoids repeating an identical
        adaptation when two selectors chose the same lesson.

        Human reference coordinates are still unopened. The exchange follows
        preselected choices, not an exhaustive search for the lesson with the
        lowest real error. Use `ST_DEPENDENCY` when queuing this array before
        the deployment array has completed.
        """),
        code("""
        exchanged = workflow.crossover(cfg, student_id=STUDENT_ID)
        show_result(exchanged)
        """),
        md("""
        ## 3. Score the exchange with the main experiment

        Notebook 06 reads these additional predictions alongside the main
        methods when crossover outputs are available. Report how many pairs
        chose different lessons, each student's own-versus-exchanged error,
        and the agreed-choice cases. Agreement provides no opportunity to
        demonstrate a personalized selection advantage.

        A student-specific gain complements, but cannot replace, the primary
        full-versus-source-progress comparison or the held-architecture result.

        Next: [06 · Measure real accuracy](06_measure_real_accuracy.ipynb).
        """),
    ])
    return result


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, notebook in render_all().items():
        # Stable cell IDs make regenerated tutorials easy to review.
        for index, cell in enumerate(notebook.cells):
            cell.id = f"{name[:2]}-{index:02d}"
        path = DESTINATION / name
        nbf.write(notebook, path)
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
