"""Build the readable Proposal 01 experiment and mechanism-diagnostic notebooks."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
DESTINATION = ROOT / "notebooks/motion_preservation"


def md(text):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text):
    return nbf.v4.new_code_cell(dedent(text).strip())


INTRO = """
Run each notebook in a fresh kernel, in order **00 → 04**. Notebook 05 is
an external visual stress test. These are offline restoration experiments:
the model may inspect the declared complete clip. They do not claim causal
forecasting or clinical diagnosis.

**The default is real data.** Export `MP_RUN_ROOT`, the AMASS and GAVD data
paths, and the model configuration before opening Jupyter. See the
[launch guide](../../slurm/motion-preservation/README.md).
For a CPU walkthrough of the mechanics, explicitly choose `MP_MODE=demo`
and a separate run directory. Demo outputs cannot establish a research result.

[Proposal](../../docs/studies/motion-preservation/protocol/proposal.md)
· [Notebook guide](README.md)
"""

SETUP = """
from pathlib import Path
import json
import os
import sys
from time import perf_counter

project_override = os.environ.get("GAVD6_ROOT")
candidates = ([Path(project_override).expanduser()] if project_override else
              [Path.cwd(), *Path.cwd().parents])
PROJECT_ROOT = next((p.resolve() for p in candidates
                     if (p / "src/gavd6_sjepa").is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError("Set GAVD6_ROOT to the gavd6 checkout.")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)  # Resolve manifest/config paths from the checkout in every kernel.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import HTML, Markdown, Video, display
from gavd6_sjepa.research_directions.motion_preservation import workflow, plots

get_ipython().run_line_magic("matplotlib", "inline")
plt.rcParams.update({"figure.figsize": (9, 3.5), "font.size": 11,
                     "axes.spines.top": False, "axes.spines.right": False})
cfg = workflow.config_from_environment()
RUN_ROOT = Path(cfg.run_root)
print(f"Mode: {cfg.mode}; run directory: {RUN_ROOT}")
if cfg.mode == "demo":
    display(Markdown("**DEMO ONLY: generated fixtures and stand-in models. "
                     "These outputs are not evidence about AMASS, GAVD, or a pretrained prior.**"))
"""


def make(title, cells):
    notebook = nbf.v4.new_notebook(cells=[md(title), md(INTRO), code(SETUP), *cells])
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"}}
    return notebook


def render_all():
    result = {}
    result["00_data_and_question.ipynb"] = make(
        """
        # 00 · The question, available data, and the first decision

        **Can we remove tracker errors without deleting movements that really
        happened?** This notebook establishes what data and public-model inputs
        are available. The first scientific result is a preservation-versus-repair
        comparison, not binary gait classification.
        """, [
        md("""
        ## 1. Why ordinary reconstruction error is not enough

        Suppose a real ankle excursion lasts only a few frames. Removing that
        excursion may slightly reduce the average error across the entire body,
        even though the useful movement information was lost.

        We need two separate measurements:

        - **Repair:** how much squared error at observed joints is removed?
        - **Preservation:** how much of the known real movement remains?

        Filling missing joints is measured separately. An interpolated gap is
        a model input, not an observation that can earn tracking-repair credit.

        We compare methods at a repair strength chosen on separate calibration
        people. A method cannot win by leaving every noisy coordinate unchanged.
        """),
        code("""
        # This table explains the design; it contains no measured results.
        display(pd.DataFrame([
            {"True event": False, "Tracking noise": False, "Desired behavior": "Keep the clean movement"},
            {"True event": True,  "Tracking noise": False, "Desired behavior": "Keep the event"},
            {"True event": False, "Tracking noise": True,  "Desired behavior": "Remove the error"},
            {"True event": True,  "Tracking noise": True,  "Desired behavior": "Preserve the event and remove the error"},
        ]))
        """),
        md("""
        ## 2. Read the existing AMASS and GAVD manifests

        AMASS supplies known body motion for controlled experiments. We use
        whole-body joints so arms and trunk can carry a true event. GAVD supplies
        in-the-wild videos for external inspection. Its estimated trajectories
        are not 3D reference truth.

        Inventory counts come from the repository manifests. Availability comes
        from the configured filesystem on this machine. A manifest entry does
        not imply that its raw file is available locally.
        """),
        code("""
        started = perf_counter()
        inventory = workflow.inventory(cfg)
        for name in ("amass", "gavd", "availability"):
            table = inventory[name]
            display(Markdown(f"### {name.capitalize()} ({len(table):,} rows)"))
            display(table.head(12))
        print(f"Inventory completed in {perf_counter() - started:.1f} seconds.")
        """),
        code("""
        # Dataset/identity summaries are descriptive, not an eligibility decision.
        amass = inventory["amass"]
        for column in ("source_dataset", "split", "role"):
            if column in amass:
                display(amass.groupby(column, dropna=False).size().rename("manifest_rows").to_frame())
        identity_column = next((c for c in ("person_id", "identity", "subject_id_candidate", "audited_subject_id")
                                if c in amass), None)
        if identity_column:
            print(f"AMASS identities/groups represented: {amass[identity_column].nunique():,}")
        gavd = inventory["gavd"]
        if "video_id" in gavd:
            print(f"GAVD source recordings represented: {gavd.video_id.nunique():,}")
        """),
        md("""
        ## 3. Keep development and final evaluation separate

        People are assigned before generating event or corruption variants. All
        variants of a person stay together. Train the gate on an arm-leg timing
        event. Use a foot-clearance event on development people for the first
        decision. Once that result affects our choices, it is development data.
        The trunk-pelvis event on final people remains unopened until notebook
        04 is explicitly launched in final mode.

        The current final configuration changes the event family, camera angle
        and corruption mechanism together. This is a combined stress test; it
        does not isolate which individual change caused a success or failure.

        Audited AMASS identity determines the grouping when available. Uncertain
        identities support only a weaker group-level claim. A new adaptation
        split does not establish absence from a public prior's training data.
        """),
        code("""
        display(pd.DataFrame([
            ("train", "Learn the small gate", "Development event and corruption settings"),
            ("calibration", "Choose strength and decision thresholds", "Separate people; no gradient fitting"),
            ("development", "48-hour continue/stop decision", "Held people and development event family"),
            ("final", "One final held-event evaluation", "Reserved people, event, and configured nuisance condition"),
        ], columns=["Role", "Purpose", "Boundary"]))
        """),
        md("""
        ## 4. Read the available backend configuration

        A real run needs a successfully loaded released motion prior and an
        image-motion estimate. The model bridge must state its representation,
        coordinates, frame rate and inverse conversion. The quick synthetic
        fixture has a different purpose and is always labelled demo.

        Missing model weights or body-model assets are practical blockers. They
        must not silently select a smoothing model and call it a pretrained prior.
        """),
        code("""
        config_path = RUN_ROOT / "config.json"
        if config_path.is_file():
            saved_config = json.loads(config_path.read_text())
            display(pd.DataFrame([{"setting": key, "value": str(value)}
                                  for key, value in saved_config.items()]))
        else:
            print("Configuration is available in cfg; the workflow has not written config.json yet.")
        """),
        md("""
        ## Decision before spending GPU time

        Continue when the selected AMASS files, body assets, prior and flow
        backend are available and there are enough distinct people to separate
        fitting from evaluation. First use a small pilot. Measure actual elapsed
        time before scaling to the proposal's planned 128 motion instances.

        Next: [01 · Make controlled pairs](01_make_controlled_pairs.ipynb).
        """),
    ])
    result["01_make_controlled_pairs.ipynb"] = make(
        """
        # 01 · Construct real-event and tracker-error pairs

        **The skeleton can be the same even when its meaning differs.** We build
        matched cases where one video supports the apparent movement and another
        does not. This is the simplest test of whether additional image evidence
        is useful.
        """, [
        md("""
        ## 1. Begin with a known motion and an edited motion

        Let `x` be clean motion. Let `x_event` contain a smooth joint-angle edit,
        converted to joint positions by forward kinematics. This keeps the
        declared bone lengths fixed. The edit is a controlled motion event, not
        a simulated medical diagnosis.

        For a matched pair, the observed skeleton `z`, confidence, timestamps and
        camera metadata are identical. Only the rendered evidence differs:

        | Case | Rendered movement | Correct reference | Observed skeleton |
        | --- | --- | --- | --- |
        | Real event | `x_event` | `x_event` | the same `z` |
        | Tracker error | `x` | `x` | the same `z` |

        A skeleton-only method has no information that separates these two
        cases. That is intentional. The pixels are the additional input.
        """),
        md("""
        ## 2. Build the development data only

        The same generation stage also crosses true event presence with
        independent tracking noise. Event and noise can affect the same joint
        at the same time. Without those mixed cases, a clip-level switch could
        appear to solve the task.

        This call creates train, calibration and development cases. It does not
        generate final-test cases. The stage saves an index and per-case arrays
        so later notebooks use exactly these observations.
        """),
        code("""
        development_roles = ("train", "calibration", "development")
        started = perf_counter()
        cases = workflow.build_pairs(cfg, roles=development_roles)
        print(f"Built/loaded {len(cases):,} cases in {perf_counter() - started:.1f} seconds.")
        display(cases.head(16))
        """),
        code("""
        # Inspect available design columns instead of inferring balance from totals.
        design_columns = [c for c in ("role", "split", "fixture", "event_family", "event_present",
                                       "noise_present", "corruption", "condition") if c in cases]
        if design_columns:
            display(cases.groupby(design_columns, dropna=False).size().rename("cases").to_frame())
        group_column = next((c for c in ("identity", "subject_id", "subject_id_candidate", "person_id")
                             if c in cases), None)
        role_column = next((c for c in ("role", "split") if c in cases), None)
        if group_column and role_column:
            display(cases.groupby(role_column)[group_column].nunique().rename("people_or_groups").to_frame())
        """),
        md("""
        ## 3. Inspect a motion before fitting a model

        The preview compares clean reference, edited reference and observations.
        Inspect the timing, amplitude and affected joints. A smooth edit should
        preserve the skeleton's lengths without creating an obvious boundary
        spike that trivially gives away the event label.

        Real mode renders the SMPL-H triangle mesh; demo mode uses procedural
        articulated tubes. Material-point transport supplies the rendering
        reference. The simple texture, camera and occluder do not simulate
        realistic clothing or clinical pathology. Natural-video transfer remains
        a separate requirement.
        """),
        code("""
        figure = plots.preview_pair(cfg, split="train")
        display(figure)
        plt.close(figure)
        """),
        md("""
        ## 4. Inspect the controls that make the result meaningful

        The important controls are scientific: identical paired skeleton inputs;
        no person split across roles; both event and error in the same clip;
        and an occluded pair where every model input is identical but the hidden
        explanation differs. For that balanced ambiguous pair, a probability near
        one half is appropriate. A confident decision is a failure.

        Duration, centroid drift, foreground area and amplitude summaries should
        not reveal the hidden explanation in the matched pair. Across other
        conditions, summarize these nuisance variables and evaluate simple
        baselines before interpreting a learned advantage.
        """),
        code("""
        control_columns = [c for c in cases if any(word in c.lower()
                           for word in ("pair", "identical", "occlu", "ambig", "camera", "noise", "event"))]
        display(cases[control_columns].head(20) if control_columns else cases.head(20))
        print("Final-test cases were not requested by this notebook.")
        """),
        md("""
        ## Decision

        Continue only if the generated changes are interpretable and the matched
        cases force the method to use measurement evidence. If a trivial scalar
        separates event from error, fix the construction before training.

        Next: [02 · Cache the prior, flow, and baselines](02_prior_flow_and_baselines.ipynb).
        """),
    ])
    result["02_prior_flow_and_baselines.ipynb"] = make(
        """
        # 02 · Ask the frozen prior, then check the video

        **Does the prior erase real motion, and can a cheap flow check already
        fix it?** This notebook caches the evidence shared by all gate fits.
        Prior and optical-flow weights stay frozen.
        """, [
        md("""
        ## 1. What the prior and flow branches contribute

        The motion prior proposes a repaired trajectory. Optical flow estimates
        how visible image locations move between adjacent frames. For projected
        path `q` and flow field `u`, the transport residual is

        `r[t] = q[t+1] - q[t] - u[t](q[t])`.

        Small residual means the path agrees with nearby image movement. It does
        not prove anatomical correctness. A sleeve or shoe can move differently
        from a joint center. Use robust local samples, quality evidence and
        occlusion flags. Pose and flow both come from the same images and are
        not independent sensors.
        """),
        code("""
        # A numerical explanation of transport residuals, not an experiment result.
        displacement = np.array([[4., 0.], [4., 0.]])
        local_flow = np.array([[3.8, 0.2], [-1., 0.]])
        residual = np.linalg.norm(displacement - local_flow, axis=-1)
        display(pd.DataFrame({"candidate_dx": displacement[:, 0],
                              "nearby_flow_dx": local_flow[:, 0],
                              "transport_residual_pixels": residual},
                             index=["agreement example", "disagreement example"]))
        """),
        md("""
        ## 2. Cache predictions on the declared development cases

        The cache records the actual backend and reference origin. An imported
        MoMask or MDM output must correspond to the same case and coordinate
        convention. A demo smoother is not a released pretrained model.

        The representation-only round trip is a separate baseline. If conversion
        itself removes the event, that loss cannot be attributed entirely to
        the pretrained prior. Keep its frame indices and report its error.

        For real flow, start with a published checkpoint or a clearly labelled
        alternate backend. Renderer-derived flow is an oracle diagnostic, never
        an estimated-flow research result. Crop transforms apply to both flow
        endpoints; camera compensation must be consistent on both trajectories.

        Read reference flow error together with its coverage columns. The
        conservative visibility check excludes occluded and unresolved surface
        points. Foreground coverage is the fraction of visible source-body
        pixels scored; all-pixel coverage uses the whole source image. A low
        error on a small visible subset does not establish reliable whole-body
        flow. MoMask inputs must use its trained rate of 20 frames per second.
        """),
        code("""
        started = perf_counter()
        predictions = workflow.cache_predictions(cfg, roles=("train", "calibration", "development"))
        print(f"Cached/loaded {len(predictions):,} rows in {perf_counter() - started:.1f} seconds.")
        display(predictions.head(16))
        """),
        code("""
        backend_columns = [c for c in predictions if any(word in c.lower()
                           for word in ("method", "backend", "prior", "flow", "reference", "origin"))]
        if backend_columns:
            display(predictions[backend_columns].drop_duplicates().head(30))
        else:
            print("Read the cached metadata and configuration to identify the executed backends.")
        """),
        md("""
        ## 3. Compare the inexpensive explanations first

        Required first-pass comparisons are raw motion, the frozen prior,
        smoothing, robust filtering, confidence gating, local flow propagation,
        and a calibrated simple flow gate. The learned adapter should add value
        beyond these. A reference-informed mixture is a diagnostic of what a
        privileged choice can achieve. It is not a certified upper bound after
        kinematic projection.

        The proposal also calls for two-tracker disagreement, MFTIQ and compatible
        HTD-Refine/H-MoRe/robust-prior-update comparisons. Check the implementation
        ledger in the guide: unavailable external results remain missing
        comparisons, not silently replaced or counted as executed baselines.
        """),
        code("""
        figure = plots.preview_pair(cfg, split="train")
        display(figure)
        plt.close(figure)
        """),
        md("""
        ## 4. Measure unmodified-prior event erasure before training

        Compare the raw input, representation-only round trip, unmodified prior
        and inexpensive baselines on development people now. This can reveal
        that conversion causes the loss, that the prior does not erase the
        selected event, or that a simple rule already repairs it.

        These are full-strength, unmatched operating points. The table and plot
        are mechanism diagnostics, not the later calibration-locked primary
        comparison. Inspect achieved noise removal as well as retention.
        """),
        code("""
        baseline_report = workflow.baseline_report(cfg, split="development")
        display(baseline_report["summary"])
        figure = plots.plot_tradeoff(baseline_report)
        figure.axes[0].set_title("Full-strength baselines: unmatched repair quality")
        display(figure)
        plt.close(figure)
        """),
        md("""
        ## 5. Does the proposed gate have room to help?

        A mixture can retain some of the raw-minus-prior residual, but it cannot
        invent a correct movement absent from both candidates. The
        reference-informed mixture diagnostic uses known truth to choose each
        weight before projection. Its post-projection score is not a mathematical
        ceiling. Poor performance is a reason to inspect whether this gate design
        has enough expressive power. If a simple flow rule succeeds equally well,
        the full learned model needs a stronger reason to exist.

        This is the 24-hour mechanism check. Do not tune using the final event
        family. A held-person development result comes after calibration in
        notebooks 03 and 04.

        Next: [03 · Train and calibrate](03_train_and_calibrate.ipynb).
        """),
    ])
    result["03_train_and_calibrate.ipynb"] = make(
        """
        # 03 · Learn what to preserve, then lock the comparison

        **Train a small gate, not a foundation model.** The prior proposes a
        repair; the gate decides how much observed movement to retain at each
        joint and time. Separate calibration people set operating strengths.
        """, [
        md("""
        ## 1. The gate has a limited job

        Let `z` be the observed trajectory and `p` the frozen prior's proposal.
        A weight `g` between zero and one yields

        `y = p + g * (z - p)`.

        The model sees raw and repaired paths, quality inputs and their flow
        agreement. It also predicts whether an apparent event is supported by
        evidence. Where a raw coordinate is missing, it must not pretend the
        missing value is an observation. The declared kinematic projection is
        applied before evaluating the final repaired motion.
        """),
        code("""
        # This plot explains the gate, not learned behavior.
        t = np.linspace(0, 1, 80)
        prior_curve = np.sin(2 * np.pi * t)
        observed_curve = prior_curve + 0.5 * np.exp(-((t - 0.55) / 0.06) ** 2)
        fig, ax = plt.subplots()
        ax.plot(t, observed_curve, label="Observed path", color="#2c6d9e")
        ax.plot(t, prior_curve, label="Prior proposal", color="#ce8737")
        ax.plot(t, prior_curve + .7 * (observed_curve - prior_curve),
                label="Illustrative gate weight 0.7", color="#277a65", linestyle="--")
        ax.set(xlabel="Clip time (illustration)", ylabel="Position (arbitrary units)")
        ax.legend(); plt.show()
        """),
        md("""
        ## 2. Fit using train people only

        Watch the optimization history for a stable fit. A lower training loss
        is not the research outcome. The actual question is preservation at the
        same repair quality on new people and a new event family.

        The first pilot should use the small default model. Increase sample size
        before adding architectural complexity. Compare the full evidence gate
        with a direct coordinate model, simple flow checks and shuffled evidence.
        """),
        code("""
        started = perf_counter()
        history = workflow.train_gate(cfg)
        print(f"Training completed in {perf_counter() - started:.1f} seconds.")
        display(history.tail(12))
        """),
        code("""
        figure = plots.plot_history(history)
        display(figure)
        plt.close(figure)
        """),
        md("""
        ## 3. Choose strength on calibration people, then lock it

        A stronger repair may remove more noise and more true movement. For each
        method, choose the operating strength on calibration people to approach
        the same target noise removal. Lock that choice before looking at
        development or final results.

        Calibrate the event-evidence probability and an abstention threshold on
        these same separate calibration observations. Report both probability
        errors and the fraction of cases receiving a decision. Synthetic
        calibration does not guarantee calibrated probabilities on GAVD.
        """),
        code("""
        operating_points = workflow.calibrate(cfg)
        display(operating_points)
        print("These operating points are selected using calibration data, not test retention.")
        """),
        md("""
        ## 4. Keep the fitted model fixed

        The next notebook evaluates the saved model, normalization and strengths.
        It does not refit them. A full prespecified strength curve is useful for
        understanding the tradeoff, but its best test point cannot replace the
        locked primary comparison.

        To fit three optimization seeds in one run, set `MP_SEEDS=17,23,42`
        before training. The workflow reuses the same people and variants and
        reports seed consistency. Use a new run directory when changing the
        method or data condition. The bootstrap groups by person; frames,
        variants and optimization seeds are not independent people.

        Next: [04 · Measure preservation and repair](04_preservation_and_repair.ipynb).
        """),
    ])
    result["04_preservation_and_repair.ipynb"] = make(
        """
        # 04 · Does the method preserve more motion at the same repair quality?

        **This is the decision notebook.** It compares locked operating points
        and asks whether the gain survives stronger baselines, uncertainty and
        held-out conditions. Default evaluation uses development cases only.
        """, [
        md("""
        ## 1. Choose the evaluation boundary explicitly

        `MP_EVALUATION_SPLIT=development` gives the 48-hour pilot decision.
        `MP_EVALUATION_SPLIT=final` opens the reserved event family and people.
        Final mode builds and caches those cases only after the gate and
        operating points exist. It never calls training or calibration.

        The implemented final setting combines a new event family, camera angle
        and corruption mechanism. It measures combined stress robustness, not
        isolated event generalization. Separate matched nuisance ablations are
        needed to attribute an effect to one of these changes.

        If you change the method after seeing the final result, that result is
        development evidence and a new independent final set is needed.
        """),
        code("""
        SPLIT = os.environ.get("MP_EVALUATION_SPLIT", "development")
        if SPLIT not in {"development", "final"}:
            raise ValueError("MP_EVALUATION_SPLIT must be development or final.")
        print(f"Evaluation role: {SPLIT}")
        if SPLIT == "final":
            workflow.require_fitted(cfg)
            print("Opening the reserved final cases with the saved gate and calibration.")
            final_cases = workflow.build_pairs(cfg, roles=("final",))
            final_cache = workflow.cache_predictions(cfg, roles=("final",))
            print(f"Final cases: {len(final_cases):,}; cached rows: {len(final_cache):,}")
        """),
        md("""
        ## 2. Read the two axes correctly

        For event descriptor `d`, clean motion `x` and edited truth `x_event`,
        event magnitude is `a = d(x_event) - d(x)`. Retention for output `y` is

        `1 - abs(d(y) - d(x_event)) / abs(a)`.

        One means the event descriptor matches its reference. Zero means an error as
        large as the event itself. Negative retention is allowed and must not
        be clipped away. Overshoot and signed errors are reported separately.

        Repair uses squared 3D error at observed joints, measured against the
        correct truth for the case, including `x_event` when an event is present.
        Noise removal of 0.25 means a quarter of that observed-joint squared
        error was removed. The raw and repaired scores use the same mask.

        Missing joints are interpolated before the prior runs. Their separate
        `completion_mse_m2` score measures gap filling. Also inspect
        `observed_mse_m2`, all-joint `mse_m2`, and `missing_fraction`. Missing
        joints cannot inflate the primary repair ratio.
        """),
        code("""
        # Worked metric example only. These numbers are not model results.
        event_magnitude = 0.10
        reference_descriptor = 0.10
        candidate_descriptors = np.array([0.10, 0.07, 0.00, 0.23])
        display(pd.DataFrame({"candidate_descriptor": candidate_descriptors,
                              "retention": 1 - np.abs(candidate_descriptors - reference_descriptor) / event_magnitude,
                              "signed_event_error": candidate_descriptors - reference_descriptor}))
        """),
        md("""
        ## 3. Evaluate the saved operating points

        Inspect achieved noise removal alongside retention. Calibration may not
        transfer perfectly, so two methods can achieve different removal on new
        people despite sharing a calibration target. Do not call a retention
        gain a matched-quality improvement if repair quality is materially worse.
        Both methods must reach the target on event-plus-noise cases, as well
        as in the overall noisy-case average.
        """),
        code("""
        started = perf_counter()
        report = workflow.evaluate(cfg, split=SPLIT)
        print(f"Evaluation completed in {perf_counter() - started:.1f} seconds.")
        display(report["summary"])
        """),
        code("""
        figure = plots.plot_tradeoff(report)
        display(figure)
        plt.close(figure)
        """),
        md("""
        ## 4. Examine the cases behind the average

        Check event-only and event-plus-noise groups separately. The latter is
        the stronger test. Examine reliable and unreliable flow, held cameras,
        missing observations and the observation-equivalent occlusion fixture.
        An aggregate can hide a method that succeeds only when no repair is needed.

        Confidence intervals should resample people with every trial and variant
        kept together. If the manifest supplies only an uncertain group identity,
        label the result accordingly. Repeated optimization seeds do not create
        new independent people.
        """),
        code("""
        scores = report["scores"]
        display(scores.head(20))
        grouping = [c for c in ("method", "fixture", "event_family", "event_present", "noise_present", "condition")
                    if c in scores]
        quantities = [c for c in ("retention", "noise_removal", "descriptor_abs_error", "descriptor_signed_error", "brier", "decided")
                      if c in scores]
        if grouping and quantities:
            display(scores.groupby(grouping, dropna=False)[quantities].mean())
        """),
        md("""
        ## 5. Make the 48-hour decision

        The proposal's provisional development target is at least **15 percentage
        points more retention** than the strongest baseline, while removing at
        least **25% of injected error**, with comparable achieved repair and a
        person-grouped interval supporting improvement. These are practical
        thresholds, not predicted effects or a promise of publication.

        Stop the method claim if simple calibrated flow matches the learned
        frontier, the gain vanishes after nuisance controls, or the prior never
        meaningfully erases true events. Use a poor reference-informed mixture
        diagnostic to reconsider the gate design, without calling it a certified
        upper bound after projection.
        Dropping an unhelpful optional V-JEPA feature branch does not invalidate
        a useful flow-supported adapter.
        """),
        code("""
        decision = report["decision"]
        display(Markdown("```json\\n" + json.dumps(decision, indent=2, default=str) + "\\n```"))
        if cfg.mode == "demo":
            print("Ignore empirical continue/stop interpretations: this run tests pipeline mechanics only.")
        """),
        md("""
        ## What comes after a passing pilot?

        Expand to three optimization seeds and more people, then evaluate the
        reserved family once. A second prior requires a new set of predictions
        for the same cases, matching the original feature convention. Keep the
        learned gate and calibration fixed if claiming transfer across priors.
        Do not call a smoother a second foundation prior.

        Natural AMASS events and GAVD inspection test whether the synthetic
        result transfers. A successful demo or synthetic-only result is not the
        full proposed ICLR contribution.

        Next: [05 · Inspect GAVD stress cases](05_gavd_visual_stress.ipynb).
        """),
    ])
    result["05_gavd_visual_stress.ipynb"] = make(
        """
        # 05 · Inspect movement preservation on GAVD video

        **Does the behavior survive real images?** Use the existing GAVD manifest
        and videos to inspect visible changes and uncertainty. This notebook
        does not treat a pose estimator or motion prior as clinical reference truth.
        """, [
        md("""
        ## 1. Select clips before comparing model errors

        GAVD provides source videos, annotated sequence boundaries and descriptive
        gait labels. Use the manifest to choose a small fixed stress set with
        clear motion, occlusion and poor tracking. Keep a source recording in
        one role. Do not infer participant identity from recording identity.

        The experiment uses full declared clips for offline restoration. It is
        not forecasting. The existing GAVD confirmation cohort must remain
        excluded. Set `MP_GAVD_RESERVATION` to that study's existing
        `config/source-reservation.csv`; without it, the stress stage reports
        that it has not run and does not decode a potentially reserved clip.
        """),
        code("""
        inventory = workflow.inventory(cfg)
        gavd_manifest = inventory["gavd"]
        useful_columns = [c for c in ("sequence_id", "video_id", "first_frame", "last_frame",
                                       "cam_view", "gait_pattern_annotation", "available") if c in gavd_manifest]
        display(gavd_manifest[useful_columns].head(15) if useful_columns else gavd_manifest.head(15))
        """),
        md("""
        ## 2. Run the configured external stress path

        The implemented first pass estimates flow on real RGB frames. It can
        compare supplied observed and repaired trajectories in full-image
        coordinates. This is a measurement stress test; it does not automatically
        apply the trained 3D gate to video. A body-model reconstruction such as
        WHAM is an estimate, not metric 3D truth.

        Optional pose exports contain `source_frames` (zero-based original video
        frame IDs), `joints2d[N,22,2]` in full-frame pixels and
        `coordinate_system="full_frame_pixels"`. Add `repaired_joints2d` when
        available. A 3D WHAM output must first be projected with its actual camera.
        It cannot be interpreted as 2D pixels by relabelling the coordinates.
        """),
        code("""
        started = perf_counter()
        stress = workflow.gavd_stress(cfg)
        print(f"GAVD stress stage completed in {perf_counter() - started:.1f} seconds.")
        display(stress)
        """),
        md("""
        ## 3. Watch the evidence alongside the repaired motion

        For each saved overlay, examine the original image, observed path, prior
        path and final path together. Useful questions are whether the visible
        motion survives, whether a clear tracking slip is reduced, and whether
        the gate reports uncertainty when neither estimate is supported.

        The viewer below embeds the first available saved video. If the stage
        produced a table of external paths instead, open those paths on HAIC.
        """),
        code("""
        video_columns = [c for c in stress if any(word in c.lower() for word in ("overlay", "video_path", "preview", "gallery"))]
        displayed_video = False
        for column in video_columns:
            for value in stress[column].dropna().head(3):
                path = Path(str(value)).expanduser()
                if not path.is_absolute():
                    path = RUN_ROOT / path
                if path.is_file() and path.suffix.lower() in {".mp4", ".webm"}:
                    display(Video(str(path), embed=True, width=720))
                    displayed_video = True
                    break
            if displayed_video:
                break
        if not displayed_video:
            print("No local video preview was returned. Inspect the stress table and stage outputs.")
        """),
        md("""
        ## 4. Record observations without inventing ground truth

        A useful review records visible evidence, an unresolved ambiguity and
        the affected time range. Avoid labels such as “clinically corrected.”
        Lower model disagreement or smoother motion does not establish that a
        person's real movement was preserved.

        Flow agreement, missingness and abstention can be measured directly as
        diagnostics. They do not replace known 3D reference motion. Binary
        normal-versus-abnormal classification is not the headline and is not
        required by this notebook.
        """),
        code("""
        # A blank review table for manual observations; no judgments are inferred.
        id_column = next((c for c in ("sequence_id", "case_id", "video_id") if c in stress), None)
        identifiers = stress[id_column].astype(str).tolist() if id_column else []
        review = pd.DataFrame({"clip": identifiers, "visible_evidence": "",
                               "time_range": "", "unresolved_ambiguity": ""})
        display(review)
        # After writing observations, save explicitly:
        # review.to_csv(RUN_ROOT / "gavd_manual_observations.csv", index=False)
        """),
        md("""
        ## Final interpretation

        A strong result combines a passing controlled preservation-versus-repair
        test, transfer across reserved events, and credible behavior on natural
        motion. A synthetic gain with visible failures on real video is a
        limitation to understand before expanding the claim.

        Return to the [notebook guide](README.md) for the execution sequence and
        the distinction between implemented experiments and external comparisons.
        """),
    ])
    result["06_diagnose_repair_mechanism.ipynb"] = diagnostic_notebook()
    return result


def diagnostic_notebook():
    notebook = make(
        """
        # 06 · Find out why a repair helps or hurts

        **Does a method remove the injected error, or does it change joints that
        were already correct?** This notebook separates motion reconstruction,
        bone-length projection, and optical-flow evidence using existing caches.
        It explains a result before we spend more time training.
        """, [
        md("""
        ## 1. Choose the existing cases and calculate the diagnostics once

        Use this notebook after notebook 02 has cached the prior and flow.
        It reads existing predictions and rendered scenes on the CPU. It does
        not load pretrained weights, train a gate, choose calibration settings,
        or open the final event family.

        By default, inspect calibration and development people separately.
        `MP_DIAGNOSTIC_ROLES` can restrict those roles; the final role is rejected.
        `MP_DIAGNOSTIC_OUTPUT_DIR` can change the report folder, and
        `MP_DIAGNOSTIC_TRACE_CASES` sets the number of example traces, initially six.
        Missing cached evidence remains missing rather than being regenerated.
        """),
        code("""
        from gavd6_sjepa.research_directions.motion_preservation import diagnostics, diagnostic_plots

        roles = tuple(value.strip() for value in
                      os.environ.get("MP_DIAGNOSTIC_ROLES", "calibration,development").split(",")
                      if value.strip())
        output_dir = os.environ.get("MP_DIAGNOSTIC_OUTPUT_DIR") or None
        trace_count = int(os.environ.get("MP_DIAGNOSTIC_TRACE_CASES", "6"))
        started = perf_counter()
        report = diagnostics.run_diagnostics(cfg, roles=roles, output_dir=output_dir,
                                             max_trace_cases=trace_count)
        print(f"Diagnostics completed in {perf_counter() - started:.1f} seconds.")
        print(f"Saved tables and notes: {report['output_dir']}")
        FIGURE_DIR = Path(report["output_dir"]) / "figures"
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)

        def show_figure(figure, name):
            figure.savefig(FIGURE_DIR / f"{name}.svg", format="svg", bbox_inches="tight")
            display(figure)
            plt.close(figure)

        print(f"Vector figures: {FIGURE_DIR}")
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(report["inventory"])
        display(report["notes"])
        """),
        md("""
        ## 2. Separate reconstruction error from loss of the event

        Start with the difficult **event plus tracking error** cases. Every method
        receives the same case, and summaries average within a person before
        averaging people. Generated variants do not create new independent people.

        - `observed_mse_m2` is squared 3D error at observed joints. Lower is better.
          The plot converts square meters to square centimeters for readability.
        - `noise_removal` is the reduction in that error relative to the raw input.
          Zero means unchanged error; a negative value means the method made it worse.
        - `retention` measures fidelity to the edited movement descriptor. One is
          exact. It is not clipped, so a very poor reconstruction can score below zero.
        - `signed_event_error` keeps the direction that retention discards. Zero is
          exact, minus one returns to the unedited descriptor, and positive values
          go beyond the edited descriptor in the direction of the event.
        - `affected_mse_m2` concerns observed positions that originally had tracker
          error. `unaffected_mse_m2` concerns observed positions that were already
          correct. These regions are defined by corruption, not by event location.

        Compare `raw` with `projected_raw` to isolate bone-length projection.
        Compare the conversion-only `bridge` with the `prior` to examine the
        learned reconstruction. Each has projected and unprojected versions.
        Oracle mixtures and reference-length controls use hidden truth and are
        diagnostic aids, not eligible practical competitors.
        """),
        code("""
        summary = report["summary"]
        difficult = summary.loc[summary.fixture.eq("factorial")
                                & summary.event_present.astype(bool)
                                & summary.noise_present.astype(bool)]
        score_columns = [column for column in (
            "role", "method", "privileged", "n_people", "n_cases",
            "observed_mse_m2", "noise_removal", "retention", "signed_event_error",
            "descriptor_signed_error", "affected_mse_m2", "unaffected_mse_m2",
            "completion_mse_m2") if column in difficult]
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(difficult[score_columns])
        figure = diagnostic_plots.plot_repair(report)
        show_figure(figure, "repair_and_preservation")
        """),
        md("""
        Also inspect clips with no event and no injected error. A method that
        damages these inputs cannot attribute every error to a difficult event.
        Missing-joint completion is a separate column because imputed positions
        were not observed tracker measurements. Its value is undefined when a
        case contains no missing joints.
        """),
        code("""
        clean_controls = summary.loc[summary.fixture.eq("factorial")
                                     & ~summary.event_present.astype(bool)
                                     & ~summary.noise_present.astype(bool)]
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(clean_controls[score_columns])
        """),
        md("""
        ## 3. Check what zero repair strength actually does

        A strength of zero should return the cached raw input, including any
        previously imputed gaps. However, the original candidate mixing step
        still projected the skeleton onto fixed bone lengths at strength zero.
        That operation can move joints even when no prior repair is accepted.
        The original step also substituted the candidate at missing joints
        regardless of strength. The `legacy_prior_strength0` row retains both
        original behaviors. The new curves below blend all cached positions,
        keeping completion error separate from observed-joint repair.

        This diagnostic shows both paths explicitly:

        ```text
        mixed motion = raw + strength × (candidate - raw)
        projection = none: return the mixed motion
        projection = fixed_lengths: project the mixed motion onto estimated lengths
        ```

        If a projected curve is already poor at zero, investigate projection
        before blaming the learned prior. If an unprojected prior curve degrades
        as strength rises, investigate the candidate reconstruction itself.

        These are descriptive curves on already inspected people. They do not
        replace the saved calibration settings or create a new official result.
        """),
        code("""
        figure = diagnostic_plots.plot_strength_curves(report)
        show_figure(figure, "strength_curves")
        curves = report["strength_curve"]
        zero = curves.loc[curves.fixture.eq("factorial")
                          & curves.event_present.astype(bool)
                          & curves.noise_present.astype(bool) & curves.strength.eq(0)]
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(zero[["role", "method", "projection", "strength",
                          "observed_mse_m2", "noise_removal", "retention"]])
        """),
        md("""
        ## 4. Ask whether the bone lengths explain the change

        A fixed-length projection preserves each bone's direction but changes
        its length. Changes near the pelvis can also move joints farther along
        the same chain. The estimated lengths come from the permitted raw input.
        The reference lengths below are used only to diagnose their accuracy.

        **Bias** is the estimated length minus the reference median. Positive
        bias means the estimate is too long. **Variation** is the reference
        length's standard deviation across frames. SMPL-H with dynamic body
        shape can produce reference joints whose distances vary slightly.

        The plot uses noisy factorial cases and takes the absolute bias before
        averaging, so positive and negative errors cannot cancel. It averages
        cases within each person, then averages people. The saved bone-length
        table retains signed bias for inspecting individual cases. Reference
        variation is summarized on the same cohort.

        Large bias suggests inaccurate length estimates. Appreciable reference
        variation means that even an accurate fixed median cannot reproduce
        every reference frame. Compare the reference-length controls in step 2
        before concluding which mechanism dominates.
        """),
        code("""
        figure = diagnostic_plots.plot_bones(report)
        show_figure(figure, "bone_lengths")
        """),
        md("""
        ## 5. Check whether the video evidence separates the paired explanations

        A skeleton-matched pair has the same tracker output but two different
        videos: one supports the event and one does not. The useful question is
        whether flow changes which candidate it supports between these videos.

        The transport gap is the prior path's disagreement with flow minus the
        raw path's disagreement. Positive gaps favor the raw path; negative gaps
        favor the prior. A useful change would favor the raw path more in the
        real-event video than in the tracking-failure video.
        The strongest pattern has a positive real-event gap and a negative
        tracking-failure gap. A change between two negative gaps does not mean
        that both explanations favor the appropriate path.

        Read every gap with its **coverage**, the fraction of relevant event
        positions that have usable flow evidence. The two videos can have
        different supported positions, so a difference between their means is
        not automatically a comparison of identical pixels. Missing support is
        not evidence of zero error. The pair table records missing or incomplete
        evidence explicitly.

        Projected event size, in pixels, provides context: a meaningful 3D change
        can be difficult to see at the rendered resolution. Renderer-reference
        EPE measures estimated flow error only where reference transport is valid;
        it also needs its own foreground-coverage denominator.
        The plot uses the maximum clean-to-edited separation, not its typical
        size, temporal velocity, or a guarantee of visibility through occlusion.
        """),
        code("""
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            display(report["flow_summary"])
            display(report["flow_pairs"])
            flow_cases = report["flow_cases"]
            flow_columns = [column for column in (
                "role", "case_id", "event_present", "event_flow_coverage",
                "event_displacement_max_px", "event_projected_in_frame_fraction",
                "flow_reference_epe", "flow_reference_coverage_foreground") if column in flow_cases]
            display(flow_cases.loc[flow_cases.fixture.eq("matched"), flow_columns])
        figure = diagnostic_plots.plot_flow(report)
        show_figure(figure, "paired_flow_evidence")
        """),
        md("""
        ## 6. Use signed traces to distinguish erasure from overshoot

        A poor retention score alone does not say whether a method erased an
        event or exaggerated it. The left-foot trace and its signed vertical
        error expose the direction: positive height error means the output is
        too high at that frame. Foot height is one coordinate, not the complete
        peak-height descriptor or an estimate of clinical foot clearance.

        The lower panels separate originally corrupted observed positions from
        previously accurate ones. A useful repair reduces the first error without
        creating large error in the second. Frames with no positions in a group
        are left blank. The cached raw trace can include interpolated missing
        joints, but these are excluded from the lower panels.

        These examples explain mechanisms. The full tables, not a selected trace,
        establish how common a behavior is. Timing-family events also require
        their descriptor table because a foot-height trace cannot measure arm-leg
        or trunk-pelvis timing by itself.
        """),
        code("""
        for number, trace in enumerate(report["traces"], 1):
            figure = diagnostic_plots.plot_trace(trace)
            show_figure(figure, f"trace_{number:02d}")
        if not report["traces"]:
            print("No trace cases were requested or available; use the full diagnostic tables.")
        """),
        md("""
        ## 7. Choose the next experiment from the evidence

        | What the diagnostics show | What to investigate next |
        | --- | --- |
        | Projection hurts raw or reference motion | Length estimation and the fixed-length assumption |
        | Conversion-only reconstruction already loses detail | The representation and inverse transform |
        | The unprojected prior damages previously accurate joints | Whether this prior is a suitable repair candidate |
        | Signed error shows overshoot | Candidate bias rather than only an event-erasure explanation |
        | Flow support is sparse or the event is barely visible | Observation resolution and evidence quality |
        | Flow does not distinguish the matched videos despite adequate support | The transport evidence or the paired construction |
        | Repair is useful and flow is discriminative | A focused adapter experiment on development people |

        Record the explanation supported by several cases and the comparisons
        that could disprove it. Do not revise the meaning of a metric to rescue
        a preferred hypothesis. Any later method change needs a new development
        comparison before an untouched final evaluation.

        The generated CSV files, notes, and SVG figures stay in the diagnostic output folder.
        Original predictions, trained models, and calibration files remain intact.
        """),
    ])
    notebook.cells[1] = md("""
        This is a CPU diagnostic of **existing caches**, usually run after 02
        or after inspecting the pilot result in 04. Run it in a fresh kernel.
        Calibration and development remain separate; final data stays closed.

        Set `MP_RUN_ROOT` to the existing run and use its saved configuration.
        The [launch guide](../../slurm/motion-preservation/README.md) describes
        the HAIC launcher. No raw-data or checkpoint download is needed.

        [Proposal](../../docs/studies/motion-preservation/protocol/proposal.md)
        · [Notebook guide](README.md)
    """)
    return notebook


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for name, notebook in render_all().items():
        nbf.write(notebook, DESTINATION / name)
        print(DESTINATION / name)


if __name__ == "__main__":
    main()
