"""Authoring source for the timestamp-aware, past-only prediction tutorial."""
from textwrap import dedent, indent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    def md(value):
        return new_markdown_cell(dedent(value).strip())

    def code(value):
        return new_code_cell(dedent(value).strip(), execution_count=None, outputs=[])

    def tracked_code(step, title, value):
        body = indent(dedent(value).strip(), "    ")
        return new_code_cell(
            f"with tutorial_progress.unit({step}, {title!r}):\n{body}",
            execution_count=None,
            outputs=[],
        )

    return new_notebook(cells=[
        md(r"""
        # 10 — Predicting future movement using only past observations

        The completed laterality experiment asks whether features describe a
        whole movement clip and respond consistently to reflection. A temporal
        world model needs an additional capability: useful prediction of what
        follows an observed segment. This notebook implements a small experiment
        that makes that distinction testable, beginning with the data preparation.

        We train a compact JEPA to predict the representation of a future pose
        window from an observed prefix. We then freeze its context encoder and
        fit a simple predictor of future landmark positions. Comparing this
        predictor with an initial encoder, direct pose inputs, and motion baselines
        tests whether future-feature pretraining makes the context more useful.

        The default run uses generated oscillating landmarks. It is a working
        example and software check, not a gait result. A separately enabled local
        run reads the original timed pose archives and retains the existing
        source-video partitions. It creates a new exploratory forecasting task;
        none of its results belongs to the completed protocol in notebooks 00–06.
        """),
        md("""
        ## Step 1 — Define the prediction boundary and the measured outcome

        We observe the first 0.8 seconds of a clip, then ask about movement 0.25,
        0.50, and 0.75 seconds after that boundary. These durations are transparent
        tutorial choices, selected to illustrate several short horizons. They
        have no clinical calibration, and their suitability should be checked
        using training recordings before a larger experiment is specified.

        The self-supervised target is a feature vector for a 0.20-second window
        ending at each horizon. The observable evaluation target is the last
        sampled landmark position in that window. Every future window begins
        strictly after the observation boundary. The predictor receives the
        requested horizon because that is known when a forecast is requested;
        it never receives the future coordinates or future visibility pattern.

        Each clip contributes one observation boundary. Longer clips therefore
        do not automatically contribute more windows. Different clips from one
        source still share their train/test role and their total evaluation weight.
        """),
        code("""
        from pathlib import Path
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from IPython.display import SVG, display
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        get_ipython().run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("svg", "png")

        def locate_suite_root():
            current = Path.cwd().resolve()
            for parent in (current, *current.parents):
                for candidate in (parent, parent / "neurips-laterality"):
                    if (candidate / "config" / "protocol.json").is_file():
                        return candidate
            raise FileNotFoundError("Run within the repository or the laterality folder.")

        SUITE_ROOT = locate_suite_root()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))

        from laterality_extensions.forecasting import (
            ForecastSpec, FutureJEPA, JOINTS, as_tensors,
            load_local_forecasting, motion_baselines, perturb_future,
            prefix_input, prepare_examples, run_forecasting_study,
            score_forecasts, synthetic_records,
        )
        from notebook_progress import NotebookTaskProgress

        torch.set_num_threads(1)
        spec = ForecastSpec()
        RUN_LOCAL_FORECAST_TRAINING = False
        tutorial_progress = NotebookTaskProgress(
            "Past-only forecasting tutorial",
            "stage",
            refresh_seconds=0.5,
        )
        tutorial_progress.start(
            6,
            profile="exploratory",
            note="Default execution uses generated timed motion; local cohort forecasting is opt-in.",
        )
        print("SYNTHETIC TUTORIAL — generated motion, not empirical gait evidence")
        display(pd.DataFrame({
            "setting": ["Observed prefix", "Forecast horizons", "Future target window", "Optimizer updates"],
            "value": [f"{spec.context_seconds:.2f} s", str(spec.horizons) + " s", f"{spec.future_window_seconds:.2f} s", spec.updates],
        }))
        """),
        tracked_code(1, "Render the past/future information boundary", '''
        display(SVG("""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 940 205" role="img" aria-labelledby="forecast-title forecast-desc">
        <title id="forecast-title">Past-only prediction boundary</title>
        <desc id="forecast-desc">Observed poses enter the context encoder. A known time horizon enters the predictor. Future observations enter only the training target and scoring.</desc>
        <defs><marker id="forecast-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#475569"/></marker></defs>
        <style>text{font-family:system-ui,sans-serif;fill:#172b4d}.label{font-size:17px;font-weight:600}.note{font-size:14px}.box{rx:10;stroke-width:1.5}.arrow{stroke:#475569;stroke-width:2;marker-end:url(#forecast-arrow)}</style>
        <rect class="box" x="20" y="38" width="220" height="68" fill="#e7f2fa" stroke="#2874a6"/>
        <text class="label" x="130" y="65" text-anchor="middle">Observed prefix</text><text class="note" x="130" y="89" text-anchor="middle">0 to 0.8 seconds</text>
        <path class="arrow" d="M242 72H300"/>
        <rect class="box" x="304" y="38" width="245" height="68" fill="#e7f2fa" stroke="#2874a6"/>
        <text class="label" x="426" y="65" text-anchor="middle">Encoder + predictor</text><text class="note" x="426" y="89" text-anchor="middle">past poses + requested horizon</text>
        <path class="arrow" d="M552 72H610"/>
        <rect class="box" x="616" y="38" width="300" height="68" fill="#f3f5f7" stroke="#64748b"/>
        <text class="label" x="766" y="65" text-anchor="middle">Future-feature comparison</text><text class="note" x="766" y="89" text-anchor="middle">prediction and separate teacher target</text>
        <rect class="box" x="616" y="140" width="300" height="48" fill="#fff2e3" stroke="#b76d1a"/>
        <text class="note" x="766" y="170" text-anchor="middle">Future poses: training target and scoring</text>
        <path class="arrow" d="M766 138V111"/>
        <text class="note" x="24" y="159">Information flows from the observed prefix to the prediction.</text>
        <text class="note" x="24" y="182">The comparison with future observations happens afterward.</text>
        </svg>"""))
        '''),
        md("""
        ## Step 2 — Prepare past observations before looking at the future

        The original 64-frame model inputs are unsuitable for this test without
        rebuilding preparation. Their time axis describes position within a clip,
        and whole-clip scaling and gap filling can use later observations. Cutting
        that array in half would not restore a valid past-only information boundary.

        Here we retain the timestamps from frame numbers and recording rate.
        The reference position is the last observed bilateral pelvis in the
        prefix, and scale is the median observed shoulder/hip width in the prefix's
        image plane. That same fixed reference and scale are used for future targets.
        We do not recenter each future frame, which would remove part of the motion
        being predicted. These coordinates have units of estimated body width,
        not meters; their relative depth is also not a calibrated 3D measurement.

        At each requested past time, we use only the latest observed landmark at
        or before that time, with a maximum allowed age of 0.20 seconds. Older or
        unavailable observations remain invalid and are masked from attention.
        Future targets use nearby actual observations within 0.055 seconds, without
        interpolation. This tolerance accommodates the available frame rates;
        the actual observation times are retained, so the requested horizon is a
        nominal time with a stated tolerance. Repeated samples of one low-rate
        observation do not create new temporal resolution.
        """),
        tracked_code(2, "Prepare timed synthetic forecast examples", """
        records = synthetic_records(seed=spec.seed)
        examples = prepare_examples(records, spec, status="SYNTHETIC — NON-EVIDENTIARY")
        display(pd.DataFrame([{
            "generated_clips": len(records),
            "accepted_clips": examples.coverage["accepted_clips"],
            "forecast_examples": examples.coverage["examples"],
            "source_groups": examples.coverage["sources"],
        }]))
        print("An example is one clip at one requested horizon; examples are not independent people.")
        """),
        md("""
        ## Step 3 — Try to break the information boundary

        A useful leakage test changes every coordinate and visibility indicator
        after the boundary, then rebuilds the input and repeats the prediction.
        The context should be identical because none of those changes was
        available at prediction time. A future-aware normalization, a hidden
        interpolation endpoint, or a future visibility mask passed to the
        predictor would invalidate that guarantee.

        The checks below verify both preparation and the model interface.
        The future teacher may use observed future positions when constructing
        the training target. It is a separate forward pass, with no attention
        connection back to the context. Target visibility can control the loss
        and scoring, but it is absent from the predictor's arguments.
        """),
        tracked_code(3, "Test the past-only information boundary", """
        original_context = prefix_input(records[0], spec)
        changed_context = prefix_input(perturb_future(records[0], spec), spec)
        for field in original_context:
            np.testing.assert_array_equal(original_context[field], changed_context[field])

        torch.manual_seed(spec.seed)
        boundary_check_model = FutureJEPA(spec.embed_dim).eval()
        def predict_from_context(context):
            with torch.no_grad():
                return boundary_check_model.predict(
                    torch.tensor(context["xyz"])[None],
                    torch.tensor(context["valid"])[None],
                    torch.tensor(context["times"])[None],
                    torch.tensor([spec.horizons[0]], dtype=torch.float32),
                )
        torch.testing.assert_close(
            predict_from_context(original_context), predict_from_context(changed_context),
            rtol=0, atol=0,
        )
        assert (examples.past_times[examples.past_valid] <= 0).all()
        assert (examples.future_times[examples.future_valid] > 0).all()
        print("PASS: changed future coordinates and visibility leave the context and prediction unchanged.")
        """),
        md("""
        ## Step 4 — Keep source groups separate and establish motion baselines

        The toy sources are reproducible groups of generated clips. We assign
        eight groups to fitting and four to testing before running any model.
        This is a demonstration of source separation, not an estimate of
        population performance. Local-data mode instead inherits one of the
        existing held-out-video folds and never redraws it after exclusions.

        **Persistence** predicts the most recent observed position. **Constant
        velocity** estimates movement from the last two distinct observed times
        and extends that velocity to the requested horizon. If a joint has only
        one recent observation, the velocity baseline falls back to persistence.
        Both baselines use past observations only and require no fitting. On
        nearly static clips they may be difficult to beat, which is a meaningful
        property of the task rather than a reason to remove them.

        Errors are calculated on the same available future joint endpoints for
        every method. We average coordinate error within an example, then give
        every source equal total weight before taking the square root. Smaller
        root-mean-square error (RMSE) is better. Coverage accompanies each score
        because a method should not appear better by being evaluated on fewer
        difficult observations.
        """),
        tracked_code(4, "Create the source split and motion baselines", """
        source_groups = sorted(set(examples.sources))
        training_sources, testing_sources = source_groups[:8], source_groups[8:]
        assert not set(training_sources) & set(testing_sources)
        test_rows = np.flatnonzero(np.isin(examples.sources, testing_sources))
        persistence, velocity = motion_baselines(examples)
        baseline_scores = score_forecasts(
            {"Persistence": persistence, "Constant velocity": velocity}, examples, test_rows,
        )
        display(baseline_scores.drop(columns="status").style.format({
            "horizon_seconds": "{:.2f}", "source_balanced_rmse": "{:.3f}",
        }))
        """),
        md(r"""
        ## Step 5 — Learn future features with a small JEPA

        The encoder projects each landmark's coordinates, validity, and actual
        observation time into tokens, adds a joint embedding, and uses one
        Transformer block. Only valid tokens contribute to the final pooled
        context. Its predictor receives that context and the requested horizon.
        A slowly updated copy of the encoder processes the separate future window.
        The future branch supplies a training target and receives no gradient.

        In the following objective, $p$ is the predicted future feature vector,
        $z_f$ is the teacher's feature vector, and $z_p$ is the context feature:

        $$L = \operatorname{mean}(p-z_f)^2
        + 0.1\,\operatorname{mean}\left[\max\{0,1-\sqrt{\operatorname{Var}(z_p)+10^{-4}}\}\right].$$

        The second term discourages an identical context representation for every
        training example. Its weight, the teacher momentum of 0.99, and the short
        update budget are illustrative settings. They are not reported optima or
        evidence that collapse has been prevented. We inspect actual variation
        and useful held-out prediction afterward.

        The training loop samples a source uniformly, then a clip/horizon from
        that source. A control model uses the same initial weights and source
        draws but receives a future window from a different training source at
        the same requested horizon. It selects an eligible different source
        uniformly before selecting a window, so sources with many clips do not
        dominate the shuffled targets. The eligible alternative-source set can
        vary with horizon coverage and should be reported in a larger study.
        Its separate random stream preserves the
        main sampling sequence. This deliberately mismatched future tests whether
        matching the observed and future movement matters, rather than merely
        training on any future-looking representation.
        """),
        md("""
        ## Step 6 — Evaluate usefulness with a fixed, observable read-out

        A latent loss can fall because future windows share static structure or
        because representations become less informative. We therefore freeze
        the context encoder and fit a regularized linear predictor of future
        coordinates using only training sources. The features include the known
        horizon and its interaction with the context. Feature scaling and each
        joint's regression coefficients are fitted on training observations only.
        The ridge penalty is fixed at 10 for this demonstration, without test-set
        selection. A larger study should choose it in a training-only validation
        design and retain that choice for the outer test.

        We compare the learned encoder with its matched initial encoder, the
        shuffled-future encoder, and a predictor given past coordinates, validity,
        and timestamps directly. The last comparison is deliberately informative:
        it asks whether the learned representation adds value over readily
        available pose information. All fitted read-outs use the same procedure.

        These scores evaluate **context-feature usefulness after future-feature
        pretraining**. The ridge read-out does not decode the JEPA predictor's
        future latent vector. The experiment therefore does not establish an
        accurate latent rollout, a simulator, or an action-conditioned world model.
        """),
        tracked_code(5, "Train and evaluate the forecasting comparison", """
        result = run_forecasting_study(
            examples, spec, train_sources=training_sources, test_sources=testing_sources,
        )
        score_table = result["scores"].pivot(
            index="method", columns="horizon_seconds", values="source_balanced_rmse",
        )
        print("SYNTHETIC RESULTS — RMSE in prefix body-width units; lower is better")
        display(score_table.style.format("{:.3f}"))
        display(pd.DataFrame([result["variation"]]).style.format({
            "mean_across_clip_std": "{:.3f}", "effective_rank": "{:.2f}",
        }))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for name, history in (("Matched future", result["history"]), ("Shuffled future", result["shuffled_history"])):
            axes[0].plot(history["update"], history["prediction_loss"], label=name)
        axes[0].set(xlabel="Optimizer update", ylabel="Training feature-prediction loss")
        axes[0].legend(frameon=False)
        for name in ("Persistence", "Past coordinates + ridge", "Future JEPA + ridge", "Initial encoder + ridge"):
            rows = result["scores"].query("method == @name")
            axes[1].plot(rows.horizon_seconds, rows.source_balanced_rmse, marker="o", label=name)
        axes[1].set(xlabel="Requested horizon (seconds)", ylabel="Held-out pose RMSE — lower is better")
        axes[1].legend(frameon=False, fontsize=8)
        fig.suptitle("SYNTHETIC TUTORIAL — optimization and useful prediction are separate checks")
        display(fig)
        plt.close(fig)
        """),
        md("""
        ### How to reason about these results

        First inspect the baseline errors at each horizon. Next compare the
        learned encoder with the initial encoder under the same ridge procedure.
        If the learned features do not improve prediction, reduced training loss
        is insufficient evidence of useful pretraining. If matched and shuffled
        futures perform similarly, the run has not established a benefit from
        learning the actual temporal association.

        Feature standard deviation describes how much the learned context varies
        across distinct test clips. Effective rank describes how broadly that
        variation is distributed across feature directions; a constant
        representation has rank zero in this calculation. These are descriptive
        diagnostics, not calibrated acceptance thresholds. A small positive
        variance does not by itself establish informative movement features.

        The example has one toy split and one initialization, so the apparent
        ranking can change. We do not tune the notebook until the JEPA wins.
        A scientific result would require repeated source-held-out fits,
        prespecified choices, and uncertainty calculated at the source level.
        Bootstrap intervals on saved predictions are conditional on those fitted
        models; variability across new training initializations should be shown
        separately. No numerical result above should be used in a paper as a
        result on people or on GAVD.
        """),
        md("""
        ## Step 7 — Run a local exploratory pilot only when requested

        The disabled cell below reads validated raw pose archives, restricts them
        to the accepted laterality cohort, and inherits fold 0's source assignments.
        It does not use the pre-resized model arrays. Some accepted laterality
        clips will be excluded because forecasting needs enough elapsed time and
        visible future observations. Their source roles do not change, and the
        returned coverage record reports those exclusions.

        Future visibility influences whether an outcome can be scored, which can
        produce a selected evaluation sample even when there is no information
        leakage. Compare coverage by horizon and training/test role, and investigate
        missingness-related selection before interpreting generalization. A full
        study should also report which landmarks and recording conditions remain
        measurable, without silently imputing target positions.

        The same 40-update budget remains a pilot, not a continuation of the
        earlier 300-epoch experiment or a validated training schedule. Model and
        result objects stay in memory. Save a separately versioned configuration,
        source assignments, model states, and predictions before treating a larger
        run as research evidence. This notebook never overwrites registered models
        or reports and does not authorize distribution of derived recordings.
        """),
        code("""
        if RUN_LOCAL_FORECAST_TRAINING:
            with tutorial_progress.unit(6, "Optionally run the local forecasting pilot"):
                local_examples, local_split = load_local_forecasting(spec, fold=0)
                print("EXPLORATORY LOCAL FORECASTING — new task, one fold and initialization")
                display(local_examples.coverage)
                local_result = run_forecasting_study(
                    local_examples, spec,
                    train_sources=local_split["train_sources"],
                    test_sources=local_split["test_sources"],
                )
                display(local_result["scores"].drop(columns="status").style.format({
                    "horizon_seconds": "{:.2f}", "source_balanced_rmse": "{:.3f}",
                }))
        else:
            print("Local-data forecasting not run. Set RUN_LOCAL_FORECAST_TRAINING = True to request the pilot.")
            tutorial_progress.start_unit(6, "Optionally run the local forecasting pilot")
            tutorial_progress.skip_unit("Local forecasting was not requested")

        tutorial_progress.complete(
            status="Tutorial complete" if RUN_LOCAL_FORECAST_TRAINING
            else "Tutorial complete · optional local forecasting not run"
        )
        """),
        md("""
        ## Step 8 — Turn the pilot into a productive research question

        A strong next comparison would hold this past-only preparation fixed and
        transfer the controlled symmetry objective from notebook 09 into temporal
        pretraining. Evaluate both prediction accuracy and whether reflecting an
        observed prefix gives the reflected prediction. The latter property can
        be built into a method, so its scientific value depends on useful future
        prediction under the same observation conditions.

        Before expanding the architecture, add a past-order control that changes
        movement order without changing the set of visible coordinates, and test
        multiple horizons on the same supported source sample. Distinguish
        landmark noise from future uncertainty. For a probabilistic predictor,
        calibration and meaningful distributional scores would become additional
        requirements; the deterministic RMSE comparison here cannot establish them.

        Latent prediction, transformation-aware features, and temporal world
        modeling already have substantial precedent. [V-JEPA 2](https://arxiv.org/abs/2506.09985)
        studies video understanding, prediction, and planning; [seq-JEPA](https://proceedings.neurips.cc/paper_files/paper/2025/file/2f63d2963526bdd9ff1b8bcc2dc9905a-Paper-Conference.pdf)
        studies invariant/equivariant representations through sequence prediction.
        This notebook is not a reproduction of either model. The possible
        contribution is a controlled account of which geometric constraints
        improve observable movement prediction when pose visibility, real timing,
        and the information boundary are handled explicitly.

        An external subject-indexed study could then test that finding beyond the
        development dataset. Notebook 06 currently checks its prerequisites; an
        actual external evaluation requires a compatible measurement endpoint
        and a separately specified procedure. Predicting subsequent observed
        movement does not identify the effects of treatment, injury, or other
        interventions, and no such conclusion follows from this tutorial.
        """),
    ])
