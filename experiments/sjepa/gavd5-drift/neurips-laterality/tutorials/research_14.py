"""Editable teaching source for the future-feature decoder comparison."""
from textwrap import dedent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    def md(value):
        return new_markdown_cell(dedent(value).strip())

    def code(value):
        return new_code_cell(dedent(value).strip(), execution_count=None, outputs=[])

    return new_notebook(cells=[
        md("""
        # 14 — Do predicted future features describe future movement?

        A useful representation of the observed past and an accurate prediction
        of the future are separate achievements. Notebook 10 fits a regression
        model from past encoder features to future coordinates. Here we also
        examine the feature vector produced by the JEPA predictor itself.

        We first fit a small decoder that translates **observed future features**
        into landmark positions, using training videos only. We then freeze this
        decoder and apply it to **predicted future features** from held-out videos.
        Applying the same decoder to observed future features provides a useful
        diagnostic: if this easier task fails, the chosen features or decoder
        cannot yet express the movement endpoint well enough to judge forecasting.

        This notebook depends on the timing and source-separation ideas in
        Notebook 10 and the encoder-versus-predictor distinction in Notebook 13.
        Its default execution uses a few updates on generated oscillating
        landmarks. These outputs are software demonstrations. The real-data
        comparison is implemented, but requires explicit enablement and has no
        empirical result until it is run and retained.
        """),
        md("""
        ## 1. State what the forecast can observe

        Each example contains the first 0.8 seconds of a clip. A requested horizon
        tells the predictor whether to estimate movement 0.25, 0.50, or 0.75
        seconds after that boundary. During training, a separate teacher encoder
        processes a 0.20-second future window ending at that horizon. Its output
        supplies the feature target; its parameters receive no gradient.

        The observable outcome is the last measured position in the future
        window for twelve specified landmarks. Camera sampling may place this
        measurement slightly before or after the requested time. We retain the
        actual measurement time in the evaluation record, while predictions
        receive only the requested horizon, which is known in advance.

        The encoder here follows Notebook 10's small forecasting architecture:
        each sampled landmark observation is a token. This is a separately
        specified model comparison from the four-step tokens used for masking in
        notebooks 11–13. Its optimizer budget alone cannot make it a reproduction
        of their training recipe.
        """),
        code("""
        from pathlib import Path
        from dataclasses import asdict, replace
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from IPython.display import display
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        set_matplotlib_formats("svg", "png")

        def locate_suite_root():
            for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
                for candidate in (parent, parent / "neurips-laterality"):
                    if (candidate / "config" / "protocol.json").is_file():
                        return candidate
            raise FileNotFoundError("Run within the repository or the laterality folder.")

        SUITE_ROOT = locate_suite_root()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))

        from laterality_extensions.forecasting import JOINTS, TimedPose, synthetic_records
        from laterality_extensions.future_comparison import (
            ALL_JOINTS, ComparisonForecastSpec, ForecastComparisonModel,
            plan_future_comparison, prepare_future_examples, prepare_prefix,
            real_forecast_spec, run_future_comparison, run_real_future_comparison,
        )

        torch.set_num_threads(1)
        synthetic_spec = ComparisonForecastSpec(updates=4)
        print("SYNTHETIC SOFTWARE DEMONSTRATION — no empirical gait result")
        display(pd.Series(asdict(synthetic_spec), name="Demonstration settings").to_frame())
        """),
        md("""
        ## 2. Follow the two evaluation routes

        The lower row in the following diagram uses an observed future window,
        so it has access to the answer period. It tests whether the frozen
        representation and training-only decoder can express the measured
        coordinates. The upper row is the forecast: every value available to
        the context encoder comes from the observed prefix.

        Both routes use the same decoder and the same measured endpoints. A
        large gap between their errors can arise when predicted features lie
        outside the distribution on which the decoder was fitted. A decoder
        applied to such features may extrapolate badly even when its observed-
        future diagnostic is accurate. This is a substantive outcome to report.
        """),
        code(r"""
        fig, ax = plt.subplots(figsize=(11, 3.7), constrained_layout=True)
        ax.set(xlim=(0, 11), ylim=(-0.5, 3.1))
        ax.axis("off")
        boxes = [
            (0.2, 2.0, "Observed past", "#e4eef6"),
            (2.6, 2.0, "Encoder + predictor", "#e4eef6"),
            (5.3, 2.0, "Predicted future features", "#e4eef6"),
            (8.5, 2.0, "Frozen decoder", "#dcecdf"),
            (0.2, 0.2, "Observed future", "#f2eadc"),
            (2.6, 0.2, "Teacher encoder", "#f2eadc"),
            (5.3, 0.2, "Observed future features", "#f2eadc"),
            (8.5, 0.2, "Same decoder", "#dcecdf"),
        ]
        for x, y, label, color in boxes:
            ax.text(x, y, label, ha="left", va="center", fontsize=10,
                    bbox={"boxstyle": "round,pad=0.45", "fc": color, "ec": "#9cabb4"})
        for y in (2.0, 0.2):
            for left, right in ((1.9, 2.35), (4.5, 5.05), (7.7, 8.25)):
                ax.annotate("", (right, y), (left, y), arrowprops={"arrowstyle": "->", "color": "#53616b"})
        ax.text(0.2, 2.65, "Forecast: only past observations are available", fontsize=11)
        ax.text(0.2, 0.85, "Diagnostic: the future is observed", fontsize=11)
        ax.text(8.5, 1.05, "Predict the same\nlandmark positions", fontsize=10, color="#365543")
        display(fig)
        plt.close(fig)
        """),
        md("""
        ## 3. Prepare both input choices with common endpoints

        A separate configuration supplies either twelve gait landmarks or all 33
        landmarks to the encoder. In both cases, the measured future coordinates
        remain the same twelve landmarks: shoulders, hips, knees, ankles, heels,
        and foot tips. The twelve-landmark prefix also determines eligibility,
        the pelvis reference position, and body scale in both configurations.

        This keeps a change in input coverage from silently changing the target
        or accepted clip set. The all-landmark configuration therefore still
        retains anatomical choices in its normalization and endpoint. Its joint
        identity table has the same parameter shape as the twelve-input model,
        allowing initial weights to be paired while the visible input changes.

        The sampler keeps the most recent valid observation at or before each
        past query time, provided it is sufficiently recent. It records that
        observation's actual time. No interpolation between a past observation
        and a future observation enters the context.
        """),
        code("""
        records = synthetic_records(sources=8, clips_per_source=1)
        examples = prepare_future_examples(records, synthetic_spec, status="SYNTHETIC DEMONSTRATION")
        all_input_examples = prepare_future_examples(
            records, replace(synthetic_spec, input_joints=ALL_JOINTS),
            status="SYNTHETIC DEMONSTRATION",
        )
        for field in ("endpoint", "endpoint_valid", "endpoint_times", "sequence_ids", "horizon"):
            np.testing.assert_array_equal(getattr(examples, field), getattr(all_input_examples, field))
        display(pd.DataFrame([
            {"Input landmarks": len(data.input_joints), "Future endpoints": data.endpoint.shape[1],
             "Clips": data.coverage["accepted_clips"], "Videos": data.coverage["sources"],
             "Clip–horizon examples": data.coverage["examples"]}
            for data in (examples, all_input_examples)
        ]))
        """),
        md("""
        These counts describe generated examples. Matching them verifies the
        input comparison's bookkeeping; it does not show that adding landmarks
        improves forecasting. In real recordings, unavailable future endpoints
        affect evaluation coverage even when the context remains valid, so their
        exclusions must accompany the prediction scores.

        ## 4. Check the information boundary directly

        We now replace every coordinate after the observation boundary, remove
        its visibility, and move its timestamp further into the future. Since all
        changed timestamps remain after the boundary, the prepared prefix and
        deterministic forecast should remain unchanged. Moving a measurement
        across the boundary would change the observations available and would
        answer a different question.
        """),
        code("""
        record = records[0]
        later = record.times - record.times[0] > synthetic_spec.context_seconds + 1e-10
        changed_xyz, changed_valid, changed_times = record.xyz.copy(), record.valid.copy(), record.times.copy()
        changed_xyz[later], changed_valid[later] = 1e6, False
        changed_times[later] += 0.4
        changed = TimedPose(record.source, record.sequence_id, changed_times, changed_xyz, changed_valid)
        original_prefix = prepare_prefix(record, synthetic_spec)
        changed_prefix = prepare_prefix(changed, synthetic_spec)
        for field in original_prefix:
            np.testing.assert_array_equal(original_prefix[field], changed_prefix[field])

        torch.manual_seed(synthetic_spec.seed)
        initial_model = ForecastComparisonModel(synthetic_spec).eval()
        def predict_from_prefix(prefix):
            with torch.no_grad():
                return initial_model.predict(
                    torch.tensor(prefix["xyz"])[None], torch.tensor(prefix["valid"])[None],
                    torch.tensor(prefix["times"])[None], torch.tensor([0.25]),
                )
        torch.testing.assert_close(predict_from_prefix(original_prefix), predict_from_prefix(changed_prefix), rtol=0, atol=0)
        print("Changed future coordinates, visibility, and later timestamps leave context and prediction unchanged.")
        """),
        md("""
        ## 5. Fit matched and mismatched future targets

        The small model encodes the prefix, appends the requested horizon, and
        predicts the teacher's future feature vector. Its loss combines mean
        squared feature error with a penalty on very low context variation.
        The variance term encourages nonconstant features; measured variation
        and movement prediction must still be checked afterward.

        The control receives a future window from a different training video at
        the same horizon. It samples an eligible video uniformly before sampling
        a clip from that video. A separate random stream for this mismatch keeps
        the two arms' initial weights, context draws, and update counts paired.
        This control asks whether the correct temporal pairing helps beyond the
        exposure to generic future windows.

        The online encoder, teacher, and predictor are retained separately at
        the final, prespecified update. Teacher features are evaluated with their
        own frozen readout; the forecasting predictor always receives online
        context features through its normal pathway.
        """),
        code("""
        source_names = sorted(set(examples.sources))
        training_sources, testing_sources = source_names[:6], source_names[6:]
        assert not set(training_sources) & set(testing_sources)
        comparison = run_future_comparison(
            examples, synthetic_spec,
            train_sources=training_sources, test_sources=testing_sources,
        )
        assert comparison["matched"]["source_schedule"] == comparison["mismatched"]["source_schedule"]
        display(pd.DataFrame([
            {"Training target": label, "Updates": len(comparison[key]["history"]),
             "Training videos": len(comparison[key]["train_sources"]),
             "Training time (seconds)": comparison[key]["runtime_seconds"]}
            for key, label in (("matched", "Matching future"), ("mismatched", "Another video's future"))
        ]).style.format({"Training time (seconds)": "{:.2f}"}))
        """),
        md("""
        ## 6. Fit decoders within the training videos

        Each coordinate decoder uses a linear regression with a penalty that
        limits large coefficients. The training videos are divided into inner
        source groups to choose that penalty. Scaling and regression fitting use
        only the inner training group for each validation split; the final
        decoder is then fitted on all outer training videos. Each video receives
        equal total weight, regardless of its number of clips.

        This inner procedure selects a supervised readout for an already fixed
        encoder. It does not validate the entire pretraining recipe: the encoder
        was pretrained on all outer training videos, including videos later used
        for readout validation. Comparing pretraining recipes through inner
        validation would also require excluding those inner validation videos
        from candidate encoder training.

        The future-feature decoder sees observed future-teacher features and
        future coordinates from training videos. After fitting, the same
        coefficients translate both observed and predicted future features from
        test videos. The requested horizon and its interaction with features are
        supplied consistently to all fitted readouts. Actual future measurement
        times and visibility are used for scoring, without entering the forecast.
        """),
        code("""
        decoder = comparison["decoders"]["Matched future-feature decoder"]
        assert set(decoder.fitted_sources) == set(training_sources)
        assert not set(decoder.fitted_sources) & set(testing_sources)
        display(pd.DataFrame({
            "Landmark index": JOINTS,
            "Ridge penalty selected within training videos": decoder.selected_alphas,
        }))
        """),
        md("""
        ## 7. Read the result by the question it answers

        The first table compares the two future-feature routes. The observed-
        future diagnostic has access to future information; its error describes
        how well the learned features and decoder express the endpoint. The
        predicted-future row is the actual forecast. The mismatched-future rows
        test whether correct temporal pairing mattered in the same setup.

        Errors are in the body-width scale estimated from the prefix, using the
        pose model's coordinate channels. They are not distances in millimeters
        or validated anatomical measurements. For each requested horizon, we
        average coordinate error over common observed landmarks in each clip,
        give every test video equal total weight, and then take the square root.
        """),
        code("""
        scores = comparison["scores"]
        decoded_rows = scores[scores.method.str.contains("future features decoded", regex=False)]
        print("SYNTHETIC RESULTS — source-balanced coordinate RMSE; lower is better")
        display(decoded_rows.pivot(index="method", columns="horizon_seconds", values="source_balanced_rmse").style.format("{:.3f}"))
        """),
        md("""
        A large forecast error is possible after this deliberately short run.
        The test requires the pathways to be correct and executable, without
        requiring the JEPA to beat a baseline. An accurate observed-future
        diagnostic together with poor decoded forecasts would locate a remaining
        difficulty in the predicted feature distribution or its alignment with
        the decoder. It would not establish which training change resolves it.

        The next table evaluates other routes from the same past observations.
        Regression from the trained past encoder answers Notebook 10's original
        question. Direct-coordinate regression checks whether an encoder adds
        useful information processing beyond the available pose data. The initial
        encoder controls for architectural features that exist before pretraining.
        Last-position and recent-velocity forecasts provide simple movement
        references, using the requested horizon and past observations only.
        """),
        code("""
        past_rows = scores[~scores.method.str.contains("future features decoded", regex=False)]
        display(past_rows.pivot(index="method", columns="horizon_seconds", values="source_balanced_rmse").style.format("{:.3f}"))
        display(comparison["prediction_coverage"])
        timing = scores[["horizon_seconds", "actual_time_min", "actual_time_max", "test_sources", "test_clips", "common_endpoints"]].drop_duplicates()
        display(timing.style.format({"horizon_seconds": "{:.2f}", "actual_time_min": "{:.3f}", "actual_time_max": "{:.3f}"}))
        """),
        md("""
        Every compared method is scored on the same available endpoints. The
        coverage table also reports each method's unavailable predictions and
        examples with no common endpoint, so the common intersection cannot hide
        a loss of coverage. Any real-data comparison between input choices must
        preserve this endpoint agreement and report the same source scope.

        A low teacher-feature error alone cannot rank the methods. Each arm has
        its own learned teacher, and a teacher with less variable features can
        make prediction easier. The normalized error below divides mean squared
        feature error by the mean coordinate-wise variance of that arm's target
        features. This makes the denominator explicit, but does not turn
        different teachers into a common measurement scale.
        """),
        code("""
        display(comparison["feature_diagnostics"].style.format({
            "mean_feature_std": "{:.3g}", "effective_rank": "{:.2f}", "mean_feature_norm": "{:.3g}",
        }))
        display(pd.DataFrame(comparison["latent_diagnostics"]).drop(columns="scope").style.format({
            "feature_mse": "{:.3g}", "target_variance_denominator": "{:.3g}", "normalized_error": "{:.3g}",
        }))
        """),
        md("""
        ## 8. Review a complete real-data workload before training

        The explicit real-data recipe below uses 1,200 updates, a 24-dimensional
        encoder, and a batch size of eight. These are declared starting settings
        for this separate forecasting task. The architecture and future-feature
        objective differ from Notebook 08, even though its update count provides
        a useful scale reference. A change in training budget should be justified
        from training sources and applied consistently to compared methods.

        Five existing outer video groups and five seeds produce 25 paired runs
        for the twelve-input configuration, containing 50 fitted forecasting
        models. Supplying both input sets is a separate, explicit comparison that
        doubles this workload. Decoder fits are additional supervised readouts.
        A smaller selection of folds or seeds is labeled as a pilot or partial
        comparison, with its actual scope retained.

        The first cell only displays the complete plan. The following cell can
        validate raw recordings and existing source assignments without training.
        Setting the training flag enables the requested runs after the plan is
        displayed. Completed compatible outputs are reused; existing incompatible,
        corrupted, or incomplete outputs fail clearly. Interrupted training is
        recorded as incomplete and has no automatic resume path.
        """),
        code("""
        real_spec = real_forecast_spec()
        REAL_FOLDS = (0, 1, 2, 3, 4)
        REAL_SEEDS = (42, 43, 44, 45, 46)
        REAL_INPUT_SETS = (JOINTS,)  # Add ALL_JOINTS only for the declared input comparison.
        REAL_OUTPUT_ROOT = SUITE_ROOT / "outputs" / "masking_extensions" / "future"
        workload = plan_future_comparison(
            real_spec, folds=REAL_FOLDS, seeds=REAL_SEEDS,
            input_sets=REAL_INPUT_SETS, output_root=REAL_OUTPUT_ROOT,
        )
        display(pd.DataFrame(workload["runs"]))
        display(pd.Series(workload["recipe"], name="Real-data recipe").to_frame())
        print(f'{workload["scope"]}: {workload["model_fits"]} forecasting model fits, '
              f'{workload["optimizer_updates"]:,} updates, '
              f'{workload["source_example_draws"]:,} source-balanced example draws.')
        print(workload["recipe_note"])
        """),
        code("""
        VALIDATE_REAL_INPUTS = False
        ENABLE_REAL_TRAINING = False
        if VALIDATE_REAL_INPUTS or ENABLE_REAL_TRAINING:
            real_request = run_real_future_comparison(
                enabled=ENABLE_REAL_TRAINING, validate_inputs=VALIDATE_REAL_INPUTS,
                spec=real_spec, folds=REAL_FOLDS, seeds=REAL_SEEDS,
                input_sets=REAL_INPUT_SETS, output_root=REAL_OUTPUT_ROOT,
            )
            print(real_request["status"])
        else:
            print("Real-data plan displayed. Input loading and training remain disabled.")
        """),
        md("""
        ## 9. What would support useful future-feature learning?

        A persuasive result would combine an informative observed-future decoder
        diagnostic with accurate decoded predicted features on excluded videos.
        The predicted-feature route should be compared with direct past-coordinate
        regression, initial features, simple motion references, and the same
        architecture trained against mismatched futures. If correct temporal
        pairing has no clear benefit, the experiment has not established that
        the learned future association improves the observable forecast.

        Repeated seeds reuse the same recordings. For a full comparison, combine
        each seed's out-of-fold predictions before reporting its source-balanced
        score, and summarize seed variability separately. Paired intervals should
        resample whole source videos with all their clip, horizon, condition, and
        seed predictions together; they describe uncertainty conditional on the
        fitted models, rather than the uncertainty of retraining everything.

        Future-target learning has substantial prior work. The August 2026
        [Human-JEPA preprint](https://arxiv.org/html/2608.21160v1) studies strictly
        past-to-future masks in human video and evaluates the role of the predictor
        head. Our small pose model is a separate implementation; it does not
        reproduce that model's pretraining, scale, or reported results. The useful
        research question here is whether a carefully controlled future-feature
        pathway improves measured movement prediction under source separation.

        The generated example establishes that the decoder comparison, matched
        controls, and information-boundary checks can run. Real gait forecasting
        remains unmeasured by this notebook until its separate experiment is
        executed. Multiple horizons from one fixed prefix are independent
        forecasts; a recursive rollout would require a separately implemented
        and tested transition from one predicted state to the next.
        """),
    ])
