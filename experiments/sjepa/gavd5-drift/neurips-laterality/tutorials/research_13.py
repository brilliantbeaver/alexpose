"""Authoring source for evaluating comparative masking encoders and predictors."""

from textwrap import dedent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    def md(text):
        return new_markdown_cell(dedent(text).strip())

    def code(text):
        return new_code_cell(dedent(text).strip(), outputs=[], execution_count=None)

    return new_notebook(cells=[
        md("""
        # 13 — Do the learned features preserve useful movement differences?

        A masking task can become easier while a useful movement difference
        becomes harder to predict. This notebook makes that possibility visible
        by evaluating two connected operations: predicting hidden teacher
        features, and predicting an observable movement score from a frozen
        encoder. Notebook 14 separately examines observable future movement.

        We first check the evaluation with a signal whose meaning we know, then
        train the two small masking examples introduced in Notebook 12. Every
        displayed score is calculated from generated data in this execution.
        These software demonstrations do not add empirical findings about GAVD.
        Three optimizer updates are enough to exercise the pathway, but provide
        no assessment of a fully trained masking method.

        You can run this notebook independently in a fresh kernel. It imports
        the helpers introduced in Notebooks 11 and 12 and prepares its own
        synthetic examples. The final section shows how the same evaluation
        applies to deliberately enabled, retained real-data experiments.
        """),
        md("""
        ## 1. Keep the prediction question visible

        | Question | Input to the predictor | Measured outcome |
        |---|---|---|
        | Can the JEPA complete its training task? | Online encoder features with selected observations hidden | Features supplied by that model's teacher |
        | Do frozen features help describe movement? | The same five bilateral feature summaries for every encoder | The original recording's coordinate-derived laterality score |
        | Do predicted future features express future movement? | A JEPA prediction based on a past-only prefix | Future landmark coordinates, studied in Notebook 14 |

        The first outcome changes when a model's teacher changes. The second
        gives every method the same target and therefore supports the main
        comparison of usefulness. For example, two teachers could differ in
        how much their features vary; predicting a nearly constant teacher
        could require little movement information.

        The original score describes a signed left–right movement difference
        computed from estimated coordinates. Its predictability does not by
        itself establish neurological interpretation or diagnostic usefulness.
        """),
        code("""
        from pathlib import Path
        import copy
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from IPython.display import display
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        get_ipython().run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("svg", "png")

        SUITE_ROOT = next(
            candidate
            for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents)
            for candidate in (parent, parent / "neurips-laterality")
            if (candidate / "laterality" / "model.py").is_file()
        )
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))

        from laterality.metrics import source_weights, weighted_r2
        from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
        from laterality_extensions.comparative_training import default_conditions, train_comparison
        from laterality_extensions.comparative_evaluation import (
            aggregate_predictions, evaluate_frozen_representations, feature_diagnostics,
            fit_source_readout, make_evaluation_mask_bank, paired_source_bootstrap,
            predictor_diagnostics, prepare_raw_missing_observations,
            prepared_observation_sensitivity,
        )

        torch.set_num_threads(1)
        print("Synthetic teaching examples; no real-data training is enabled.")
        """),
        md("""
        ## 2. Check that the readout can recover a known signal

        Suppose two measured feature channels describe left and right movement,
        and the target is twice the first channel minus half the second. We
        generate this relation directly, so a linear readout should recover it.
        Two additional channels contain irrelevant variation; some values in
        one channel are missing. This lets us exercise training-only imputation
        as well as regression.

        Six generated sources supply training clips and two supply test clips.
        Within the training sources, we choose the ridge penalty by repeatedly
        holding out whole sources. Each inner split fits its own imputation
        and scaling. We pool the inner predictions before comparing penalties,
        refit the chosen readout on the outer training sources, and evaluate
        the remaining sources once. A failure here would raise a software or
        evaluation concern before we interpreted any weak encoder result.
        """),
        code("""
        rng = np.random.default_rng(81)
        signal_sources = np.repeat(np.array(list("abcdefgh")), 5)
        signal_features = rng.normal(size=(40, 4))
        signal_target = 2 * signal_features[:, 0] - 0.5 * signal_features[:, 1]
        signal_features[::3, 3] = np.nan
        signal_readout = fit_source_readout(
            signal_features, signal_target, signal_sources,
            train_sources=tuple("abcdef"), test_sources=tuple("gh"),
            alphas=(0.001, 0.1, 10), inner_folds=3,
        )
        signal_test = np.isin(signal_sources, tuple("gh"))
        recovered_signal = signal_readout.predict(signal_features[signal_test])
        signal_r2 = weighted_r2(
            signal_target[signal_test], recovered_signal,
            source_weights(signal_sources[signal_test]),
        )
        display(pd.DataFrame([{
            "Example": "Known linear movement signal",
            "Test clips": int(signal_test.sum()), "Test sources": 2,
            "R²": signal_r2, "Selected ridge penalty": signal_readout.selected_alpha,
        }]).round(3))
        assert np.mean((recovered_signal - signal_target[signal_test]) ** 2) < 0.01
        """),
        md("""
        This constructed result establishes that the readout procedure can
        recover a signal expressed by its inputs. It does not establish that
        a real encoder preserves that signal. We use the same source-separated
        procedure below, including for direct pose summaries and the initial
        encoder, so a method receives no special tuning advantage.

        The validation here selects a supervised readout after pretraining.
        Choosing an entire pretraining recipe would require an additional
        source boundary: inner validation videos must also be excluded from
        training each candidate encoder. Selecting a mask using outer-test
        scores would defeat the comparison's intended separation.
        """),
        md("""
        ## 3. Prepare two matched encoders and declared checkpoints

        We use 33 input landmarks and compare scattered targets from the twelve
        gait landmarks with scattered targets from all landmarks. Both models
        retain the twelve-landmark regularizer and five-pair laterality summary.
        This changes the eligible prediction targets while preserving the
        other anatomical choices.

        The three-update teaching run stores states after updates 1 and 3.
        Update 3 is the declared evaluation checkpoint; update 1 is retained
        to demonstrate inspection and recovery of a specified earlier state.
        We pair initial weights, sampled source clips, geometric views, and
        hidden-token counts before interpreting the comparison.
        """),
        code("""
        dataset = load_learning_dataset(real=False, fold=0)
        settings = LearningSettings(steps=3, batch_size=4, embed_dim=8, heads=2, seed=7)
        conditions = default_conditions()
        comparison = train_comparison(
            dataset, settings, conditions, checkpoint_steps=(1, 3),
        )
        assert all(comparison["pairing"].values())
        display(pd.DataFrame([
            {"Control": key.replace("_", " "), "Paired": value}
            for key, value in comparison["pairing"].items()
        ]))

        checkpoint_rows = []
        for name, run in comparison["runs"].items():
            for step, state in run["checkpoints"].items():
                checkpoint_rows.append({
                    "Masking condition": name.replace("_", " "), "Update": step,
                    "Online encoder stored": any(key.startswith("view_encoder.") for key in state),
                    "Teacher stored": any(key.startswith("target_encoder.") for key in state),
                })
        display(pd.DataFrame(checkpoint_rows))
        first_run = next(iter(comparison["runs"].values()))
        earlier_model = copy.deepcopy(first_run["model"])
        earlier_model.load_state_dict(first_run["checkpoints"][1], strict=True)
        """),
        md("""
        The online encoder receives gradients. The teacher follows the online
        encoder through an exponential moving average and supplies targets
        without receiving gradients. These two stored states can differ even
        when their architecture is identical. We fit separate frozen readouts
        for them, while preserving the usual online-encoder input to the JEPA
        predictor. Feeding teacher features into that predictor would introduce
        another experimental change.
        """),
        md("""
        ## 4. Give every model the same evaluation gaps

        The bank below contains scattered gaps and one consecutive missing
        region on each leg. A leg region contains the knee, ankle, heel, and
        foot tip, joined through explicit anatomical connections. Both sides
        are tested. The scattered condition matches the left-region hidden
        count for each clip; naturally missing measurements can give the right
        region a different count, which we record.

        The masks are generated once from validity and a separate evaluation
        seed, then reused for every model. A deliberately hidden observation
        remains a valid teacher target during the feature-prediction diagnostic.
        Naturally missing observations cannot supply such a target. Requested
        leg regions keep their complete pattern even when some measurements
        were already missing. We count valid hidden targets separately; a
        corruption that removes no observed token is recorded as infeasible.
        """),
        code("""
        patch_valid = dataset.valid.reshape(len(dataset.xyz), -1, 4, 33).all(axis=2)
        evaluation_masks = make_evaluation_mask_bank(patch_valid, seed=813, interval_length=1)
        coverage_rows = []
        for name, mask in evaluation_masks.items():
            for row in dataset.test_rows:
                coverage_rows.append({
                    "Gap": name.replace("_", " "),
                    "Requested missing positions": int(mask[row].sum()),
                    "Hidden valid tokens": int((mask[row] & patch_valid[row]).sum()),
                    "Remaining valid tokens": int((patch_valid[row] & ~mask[row]).sum()),
                    "Feasible target": bool((mask[row] & patch_valid[row]).any()
                                            and (patch_valid[row] & ~mask[row]).any()),
                })
        display(pd.DataFrame(coverage_rows).groupby("Gap", as_index=False).agg(
            smallest_hidden_count=("Hidden valid tokens", "min"),
            largest_hidden_count=("Hidden valid tokens", "max"),
            smallest_remaining_context=("Remaining valid tokens", "min"),
            feasible_clips=("Feasible target", "sum"),
        ))
        """),
        md("""
        ## 5. Inspect feature prediction together with variation

        The predictor receives hidden online-encoder inputs and predicts the
        matching teacher vectors. We calculate squared error per hidden token,
        average within each clip, then give each source the same total weight.
        The normalized error divides this quantity by the correspondingly
        weighted mean squared teacher-channel value. That denominator controls
        magnitude; it does not make different teachers equivalent outcomes.

        We also compare each prediction with a teacher target from another
        evaluation source, at the same valid landmark and time positions. This
        fixed-seed diagnostic breaks the clip correspondence without fitting
        any new parameters. A similar error for correct and mismatched targets
        would weaken the case that predictions retain clip-specific content.
        The initial model supplies another reference.
        """),
        code("""
        diagnostic_tables = [predictor_diagnostics(
            first_run["initial_model"], dataset, evaluation_masks, condition="Initial model",
        )]
        for name, run in comparison["runs"].items():
            diagnostic_tables.append(predictor_diagnostics(
                run["model"], dataset, evaluation_masks,
                condition=name.replace("_", " "),
            ))
        feature_results = pd.concat(diagnostic_tables, ignore_index=True)
        display(feature_results[[
            "condition", "evaluation_mask", "evaluated_clips", "evaluated_sources",
            "feature_mse", "mismatched_target_mse", "normalized_error",
            "target_clip_mean_channel_sd", "prediction_clip_mean_channel_sd",
        ]].round(3))
        """),
        md("""
        The full table also retains feature norms, covariance effective rank,
        unavailable clips, and near-constant-feature flags. Effective rank
        describes how many feature directions carry appreciable variation.
        It is computed from covariance eigenvalues and has no clinical scale.
        We inspect both individual token vectors and a mean vector per clip,
        because variation across landmark positions can coexist with little
        variation between recordings. Different hidden locations can also
        affect the clip averages, so these remain diagnostic descriptions.

        The following failure case produces no between-clip variation. A
        predictor matching these constant targets could have zero feature
        error while conveying no differences among the clips. A lower feature
        loss alone therefore cannot rank the usefulness of independently
        learned teachers. A shared frozen teacher would support a separately
        trained comparison with a common target space.
        """),
        code("""
        constant_features = np.ones((8, 4))
        constant_check = feature_diagnostics(constant_features, np.array(list("abcdefgh")))
        display(pd.DataFrame([constant_check]).round(3))
        assert constant_check["near_constant"]
        """),
        md("""
        ## 6. Predict the same movement score from each frozen representation

        We fit the online, teacher, and initial-encoder readouts separately.
        Direct pose summaries give a reference using input measurements without
        a learned encoder. The mean target from training sources gives a
        prediction with no clip-specific information. Every fitted readout
        uses the same inner source splits and ridge-penalty candidates.

        To test prepared-coordinate sensitivity, we also remove the fixed bank's
        observations from already prepared test inputs. The readouts remain
        those fitted on unaltered training inputs, and the target remains the
        original recording's movement score. This asks whether missing features
        disrupt an existing predictor; the following section separately checks
        removal before input preparation.
        """),
        code("""
        sensitivity_inputs = {
            name + "_prepared_sensitivity": prepared_observation_sensitivity(dataset, mask)
            for name, mask in evaluation_masks.items()
        }
        evaluated = {}
        for name, run in comparison["runs"].items():
            evaluated[name] = evaluate_frozen_representations(
                run["model"], run["initial_model"], dataset, settings,
                condition=name, checkpoint="update_3", alphas=(0.01, 1.0, 100.0),
                observation_datasets=sensitivity_inputs,
                comparison_id="synthetic_three_update_masking_example",
            )
        predictions = pd.concat([result["predictions"] for result in evaluated.values()], ignore_index=True)
        expected = pd.DataFrame({
            "sequence_id": dataset.sequence_ids[dataset.test_rows],
            "source_id": dataset.source_ids[dataset.test_rows], "fold": dataset.fold,
        })
        representations = ("pretrained_online", "pretrained_teacher", "initial_online", "direct_pose", "training_mean")
        scores = aggregate_predictions(
            predictions, expected, seeds=(settings.seed,), conditions=tuple(conditions),
            representations=representations,
            observations=("unaltered", *sensitivity_inputs),
        )
        display(scores["per_seed"].query("observation == 'unaltered'")[[
            "condition", "representation", "r2", "mae", "evaluated_clips", "evaluated_sources",
        ]].round(3))
        """),
        md("""
        Source-balanced R² measures predictive fit while giving every source
        video the same total influence. A negative score means the squared
        error exceeds that of a constant equal to the evaluated targets' mean;
        that reference mean defines the metric and is never a fitted prediction
        baseline. The separately reported training-mean predictor uses only
        training targets. Mean absolute error gives the average size of an
        error in the movement score's original units.

        This example contains one declared outer fold and one training seed.
        In the full comparison, each seed must contribute predictions from all
        five outer folds before R² is computed. We then summarize these five
        seed-specific scores. Averaging fold R² values would change the metric;
        averaging predictions across seeds would instead evaluate an ensemble.
        The coverage check rejects duplicate rows, source overlap, or missing
        declared predictions before a complete comparison is reported.
        """),
        code("""
        readable_names = {
            "pretrained_online": "Pretrained online", "pretrained_teacher": "Pretrained teacher",
            "initial_online": "Initial encoder", "direct_pose": "Direct pose", "training_mean": "Training mean",
        }
        intact = scores["per_seed"].query("observation == 'unaltered'")
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
        for ax, condition in zip(axes, conditions):
            rows = intact[intact.condition == condition].set_index("representation").loc[list(representations)]
            ax.barh(np.arange(len(rows)), rows.mae, color="#387b9a", height=0.65)
            ax.set_yticks(np.arange(len(rows)), [readable_names[name] for name in rows.index])
            ax.invert_yaxis()
            ax.set_xlabel("Mean absolute error (lower is better)")
            ax.set_title(condition.replace("_", " ").capitalize())
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle("Generated data, three training updates: software demonstration", fontsize=12)
        plt.show()
        """),
        md("""
        Each panel reports absolute error, so the reader can compare pretrained
        and initial features without decoding a subtraction label. Any ordering
        here concerns a brief synthetic execution. A useful real-data result
        would require better prediction across excluded source videos, with
        comparable coverage and uncertainty, under the declared training budget.

        If teacher prediction becomes easier but both frozen readouts remain
        weak, the current feature objective has not established useful movement
        learning. If the initial encoder performs similarly, the learned
        changes have not demonstrated added value. Improvement only when a
        specific region is missing would support a more limited sensitivity
        finding that should be reported at that observation condition.
        """),
        code("""
        sensitivity = scores["per_seed"].query("representation == 'pretrained_online'")
        display(sensitivity[[
            "condition", "observation", "mae", "evaluated_clips", "evaluated_sources", "unavailable_clips",
        ]].round(3))
        """),
        md("""
        ## 7. Remove raw observations before normalization and interpolation

        Removing coordinates after normalization can leave information from
        hidden measurements inside the remaining input. For example, a removed
        hip could already have contributed to the pelvis center or body scale.
        A claim about genuinely missing measurements requires removal before
        those calculations.

        The next independent example constructs raw coordinates with visibility
        values, removes a fixed ankle interval, and calls the existing pose
        preparation using only remaining observations. Changing the withheld
        coordinates cannot change the prepared result. The original movement
        score is supplied separately and remains unchanged. Short gaps may be
        interpolated from permitted endpoints, so we report raw removed counts
        separately from prepared valid-token counts.
        """),
        code("""
        raw_rng = np.random.default_rng(11)
        raw = raw_rng.normal(size=(16, 33, 4))
        raw[..., 3] = 1
        raw[:, 23, :3] = [-0.5, 0, 0]
        raw[:, 24, :3] = [0.5, 0, 0]
        withheld = np.zeros((16, 33), dtype=bool)
        withheld[4:12, [27, 29, 31]] = True
        preparation = dict(original_target=0.2, frames=16, max_interpolation_gap=2)
        prepared = prepare_raw_missing_observations(raw, np.arange(16), 30, withheld, **preparation)
        changed = raw.copy()
        changed[withheld, :3] = 10000
        prepared_again = prepare_raw_missing_observations(changed, np.arange(16), 30, withheld, **preparation)
        np.testing.assert_array_equal(prepared["xyz"], prepared_again["xyz"])
        np.testing.assert_array_equal(prepared["valid"], prepared_again["valid"])
        display(pd.DataFrame([{key: value for key, value in prepared.items() if key not in ("xyz", "valid")}]))
        """),
        md("""
        The raw example tests the missing-information boundary. The sensitivity
        scores above use already prepared synthetic inputs and retain that
        narrower interpretation. A real missing-data evaluation must build each
        corrupted input from its raw archive with this boundary, retain the
        unaltered recording's target, and apply the same prepared inputs to
        every model. Cases with no usable input remain explicit unavailable
        predictions, with their source and coverage information retained.
        """),
        md("""
        ## 8. Keep source uncertainty separate from training variation

        To compare the two masking conditions, we resample complete source
        videos. Each sampled source carries all its clips, both methods, and
        every seed's prediction together. We calculate the metric within each
        seed and average the paired differences. The interval conditions on
        the fitted models; it does not estimate what would happen if models
        were retrained on every resampled cohort.

        The example uses 100 resamples to keep execution small. Its tiny test
        set and one synthetic training seed cannot support a research claim.
        A real analysis should predeclare its larger resampling budget and
        report seed variability separately. Sign agreement, when added, also
        needs a declared near-zero exclusion rule: a tiny signed target can
        change direction under a small measurement error.
        """),
        code("""
        uncertainty = paired_source_bootstrap(
            predictions, first="all_landmark_targets", reference="gait_targets",
            representation="pretrained_online", repetitions=100,
        )
        display(pd.DataFrame([uncertainty]).round(3))
        """),
        md("""
        ## 9. Apply the evaluation when real experiments have been enabled

        Notebook 12 displays the real experiment's entire workload before
        training can start, beginning with the retained 1,200-update recipe.
        Its real runner saves compatible trained states in a separate location.
        The same `evaluate_frozen_representations` call used above evaluates
        each declared fold, training seed, and checkpoint. Per-clip prediction
        files must retain source, fold, seed, condition, reference target, and
        availability before their results are aggregated.

        Evaluate the predeclared checkpoint and observation bank consistently.
        Refit each imputer, scaler, and selected ridge readout using that fold's
        training sources. Pool all outer predictions within each seed, include
        initial, direct-pose, and training-mean controls, then report the shared
        observable outcome alongside feature diagnostics. The exploratory GAVD
        results already examined have informed this development; independent
        recordings would strengthen a later confirmation.

        No new real-data scores are supplied by this notebook. The completed
        comparison in Notebook 08 found no established predictive advantage
        for gait-only over all-landmark scattered targets under its tested
        recipe. The new implementation makes additional hypotheses testable:
        whether another missing-information task helps frozen movement
        prediction, whether its benefit depends on observation gaps, and
        whether the JEPA predictor's easier task corresponds to a useful
        change in its encoder.
        """),
    ])
