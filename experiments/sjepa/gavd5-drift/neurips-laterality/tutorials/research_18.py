"""Editable tutorial: recover movement information before expanding training."""
from nbformat.v4 import new_notebook
from .masking_shared import md, code, setup_cell


def build_notebook():
    return new_notebook(cells=[
        md('''
        # 18 — Does the readout retain movement information?

        The [tutorial's Direction A](docs/TUTORIAL.md#direction-a-explain-and-recover-access-to-movement-information)
        has the clearest near-term question: why can the JEPA predictor learn
        clip-related information without improving the movement endpoint?
        Whole-window means can discard oscillation amplitude. We test that
        explanation with a known synthetic signal, then apply two declared
        summaries equally to trained and matched initial encoders.

        This notebook executes independently using small generated examples.
        It does not establish a new GAVD result. Notebook 17's explicitly enabled
        real runner uses the same readout helpers and saves the full predictions.
        ''') ,
        setup_cell(),
        code('''
        from laterality_extensions.motion_readout import (
            pooling_positive_control, evaluate_motion_readouts, aggregate_motion_study,
        )
        control_scores, fixture = pooling_positive_control()
        display(control_scores)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
        for row in (0, 1, 2):
            axes[0].plot(fixture["tokens"][row, :, 27, 0], label=f"Generated example {row + 1}")
        axes[0].set(xlabel="Prepared time block", ylabel="Toy feature value", title="Different amplitudes, zero means")
        axes[0].legend(fontsize=8)
        axes[1].bar(control_scores.summary, control_scores.r2, color=["#9fb0bc", "#21679b"])
        axes[1].set(ylabel="Held-out source R²", title="Known synthetic amplitude contrast")
        display(fig); plt.close(fig)
        assert control_scores.set_index("summary").loc["mean_motion", "r2"] > 0.99
        ''') ,
        md('''
        ## 1. An evaluation positive control

        Both sides follow a complete sinusoidal cycle. Their mean is zero, while
        their amplitudes differ. The target is the known amplitude difference,
        so a summary containing temporal variation can express it. Source-separated
        ridge selection must recover that signal on generated test sources.
        This verifies a recoverable example; it does not promise improvement on
        natural gait or show that real JEPA tokens encode amplitude similarly.

        The `mean` representation keeps the existing five left/right sums and
        differences of mean features. `mean_motion` appends the corresponding
        standard deviations, mean absolute consecutive feature changes, and
        valid-support fractions. Only common bilateral support and adjacent
        observed transitions contribute. These statistics describe prepared
        tokens and do not restore the original physical timing.

        ## 2. Apply the same summaries to every control

        We widen the ridge grid through 10,000 and select the penalty on inner
        training-source folds. Boundary choices are reported. Both summaries
        remain declared outputs: outer-test scores never choose a winning
        summary or checkpoint. Pretraining can use all outer-training sources,
        so this inner validation selects the readout only. Selecting an entire
        pretraining recipe would also require excluding inner validation sources
        from candidate encoder training.
        ''') ,
        code('''
        from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
        from laterality_extensions.motion_structured_training import train_mask_study
        data = load_learning_dataset()
        settings = LearningSettings(steps=2)
        comparison = train_mask_study(data, settings, experiment="motion")
        evaluation = evaluate_motion_readouts(comparison, data, settings)
        expected = pd.DataFrame({"sequence_id": data.sequence_ids[data.test_rows],
            "source_id": data.source_ids[data.test_rows], "fold": data.fold})
        predictions = evaluation["predictions"].assign(experiment="motion")
        demonstration_plan = {"experiments": ("motion",), "seeds": (settings.seed,),
                              "arms": {"motion": comparison["runs"]}}
        scored = aggregate_motion_study(predictions, expected, demonstration_plan)
        display(scored["per_seed"][["condition", "representation", "r2", "mae",
                                     "evaluated_clips", "evaluated_sources"]])
        chosen = evaluation["selection"].query("selected")
        display(chosen[["condition", "representation", "alpha", "at_grid_boundary"]])
        display(evaluation["diagnostics"][["condition", "representation", "effective_rank", "near_constant"]])
        ''') ,
        md('''
        Two updates and the small generated cohort are sufficient for software
        checks. Rankings in this table are not empirical evidence for one mask.
        Initial online features, direct-pose statistics and the training mean
        remain visible alongside both trained encoders. Added feature dimensions
        change the readout's capacity; compare mean-motion summaries against
        equally summarized initial features before attributing a gain to learning.

        ## 3. Ask what the JEPA predictor has learned separately

        Use the same prespecified scattered and bilateral leg-gap masks for
        every model. The normal pathway remains online encoder to predictor,
        with the full-input teacher providing targets. Cross-source mismatched
        targets test clip correspondence, while feature variation and norms
        diagnose uninformative representations. Each model supplies its own
        teacher, so raw or normalized loss does not define a common semantic scale.
        ''') ,
        code('''
        from laterality_extensions.comparative_evaluation import (
            make_evaluation_mask_bank, predictor_diagnostics,
        )
        valid = data.valid.reshape(len(data.xyz), -1, 4, 33).all(2)
        bank = make_evaluation_mask_bank(valid, seed=1801)
        diagnostic_rows = [predictor_diagnostics(run["model"], data, bank, condition=name)
                           for name, run in comparison["runs"].items()]
        diagnostic_rows.append(predictor_diagnostics(next(iter(comparison["runs"].values()))["initial_model"],
                                                     data, bank, condition="initial"))
        predictor_table = pd.concat(diagnostic_rows, ignore_index=True)
        display(predictor_table[["condition", "evaluation_mask", "evaluated_clips", "evaluated_sources",
                                 "feature_mse", "mismatched_target_mse", "normalized_error"]])
        ''') ,
        md('''
        The normalized error divides feature MSE by the source-balanced mean
        squared teacher-channel value. It is not a centered variance ratio and
        does not remove all effects of changing teacher scale. The complete table
        retains mismatch availability and target/prediction variation diagnostics.

        ## 4. Continue from retained encoders before spending more compute

        The optional cell below reanalyses one retained Notebook 12 job without
        encoder training. Set `LATERALITY_RETAINED_COMPARISON` to its result
        directory to enable it. The loader checks compatibility against current
        data, code and runtime before extracting features. New readout artifacts
        have a separate output directory. Local models are required; tracked
        numerical summaries alone cannot supply token features. One job's output
        is a fold/seed diagnostic, not the full declared evaluation grid.
        ''') ,
        code('''
        from laterality_extensions.motion_readout import evaluate_retained_comparison
        retained_directory = os.getenv("LATERALITY_RETAINED_COMPARISON", "")
        if retained_directory:
            retained = evaluate_retained_comparison(retained_directory)
            display(retained["selection"].query("selected")[[
                "condition", "representation", "selected_alpha", "at_grid_boundary", "fold", "seed"]])
            print("Retained job reanalysed; no encoder training. Per-clip predictions saved separately.")
        else:
            print("Retained-encoder reanalysis not requested; no empirical results substituted.")
        ''') ,
        md('''

        A gain shared by initial and trained encoders points toward readout design.
        A learned-over-initial gain confined to temporal summaries would support
        the hypothesis that averaging obscured useful content. If neither gains,
        inspect preparation timing and the objective before adding mask mixtures.
        Report unfavorable results and any near-constant feature diagnostics.

        Missing-observation prediction remains a separate evaluation. Removing
        already prepared tokens tests representation sensitivity. A claim about
        genuinely missing measurements requires removing raw observations before
        interpolation and normalization, as in Notebook 13. Future-feature
        decoding remains Notebook 14's past-only task, with its observed-future
        decoder check, persistence, velocity, direct-past and mismatched-future
        controls. None of the completion masks here establishes forecasting.

        The [research specification](docs/MOTION_STRUCTURED_MASKING.md) records
        the proposed real comparisons and distinguishes implemented software,
        completed synthetic checks and unrun empirical analyses.
        ''')
    ])
