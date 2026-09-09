"""Editable GAVD tutorial: reopen the full trained grid and interpret held-out readouts."""
from nbformat.v4 import new_notebook
from .motion_results_20260908 import add_saved_result_interpretation
from .masking_shared import (md, code, setup_cell, data_instructions, configuration_cell, inputs_cell, plan_cell)


def build_notebook():
    notebook = new_notebook(cells=[
        md('''
        # 18 — Evaluate motion information from the saved GAVD grid

        This notebook loads the **same five-fold, five-seed grid trained in
        Notebook 17**, including real train/test source membership. It never
        starts encoder training. In a fresh kernel, it reconstructs expected
        data/model identities, locates every saved job, and reuses or recomputes
        its frozen readouts. A missing job produces an explicit status table;
        a partial grid cannot become a full-grid result.

        The central question from [TUTORIAL.md](docs/TUTORIAL.md) is whether
        training learns useful movement information that simple averaging hides.
        We compare temporal summaries against the same summaries of initial
        features, then inspect predictor correspondence separately. This makes
        favorable and unfavorable results equally interpretable.
        '''),
        setup_cell(load_environment=True), data_instructions(), configuration_cell(), inputs_cell(),
        md('''
        ## 2. Reconstruct the exact training grid before loading results

        Keep `DATA_MODE`, `FOLDS`, `SEEDS`, `EXPERIMENTS`, `DEVICE` and
        `OUTPUT_ROOT` identical to Notebook 17. The `RUN_TRAINING` variable has
        no effect in this notebook. Model initialization settings and the resolved
        backend are part of compatibility; changing them identifies a different
        job. A saved manifest is not its own proof of compatibility.

        The primary declaration expects 50 paired jobs containing 125 encoders.
        The census verifies test coverage independently of the prediction files.
        With all five folds, every accepted sequence must appear in a test set
        exactly once per seed. Changing the scope to a subset creates a pilot.
        '''),
        plan_cell(),
        code('''
        checkpoint_progress = NotebookTaskProgress("Saved GAVD checkpoint inspection", "stage")
        status = grid_status_with_progress(plan, inputs, progress=checkpoint_progress)
        display(status)
        availability = status.groupby(["experiment", "training_status"]).size().unstack(fill_value=0)
        ax = availability.plot.bar(stacked=True, figsize=(8, 3.5), title=f"{DATA_MODE.upper()}: saved paired jobs")
        ax.set(xlabel="Experiment", ylabel="Fold/seed jobs")
        ax.figure.tight_layout(); display(ax.figure); plt.close(ax.figure)
        '''),
        md('''
        ## 3. Separate representation learning from readout design

        | Representation | What is frozen? | Inference it supports |
        |---|---|---|
        | `pretrained_online__mean` / `__mean_motion` | Trained online encoder | Content available without the JEPA predictor |
        | `pretrained_teacher__mean` / `__mean_motion` | Final EMA teacher | Teacher representation quality |
        | `initial_online__mean` / `__mean_motion` | Same job's initial encoder | Architecture and readout control |
        | `direct_pose` | Prepared coordinates | Information already accessible from pose summaries |
        | `training_mean` | Training-source target mean | No-feature prediction baseline |

        `mean` retains the existing five bilateral sums and differences of
        average features. `mean_motion` also includes temporal standard
        deviation, mean absolute consecutive feature changes and valid-support
        fractions. Only common bilateral support and adjacent valid transitions
        contribute. The summaries describe prepared tokens; they cannot restore
        physical timing lost during input resizing.

        The ridge grid is 0.01, 0.1, 1, 10, 100, 1,000 and 10,000. Preprocessing
        and alpha selection use outer-training sources only. This extension uses
        the comparative readout helper's **three source-separated inner groups**;
        these are not the registered protocol's four inner model-selection
        folds. Pretraining sees all outer-training sources, so this inner step
        selects the readout only. Tuning an entire encoder recipe would require
        excluding inner-validation sources from candidate pretraining too.

        Both summaries are declared outputs. Outer-test scores cannot choose a
        summary, checkpoint or alpha. A boundary alpha is reported for inspection;
        a wide grid alone does not guarantee adequate regularization.
        '''),
        code('''
        evaluation_progress = NotebookTaskProgress("Saved-grid readouts and source-level reporting", "stage")
        results = collect_gavd_grid_with_progress(plan, inputs, progress=evaluation_progress)
        print(results["status"])
        COMPLETE = results["status"] == "Complete"
        if COMPLETE:
            print("Verified grid report:", results["directory"])
            display(results["jobs"])
            display(results["per_seed"][["experiment", "condition", "representation", "seed",
                "r2", "mae", "evaluated_clips", "evaluated_sources"]])
        else:
            display(results["jobs"].query("training_status == 'missing'"))
            print("Complete these jobs in Notebook 17, then rerun this cell. No substitute model was trained.")
        '''),
        md('''
        ## 4. Pool predictions correctly and inspect paired differences

        R² and MAE are calculated after pooling all declared outer-test folds
        for each seed, with equal total weight per source video. With the full
        GAVD grid, each arm/summary/seed has 625 predictions from 93 videos.
        Fold-specific R² has a different denominator and must not be averaged.
        A negative pooled R² means worse squared error than the pooled weighted
        test-target mean; the deployable training-mean control is shown separately.

        For the tables below, positive `delta_r2` and negative `delta_mae` favor
        the first representation in the contrast. Compare trained against initial
        within the **same** summary before attributing gains to learning.
        The saved `paired_intervals` additionally compares each mask arm with its
        own uniform control using teacher mean-motion features and 2,000 paired
        source-video bootstrap resamples. Whole videos, clips and paired seed
        predictions stay together. Those intervals condition on fitted models;
        they do not measure full retraining uncertainty. Seed variation is separate.
        '''),
        code('''
        if COMPLETE:
            display(results["summary"])
            contrasts = readout_contrasts(results["per_seed"])
            display(contrasts)
            display(contrasts.groupby(["experiment", "condition", "contrast"], sort=False).agg(
                mean_delta_r2=("delta_r2", "mean"), seed_sd_delta_r2=("delta_r2", "std"),
                mean_delta_mae=("delta_mae", "mean")))
            display(results["paired_intervals"])
            selected = results["selection"].query("selected")
            display(selected[["experiment", "condition", "representation", "fold", "seed",
                              "selected_alpha", "at_grid_boundary"]])
            display(results["diagnostics"][["experiment", "condition", "representation", "fold", "seed",
                                            "effective_rank", "near_constant"]])
        else:
            print("No complete declared grid: performance and learned-over-initial inference remain pending.")
        '''),
        md('''
        ## 5. Ask whether the predictor preserves clip correspondence

        Frozen encoder readout and JEPA prediction answer different questions.
        Every saved job also uses the same prespecified evaluation masks (bank
        seed 1801), including scattered targets and bilateral leg gaps. The normal
        pathway is online encoder → predictor, with full-input teacher targets.
        Initial-model controls undergo the same diagnostics.

        Compare `mismatched_target_mse` against
        `matched_target_mse_on_control_clips`: these use the same subset where a
        cross-source mismatch exists. A larger mismatch error supports sensitivity
        to clip correspondence. A smaller raw own-teacher MSE across two trained
        arms cannot rank semantics, because their teacher spaces and scales differ.
        `normalized_error` divides by mean squared teacher-channel value, not
        centered variance. Inspect feature variation and effective rank as well.
        '''),
        code('''
        if COMPLETE:
            predictor = results["predictor_diagnostics"].copy()
            predictor["correspondence_gap"] = (
                predictor.mismatched_target_mse - predictor.matched_target_mse_on_control_clips)
            display(predictor[["experiment", "condition", "fold", "seed", "evaluation_mask",
                "evaluated_clips", "evaluated_sources", "feature_mse", "normalized_error",
                "matched_target_mse_on_control_clips", "mismatched_target_mse", "correspondence_gap"]])
        else:
            print("Predictor diagnostics await the same saved GAVD grid; no synthetic scores substituted.")
        '''),
        md('''
        ## 6. Turn the evidence into the next experiment

        | Observed pattern | Supported interpretation | Next investigation |
        |---|---|---|
        | Temporal summaries help initial and trained features similarly | Readout design explains much of the gain | Keep the matched initial control; avoid crediting JEPA alone |
        | Trained-over-initial gain appears under temporal summaries | Averaging may hide learned movement content | Check seeds, source contrasts and regularization diagnostics |
        | A mask beats its uniform control but not initial features | Relative masking improvement without demonstrated learning benefit | Inspect the objective and preparation before broader mixtures |
        | Low predictor error accompanies near-constant features | Error reduction may reflect an uninformative target space | Inspect teacher variation and mismatch controls |
        | Neither readout nor predictor gives useful contrasts | Current representation/task may miss the endpoint | Revisit timing preparation or a focused objective change |

        These are conditional interpretations, not conclusions manufactured from
        absent results. Report unfavorable outcomes, boundary penalties and
        unstable seeds. GAVD has informed these hypotheses; even a full new grid
        remains development evidence for a coordinate-derived endpoint.

        Missing raw measurements require the before-preparation tests in Notebook
        13. Future prediction requires Notebook 14's past-only inputs and
        persistence, velocity, mismatched-future and observed-future controls.
        Completion masks do not establish forecasting or clinical validity.
        '''),
        md('''
        ## 7. Optional teaching check: a signal that temporal averaging destroys

        This last cell is explicitly generated data, separate from the GAVD grid.
        Two sinusoidal limbs have different amplitudes but zero temporal means.
        A temporal summary should recover their known amplitude contrast while a
        mean cannot. This validates a recoverable construction; it does not prove
        that real JEPA tokens encode amplitude in the same way.
        '''),
        code('''
        from laterality_extensions.motion_readout import pooling_positive_control
        teaching_progress = NotebookTaskProgress("Generated amplitude control", "stage")
        teaching_scores, generated = run_notebook_task(pooling_positive_control,
            progress=teaching_progress, label="Fit and evaluate the two generated-data summaries")
        display(teaching_scores.assign(evidence="generated amplitude control, not GAVD"))
        fig, ax = plt.subplots(figsize=(7, 2.8), constrained_layout=True)
        ax.plot(generated["wave"], label="Unit oscillation (mean zero)")
        ax.axhline(0, color="gray", linewidth=0.7)
        ax.set(title="Generated teaching control only", xlabel="Prepared step", ylabel="Amplitude")
        ax.legend(); display(fig); plt.close(fig)
        '''),
        md('''
        ## 8. Optional: reuse a retained Notebook 12 encoder first

        If compatible older checkpoints are available, this is the cheaper
        readout test recommended in the tutorial. Set
        `LATERALITY_RETAINED_COMPARISON` to one Notebook 12 job directory.
        The loader checks current data, code and runtime, then saves separate
        readouts without changing the original checkpoint. This optional
        single-job diagnostic cannot substitute for Notebook 17's full grid.
        '''),
        code('''
        retained_directory = os.getenv("LATERALITY_RETAINED_COMPARISON", "")
        retained_progress = NotebookTaskProgress("Optional Notebook 12 readout reanalysis", "stage")
        retained = evaluate_retained_motion_with_progress(retained_directory,
            progress=retained_progress, enabled=bool(retained_directory))
        if retained_directory:
            display(retained["selection"].query("selected")[[
                "condition", "representation", "fold", "seed", "selected_alpha", "at_grid_boundary"]])
            print("One retained Notebook 12 job reanalysed without encoder training; separate from the full grid.")
        else:
            print("Optional older-checkpoint reanalysis not configured.")
        '''),
    ])
    return add_saved_result_interpretation(notebook, 18)
