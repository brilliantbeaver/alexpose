"""Authoring source for the matched-target-budget research tutorial."""
from textwrap import dedent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    def md(text):
        return new_markdown_cell(dedent(text).strip())

    def code(text):
        return new_code_cell(dedent(text).strip())

    return new_notebook(cells=[
        md("""
        # 08 — Does choosing gait landmarks improve self-supervised learning?

        The completed laterality experiment used the same set of gait-related
        prediction targets throughout training. Its results therefore cannot tell
        us whether choosing those landmarks helped the representation. This
        notebook turns that open question into a controlled comparison: we train
        two fresh encoders that differ in where they predict hidden features,
        while keeping the number of hidden features and the rest of the recipe
        fixed.

        The default run uses generated movement sequences and a few CPU updates.
        It demonstrates the complete method, including genuinely held-out source
        groups, but its scores are **synthetic software checks, not research
        findings about human gait**. A separate, disabled-by-default section can
        run the same comparison on the existing local pose cohort. Neither mode
        changes the registered notebooks, checkpoints, or reported conclusions.

        By the end, you should understand what a fair masking comparison holds
        fixed, how to check whether pretraining adds useful information, and
        which additional evidence would make a positive result convincing.
        """),
        md("""
        ## Step 1 — State the scientific question before training

        A joint-embedding predictive architecture, or JEPA, predicts features of
        hidden input regions from the visible context. Here, a region is one
        landmark over four consecutive samples. The student encoder receives a
        partly hidden pose sequence; a slowly updated teacher supplies the
        target features. Dataset condition names and the laterality measurement
        do not enter this representation-learning objective.

        Our question is whether concentrating prediction on gait-relevant
        landmarks makes the learned features more useful for a held-out
        movement task. The selected landmarks are the shoulders, hips, knees,
        ankles, heels, and foot tips: six bilateral pairs, or twelve landmarks.
        The remaining landmarks remain available as context.

        | Recipe | Eligible hidden targets | What stays fixed |
        |---|---|---|
        | Gait targets | The twelve predefined landmarks | Actual hidden-token count, architecture, initialization, training sources, update count, and read-out |
        | Uniform targets | Any of the thirty-three valid landmarks | The same controls as the gait-target recipe |

        A plausible benefit is that the prediction problem concentrates on
        movement relevant to the laterality probe. A plausible cost is that it
        neglects other useful body information. Both outcomes deserve reporting.
        Anatomical and motion-dependent masking already appear in skeleton
        learning, so a useful contribution would be a carefully explained
        empirical advantage in this setting, rather than a claim that selecting
        body regions is new by itself.
        """),
        code("""
        from pathlib import Path
        import importlib
        import sys
        import os
        from dataclasses import replace
        import warnings

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import torch
        from IPython.display import display
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        get_ipython().run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("svg", "png")

        def locate_suite(start=None):
            start = (start or Path.cwd()).resolve()
            for ancestor in (start, *start.parents):
                for candidate in (ancestor, ancestor / "neurips-laterality"):
                    if (candidate / "laterality_extensions").is_dir():
                        return candidate
            raise FileNotFoundError("Open this notebook from the project directory.")

        SUITE_ROOT = locate_suite()
        suite_entry = str(SUITE_ROOT)
        if suite_entry in sys.path:
            sys.path.remove(suite_entry)
        sys.path.insert(0, suite_entry)

        # Jupyter keeps imported modules alive across cell reruns. Import the
        # module object first, verify its origin, and reload it before binding
        # names that may have been added since this kernel started.
        import laterality_extensions.masked_learning as masked_learning

        expected_module = (SUITE_ROOT / "laterality_extensions" / "masked_learning.py").resolve()
        loaded_module = getattr(masked_learning, "__file__", None)
        if loaded_module is None or Path(loaded_module).resolve() != expected_module:
            raise ImportError(
                "A different laterality_extensions package is already loaded. "
                "Restart the kernel and open this notebook from the project directory."
            )
        importlib.invalidate_caches()
        masked_learning = importlib.reload(masked_learning)
        required_masking_api = (
            "GAIT_JOINTS", "LearningSettings", "comparison_cache_path",
            "configure_learning_runtime", "load_cached_comparison",
            "load_learning_dataset", "matched_target_masks",
            "resolve_learning_device", "run_matched_comparison",
            "save_comparison", "summarize_cross_fitted_predictions",
        )
        missing_masking_api = [
            name for name in required_masking_api
            if not hasattr(masked_learning, name)
        ]
        if (
            getattr(masked_learning, "MASKING_RUNNER_API_VERSION", 0) < 2
            or missing_masking_api
        ):
            raise ImportError(
                "The masking runner on disk is incomplete after reload; missing "
                + ", ".join(missing_masking_api or ["API version 2"])
            )
        GAIT_JOINTS = masked_learning.GAIT_JOINTS
        LearningSettings = masked_learning.LearningSettings
        comparison_cache_path = masked_learning.comparison_cache_path
        configure_learning_runtime = masked_learning.configure_learning_runtime
        load_cached_comparison = masked_learning.load_cached_comparison
        load_learning_dataset = masked_learning.load_learning_dataset
        matched_target_masks = masked_learning.matched_target_masks
        resolve_learning_device = masked_learning.resolve_learning_device
        run_matched_comparison = masked_learning.run_matched_comparison
        save_comparison = masked_learning.save_comparison
        summarize_cross_fitted_predictions = masked_learning.summarize_cross_fitted_predictions
        from notebook_progress import NotebookTaskProgress

        torch.set_num_threads(1)  # Small demonstration batches run well on one CPU thread.
        warnings.filterwarnings("ignore", message="enable_nested_tensor is True")
        LABEL = "SYNTHETIC DEMONSTRATION — NOT RESEARCH EVIDENCE"
        print(LABEL)
        """),
        md("""
        ## Step 2 — Separate sources before any learning

        A source is a recording group, with several clips belonging to that
        group. In the generated example, source identities are invented; in the
        real-data option, they are the original source videos. Every clip from a
        source stays on one side of the training/test boundary. The synthetic
        example has twenty training sources and five test sources.

        We sample training sources uniformly and then choose a clip within each
        sampled source. Thus, a source with many clips does not automatically
        receive more optimizer updates. Test sources never enter representation
        training or the fitted read-out. A source-video split still cannot prove
        that the same person is absent from another video; the real-data claim
        remains restricted to held-out source videos.
        """),
        code("""
        dataset = load_learning_dataset(real=False, fold=0)
        assert not set(dataset.train_sources) & set(dataset.test_sources)
        display(pd.DataFrame({
            "partition": ["training", "held-out test"],
            "sources": [len(dataset.train_sources), len(dataset.test_sources)],
            "sequences": [len(dataset.train_rows), len(dataset.test_rows)],
        }))
        print("Pose array:", dataset.xyz.shape, "= sequences × samples × landmarks × coordinates")
        """),
        md("""
        ## Step 3 — Match the number of hidden targets

        Matching a percentage is insufficient when the candidate sets have
        different sizes. With four temporal segments, twelve landmarks supply
        forty-eight possible targets; thirty-three landmarks supply 132.
        Hiding half of each would create two different prediction budgets.

        Instead, we first choose a count using the available gait targets and
        then hide that many valid tokens in either recipe. Within a batch, the
        smallest available candidate set determines a feasible common count.
        At least one valid context token remains. Shared random priorities keep
        draws paired where candidate sets overlap, without forcing both recipes
        to select the same positions.

        The code below displays the selected locations for one example. Dark
        cells are hidden targets. Their distribution differs, but their counts
        must agree; a failed assertion stops the comparison before interpretation.
        """),
        code("""
        patch_valid = dataset.valid[:2].reshape(2, -1, 4, 33).all(axis=2)
        masks = matched_target_masks(
            patch_valid, mask_fraction=0.5, rng=np.random.default_rng(12),
        )
        counts = {name: mask.sum(axis=(1, 2)).tolist() for name, mask in masks.items()}
        assert counts["gait"] == counts["uniform"]
        display(pd.DataFrame(counts).rename_axis("example"))

        fig, axes = plt.subplots(2, 1, figsize=(10, 3.5), constrained_layout=True)
        for ax, (name, mask) in zip(axes, masks.items()):
            ax.imshow(mask[0], aspect="auto", cmap="Blues", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.set_title(f"{name.capitalize()} targets: {int(mask[0].sum())} hidden tokens", loc="left")
            ax.set_ylabel("Time segment")
            ax.set_yticks(range(mask.shape[1]))
        axes[-1].set_xlabel("Landmark index (0–32)")
        fig.suptitle(LABEL, fontsize=10)
        display(fig)
        plt.close(fig)
        """),
        md("""
        ## Step 4 — Keep the rest of the learning recipe unchanged

        Both encoders start from the same parameter values and receive the same
        source draws and geometric views. The teacher follows the student with
        an exponential moving average. Its features are detached from the
        gradient calculation, so the student must learn to predict them rather
        than update both sides directly through the same loss.

        The masked-prediction loss compares distributions over feature channels.
        A second term encourages variation across examples and agreement
        between two slightly transformed views. This is the existing VICReg
        regularizer. Its pooling landmarks stay fixed across the masking
        recipes; otherwise, changing the mask would also change a second part
        of the training objective.

        The demonstration uses eight updates and a sixteen-dimensional encoder.
        These choices make the notebook quick to execute; they do not establish
        a scientifically adequate training budget. A fixed ridge penalty of one
        is used for every frozen-feature read-out, with no test-set selection.
        For a research run, this penalty can be replaced by a predeclared
        training-source-only selection procedure applied equally to every arm.
        """),
        code("""
        settings = LearningSettings(
            seed=7, fold=0, steps=8, batch_size=5,
            embed_dim=16, encoder_depth=1, predictor_depth=1, heads=2,
            mask_fraction=0.5, ridge_alpha=1.0,
        )
        comparison = run_matched_comparison(dataset, settings)
        display(pd.Series(comparison["pairing"], name="comparison checks passed"))
        assert all(comparison["pairing"].values())
        """),
        md("""
        ## Step 5 — Evaluate useful content, including the untrained baseline

        After training, we freeze the teacher encoder and summarize its tokens
        with the same bilateral features used by the laterality workflow. A
        linear read-out predicts the coordinate-derived motion contrast from
        training sources and is then evaluated on test sources. Imputation,
        scaling, and read-out coefficients are fitted using training rows only.

        Two additional comparisons help interpret the result. The paired
        untrained encoder tests whether pretraining improves on information
        already available through the architecture and pose inputs. Direct pose
        summaries test whether simple movement statistics suffice without an
        encoder. The direct baseline has a different feature dimension, so it
        is a practical reference rather than a capacity-matched neural control.

        Each test source receives equal total weight. R² compares prediction
        error with variation around the held-out target mean; higher is better,
        and negative values are possible. Mean absolute error reports the
        average prediction error in the dimensionless target units; lower is
        better. These are representation-probe measurements, not diagnostic
        accuracy or validated clinical mobility scores.
        """),
        code("""
        metrics = comparison["metrics"]
        readable = metrics[metrics.representation == "learned_encoder"].copy()
        controls = metrics[(metrics.variant == "gait_targets") &
                           (metrics.representation != "learned_encoder")].copy()
        display(pd.concat([readable, controls], ignore_index=True).round({"r2": 3, "mae": 3}))

        learned = readable.set_index("variant")
        differences = pd.Series({
            "R² difference: gait-target recipe minus uniform-target recipe":
                learned.loc["gait_targets", "r2"] - learned.loc["uniform_targets", "r2"],
            "Absolute-error difference: gait-target recipe minus uniform-target recipe":
                learned.loc["gait_targets", "mae"] - learned.loc["uniform_targets", "mae"],
        })
        display(differences.round(3))
        print("These differences describe this generated example only.")
        """),
        md("""
        ## Step 6 — Read the outcome without turning a demonstration into a finding

        First check the pairing assertions, then compare the two trained
        encoders, and finally compare each with its paired initialization. A
        lower training loss alone would only show that one prediction task was
        easier under its own targets. It would not establish a better movement
        representation, because the recipes predict different positions.

        A useful real-data result would show a consistent held-out advantage for
        gait targets and a meaningful improvement over the untrained encoder.
        Similar scores would leave the benefit uncertain; an adverse result
        would identify a limitation of this anatomical choice. The short
        synthetic run commonly leaves trained and untrained scores close, which
        is unsurprising after so few teacher updates. Do not rank research
        directions from the sign of that small numerical difference.
        """),
        code("""
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
        for name, history in comparison["history"].groupby("variant", sort=False):
            axes[0].plot(history.step, history.loss, marker="o", label=name.replace("_", " "))
        axes[0].set(xlabel="Optimizer update", ylabel="Combined training loss",
                    title="Optimization check")
        axes[0].legend(frameon=False, fontsize=9)
        reference = metrics[metrics.variant == "gait_targets"].set_index("representation")
        values = [learned.loc["gait_targets", "r2"], learned.loc["uniform_targets", "r2"],
                  reference.loc["initial_encoder", "r2"], reference.loc["raw_pose", "r2"]]
        axes[1].barh(["Gait targets", "Uniform targets", "Untrained encoder", "Direct pose"], values,
                     color=["#2563eb", "#0d9488", "#94a3b8", "#cbd5e1"])
        axes[1].axvline(0, color="#475569", linewidth=0.8)
        axes[1].set(xlabel="Held-out R²; higher is better", title="Useful information check")
        axes[1].invert_yaxis()
        fig.suptitle(LABEL, fontsize=10)
        display(fig)
        plt.close(fig)
        """),
        md("""
        ## Step 7 — Add controls that distinguish anatomy from other explanations

        Uniform masking changes both the target locations and their distribution
        over body regions. A stronger study therefore includes several
        preselected random twelve-landmark subsets. Each subset must be chosen
        without seeing its test performance. Comparing one favorable subset
        after trying many would bias the conclusion.

        The helper supports a random-subset arm. When it joins a comparison, the
        hidden-token budget is matched across all three policies. If visibility
        makes that subset less available, the common budget may decrease, so
        rerun the gait and uniform arms alongside it. Report availability and
        target counts rather than comparing runs with different difficulties.

        Motion-aware masking is another important prior-art control when making
        a claim about target-selection methods more broadly. It is not
        implemented here, and this notebook does not claim superiority to it.
        Changing the pool, adding a label loss, or tuning a special read-out
        only for one recipe would require a separate experiment.
        """),
        code("""
        RUN_RANDOM_SUBSET_DEMONSTRATION = False
        if RUN_RANDOM_SUBSET_DEMONSTRATION:
            random_control = run_matched_comparison(
                dataset, settings,
                variants={
                    "gait_targets": {"mask_policy": "gait"},
                    "uniform_targets": {"mask_policy": "uniform"},
                    "random_twelve": {"mask_policy": "random_subset"},
                },
            )
            display(random_control["metrics"].round({"r2": 3, "mae": 3}))
        else:
            print("Optional random-subset demonstration was not run.")
        """),
        md(r"""
        ## Step 8 — Run a separately declared local research experiment

        This section is disabled until you set
        `LATERALITY_RESEARCH_RUN_REAL=1` before starting the kernel. It loads the
        registered cohort and source partitions read-only. Missing or stale
        artifacts stop the run; the code never substitutes synthetic data.

        The full grid below contains five source folds and five training seeds.
        With two recipes, it contains fifty encoder fits. A completed exact-match
        comparison is validated and reused, so rerunning the cell does not
        deliberately train the same pair again. The suggested 1,200-update
        budget is a compute-matching choice, not a known minimum or optimum. In
        Notebook 03, each registered paper encoder ran for 300 epochs.
        An epoch contained four optimizer updates because each outer-training
        fold had 74 or 75 sources and the batch size was 20:

        $$\lceil 75/20\rceil=4,\qquad 300\times4=1{,}200\text{ updates}. $$

        Notebook 08 states the budget directly in optimizer updates. It uses the
        same five folds, five seeds, and two-recipe grid, so both studies contain
        50 encoders and 60,000 optimizer updates in total. Each encoder draws
        1,200 batches of 20 sequence positions, or 24,000 sampled positions.
        Across the full grid that is 1.2 million sampled positions. These are
        repeated training draws, not additional sequences or independent data.

        | Setting | Choice in the optional grid | Reason for the choice |
        |---|---:|---|
        | Outer folds | 5 | Reuse the registered source-video partitions without redrawing the test roles |
        | Seeds | 42–46 | Match the five registered initializations and expose training variation |
        | Masking recipes | 2 | Compare gait targets with uniform targets under paired conditions |
        | Updates per encoder | 1,200 | Match Notebook 03's 300 epochs × 4 updates, limiting update count as a confound |
        | Batch size | 20 | Match the registered paper batch size and source-balanced exposure scale |
        | Encoder | 96 dimensions, 4 encoder blocks, 2 predictor blocks, 4 heads | Match the registered paper architecture |
        | Learning rate / weight decay / VICReg weight | 0.001 / 0.05 / 0.05 | Reuse the registered starting values and regularization weights |
        | EMA momentum | 0.999, fixed | Start from the registered paper value; unlike Notebook 03, this exploratory runner does not increase it toward 1.0 |
        | Mask fraction | 0.5 | A transparent exploratory choice; both policies hide the same realized number of valid tokens |
        | Ridge penalty | 1.0, fixed | Keep held-out outcomes out of tuning in this tutorial; a final study should select it using training sources only |
        | Device | CUDA or MPS when available; otherwise CPU | Use an available accelerator; an explicitly requested unavailable accelerator stops rather than silently falling back |
        | Execution layout | Fold tensors resident on the device; two VICReg views augmented and encoded as one larger batch | Remove repeated transfers and expose more parallel work without changing examples, masks, losses, or optimizer updates |

        Equal update counts make the two masking arms fair to each other and put
        their cost on the same scale as the completed experiment. They do not
        guarantee convergence or make this an exact reproduction. Notebook 03
        used cosine learning-rate decay, an EMA schedule from 0.999 toward 1.0,
        and a mask fraction of 0.6. This runner uses a constant learning rate,
        fixed EMA momentum, and a mask fraction of 0.5 while changing which
        landmarks may become targets.

        The completed Notebook 03 loss curves provide context but do not prove
        that 1,200 updates are sufficient for the new objective. Most registered
        loss reduction occurred early, with smaller continued improvement from
        epoch 200 to 300. A future budget study should compare prespecified
        checkpoints using training-only diagnostics or validation. It must not
        choose the stopping point after examining outer-test results. A smaller
        grid can estimate runtime, but it must be labeled as a pilot rather than
        treated as equivalent evidence. The complete rationale is recorded in
        [the matched-budget parameter reference](docs/MATCHED_BUDGET_MASKING.md).

        The default `full` scope retains all folds and seeds. Setting
        `LATERALITY_RESEARCH_MASKING_SCOPE=single_seed_pilot` keeps all five
        held-out folds but runs only seed 42, reducing the grid from 50 to 10
        encoders. That mode is useful for timing and debugging, but its result is
        conditional on one initialization and is not equivalent to the full
        seed-robustness analysis.

        Automatic selection honors `LATERALITY_DEVICE`; for example, setting it
        to `mps` requires MPS and fails visibly if MPS is unavailable. CPU fallback
        uses up to eight threads by default and can be set explicitly with
        `LATERALITY_RESEARCH_CPU_THREADS`. Each comparison is saved atomically in
        a content-addressed private directory with settings, source roles,
        predictions, model states, and file digests. A cache candidate counts as
        reused only after its lineage, contents, and held-out row coverage pass
        validation. The original registered artifacts are never overwritten.
        Saved source identifiers remain private and are not automatically
        approved for sharing.
        """),
        code("""
        # A notebook file can be reloaded while its kernel still holds an older
        # version of this module. Refresh all related symbols together if the
        # in-memory runner predates progress, acceleration, or validated reuse.
        import importlib
        import inspect
        import laterality_extensions.masked_learning as masked_learning

        def masking_api_is_current():
            return (
                getattr(masked_learning, "MASKING_RUNNER_API_VERSION", 0) >= 2
                and "progress_callback" in inspect.signature(
                    masked_learning.run_matched_comparison
                ).parameters
                and all(hasattr(masked_learning, name) for name in (
                    "comparison_cache_path", "configure_learning_runtime",
                    "load_cached_comparison", "resolve_learning_device",
                ))
            )

        if not masking_api_is_current():
            importlib.invalidate_caches()
            masked_learning = importlib.reload(masked_learning)

        if not masking_api_is_current():
            raise RuntimeError(
                "The masking runner is stale and could not be refreshed. Restart the kernel, "
                "rerun the imports cell, and then rerun this cell."
            )

        # Rebind unconditionally. This also repairs notebook globals left over
        # from an older successful import in the same persistent kernel.
        GAIT_JOINTS = masked_learning.GAIT_JOINTS
        LearningSettings = masked_learning.LearningSettings
        comparison_cache_path = masked_learning.comparison_cache_path
        configure_learning_runtime = masked_learning.configure_learning_runtime
        load_cached_comparison = masked_learning.load_cached_comparison
        load_learning_dataset = masked_learning.load_learning_dataset
        matched_target_masks = masked_learning.matched_target_masks
        resolve_learning_device = masked_learning.resolve_learning_device
        run_matched_comparison = masked_learning.run_matched_comparison
        save_comparison = masked_learning.save_comparison
        summarize_cross_fitted_predictions = masked_learning.summarize_cross_fitted_predictions

        RUN_LOCAL_RESEARCH = os.getenv("LATERALITY_RESEARCH_RUN_REAL", "0") == "1"
        REAL_SCOPE = os.getenv("LATERALITY_RESEARCH_MASKING_SCOPE", "full").strip().lower()
        REAL_FOLDS = (0, 1, 2, 3, 4)
        if REAL_SCOPE == "full":
            REAL_SEEDS = (42, 43, 44, 45, 46)
        elif REAL_SCOPE == "single_seed_pilot":
            REAL_SEEDS = (42,)
        else:
            raise ValueError(
                "LATERALITY_RESEARCH_MASKING_SCOPE must be 'full' or 'single_seed_pilot'"
            )
        REAL_STEPS = 1200
        REAL_VARIANTS = 2
        print("Run scope:", REAL_SCOPE)
        print("Planned encoder slots before cache validation:",
              len(REAL_FOLDS) * len(REAL_SEEDS) * REAL_VARIANTS)
        print("Real-data training enabled:", RUN_LOCAL_RESEARCH)

        if RUN_LOCAL_RESEARCH:
            real_device = resolve_learning_device("auto")
            runtime = configure_learning_runtime(real_device)
            print("Training device:", runtime["device"])
            if runtime["cpu_threads"] is not None:
                print("CPU fallback threads:", runtime["cpu_threads"])
            if REAL_SCOPE == "single_seed_pilot":
                print("PILOT ONLY: results are conditional on seed 42.")

            def make_real_settings(fold, seed):
                return LearningSettings(
                    fold=fold, seed=seed, steps=REAL_STEPS, batch_size=20,
                    embed_dim=96, encoder_depth=4, predictor_depth=2, heads=4,
                    mask_fraction=0.5, ridge_alpha=1.0,
                    weight_decay=0.05, vicreg_weight=0.05, ema_momentum=0.999,
                    device=runtime["device"], confirm_real_run=True,
                )

            real_datasets = {
                fold: load_learning_dataset(real=True, fold=fold)
                for fold in REAL_FOLDS
            }
            real_settings = {
                (fold, seed): make_real_settings(fold, seed)
                for fold in REAL_FOLDS for seed in REAL_SEEDS
            }
            cache_candidates = {
                key: comparison_cache_path(real_datasets[key[0]], settings, study="masking").exists()
                for key, settings in real_settings.items()
            }
            planned_comparisons = len(REAL_FOLDS) * len(REAL_SEEDS)
            local_training_progress = NotebookTaskProgress(
                "Local matched-budget training",
                "fold/seed comparison",
                refresh_seconds=0.5,
            )
            local_training_progress.start(
                planned_comparisons,
                profile=f"exploratory real-data · {REAL_SCOPE} · {runtime['device']}",
                cached_candidate_units=sum(cache_candidates.values()),
                note=(
                    f"Each uncached comparison trains {REAL_VARIANTS} encoders for "
                    f"{REAL_STEPS:,} optimizer updates apiece; cached results must pass validation."
                ),
            )
            prediction_frames = []
            completed_runs = 0
            for fold in REAL_FOLDS:
                real_dataset = real_datasets[fold]
                for seed in REAL_SEEDS:
                    job_index = completed_runs + 1
                    key = (fold, seed)
                    candidate_cached = cache_candidates[key]
                    local_training_progress.start_unit(
                        job_index,
                        f"Fold {fold}, seed {seed}",
                        candidate_cached=candidate_cached,
                        total_steps=0 if candidate_cached else REAL_VARIANTS * REAL_STEPS,
                    )
                    try:
                        settings_for_job = real_settings[key]
                        result = load_cached_comparison(
                            real_dataset, settings_for_job, study="masking",
                        )
                        reused = result is not None and result.get("cache_reused") is True

                        def report_training(update):
                            local_training_progress.update_unit(
                                completed_steps=update["comparison_completed_steps"],
                                detail=(
                                    f"{update['variant'].replace('_', ' ')} · "
                                    f"update {update['step']}/{update['total_steps']} · "
                                    f"loss {update['loss']:.4f}"
                                ),
                            )

                        if result is None:
                            result = run_matched_comparison(
                                real_dataset,
                                settings_for_job,
                                progress_callback=report_training,
                            )
                            saved_to = save_comparison(result, study="masking")
                        else:
                            saved_to = result["artifact_path"]
                            local_training_progress.update_unit(
                                detail="Lineage and file digests validated",
                                completed_steps=0,
                                total_steps=0,
                            )
                    except BaseException as error:
                        local_training_progress.fail(error)
                        raise
                    else:
                        local_training_progress.complete_unit(reused=reused)
                    completed_runs += 1
                    action = "reused" if reused else "trained and saved"
                    print(f"Fold {fold}, seed {seed}: {action} {saved_to.name}")
                    for variant, run in result["runs"].items():
                        prediction_frames.append(run["predictions"].assign(
                            fold=fold, seed=seed, variant=variant,
                        ))
            real_predictions = pd.concat(prediction_frames, ignore_index=True)
            real_scores = summarize_cross_fitted_predictions(real_predictions)
            display(real_scores.round({"r2": 3, "mae": 3}))
            local_training_progress.complete(
                status=("Full local grid complete" if REAL_SCOPE == "full"
                        else "Single-seed pilot complete · not full evidence")
            )
        else:
            print("Real-data research experiment not run. All numerical outputs above are synthetic.")
        """),
        md("""
        ## Step 9 — Decide what the research result would support

        For each training seed, pool held-out predictions across source folds
        before computing the source-weighted metric. The preceding helper does
        this explicitly; averaging five fold R² values would mix denominators
        from different target distributions. Keep training-seed variation
        visible, and estimate data uncertainty with paired source-video
        resampling so all clips and recipes from a sampled source move
        together. Confidence intervals conditional on these fitted models do
        not include uncertainty from new training runs or alternative splits.

        A positive laterality-probe result would support a narrower claim that
        this target selection helps preserve useful signed movement information
        in this dataset. It would still need complementary movement tasks and
        independently validated endpoints to justify a broader health or
        foundation-model contribution. This notebook performs masked feature
        prediction within a clip; it does not evaluate forecasting. Notebook10
        addresses that separate world-model question with an explicit
        observed-past/future boundary.

        Useful prior work includes [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf),
        [Masked Motion Predictors](https://openaccess.thecvf.com/content/ICCV2023/html/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.html),
        and the body-part comparison in [SkeletonMAE](https://arxiv.org/html/2307.08476v1).
        Their different datasets and objectives should be described when
        positioning any new finding. None supplies evidence that our particular
        gait-landmark choice already improves learning.
        """),
    ])
