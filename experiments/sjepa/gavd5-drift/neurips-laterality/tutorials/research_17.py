"""Editable tutorial: paired motion/structure JEPA training and real-data planning."""
from nbformat.v4 import new_notebook
from .masking_shared import md, code, setup_cell


def build_notebook():
    return new_notebook(cells=[
        md('''
        # 17 — Controlled pretraining with motion and structured masks

        This notebook connects the samplers in [15](15_motion_weighted_masking.ipynb)
        and [16](16_structured_masking_and_context.ipynb) to the existing S-JEPA
        model and objective. Default execution trains five small encoders for
        two updates each on generated coordinates. The real-data path is
        explicitly enabled and displays its workload before training.

        ## 1. Predeclare two questions

        | Experiment | Arms | Shared control |
        |---|---|---|
        | Motion | Uniform, MAMP code convention, median/clipped motion mixture | Same hidden count in each batch, all-landmark eligibility |
        | Regions | Connected six-landmark half-window interval, uniform reference | Same realized hidden count per clip |

        Three full trajectories and an interior time gap are separate, optional
        follow-ups. They each require their own scattered reference. All arms
        retain 33-landmark input, four-step tokens, the existing twelve-landmark
        regularizer pool, and the five bilateral readout pairs. These are masking
        comparisons within our gait adaptation, not paper reproductions.

        The objective remains centered teacher-feature cross-entropy plus the
        existing feature-variation regularizer. Every clip contributes its mean
        hidden-token loss. The teacher receives no gradient and follows the online
        encoder by EMA. The unmasked regularizer remains part of every arm.
        ''') ,
        setup_cell(),
        code('''
        import torch
        from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
        from laterality_extensions.motion_structured_training import (
            train_mask_study, plan_mask_study, run_mask_study,
        )
        data = load_learning_dataset()
        settings = LearningSettings(steps=2, batch_size=5)
        runs = {experiment: train_mask_study(data, settings, experiment=experiment, checkpoint_steps=(1,))
                for experiment in ("motion", "regions")}
        display(pd.DataFrame({e: result["pairing"] for e, result in runs.items()}))
        assert all(all(result["pairing"].values()) for result in runs.values())
        ''') ,
        md('''
        ## 2. Check paired exposure before looking at losses

        Initialization, source schedules and geometric views use separate random
        streams. Mask draws use an arm-specific stream. Changing how many random
        numbers one sampler consumes cannot change which clips another arm sees.
        All scheduled masks are checked before optimization begins. A structure
        that cannot leave a valid target and context fails explicitly.

        The source schedule uses outer-training videos only. Checkpoints retain
        the online encoder, EMA teacher, predictor and target center together.
        Completed compatible jobs are saved atomically and can be reused.
        Interrupted jobs restart from their seed; this new runner does not claim
        optimizer-resume equivalence. Existing Notebook 12 caches are preserved.
        ''') ,
        code('''
        histories = pd.concat([run["history"].assign(experiment=e, condition=name)
            for e, result in runs.items() for name, run in result["runs"].items()], ignore_index=True)
        display(histories[["experiment", "condition", "step", "hidden_count_min", "hidden_count_max"]])
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.3), constrained_layout=True)
        for ax, (experiment, group) in zip(axes, histories.groupby("experiment", sort=False)):
            for condition, values in group.groupby("condition", sort=False):
                ax.plot(values.step, values.masked_prediction_loss, marker="o", label=condition.replace("_", " "))
            ax.set(title=f"{experiment}: two synthetic updates", xlabel="Update", ylabel="Own-teacher loss")
            ax.legend(fontsize=8)
        display(fig); plt.close(fig)
        ''') ,
        md('''
        These traces check finite optimization and retained exposure. Each arm's
        teacher evolves differently, so loss values do not share a fixed target
        scale. Their ordering cannot rank useful movement learning. Notebook 18
        tests initial and trained representations with the same frozen readouts.

        ## 3. Inspect the real-data recipe without starting it

        The tracked numerical summary from Notebook 08 records the starting
        recipe: 1,200 updates, batch 20, width 96, four encoder layers, two predictor
        layers and four attention heads. The plan reads all its settings, rather
        than inheriting the tiny teaching configuration. Raw local checkpoints
        are not required to display this plan. Their absence must not be mistaken
        for absence of the completed evidence described in the tutorial.

        Five outer video groups and five seeds produce 75 encoders for the motion
        comparison and 50 for the region comparison: 125 encoders and 150,000
        optimizer updates. A selected fold/seed subset is a pilot. GAVD has already
        informed these hypotheses, so the full plan remains development evidence.
        ''') ,
        code('''
        VALIDATE_REAL_INPUTS = os.getenv("LATERALITY_MOTION_VALIDATE_REAL", "0") == "1"
        RUN_REAL_TRAINING = os.getenv("LATERALITY_MOTION_RUN_REAL", "0") == "1"
        REAL_EXPERIMENTS = tuple(os.getenv("LATERALITY_MOTION_EXPERIMENTS", "motion,regions").split(","))
        real_plan = plan_mask_study(experiments=REAL_EXPERIMENTS,
            device=os.getenv("LATERALITY_DEVICE", "auto"),
            validate_inputs=VALIDATE_REAL_INPUTS or RUN_REAL_TRAINING)
        display(pd.Series({key: real_plan[key] for key in
            ("scope", "training_runs", "optimizer_updates", "recipe_source", "output_dir")}, name="Workload"))
        display(pd.Series(real_plan["settings"], name="Full starting configuration"))
        display(real_plan["workload"].groupby("experiment").agg(
            encoders=("condition", "size"), updates=("updates", "sum")))
        display(pd.DataFrame([{"experiment": e, **arm} for e, arms in real_plan["arms"].items()
                              for arm in arms.values()]))
        if not real_plan["input_checks"].empty:
            display(real_plan["input_checks"])
        # The complete job table remains available as real_plan["workload"].
        real_result = run_mask_study(real_plan, enabled=RUN_REAL_TRAINING)
        print(real_result["status"])
        ''') ,
        md('''
        Real runs require the already prepared cohort and source-split artifacts.
        Validation reports missing prerequisites directly; it does not download
        data, rebuild the registered cohort, or substitute synthetic examples.
        All artifacts go under `artifacts/motion_structured`.

        The runner uses AdamW with betas (0.9, 0.95), constant learning rate and
        teacher momentum, target/predictor temperatures 0.06/0.10, center momentum
        0.9, gradient clipping at 1, rotation up to eight degrees, and translation
        up to 0.03 prepared-coordinate units. CPU-generated geometric views are
        shared across arms; the selected CPU, CUDA or MPS device runs the model.
        Report wall time as well as updates: the dense path and geometry audit
        add costs, and this implementation need not match old runtime.

        ## 4. Use outcomes to decide the next step

        The real runner saves every clip prediction, training-source ridge
        selection, feature-variation diagnostic and complete training job. It
        pools outer-fold predictions per seed before calculating source-balanced
        metrics. Source-bootstrap contrasts retain paired conditions and seeds
        and condition on fitted models; seed variation is reported separately.

        A motion-sensitive readout that helps trained and initial encoders equally
        improves evaluation without establishing a pretraining gain. A repeated
        trained-over-initial advantage would support useful learning. Broader
        masking mixtures, loss changes and future decoding follow only after
        these simpler contrasts become informative. The endpoint remains a
        coordinate-derived movement quantity, with no new clinical claim.
        ''')
    ])
