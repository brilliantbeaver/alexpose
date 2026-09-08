"""Editable source for the controlled comparative masking tutorial."""
from textwrap import dedent
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def build_notebook():
    def md(text): return new_markdown_cell(dedent(text).strip())
    def code(text): return new_code_cell(dedent(text).strip())
    return new_notebook(cells=[
        md("""
        # 12 — Training encoders with a fair masking comparison

        Which missing observations help a JEPA preserve useful movement? Notebook
        11 defines where the observations are hidden. This notebook trains paired
        encoders while holding their other training choices constant. Notebook 13
        then measures how well the resulting features support movement prediction.

        Default execution uses generated poses and three optimizer updates per
        condition. These outputs demonstrate the software and have no empirical
        interpretation about human gait. The real-data section displays the
        retained training recipe and a complete workload before its training
        switch can be enabled. See [the experiment specification](docs/COMPARATIVE_MASKING_PLAN.md)
        for the questions assigned to each comparison.
        """),
        md("""
        ## 1. Keep the comparison focused

        First compare where scattered targets are eligible: twelve gait landmarks,
        all 33 landmarks, three preselected random sets, or a soft gait preference.
        Every condition receives the same number of hidden observations. Random
        sets are selected before examining outcomes and should be reported together.

        Next compare a structured mask with a scattered reference at its realized
        count. For example, hiding two entire joint trajectories and hiding the
        same number of scattered tokens create different prediction tasks. The
        regularizer continues to pool the twelve gait landmarks, and all 33
        landmarks remain input to the encoder. Those anatomical choices are common
        to the compared methods.
        """),
        code("""
        from pathlib import Path
        import os
        import sys
        from dataclasses import asdict, replace
        import tempfile
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        from IPython import get_ipython
        get_ipython().run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("svg", "png")
        candidates = [Path.cwd(), *Path.cwd().parents]
        SUITE_ROOT = next(p if (p / "laterality").is_dir() else p / "neurips-laterality"
            for p in candidates if (p / "laterality").is_dir() or (p / "neurips-laterality/laterality").is_dir())
        sys.path.insert(0, str(SUITE_ROOT))
        from laterality_extensions.masked_learning import LearningSettings, load_learning_dataset
        from laterality_extensions.comparative_masks import MaskPolicy, MaskBudget
        from laterality_extensions.comparative_training import (
            train_comparison, default_conditions, anatomical_conditions, masks_for_batch,
            saved_reference_recipe, plan_real_comparison, run_real_comparison,
        )
        from notebook_progress import (
            NotebookTaskProgress, run_real_masking_comparison_with_progress,
        )
        data = load_learning_dataset()
        settings = LearningSettings(steps=3, batch_size=5)
        print("Synthetic demonstration: generated movement, three updates per encoder.")
        display(pd.DataFrame([{"training sources": len(data.train_sources),
            "test sources": len(data.test_sources), "generated clips": len(data.xyz)}]))
        """),
        md("""
        ## 2. Inspect actual hidden counts before training

        One token contains four consecutive samples of one landmark. A requested
        fraction of the gait pool differs from that fraction of the whole input.
        The helper chooses a count feasible for every example and policy in the
        batch, leaving visible context. A smaller common budget also applies to
        its reference conditions.

        The automatic scattered budget retains Notebook 08's fraction of the
        available gait pool, even when both policies can hide any landmark.
        This preserves the reference exposure while keeping an anatomical
        influence on the budget. An explicit shared hidden count can test
        other exposures as a separately declared comparison.

        The following example checks several fixed random pools as well as the
        two original target choices. It only constructs masks. The deliberately
        small training example below uses the two original choices.
        """),
        code("""
        conditions = anatomical_conditions()
        masks, coverage = masks_for_batch(data, data.train_rows[:5], settings, conditions)
        display(pd.DataFrame([{"condition": name,
            "smallest_hidden_count": int(mask.sum((1, 2)).min()),
            "largest_hidden_count": int(mask.sum((1, 2)).max())}
            for name, mask in masks.items()]))
        assert len({tuple(mask.sum((1, 2))) for mask in masks.values()}) == 1
        """),
        md("""
        ## 3. Pair the training choices that could otherwise explain a difference

        Each condition starts with identical encoder, teacher, predictor, and
        regularizer weights. The source-video schedule and geometric views are
        shared. Policy-specific random streams generate masks independently, so
        a sampler's additional random draws cannot change its training clips.

        The student predicts features supplied by an unmasked teacher. Its loss
        averages hidden targets within each clip and then averages clips. This
        preserves the comparison when missing measurements give structured masks
        different counts in different clips. The target center uses the same clip
        weighting. Invalid measurements never become targets.
        """),
        code("""
        comparison = train_comparison(data, settings, default_conditions(), checkpoint_steps=(1, 3))
        display(pd.DataFrame([comparison["pairing"]]))
        assert all(comparison["pairing"].values())
        history = pd.concat([run["history"].assign(condition=name)
            for name, run in comparison["runs"].items()], ignore_index=True)
        fig, ax = plt.subplots(figsize=(6.2, 3.1))
        labels = {"gait_targets": "Gait targets", "all_landmark_targets": "Targets across the body"}
        for name, group in history.groupby("condition", sort=False):
            ax.plot(group.step, group.masked_prediction_loss, marker="o", label=labels[name])
        ax.set(xlabel="Optimizer update", ylabel="Feature-prediction training loss",
               title="Synthetic execution: optimization check")
        ax.set_xticks([1, 2, 3]); ax.legend(frameon=False); fig.tight_layout()
        display(fig); plt.close(fig)
        """),
        md("""
        Three updates establish that the learning path executes. Even a falling
        loss would leave the movement question open: the teacher also changes
        during training, and less variable teacher features can be easier to
        predict. Notebook 13 evaluates a shared movement outcome and compares
        both trained encoders with their initial states.
        """),
        md("""
        ## 4. Match each intact structure with its own reference

        A complete trajectory stays hidden throughout the window. Missing
        measurements may lower its count without shortening its temporal span.
        The scattered reference below matches that actual count separately in
        every clip. Both models receive the same clip weight in the loss. The two
        arms are interleaved, so the displayed wall time describes their paired
        job and should not be added across condition rows.

        A temporal gap cannot always share that count. With 16 time blocks and
        33 visible landmarks, trajectories add 16 tokens at a time and complete
        time blocks add 33. Their only common positive count hides the whole
        input. These families therefore need separate matched references.
        """),
        code("""
        shape_conditions = {"whole_trajectories": MaskPolicy("whole_trajectory"),
                            "scattered_reference": MaskPolicy("uniform")}
        shape_budgets = {"whole_trajectories": MaskBudget(trajectories=2)}
        shape_example = train_comparison(data, replace(settings, steps=2), shape_conditions,
            budgets=shape_budgets, matched_to="whole_trajectories")
        display(pd.DataFrame([shape_example["pairing"]]))
        display(pd.DataFrame([{"condition": name, "updates": len(run["history"]),
            "paired_job_wall_seconds": run["elapsed_training_seconds"]}
            for name, run in shape_example["runs"].items()]).round(2))
        """),
        md("""
        ## 5. Retain compatible results without overwriting them

        A saved comparison includes configuration, data and source partitions,
        mask coverage, source schedules, losses, and declared encoder checkpoints.
        Reuse verifies both compatibility and file contents. During a long
        real-data job, both arms are checkpointed at the same update boundary.
        If the kernel stops, the next identical request resumes from the last
        verified boundary instead of discarding all work for that job.

        This short temporary-directory example demonstrates reuse. Real runs use
        a dedicated output directory, and their readout predictions are retained
        separately from training checkpoints.
        """),
        code("""
        with tempfile.TemporaryDirectory(prefix="masking_teaching_") as temporary:
            first = train_comparison(data, replace(settings, steps=1), output_dir=temporary)
            repeated = train_comparison(data, replace(settings, steps=1), output_dir=temporary)
            print("Completed compatible comparison reused:", repeated["reused"])
        """),
        md("""
        ## 6. Inspect and optionally run the real-data grid

        The settings below are read from all 25 retained Notebook 08 comparisons.
        Their agreement is checked before the reference is returned. The recipe
        has a constant learning rate and teacher momentum; it is not the small
        synthetic setup used above. The original [S-JEPA paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
        provides the feature-prediction motivation, while our comparative training
        retains the current project's architecture and regularizer.

        The full two-condition plan contains 50 encoders and 60,000 optimizer
        updates across five outer video groups and five seeds. Input validation
        reads the local prepared data and checks mask feasibility without training.

        The runner uses PyTorch's native CUDA backend on an NVIDIA GPU or its MPS
        backend on an Apple GPU. It keeps each fold's poses, validity masks, and
        presampled target masks on that device; generates one pair of augmented
        views for both masking arms; and reduces feature tensors before copying
        summaries back to the CPU. On a single GPU, fold/seed jobs run one at a
        time so that concurrent models do not compete for accelerator memory.
        Tensor operations within each job are already batched and parallel.

        MLX is not selected automatically. It is not a drop-in execution backend
        for this PyTorch model, optimizer, and checkpoint format; a native MLX
        port would be a separate numerical implementation that needs its own
        equivalence study. MPS therefore provides the valid Metal-accelerated path
        for this comparison today.

        Set `LATERALITY_RESEARCH_RUN_REAL=1` to enable training and optionally set
        `LATERALITY_DEVICE` to `mps`, `cuda`, or `cpu` (otherwise `auto` is used). The log
        first explains the total work, hardware, memory strategy, and recovery
        policy. A single updating `notebook_progress.py` display then shows the
        active fold, seed, masking arm, optimizer position, latest sampled loss,
        elapsed time, and adaptive ETA. The ordinary text log still reports only
        one start and one plain-language result per fold/seed job instead of
        printing the 50-row workload or every optimizer update. Complete jobs are
        reused. Interrupted jobs save both arms together
        every 300 updates per arm. Because each recovery file contains both models
        and both optimizer states, this interval cuts checkpoint traffic by two
        thirds relative to saving every 100 updates while limiting repeated work
        after an interruption to at most 300 updates per arm.

        The working notebook retains its previously inspected outputs. Those
        historical lines can still show the older verbose runner and its
        interruption; rerun the next cell to replace them with the concise log
        produced by the current source. They are not used as current results.
        """),
        code("""
        recipe = saved_reference_recipe()
        display(pd.DataFrame([
            {
                "part of the run": "Exposure",
                "chosen setting": f"{recipe['steps']:,} updates × batch {recipe['batch_size']} per encoder",
                "reason": "Matches the retained Notebook 08 training budget",
            },
            {
                "part of the run": "Encoder",
                "chosen setting": (
                    f"{recipe['embed_dim']}-D; encoder depth {recipe['encoder_depth']}; "
                    f"predictor depth {recipe['predictor_depth']}; {recipe['heads']} heads"
                ),
                "reason": "Keeps architecture fixed so only the masking policy changes",
            },
            {
                "part of the run": "Optimization",
                "chosen setting": (
                    f"{recipe['optimizer']}; learning rate {recipe['learning_rate']}; "
                    f"weight decay {recipe['weight_decay']}; teacher momentum {recipe['ema_momentum']}"
                ),
                "reason": "Uses the same optimizer recipe for both independent arms",
            },
        ]))
        VALIDATE_REAL_INPUTS = os.getenv("LATERALITY_RESEARCH_VALIDATE_REAL", "1") == "1"
        RUN_REAL_TRAINING = os.getenv("LATERALITY_RESEARCH_RUN_REAL", "0") == "1"
        REAL_DEVICE = os.getenv("LATERALITY_DEVICE", "auto")
        real_plan = plan_real_comparison(conditions=default_conditions(),
            validate_inputs=VALIDATE_REAL_INPUTS, device=REAL_DEVICE)
        display(pd.DataFrame([{
            "scope": real_plan["scope"],
            "fold/seed comparisons": len(real_plan["folds"]) * len(real_plan["seeds"]),
            "encoder arms": real_plan["training_runs"],
            "updates": real_plan["optimizer_updates"],
            "clip presentations": real_plan["sample_presentations"],
            "requested device": REAL_DEVICE,
            "real training enabled": RUN_REAL_TRAINING,
        }]))
        if len(real_plan["input_checks"]):
            display(real_plan["input_checks"])
        if RUN_REAL_TRAINING:
            real_training_progress = NotebookTaskProgress(
                "Real controlled-masking experiment",
                "stage",
                refresh_seconds=0.5,
            )
            real_result = run_real_masking_comparison_with_progress(
                real_plan,
                progress=real_training_progress,
                progress_interval=100,
                resume_interval=300,
            )
        else:
            real_result = run_real_comparison(
                real_plan,
                enabled=False,
                resume_interval=300,
            )
        """),
        md("""
        ## 7. Decide what the comparison would establish

        Source-separated inner validation can select a readout penalty or check
        feasible settings before a declared full comparison. If choosing a
        pretraining recipe, inner validation videos must also be excluded from
        that candidate encoder's training. Using representations pretrained on
        the outer training set to tune only the readout is a different procedure.

        Better feature prediction becomes useful evidence when frozen features
        also support better movement prediction on excluded videos. A benefit
        limited to a specific missing-region test would support that observation
        condition. Similar or poorer movement results are valid outcomes and
        should retain the same baselines and evaluated recordings. Continue to
        [Notebook 13](13_masking_encoder_and_predictor_evaluation.ipynb) to make
        these distinctions measurable.
        """),
    ])
