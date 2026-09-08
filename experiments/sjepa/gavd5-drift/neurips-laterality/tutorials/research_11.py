"""Editable tutorial: what different masking policies hide and leave visible."""

from textwrap import dedent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    def md(text):
        return new_markdown_cell(dedent(text).strip())

    def code(text):
        return new_code_cell(dedent(text).strip())

    return new_notebook(cells=[
        md("""
        # 11 — Which missing observations should a movement encoder learn to predict?

        A brief ankle gap leaves nearby observations of the same ankle available.
        Hiding its complete trajectory removes those clues. These are different
        learning problems even when both hide the same number of values, and a
        useful comparison must describe the observations available to solve each.

        This notebook constructs and checks eight masking policies before model
        training. It needs NumPy, pandas, and Matplotlib; every example uses small
        generated arrays. The output verifies definitions and counts, with no
        empirical claim about human gait. Notebook 12 uses these masks to train
        paired encoders, and Notebook 13 tests their features and predictions on
        recordings excluded from training. Notebook 14 handles future prediction.

        Notebook 08 already completed a gait-only versus all-landmark target
        comparison. Both encoders received all 33 input landmarks. The extension
        asks whether broader target eligibility, movement-weighted selection, or
        longer connected gaps produce a more informative learning task.
        """),
        md("""
        ## 1. Separate the questions before choosing a mask

        | Question | What changes | What is kept fixed |
        |---|---|---|
        | Which landmarks can become targets? | Gait-only, all landmarks, fixed random twelve-landmark sets, or a soft gait preference | Scattered geometry and the realized hidden count |
        | Does the pattern of missing information matter? | Scattered, movement-weighted, whole trajectories, or connected regions | Declared landmark pool; each structure has its own count-matched scattered reference |
        | Does learning to continue movement help? | Interior completion versus future prediction | A separately declared information boundary and future endpoints |
        | Does another representation choice help? | Input landmarks, feature summaries, regularizer pooling, predictor size, or loss | These require separate comparisons |

        For the first masking comparisons, all 33 landmarks remain encoder inputs.
        The regularizer still pools twelve gait landmarks from unmasked views,
        and the laterality readout still uses five bilateral pairs. All-landmark
        target eligibility therefore retains anatomical choices elsewhere in the
        pipeline. The exact twelve-landmark set has not been clinically validated.
        """),
        code("""
        from pathlib import Path
        import sys
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from IPython.display import display
        from matplotlib_inline.backend_inline import set_matplotlib_formats

        def locate_suite():
            for parent in (Path.cwd(), *Path.cwd().parents):
                for candidate in (parent, parent / "neurips-laterality"):
                    if (candidate / "laterality_extensions" / "comparative_masks.py").is_file():
                        return candidate.resolve()
            raise FileNotFoundError("Open the notebook from the research project.")

        SUITE_ROOT = locate_suite()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))
        from laterality_extensions.comparative_masks import (
            GAIT_JOINTS, LANDMARK_NAMES, InfeasibleMaskBudget, MaskPolicy, MaskBudget,
            sample_mask, fixed_subset, shared_scattered_count, connected_region_bank,
            motion_scores, reflect_sample,
        )
        from laterality_extensions.mask_figures import plot_mask_examples
        set_matplotlib_formats("svg")
        pd.set_option("display.precision", 3)
        print("Generated examples: mask construction only; no real-data training.")
        """),
        md("""
        ## 2. Count observed tokens before counting deliberate masks

        One token contains four prepared time steps of one landmark. A token is
        valid only when all four observations are valid. If an ankle observation
        was never measured, its token cannot become a supervised target. Removing
        an observed ankle deliberately for pretraining is a different operation.

        The example below has 24 prepared steps, hence six time blocks. It makes
        the left ankle move faster than the right ankle and includes one missing
        wrist observation. These settings make the sampling rules easy to inspect;
        their movement amplitudes are teaching choices with no clinical meaning.
        """),
        code("""
        prepared_steps = 24
        time = np.arange(prepared_steps, dtype=float)
        xyz = np.zeros((prepared_steps, 33, 3))
        xyz[:, :, 1] = np.arange(33) / 33
        xyz[:, 27, 0] = time * 0.10
        xyz[:, 28, 0] = time * 0.02
        observed = np.ones((prepared_steps, 33), dtype=bool)
        observed[6, 15] = False
        xyz[6, 15] = np.nan
        token_valid = observed.reshape(6, 4, 33).all(axis=1)
        display(pd.DataFrame([{
            "potential tokens": token_valid.size,
            "valid tokens": int(token_valid.sum()),
            "naturally missing tokens": int((~token_valid).sum()),
        }]))
        assert not token_valid[1, 15]
        """),
        md("""
        ## 3. Compare target eligibility at a shared scattered budget

        The gait policy draws fresh joint–time targets from shoulders, hips,
        knees, ankles, heels, and foot tips. Uniform masking draws from every
        valid landmark. A fixed random subset uses its own selection seed so the
        anatomical list stays unchanged across masks, folds, and training seeds.

        Predeclare several random sets and report their group result. Random
        individual landmarks can include the nose and unpaired points; selecting
        six bilateral pairs preserves pair structure and is a separate control.
        Neither control should be chosen after inspecting prediction scores.

        The soft preference assigns half its initial probability mass through
        uniform sampling over all candidates and half through the gait pool.
        Sampling without replacement does not guarantee that half the final
        targets belong to each component. With no valid gait candidates, the
        sampler uses uniform probabilities. The mixture weight is a proposed
        setting, with no evidence that it is optimal.
        """),
        code("""
        subset_selection_seeds = (31, 47, 59)
        eligibility_policies = [MaskPolicy("gait"), MaskPolicy("uniform")]
        eligibility_policies += [
            MaskPolicy("random_subset", subset_seed=seed)
            for seed in subset_selection_seeds
        ]
        eligibility_policies.append(MaskPolicy("soft_gait", gait_weight=0.5))
        common_count = shared_scattered_count(token_valid, eligibility_policies, requested_count=18)
        budget = MaskBudget(hidden_count=common_count)
        eligibility_results = [
            sample_mask(xyz, token_valid, policy, budget, np.random.default_rng(12),
                        observation_valid=observed)
            for policy in eligibility_policies
        ]
        rows = []
        for policy, result in zip(eligibility_policies, eligibility_results):
            label = policy.name.replace("_", " ")
            if policy.name == "random_subset":
                label = f"fixed random individual set, selection seed {policy.subset_seed}"
            rows.append({"target eligibility": label, "eligible landmarks": len(result.coverage["eligible_landmarks"]),
                         "hidden": result.coverage["hidden_tokens"], "context": result.context_count})
        display(pd.DataFrame(rows))
        assert len({result.coverage["hidden_tokens"] for result in eligibility_results}) == 1
        """),
        code("""
        display(pd.DataFrame({
            "selection seed": subset_selection_seeds,
            "fixed twelve-landmark identities": [
                ", ".join(LANDMARK_NAMES[joint] for joint in fixed_subset(MaskPolicy("random_subset", subset_seed=seed)))
                for seed in subset_selection_seeds
            ],
        }))
        """),
        md("""
        The rows have equal target counts and different candidate pools. This
        checks the eligibility comparison; it does not show that any pool is
        useful for a held-out movement endpoint. If adding another subset lowers
        the feasible count, repeat every reference at that newly declared count.

        ## 4. Change mask shape while preserving its definition

        Whole trajectories hide selected landmarks across the complete clip,
        wherever observations are valid. Connected regions hide an anatomical
        group for a consecutive interval. The region bank grows along explicit
        anatomical connections from every eligible landmark and includes reflected
        counterparts. The complete bank covers the candidate pool, including the
        face and hands. Head-to-mouth and ear-to-shoulder connections are declared
        masking adjacency links; they are not measured bones.

        An interior temporal gap hides every valid landmark during a missing
        interval while retaining observations before and after it. This tests
        completion. A model supplied with those later observations cannot be
        interpreted as forecasting from the past alone.

        Each structured mask below has a uniform scattered reference with its
        own realized count. The trajectory and connected-region counts need not
        match the temporal-gap count, and their effects cannot be ranked as an
        identical-budget experiment.
        """),
        code("""
        structured_specs = [
            ("Whole trajectories", MaskPolicy("whole_trajectory"), MaskBudget(trajectories=3)),
            ("Connected region interval", MaskPolicy("connected_region"), MaskBudget(region_size=3, interval_blocks=3)),
            ("Interior temporal gap", MaskPolicy("temporal_gap"), MaskBudget(interval_blocks=2)),
        ]
        structured_results = []
        comparisons = []
        for label, policy, shape_budget in structured_specs:
            result = sample_mask(xyz, token_valid, policy, shape_budget, np.random.default_rng(21),
                                 observation_valid=observed)
            reference = sample_mask(xyz, token_valid, MaskPolicy("uniform"),
                                    MaskBudget(hidden_count=result.coverage["hidden_tokens"]),
                                    np.random.default_rng(22), observation_valid=observed)
            assert result.coverage["hidden_tokens"] == reference.coverage["hidden_tokens"]
            structured_results.append((label, result))
            comparisons.append({"structured task": label,
                                "structured targets": result.coverage["hidden_tokens"],
                                "scattered reference targets": reference.coverage["hidden_tokens"],
                                "covered landmarks": result.coverage["landmark_count"],
                                "declared interval blocks": result.coverage["interval_blocks"],
                                "left targets": result.coverage["left_hidden_tokens"],
                                "right targets": result.coverage["right_hidden_tokens"]})
        display(pd.DataFrame(comparisons))
        """),
        code("""
        examples = [("Scattered all-landmark targets", eligibility_results[1]), *structured_results]
        figure = plot_mask_examples(examples, token_valid)
        display(figure)
        # This teaching notebook saves vector illustrations in a new output area.
        figure_dir = SUITE_ROOT / "outputs" / "comparative_masking" / "teaching_figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_dir / "masking_patterns.svg", bbox_inches="tight")
        figure.savefig(figure_dir / "masking_patterns.pdf", bbox_inches="tight")
        plt.close(figure)
        """),
        md("""
        Blue cells show deliberate targets; pale cells are observed context and
        grey cells were naturally missing. The whole-trajectory mask removes
        immediate observations of the selected landmark throughout the window.
        The region mask removes neighboring body observations together, while
        the temporal gap keeps context on both sides of time. Greater difficulty
        may help learning, or may remove too much information for the endpoint.

        Tube masking in [VideoMAE](https://arxiv.org/abs/2203.12602) motivates
        withholding the same spatial position throughout time. Skeleton-specific
        connected-region masking appears in [SLiM](https://arxiv.org/html/2603.10648v3),
        which also changes other training components. Our masks are explicit
        comparisons within the current learning recipe, rather than reproductions
        of those complete methods or new inventions of their mask families.
        """),
        md("""
        ## 5. Show an impossible count before it becomes a training failure

        A fully observed 64-step input supplies 16 time blocks and 33 landmarks.
        A whole trajectory contains 16 tokens, so six trajectories hide 96. A
        complete time block contains 33 tokens; no integer number of complete
        time blocks hides 96. In fact, the two families share no positive count
        below the entire 528-token input, which would leave no context.

        The sampler therefore raises a clear error for an impossible exact
        count. It never shortens a trajectory or adds isolated cells to make
        the arithmetic work. Real missingness can make whole-trajectory counts
        unequal across clips; Notebook 12 preserves each clip's target indices
        and averages its target loss before averaging examples.
        """),
        code("""
        complete_xyz = np.zeros((64, 33, 3))
        complete_valid = np.ones((16, 33), dtype=bool)
        six_trajectories = sample_mask(
            complete_xyz, complete_valid, MaskPolicy("whole_trajectory"),
            MaskBudget(hidden_count=96), np.random.default_rng(5),
        )
        print("Six complete trajectories hide", six_trajectories.coverage["hidden_tokens"], "tokens.")
        try:
            sample_mask(complete_xyz, complete_valid, MaskPolicy("temporal_gap"),
                        MaskBudget(interval_blocks=3, hidden_count=96), np.random.default_rng(5))
        except InfeasibleMaskBudget as error:
            print("Expected impossible-budget example:", error)
        trajectory_counts = {16 * number for number in range(1, 33)}
        complete_block_counts = {33 * number for number in range(1, 16)}
        assert not trajectory_counts.intersection(complete_block_counts)
        """),
        md("""
        ## 6. Let reliable motion influence sampling without excluding slower regions

        [MAMP](https://arxiv.org/html/2308.07092) gives greater masking probability
        to tokens with larger motion, and
        [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
        adopts that strategy. Here we keep the JEPA feature target and implement a
        declared sampling adaptation suited to missing and noisy pose observations.

        Movement spans four prepared steps. For each token, we divide displacement
        by elapsed time and take the median of its valid transition speeds. At
        least two transitions are required. The first block reuses the second
        block's score where both blocks are valid. With supplied timestamps, speed
        is expressed per second; otherwise it is per prepared time step. Prepared
        steps are not assumed to have a fixed duration in real recordings.

        The median suppresses an isolated one-frame tracking jump. We cap positive
        scores at their 95th percentile, normalize them to sum to one, and mix
        75% motion probability with 25% uniform probability. Stationary or unusable
        motion falls back to uniform sampling. These safeguards and mixture values
        are proposed settings, rather than the original MAMP formula or validated
        clinical parameters. Slower limbs retain a chance to become targets.
        """),
        code("""
        motion_policy = MaskPolicy("motion", motion_weight=0.75, motion_clip_quantile=0.95)
        scores, motion_note = motion_scores(xyz, token_valid, observation_valid=observed)
        rng = np.random.default_rng(42)
        selection_counts = np.zeros_like(token_valid, dtype=int)
        for _ in range(200):
            result = sample_mask(xyz, token_valid, motion_policy, MaskBudget(hidden_count=1), rng,
                                 observation_valid=observed)
            selection_counts += result.mask
        display(pd.DataFrame({
            "landmark": ["left ankle: greater generated movement", "right ankle: smaller generated movement", "left elbow: stationary"],
            "times selected in 200 draws": [selection_counts[:, joint].sum() for joint in (27, 28, 13)],
            "mean motion score": [scores[:, joint].mean() for joint in (27, 28, 13)],
        }))
        print("Motion units:", motion_note["motion_units"])
        """),
        md("""
        This repeated-sampling example checks whether the configured rule favors
        greater reliable movement. It provides no evidence that a trained encoder
        preserves more useful gait information. A slower limb may be essential to
        an asymmetric movement score, so the prediction comparison must retain
        that shared observable outcome.

        Motion-based selection uses the complete permitted input during ordinary
        self-supervised masking. A future-prediction input has a stricter boundary:
        its motion scores must use only the observed prefix. Notebook 14 tests that
        changing later observations cannot alter the prepared context or forecast.
        """),
        md("""
        ## 7. Reflect observations and masks together

        Reflection reverses the horizontal coordinate and exchanges left and right
        landmark identities. The same exchange must apply to natural validity and
        deliberate target locations. Otherwise a comparison would hide different
        anatomical observations while calling the inputs corresponding reflections.
        The check below applies the transformation twice and recovers the original
        arrays, including missing values.
        """),
        code("""
        mask = eligibility_results[0].mask
        mirrored = reflect_sample(xyz, token_valid, mask)
        restored = reflect_sample(*mirrored)
        for original, returned in zip((xyz, token_valid, mask), restored):
            np.testing.assert_equal(original, returned)
        np.testing.assert_array_equal(mirrored[2][:, 28], mask[:, 27])
        print("Reflection preserved corresponding observations, validity, and targets.")
        print("Connected region bank covers", len(set().union(*map(set, connected_region_bank(tuple(range(33)), 3)))), "landmarks.")
        """),
        md("""
        ## What these examples establish

        The implementation constructs reproducible, label-blind masks, retains
        intact structured definitions, and records the amount and location of
        remaining information. The examples also expose budgets that cannot be
        matched and natural missingness that prevents supervision.

        Useful representation learning remains an empirical question. Notebook 12
        pairs training conditions while keeping model and data exposure fixed;
        Notebook 13 compares hidden-feature prediction with a common movement
        endpoint. A lower feature loss without improved movement prediction would
        weaken the hypothesis that the new missing-information task preserves more
        useful movement differences. No real-data training is run in this notebook.
        """),
    ])
