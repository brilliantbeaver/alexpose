"""Editable tutorial: structured target geometry and context-cue coverage."""
from nbformat.v4 import new_notebook
from .masking_shared import md, code, setup_cell


def build_notebook():
    return new_notebook(cells=[
        md('''
        # 16 — Structured masking and the context available for prediction

        We extend Notebook 11 by measuring which nearby observations a mask
        leaves visible. An intact trajectory removes same-joint temporal clues;
        a connected region also removes nearby landmarks. These are testable
        differences in available information. They do not establish how a JEPA
        actually solves its prediction task.

        [VideoMAE, NeurIPS 2022, §3.3](https://arxiv.org/html/2203.12602)
        repeats a spatial mask across video frames. Here the analogue hides a
        landmark's full trajectory. [I-JEPA, CVPR 2023, §3](https://arxiv.org/html/2301.08243v3)
        predicts several image regions using surrounding context; its geometry
        motivates comparison rather than a literal skeleton implementation.
        [SLiM, arXiv v3, §3.3](https://arxiv.org/html/2603.10648v3)
        masks connected anatomical subsets across consecutive time spans, varying
        duration inversely with spatial size. Our primary region comparison fixes
        region size and duration so its effect is easier to interpret. It does
        not reproduce SLiM's compact tokens, contrastive objective or full recipe.

        All examples are synthetic. Dependencies are NumPy, pandas and Matplotlib.
        [Notebook 17](17_motion_and_structure_pretraining.ipynb) runs these policies
        through a common JEPA objective.
        ''') ,
        setup_cell(),
        code('''
        from laterality_extensions.comparative_masks import (
            MaskPolicy, MaskBudget, InfeasibleMaskBudget, sample_mask, connected_region_bank,
        )
        from laterality_extensions.mask_figures import plot_mask_examples
        xyz = np.zeros((64, 33, 3))
        valid = np.ones((16, 33), dtype=bool)
        examples, records = [], []
        for experiment in ("regions", "trajectories", "completion"):
            arm = list(study_arms(experiment, 16).values())[1]
            structured = sample_study_mask(xyz, valid, arm, 1, np.random.default_rng(1601))
            reference = sample_study_mask(xyz, valid, StudyArm("uniform"),
                int(structured.mask.sum()), np.random.default_rng(1602))
            for label, result in ((arm.name, structured), (experiment + " scattered reference", reference)):
                examples.append((label.replace("_", " "), result))
                records.append({"condition": label, **context_cue_audit(result.mask, valid),
                                "fraction hidden": result.coverage["hidden_fraction"],
                                "longest hidden run": result.coverage["longest_observed_hidden_run_blocks"]})
        display(pd.DataFrame(records))
        fig = plot_mask_examples(examples, valid, title="Each structure has its own count-matched reference")
        display(fig); plt.close(fig)
        ''') ,
        md('''
        ## 1. Quantify local cues, with explicit limits

        The temporal-bracket fraction counts targets with the same landmark
        visible in both immediately adjacent time blocks. The neighbor fraction
        counts targets with at least one visible anatomical graph neighbor at
        the same time. Region definitions use the graph from Notebook 11 and
        collectively cover all 33 landmarks. Head-to-body links are declared
        masking connections, not newly measured bones.

        These counts audit an interpolation hypothesis. A contextualized teacher
        feature can require information beyond local coordinate interpolation.
        Hiding more cues can also make prediction ambiguous, especially for an
        asymmetric limb. The downstream test must determine whether this helps.
        ''') ,
        code('''
        regions = connected_region_bank(tuple(range(33)), 6)
        assert set().union(*map(set, regions)) == set(range(33))
        print(f"{len(regions)} declared connected regions collectively cover all 33 landmarks.")
        # With complete observations, intact trajectories require multiples of 16,
        # and whole-frame gaps require multiples of 33. No proper positive common count exists.
        common_counts = sorted(set(range(16, 528, 16)) & set(range(33, 528, 33)))
        assert common_counts == []
        try:
            sample_mask(xyz, valid, MaskPolicy("whole_trajectory"), MaskBudget(hidden_count=33),
                        np.random.default_rng(16))
        except InfeasibleMaskBudget as error:
            print("Declared infeasibility:", error)
        ''') ,
        md('''
        ## 2. Missing measurements change realized coverage

        A six-landmark interval of eight blocks nominally occupies 48 tokens.
        Naturally missing measurements reduce its supervised count. We keep the
        nominal region intact and sample the scattered reference at that exact
        realized count for the same clip. The training loss averages over each
        clip's targets before averaging clips, preserving identities when counts
        differ within the batch. Missing cells never become supervised targets.
        ''') ,
        code('''
        imperfect = valid.copy()
        imperfect[3:8, 27] = False
        observed = np.repeat(imperfect, 4, axis=0)
        imperfect_xyz = xyz.copy(); imperfect_xyz[~observed] = np.nan
        trajectory = sample_study_mask(imperfect_xyz, imperfect,
            StudyArm("whole_trajectory", trajectories=3), 1, np.random.default_rng(16),
            observation_valid=observed)
        reference = sample_study_mask(imperfect_xyz, imperfect, StudyArm("uniform"),
            int(trajectory.mask.sum()), np.random.default_rng(17), observation_valid=observed)
        assert trajectory.mask.sum() == reference.mask.sum()
        assert not (trajectory.mask & ~imperfect).any()
        display(pd.DataFrame([trajectory.coverage, reference.coverage])[
            ["valid_tokens", "hidden_tokens", "context_tokens", "naturally_missing_tokens"]])
        ''') ,
        md('''
        ## 3. Keep the next experiment focused

        The primary new training plan compares motion rules and connected regions.
        Whole trajectories are a prespecified follow-up that separates missing
        one trajectory from missing a neighboring group. Interior temporal gaps
        are a separate completion comparison: observations on both sides remain
        visible. Notebook 14 supplies the stricter past-only forecasting pathway.

        Do not combine these tasks by imposing one infeasible hidden count, or
        select whichever condition happens to score best on the outer test.
        A favorable result on this already inspected cohort is development evidence.
        ''')
    ])
