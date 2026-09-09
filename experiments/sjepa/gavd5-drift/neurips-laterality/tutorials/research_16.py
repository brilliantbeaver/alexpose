"""Editable GAVD tutorial: structured masks, missingness and contextual cues."""
from nbformat.v4 import new_notebook
from .motion_results_20260908 import add_saved_result_interpretation
from .masking_shared import md, code, setup_cell, data_instructions, configuration_cell, inputs_cell


def build_notebook():
    notebook = new_notebook(cells=[
        md('''
        # 16 — Structured masking on the real GAVD training partitions

        Scattered targets often retain nearby time points and landmarks. Does
        removing those clues encourage a useful movement representation?
        We extend Notebook 11 by inspecting connected regions, full landmark
        trajectories and interior gaps on **every real outer-training clip**,
        for five folds and five seeds. Each structure gets its own scattered
        reference with exactly the same realized target count.

        [VideoMAE, NeurIPS 2022, §3.3](https://arxiv.org/html/2203.12602) repeats
        spatial masks across frames, motivating full landmark trajectories here.
        [I-JEPA, CVPR 2023, §3](https://arxiv.org/html/2301.08243v3) predicts
        image regions from surrounding context. [SLiM, §3.3](https://arxiv.org/html/2603.10648v3)
        masks connected anatomical subsets over consecutive time spans. Our
        fixed region size and duration isolate a simpler skeleton comparison;
        these are adaptations, not reproductions of those complete recipes.
        '''),
        setup_cell(), data_instructions(), configuration_cell(), inputs_cell(),
        md('''
        ## 2. Define three separate prediction problems

        | Experiment | Structured target on complete 64-step input | Nominal tokens | Visible context |
        |---|---|---:|---|
        | `regions` (primary) | Six connected landmarks for eight blocks | 48 | Other landmarks and times |
        | `trajectories` (follow-up) | Three landmarks for all 16 blocks | 48 | Other landmark trajectories |
        | `completion` (follow-up) | All 33 landmarks for four interior blocks | 132 | Earlier and later observations |

        Region definitions use anatomical graph connections from Notebook 11.
        Their union covers all 33 landmarks. Array-neighboring IDs do not define
        a body region; head-to-body links are explicit masking connections.

        Missing observations reduce realized counts. The structured mask keeps
        its geometry, then its scattered reference hides exactly that many valid
        tokens in the same clip. No missing coordinate becomes a target. We do
        not trim full trajectories to fit a count. A structure with no feasible
        target/context fails before optimization.

        The two nominally 48-token structures need not realize equal counts with
        missing data. Completion has a different budget. Compare each against
        its own reference; a raw cross-experiment ranking confounds geometry
        with the amount of hidden information.
        '''),
        code('''
        from laterality_extensions.comparative_masks import connected_region_bank
        regions = connected_region_bank(tuple(range(33)), 6)
        assert set().union(*map(set, regions)) == set(range(33))
        print(f"{len(regions)} declared regions collectively cover all 33 landmarks.")
        mask_progress = NotebookTaskProgress("Structured-mask audit", "fold/seed pass")
        structure_audit = audit_training_masks_with_progress(inputs,
            experiments=("regions", "trajectories", "completion"), progress=mask_progress)
        display(structure_audit["summary"])
        assert structure_audit["per_clip"].groupby([
            "experiment", "fold", "seed", "sequence_id"]).hidden_tokens.nunique().eq(1).all()
        print(f"Audited {len(structure_audit['per_clip']):,} training-clip mask draws.")
        '''),
        md('''
        ## 3. See what a real clip leaves available

        Each row compares one structured family to its own uniform draw. Gray
        marks natural missingness, blue visible context, and orange deliberate
        targets. Panels identify the same first training clip selected before
        outcome analysis. Geometry remains intact even when gray cells interrupt
        its observed target support.
        '''),
        code('''
        from matplotlib.colors import ListedColormap
        fig, axes = plt.subplots(3, 2, figsize=(11, 9), constrained_layout=True)
        example_rows = []
        for row, experiment in enumerate(("regions", "trajectories", "completion")):
            examples = [(name, value) for (e, name), value in structure_audit["examples"].items() if e == experiment]
            for ax, (name, example) in zip(axes[row], examples):
                state = np.where(example["valid"], 1, 0); state[example["mask"]] = 2
                ax.imshow(state.T, origin="lower", aspect="auto", vmin=0, vmax=2,
                          cmap=ListedColormap(["#d4d4d4", "#72a8cf", "#df9340"]))
                ax.set(title=f"{experiment}: {name}, K={example['mask'].sum()}",
                       xlabel="Four-step block", ylabel="Landmark ID")
                example_rows.append({"experiment": experiment, "condition": name,
                    **{k: example[k] for k in ("sequence_id", "source_id", "fold", "seed")},
                    **context_cue_audit(example["mask"], example["valid"])})
        display(fig); plt.close(fig)
        display(pd.DataFrame(example_rows))
        '''),
        md('''
        ## 4. Quantify contextual cues across all folds and seeds

        `temporal_bracket_fraction` counts targets with the same landmark visible
        in **both immediately adjacent** time blocks. Full trajectories should
        have zero such brackets. `visible_neighbor_fraction` counts targets
        with at least one visible graph neighbor at the same time. A lower value
        means fewer of these specific clues remain.

        These are descriptive audits. Contextualized teacher features can
        require more than coordinate interpolation; fewer clues can also make
        asymmetric movement ambiguous. Neither fraction measures learned
        semantics. Each fold/seed mean weights its training videos equally.
        Averaging these dependent rows does not create independent observations.
        '''),
        code('''
        summary = structure_audit["summary"]
        display(summary.groupby(["experiment", "condition"], sort=False).agg(
            smallest_realized_count=("hidden_min", "min"), largest_realized_count=("hidden_max", "max"),
            mean_hidden_fraction=("hidden_fraction", "mean"),
            temporal_brackets=("temporal_bracket_fraction", "mean"),
            visible_neighbors=("visible_neighbor_fraction", "mean")))
        cue = summary.groupby(["experiment", "condition"], sort=False)[[
            "temporal_bracket_fraction", "visible_neighbor_fraction"]].mean()
        ax = cue.plot.barh(figsize=(10, 4.5), xlim=(0, 1), title=f"{DATA_MODE.upper()}: available local cues")
        ax.set_xlabel("Source-balanced fraction of hidden targets")
        ax.figure.tight_layout(); display(ax.figure); plt.close(ax.figure)
        '''),
        md('''
        ## 5. Predeclare the next inference

        Notebook 17's primary grid includes connected regions and the three
        motion arms from Notebook 15. Trajectories and completion are declared
        follow-ups; auditing them here does not claim they have been trained.
        Add their experiment names to the configuration in both 17 and 18 to
        conduct those comparisons.

        If regions remove local cues yet fail to improve trained-over-initial
        readout, cue removal alone is insufficient. If regions help against
        their scattered reference and initial features, the trajectory follow-up
        can distinguish spatial from temporal context effects. Avoid selecting
        geometry by outer-test rank on this already inspected development cohort.

        Interior gaps have context on both sides and test completion. Forecasting
        requires the past-only inputs in Notebook 14. Removing prepared tokens
        also differs from losing raw observations before interpolation and
        normalization, addressed in Notebook 13. Continue with
        [17](17_motion_and_structure_pretraining.ipynb) and
        [18](18_motion_information_and_readout.ipynb).
        '''),
    ])
    return add_saved_result_interpretation(notebook, 16)
