"""Editable GAVD tutorial: motion-weighted masks across five folds and seeds."""
from nbformat.v4 import new_notebook
from .motion_results_20260908 import add_saved_result_interpretation
from .masking_shared import md, code, setup_cell, data_instructions, configuration_cell, inputs_cell


def build_notebook():
    notebook = new_notebook(cells=[
        md(r'''
        # 15 — Motion weighting on the real GAVD training partitions

        The completed comparisons in [TUTORIAL.md](docs/TUTORIAL.md) did not
        establish that scattered-mask pretraining improved movement readout over
        initial features. Motion weighting is a focused next comparator: does
        hiding reliably moving tokens change what the encoder learns?

        This notebook loads **real GAVD**, displays all five folds and five seeds,
        then audits masks on every outer-training clip. Notebook 16 examines
        structured masks, 17 trains the paired grid, and 18 evaluates held-out
        test predictions. This inspection establishes the intervention to test;
        it does not train a model or report downstream performance.
        '''),
        setup_cell(), data_instructions(), configuration_cell(), inputs_cell(),
        md(r'''
        ## 2. Translate the authoritative method into a precise sampler

        [MAMP, ICCV 2023, §3.4](https://arxiv.org/html/2308.07092) scores motion
        across one token length, applies a softmax and samples targets without
        replacement through Gumbel ranking. [S-JEPA, ECCV 2024, §3](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf)
        adopts motion-based masking while predicting teacher features.

        The [official MAMP code](https://github.com/maoyunyao/MAMP/blob/main/model_mamp/transformer.py)
        averages absolute displacement over four offsets and three axes, copies
        the second block's intensity into the first, and normalizes by the clip
        maximum. That normalization is absent from the paper's written softmax.
        Here `mamp_motion` follows the code on fully observed inputs, with
        explicit handling of missing transitions for GAVD.

        Define $a_i=I_i/(\max_j I_j\,\tau+10^{-10})$, with $\tau=0.8$.
        Draw $u_i\sim U(0,1)$ and hide the $K$ largest
        $a_i-\log(-\log u_i)$. Softmax normalization cancels in this ranking.
        These weights are not marginal inclusion probabilities.

        | Arm | What increases target probability? | Declared control |
        |---|---|---|
        | `uniform` | Nothing: all valid tokens equally eligible | Reference |
        | `mamp_motion` | Mean absolute block displacement, max-normalized | Temperature 0.8 |
        | `robust_motion` | Median Euclidean displacement, clipped at positive 95th percentile | 75% motion / 25% uniform mixture |

        All 33 landmarks remain eligible. The common count is half the minimum
        valid twelve-landmark gait-token count in the current batch, rounded
        down and bounded below by one, following the existing budget convention.
        **It does not hide 50% of all 33-landmark tokens.** The three arms share
        the same realized count per clip. Stationary clips fall back to uniform
        weights. Missing cells cannot be targets or motion evidence.

        Prepared steps have been resized to length 64. Scores describe prepared
        coordinates, not meters per second. A uniform mixture keeps slow
        landmarks eligible but cannot guarantee both legs in every draw.
        '''),
        code('''
        mask_progress = NotebookTaskProgress("Motion-mask audit", "fold/seed pass")
        motion_audit = audit_training_masks_with_progress(inputs, experiments=("motion",), progress=mask_progress)
        display(motion_audit["summary"])
        per_clip = motion_audit["per_clip"]
        assert per_clip.role.eq("train").all()
        assert per_clip.groupby(["fold", "seed", "sequence_id"]).hidden_tokens.nunique().eq(1).all()
        print(f"Audited {len(per_clip):,} clip/seed/fold/arm draws, using training clips only.")
        '''),
        md('''
        ## 3. Read coverage before interpreting motion preference

        Each row above covers one arm, fold and seed. Every training clip is
        visited once for inspection. Descriptive means give each video equal
        total weight. These fixed inspection batches differ from the
        source-balanced random schedule used in pretraining. Seeds estimate
        mask-draw variability; overlapping training folds remain dependent.

        `target_motion` and `eligible_motion` use the **same robust diagnostic
        score for all arms**. Their difference describes selection preference
        without comparing unlike sampler score units. Higher values show more
        movement under this diagnostic; they cannot establish useful features.
        `both_legs_targeted` checks whether both sides have at least one target
        among the hip, knee, ankle, heel or foot landmarks.
        '''),
        code('''
        overview = motion_audit["summary"].copy()
        overview["motion_enrichment"] = overview.target_motion - overview.eligible_motion
        display(overview.groupby("condition", sort=False)[[
            "hidden_tokens", "hidden_fraction", "motion_enrichment", "both_legs_targeted"]].mean())
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
        for name, group in overview.groupby("condition", sort=False):
            by_seed = group.groupby("seed").mean(numeric_only=True)
            axes[0].plot(by_seed.index, by_seed.motion_enrichment, marker="o", label=name)
            axes[1].plot(by_seed.index, by_seed.both_legs_targeted, marker="o", label=name)
        axes[0].set(xlabel="Mask seed", ylabel="Target minus eligible motion",
                    title=f"{DATA_MODE.upper()}: training-mask motion preference")
        axes[1].set(xlabel="Mask seed", ylabel="Fraction with both legs targeted", ylim=(0, 1.05))
        axes[0].legend(fontsize=8)
        display(fig); plt.close(fig)
        '''),
        md('''
        ## 4. Inspect an identified training clip

        The masks below use the same first training clip and seed, selected by
        row order before outcome analysis. Gray marks missing observations,
        blue visible context, and orange hidden targets. Axes show anatomical
        landmark IDs and prepared time blocks. These are actual sampled masks,
        not an idealized illustration or a representative clinical example.
        '''),
        code('''
        from matplotlib.colors import ListedColormap
        examples = [(name, value) for (experiment, name), value in motion_audit["examples"].items()]
        fig, axes = plt.subplots(1, len(examples), figsize=(12, 4), constrained_layout=True)
        for ax, (name, example) in zip(np.atleast_1d(axes), examples):
            state = np.where(example["valid"], 1, 0); state[example["mask"]] = 2
            ax.imshow(state.T, origin="lower", aspect="auto", vmin=0, vmax=2,
                      cmap=ListedColormap(["#d4d4d4", "#72a8cf", "#df9340"]))
            ax.set(title=f"{name}: {example['mask'].sum()} targets", xlabel="Four-step block", ylabel="Landmark ID")
        display(fig); plt.close(fig)
        display(pd.Series({k: examples[0][1][k] for k in ("sequence_id", "source_id", "fold", "seed")}))
        '''),
        md('''
        ## 5. Carry a falsifiable hypothesis into training

        Motion weighting can emphasize tracking jumps as well as real movement.
        The robust alternative is designed to reduce isolated-jump sensitivity;
        coverage alone cannot prove downstream benefit. Compare learned and
        initial encoders with identical summaries and ridge selection in
        Notebook 18 before attributing any gain to JEPA training.

        The sampler is label-blind. It may inspect the complete permitted
        training clip to choose hidden locations, which can themselves convey
        motion information. A past-only task must instead compute motion from
        its observed prefix, as in Notebook 14. Outer-test clips here are
        enumerated for coverage but never tune temperature or mixture weight.

        Continue with [16](16_structured_masking_and_context.ipynb), then
        [17](17_motion_and_structure_pretraining.ipynb) and
        [18](18_motion_information_and_readout.ipynb). The
        [source review](docs/MOTION_STRUCTURED_MASKING.md) records the adaptations
        and remaining anatomical assumptions.
        '''),
    ])
    return add_saved_result_interpretation(notebook, 15)
