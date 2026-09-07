"""Authoring source for the evidence and information-path tutorial."""

from textwrap import dedent, indent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    def md(text):
        return new_markdown_cell(dedent(text).strip())

    def code(text):
        return new_code_cell(dedent(text).strip(), execution_count=None, outputs=[])

    def tracked_code(step, title, text):
        body = indent(dedent(text).strip(), "    ")
        return new_code_cell(
            f"with tutorial_progress.unit({step}, {title!r}):\n{body}",
            execution_count=None,
            outputs=[],
        )

    cells = [
        md(r"""
        # 07 — What the completed study teaches us, and what to test next

        The completed experiment established a useful distinction: exposing an
        encoder to mirrored training clips improved one measure of feature
        consistency, while its benefit for predicting signed movement remained
        uncertain. This gives us a concrete starting point for further research,
        but it leaves several possible explanations for the weak prediction.

        This notebook follows information from the observed pose through input
        preparation, the encoder, temporal pooling, and the final read-out. We
        first read a small set of saved results, then use a constructed example
        to understand what temporal averaging can remove. An optional, inexpensive
        analysis asks how closely the prepared model input reproduces the original
        target calculation. No encoder is trained or evaluated here.

        By the end, you should be able to distinguish an established finding from
        a plausible explanation and choose a follow-up experiment that separates
        those explanations. The new analyses are exploratory, developed after
        the original results. They do not alter the frozen experiment in 00–06.
        """),
        md("""
        ## Step 1 — Keep the three kinds of evidence separate

        | Material in this notebook | What it can establish |
        |---|---|
        | Saved real-data report | What the completed source-held-out experiment reported |
        | Constructed two-side motion example | How averaging can lose a movement distinction in a simple case |
        | Optional input reconstruction | How two measurement paths agree on the available processed poses |

        None of the new cells tests a clinical outcome, generalization to unseen
        people, or a future-prediction model. The saved study contains 625 accepted
        pose sequences from 93 source videos, with five source-level outer folds,
        five training seeds, and two training recipes. Its independent data unit
        is the source video; 50 fitted encoders do not create 50 independent datasets.

        Default execution reads available aggregate reports and runs the small
        constructed example. The real input-reconstruction cell requires an
        explicit switch. It uses local derived poses only and does not open video,
        reveal source identifiers, overwrite artifacts, or run checkpoint inference.
        """),
        code("""
        from pathlib import Path
        import sys

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from IPython.display import display
        from IPython import get_ipython
        from matplotlib_inline.backend_inline import set_matplotlib_formats
        get_ipython().run_line_magic("matplotlib", "inline")
        set_matplotlib_formats("svg", "png")

        def locate_suite_root():
            start = Path.cwd().resolve()
            for ancestor in (start, *start.parents):
                for candidate in (ancestor, ancestor / "neurips-laterality"):
                    if (candidate / "config" / "protocol.json").is_file() and (candidate / "laterality").is_dir():
                        return candidate
            raise FileNotFoundError("Run from the repository or its neurips-laterality folder.")

        SUITE_ROOT = locate_suite_root()
        if str(SUITE_ROOT) not in sys.path:
            sys.path.insert(0, str(SUITE_ROOT))

        from laterality.config import load_context
        from laterality_extensions.diagnostics import (
            read_registered_results,
            reconstruct_processed_target,
            summarize_reconstruction,
            temporal_pooling_example,
        )
        from notebook_progress import NotebookTaskProgress

        RUN_REAL_INPUT_RECONSTRUCTION = False
        tutorial_progress = NotebookTaskProgress(
            "Evidence and information-path tutorial",
            "stage",
            refresh_seconds=0.5,
        )
        tutorial_progress.start(
            5,
            profile="exploratory",
            note="Default execution reviews saved evidence and runs constructed checks; no encoder training.",
        )
        print("Default: read-only report review and constructed examples; no training.")
        """),
        md("""
        ## Step 2 — Read the completed results without overloading the comparison

        The next cell selects five report rows. Their confidence intervals resample
        whole source videos while keeping their clips and paired seed results
        together. They describe uncertainty conditional on the fitted pipeline;
        they do not include the variation from repeating the entire study with
        new videos, folds, and training seeds.

        A positive change in R² favors the first prediction method. For the strict
        token error, a negative change means the two transformed feature arrays
        agree more closely. That comparison uses the predefined anatomical joint
        exchange while keeping feature channels fixed. The error therefore tests
        one specified transformation rule, not every possible representation of
        reflection.

        Reading these files checks the declared protocol and cohort metadata and
        selects unambiguous report rows. It does not repeat the original bootstrap
        or revalidate all 50 checkpoints. If the report is absent, the cell says so
        instead of substituting synthetic scores.
        """),
        tracked_code(1, "Read the registered results", """
        paper_context = load_context(SUITE_ROOT / "config" / "protocol.json", profile="paper")
        saved_evidence = read_registered_results(paper_context)
        print(saved_evidence["status"])
        if saved_evidence["rows"]:
            print(f"Cohort: {saved_evidence['sequences']} sequences from {saved_evidence['sources']} source videos.")
            result_table = pd.DataFrame(saved_evidence["rows"])
            display(result_table.style.format({"estimate": "{:.3f}", "ci95_low": "{:.3f}", "ci95_high": "{:.3f}"}))
        """),
        md("""
        ### What the saved values mean

        The original prediction result was R² = 0.060, with a 95% interval from
        −0.025 to 0.126. Its difference from the matched untrained encoder was
        −0.018, with an interval from −0.039 to 0.002. These results do not establish
        a predictive benefit from this pretraining recipe. They also do not show
        that all information about the target has disappeared from the encoder.

        Mirrored training clips reduced strict token error by about 0.0084
        (interval −0.0102 to −0.0069). Their estimated prediction benefit was much
        smaller and uncertain: a change in R² of 0.004, with an interval from
        −0.006 to 0.013. Feature transformation consistency and useful prediction
        can therefore respond differently to the same training change.

        The trained encoder without mirrored clips had higher strict error than
        its initialization by 0.031 (interval 0.016 to 0.048). A low error at random
        initialization can partly reflect common or input-insensitive features,
        so future consistency analyses also need feature-variation and prediction
        checks. No formal power analysis or equivalence test justifies calling
        the uncertain predictive contrasts a universal failure of JEPA.
        """),
        md("""
        ## Step 3 — Locate the unanswered question along the information path

        The target uses observed, paired-valid motion transitions before temporal
        resizing. Encoder inputs follow another path: short gaps can be filled,
        coordinates are normalized, and each clip is represented by 64 time steps.
        These paths were intentionally separated to keep invented observations
        out of the target. Their difference also creates an empirical question:
        how much of the original movement contrast remains easy to recover from
        the processed input?

        The existing read-out adds further choices. It averages contextual encoder
        tokens over time for each anatomical side, combines side differences and
        sums, and fits a linear ridge model. An encoder can retain information that
        this particular summary or linear read-out does not recover. Conversely,
        a more elaborate read-out can exploit structure already present in an
        untrained encoder, which is why paired initialization controls remain useful.
        """),
        tracked_code(2, "Map the information path", """
        fig, ax = plt.subplots(figsize=(11, 2.4))
        stages = ["Observed\\npose stream", "Prepared\\n64-step input", "Contextual\\nencoder tokens", "Time-averaged\\nside features", "Linear\\nprediction"]
        for index, label in enumerate(stages):
            ax.text(index, 0.5, label, ha="center", va="center", fontsize=11,
                    bbox={"boxstyle": "round,pad=0.6", "facecolor": "#edf3f8", "edgecolor": "#56718a"})
            if index < len(stages) - 1:
                ax.annotate("", xy=(index + 0.65, 0.5), xytext=(index + 0.35, 0.5),
                            arrowprops={"arrowstyle": "->", "color": "#56718a"})
        ax.set(xlim=(-0.6, 4.6), ylim=(0, 1))
        ax.set_title("Where could a useful movement distinction become harder to recover?", fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ## Step 4 — See why equal average positions can hide different movement

        Consider two sides moving along a line over four equally spaced samples.
        In example A, the left side alternates between 0 and 1, while the right
        side changes position only once. Both have mean position 0.5. In example
        B, their trajectories are exchanged, so both means remain 0.5 even though
        the signed speed contrast reverses.

        The next cell writes out the calculation: take consecutive differences,
        convert them to nonnegative speeds, then compare the two median speeds.
        This deliberately simple example shows a limitation of averaging raw
        positions. It does **not** prove that averaging contextual encoder tokens
        has the same limitation, because those tokens may already encode motion.
        That distinction must be tested with the actual representations.
        """),
        tracked_code(3, "Run the temporal-pooling example", """
        toy = temporal_pooling_example()
        trajectories = toy["trajectories"]
        median_speed = np.median(np.abs(np.diff(trajectories, axis=1)), axis=1)
        signed_contrast = (median_speed[:, 0] - median_speed[:, 1]) / median_speed.sum(axis=1)

        display(pd.DataFrame({
            "constructed example": ["A", "B"],
            "left mean position": trajectories[:, :, 0].mean(axis=1),
            "right mean position": trajectories[:, :, 1].mean(axis=1),
            "signed median-speed contrast": signed_contrast,
        }))
        assert np.array_equal(toy["mean_position"][0], toy["mean_position"][1])
        assert signed_contrast[0] == -signed_contrast[1]

        fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2), sharey=True)
        for index, ax in enumerate(axes):
            ax.plot(trajectories[index, :, 0], "o-", label="Left", color="#2166ac")
            ax.plot(trajectories[index, :, 1], "s--", label="Right", color="#b35806")
            ax.set(title=f"Example {'AB'[index]}: contrast {signed_contrast[index]:+.0f}", xlabel="Sample index", xticks=range(4))
            ax.axhline(0.5, color="0.65", linewidth=1, label="Both means")
            ax.legend(fontsize=9)
        axes[0].set_ylabel("Position in arbitrary units")
        fig.suptitle("CONSTRUCTED EXAMPLE — not a gait result", fontsize=11)
        plt.tight_layout()
        plt.show()
        """),
        md("""
        ### Why reversing time is not a failure test for this target

        This target compares speed magnitudes, so reversing an equally sampled
        trajectory preserves its median speed. We should not demand a sign change
        under time reversal: the sign refers to anatomical side, not forward or
        backward time. Randomly shuffling frames can alter speed, but it need not
        eliminate every left–right difference either. A temporal control is useful
        only when its expected effect follows from the quantity being measured.

        For a future-motion benchmark, shuffled or missing history can instead
        test whether a predictor benefits from its temporal context. That is a
        different question from the current whole-clip movement contrast.
        """),
        tracked_code(4, "Check the time-reversal expectation", """
        reversed_speed = np.median(np.abs(np.diff(trajectories[:, ::-1], axis=1)), axis=1)
        assert np.array_equal(median_speed, reversed_speed)
        print("Constructed check passed: reversing time preserves both median speeds.")
        """),
        md("""
        ## Step 5 — Optionally compare the original target with the processed input

        Set `RUN_REAL_INPUT_RECONSTRUCTION = True` in Step 1 and rerun this
        notebook to enable the next calculation. It reads the validated cohort,
        applies the same five-pair formula to the 64-step model inputs, and compares
        that result with the original observed-coordinate target. It takes no
        optimization steps and does not change the registered target or cohort.

        The calculation uses equally spaced relative sample positions. Because
        the contrast divides one speed difference by the speed sum, a common time
        factor cancels apart from the small numerical stabilizer. This observation
        does not turn the resized input into a physically timed forecasting input.

        The formula can become undefined after processing even when it was defined
        on the original stream. We keep those failures visible and calculate
        agreement only where both values are finite, giving each remaining source
        equal total weight. Direct agreement R² below one shows that the two
        calculations differ; it is neither a model score nor an upper bound on
        what a sufficiently informative representation could predict.
        """),
        code("""
        reconstruction_result = None
        if not RUN_REAL_INPUT_RECONSTRUCTION:
            print("Real input reconstruction: not run. Enable the explicit switch in Step 1 to compute it.")
            tutorial_progress.start_unit(5, "Optionally reconstruct the target from processed inputs")
            tutorial_progress.skip_unit("Real input reconstruction was not requested")
        else:
            with tutorial_progress.unit(5, "Optionally reconstruct the target from processed inputs"):
                from laterality.data import load_cohort

                cohort = load_cohort(paper_context)
                recomputed = reconstruct_processed_target(
                    cohort.model_xyz, cohort.model_valid, paper_context.protocol["target"]
                )
                reconstruction_result = summarize_reconstruction(
                    cohort.targets, recomputed, cohort.table["video_id"].to_numpy()
                )
                count_labels = {
                    "input_sequences": "Original sequences",
                    "input_sources": "Original source videos",
                    "finite_sequences": "Sequences with both calculations",
                    "finite_sources": "Source videos in that overlap",
                    "excluded_sequences": "Sequences without a finite recomputation",
                }
                display(pd.DataFrame({"quantity": list(count_labels.values()),
                                      "count": [reconstruction_result[key] for key in count_labels]}))
                metric_labels = {
                    "direct_agreement_r2": "Direct agreement R²",
                    "weighted_correlation": "Source-balanced correlation",
                    "weighted_mae": "Mean absolute difference",
                    "weighted_original_target_sd": "Original-target standard deviation",
                }
                display(pd.DataFrame({"quantity": list(metric_labels.values()),
                                      "value": [f"{reconstruction_result[key]:.3f}" for key in metric_labels]}))
                print(f"Source-balanced sign agreement: {100 * reconstruction_result['weighted_sign_agreement']:.1f}%.")

                finite = np.isfinite(cohort.targets) & np.isfinite(recomputed)
                fig, ax = plt.subplots(figsize=(4.5, 4.0))
                ax.scatter(cohort.targets[finite], recomputed[finite], s=10, alpha=0.35, color="#2166ac")
                limits = [min(cohort.targets[finite].min(), recomputed[finite].min()),
                          max(cohort.targets[finite].max(), recomputed[finite].max())]
                ax.plot(limits, limits, color="0.5", linestyle="--", linewidth=1)
                ax.set(xlabel="Original observed-coordinate target", ylabel="Formula on processed input",
                       title="Exploratory measurement-path comparison")
                ax.text(0.02, 0.98, "Each point is a clip; numerical summary balances sources.",
                        transform=ax.transAxes, va="top", fontsize=8)
                plt.tight_layout()
                plt.show()

        tutorial_progress.complete(
            status="Tutorial complete" if RUN_REAL_INPUT_RECONSTRUCTION
            else "Tutorial complete · optional input reconstruction not run"
        )
        """),
        md("""
        ### A recorded exploratory check, and the limits of its interpretation

        A read-only check during this tutorial's preparation on September 7, 2026
        produced finite recomputations for 623 of 625 sequences from 92 of 93
        sources. On that overlap, source-balanced direct agreement was R² = 0.218,
        correlation 0.652, and mean absolute difference 0.041. The original target's
        source-balanced standard deviation was about 0.060. Sign agreement was
        about 70%. These are rounded descriptions of a new processing diagnostic,
        separate from the registered held-out results in Step 2.

        The discrepancy makes input preparation a credible contributor to weak
        decoding. It does not identify which processing operation caused it,
        establish irreversible information loss, or prove that a different encoder
        would fail. Interpolation, temporal resizing, visibility changes, and their
        interaction with median speeds all merit controlled comparison. The two
        excluded recalculations must remain excluded transparently from this
        comparison rather than being silently assigned zero.

        If the execution switch remains off, the paragraph above is a recorded
        prior check, not output produced by the current notebook run. If enabled,
        compare the newly displayed counts and values with this record. Investigate
        differences before combining results from changed data or methodology.
        """),
        md("""
        ## Step 6 — Choose experiments that distinguish the remaining explanations

        A useful next experiment changes one meaningful ingredient and asks whether
        a prespecified outcome changes with it. Start with inexpensive measurement
        and read-out comparisons before committing to a larger pretraining grid.

        | Question | Controlled comparison | What a positive result would support |
        |---|---|---|
        | Does input preparation obscure movement? | Original versus interpolated versus resized calculation on the same eligible sequences | A specific processing choice alters the measured signal |
        | Does the read-out miss available information? | Time-averaged versus ordered or motion-sensitive features, with train-only selection | A different summary recovers more held-out information |
        | Does target selection improve learning? | Gait-informed versus generic masks with matched numbers of hidden valid tokens | This selection helps the stated training recipe and endpoint |
        | Is the token test sensitive to feature coordinates? | Fixed identity action versus a train-fitted constrained reflection action | Some mismatch depends on how reflection is represented in feature channels |
        | Does the representation support dynamics? | Past-only prediction versus persistence and constant-velocity baselines | Useful information about subsequent pose beyond simple continuation |

        Every learned-feature comparison should keep the corresponding untrained
        encoder, and every new read-out must fit and tune on training sources only.
        Matching feature dimension or using train-only dimensionality reduction
        helps separate a temporal-summary change from a capacity increase. Because
        the existing test results motivated these directions, new results on the
        same sources remain post-development evidence. Strong confirmation would
        benefit from new, separately governed sources or an external dataset.
        """),
        md("""
        ## Step 7 — Decide what would count as progress

        The strongest next contribution would explain why a representation becomes
        more useful, not merely reduce a consistency error. Masking experiments can
        test which hidden prediction tasks retain movement information. A
        feature-coordinate analysis can clarify the meaning of the existing strict
        reflection result. Past-only prediction can move the research toward
        observable body dynamics, provided future coordinates, validity, and
        normalization never enter the predictor.

        A lower reconstruction discrepancy alone would improve measurement
        understanding. A stronger probe alone would improve the evaluation design.
        A learned representation that improves a prespecified held-out motion task
        beyond matched initialization and direct-pose baselines would provide
        evidence of useful learning. These are worthwhile but different outcomes,
        and the notebook should report whichever the experiment supports.

        The following extension notebooks develop these comparisons. Their
        constructed demonstrations explain and test the proposed machinery;
        real training and confirmatory results require their explicit experimental
        steps. The existing GAVD data remain source-indexed video-derived poses,
        with no basis here for diagnosis, unseen-person generalization, or
        action-conditioned physical control.
        """),
    ]
    return new_notebook(cells=cells)
