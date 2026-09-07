"""Source for Notebook 09: a controlled symmetry-objective tutorial."""

from __future__ import annotations

from textwrap import dedent, indent

from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def build_notebook():
    cells = []

    def prose(text):
        cells.append(new_markdown_cell(dedent(text).strip()))

    def code(text):
        cells.append(new_code_cell(dedent(text).strip()))

    def tracked_code(step, title, text):
        body = indent(dedent(text).strip(), "    ")
        cells.append(new_code_cell(f"with tutorial_progress.unit({step}, {title!r}):\n{body}"))

    prose(r"""
    # 09 — Can an explicit reflection loss improve useful JEPA features?

    This notebook introduces a proposed learning experiment that follows the
    completed laterality study. We keep the skeleton encoder architecture fixed
    and compare three training choices: the base objective, mirrored training
    clips, and an additional loss that asks corresponding body landmarks to
    produce consistent features under reflection.

    **The default run uses a small synthetic dataset. Its outputs test the
    implementation and illustrate how to read a comparison; they are not new
    GAVD results.** The optional local-data section is disabled. Nothing here
    changes the registered notebooks, checkpoints, or results in 00–06.

    The scientific question is whether a geometric constraint improves useful
    prediction on unseen sources. Reducing the same reflection error that we
    explicitly optimize would establish a narrower result. We will therefore
    examine predictive performance and feature variation beside consistency.
    """)
    prose(r"""
    ## 1. Understand what the earlier experiments leave open

    The registered study used 625 accepted pose sequences from 93 source videos,
    with a new encoder trained for each held-out source group and training seed.
    It separated three questions: whether a read-out predicts the signed motion
    target, whether that prediction changes sign under reflection, and whether
    the encoder tokens themselves transform consistently. Notebook 07 reviews
    the current numerical evidence and its uncertainty.

    An analytically constrained read-out can obey a sign rule before any encoder
    training. Its success on that rule does not tell us whether pretraining has
    added useful information. Reflection augmentation is a different intervention:
    it changes the examples encountered during learning but does not directly
    require any particular relation between their feature vectors.

    We now test an explicit relation inside the representation. This is a new
    exploratory objective comparison, rather than a reinterpretation of the
    earlier registered conditions. A favorable result would still need repeated
    source-held-out evaluation, paired initial-encoder controls, and independent
    movement outcomes before supporting a broader body-modeling claim.
    """)
    code("""
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    import torch
    import matplotlib.pyplot as plt
    from IPython.display import display
    from IPython import get_ipython
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    get_ipython().run_line_magic("matplotlib", "inline")
    set_matplotlib_formats("svg", "png")  # Keep an editable vector display too.

    candidates = [Path.cwd(), *Path.cwd().parents]
    SUITE_ROOT = next(
        path if (path / "laterality").is_dir() else path / "neurips-laterality"
        for path in candidates
        if (path / "laterality").is_dir() or (path / "neurips-laterality" / "laterality").is_dir()
    )
    if str(SUITE_ROOT) not in sys.path:
        sys.path.insert(0, str(SUITE_ROOT))

    from laterality_extensions.symmetry_learning import (
        feature_variation, reflect_training_batch, token_reflection_error,
        run_symmetry_comparison,
    )
    from laterality_extensions.masked_learning import (
        LearningSettings, load_learning_dataset,
    )
    from notebook_progress import NotebookTaskProgress

    tutorial_progress = NotebookTaskProgress(
        "Symmetry-aware JEPA tutorial",
        "stage",
        refresh_seconds=0.5,
    )
    tutorial_progress.start(
        7,
        profile="exploratory",
        note="Default execution runs a short synthetic three-arm comparison; local cohort training is opt-in.",
    )
    print("SYNTHETIC TUTORIAL — NON-EVIDENTIARY")
    """)
    prose(r"""
    ## 2. Distinguish a useful reflection rule from lost information

    Consider two simplified measurements: left-side motion and right-side
    motion. Their average is unchanged when the sides are exchanged. Their
    difference changes sign. Both are legitimate summaries, but they answer
    different questions.

    If a feature vector is identical for two mirrored inputs, any deterministic
    read-out of that vector must return the same prediction for both. It cannot
    recover opposite nonzero target values. A representation can instead retain
    the two measurements in separate landmark positions. Reflection exchanges
    those positions, allowing a read-out to preserve the signed difference.

    This small example explains the choice of training loss below. We align
    corresponding landmarks before comparing features. We do not force every
    whole-body summary to become unchanged under reflection. The arithmetic
    example is deliberately simpler than the full coordinate reflection, which
    also reverses the horizontal coordinate.
    """)
    tracked_code(1, "Check the arithmetic reflection rule", """
    left, right = 3.0, 2.0
    example = pd.DataFrame([
        {"Input": "Original", "Left": left, "Right": right,
         "Average": (left + right) / 2, "Signed difference": left - right},
        {"Input": "Sides exchanged", "Left": right, "Right": left,
         "Average": (left + right) / 2, "Signed difference": right - left},
    ])
    display(example)
    assert example["Average"].nunique() == 1
    assert example["Signed difference"].iloc[0] == -example["Signed difference"].iloc[1]
    """)
    prose(r"""
    ## 3. Transform observation masks and prediction targets together

    A pose contains both coordinates and information about which observations
    are usable. If a left knee is missing, reflecting the coordinates while
    leaving the validity flags unchanged would attach missingness to the wrong
    anatomical point. The same issue applies to the locations hidden during
    masked prediction.

    Our helper reverses the horizontal coordinate and exchanges every supplied
    left/right landmark pair. It applies the joint permutation to both masks.
    Time stays in the same order. Reflecting twice must recover the input and
    both masks, and no hidden target may move to an invalid position.

    The code uses four temporal patches, each covering four frames. A patch is
    eligible only when its required observations are valid. This strict rule
    avoids teaching the model to predict arbitrary placeholders for missing
    coordinates. The transformation tests are mathematical checks; passing them
    provides no evidence of predictive usefulness.
    """)
    tracked_code(2, "Verify reflection and mask transformations", """
    generator = torch.Generator().manual_seed(7)
    xyz = torch.randn(2, 16, 33, 3, generator=generator)
    valid_patch = torch.ones(2, 4, 33, dtype=torch.bool)
    valid_patch[:, :, 11] = False  # An unavailable left-shoulder patch.
    hidden_patch = torch.zeros_like(valid_patch)
    hidden_patch[:, 1, 12] = True  # Hide its available right-side counterpart.

    mirrored, mirrored_valid, mirrored_hidden = reflect_training_batch(
        xyz, valid_patch, hidden_patch
    )
    restored, restored_valid, restored_hidden = reflect_training_batch(
        mirrored, mirrored_valid, mirrored_hidden
    )
    torch.testing.assert_close(restored, xyz)
    assert torch.equal(restored_valid, valid_patch)
    assert torch.equal(restored_hidden, hidden_patch)
    assert not (mirrored_hidden & ~mirrored_valid).any()
    print("Reflection preserves valid target locations and restores the input when applied twice.")
    """)
    prose(r"""
    ## 4. Write the objective in terms we can test

    Let \(Z(x)\) denote the encoder tokens, \(M\) the input reflection, and
    \(S\) the corresponding joint permutation in the tokens. We compare
    \(Z(Mx)\) with \(S Z(x)\), leaving the feature channels unchanged. The
    added term is the squared difference, divided by the combined feature
    energy, averaged over sequences. Only aligned valid positions contribute:

    \[
    L_{\mathrm{total}} = L_{\mathrm{masked\ prediction}} +
    L_{\mathrm{feature\ regularization}} + \lambda L_{\mathrm{reflection}},
    \qquad
    L_{\mathrm{reflection}} = \frac{1}{B}\sum_i
    \frac{\lVert Z(Mx_i)-S Z(x_i)\rVert_C^2}
    {\operatorname{stopgrad}(\lVert Z(Mx_i)\rVert_C^2+
    \lVert S Z(x_i)\rVert_C^2)+\epsilon}.
    \]

    The denominator gives the penalty a comparable scale across sequences.
    Stopping its gradient prevents this extra loss from encouraging larger
    feature magnitudes simply to enlarge the denominator. A very small positive
    numerical floor prevents division by zero during optimization.

    The supplied anatomy and channel action are assumptions of this objective.
    We are asking whether enforcing this particular structure is useful. We are
    not testing whether a model discovers anatomical names without supplied
    information. A different action on the feature channels could also represent
    reflection, but that would require a separately specified experiment.
    """)
    prose(r"""
    ## 5. Check the failure case before looking for an improvement

    A model that always emits the same nonzero vector can have zero reflection
    error. It has discarded the differences between clips. The next cell makes
    this failure visible without training anything.

    Two complementary summaries are useful. Average channel standard deviation
    measures how much features change across inputs. Effective rank summarizes
    how broadly this variation is distributed across feature directions. Here
    rank is computed from the source-weighted centered feature covariance, so
    duplicating an identical clip within one source does not give that source
    extra influence.

    These are descriptive safeguards, rather than guarantees. Nonconstant,
    high-rank features can still encode camera artifacts instead of useful
    movement. Conversely, low rank is not automatically a failure when a task
    needs only a few meaningful quantities. Predictive testing remains necessary.
    """)
    tracked_code(3, "Expose the constant-feature failure case", """
    constant_tokens = torch.ones(3, 4, 33, 8)
    all_valid = torch.ones(3, 4, 33, dtype=torch.bool)
    error = token_reflection_error(constant_tokens, constant_tokens, all_valid, all_valid)
    variation = feature_variation(np.ones((3, 8)), np.array(["a", "b", "c"]))
    display(pd.DataFrame([{
        "Reflection error": float(error.mean()),
        "Feature standard deviation": variation["mean_channel_standard_deviation"],
        "Effective rank": variation["effective_rank"],
        "Constant features": variation["constant_features"],
    }]))
    print("Perfect agreement can coexist with no information distinguishing the inputs.")
    """)
    prose(r"""
    ## 6. Fix the comparison before fitting any model

    We use three conditions with the same architecture, initial parameters,
    training sources, number of updates, and masking policy. Sampling selects a
    source uniformly and then one of its sequences, so a source with many clips
    does not dominate learning. Separate random streams keep the source draws,
    target masks, and augmentation decisions reproducible.

    | Condition | Mirrored training clips | Added reflection penalty |
    |---|---|---|
    | Base objective | No | None |
    | Mirrored training clips | Probability 0.5 | None |
    | Explicit reflection loss | No random replacement | Weight \(\lambda=1\) |

    The penalty still evaluates both original and reflected inputs; the final
    row means that the ordinary masked-prediction input is not randomly replaced.
    This comparison isolates a practical objective change. It does not separate
    every possible interaction between the penalty and augmentation. A later
    factorial experiment can add a fourth condition containing both.

    The value one is a transparent tutorial setting, not an established optimum.
    A research run must either predeclare it or select a value using training-only
    validation. The shared tutorial read-out also uses a fixed ridge penalty;
    this small demonstration does not reproduce the registered nested selection
    procedure. No held-out result should be used to choose either penalty.

    Equal updates are not equal computation. The explicit penalty adds two
    encoder forwards and their gradients. Record elapsed time and computation
    counts, and include a separate compute-matched baseline before claiming
    efficiency. The small run below is intended to finish quickly on a CPU.
    """)
    tracked_code(4, "Train the synthetic three-arm comparison", """
    dataset = load_learning_dataset(real=False, fold=0)
    settings = LearningSettings(
        seed=7, fold=0, steps=8, batch_size=5,
        embed_dim=16, encoder_depth=1, predictor_depth=1,
        heads=2, mask_policy="gait", device="cpu",
    )
    print("Synthetic data:", dataset.synthetic)
    print("Training sources:", len(dataset.train_sources))
    print("Held-out sources:", len(dataset.test_sources))
    assert set(dataset.train_sources).isdisjoint(dataset.test_sources)
    comparison = run_symmetry_comparison(dataset, settings, symmetry_weight=1.0)
    """)
    prose(r"""
    ## 7. Read prediction and consistency as separate outcomes

    Each trained encoder is frozen before the read-out is fitted. The read-out
    sees targets only from training sources. Its test score gives every held-out
    source equal total weight, regardless of how many clips it contributes.
    The paired initial encoder uses the same architecture and feature extraction,
    making it possible to ask whether training added useful information.

    A direct pose-summary baseline is also important. The target in this
    experiment is computed from coordinates, so simple coordinate features are a
    demanding and informative comparator. The raw-pose baseline should not be
    confused with a clinical measurement or an independent ground truth.

    For prediction, larger \(R^2\) values indicate smaller errors relative to
    a weighted constant reference. Negative values are possible. For reflection
    error, values closer to zero indicate closer aligned-token agreement.
    These measures have different meanings and should not be combined into one
    score. Read feature-variation diagnostics beside them to detect obvious
    degenerate solutions, then inspect performance differences across repeated
    runs before drawing a research conclusion.
    """)
    tracked_code(5, "Summarize usefulness, consistency, and compute", """
    # The shared runner returns one row per evaluated feature condition.
    summary = comparison["summary"]
    display(summary.round(3).fillna("—"))
    print("Matched comparison checks:", comparison["pairing"])
    print(comparison["compute_scope"])
    compute = pd.DataFrame([
        {"Condition": label.replace("_", " "),
         "Training seconds": run["elapsed_training_seconds"],
         "Encoder forwards": run["encoder_forward_calls"]}
        for label, run in comparison["runs"].items()
    ])
    display(compute.round({"Training seconds": 2}))
    """)
    prose(r"""
    ### A compact view of the synthetic comparison

    The left panel compares predictive usefulness. The right panel compares
    reflection consistency for the same encoders. The initial encoder appears
    as a reference in both; the direct-pose reference applies only to prediction.
    The two horizontal axes intentionally use their own units. Dashed references
    are baselines, not confidence intervals, and this single-seed demonstration
    provides no uncertainty estimate.
    """)
    tracked_code(6, "Plot prediction and reflection outcomes", """
    learned = summary.iloc[:3]
    initial = summary.loc[summary["Condition"] == "Paired initial encoder"].iloc[0]
    raw = summary.loc[summary["Condition"] == "Direct pose summaries"].iloc[0]
    positions = np.arange(len(learned))
    colors = ["#52667A", "#2E7D75", "#8059A0"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].barh(positions, learned["Held-out R²"], color=colors)
    axes[0].axvline(initial["Held-out R²"], color="#30343B", linestyle="--", label="Initial encoder")
    axes[0].axvline(raw["Held-out R²"], color="#C07830", linestyle=":", label="Direct pose summaries")
    axes[0].set_yticks(positions, learned["Condition"])
    axes[0].set_xlabel("Prediction R² · larger is better")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].barh(positions, learned["Reflection error"], color=colors)
    axes[1].axvline(initial["Reflection error"], color="#30343B", linestyle="--", label="Initial encoder")
    axes[1].set_yticks(positions, [""] * len(positions))
    axes[1].set_xlabel("Reflection error · closer to zero is better")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.invert_yaxis()
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("SYNTHETIC DEMONSTRATION — NOT RESEARCH EVIDENCE", fontsize=11)
    fig.tight_layout()
    plt.show()
    """)
    prose(r"""
    ## 8. Decide which inference the results can support

    Use the following questions in order. Did the software finish with valid
    masks, finite losses, and disjoint source groups? Did the added loss reduce
    held-out reflection error? Do the resulting features retain variation? Does
    their predictive performance improve beyond the paired initial encoder and
    the base objective, and how do they compare with direct pose summaries?

    Several outcomes would be informative. Lower reflection error with unchanged
    prediction would show a consistency benefit without demonstrated predictive
    benefit. Lower error with worse prediction would motivate investigating an
    overly restrictive action or penalty strength. Improvements in both outcomes
    would justify replication and transfer experiments. Constant features would
    invalidate a favorable interpretation of consistency alone.

    A short synthetic run cannot choose among these explanations for GAVD.
    Its source count, noise process, and motion patterns are generated by code.
    It is useful for testing the complete comparison and displaying failure
    cases whose answers are known. Do not copy its numbers into a paper table
    describing real gait data, even if they appear favorable.
    """)
    prose(r"""
    ## 9. Optional local-data research run — disabled by default

    The next cell is an explicit gate. Leaving it unchanged performs no local
    pose loading or additional training. Enabling it requests a new exploratory
    run, separate from the registered evidence. Review the data-use permissions,
    cohort availability, chosen fold, settings, and expected runtime first.

    A single fold and seed are a pilot, not a confirmatory result. Before a full
    comparison, freeze all conditions and budgets; repeat the source-held-out
    folds and seeds; fit scaling and read-outs inside training data; and use
    source-cluster uncertainty that keeps training-seed variation identifiable.
    If existing GAVD outcomes guide these choices, label the resulting study as
    post-development and seek independently held-out data for confirmation.

    Save outputs only under a new research configuration. Never overwrite the
    registered models or reuse an old checkpoint as if it had been trained with
    the added objective. A failed or incomplete run should remain visibly
    incomplete, with no substitution of synthetic results.
    """)
    code("""
    RUN_LOCAL_RESEARCH = False
    if RUN_LOCAL_RESEARCH:
        with tutorial_progress.unit(7, "Optionally run the local symmetry pilot"):
            local_dataset = load_learning_dataset(real=True, fold=0)
            local_settings = LearningSettings(
                seed=7, fold=0, steps=100, batch_size=5,
                embed_dim=96, encoder_depth=4, predictor_depth=2,
                heads=4, mask_policy="gait", device="cpu", confirm_real_run=True,
            )
            local_comparison = run_symmetry_comparison(
                local_dataset, local_settings, symmetry_weight=1.0
            )
            display(local_comparison["summary"].round(3))
    else:
        print("Local-data research comparison: NOT RUN.")
        tutorial_progress.start_unit(7, "Optionally run the local symmetry pilot")
        tutorial_progress.skip_unit("Local symmetry training was not requested")

    tutorial_progress.complete(
        status="Tutorial complete" if RUN_LOCAL_RESEARCH
        else "Tutorial complete · optional local training not run"
    )
    """)
    prose(r"""
    ## 10. Place the contribution in the existing literature

    Explicit symmetry in representation learning is established. [SIE (ICML
    2023)](https://proceedings.mlr.press/v202/garrido23b.html) separates invariant
    and equivariant representations. [Soft-equivariance regularization (ICML
    2023)](https://proceedings.mlr.press/v202/kim23p.html) investigates approximate
    symmetries and includes motion forecasting. [seq-JEPA (NeurIPS
    2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/2f63d2963526bdd9ff1b8bcc2dc9905a-Paper-Conference.pdf)
    learns representations for invariant and equivariant tasks using sequences
    of views and transformation conditioning.

    Our potential contribution is a controlled answer about signed body dynamics
    measured through imperfect pose estimates: when does enforcing a known
    geometric relation improve useful prediction, and when does it constrain the
    representation without helping? Such a claim needs reproducible comparative
    evidence. Adding a familiar regularizer to a skeleton JEPA does not by itself
    establish a new class of world model.

    The next step is to test past-only movement prediction in Notebook 10 and,
    eventually, an independently measured movement outcome. A reflection rule
    can make predictions geometrically coherent, but clinical usefulness and
    prediction of future motion remain separate empirical questions.
    """)
    return new_notebook(cells=cells)
