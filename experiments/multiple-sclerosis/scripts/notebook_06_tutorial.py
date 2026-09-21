"""Prose-only tutorial for notebook 06's retained full-budget laptop run.

Evidence: capstone-20260921T011305448893Z, registry 9496e61b050f.
The 88 saved OOF rows reproduce results.json metrics exactly. The RF description
also reflects the inspected feature cache and ambient's legacy angle extractor.
This module does not train models, change predictions, or repair that extractor.
"""

from textwrap import dedent


def add_tutorial(original, md):
    """Keep original executable cells, including any saved outputs, unchanged."""
    stem = "06_capstone_rf_vs_sjepa"
    for i, cell in enumerate(original):
        cell.setdefault("id", f"{stem}-{i:02d}")

    def note(key, text):
        cell = md(dedent(text).strip())
        cell["id"] = f"06-tutorial-{key}"
        return cell

    return [original[0], note("intro", """
        # 06 · What did our five-system comparison actually teach us?

        Can a model that learns from moving skeletons distinguish the dataset's
        Normal, MS, and Parkinson's disease (PD) labels better than simpler
        descriptions of the same clips? That is the question we test here.
        These are collection labels, not diagnoses established by this experiment.

        The saved full-budget run has completed: **88 clips from 41 source videos,
        five test folds, and five comparison systems**. S-JEPA's overall
        source-weighted macro-F1 is **0.457**, compared with **0.452** for mean
        pose. That small overall lead is not consistent across folds: mean pose
        scores higher in four of the five. We therefore have a useful controlled
        pilot, but have not established an advantage from learning temporal motion.

        Follow the steps below to understand the data, the five systems, the
        training procedure, the scores, and the conclusions. The final sections
        explicitly assess each finding's value as a research contribution and
        what a credible workshop paper would still need.

        All numerical interpretations refer to this saved **laptop, seed-42,
        800+400-update-per-fold** run, not smoke mode or older experiments.
        A code review also found an RF feature-construction error, explained
        below. Its saved result is reported transparently, not silently corrected.
        If you rerun with different settings, update these written observations
        to match the new outputs. The [evidence and submission notes](docs/16-capstone-results-and-contributions.md)
        identify the retained artifacts and current workshop options.
        """), *original[2:5], note("data", """
        ## Step 1 · Decide what counts as a separate test example

        A **frame** is one instant of a video. A **window** here contains 32
        consecutive frames. A **clip** can contain many windows, but receives
        one final predicted label. Several clips can come from one **source
        video**, so they are not necessarily independent examples.

        We keep clips from the same source together. For example, two segments
        cut from one recording cannot be on opposite sides of the training/test
        split. Otherwise, the model could be rewarded for recognizing a familiar
        recording instead of transferring to another source.

        The registry below fixes five rounds, called **folds**. Each round has
        training sources for fitting models, validation sources for choosing
        between two S-JEPA stages, and test sources for the final check. Across
        the five rounds, every usable clip is tested exactly once. Its
        **out-of-fold (OOF) prediction** comes from a model that did not train
        or choose its stage using that source. We do not test one shared trained
        model five times; each fold starts with fresh model weights.
        """), original[6], note("counts", """
        ### Read the split table

        The usable collection has 24 Normal clips from 16 sources, 29 MS clips
        from 13 sources, and 35 PD clips from 12 sources. Three other clips were
        excluded at the earlier pose-quality gate. All five systems use the same
        retained clips and source assignments.

        As a concrete example, Fold 0 uses **51 training clips from 24 sources,
        19 validation clips from 8 sources, and 18 test clips from 9 sources**.
        The other folds change which sources play each role. Within the outer
        development sources, only the first split of a four-fold splitter is
        used for validation; this is one inner holdout, not a full inner search.

        A source identifier is derived from a recording's filename. It is not
        a verified person identifier. Different videos could show the same
        person or reposted footage. Source grouping controls known within-video
        overlap, but does not establish participant independence. See the
        [split methodology](docs/11-full-data-splits.md) for the exact procedure.
        """), note("systems", """
        ## Step 2 · Understand the five systems

        A **feature vector** is simply a list of numbers describing one clip.
        A **classifier** uses that list to predict a label. These systems differ
        mainly in how they build the list and which classifier reads it.

        | System | Description supplied to the classifier | How it predicts a label | Question it helps answer |
        |---|---|---|---|
        | `rf` | Joint-angle averages, left/right differences, and ranges; 15 varying feature columns in this run | Random Forest: 100 decision trees, maximum depth 5 | How well does a small set of designed angle summaries work? |
        | `sjepa` | 96 learned features, averaged into one vector per clip | Frozen teacher encoder plus a separately fitted linear classifier | Does self-supervised representation learning provide useful label information? |
        | `visibility` | 33 average visibility values and 33 standard deviations: 66 numbers | The same linear-classifier recipe as S-JEPA | How predictive are detection/occlusion-related summaries alone? |
        | `mean_pose` | Average normalized x and y for each of 33 landmarks: 66 numbers | The same linear-classifier recipe as S-JEPA | How competitive is a description that discards frame order? |
        | `majority` | No skeleton features | Always predict the most common training-clip label in that fold | How well can a rule do without examining the test clip? |

        ### `rf`: summarize the angles, then combine simple decisions

        The RF branch starts with the cached pixel-coordinate landmarks, not
        the video pixels. It calculates left and right hip, knee, and ankle
        angles. Its intended summaries are six means, three absolute left/right
        mean differences, and six ranges. For a teaching example, knee angles
        of 100°, 120°, and 140° have mean 120° and range 40°. A decision tree
        learns sequences of feature-threshold questions; the forest combines
        the trees' class probabilities to choose a label. Maximum depth 5
        limits a path through a tree to five successive splits.

        The feature container has 82 slots, but this legacy extraction path
        does not calculate the comprehensive gait measures suggested by all
        those names. Only 15 columns vary in each fold's training data. Constant
        columns are removed using training data alone, and the remaining
        columns are standardized. RF uses `max_features='sqrt'`, balanced class
        weights, seed 42, and one tree-fitting worker in the saved run.

        **Important implementation limitation:** the legacy
        `GaitFeatureVector.from_joint_angles` method fills `left_ankle_range`
        from the **right** ankle's statistics. Both ankle-range columns are
        therefore identical in all 88 cached feature rows. This omits the true
        left-ankle range and duplicates another input; its effect on performance
        has not been measured. The RF scores below describe that implementation.
        A corrected, regression-tested rerun is necessary before claiming a fair
        S-JEPA advantage over the intended RF baseline. This tutorial update
        has not changed the extractor or the saved predictions.

        ### `sjepa`: learn a description before learning the labels

        S-JEPA first trains on skeleton motion without using condition labels
        in its learning loss. A student sees part of a window and predicts
        feature targets supplied by a teacher that sees the complete window.
        The teacher follows a slow moving average of the student's weights.
        Neither network is asked to predict MS during this stage.

        To describe a finished clip, we use the frozen teacher. It sees each
        complete input window. A fixed seed-0 mask selects which **output token
        features to average**; this readout mask does not hide input joints.
        Averaging those features, then averaging across windows, produces 96
        numbers per clip. A separate classifier learns Normal/MS/PD labels from
        the training clips' vectors. This is a **linear probe**: a simple reader
        of the learned representation, with no label-driven encoder updates.

        ### `visibility`: retain visibility, discard coordinates

        For each landmark, this control records the mean and standard deviation
        of the pose detector's visibility values over the full clip. Standard
        deviation measures how much a value varies around its average. For example,
        an ankle that is frequently obscured can have a different summary from
        one that stays visible. This uses neither x-y positions nor frame order.
        Visibility is not diagnostic confidence or a pure camera-quality measure:
        framing, occlusion, pose, and movement can all affect it.

        ### `mean_pose`: retain the average shape, discard frame order

        This control averages each landmark's root-centered, torso-scaled x-y
        coordinates over the full clip. As an illustrative example, normalized
        ankle x-values of −0.2, 0.0, and 0.2 have average 0.0 regardless of their
        order. The model cannot tell which happened first. It receives no direct
        visibility features and no explicit motion-variation measurements.
        Average posture can still reflect how long someone occupies different
        positions, body shape, and camera view. It is a useful order-free control,
        not a guarantee that all movement-related information has disappeared.

        ### `majority`: learn only the most common training label

        This rule predicts MS in folds 0, 2, and 3, and PD in folds 1 and 4,
        because those labels have the most **training clips** in those folds.
        It never predicts Normal in this run. It does not use the test labels
        to choose its answer, and it is not a random-guessing baseline.

        ### What is shared, and what is not?

        S-JEPA, visibility, and mean pose each fit their own training-only
        `StandardScaler` and logistic-regression classifier with `C=1.0`,
        `max_iter=2000`, `class_weight='balanced'`, and seed 42. Logistic
        regression combines features with learned weights to score each label;
        despite its name, it is used here for classification. `C` controls how
        strongly large fitted weights are discouraged; its value is fixed here,
        not tuned on test results. `max_iter` is an iteration limit, not a claim
        that every fit takes 2,000 iterations. Class balancing
        reduces the influence of more frequent labels in the fitting loss.

        Class balancing does **not** balance sources within a label. One Fold-0
        MS source supplies 13 of its 23 MS training clips, so it contributes
        13/23 of the MS classifier-loss weight. Source-uniform sampling is used
        for S-JEPA's self-supervised windows, but not for fitting these heads.
        Equal-source evaluation, explained below, does not change that training
        choice. Also, only S-JEPA gets a validation choice between two stages;
        the controls have fixed settings. These are comparisons of complete
        procedures, not perfectly matched feature dimensions or training budgets.
        """), note("training", """
        ## Step 3 · Follow the training and selection procedure

        The saved laptop profile uses 33 landmarks with three channels each:
        normalized x, normalized y, and visibility. Each window has 32 frames
        at a target sampling rate of 15 frames/second, about 2.1 seconds of
        motion. Consecutive windows start 16 frames apart. Four adjacent frames
        form a time token, giving `8 time positions × 33 joints = 264 tokens`.

        The encoder has 3 transformer layers, feature width 96, and 4 attention
        heads; the training predictor has 2 layers and width 96. A layer is a
        processing stage, width 96 means 96 numbers in each token's description,
        and the attention heads learn several ways to combine information from
        different tokens. Training uses
        batches of 32 windows, AdamW with learning rate 0.001 and weight decay
        0.05, and a 10%-of-updates learning-rate warmup. Structured random masks
        aim to target about 60% of tokens, though selecting whole anatomical
        groups can overshoot that target. All three condition groups in the
        training partition supply windows; labels do not enter the encoder loss.

        In each fold we:

        1. Start a fresh model and train for **800 updates** (`ssl`). An update
           is one batch used to adjust weights, not one complete pass through
           all clips.
        2. Freeze the teacher, fit the linear probe on training clips, and
           measure its source-weighted macro-F1 on validation clips.
        3. Continue from those model weights for **400 additional updates**
           (`continued`), then fit and validate a new probe.
        4. Keep the stage with higher validation macro-F1. A tie keeps `ssl`.
           Do not refit the selected probe on validation data.
        5. Evaluate that selected model and the four baselines on test clips.

        The second stage retains model and teacher weights but starts a new
        optimizer, loss center, and learning-rate/teacher schedules. Its seeded
        sampling streams also restart. Thus `800+400` means two stages, not
        necessarily the same optimization path as one uninterrupted 1,200-update
        call. Historical epoch and VICReg fields in the configuration do not
        control this repaired training loop; there is no class-aware VICReg
        or progressive addition of diagnoses here.

        ### Run efficiently without changing the question

        The cell below performs `5 × (800 + 400) = 6,000` updates in a normal
        run. Smoke mode instead uses 4+2 updates and a tiny model; its results
        only check execution. Compatible cached stages and completed folds can
        be reused. CPU-only runs can process separate folds in parallel; one
        MPS/CUDA accelerator trains one fold at a time. Batched operations and
        per-clip feature caching reduce repeated work while fitting and model
        selection remain fold-specific.

        Keep the cache directory stable between reruns and read `computed`
        versus `cache hit` in the output. An unfinished stage restarts; this is
        not mid-update recovery. Data, configuration, code, and environment
        checks prevent incompatible reuse. The [performance tutorial](docs/15-capstone-performance.md)
        explains the safeguards, worker controls, and memory tradeoffs in detail.
        """), original[8], note("selection", """
        ### Read what this run actually selected

        Both stages completed in every fold. The validation scores were:

        | Fold | Original: 800 updates | Continued: 800+400 updates | Stage selected before testing |
        |---|---:|---:|---|
        | 0 | 0.294 | 0.276 | Original |
        | 1 | 0.491 | 0.344 | Original |
        | 2 | 0.325 | 0.257 | Original |
        | 3 | 0.266 | 0.284 | Continued |
        | 4 | 0.375 | 0.560 | Continued |

        Extra training was selected in **two of five folds**. Although the
        final self-supervised loss was lower after continuation in every fold,
        its validation classification score was lower in three. Learning the
        masked-feature task better therefore did not consistently improve this
        label-prediction check. That is a practical reason to select on the
        downstream validation score rather than on training loss.

        The saved execution took **216 seconds, about 3.6 minutes**, on Apple
        MPS, with one fold worker and two CPU feature workers. All five fold
        statuses are `computed`. This is an observed full-run time, not a
        measured speedup: there is no matched before/after timing comparison.
        A compatible cached rerun performs less work and must be labeled as such.
        """), note("metrics", """
        ## Step 4 · Understand the score before looking for a winner

        For MS, **recall** asks, “Of the actual MS examples, how many did we
        recognize?” **Precision** asks, “Of the examples we called MS, how many
        really have the MS label?” F1 balances both: a model is penalized for
        missing MS and for labeling other examples MS. Its formula is
        `2 × precision × recall / (precision + recall)` when the denominator
        is nonzero; this evaluation reports zero for undefined cases.

        **Macro-F1** calculates F1 separately for Normal, MS, and PD, then
        averages the three. For a teaching example, class F1 scores of 0.6,
        0.0, and 0.6 give macro-F1 `(0.6 + 0.0 + 0.6) / 3 = 0.4`. A complete
        failure on MS remains visible even if the other labels do well. Higher
        is better, with 1 meaning perfect predictions, but **0.457 macro-F1
        does not mean 45.7% of clips were classified correctly**.

        We also have to decide how much each clip counts:

        - **Clip-weighted:** each of the 88 clips has weight one. A source with
          13 clips influences the counts 13 times as much as a one-clip source.
        - **Source-weighted:** each source has total weight one, shared among
          its clips. That 13-clip source gives each clip weight `1/13`; a
          one-clip source gives its clip weight one. Total weight is 41.

        We form weighted confusion counts first, calculate each class's F1,
        and then average across classes. We do not average an F1 per source
        or merge its clips into one predicted diagnosis. Giving sources equal
        weight and giving classes equal weight solve different imbalance problems.

        The **pooled** score uses all 88 OOF predictions together. The **fold
        mean** averages five separately calculated scores. Because F1 is a ratio,
        those operations need not give the same answer. **Fold SD** is the
        standard deviation across the five fold scores, describing their spread;
        it is not a confidence interval or a significance test. Their training
        sets overlap. No repeated training seeds are included in this run.
        """), original[10], note("results", """
        ## Step 5 · Interpret the overall results and the individual folds

        Rounded to three decimals, the saved table says:

        | System | Pooled source macro-F1 | Pooled clip macro-F1 | Mean fold source macro-F1 | Fold SD |
        |---|---:|---:|---:|---:|
        | `rf` — current extractor | 0.411 | 0.397 | 0.395 | 0.098 |
        | `sjepa` | 0.457 | 0.397 | 0.438 | 0.196 |
        | `visibility` | 0.318 | 0.318 | 0.285 | 0.104 |
        | `mean_pose` | 0.452 | 0.410 | 0.427 | 0.132 |
        | `majority` | 0.259 | 0.202 | 0.159 | 0.022 |

        **Start with S-JEPA versus mean pose.** The source-weighted difference
        is only about **0.005 F1** on the 0–1 scale. With equal clip weights,
        mean pose leads instead. We have no uncertainty analysis showing a
        reliable S-JEPA advantage, and no equivalence test showing the methods
        are interchangeable. Their similar overall scores can hide different
        predictions and different strengths across sources.

        **Then consider RF.** S-JEPA's pooled source score is about **0.045 F1**
        higher than the saved RF result, but their clip-weighted scores both
        round to 0.397. The identified RF extraction error further limits this
        comparison. It cannot support a broad claim that learned features beat
        a correctly implemented, well-tuned classical gait system.

        **Finally examine the simpler checks.** Both S-JEPA and mean pose score
        above the majority rule in this run. That supports the narrow statement
        that these feature-based procedures outperform this no-input rule here.
        Visibility alone is weaker than either, but its nonzero performance
        means it is a relevant comparison. It neither proves nor rules out
        recording-related shortcuts in S-JEPA. Also, 1/3 is not a universal
        chance level for macro-F1; random-rule performance depends on class
        frequencies and the rule's predictions.

        ### Why the fold breakdown changes the interpretation

        The following source-weighted scores are recalculated from the saved
        OOF predictions, using the same scoring function as the overall table:

        | Fold | `rf` | `sjepa` | `visibility` | `mean_pose` | `majority` |
        |---|---:|---:|---:|---:|---:|
        | 0 | 0.356 | 0.515 | 0.236 | 0.554 | 0.167 |
        | 1 | 0.450 | 0.363 | 0.222 | 0.500 | 0.182 |
        | 2 | 0.553 | 0.780 | 0.272 | 0.187 | 0.133 |
        | 3 | 0.351 | 0.316 | 0.488 | 0.510 | 0.182 |
        | 4 | 0.267 | 0.217 | 0.206 | 0.386 | 0.133 |

        S-JEPA beats mean pose only in Fold 2, where its lead is large. Mean
        pose is ahead in the other four. S-JEPA ranges from **0.217 to 0.780**;
        its average is not a promise about the next unseen source. Fold 4 also
        illustrates why validation is not a guarantee: its selected model scored
        0.560 on validation and 0.217 on test.

        These observations motivate a stability study; they do not tell us
        whether source composition, limited data, camera differences, training
        randomness, or particular movement patterns caused the variation.
        Five overlapping training folds are not five independent clinical trials.
        Do not use this test table to retrospectively change which checkpoint
        each fold selected or to report only the favorable fold.
        """), note("confusion", """
        ## Step 6 · Read the errors, especially for MS

        In each heatmap, a **row is the true dataset label** and a **column is
        the predicted label**. Diagonal cells are correct; off-diagonal cells
        are confusions. For example, the MS row and PD column show MS examples
        that were labeled PD. Darker cells have larger weighted counts.

        These are source-weighted counts. A displayed value of 5.6 is a sum of
        fractional clip weights, not 5.6 people. The row totals before rounding
        are 16 for Normal, 13 for MS, and 12 for PD, matching source counts.
        We display RF and S-JEPA here; all five systems' confusion matrices
        remain available in the saved results.
        """), original[11], note("ms", """
        ### MS is no longer at zero, but important errors remain

        S-JEPA's pooled **source-weighted class F1** is 0.467 for Normal,
        **0.443 for MS**, and 0.460 for PD. For MS, source-weighted precision is
        0.459 and recall is 0.429. Those are different from the raw clip counts.

        Counting clips equally, S-JEPA correctly labels **8 of 29 MS clips**;
        it calls 8 Normal and 13 PD. Thus 21 MS clips are missed, and raw MS
        recall is `8/29 = 0.276`. Of the 21 clips it predicts as MS, 8 really
        have that label, giving precision `8/21 = 0.381` and MS F1 **0.320**.
        The gap from source-weighted MS F1 reflects unequal numbers of clips
        per source. Both views matter; neither should be silently substituted
        for the other.

        Mean pose also recognizes 8 of 29 MS clips, though not necessarily the
        same ones; its source-weighted MS F1 is 0.391. The current RF recognizes
        5 of 29, with source-weighted MS F1 0.272. The learned procedure has not
        become a reliable MS detector merely because its pooled F1 is nonzero.

        Notebook 04's zero MS F1 described two MS **validation** clips in one
        fold. This result describes **test predictions pooled across five folds**.
        They are different samples and evaluation roles. The present result
        shows that the cross-validation procedure does make correct MS
        predictions, but it does not erase the earlier failure or demonstrate
        clinical diagnostic performance.
        """), note("conclusions", """
        ## Step 7 · State the conclusions and their contribution value

        A **research contribution** is something another researcher can learn
        from or reuse. It need not be a higher score. However, a disappointing
        score alone is not an explanation, and a careful implementation is not
        automatically a new scientific method.

        The value assessments below are judgments about this evidence, not
        probabilities of paper acceptance.

        | Specific conclusion supported now | Evidence | Value as a contribution, and its boundary |
        |---|---|---|
        | The evaluation can compare all five procedures on the same held-out sources. | 88 unique OOF rows cover the registry exactly; recalculation reproduces all reported aggregate metrics. | **Useful reproducibility foundation.** Grouped splitting and OOF scoring are established methods, so their implementation alone offers limited methodological novelty. It does not verify person-level independence. |
        | An advantage over average pose has not been demonstrated. | S-JEPA 0.457 versus mean pose 0.452 source macro-F1; mean pose leads in four folds and in the clip-weighted aggregate. | **Strongest candidate empirical contribution.** A simple order-free control changes the interpretation of a learned-model result. Its generality needs repeated seeds and targeted controls. It does not prove temporal learning is useless or that the two methods are equivalent. |
        | The apparent model ordering depends on the evaluation weights. | S-JEPA leads with equal source weights; mean pose leads with equal clip weights. | **Useful evaluation lesson for this collection.** Report both and explain the unit. The general principle is familiar, so this is supporting evidence rather than a new metric. |
        | More self-supervised training does not consistently improve validation classification in this recipe. | Continued loss is lower in all folds, but validation F1 improves in only two of five. | **Useful training-selection observation.** It supports checking downstream performance. It does not establish that 800 updates is universally optimal or that other continuation recipes fail. |
        | Performance varies substantially across the held-out source groups. | S-JEPA fold F1 ranges from 0.217 to 0.780; SD is 0.196. | **Important limitation and research question.** Identifying the reason could become a stronger contribution; variation by itself does not identify a failure mechanism. |
        | Useful label prediction remains limited, including for MS. | Pooled source macro-F1 is 0.457; 21 of 29 MS clips are missed under raw clip counting. | **Honest boundary on application claims.** This supports a pilot study, not deployment, screening, verified biomarkers, or patient diagnosis. |
        | The intended RF comparison is not yet fully validated. | The left-ankle range feature duplicates the right-ankle range. | **Necessary correction, not a scientific success.** The current score is auditable, but fixing a coding error alone is not evidence against classical gait methods. |

        The most defensible central conclusion is:

        > On this small source-grouped collection, a validation-selected S-JEPA
        > procedure did not demonstrate a consistent advantage over an average-pose
        > classifier. The comparison shows why an order-free control, reporting
        > both source and clip weighting, and examining every fold are important
        > before attributing performance to learned temporal gait information.

        That is a bounded finding about this dataset and recipe. We have not
        measured a causal benefit from temporal learning, proved that visibility
        is the shortcut, established statistical equivalence, or discovered a
        disease-specific movement marker. Those would require additional tests.
        """), note("paper", """
        ## Step 8 · Are we ready for a worthwhile workshop paper?

        **We have enough to start a focused pilot-paper draft, but not yet enough
        to call the methodology fully validated or claim a superior model.**
        The work already has concrete strengths: retained predictions, explicit
        split boundaries, simple controls, and an informative result that resists
        an overly favorable interpretation. The lack of a decisive S-JEPA win
        does not make the work wasted. It changes the paper's question from
        “Did we beat the baseline?” to “What does this evaluation actually show
        about the value of learned motion features?”

        Our strongest direction is an **empirical study of controls and limits
        in small-data skeleton representation learning**. S-JEPA is an existing
        method, so applying it here is not, by itself, a new architecture; see
        the [original S-JEPA project](https://sjepa.github.io/). Standard caching
        and parallelism are useful supporting engineering. Without matched
        timing and correctness comparisons, we should not make them a measured
        acceleration contribution either.

        Before submitting, prioritize these bounded additions rather than an
        open-ended search for a higher score:

        1. **Correct and test the RF extractor, then rerun the affected comparison.**
           Preserve this run as the original and label the corrected version.
           A regression test should use different known left/right ankle ranges.
           Do not hide an unfavorable corrected result.
        2. **Test what learning and time order contribute.** Compare the trained
           encoder with an untrained encoder using the same readout and probe.
           Add a predefined temporal-order ablation, with its training and
           evaluation transformations stated explicitly. Shuffling only test
           frames measures sensitivity to changed input, not necessarily the
           benefit of temporal learning. These studies have not been run here.
        3. **Check stability and paired uncertainty.** Repeat a declared set of
           training seeds and report them all. Compare systems on the same
           held-out sources. Any bootstrap should resample whole sources, not
           treat related clips as independent. A bootstrap of fixed OOF
           predictions is conditional on these fitted models; it does not
           capture all training or split-selection uncertainty.
        4. **Review the collection and release responsibly.** Check duplicates,
           possible repeat participants, label provenance, exclusions, and
           source-media permissions. Release allowable code, split manifests,
           configurations, and predictions with clear reproduction instructions.
           Do not infer video rights from the code's license.

        Since we have now inspected these test results, improvements motivated
        by them are further development on this collection. Freeze the next
        comparisons before running them, disclose the development history, and
        use an untouched external collection if making a stronger generalization
        claim. More seeds improve a stability assessment; they do not create
        more participants or remove recording-related bias.

        ### Which workshop fits?

        An ICLR workshop on representation learning, small-data evaluation,
        temporal data, or health applications is the closest thematic direction.
        As checked on **September 20, 2026**, the official 2027 call schedules
        workshop-selection notifications for **November 29, 2026** and suggests
        **February 1, 2027** for contribution submissions. Those are not a
        confirmed deadline for a chosen workshop. The October 9 deadline is
        for people proposing to **organize** a workshop, not for our paper.
        Check the eventual individual call. [Official ICLR workshop call](https://iclr.cc/Conferences/2027/CallForWorkshops).

        Reputable IEEE agent workshops also have upcoming calls, but this study
        currently evaluates neither an agent nor an LLM. An agent wrapper would
        not establish an agent-research contribution. A better IEEE alternative
        to investigate is **PerFail 2027**, which invites substantive lessons
        from negative results in pervasive computing, with a listed paper
        deadline of **November 17, 2026**. Its call explicitly excludes findings
        based merely on coding bugs; a corrected study and a convincing sensing
        connection would be essential. [PerFail call](https://perfail-workshop.github.io/).

        The [submission assessment](docs/16-capstone-results-and-contributions.md#workshop-options)
        compares ICLR, ACM, and IEEE possibilities with verified links. A suitable
        workshop could value the bounded empirical lesson, but acceptance cannot
        be predicted from these scores. Our next milestone is a corrected,
        reproducible comparison with a clear explanation of what was learned.
        """)]
