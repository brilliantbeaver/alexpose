"""Tutorial notes for notebook 05's retained laptop/fold-0 run.

Evidence: saved t-SNE and UMAP cell outputs and printed silhouette scores;
training_embeddings.npz SHA-256:
a02aab4dd50036b688480f5aea88e520337b674240ad7de078031342279f471b.
The NPZ's clip/source/label ordering and registry match the training partition.
Its learned embeddings agree with CPU inference from the existing checkpoints
(maximum absolute difference below 1e-6); visibility features match exactly.
Silhouettes recomputed from these standardized vectors:
ssl -0.0043661827221512794; continued -0.011896262876689434;
visibility -0.004153621848672628.
This module adds prose only and runs no models or projections on import.
"""

from textwrap import dedent


def add_tutorial(original, md):
    """Surround the original executable cells with explanations of their results."""
    stem = "05_representation_visualization"
    for i, cell in enumerate(original):
        cell.setdefault("id", f"{stem}-{i:02d}")

    def note(key, text):
        cell = md(dedent(text).strip())
        cell["id"] = f"05-tutorial-{key}"
        return cell

    return [original[0], note("intro", """
        # 05 · What do the learned descriptions of walking look like?

        Imagine making a card for each walking clip. Each card contains a list
        of numbers describing that clip. Two clips with similar lists have
        similar **representations**, also called **feature vectors**. S-JEPA
        learned how to produce those lists in notebooks 03 and 04.

        Here we ask: **do clips with the same dataset label tend to have similar
        representations, and did the extra training make that grouping clearer?**
        We make two kinds of map and calculate a grouping score called
        **silhouette**. Each map dot represents one clip.

        The saved results show mixed labels and silhouette scores close to zero.
        Extra training did not improve the measured grouping by condition. We
        will work through how to read that finding, why it matters, and why it
        is different from notebook 04's classification score.

        Follow the tutorial in order: identify the clips, understand their
        feature vectors, read the maps, understand silhouette, then connect
        the evidence across the three notebooks. All numbers and observations
        below describe this saved laptop-profile, Fold 0 run.
        """), *original[2:7], note("examples", """
        ## Step 1 · Know what one dot represents

        Look at **Fold 0's training row** in the table above. This notebook uses
        its **51 clips from 24 source videos**: 15 Normal clips, 23 MS clips,
        and 13 PD clips. The legend uses blue for Normal, orange for MS, and
        green for PD. These colors are the clips' known dataset labels, not
        predictions made by a classifier.

        A clip may contain several overlapping motion windows. We summarize
        those windows into one vector, so a long clip still contributes one
        dot. A source video may contribute several clips, however. Those clips
        appear as several dots even though they come from the same recording.
        In this fold, **13 of the 23 MS clips come from one source**. A cluster
        of orange dots could therefore reflect a shared recording as well as
        movement associated with the label. The current plots do not identify
        sources, so they cannot distinguish these explanations.

        These are the encoder's **training clips**. Notebook 05 excludes the
        validation and test partitions from feature extraction, scaling, maps,
        and silhouette calculations. We are inspecting patterns in material the
        encoder has practiced on. We will need held-out evaluation to find out
        whether useful patterns transfer to different sources.
        """), note("features", """
        ## Step 2 · Compare three descriptions of the same clips

        The three panels use the same 51 clips in the same order. What changes
        is the list of numbers used to describe each clip:

        | Name in the code and plots | Description of one clip | Numbers per clip |
        |---|---|---:|
        | `ssl` | Features from the original 800-update checkpoint | 96 |
        | `continued` | Features after a further 400-update training stage | 96 |
        | `visibility` | Mean and standard deviation of each landmark's visibility score | 66 |

        For the learned descriptions, the helper passes each complete skeleton
        window through the saved **target encoder**, the slowly updated teacher.
        It averages features from a fixed set of joint-time positions and then
        averages across the clip's windows. This is the same feature-extraction
        recipe used in notebook 04. The fixed mask selects **output positions
        to average**; it does not hide the input joints at this stage. The 96
        features are learned numbers, not 96 named gait measurements.

        Visibility provides a simpler comparison. For each of 33 landmarks,
        we measure its average pose-detector visibility and how much that value
        varies during the clip: `33 × 2 = 66` numbers. For example, a frequently
        obscured ankle may have a lower average visibility or greater variation.
        This description does not use the landmark's x-y coordinates directly.
        Camera framing, occlusion, and pose can all affect it, so it is a check
        for alternative cues rather than a pure measure of camera quality.

        The next cell loads the trained checkpoints and saves these descriptions
        in `training_embeddings.npz`, together with clip names, sources, labels,
        and split information. It does not train the encoder again. The saved
        output, `training clips plotted: 51`, confirms the number of vectors
        prepared for each panel.
        """), original[7], note("maps", """
        ## Step 3 · Turn a long list of numbers into a two-dimensional map

        We cannot directly draw a point with 96 coordinates on a flat page.
        **t-SNE** is a method that places those points on a map while trying
        to preserve nearby relationships. It is a simplified view of the
        features, so some information is inevitably lost.

        First, `StandardScaler` puts each feature on a comparable scale using
        these training clips. For each feature it subtracts its mean and divides
        by its standard deviation when that deviation is nonzero. Each of the
        three representations gets its own scaler. Then t-SNE makes a separate
        map for each one. Its `perplexity=15` controls the scale of neighborhood
        relationships, and `random_state=42` fixes the seed. The condition labels
        are used only to color the finished map. This code does not tell t-SNE
        to create three condition groups. See the [scikit-learn explanation of
        t-SNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne).

        When looking at each panel:

        1. Find a small group of nearby dots. Are its colors similar or mixed?
        2. Check whether that pattern holds throughout the panel. One orange
           patch does not mean all MS clips form a distinct group.
        3. Compare the pattern of color mixing across panels, rather than the
           exact dot positions. Each map has its own coordinate system.

        The horizontal and vertical axes do not represent walking speed,
        disease severity, or time. A dot on the right of one panel is not
        necessarily more similar to a dot on the right of another panel.
        """), original[9], note("tsne-results", """
        ### Read the saved t-SNE figure

        In the **original `ssl` panel**, blue, orange, and green dots occur
        near one another in several regions. There are local patches with
        more of one color, but the panel does not show three clean groups
        corresponding to the three conditions.

        In the **`continued` panel**, the arrangement changes. A small green
        group appears near the upper right, but other green dots remain near
        orange and blue dots elsewhere. Extra training has changed the feature
        geometry; that particular green patch is insufficient evidence that
        all PD clips have become easy to distinguish.

        In the **visibility panel**, orange dots are relatively common toward
        the upper right, while colors overlap elsewhere. This reminds us that
        a map can show structure even when it is based only on visibility
        summaries. We cannot conclude from this picture that the learned
        encoder relies on visibility, or that visibility predicts labels well
        on unseen sources. Those are separate questions to test.
        """), note("umap", """
        ## Step 4 · View the features through a second mapping method

        **UMAP** makes another two-dimensional map of the same standardized
        vectors. Here it uses `n_neighbors=15` and `min_dist=0.3`, settings that
        affect neighborhood relationships and how closely dots can pack. It
        also uses seed 42. It is a second view of the same evidence, not an
        independent dataset or a second classification experiment.

        UMAP and t-SNE can make gaps look stronger or split a continuous group
        into apparent islands. The [UMAP documentation explains this limitation](https://umap-learn.readthedocs.io/en/latest/clustering.html).
        Look for whether the islands actually match the label colors before
        interpreting them as condition groups.
        """), original[10], note("umap-results", """
        ### Read the saved UMAP figure and its warnings

        The **original panel** again mixes colors, including within its small
        upper group. The **continued panel** has a striking gap between a small
        group at the lower left and a larger group on the right. However,
        **both sides contain multiple label colors**. That is the key reading:
        a large gap in the map does not by itself separate Normal, MS, and PD.

        The **visibility panel** also contains all three colors distributed
        through the map. Taken together, these figures do not show a consistent
        division into three condition-specific groups. They also cannot prove
        that no useful information remains in the full feature vectors.

        This cell completed and displayed its figure despite the messages above
        it. `IProgress not found` concerns the notebook's progress-bar widget.
        The `n_jobs` warning concerns UMAP's use of a fixed seed and single-worker
        execution. Neither message reports failure of this completed projection.
        They do not explain the mixed colors or the low silhouette values below.
        """), note("silhouette", r"""
        ## Step 5 · Measure grouping with a silhouette score

        Silhouette asks whether clips are closer, on average, to clips with
        their own label than to clips in the closest other label group.
        Here the groups are the known Normal, MS, and PD labels; the code
        does not first discover groups with a clustering algorithm.

        For a particular clip, let **a** be its average distance to the other
        clips with its label. Let **b** be the smaller of its average distances
        to the two other label groups. Its score is:

        $$s = rac{b-a}{max(a,b)}.$$

        For illustration, suppose an MS clip has `a=2`, average distance 5 to
        Normal, and average distance 6 to PD. Then `b=5` and its score is
        `(5−2)/5 = 0.6`: it is closer to its own group. If instead `a=5` and
        `b=2`, the score is `−0.6`. These are teaching examples, not our results.

        A value near **+1** indicates strong separation for that point; near
        **0** means similar within-group and nearest-other-group distances;
        a **negative** value means the other group is closer on average.
        The notebook reports the mean across clips. The [scikit-learn metric
        documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)
        describes this calculation.

        Crucially, the code calculates distances in the **standardized original
        feature vectors**: 96 dimensions for each learned representation and
        66 for visibility. It uses `scaled`, not the two-dimensional `xy` map.
        The printed score therefore measures the features behind the picture,
        rather than scoring the appearance of the picture itself.

        Each clip counts equally in this mean. Unlike notebook 04's validation
        score, it does **not** give every source equal total weight or every
        condition one third of the final average. Sources with more clips can
        influence both the distances and the average more strongly.
        """), original[11], note("scores", """
        ## Step 6 · Interpret our three scores

        The saved output reports:

        | Representation | Training silhouette | Interpretation for this run |
        |---|---:|---|
        | Original, `ssl` | **−0.004** | Little average separation by condition under this distance measure |
        | Extra training, `continued` | **−0.012** | Slightly lower measured separation after continuation |
        | Visibility summaries | **−0.004** | Also little average separation by condition |

        These values are all close to zero. They provide little evidence that
        each condition forms a compact group well separated from the other
        conditions in the standardized feature space. A mean near zero can
        also combine positive scores for some clips and negative scores for
        others; it does not mean every clip has the same relationships.

        The original and visibility scores match at the three decimal places
        printed here. They are close, not exactly equal. That similarity does
        not establish that the two representations encode the same information
        or would achieve the same classification score.

        Continuation lowers silhouette by about **0.008**. This goes in the
        same unfavorable direction as notebook 04's overall validation macro-F1
        change, but the two scores measure different things. No uncertainty
        estimate or repeat-run comparison is provided for the silhouette change,
        so we should not describe it as a statistically established deterioration.

        A negative silhouette is possible and valid. It is not negative
        accuracy, a percentage of missed clips, or evidence that a label was
        entered incorrectly. There is also no fitted classifier in this cell.
        Classification can use particular combinations of features even when
        the labels do not form compact groups under overall Euclidean distance.
        """), note("progress", """
        ## Step 7 · Connect the results across notebooks 03, 04, and 05

        Each notebook asks a different question:

        | Notebook | Question | Evidence so far |
        |---|---|---|
        | 03: training | Does the model improve at predicting teacher features and retain varied embeddings? | Loss falls and effective rank stays well above one |
        | 04: validation | Can a fitted classifier use the features to recognize labels on different sources? | Macro-F1 is 0.294 for the original model and 0.276 after continuation; the original model misses both MS validation clips |
        | 05: inspection | Do the training features visibly and numerically group clips by condition? | Maps mix colors and training silhouette stays near zero for all three representations |

        **The learning process is working, but useful separation of the condition
        labels remains weak in the checks performed so far.** Notebook 03 showed
        that the model could learn its masked feature-prediction task. That task
        never required its features to arrange the three condition labels into
        distinct groups. Notebook 05 helps make that distinction visible.

        There is no contradiction between seeing some orange neighbors here and
        obtaining MS F1 of zero in notebook 04's original model. This figure
        contains 23 **training** MS clips, many sharing a source. The two missed
        MS clips came from **different validation sources** and do not appear
        in these maps. A familiar recording can look consistent within training
        while a new recording is still misclassified.

        The continued model's extra training has not produced a better overall
        result in this example: its validation macro-F1 and training silhouette
        are both lower. That supports retaining notebook 04's original checkpoint
        under the existing selection rule. It does not identify why transfer is
        weak or establish that every longer training recipe would fail.

        Visibility is a useful comparison because it asks whether simple
        detection-related information also has structure. Its near-zero score
        does not prove recording conditions are harmless, and similarity between
        maps does not prove the encoder learned a camera shortcut. We still need
        controlled comparisons on held-out sources to assess those possibilities.
        """), note("next", """
        ## Step 8 · Decide what this inspection justifies next

        A useful next development check is to label or color training dots by
        **source video**, or to inspect a source-balanced sample. That would
        help distinguish repeated-recording groups from groups spanning several
        independent sources. It is a proposed check, not a result shown here.
        Inspecting skeleton quality and the most confusing training examples can
        also help formulate a specific improvement to test.

        Keep the saved seed and settings as the reference. Trying many projection
        settings until the colors look separated would not demonstrate improved
        classification. Any sensitivity study should report the variation,
        including views that make the grouping look less convincing.

        Notebook 06 tests the fixed training-and-selection procedure across five
        source-grouped folds, with a fresh model in each fold. Its Random Forest,
        visibility, mean-pose, and majority-label comparisons help judge whether
        S-JEPA offers an advantage over simpler alternatives. Choose any changes
        to the development recipe before using test results to assess it.

        The conclusion supported here is: “For 51 training clips from 24 sources,
        the original, continued, and visibility representations had silhouette
        scores near zero, and both mapping methods showed mixed condition labels.
        The additional training stage did not improve this grouping measure.
        These are descriptive training results; performance on held-out sources
        must be assessed separately.”
        """)]
