"""Teaching commentary for notebook 04; preserve its executed cells and outputs.

Evidence: saved laptop/fold-0 output, registry 9496e61b050f. A CPU replay of
the existing validation pipeline (torch 2.13.0, sklearn 1.9.0) reproduced both
printed aggregate scores exactly and supplied the explicitly labeled per-class
breakdown. Checkpoint SHA-256 values for that check:
ssl: 4bccf2a730f71fba4e366f884372fc0863545578c4fd6d9535a95d20bebc0d9d
continued: 92df9bbcedae0b02d104f43a1e85dcb0ee88d82ce9ae16a6ca0616dabda6aae6
No encoder training or test scoring is performed by this module.
"""

from textwrap import dedent


def add_tutorial(original, md):
    """Add explanations around the original thirteen cells without editing code."""
    stem = "04_progressive_finetune_ms_pd_vicreg"
    for i, cell in enumerate(original):
        cell.setdefault("id", f"{stem}-{i:02d}")

    def note(key, text):
        cell = md(dedent(text).strip())
        cell["id"] = f"04-tutorial-{key}"
        return cell

    return [original[0], note("intro", """
        # 04 · Does additional training help classify walking clips?

        Notebook 03 showed that S-JEPA could improve at predicting hidden motion
        features while retaining variation in its representations. We now ask a
        different question: **do those learned features help a simple classifier
        distinguish the dataset's Normal, MS, and PD labels on new source videos,
        and does another training stage improve that result?**

        The saved Fold 0 comparison favors the original checkpoint: validation
        macro-F1 is **0.294 after the first 800 updates**, compared with **0.276
        after 400 additional updates**. The extra training did not improve the
        chosen overall score in this run. This is a useful result for selecting
        a training budget, although classification remains weak on this small
        validation set.

        We will follow the data split, compare the two checkpoints, explain how
        learned features become label predictions, and read the results one claim
        at a time. The numerical interpretation describes the saved laptop run;
        a different run must be interpreted using its own outputs.

        The filename retains older terminology. The active experiment continues
        label-free training on all three conditions in the training partition.
        It uses no class-aware VICReg loss. Condition labels are introduced later
        when fitting the small classifier, called a **linear probe**.
        """), *original[2:7], note("partitions", """
        ## Step 1 · Identify what each data partition is allowed to do

        The table above lists five folds, but this notebook works with **Fold 0**.
        A source video can produce several clips, and each clip can produce
        overlapping windows. Keeping all of them in one partition prevents the
        same recording from serving as both practice material and an evaluation.

        | Fold 0 partition | Clips | Source videos | Role in this notebook |
        |---|---:|---:|---|
        | Training | 51 | 24 | Train the encoder; fit each scaler and classifier |
        | Validation | 19 | 8 | Compare the two trained candidates and choose one |
        | Test | 18 | 9 | Reserved for the evaluation procedure in notebook 06 |

        Validation contains **3 Normal clips from 3 sources, 2 MS clips from
        2 sources, and 14 PD clips from 3 sources**. The many PD clips therefore
        represent only three source videos. They do not provide fourteen
        independent recordings. Source IDs also do not establish unique people.

        The next cell loads `ssl.pt`, the checkpoint from notebook 03, and checks
        that its configuration and training membership match this split. If the
        cell displays a long dictionary of tensors, those are stored weights and
        training state returned by the loader. The classification results appear
        in step 5 below.
        """), original[7], note("continuation", """
        ## Step 2 · Give the same model another stage of practice

        We compare two candidates that share the first training stage:

        | Checkpoint | How it was trained | Total updates in this workflow |
        |---|---|---:|
        | `ssl.pt` | Original label-free training from notebook 03 | 800 |
        | `continued.pt` | Start from `ssl.pt`, then train on the same sources for 400 more updates | 1,200 |

        An **update** means drawing a batch, making masked prediction targets,
        calculating the loss, and adjusting the model once. With the laptop
        batch size of 32, the extra stage draws `400 × 32 = 12,800` windows,
        including repetitions. No new source videos or condition labels are
        introduced into the encoder loss.

        The learned student, predictor, mask token, and teacher weights carry
        forward. The optimizer, learning-rate schedule, and running feature
        center start fresh. Thus this comparison measures the effect of this
        particular **400-update continuation recipe**. It is not the same as
        training from scratch for 1,200 updates under one uninterrupted schedule.
        Smoke mode instead compares 4 initial updates with 2 extra updates and
        is only an execution check.

        The next cell saves `continued.pt` and may finish without printing a
        message. It does not calculate classification performance. Rerunning it
        trains the current in-memory model again and replaces that checkpoint;
        reload `ssl.pt` first if you intend to repeat the same comparison.
        """), original[9], note("probe", """
        ## Step 3 · Turn a clip into features, then predict its label

        Each candidate gets its own scaler and linear probe. During this part,
        the encoder is **frozen**: fitting the classifier does not change its
        learned weights. Think of the encoder as describing a walk with numbers
        and the probe as learning how to combine those numbers into a label.

        The comparison cell follows the same procedure for both checkpoints:

        1. Split each clip into windows and pass each complete window through
           the **target encoder**, the EMA teacher learned during pretraining.
        2. Average features from a fixed set of joint-time token positions, then
           average the window vectors into one vector per clip. The laptop
           profile produces 96 numbers per clip. The fixed seed-0 readout selects
           output tokens to average; it does not hide input coordinates here.
        3. Fit a **scaler** on the 51 training vectors. It puts feature dimensions
           on comparable scales using training means and standard deviations.
        4. Fit **logistic regression**, a classifier that learns weighted sums
           of those features, using the training condition labels. Both candidates
           use the same settings: `C=1`, balanced class weights, and at most
           2,000 solver iterations.
        5. Apply that fitted scaler and classifier to validation clips. Their
           true labels are used to score predictions and choose the checkpoint.

        Class balancing increases the fitting weight of less common **training
        labels**. It differs from the equal-source weighting used for validation
        below. The probe is trained on clips and is not fitted with equal-source
        weights. Neither its scaler nor its classifier is fitted on validation.

        The result measures how useful each frozen representation is to this
        particular simple classifier. A representation can contain variation
        that helps masked prediction but contributes little to label separation.
        """), note("metric", r"""
        ## Step 4 · Understand the score before comparing numbers

        **Precision** asks: of the clips predicted as MS, how many really have
        the MS dataset label? **Recall** asks: of the clips labeled MS, how many
        did the classifier find? We calculate these quantities for each of the
        three labels. F1 combines precision and recall:

        $$F1 = \frac{2\,\text{precision}\,\text{recall}}
                     {\text{precision}+\text{recall}}.$$

        For an illustrative example, precision 0.75 and recall 0.50 give F1 0.60.
        This example only explains the calculation; it is not a measured result.
        F1 rewards both finding the labeled clips and avoiding false predictions.
        The code uses zero for an undefined class score.

        **Macro-F1** is the simple average of the Normal, MS, and PD F1 scores.
        Each condition therefore receives one third of the final score. Scores
        range from 0 to 1, with larger values better. An F1 of 0.294 does not
        mean that 29.4% of clips were classified correctly; accuracy is a
        different measure.

        **Source weighting** operates before computing those class scores. Each
        source contributes total weight one, divided equally among its clips.
        For example, the validation source with seven PD clips gives each clip
        weight `1/7`; a source with one clip gives it weight `1`. This prevents
        a long recording from counting seven times as much in the score. The
        code still scores individual clip predictions, rather than voting for
        one label per source or averaging eight separate source F1 scores.

        Selection uses the full-precision scores: choose `continued` only when
        its source-weighted macro-F1 is strictly higher. A tie keeps `ssl`.
        """), original[11], note("results", """
        ## Step 5 · Read the completed comparison

        The saved output above reports the following validation results for
        the laptop profile and Fold 0:

        | Candidate | Source-weighted macro-F1 | Decision |
        |---|---:|---|
        | Original checkpoint, `ssl` | **0.293979** | Selected |
        | Extra training, `continued` | 0.276105 | Not selected |
        | Change, continued minus original | **−0.017874** | Lower overall score after continuation |

        On a scale from 0 to 100, this is a decline of about **1.79 F1 points**.
        Extra training used 50% more encoder updates, yet did not improve the
        selection score. The rule therefore prints `Selected stage: ssl`.
        That identifies the better of these two candidates for this fold;
        it does not certify that either candidate performs well enough.

        The printed choice is a stage name. The loop's `m` and `probe` variables
        still refer to its final iteration, `continued`. Any later use of the
        winner must explicitly load `ssl` and fit or retain its matching probe.
        Notebook 06 handles that selection before test prediction.

        The class breakdown below comes from an **additional validation check
        of the two saved checkpoints**, using the same embedding, probe, and
        scoring functions on CPU. Both aggregate scores exactly matched the
        notebook's saved output. No encoder retraining or test scoring was
        needed. The original cell prints only the aggregate, so this table
        supplies detail beyond that output.

        | Dataset label | Original F1 | Continued F1 | Change |
        |---|---:|---:|---:|
        | Normal | 0.641 | 0.393 | −0.247 |
        | MS | 0.000 | 0.302 | +0.302 |
        | PD | 0.241 | 0.133 | −0.108 |
        | Macro average | **0.294** | **0.276** | **−0.018** |

        Values and differences are rounded independently from the full-precision
        calculations. All class F1 values use source weights.

        The change is a **trade-off across labels**. The original model correctly
        labels two of the three Normal validation clips but misses both MS clips.
        The continued model correctly labels one of the three Normal clips and
        one of the two MS clips. PD F1 also falls. The improvement for MS is
        outweighed by the combined declines for Normal and PD, so the macro
        average decreases. Choosing only the improved MS score afterward would
        change the selection rule after seeing the results.
        """), note("interpretation", """
        ## Step 6 · What this says about the experiment so far

        **Training is functioning, but the current classification result is
        weak.** Notebook 03's decreasing prediction loss and varied embeddings
        were useful checks of the learning process. Notebook 04 asks whether
        those features transfer to distinguishing labels on different sources.
        The original checkpoint's macro-F1 of about 0.294, including zero MS F1
        in this fold, shows that healthy training diagnostics have not yet
        translated into reliable three-condition classification here.

        **The extra stage did not help the chosen overall objective in this
        comparison.** Its MS improvement is real for these validation clips,
        but the overall result favors keeping the earlier checkpoint. This is
        useful feedback: simply adding this training stage is not supported as
        an improvement by the current selection score.

        **We cannot identify the cause from these scores alone.** Possible
        explanations include variation that is poorly aligned with the labels,
        differences between training and validation recordings, or the restarted
        optimization schedule. Overfitting is one possibility, but two validation
        scores without matching training comparisons do not establish it. The
        scores also do not prove encoder collapse or that all longer training
        budgets would perform worse.

        **The validation evidence is small.** There are only eight source videos,
        including two MS sources. A few changed predictions can materially move
        the scores, and the clips within a source are related. This run provides
        no confidence interval or repeated-seed estimate for the 0.018 difference.
        The deterministic selection rule can still choose `ssl`; that choice is
        not a statistical significance test.

        Do not interpret 1/3 as a universal chance threshold for macro-F1 just
        because there are three labels. A baseline depends on the prediction
        rule, label frequencies, and weighting. This comparison does not include
        a matched baseline or an untrained encoder, so it cannot establish that
        pretraining beats either of them.

        A result statement supported by the evidence is: “On Fold 0's validation
        set of 19 clips from eight sources, the original checkpoint achieved
        source-weighted macro-F1 0.294, compared with 0.276 after the specified
        400-update continuation stage. We retained the original checkpoint.
        Classification remained weak, and the benefit of the approach on held-out
        test sources remains unresolved by this notebook.”
        """), note("next", """
        ## Step 7 · Use this result to guide the next evaluation

        Notebook 05 can inspect training representations and acquisition-related
        features for clues about the weak label separation. Those pictures may
        suggest hypotheses; they do not replace performance measurement.

        Notebook 06 evaluates the fixed procedure across all five source-grouped
        folds. It trains fresh models in each fold, chooses between the two
        checkpoints using that fold's validation sources, then measures the
        selected candidate on that fold's test sources. Other folds may choose
        a different checkpoint. It also compares Random Forest and fixed controls
        based on visibility, mean pose, and the most common training label.

        Those comparisons will help answer whether the method works across
        sources and whether simpler recording cues explain similar performance.
        Keep the evaluation procedure fixed before reading test results. If
        validation exploration leads to a new recipe, document that change before
        testing it. The scores above are development evidence used for a choice;
        they should not be reported as the final test performance.
        """)]
