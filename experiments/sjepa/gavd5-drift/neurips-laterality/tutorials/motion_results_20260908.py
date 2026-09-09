"""Dated interpretation of saved results; no execution or artifact loading.

The notebook builders include this prose so regeneration retains the analysis.
It is a historical snapshot, not a claim about outputs from a subsequent run.
"""
from textwrap import dedent

from nbformat.v4 import new_markdown_cell


GRID = "artifacts/motion_structured/grids/292443b0fab5339f5da7ca566a85d6172ffc5b64abe5febf2546681a0152ff57"
ARCHIVED_15 = "executed/motion_structured/gavd_5yc3ve5h/15_motion_weighted_masking.ipynb"

OPENING = {
    15: """
    ### What to look for in the reviewed results — 2026-09-08

    Follow two questions separately: did motion weighting change the targets,
    and did that change improve held-out movement readout? The retained real
    mask audit answers the first positively; the completed grid in Notebook 18
    does not demonstrate the second. The current copy of Notebook 15 had no
    saved code outputs when reviewed. Section 6 explicitly cites the archived
    audit and the completed grid; no execution outputs have been inserted here.
    """,
    16: """
    ### What to look for in the reviewed results — 2026-09-08

    The saved figures establish that connected regions remove immediate
    temporal brackets and reduce visible anatomical neighbors. Check whether
    this intervention also improves readout: Notebook 18's completed region
    comparison does not demonstrate a benefit. Section 6 below interprets the
    actual audit, explains the different mask denominators, and prioritizes
    the next investigation. All existing figures and tables are retained.
    """,
    17: """
    ### What to look for in the reviewed results — 2026-09-08

    The full motion/region grid has completed: Notebook 18 and its saved
    artifacts verify 50 paired jobs, 125 trained encoders and 150,000 updates.
    This copy of Notebook 17 had no saved code outputs when reviewed, so
    Section 6 uses explicitly identified artifacts to interpret training.
    Falling prediction loss and useful movement readout give different answers
    here. The completed grid used **CUDA BF16 training and FP32 evaluation**;
    that setting matters when reopening it from the configuration below.
    """,
    18: """
    ### What to look for in the reviewed results — 2026-09-08

    The complete real-data grid supports a specific conclusion: temporal
    summaries improve the initial encoder's readout more than the trained
    encoders' readout. Initial mean-motion features reach mean R² **0.223**;
    trained teachers reach **0.101–0.114**, with every teacher arm below its
    matched initial control in all five seeds. Mask-versus-uniform intervals
    include zero. Section 9 explains these results, the positive predictor
    diagnostic, and a concrete sequence of follow-up analyses. It also states
    what the evidence cannot distinguish. Existing outputs remain unchanged.
    """,
}

ANALYSIS = {
    15: r"""
    ## 6. Interpretation of the retained results and how to proceed

    **Evidence snapshot: 2026-09-08.** This source notebook has no retained
    execution outputs. The most recent retained real-data execution found for
    Notebook 15 is the [archived GAVD audit]({archive}); it reports 37,500
    training-clip/fold/seed/arm draws. The numbers below transcribe that audit,
    rather than asserting a new execution of the current source. Downstream
    scores come from Notebook 18's [completed grid summary]({grid}/summary.csv).

    ### Step 1: Check what the sampler actually changed

    | Arm | Mean targets per clip | Mean fraction of valid tokens hidden | Target minus eligible motion | Both legs targeted |
    |---|---:|---:|---:|---:|
    | Uniform | 80.771 | 0.170 | -0.000105 | 1.000 |
    | MAMP convention | 80.771 | 0.170 | +0.01749 | 0.999 |
    | Robust motion mixture | 80.771 | 0.170 | +0.03784 | 1.000 |

    These are averages of source-balanced fold/seed summaries, rounded as in
    the archived display. The common target counts support a matched-budget
    comparison. The fraction is about **17% of valid all-landmark tokens**,
    despite `mask_fraction=0.5`: the count is derived from the twelve-landmark
    gait budget and then sampled over all 33 landmarks. Do not describe this
    as a 50% all-token mask experiment.

    In the archived seed plot, robust motion stays above MAMP and uniform
    stays near zero. This shows a repeatable change in target selection under
    the common robust displacement diagnostic. Because that diagnostic is
    closely related to the robust sampler's own score, its larger enrichment
    is not an independent measure of tracking quality or useful semantics.
    The nearly saturated bilateral-coverage plot is a coarse check: targeting
    one token on each side neither balances their counts nor verifies quality.
    The first-clip panels each contain 90 targets; that example is not the
    80.771-target average. Fixed audit batches also differ from training's
    random source-balanced batches.

    ### Step 2: Test the learning consequence using the same readout

    The completed CUDA BF16 grid evaluates each encoder in FP32. Its final
    teacher mean-motion readout gives:

    | Motion arm | Mean pooled R² | Mean MAE | R² difference from motion-uniform [95% source bootstrap interval] |
    |---|---:|---:|---|
    | Uniform | 0.113725 | 0.043658 | Reference |
    | MAMP convention | 0.112890 | 0.043751 | -0.000834 [-0.020372, 0.015296] |
    | Robust motion mixture | 0.114209 | 0.043595 | +0.000485 [-0.015497, 0.015173] |
    | Matched initial encoder, same summary | 0.222544 | 0.041546 | Learning control, shared across arms |

    Each R² is first computed from all five held-out folds within a seed,
    weighting videos equally, then averaged over seeds. The mask intervals
    use 2,000 paired video resamples conditional on the fitted models. They
    do not establish equivalence, and they do not test trained versus initial.
    The tiny point differences do not identify a winning motion sampler.
    Every motion teacher has lower R² and higher MAE than its matched initial
    encoder in every seed. Selecting high-motion targets therefore changed
    the intervention without demonstrating improved readout for this endpoint.

    ### Step 3: Choose the next action

    1. Preserve this comparison and its fixed temperature/mixture settings.
       A new temperature or mixture search would be an exploratory study
       informed by these development results.
    2. First reuse the saved encoders for the regularization and summary
       ablations specified in Notebook 18, Section 9. Those tests address the
       shared trained-versus-initial deficit across all motion arms.
    3. Before interpreting another motion mask as anatomically preferable,
       inspect training-only coverage by landmark, left/right target counts,
       validity, and large pose jumps. The current enrichment score cannot
       distinguish true movement from tracking artifacts. Retain source-level
       reporting and inspect ordinary clips as well as high-enrichment clips.
    4. Revisit motion-mask tuning only after a fixed readout/objective test
       shows reproducible learning benefit under matched initial controls.

    **Takeaway.** Motion targeting works as a sampling intervention in the
    retained audit. Its greater motion enrichment has not translated into
    better laterality readout in the completed grid. The most informative
    next step is to diagnose the common representation/readout problem.
    """,
    16: r"""
    ## 6. Interpretation of the saved structure audit and next decision

    **Evidence snapshot: 2026-09-08.** Executions 3–6 above retain the real GAVD
    census, structure audit, example masks and cue plot. The audit visits
    75,000 mask draws: 2,500 training-clip memberships across the five folds,
    multiplied by five seeds and six experiment/arm combinations. These are
    repeated draws from **625 clips and 93 videos**, not 75,000 independent
    observations. The 17 excluded archives are QC attrition; the five dataset
    condition counts describe this cohort and are not diagnostic outcomes.

    ### Step 1: Read the cue plot as a test of the intervention

    | Experiment / arm | Observed target-count range | Mean hidden fraction | Immediate temporal brackets | Visible graph neighbor |
    |---|---:|---:|---:|---:|
    | Regions / uniform | 6–48 | 0.099 | 0.699 | 0.988 |
    | Connected regions | 6–48 | 0.099 | 0.000 | 0.512 |
    | Trajectories / uniform | 3–48 | 0.095 | 0.706 | 0.989 |
    | Whole trajectories | 3–48 | 0.095 | 0.000 | 0.995 |
    | Completion / uniform | 24–132 | 0.253 | 0.482 | 0.941 |
    | Interior temporal gap | 24–132 | 0.253 | 0.000 | 0.000 |

    Fractions use valid-token support and source-balanced summaries within
    each fold/seed. A shorter bar means fewer of the specified local cues;
    it does not by itself indicate a better learning problem.

    Connected regions remove immediate two-sided temporal brackets and reduce
    targets with a visible graph neighbor from **98.8% to 51.2%**, a decrease
    of 47.6 percentage points. They still leave such a spatial cue for about
    half of their targets. Whole trajectories remove the temporal brackets
    while leaving a visible graph neighbor for **99.5%** of targets. They
    isolate temporal removal more than spatial removal. Interior gaps remove
    both measured local cues, but context still exists before and after the
    gap. A zero bracket score concerns the two *immediately adjacent* blocks;
    it does not mean that all temporal information has disappeared.

    The six-panel example confirms the intended geometry on one identified
    training clip. Scattered targets are dispersed; region targets span a
    connected anatomical subset; trajectory targets extend across the clip;
    the gap is one interior band. Anatomical connectivity follows the graph,
    not adjacent landmark ID numbers. Gray cells are naturally missing, not
    extra targets. The trajectory example has 45 targets because three of
    its nominal 48 tokens fall in the missing first block. Its paired uniform
    reference also has 45. The example's 48/45/132 counts are not cohort means.

    ### Step 2: Keep the comparisons that answer the same question

    Regions and trajectories nominally hide 48 tokens, but missingness makes
    their realized fractions differ. Completion hides about 25.3% of valid
    tokens, versus about 9.5–9.9% for the other structures. Each structure
    must be compared with its own count-matched uniform reference. Cross-family
    scores would mix target amount and geometry. Similarly, the region-uniform
    training arm is distinct from Notebook 15's higher-budget motion-uniform.

    The smallest realized counts, 6/3/24, also show that some draws have much
    less supervised support than their nominal structure. Per-clip loss
    averaging preserves clip weight despite this variation. A useful future
    audit should stratify error and target coverage by support, rather than
    interpreting the nominal budget as a constant amount of information.

    ### Step 3: Connect cue removal to the completed learning result

    Only **regions** from this notebook were trained in the primary grid.
    Notebook 18's [saved teacher mean-motion comparison]({grid}/paired_intervals.csv)
    gives R² **0.100777** for connected regions versus **0.109446** for their
    uniform reference: difference **-0.008669**, with 95% source bootstrap
    interval **[-0.033290, 0.017666]**. Both trail the matched initial encoder
    (**0.222544**). The interval permits modest effects in either direction;
    it supports neither a superiority claim nor an equivalence claim.

    A small region-teacher mean-only increase over initial features
    (**+0.002928 R²** averaged over seeds) does not rescue the hypothesis:
    the temporal-summary contrast is **-0.121766**, negative in every seed.
    The evidence shows that removing these local cues is insufficient for
    a demonstrated benefit under this recipe and endpoint. It does not show
    that structured masking cannot help a different task.

    ### Step 4: Proceed in this order

    1. Diagnose the regularization and summary effects in Notebook 18 using
       the saved encoders. Keep regions as an informative completed comparison.
    2. Defer a full trajectory/completion grid until a specific remaining
       hypothesis justifies it. Neither family has downstream results here.
    3. If a focused context experiment is warranted, trajectories test whether
       retaining spatial neighbors while removing a landmark's time series
       changes the outcome. Completion addresses reconstruction with both
       earlier and later context. Predeclare that task and its controls.
    4. Use Notebook 13 for observations removed before preparation and Notebook
       14 for past-only forecasting. The present figures support neither a
       raw-sensor robustness claim nor a future-prediction claim.

    **Takeaway.** The geometry audit succeeded: the mask families leave
    measurably different local information. The trained region comparison
    provides no demonstrated readout advantage, so harder masks alone are
    not the next priority.
    """,
    17: r"""
    ## 6. Interpretation of the completed training artifacts

    **Evidence snapshot: 2026-09-08.** This copy of Notebook 17 has no saved
    code outputs. Completion is established by Notebook 18's executions 5–6
    and its [grid manifest]({grid}/manifest.json), rather than inferred from
    this notebook's workload declaration. The review independently checked
    the grid, training and evaluation file hashes, and all 125 training
    histories contain steps 1 through 1,200. No model was retrained for this
    analysis. The conclusions below concern that specific saved grid.

    ### Step 1: Separate completion from effectiveness

    | Verified item | Meaning |
    |---|---|
    | 25 motion jobs × 3 arms, 25 region jobs × 2 arms | 50 paired jobs and 125 trained encoder runs |
    | 1,200 updates in each history | 150,000 arm-specific optimizer updates; 60,000 paired-job update rounds |
    | Five folds × seeds 42–46 | New stochastic training per fold/seed, with the same video partitions reused across seeds |
    | CUDA BF16 training; FP32 weights, loss reductions and frozen evaluation | One explicitly identified numerical condition |
    | 125,000 held-out prediction rows | 625 clips × five seeds × five arms × eight representations, not 125,000 participants |

    Notebook 18's completion bars show 25 jobs per experiment. Their height
    is an availability check, not a performance score. The grid contains all
    625 accepted clips from 93 videos for each arm/representation/seed, and
    the review recomputed all 200 pooled R²/MAE rows from saved predictions.

    The current configuration defaults to FP32 unless overridden. To inspect
    this completed run, use `PRECISION="bf16"`, the CUDA kernel, the same
    fold/seed scope and experiment list, and its existing output root in both
    notebooks. An FP32 configuration names a different experiment and may
    correctly report these jobs as missing. Do not interpret that as loss of
    the BF16 results or launch another grid merely to refill this notebook's
    empty output cells. The short FP32/BF16 timing pilots do not establish
    numerical equivalence or identify the cause of the scientific outcome.

    ### Step 2: Interpret the training histories

    These values were read from the 25 histories per arm and averaged at
    their first and final steps. They are endpoint summaries, not evidence
    of monotonic convergence between those steps.

    | Experiment / arm | Masked prediction loss, step 1 → 1,200 | Total loss, step 1 → 1,200 |
    |---|---:|---:|
    | Motion / uniform | 15.078 → 0.849 | 16.247 → 1.335 |
    | Motion / MAMP | 15.042 → 0.835 | 16.211 → 1.316 |
    | Motion / robust mixture | 15.024 → 0.864 | 16.193 → 1.348 |
    | Regions / uniform | 15.232 → 0.839 | 16.402 → 1.324 |
    | Connected regions | 15.125 → 0.948 | 16.295 → 1.452 |

    Optimization substantially reduces its objective. The total includes
    `masked_prediction_loss + 0.05 × variance_regularizer`; the latter
    contributes roughly 0.48–0.50 at the final averaged step. That scalar
    contribution is not a measurement of its gradient influence. MAMP's
    slightly lower prediction loss does not make it the best representation:
    each arm has a changing teacher, target distribution and feature space.
    Longer training cannot be justified solely by comparing these losses.

    ### Step 3: Ask what learning achieved

    The final teacher with temporal summaries reaches mean R² **0.101–0.114**;
    the initial encoder under the same summary reaches **0.223**. All five
    trained arms have lower R² and higher MAE than initial features in all
    five seeds. The teacher generally outperforms the final online encoder
    in the averaged table, but it does not recover the initial readout level.

    The predictor has nevertheless learned clip correspondence: across the
    75 fold/seed/evaluation-mask rows per trained arm, mismatched targets
    always yield larger error than matched targets on the same eligible
    clips. This can coexist with an inferior endpoint readout. The evidence
    is consistent with learning information that is less useful to this
    particular summary/ridge/target combination; it does not identify which
    component is responsible or prove that movement information was erased.

    ### Step 4: Spend the next computation on a discriminating test

    1. Reuse the saved encoders for a common expanded ridge grid and the
       summary-component ablations in Notebook 18. Those checks do not
       require repeating these 150,000 updates.
    2. If the deficit survives, run one source-separated training pilot with
       initialization and prespecified intermediate checkpoints, for example
       steps 0, 100, 300, 600 and 1,200. Reserve validation sources from that
       pilot's encoder training. Track readout, loss, unscaled feature
       variation and correspondence together; do not select a checkpoint
       on the already inspected outer-test scores.
    3. Change one candidate cause at a time. A focused regularizer-weight or
       target-objective comparison is interpretable; changing masking,
       precision, augmentation, width and duration together is not. An FP32
       sensitivity pilot is a separate numerical comparison if needed.
    4. Expand a revised recipe to all folds/seeds only after the pilot defines
       a fixed testable hypothesis. Report the present unfavorable result
       alongside any later improvement on this development cohort.

    **Takeaway.** The full training milestone is complete. It improved the
    training objective and predictor correspondence, but it did not improve
    the tested laterality readout over initialization. Diagnose that gap
    before increasing the model, training duration or mask menu.
    """,
    18: r"""
    ## 9. Evaluation of the actual results and a specific next plan

    **Evidence snapshot: 2026-09-08.** The analysis below interprets the saved
    executions above and the exact [grid manifest]({grid}/manifest.json).
    All 50 paired jobs are complete, with CUDA BF16 training and FP32 frozen
    evaluation. The review checked artifact file hashes, confirmed 125
    complete 1,200-step histories, and independently recomputed the 200 pooled
    R²/MAE rows from 125,000 predictions. Every arm/representation/seed covers
    625 clips from 93 videos. These are development results for a
    coordinate-derived movement contrast, with video rather than verified
    person separation.

    ### Step 1: Compare learning under an identical readout

    Values below are from [summary.csv]({grid}/summary.csv). R² is computed
    after pooling five outer folds within each seed with equal weight per
    video, then averaged over five seeds. “±” denotes **seed standard
    deviation**, not a confidence interval. MAE is in the dimensionless target
    contrast's units; lower is better.

    | Representation / training arm | Mean pooled R² ± seed SD | Mean MAE |
    |---|---:|---:|
    | Training-source mean | -0.0106 ± 0.0000 | 0.046172 |
    | Direct pose summaries | 0.0345 ± 0.0000 | 0.044960 |
    | Initial encoder, mean | 0.0708 ± 0.0186 | 0.044344 |
    | Initial encoder, mean-motion | **0.2225 ± 0.0268** | **0.041546** |
    | Motion-uniform teacher, mean-motion | 0.1137 ± 0.0110 | 0.043658 |
    | MAMP teacher, mean-motion | 0.1129 ± 0.0306 | 0.043751 |
    | Robust-motion teacher, mean-motion | 0.1142 ± 0.0210 | 0.043595 |
    | Region-uniform teacher, mean-motion | 0.1094 ± 0.0088 | 0.044037 |
    | Connected-region teacher, mean-motion | 0.1008 ± 0.0092 | 0.043756 |

    The initial/direct controls are reused across arms; repeated rows are not
    independent replications. The trained online encoders' mean-motion R²
    values are also below initial: **0.0805, 0.0809, 0.0740, 0.0657 and
    0.0636** in the corresponding five-arm order. The finding therefore is
    not explained simply by evaluating the teacher instead of the online model.

    Reading [per_seed.csv]({grid}/per_seed.csv) shows teacher-minus-initial
    mean-motion R² differences from **-0.1083 to -0.1218** after averaging
    seeds. Every arm has a negative difference and higher MAE in **all five
    seeds**; MAE increases average **0.00205–0.00249**. This is consistent
    unfavorable evidence for this training/readout combination. Five seeds
    reuse the same videos, so their agreement is not five independent cohort
    replications. The saved mask intervals below are a different comparison.

    ### Step 2: Interpret the temporal-summary gain correctly

    Adding SD, absolute feature increments and support fractions improves the
    initial encoder by **+0.1517 R²**, from 0.0708 to 0.2225. The same change
    improves trained teachers by only **+0.0270 to +0.0468**. Thus simple
    averaging does omit useful predictive features, but restoring those
    summary components does not reveal a trained-over-initial advantage.

    This experiment changes several readout components together. Temporal SD
    does not depend on order, and support fractions describe missingness.
    The gain cannot yet be assigned specifically to ordered movement,
    amplitude, or missingness. The strong initial control can exploit the
    architecture, anatomical pooling and prepared input without pretraining.
    It is evidence of accessible predictive structure, not proof of learned
    dynamics. Direct pose here is a particular summary-plus-linear-ridge
    baseline; its lower score is not an upper bound on information in the
    coordinates or on stronger pose baselines.

    The generated amplitude control in execution 9 is correctly separate:
    mean R² is **-0.473**, while mean-motion R² rounds to **1.000** with MAE
    **0.00001733** on 12 clips from six generated sources. Its sine plot shows
    why a zero mean loses amplitude. It validates that construction and the
    summary calculation; it adds no real GAVD participants or clinical evidence.
    Execution 10 confirms that the optional Notebook 12 reanalysis was skipped.

    ### Step 3: Assess the masking hypothesis with paired uncertainty

    The saved [paired intervals]({grid}/paired_intervals.csv) use the final
    teacher's mean-motion representation and 2,000 joint source-video
    bootstrap resamples, retaining each video's clips and all five seeds.

    | Contrast against its own uniform reference | Mean ΔR² | 95% interval |
    |---|---:|---:|
    | MAMP motion | -0.000834 | [-0.020372, 0.015296] |
    | Robust motion | +0.000485 | [-0.015497, 0.015173] |
    | Connected regions | -0.008669 | [-0.033290, 0.017666] |

    All intervals include zero. No alternative mask demonstrates a benefit
    on this readout. The intervals allow modest gains and losses, so “all
    masks are equivalent” would also overstate the evidence. They condition
    on fitted models and exclude variability from retraining on resampled
    cohorts. A paired trained-versus-initial interval would need its own
    calculation; these intervals cannot be reused for that claim.

    Notebook 16 demonstrates a real reduction in local cues for regions.
    That reduction did not produce a demonstrated readout gain. Trajectories
    and completion were audited but are absent from this training grid.

    ### Step 4: Reconcile the predictor result with the readout result

    The full [predictor table]({grid}/predictor_diagnostics.csv) contains 525
    rows: 375 trained-arm rows and 150 initial-control rows. Each arm contributes
    25 fold/seed jobs × three evaluation masks. For every trained row,
    `mismatched_target_mse - matched_target_mse_on_control_clips` is positive.
    Its descriptive average is **0.433** for MAMP, **0.428** for robust motion,
    **0.435** for motion-uniform, **0.345** for regions, and **0.418** for
    region-uniform. The initial controls average **-0.003** and are positive
    in only 33 of 75 rows per experiment. These averages describe diagnostic
    rows; they are not pooled patient outcomes or independent significance tests.

    The predictor distinguishes a matching clip from a different source's
    target much more consistently after training. That is a useful positive
    finding. It could reflect movement, pose, appearance-related geometry,
    recording properties or other clip-specific structure. It does not
    establish laterality semantics, forecasting, or a clinically useful
    world model. Raw errors and correspondence gaps live in each arm's own
    teacher space, so their magnitudes cannot select the best mask. Use the
    matched-control subset, not all-clip error, when forming a mismatch gap.

    All 875 standardized readout-feature diagnostic rows have
    `near_constant=False`. Their mean effective ranks are about **32.2** for
    initial mean-motion, **34.0** for teacher mean-motion, and **39.2** for
    online mean-motion. A representation with more varying directions can
    still predict this endpoint less well. These diagnostics follow scaling;
    they do not by themselves exclude weak raw variation or partial loss of
    useful directions. The retained predictor's raw clip/token target and
    prediction constant flags are also all false. The present evidence does
    not support describing the outcome as complete constant-feature collapse.

    ### Step 5: Resolve the strongest readout concern before retraining

    The saved [selection table]({grid}/selection.csv) shows repeated selection
    of **alpha=10,000**, the largest candidate:

    | Readout | Upper-bound selections / reported fits |
    |---|---:|
    | Initial mean-motion | 0/125 (0%) |
    | Teacher mean-motion | 49/125 (39.2%) |
    | Online mean-motion | 96/125 (76.8%) |
    | Teacher mean | 109/125 (87.2%) |

    Control fits are repeated across arms in these counts. Boundary selection
    suggests that the readout search may not yet cover enough regularization
    for some trained features; it does not prove that a larger penalty will
    recover their deficit. The same nominal grid can have different effects
    on representations with different covariance and dimension. At width 96,
    `mean` has 960 features, while `mean_motion` has 2,890, including ten
    support fractions, fitted from 74–75 training videos per outer fold.

    ### Step 6: Follow this staged investigation

    1. **Readout regularization, using existing checkpoints.** Predeclare one
       expanded grid for every representation, such as the existing values
       plus 100,000 and 1,000,000. Select alpha using training-source inner
       predictions only, save a new readout identity, and report all arms and
       summaries. Inspect validation curves, boundary frequency, MAE and
       paired differences. Keep the current scores as the original result.
       This follow-up is exploratory because the development outcomes have
       already been inspected; it must not silently replace the first grid.
    2. **Identify the source of the summary gain.** On the same frozen initial,
       online and teacher encoders, compare mean; mean plus support; mean plus
       SD plus support; mean plus absolute increments plus support; and the
       full summary. Add support-only and corresponding direct-coordinate
       controls. Hold folds, seeds, scaler fitting and alpha selection fixed.
       A support-only gain points to observation patterns; an additional
       increment gain motivates a more specific temporal-order test.
    3. **Audit the endpoint/preparation relationship.** The target averages
       bilateral contrasts of median displacement rates on common observed
       transitions in the original target lane. Readout inputs are interpolated,
       normalized and resized to 64 steps. Compare a clearly labeled
       target-formula diagnostic computed on prepared inputs with the original
       target, stratified by support and timing. This tests information changed
       by preparation; it is not an independent learned baseline. A uniform
       time-scale factor largely cancels in the normalized bilateral ratio
       apart from epsilon, so absent duration alone is not an established
       explanation. Irregular intervals, interpolation and changed valid
       support remain testable possibilities.
    4. **Only then change training.** If the trained deficit persists, use a
       training-only, source-separated pilot with initial and intermediate
       checkpoints to find when readout deteriorates. Compare one objective
       or regularizer change at a time, retaining paired initialization,
       exposure and masking. Keep validation sources outside candidate
       encoder training; the current three-group inner procedure selects
       only the readout. A numerical sensitivity test needs its own FP32
       comparison because this completed grid used BF16.
    5. **Set a progression criterion before a new full grid.** Require a
       consistent trained-over-initial improvement under the same prespecified
       summary on the pilot's reserved sources. Then repeat the fixed recipe
       over the full declared grid and report paired source uncertainty and
       seed variation. Keep later confirmation on an untouched cohort or
       setting separate from this repeatedly inspected development dataset.

    **Takeaway.** The useful result is the combination: training improves
    clip correspondence while the best tested movement readout comes from
    initial temporal-summary features. Motion weighting and region masking
    do not resolve that gap. Prioritize regularization, summary attribution
    and endpoint/preparation checks; a larger mask sweep or longer training
    currently lacks a discriminating rationale.
    """,
}


def interpretation_text(number):
    """Return static, provenance-labelled prose for one notebook."""
    return (
        dedent(OPENING[number]).strip(),
        dedent(ANALYSIS[number]).strip().format(grid=GRID, archive=ARCHIVED_15),
    )


def add_saved_result_interpretation(notebook, number):
    opening, analysis = interpretation_text(number)
    notebook.cells[0].source += "\n\n" + opening
    notebook.cells.append(new_markdown_cell(
        analysis,
        metadata={"tags": ["results-interpretation", "review-2026-09-08"]},
    ))
    return notebook
