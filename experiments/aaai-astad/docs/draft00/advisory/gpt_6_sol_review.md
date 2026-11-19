# Independent research review

**Model:** GPT-6.1-sol. **Date:** November 14, 2026. This is advice, not an experimental replicate. I read the brief, archived plans, local GAVD audits, saved results, and annotation metadata. I did not train a model or download media. Every proposed result below is a hypothesis until measured.

## The opportunity and its real denominator

I would aim higher than a short-run JEPA failure report. The stronger question is whether a normal-reference gait score measures unusual **movement** across unfamiliar patterns, or observation and recording conditions. A source-blocked study with a mechanistic score and controlled interventions could answer that question. The workshop welcomes anomaly benchmarks, robustness, uncertainty, and healthcare applications ([ASTAD call](https://ismart.ece.mcgill.ca/astad4/)).

The local annotation inventory has 1,874 sequences in 348 source IDs. Here is the complete label support before media availability and pose quality checks:

| GAVD gait-pattern folder | Sequences | Sources |
| --- | ---: | ---: |
| normal | 291 | 32 |
| abnormal | 767 | 117 |
| antalgic | 35 | 10 |
| cerebral palsy | 64 | 11 |
| exercise | 234 | 98 |
| inebriated | 23 | 8 |
| myopathic | 188 | 30 |
| parkinsons | 47 | 11 |
| pregnant | 6 | 2 |
| prosthetic | 39 | 8 |
| stroke | 76 | 19 |
| style | 104 | 3 |

For the proposed binary endpoint, the normal folder is normal gait and every other folder is an abnormal-gait annotation. Exercise, style, and pregnant belong on the positive side. These are **not** confirmed medical diagnoses of individual walkers. The original [GAVD paper](https://arxiv.org/abs/2407.04190) describes clinical visual screening, but the local files lack participant identifiers and diagnosis verification. The separate `dataset` field says “Abnormal Gait” for 5,365 rows in the normal folder. It may denote the parent collection. Disclose its semantics, and do not relabel those sequences from that field.

There are 245 source IDs outside the original five-class selection, and none is normal. Just one source spans two folders, abnormal and cerebral palsy. There are no normal-versus-abnormal same-source pairs. The inventory has 213 sources with multiple camera-view labels; among 91 pose-qualified sources, 63 have multiple views but only four retained sequences change view. These are candidate comparisons, not verified same-person, synchronized views. Only 49 of the 245 extra source IDs have a local media file by ID, with decoding and pose quality unknown. A full-label paper needs media acquisition and an attrition ledger. Extra labels provide **no new normal calibration sources**.

## Three ranked paper concepts

### 1. Coordination-specific surprise across unseen abnormal gait categories

**Central claim.** A score based on the *incremental value of other joints given a target joint's own history* transfers more reliably than raw masked-prediction error to abnormal-gait categories excluded from method selection, while responding less to pose degradation. This is a hypothesis. The asset is the full 12-label GAVD inventory: train normal-only models on existing normal sources, select settings without new nonnormal sources, and prospectively evaluate the 245 outside sources after media and QC. Treat each abnormal folder as positive, with per-folder support shown. This tests semantic transfer, not diagnosis.

**Difference from prior work.** [S-JEPA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04755.pdf) already predicts masked skeleton latents. [FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) separates postural reconstruction from temporal prediction. [GenGait](https://arxiv.org/abs/2604.01997) uses normal-only masked gait modeling. A worthwhile contribution would identify cross-joint predictive information, test its response to observation interventions, and challenge it with held-out categories. Subtraction alone is not novel; background-corrected anomaly scores have precedent in [Ren and colleagues](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html).

**Decisive experiment.** Compare raw JEPA discrepancy with a matched full-joint-versus-own-joint contrast on the same normal sources, local targets, masks, and update budget. Freeze scoring before opening new-category scores. Test source-level normal-versus-abnormal ranking on eligible full-label sources, within view and pose-quality bands, and on paired corruptions. Test whether other-joint context adds information beyond the target joint's own past. Failure means lost abnormal-gait sensitivity, no useful gain over raw discrepancy, or equal sensitivity to observation-only interventions. Feasibility depends on run time and new pose-qualified sources. An incomplete cohort supports a transfer pilot, not an all-label claim.

### 2. Two-axis uncertainty: observation reliability and movement deviation

**Central claim.** A detector can say separately “the pose is unreliable” and “the movement departs from the normal reference.” The asset is the 450-sequence pose cohort with cached sequences and confidence traces. Compare each skeleton with mild, checked coordinate dropout, joint swaps, or jitter. This creates paired measurement interventions; severe edits may alter perceived gait. Existing videos could support a second pose extractor or manual spot checks, although correlated extractor errors are not ground truth.

**Difference from prior work.** Simple confidence weighting is already in [SeeKer](https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html). A generic corruption table is also insufficient: [RobustGait](https://openaccess.thecvf.com/content/WACV2026/html/Sayera_RobustGait_Robustness_Analysis_for_Appearance_Based_Gait_Recognition_WACV_2026_paper.html) benchmarks corruption and extractor bias for appearance-based gait recognition, and a [markerless gait anomaly study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12658200/) already addresses pose noise. The new mechanism would use paired observations to constrain two outputs, with a movement score stable under recoverable observation damage and a reliability output that rises when evidence is lost. Compare a full two-output method against confidence weighting and an ordinary denoising baseline.

**Decisive experiment and failure.** Cross clean/degraded observations with normal/abnormal gait; report discrimination and reliability or coverage-risk jointly. Failure occurs if the movement score tracks corruption, uncertainty suppresses abnormal signals, or abstention discards most abnormal sources. A controlled skeleton demonstration is plausible this week. Claims about real pose uncertainty need independent checked poses or image-level interventions. I rank this second because identification is harder by November 20.

### 3. Cross-view consistency of an open-set gait decision

**Central claim.** A truly movement-sensitive score should agree across verified views of the same walker more than a view-sensitive nuisance score, without erasing view-dependent information needed for detecting asymmetry. The asset is the 213 multi-view source candidates, especially 63 in the already pose-qualified cohort. The first step is a blinded verification sample of person identity and time alignment in existing media. If enough genuine same-walker pairs survive, learn or assess a view-conditioned agreement constraint using training sources only and evaluate held-out source pairs. Include all abnormal gait folders where usable; do not infer a pair from a shared source ID alone.

**Difference from prior work and decisive test.** View adaptation is established in gait recognition, so the contribution would be a source-paired *anomaly decision* study with a measured sensitivity-versus-invariance tradeoff. Compare raw scores, explicit view adjustment, and paired agreement training on genuine pairs. The test fails if verified pairs are too scarce, within-person score agreement is no better than source-matched unpaired clips, or agreement comes from flattening both normal and abnormal scores. This is a high-value but low-certainty six-day option. The current metadata show only candidate pairs, and video compilations may contain multiple demonstrators.

## My choice and a specific mechanism

I choose concept 1. The archived draft correctly rejects an unsupported new-detector claim, but its fallback could aim too low if it ends at “briefly trained JEPA failed.” Its proposed observation-versus-movement uncertainty separation asks for independent pose truth that cannot be established from this inventory this week. By contrast, the extra abnormal-gait categories provide a real, if acquisition-dependent, transfer challenge; paired corruptions of an identical cached sequence provide an intervention that cancels person, source, label, and camera content. Combining those with a temporal-information mechanism can produce a more substantial workshop paper.

One implementation freezes an adequately trained normal teacher and trains two equal-budget predictors. The full predictor sees all valid joint patches. The control sees only the target joint's valid history, while retaining the same target position and visibility summary. Require sufficient observed own-joint history and score identical target tokens under identical uniform masks. Define `C = CE(teacher, full prediction) - CE(teacher, own-joint prediction)`. More negative values mean other joints helped more on normal gait; a relatively high `C` is the hypothesized abnormal score. Freeze this sign before new-category evaluation. The shared target makes `C` a difference of latent discrepancies, **not** a gait-sequence likelihood ratio.

A fully contextualized teacher target already encodes other joints, so this contrast alone does not identify coordination. Repeat it with patch-local teacher features or normalized coordinate targets. The full predictor receives more information by design; the question is whether its *incremental* benefit is specific to inter-joint movement. As a diagnostic, cyclically phase-shift a whole leg in high-coverage normal sequences while preserving each joint's trajectory, then check limb geometry and compare with a framewise static predictor. A common time shift of all joints is a control. If the edit creates implausible poses or alters missingness, it does not isolate coordination. These synthetic edits are not new patient labels.

Apply clean-versus-corrupted paired tests with the same masks. Corrupt context and target separately to expose observation effects. A clean teacher target can be held fixed for an oracle analysis, but the deployable score must also work with degraded observations alone. This design is more diagnostic than adding positional noise or a stochastic latent prior, neither of which currently has a measured benefit.

## Minimal decisive matrix and evaluation policy

First time one normal-only run. Use a fixed source partition and paired initialization, data draws, masks, and update checkpoints, for example 64, 256, and 1,024 updates. These budgets are diagnostics, not guaranteed convergence. Measure normal training and validation loss, online/teacher drift, representation variance and effective rank, and whether full joint context beats own-history or swapped other-joint context. The current normal anchors received only 45, 66, and 63 updates, with substantial loss change and EMA initialization coefficients 0.835, 0.768, and 0.777. A negative detector result before training adequacy is established would mostly indict the recipe.

The primary normal-only comparison needs raw masked JEPA discrepancy, the contrast, own-history loss, a simple normal embedding distance or coordinate forecast, and one published normal-only skeleton method or faithful adaptation, preferably SeeKer, FSGait, or STG-NF. Without the last arm, do not claim superiority over the field. Match normal-source access, observation policy, and aggregation; report a common cohort if a baseline requires complete cases. The supervised comparison is a **different task**: RF, mean pose, nuisance-only features, untrained encoder probe, and trained JEPA probe, all fitted with the same five labels and split. RF binary AUROC is not normal-only detection.

The primary evaluation unit is a source ID, with sequence scores aggregated before one source score. Keep the single mixed abnormal/cerebral-palsy source in one split and count it once for overall binary detection. Report both source-weighted AUROC for the pooled binary task and a label-balanced summary with per-label counts and uncertainty. The latter will be unstable for pregnant with two sources and style with three. Report abnormal average precision beside its source-level prevalence baseline, since positives may dominate. Show QC attrition by label, view, and source. Bootstrap paired *source* differences for method and corruption comparisons, resampling normal and abnormal sources separately. If the analysis uses source-weighted macro-F1 for five-class recognition, its interval must estimate that same weighted quantity; the existing sequence-weighted intervals cannot be attached to it. Do not treat overlapping windows or three reused source splits as independent samples.

All three existing test splits have been inspected. Their results remain descriptive and every new analysis on them is labeled exploratory. The 245 additional source IDs can serve as prospective abnormal-category challenges if rules are frozen before their scores are viewed, but there is no untouched normal pool to yield a fully independent normal-versus-abnormal AUROC or a new specificity estimate. Use the old normal validation sources for development only, avoid re-partitioning reviewed sources to manufacture a fresh test, and document any earlier local use of extra sources. At 5% false-positive target, five or six normal calibration sources imply the current infinite finite-sample threshold; 19 independent units are only the algebraic minimum for a finite threshold. The 32 annotated normal sources do not become hundreds of calibration units through windows.

The central interval should address a prespecified useful effect, perhaps five AUROC points, chosen before new-category scoring. An interval containing both zero and that gain is **inconclusive**. An upper bound below it excludes that magnitude under the stated protocol, not universally. A negative result becomes informative if training stabilizes, the order control works, a strong baseline is present, and paired interventions identify the dominant signal. Otherwise explain the bottleneck.

## Main confounds and six-day decision schedule

The hardest confounds are repeated people across source IDs, camera views, annotation-frame alignment, label-dependent pose attrition, and visibility patterns that reveal recording conditions. The accepted cohort is 450 of 666 selected sequences across 91 sources; normal token validity is lower than in each selected abnormal category. Source grouping does not block repeated people across uploads. Only 758 event-annotated rows exist, with no normal contacts; phase-ground-truth claims would be weak. Coordinate corruption tests a specified intervention, not every real image artifact. [Asleep at the Wheel](https://arxiv.org/abs/2608.01336) and the [physical-surprise study](https://arxiv.org/abs/2611.01815) reinforce the difference between representation content and prediction error. I have not verified priority for this contrast; check literature before claiming it.

| Day | Action and explicit decision |
| --- | --- |
| November 14 | Freeze concept, score sign, binary label rule, primary source unit, useful gain, and an acquisition/QC ledger. Time one training run and start processing the extra abnormal-source media. |
| November 15 | Audit a stratified sample for identity and frame alignment. If serious misalignment appears, repair and disclose it before interpreting scores. Count actual pose-qualified extra sources by label. |
| November 16 | Finish the update ladder, local-target control, and context diagnostics. If other-joint context offers no gain or training remains unstable, investigate optimization. |
| November 17 | Run the normal-only arms and paired context/target corruption interventions. If a published baseline cannot be made cohort-comparable, narrow the comparative claim. |
| November 18 | Freeze the manuscript result table. Evaluate new-category sources once, with source-cluster paired intervals and attrition. If category support is sparse, present only supported categories and explicitly mark the full-label ambition incomplete. |
| November 19 to November 20 | Write and audit the seven-page paper, check claims against saved predictions and primary citations, verify double blindness and the submission cutoff in OpenReview, then submit by the workshop deadline. |

My recommendation is to spend the six days on one identifiable motion-versus-measurement question and its unseen-category test. Current results alone establish neither a positive nor a bounded negative answer.

## Second-round addendum: cross-joint information

I now favor full-visible-joint versus target-joint-own-history prediction over the earlier ordered-versus-orderless idea. It tests inter-joint coordination more directly and uses one normal teacher with two matched predictors. Its main identification risk is a fully contextualized teacher target that already encodes other joints. Require a patch-local or coordinate target control, common masks and target support, and the *same clean-window normalization* for both predictors and every paired edit. A whole-leg phase shift must preserve articulated limb geometry and each joint's trajectory; compare it with a whole-body shift and a framewise predictor. Otherwise any response may reflect broken poses, changed missingness, or rescaling rather than coordination. The strongest evidence couples selective phase sensitivity and observation robustness with source-level transfer to unseen abnormal-gait categories.
