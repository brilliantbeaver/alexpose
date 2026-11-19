# Research directions and experiments that would change the assessment

This document accompanies the [critical review](README.md). Everything below is a proposed experiment unless explicitly described as an existing result. The aim is to produce a supported contribution, not to add every possible extension before the workshop deadline.

## 1. First establish what the current model has learned

The normal anchors received only 45–66 optimizer updates. Before interpreting their performance as a property of latent prediction, investigate training sufficiency on development data. Use separate variables for the source split and model initialization seed, so that their effects can be distinguished.

| Priority | Controlled experiment | Keep fixed | What it can establish |
| --- | --- | --- | --- |
| Essential | A prespecified update ladder, for example 64, 256, 1,024, and 4,096 updates | Training sources, architecture, loss, joint set, and sampling policy | Whether the result depends on a very short optimization budget. These are diagnostic budgets, not a claim that 4,096 guarantees convergence. |
| Essential | Teacher warm-up or momentum schedule versus the current constant EMA | Total updates, initialization, data draws, and masks | Whether teacher dynamics explain a meaningful part of the short-run result. |
| Essential | Online-encoder and target-encoder probes | Checkpoint, train/validation sources, probe search, and evaluation | Whether the chosen readout contributes to the apparent failure. The current readout differs from the original Skeleton S-JEPA paper. |
| Essential for a curriculum claim | Progressive, pooled, and reversed-order condition exposure | Total updates, approximately equal per-condition exposure, normal replay, and optimizer-reset policy | Whether order contributes beyond additional compute and data. Include a normal-only continuation with the same update budget. |
| Essential for a dynamics claim | Correct context, swapped context, and orderless-context prediction | Teacher targets, allowed observations, target tokens, and scoring | Whether useful prediction depends on the temporal context. |
| Essential for an anomaly claim | A normal-only skeleton anomaly baseline | Cohort, observation policy, training-source access, and aggregation | Whether the failure or improvement is specific to the proposed scoring/model choice. |

Record training and normal-validation prediction loss, encoder variance and effective rank, projector variance, teacher-to-online drift, and downstream validation probes. Keep the normal-only anomaly setting separate from the supervised probe setting. A probe trained with abnormal labels does not become unsupervised because its encoder was pretrained on normal data.

Compare against an appropriate subset of [STG-NF](https://github.com/orhir/STG-NF), [SeeKer](https://github.com/adelic99/seeker), and the concepts in [FSGait](https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19), in addition to the existing distance score. A simple coordinate reconstruction or motion-prediction baseline can help isolate the value of latent targets. Document adaptations to the 16-joint topology, missing observations, and sequence duration. Do not compare a complete-case baseline against a different, easier cohort without also reporting a common-cohort comparison.

The strongest useful outcome could be either positive or negative. If adequate training closes the untrained-encoder gap, the original failure was at least partly a recipe issue. If it does not, a well-controlled failure analysis becomes more credible. Neither outcome should be decided in advance.

## 2. Near-term direction: isolate the information in predictive surprise

### Research hypothesis

After sufficient optimization, raw latent discrepancy may combine static geometry, ordinary prediction difficulty, observation quality, and temporal coordination. The additional predictive value of ordered context may be more informative about coordination than the raw discrepancy alone.

This is a hypothesis, not a conclusion from the current AUROCs. Some pathological gait can remain highly regular and predictable. A temporal score therefore cannot replace a separately assessed static-posture or normal-space score.

### A concrete experiment

Freeze one adequately trained normal-reference teacher, including its centering statistics and temperature. Train two predictors using only the same normal training sources and the same observed inputs. The dynamic predictor receives an ordered temporal context. The comparison predictor receives a permutation-invariant summary of the same context. Both retain joint identities and target positions. Match capacity and training budget, and record exactly how confidence and visibility information enter each predictor.

Define the removed order precisely. One implementation could embed each visible four-frame joint patch independently, without a context-time embedding, and pool patches within each joint before any temporal contextualization. It would preserve motion within a four-frame patch while removing order between those patches. Pooling already contextualized transformer outputs would not achieve this control. Test that permuting the eligible context patches within each joint, while keeping the target queries and visibility summaries fixed, leaves the comparison predictor's output unchanged. Label the control as removing order between patches rather than all temporal information.

For each common valid target token, let `p` be the frozen teacher's feature-channel distribution, `q_dynamic` the dynamic prediction, and `q_orderless` the orderless prediction. Define the exploratory contrast:

```text
temporal_contrast = mean_valid_targets[
    CE(p, q_dynamic) - CE(p, q_orderless)
]
```

More negative values mean that ordered context provides greater predictive benefit. A relatively high value means reduced benefit relative to the normal reference distribution. The two losses must use identical targets, temperatures, and valid-token support. This contrast is also a difference of KL discrepancies because the shared target entropy cancels. **It is not a likelihood ratio over skeleton sequences:** the channel softmax is not a normalized probability density over possible motions.

Removing order can also remove the temporal pattern of missingness. Control this explicitly through identical visibility summaries and a matched high-coverage subset, and repeat the test with controlled observation masks. Otherwise the experiment may measure missingness order rather than movement order. A position-only predictor and context swapping provide complementary checks with different failure modes.

Use uniform masking over the existing 16-joint grid for the main completion experiment. A causal suffix experiment is separate and must use prefix-only normalization and inputs. Preserve the original calibration and test procedures while introducing the contrast as a new exploratory score. Fix its sign before evaluating new test data.

The background-correction idea itself has substantial prior art, including [Ren et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/1e79596878b2320cac26dd792a6c51c9-Abstract.html). The possible contribution is a demonstrated gait-specific explanation and useful scoring behavior under controlled confounds, not priority over subtracting a background score.

### Evidence that would support or falsify it

Measure raw discrepancy and the contrast on normal and abnormal held-out sources, then repeat on a prespecified high-quality subset and within supported view strata. Use the same source groups for paired comparisons. Report AUROC, prevalence-aware AP, and the response to controlled perturbations. Do not pool source windows as independent observations.

The hypothesis is weakened if correct and swapped context perform similarly, if the contrast is driven mainly by observation coverage, or if it removes sensitivity to annotated gait differences. Prespecify a minimum useful gain before independent evaluation and calculate a paired interval. An interval that excludes that gain can reject the proposed improvement at that magnitude; a wide interval spanning both meaningful benefit and no benefit is inconclusive. A negative result remains informative if the controls establish what the model actually uses and the uncertainty supports the stated conclusion.

## 3. Longer-term direction: separate observation uncertainty from movement uncertainty

### The contribution worth pursuing

A useful detector should distinguish a poorly observed ordinary movement from a well-observed movement outside the normal reference. Simply treating a large residual or high predictive variance as abnormal cannot make this distinction. Confidence-weighted joint scores already exist in [SeeKer](https://openaccess.thecvf.com/content/ICCV2025/html/Delic_Sequential_keypoint_density_estimator_an_overlooked_baseline_of_skeleton-based_video_ICCV_2025_paper.html).

A stronger contribution would be an identifiable separation between uncertainty in the measurement and variability in the movement, supported by paired observations. The model should have an observation-quality output and a movement-deviation output whose behavior can be tested separately. If evidence is too poor, an explicit “insufficient observation” outcome is preferable to a confidently interpreted movement score, but its coverage and errors must also be evaluated.

### Data and model design

Obtain a small reference set with repeated views, independently checked keypoints, or a validated motion reference. Two pose estimators on the same image provide useful disagreement measurements, but their agreement is not ground truth and their errors can be correlated. Manual or external reference measurements are needed to interpret that disagreement.

Train an observation model on paired measurements and a normal-motion predictor on the normal training partition. If learned stochastic latents are used, distinguish the conditional prior available at inference from the target-informed posterior used during training. Use a stated probabilistic objective and evaluate prior samples. Report negative log predictive density only if the implementation defines a normalized predictive density; otherwise use an appropriate proper sample score in a fixed target space. Evaluate representation quality separately so that uncertainty improvements do not hide a loss of useful information.

The [November 2026 stochastic-latent JEPA preprint](https://arxiv.org/abs/2611.09302) already motivates learned predictive uncertainty and diverse completions. Transferring that machinery is a baseline. The proposed new contribution would be separating the sources of uncertainty and demonstrating better movement decisions under realistic pose error.

### Cross movement status with observation quality

| Evaluation condition | Prespecified expected behavior |
| --- | --- |
| Normal movement, reliable observation | Low movement-deviation score and low observation uncertainty. |
| The same normal movement, degraded observation | Increased observation uncertainty; movement decisions should remain stable where adequate information survives. |
| Independently annotated movement difference, reliable observation | Increased movement-deviation score, without requiring poor tracking. |
| Movement difference with degraded observation | Calibrated uncertainty or reduced coverage; report whether the detector misses the movement difference. |

Use paired image-level degradations when testing the entire visual pipeline. Coordinate noise is a useful controlled skeleton experiment, but it does not reproduce all failures of a visual pose estimator. Prespecify degradation strengths and have the retained movement content checked. A temporal shuffle or synthetic asymmetry is a controlled perturbation, not a newly labeled patient or a validated disease simulation.

Evaluate discrimination, normal false alarms, error-versus-uncertainty association, calibration where applicable, and coverage-versus-risk curves. Show abstention rates separately by condition and source. A model that improves conditional accuracy by rejecting most difficult cases has not necessarily improved screening usefulness.

The hypothesis fails if the proposed movement score rises mainly with tracking error, if the model suppresses genuine movement differences as uncertainty, or if gains disappear against a simple confidence-aware baseline. An independent participant-identified cohort would be the decisive test of impact.

## 4. Longer-term direction: preserve relevant movement sensitivity

A second method question is whether representation learning can become insensitive to nuisance variation while retaining subtle asymmetry and temporal coordination. Adding VICReg does not specify which variations should be preserved. Conversely, making every normal gait look identical can erase useful differences.

One concrete design is to distinguish transformations that should preserve the prediction from changes that should affect a measured kinematic output. Preserve left/right anatomy, estimate phase using the available context only, and retain physical timestamps or cadence alongside any phase-aligned representation. Whole-sequence phase alignment in a forecasting input could leak future information. Time normalization can also erase the timing abnormality being studied.

An auxiliary target could be a predeclared, independently validated set of asymmetry and inter-joint timing measurements. Compare the model with and without the target, with an equal-budget coordinate-prediction baseline, and with direct classical features predicting the same measurements. Since this adds kinematic supervision or a structured pretext target, describe that supervision clearly. It is not purely generic latent self-supervision anymore.

The important result would be a controlled tradeoff: nuisance transformations have less effect on the representation while expert-validated movement differences remain detectable. Measure both sides. Synthetic timing changes can test sensitivity, but establishing clinical relevance requires real annotations and agreement analysis. [FSGait](https://link.springer.com/chapter/10.1007/978-981-96-0960-4_19) and [GenGait](https://arxiv.org/abs/2604.01997) already make fine-grained gait analysis relevant prior art; joint localization alone is not enough.

I would defer a normative reconstruction or treatment recommendation claim. The present system has no validated coordinate decoder, and producing a plausible alternative movement does not establish a safe or appropriate rehabilitation target.

## 5. Treat adaptation and stochasticity as focused secondary experiments

For staged condition adaptation, keep the normal checkpoint immutable, as the code already does. A stronger future question is whether one shared representation can preserve normal-score calibration while improving recognition. Compare a frozen anchor, ordinary replay, distillation, pooled training, and a proposed explicit score-preservation constraint under matched budgets. The new claim must concern a measured retention-versus-recognition tradeoff. Keeping a backup checkpoint alone is good practice rather than a new continual-learning method.

For positional stochasticity, first establish the deterministic reference and then repeat matched runs with a small, prespecified noise grid. Preserve initialization, masks, batch draws, and update counts. Investigate whether changes affect pose-noise sensitivity and movement sensitivity differently. The existing single seed supplies a pilot observation only.

For stochastic latent variables, compare a deterministic predictor, simple fixed noise, and a learned prior/posterior model at matched encoder capacity and training budget. At test time, sample from the prior without the target. Report sample count, runtime, diversity, calibration, and representation quality. A fixed-sample-count minimum residual can be evaluated retrospectively once the future is observed, but it is not a proper predictive-density score or a forecast-selection rule available before that observation. It depends on sample count and needs separate calibration and matched sampling cost. Learned uncertainty should not be added until there is a concrete uncertainty question and data to test it.

## 6. Preserve a valid evaluation after inspecting the current tests

All three current test splits have been inspected. They can support the original descriptive results and transparently labeled follow-up analyses. A new partition of the same reviewed sources does not automatically become untouched confirmatory evidence. Preserve the existing results, record the timing of every new hypothesis, and use genuinely unexamined participant or source data for a new confirmatory claim.

For development, source-grouped internal resampling can help choose schedules without introducing window leakage. For final evidence, audit repeated people and duplicated media as far as the available metadata permit. Record verified identity groups separately from source IDs. More windows from the same recording do not solve a shortage of independent people.

Choose the evaluation unit before analysis. For five-class recognition, retain the current source-weighted macro-F1 as a primary outcome and calculate paired uncertainty for that same weighting. Keep sequence-weighted results as a separate outcome. For normal-only detection, state exactly how mixed-label source rows and source clustering are handled. Report per-class support, attrition, and uncertainty rather than only pooled performance.

For the implemented rule, the finite-sample threshold rank is `ceil((n + 1) * (1 - alpha))`. At `alpha = 0.05`, at least 19 normal calibration units are needed even to obtain a finite threshold. Nineteen is a mathematical minimum, not a sample-size recommendation for precise clinical performance. The units and test population also need the relevant exchangeability assumptions; unknown repeated people and distribution shift weaken that interpretation. See [Angelopoulos and Bates](https://arxiv.org/abs/2107.07511) for the broader calibration framework.

Avoid strengthening calibration language by using multiple windows from five people as if they were many independent normal subjects. If the available data cannot support a useful operating point, report ranking and the limitation honestly. A separately prespecified exploratory operating point can be informative, but does not establish a 5% guarantee.

## 7. A realistic plan for the remaining days

This is a prioritized schedule, not a promise that all training will finish within a laptop budget. Measure one run's duration and choose a defensible scope. Quality of the comparison matters more than the number of variants.

| Date | Main deliverable | Decision rule |
| --- | --- | --- |
| November 14–15 | Fix the paper question; audit a stratified sample of timing/person alignment; inspect training and teacher dynamics on development data. | If annotation problems are material, repair and disclose them before interpreting model behavior. Preserve the original results as superseded where appropriate. |
| November 15–16 | Run the update-budget diagnostic and one normal-only anomaly comparison. | If pretraining remains far from a meaningful training regime, prioritize this over positional or latent stochasticity. |
| November 16–17 | Complete one temporal-context or pose-quality experiment, with a prespecified scoring rule. | Choose one mechanism for the paper. Treat hypotheses formed after the current tests as exploratory. |
| November 17–18 | Finish matched evaluation, consistent intervals, attrition summaries, and a small representative failure gallery. | If no specific finding is supported, narrow the manuscript to an explicitly exploratory report. |
| November 18–19 | Write the seven-page paper, check every numerical claim, and review anonymity. | Do not turn proposals into completed contributions or omit unfavorable controls. |
| November 20 | Final source check and submission before the verified system cutoff. | Confirm the cutoff timezone in the submission system; the workshop page does not state it. |

A compact seven-page allocation could be one page for the question and closest prior work, one for cohort and protocol, one for the model and diagnostic contrast, two for controlled experiments, and two for failure analysis, interpretation, and limitations. Adapt the allocation to the actual findings. The [ASTAD requirements](https://ismart.ece.mcgill.ca/astad4/) allow two additional pages for references only.

The strongest paper will answer one difficult question clearly. If training is the main explanation, document that mechanism rather than claiming a general failure of world models. If pose quality dominates, show the controlled evidence and the limits of the mitigation. If temporal context adds useful information under matched conditions, demonstrate that contribution without implying diagnosis, disease progression, or clinical deployment.
