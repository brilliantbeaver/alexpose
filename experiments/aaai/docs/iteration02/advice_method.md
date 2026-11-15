<!-- iteration02-execution-status -->
> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.

# Methodological advice for the second iteration

Prepared November 15, 2026 after inspecting the completed manuscript, the budget and control results, and `context_prediction.py` and `linear_context.py`. This document proposes a new exploratory study. It contains no newly trained model or new performance claim.

## Recommended direction

Use a small observable-coordinate predictor that preserves a competent own-leg solution, then test whether the opposite leg adds information. Keep a separate measure of unusual local movement. Examine the same predictor under source-matched timing changes to determine whether it uses alignment between the legs. This is a better use of the remaining week than increasing transformer size or adding a stochastic latent variable to the current teacher-target score.

The plain question is: when part of the left ankle movement is hidden, does seeing the right leg help after the model has already seen the available movement of the left leg? A second comparison asks whether the right leg helps because its timing is correct, rather than because any extra smooth trajectory changes the predictor.

This differs from the earlier own-joint comparison. Knee, heel and foot geometry can help reconstruct a hidden ankle without establishing a relationship between body regions. Make own-leg to both-legs the main information increment. Keep own-joint to own-leg as a geometry control. These information sets remain conditional on the shared pelvis and torso normalization, which already uses a body reference.

The candidate contribution is an adequately tested conditional prediction score that distinguishes local movement from cross-leg timing, with fixed targets, a retained baseline and matched observation support. Its novelty and utility require completed evidence. A residual connection or timing edit is not independently a strong contribution.

## What the existing results require us to change

The learned coordinate profile achieves primary AUROC 0.686, versus 0.701 for a random encoder with fitted coordinate heads. The longer-run values are 0.690 and 0.693. The common-coordinate control gives no reason to insist that the new score must depend on learned JEPA features.

The full-context coordinate head also has worse normal-development MSE than the own-context head in all four primary folds and three longer-run folds. In the current code, restricted inputs are separately re-encoded through the frozen transformer. A richer raw input therefore does not preserve the simpler branch's final representation. The next architecture should explicitly retain the simpler prediction.

The current ridge control is weak evidence about linear prediction limits. It fits separate joint, time and context regressions from 256 sampled windows using a fixed regularization value. Many target heads have few eligible observations compared with their flattened input dimension. A source-validated shared low-dimensional ridge predictor is a necessary next comparator.

Finally, the timing diagnostic retained only five fitting-normal sources across every edit. A new interpretation claim cannot rely on another similarly selected tiny subset. Audit available temporal support and normal-source retention before selecting the diagnostic.

## Candidate routes and priorities

The following scores are judgments from 1 to 5, not acceptance probabilities. Novelty potential assumes a completed, useful empirical result.

| Direction | Novelty potential | Deadline feasibility | Decision |
| --- | ---: | ---: | --- |
| Retained own-leg prediction with opposite-leg corrections and timing controls | 3.5 | 4 | Prioritize. It directly addresses the present estimator failure and creates a focused interpretation test. |
| Bilateral reflection-equivariant prediction | 2.5 | 4 | Use as an efficiency or consistency control. Preserve asymmetry magnitude rather than forcing the two legs to be identical. |
| Phase-conditioned recurrence or a small dynamical model | 3 | 3 | Gate on reliable cycle support. Most normal sequences are too short to assume good phase estimates. |
| A physical simulator producing sets of compatible neurological or muscular explanations | 4 | 1 | Defer. GAVD lacks forces, EMG and participant measurements needed to identify and validate the mechanisms. |

Do not combine all four into one new system. The neurological motivation supports examining coordination, amplitude and adaptation, but it does not justify a large architecture search on 28 previously explored normal sources.

## Prior-art boundary

[GenGait](https://arxiv.org/abs/2604.01997v2) already uses normal-only masked gait reconstruction, joint inconsistency scores and normative reconstruction. Its primary abstract describes validation with simulated abnormal patterns in held-out normative participants. Masked normal gait modeling and joint localization cannot be claimed as new here.

[FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) already separates posture and temporal abnormality, with sequence, frame and joint localization. A new spatial-plus-temporal diagram would not distinguish our contribution.

[Beyond Marginals](https://arxiv.org/abs/2509.15033) explicitly separates marginal, temporal and inter-variable structure in multivariate anomaly detection. The broad idea of separating individual movement from relationships is established. Our prospective distinction would be a specific retained-baseline estimator, anatomy-aware information sets, and verified controls on real heterogeneous gait video.

[Neural Granger Causality](https://arxiv.org/abs/1802.05842) already studies nonlinear temporal predictive dependencies. Our completion task can use observations after the hidden interval and should not be called Granger causality or a causal model.

Multivariate surrogate methods are also established. [Prichard and Theiler](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.73.951) make clear that a surrogate's retained and altered dependence structure must be specified. Their construction preserves cross-correlations for its intended null, whereas our timing diagnostic would intentionally alter selected alignment. It is not the same hypothesis test.

The clinical motivation is not itself novel. A [primary Parkinson's study](https://pubmed.ncbi.nlm.nih.gov/18412621/) relates bilateral stepping coordination to freezing in a specified patient comparison. It motivates studying timing but does not establish that our short monocular clips measure the same clinical quantity or that every Parkinsonian gait has poor coordination.

## Define the information sets and predictor

Let `O` contain permitted visible observations from the target leg, `C` contain permitted observations from the other leg, and `Y` contain the hidden local-coordinate patch. Left-leg membership is BlazePose 23, 25, 27, 29 and 31; right-leg membership is 24, 26, 28, 30 and 32. Keep the full existing 16-joint training and uniform masking allowlist. Shoulders and arms can remain in JEPA pretraining, but should not silently enter the main leg-only prediction branches.

Use the same target, sample weights and selected mask in all branches. Missing measurements must be excluded before input construction or attention. An availability indicator is permitted when supplied consistently. No interpolation or feature construction may recover a hidden target from its withheld coordinates.

Fit a competent local predictor `f(O)` using normal fitting sources. Compare a small declared set: coordinate interpolation where possible, shared ridge, and one small temporal model. Choose using source-grouped validation within the fitting pool. Pool sufficient examples across positions rather than fitting hundreds of nearly unsupported heads.

Freeze that selected prediction, then fit a correction:

`prediction_both = f(O) + g_theta(O, C)`.

Initialize the final correction layer to zero and include exactly zero correction as a selectable solution. The base prediction must be retained unchanged. This provides function-class inclusion and an exact software identity when the correction is disabled. It does not guarantee better error on unseen sources.

Fit a capacity-matched own-leg correction as well:

`prediction_own_extra = f(O) + h_phi(O)`.

Otherwise, improvement may result from extra fitting capacity rather than opposite-leg information. Use matched widths, target draws and optimizer budgets. Select a small residual-strength or regularization grid, including zero, using grouped internal normal validation. Never evaluate both alternatives on each test target and choose whichever predicts its known answer better.

A shared ridge residual is the first implementation. An S-JEPA residual is an additional learned-feature comparison with exactly the same base, observable target and information restrictions. A random encoder uses the same target too. Keep the current uniform stochastic masks, EMA teacher and anti-collapse regularization if pretraining is used; these are inherited components, not the new contribution.

## The timing diagnostic and its limits

For the same frozen full predictor, define:

`alignment_help = mean_k loss(Y, f(O) + g(O, shifted_C_k)) - loss(Y, f(O) + g(O, aligned_C))`.

Positive values mean that actual alignment helps this fitted predictor relative to the specified offsets. Do not fit a separate predictor for each offset. Keep target, own-leg observations, weights and observation support fixed. This is a model-response statistic, not neural coupling, mutual information or a physiological causal effect.

Use finite no-wrap offsets from the same source and annotated sequence. A fixed example is plus or minus four frames, with zero as an identity check, when the original interval is sufficiently long. The four-frame choice is a proposed engineering setting, not a validated optimal biological scale. Freeze the setting before new outer ranking.

Do not call this construction exactly marginal-preserving. Different interior crops can change empirical amplitude, posture and missingness. A circular shift preserves the finite list of values but introduces a seam; filtering that seam destroys the exact preservation argument. The short normal intervals make both issues consequential.

Move an observed limb-local block together rather than shifting individual joints independently. Each donor block can preserve its observed internal limb geometry. If visualization reconstructs a whole body at a current attachment point, check segment lengths and endpoint continuity, and do not claim that the resulting cross-limb combination is dynamically feasible. It is a conditioning diagnostic, not a simulated stroke or Parkinsonian patient.

Intersect context visibility and target support across aligned and shifted versions before scoring. Apply the same support mask to every version. Report raw support, retained support, dropped sources and marginal drift in amplitude, speed and bone-vector distributions. A low-variance or distant shift should not be selected retrospectively because it produces a strong result.

Controls should include zero shift, a matched whole-body temporal change, small coordinate noise, and missing-observation degradation. A whole-body temporal change also changes the target measurements, so it is a separate crop/phase sensitivity check rather than the same fixed-target intervention. Do not claim exact invariance to it or subtract it as if it isolates a causal mechanism. They answer different questions, so do not require every source to satisfy all four before displaying the basic aligned-versus-relative comparison. Use separate clearly labeled common-support cohorts for each paired comparison, plus a smaller all-controls intersection as a sensitivity. Never compare scores from different support cohorts as if they were paired.

A source rank among shifts is descriptive. It is not a randomization p-value or conformal guarantee. Autocorrelation, short duration, nonstationarity and selected support prevent an automatic exchangeability argument.

## Protect against a completion ceiling and phase leakage

Own-leg same-time knee, ankle, heel and foot geometry may nearly solve the completion target. A small opposite-leg gain can therefore reflect little remaining prediction error, not only an inadequate learner. Report base error, noise sensitivity and available improvement before interpreting a zero gain.

A separately declared contiguous temporal-gap diagnostic can create more demanding evaluation targets, while pretraining still uses the required uniform masks. This is a supporting evaluation, not a silent replacement for the main uniform-mask experiment. It must be fixed using normal development data and reported under its own target protocol.

Phase or cadence estimated from the full unmasked target trajectory leaks information into prediction. If phase is used, estimate it from permitted visible observations only and give the same clock to all branches. Add a clock-only control. Because the normal retained sequences are short, reliable-cycle counting should be an eligibility gate, and cycle-conditioned modeling should remain optional if it removes most normal sources.

## Preserve local movement and source-level evaluation

Do not discard local amplitude and posture. A person can have reduced arm excursion, crouch or restricted range while retaining regular cross-leg timing. Retain a local-motion score alongside alignment-help deviation. If combining them, declare the formula before outer scoring and compare the combination with each component on the same sources. Do not choose a fusion weight from abnormal test AUROC.

The data review's proposed lower-limb support rule retains 304 sources, including 28 normal sources, and a 90-source focal-condition cohort. These are prospective cohort counts, not results of the new method. Every retained normal source remains previously explored. A less restrictive support rule increases coverage but does not create independent participants or erase prior model selection.

Preserve all old runs. A new primary question may focus on normal versus the pooled Parkinsons, stroke, cerebral palsy and myopathic folders, but it must be called a new exploratory endpoint chosen after the broader study. Keep all 12 folders as a separate broad non-normal movement evaluation, and report exercise separately. Do not replace the completed 251-source primary result.

Refit every comparator under the new cohort and roles. An old-model score trained on the earlier source allocation is not a fair paired comparison with a new model trained on additional normal sources. Also report a common-source sensitivity, while acknowledging that even this does not equalize old and new training unless all models are refitted.

Within each outer fold, reserve five normal reference sources before model selection. Use grouped internal validation only within the other fitting normals. Refit selected choices on the fitting pool, freeze them, estimate reference parameters from the reserved normals, and score outer sources once. Report the fold counts and every role. Source videos remain grouping units rather than confirmed independent people.

Use paired source-level differences and declared within-fold aggregation. Windows, masks, edits and seeds are not additional participants. Five reference normals imply empirical rank granularity of at least 1/6; a precise low false-alarm claim would be untenable even before checking exchangeability. A newly named fold cannot make this a confirmatory study.

## Minimum comparison and decision gates

Compare the own-leg base, capacity-matched own correction, aligned both-leg correction, random-encoder coordinate correction, the established motion-feature distance, the earlier calibrated full-error method refitted on the new roles, and a simple lagged cross-correlation or waveform asymmetry descriptor. Call timing descriptors proxies unless event timing has independent validation.

Attempt one published pose-anomaly method within a bounded parallel workstream. [STG-NF](https://openaccess.thecvf.com/content/ICCV2023/html/Hirschorn_Normalizing_Flows_for_Human_Pose_Anomaly_Detection_ICCV_2023_paper.html) is lightweight in its native setting, but its missing-input and joint-layout assumptions still require validation. The existing feasibility report explains why a zero-fill adapter is not an exact reproduction. Report an adaptation as an adaptation, and keep the comparison gap visible if it cannot be resolved.

| Gate | Required observation | Consequence if it fails |
| --- | --- | --- |
| Observation support | Count eligible normal sources and paired target support before training. Inspect real pose overlays. | A small timing subset remains a pilot rather than the central validation. |
| Known dependencies | Recover added-context value on held-out simulated coupled processes; avoid systematic gain under conditional independence. | Fix the estimator before interpreting real-gait scores. |
| Normal adequacy | Both-leg correction improves grouped normal development against the frozen base and the capacity-matched own correction. | Select zero correction where appropriate and report the estimation failure. |
| Specificity | Alignment response survives fixed support and is distinguishable from the declared observation controls. | Do not attribute it specifically to cross-leg timing. |
| Detection | Useful paired improvement over local motion and calibrated controls on the declared focal endpoint, with broad results retained. | Do not claim a stronger detector. An estimator/control study may still be supportable. |
| Interpretation | Any anatomical explanation agrees with independent movement descriptions and observed quality. | Keep the output as a model statistic, not a disease explanation. |

Days one and two should establish support, the retained-baseline implementation and known-dependency checks. Day three should test normal adequacy and the timing controls. Day four should perform one frozen real-data comparison. Day five should write only the supported result and obtain an independent review. A failed gate must narrow the claim. Neither a larger model nor a favorable new seed repairs it.
