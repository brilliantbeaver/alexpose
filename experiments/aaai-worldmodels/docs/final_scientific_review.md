# Independent review of the released empirical study

The completed experiments support a small, controlled methodological case study. They do not establish that JEPA outperforms Random Forest, that accurate forecasts reliably identify abnormal gait, or that latent dispersion provides clinically calibrated uncertainty. Their most useful finding is that forecasting error, abnormality ranking, and uncertainty quality can give different answers on the same observations.

The reviewer participated in literature review and protocol design but did not author the model, data adapters, evaluation implementations, or result artifacts. Implementation review was independent of their authors. Real final outcomes were inspected only after the coordinator explicitly confirmed the joint release on November 16, 2026. Before reading outcomes, the reviewer verified the completion status, its matching joint-release hash, all four final result JSON hashes, and all 84 pinned CSV hashes.

Primary evidence is the [joint release](test_release.json), [completion record](test_execution.json), [protocol](revision_protocol.md), [amendments](revision_amendments.md), and [implementation review](revision_review.md). No model, score sign, threshold, or hyperparameter was selected from the final results. Earlier software had already computed quarantined test outputs; the authors report not inspecting their values for development decisions. The study must therefore be described as an internal source holdout with this limitation, not an independently preregistered untouched clinical test set.

## Observation and execution scope

| Experiment | Final observations | Training / validation / test sources | Held-out label support |
| --- | --- | --- | --- |
| Sixteen-frame classifier cohort | 405 windows, 89 sources | 53 / 19 / 17 | Normal 6; Parkinson's 2; stroke 2; cerebral palsy 1; myopathic 6. |
| Thirty-two-frame forecasting and quality cohort | 407 windows, 91 sources | 53 / 19 / 19 | Normal 6; Parkinson's 2; stroke 4; cerebral palsy 1; myopathic 6. |

These cohorts are different and must not be pooled into a single test population or treated as matched cross-task comparisons. Sources are video groups, not verified unique patients. Repeated windows do not increase the independent source count. The dataset labels are annotations rather than independently adjudicated diagnoses.

The forecasting models use 84 normal training windows from 17 sources and 27 normal validation windows from six sources. Three seeds share 200 normal-only uniform-pretraining updates, followed by 300 forecast updates per arm. Each forecast uses sixteen past and sixteen future sampled frames. Sampling every second native frame creates different physical durations across sources: the corrected full-cohort future interval ranges from about 1.07 to 2.13 seconds, while the corrected test interval ranges from about 1.07 to 1.33 seconds. The earlier 3.2-second maximum belonged to an excluded source and does not describe the final cohort. Prediction concerns torso-scaled, body-relative 2D articulation, not physical displacement or clinical progression.

Independent visual review identified person switches and scene cuts in one abnormal training source and one abnormal validation source. Both whole sources were removed. The long cohort's normal training, normal validation, and complete test inputs remained exactly unchanged, allowing documented forecast-checkpoint inheritance. The short cohort was freshly gated and all its models were refitted. Its test support decreased from 18 to 17 sources through the fixed gate, before test-outcome inspection. Source exclusions, pose filtering, and the short refit's effective stage budget changed together, so their effects cannot be attributed to a single isolated cause.

## Primary supervised comparison

The primary JEPA comparator remained the mixed-adaptation probe, fixed before the clean refit. Its result must not be replaced by the best secondary test model.

| Direct binary model | Balanced accuracy | AUROC | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Random Forest | 0.788 | 0.894 | 0.909 | 0.667 |
| Untrained-encoder probe | 0.818 | 0.788 | 0.636 | 1.000 |
| Normal-only JEPA probe | 0.780 | 0.742 | 0.727 | 0.833 |
| Staged-adaptation JEPA probe, secondary | 0.909 | 0.909 | 0.818 | 1.000 |
| Mixed-adaptation JEPA probe, primary | 0.818 | 0.848 | 0.636 | 1.000 |

On the 17 held-out sources, the primary mixed JEPA minus RF balanced-accuracy difference is **0.030**, with a paired 95% source-bootstrap interval of **[−0.189, 0.250]**. This provides no clear evidence of superiority. The primary JEPA model trades lower abnormal sensitivity for higher normal specificity. Specificity 1.0 means six of six normal sources were classified correctly; it is not a reliable estimate of perfect future clinical specificity.

The untrained probe reaches the same balanced accuracy as the primary mixed probe. This is a consequential control: the primary readout result alone does not establish a benefit from JEPA pretraining. The staged probe's favorable secondary point estimate is worth reporting, but its paired difference against RF also has an interval spanning zero, **0.121 [−0.091, 0.333]**, and it was not the prespecified primary representation. Classification used one encoder seed, so training-seed stability remains untested for this task.

The direct binary objective corrects a real endpoint mismatch. Summing four abnormal class probabilities from a balanced five-class model and thresholding at 0.5 classified every held-out source as abnormal, producing balanced accuracy 0.5. Direct binary training improves that decision rule. However, the five-class RF's derived abnormal score already had AUROC **0.955**, exceeding the directly trained binary RF's 0.894. Therefore the correction improves the fixed decision endpoint, not every aspect of ranking or calibration. It is not evidence that the earlier representation lacked class information.

Five-class recognition remains limited: RF macro-F1 is 0.310, mixed JEPA 0.326, staged JEPA 0.314, and the untrained and normal-only probes both 0.520. These secondary models retain the inherited hyperparameter-selection rule based on binary validation balanced accuracy and the first candidate on ties; they were not optimized for five-class macro-F1. The single cerebral-palsy source and two Parkinson's/stroke sources make condition-specific recalls unstable. These results do not establish robust differentiation of the five health conditions.

## Primary forecasting question

The prespecified normal-validation gain criterion failed for all three deterministic residual seeds. Final test evaluation retained every arm as planned.

| Predictor | Seed 27 normal MSE | Seed 41 normal MSE | Seed 73 normal MSE |
| --- | ---: | ---: | ---: |
| Persistence | 0.055116 | 0.055116 | 0.055116 |
| Direct deterministic | 0.074278 | 0.071690 | 0.080599 |
| Residual deterministic, primary | 0.054642 | 0.054046 | 0.055124 |
| Residual latent-variable | 0.055575 | 0.054759 | 0.054778 |
| Residual deterministic, weight-10 sensitivity | 0.054743 | 0.053948 | 0.055265 |

The primary source-wise mean persistence-minus-residual gain is **0.000512**, with interval **[−0.000811, 0.001859]**, across six normal test sources after averaging the three seeds within each source. This is a small, uncertain difference. Two seeds have positive gains; the third is slightly negative. A positive point estimate on the final set does not erase the failed development criterion or establish a reliable forecasting improvement.

The direct predictor is worse than persistence in every seed, despite reducing its coordinate training loss. Simple references help interpret this: population-static normal MSE is 0.058918, an untrained zero direct readout is 0.504937, and constant-velocity extrapolation is 3.703764. The learned direct model improves over its zero initialization, but does not exceed the strongest elementary reference. The residual architecture begins exactly at persistence, so its near-persistence result cannot itself demonstrate learned temporal dynamics.

Temporal controls weaken a motion-dynamics interpretation. Replacing the residual model's visible past with a static prefix reduces MSE by **0.000962**, with paired interval **[−0.001604, −0.000325]** for controlled-minus-original error. Shuffling the first fifteen past frames changes MSE by only **0.000040**, with interval **[−0.000072, 0.000207]**. Static prefixes also improve the latent residual arm. These findings do not support a claim that richer ordered temporal context is responsible for the residual model's performance.

The single weight-10 sensitivity was motivated by training-only gradients, not final performance. Its final gain over the primary weight-1 model is **−0.000048**, with interval **[−0.000212, 0.000090]**. It does not establish a remedy. All three sensitivity seeds remain reported, and no additional objective-weight search is warranted by this study.

## Stochastic predictions and uncertainty

The latent-variable model improves normal-source energy score relative to the deterministic residual model in every seed. The source-paired mean gain is **0.007270 [0.006610, 0.007890]**. Its mean-coordinate MSE is within the declared 10% descriptive tolerance in every seed. This supports a narrow statement that the complete sampled predictor has a better energy score in this experiment.

It does not establish useful error-based deferral. On normal test sources, uncertainty AURC versus prefix-quality AURC is 0.0483 versus 0.0389, 0.0460 versus 0.0381, and 0.0592 versus 0.0384 for seeds 27, 41, and 73. Lower is better, so simple prefix quality wins in all three seeds. The same direction holds over all 19 test sources. Normal-source dispersion/error Spearman correlations are 0.486, 0.657, and −0.371, showing seed dependence. Adjusted correlations are unavailable on the six normal sources because the nuisance model leaves too few residual degrees of freedom.

The stochastic model has additional parameters, and there is no equal-parameter deterministic model or unconditional-noise comparator. Better energy score therefore cannot be attributed uniquely to conditional stochasticity. No clinical probability calibration, prospective deferral policy, or safety benefit is established. Six normal calibration sources also cannot support a finite threshold with the proposed 5% conformal false-positive guarantee.

## Abnormality ranking is a separate endpoint

The anomaly scores and their higher-is-abnormal direction were declared before final release. The direction must remain fixed, including when a score ranks below chance. Abnormal prevalence is 13/19, or 0.684, on this cohort.

| Predictor or reference | Raw MSE AUROC | Energy AUROC | Excess MSE AUROC |
| --- | ---: | ---: | ---: |
| Persistence | 0.167 | Not the primary reference score | Not applicable |
| Direct deterministic, mean score across seeds | 0.744 | 0.705 | 0.923 |
| Residual deterministic, mean score across seeds | 0.167 | 0.167 | 0.654 |
| Residual latent-variable, mean score across seeds | 0.167 | 0.167 | 0.615 |

Excess MSE means model MSE minus matched persistence MSE. The direct model's excess-score AUROCs are 0.962, 0.923, and 0.846 by seed. Its mean-seed score has AUROC **0.923 [0.744, 1.000]** and average precision **0.959 [0.874, 1.000]**. Its paired AUROC gain over persistence is **0.756 [0.385, 1.000]**. This is a favorable **secondary** anomaly-ranking result from a model that is comparatively poor at normal forecasting. It cannot be used to claim that forecasting skill caused discrimination, or to replace the residual forecast primary with a post hoc winning arm.

The nine mean-seed arm/score comparisons and their per-seed counterparts are all retained in [the full tables](results_tables.md). These intervals are descriptive and are not multiplicity-adjusted simultaneous confidence statements. Residual excess-score intervals are broad: deterministic AUROC 0.654 [0.308, 0.962] and latent-variable AUROC 0.615 [0.269, 0.936]. A gain over the below-chance persistence score does not by itself establish reliable above-chance discrimination.

Persistence error averages 0.0551 on normal sources and 0.0218 on abnormal sources. Thus, greater raw prediction error is not a reliable synonym for the annotated abnormality in this sample. This observation does not identify a causal neurological mechanism. Differences in visible motion, view, detection quality, activity, or source composition remain plausible explanations.

The acquisition-only classifier uses the same 19-source forecast cohort. Prefix quality yields AUROC 0.500 and balanced accuracy 0.513; a separately labeled retrospective model using future quality yields AUROC 0.679 and balanced accuracy 0.596. These limited features do not exhaust acquisition confounding, so a weak prefix classifier cannot prove that the motion score is free of shortcuts. The quality classifier uses abnormal training labels while the forecasting models train only on normal observations, so their comparison is descriptive rather than a matched learning-regime superiority test.

## Defensible paper claims and remaining limits

A suitable narrow framing is **a controlled case study separating forecast skill, predictive uncertainty, and gait-abnormality ranking**. The evidence supports reporting a source-aware protocol, identity-quality failures caught before final release, useful endpoint controls, a negative or inconclusive primary forecast comparison, a narrow energy-score gain, and the mismatch between forecasting skill and secondary abnormality discrimination. Existing gait forecasting, normative reconstruction, and stochastic skeleton-anomaly literature preclude a credible first-method claim for the generic combination.

Do not claim clinical diagnostic validity, external-site generalization, causal neurological explanation, patient-specific disease simulation, treatment-response prediction, reliable five-condition recognition, JEPA superiority over RF, or calibrated medical uncertainty. The small source count, unknown patient overlap, observational labels, manual review limited to flagged identity failures, bounded sequence prefixes, variable-duration horizons, compact compute budget, and multiple descriptive comparisons constrain interpretation. Confidence intervals condition on the observed sources and trained seeds rather than representing a future clinical population.

The user-requested final real-notebook replay is now independently confirmed: the current `notebook_execution_real.json` matches all eight source and executed-output hashes, all 88 code cells completed, and no error outputs remain. The separate direct-binary video adapter has also passed the execution check recorded below. These checks support proceeding with a cautiously framed empirical draft.

## Analysis and figure acceptance

The completed [analysis report](research_results.json) was checked against its pinned analysis-code identity. Its principal effects agree with independent calculations from the released source CSVs: normal-source persistence gain, static/shuffle control effects, stochastic energy gain, and every displayed mean-seed anomaly AUROC and average precision. The full table preserves all nine secondary scores and the additional fixed forecasting baselines.

The reviewer opened all four rendered PNG figures: binary comparison, forecast comparison, anomaly comparison, and uncertainty tradeoff. Text, axes, and intervals are readable without overlapping labels. The diagrams correctly distinguish the 17-source binary task, per-seed forecasting gains, all nine anomaly-score comparisons, and better energy scores accompanied by worse selective prediction. The SVG versions preserve editable vector content. A figure's positive point estimate must still be read with its paired comparison and source-count limitations in the text.

Statistical and visual review is accepted for a narrowly framed empirical report. The separately verified final notebook replay and binary adapter complete the recorded execution checks. These acceptances do not establish clinical validity.

## Direct-binary video adapter acceptance

The new standalone `worldmodels/binary_inference.py`, SHA256 `479dd6aeb1e9b8323a030e8f4ab7a3cba023bf767e809493d0d9ea0b124d8881`, serves the actual sealed direct-binary RF and fixed mixed-JEPA classifiers. It does not substitute the five-class-derived abnormal probability. The reviewer verified its artifact checks, fitted-transform reuse, probability-column alignment, window averaging, fixed threshold, quality refusal, and retained person-review requirement. Five focused tests passed independently in 0.94 seconds.

The recorded training-source smoke artifact has SHA256 `95e7755e40a8180dfef9c2b4e3cfa7c61d2237cec446092e78f5b0285ca0425d`. The reviewer verified that source `4KU6G7Iv8ZM`, sequence `cljxm3ajs004m3n6lk7unsn04`, belongs to the clean training partition, then independently reran actual video decoding and MediaPipe inference over frames 1376–1406. Both classifiers reproduced the saved single-window probabilities: RF abnormal probability 0.2276488 and mixed JEPA 0.5151616. Their different decisions at threshold 0.5 are retained honestly.

This completes a real video-to-binary-prediction software check without changing frozen study modules or training new models. It is not additional held-out efficacy evidence. Arbitrary input videos lack GAVD annotation boxes, so manual review of the intended person and scene continuity remains required.
