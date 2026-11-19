# A positive, evidence-based direction for draft03

The strongest positive story is that **the other leg can improve normal-motion completion under a controlled observation task, and this improvement can be measured without treating it as a diagnosis**. The completed study supports that statement in both web-video skeletons and an independent laboratory-angle dataset. It does not support improved anomaly detection by the fixed score fusion, a JEPA advantage, or recovery of a neurological mechanism.

This document proposes a change in emphasis, not a change in results. The focused challenge was a declared secondary diagnostic. Giving it more prominence in draft03 is a post-outcome editorial choice and must not turn it into a retrospectively declared primary endpoint. The original detection contrasts should remain visible and unchanged.

## The positive result worth leading with

Use the U-trained reference model as the principal example. This avoids choosing the largest observed gain from the three training policies.

On the focused E_leg completion task, the U model reduces source-mean MSE from **0.080769 for the own-capacity predictor to 0.065020 for the both-leg predictor**. The absolute benefit is **0.015748**, with the saved conditional interval **[0.010721, 0.021300]**, over **28 held-out normal recordings**. This is a **19.5% reduction relative to the own-capacity mean MSE**. The percentage is a ratio of the reported source means, not a mean of participant-specific percentages.

All three training policies have positive focused-task benefit intervals. Their mean MSE reductions relative to their own-capacity controls are approximately 13.1% for J and 23.4% for A. Those figures describe the fitted families on the declared challenge; they do not establish a superior masking policy or an effect independent of effective model capacity.

| Correction training | Focused own-capacity MSE | Focused both-leg MSE | Absolute benefit | Conditional benefit interval |
| --- | ---: | ---: | ---: | --- |
| U, the reference | 0.080769 | 0.065020 | 0.015748 | [0.010721, 0.021300] |
| J | 0.080057 | 0.069567 | 0.010490 | [0.007229, 0.014019] |
| A | 0.078371 | 0.060009 | 0.018362 | [0.012688, 0.024636] |

These values come from the frozen `normal_error_table.csv` and saved normal-benefit summaries. The intervals use 5,000 source resamples conditional on the fitted models and split. They are not three independent replications, because the same recordings and retained baselines are reused.

The independent laboratory experiment provides a second positive anchor. For 128 held-out healthy participants, mean MSE is **201.038 for f+h and 190.584 for f+g**, in squared degrees. The aggregate reduction is about **5.2%**, with fold-specific reductions of **4.78% to 5.52%**. The post-outcome interpolation audit gives **309.838 for direct interpolation** and **206.262 for the shared ridge baseline**. The ordering interpolation, f, f+h, f+g holds in every fold. This makes it difficult to dismiss the positive result as improvement over only an obviously inadequate local interpolation rule.

The video and laboratory results use different observation systems and targets. They support replication of a useful prediction comparison, not transfer of weights or equality of physical errors.

## A useful research question for the paper

Ask: **When do observations of the other leg help complete a missing movement, and what must be checked before using that help as anomaly evidence?**

This question has a positive answer with informative limits. The other leg helps the fitted normal reference on the focused missing-leg task and on the laboratory angle task. Much smaller benefits appear on ordinary uniform masking, where nearby own-leg geometry can make completion easier. The timing experiment then shows that the same predictor cannot be assumed competent after visibility intersections remove much of its context.

The contribution is an observable test of predictive help with explicit controls. It is not a claim that conditional prediction, tube masking, residual fitting or normal-distance scoring is new by itself. Its useful combination is:

1. Both predictors answer the same physical-coordinate or angle question. Different contextual teacher representations cannot silently change the target.
2. A retained own-leg solution and an additional own-leg branch make the comparison stronger than a weak local model versus a larger whole-body model.
3. Structured training and structured evaluation are separated. The reader can see whether a result comes from learning or from changing the missing-data question.
4. The same evidence sequence is checked on a separate participant dataset, with explicit fitting, reference and test roles.
5. Completion benefit, support adequacy and detection improvement remain separate outcomes. A successful first step does not authorize a clinical claim about the last one.

The retained solution and zero option are useful design properties. They preserve an available local prediction when a correction is switched off. They do not guarantee that a correction selected on one support distribution remains helpful on another support distribution, as the timing results demonstrate.

## How to connect the positive result to neurological reasoning

Use neurological knowledge to explain why the question is worth asking, rather than to name the learned quantity after a neural mechanism. A coordinated walking pattern creates temporal and geometric relationships between the legs. Stroke and other conditions can change movement amplitude, timing, compensation and support needs. A normal-trained predictor may therefore find the other leg useful in some circumstances and misleading in others.

The external results make this distinction concrete. The both-leg branch helps the healthy group in every test fold. In the stroke group, its mean gain is negative in three folds and approximately zero in one. This is a descriptive group pattern already present in `heldout_prediction_error.csv`. It is compatible with a mismatch between the normal prediction rule and some stroke walking patterns. It is not a direct measurement of neural coupling or proof of its cause.

Several alternatives remain: walking speed, movement range, the short observed interval, shared PiG geometry, unequal basis ranks, model misspecification, or a different relationship between visible and missing angles. Stroke errors can be lower than healthy errors because less movement can be easier to predict. The manuscript should not equate prediction error, benefit, or regularity with severity.

A useful phrase is **“normal-pattern completion reference.”** Avoid “neural coordination index,” “brain world model,” or a disease-specific causal explanation. The experiment measures the behavior of a fitted predictor under a specified observation rule.

## Three positive claims, ranked by strength

| Claim | Strength | Boundary |
| --- | --- | --- |
| The fitted both-leg model improves normal completion on the focused video task and the independent laboratory-angle task. | Strongest completed positive claim. | It concerns these fitted bases, targets, support rules and cohorts. |
| Evaluation design changes how much useful context is exposed. | Strong and practically useful. | E_leg is a different task. It is not an isolated effect of training policy, nor proof of performance under natural occlusion. |
| The controlled workflow can distinguish useful prediction from unsupported anomaly interpretation. | Strong methodological lesson. | The study demonstrates the distinction; it does not prove improved clinical decisions or deployment safety. |
| A normal context-benefit pattern may differ in stroke walking. | Exploratory descriptive direction. | Existing group errors support the observation, but speed, amplitude, capacity and model assumptions remain unresolved. |
| Anatomical masking is the best training method, or relational fusion improves detection. | Unsupported. | The fixed-U masking contrast is inconclusive, and S underperforms M. |

The most defensible workshop paper combines the first three claims and treats the fourth as motivation for later work.

## Figure sequence for a positive narrative

**First, show the question with one real held-out example.** Place the measured target beside f, f+h and f+g predictions. Identify the permitted own-leg and opposite-leg observations, and show the exact mask. Use a deterministic example-selection rule and display a counterexample too. This makes the meaning of “help” visible without selecting only attractive results.

**Second, show useful completion on two observation tasks.** Keep U and E_leg in separate panels. Lead with the U-trained reference on each task, then show J and A as the declared secondary comparison. Annotate the 19.5% focused-task reduction as a ratio of means. Do not divide the focused benefit by the tiny uniform benefit and advertise a large improvement factor; the uniform benefit is uncertain and the tasks differ.

**Third, show the independent laboratory ordering.** Display interpolation, f, f+h and f+g normal MSE, with all four folds visible. Label the interpolation check as a post-outcome descriptive audit. It demonstrates that the positive prediction result is not limited to a single weak local comparator.

**Fourth, show the boundary of usefulness.** Place the original-support completion result next to the sparse paired-support timing result. The latter retains about 1.75 editable distal tokens and fails every paired-support adequacy gate. This explains why a predictor response is not enough to justify a coordination interpretation.

**Finally, keep the detection result visible.** Show M, R and S separately. The laboratory R score has AUROC 0.694, but the fixed fusion reduces M's 0.958 to 0.892. On the GAVD focal task, M falls from 0.694 to S's 0.576. These observations motivate better measurement and calibration research; they do not erase the completion result.

These figures can be assembled from saved predictions and source-level artifacts. They do not require new fitting or altered endpoints. Any new descriptive display should identify its derivation and avoid claiming preselection if it was designed after the outcomes were known.

## Suggested title and abstract direction

Suggested title: **When the Other Leg Helps: Controlled Gait Completion for Anomaly Analysis**.

A possible abstract is:

> Observations of the other leg can help complete missing gait motion, but this predictive help needs a controlled interpretation. We compare a retained own-leg predictor, an additional own-leg correction and a both-leg correction against identical observable targets. Uniform, joint-tube and anatomical-tube training share fixed evaluation tasks and explicit input-access checks. In an exploratory GAVD study, the uniform-trained both-leg model reduces normal completion MSE by 19.5% relative to the own-capacity control on a declared focused missing-leg diagnostic. An independent laboratory-angle experiment improves held-out healthy MSE by approximately 5.2%, with consistent improvement in all four folds and over direct interpolation. These positive prediction results do not imply detection improvement: the frozen relational-distance fusion underperforms the movement baseline in both datasets. A timing diagnostic also fails prediction adequacy after permitted context becomes sparse. The study establishes a reproducible comparison for measuring when contralateral observations help, and separates successful completion from unsupported anomaly or neurological conclusions.

The paper should state explicitly that its stronger emphasis on the focused diagnostic is an exploratory interpretation of already completed results. The original primary detection endpoint remains reported.

## Limitations that the positive framing must preserve

Equal feature width is not equal effective capacity. The GAVD residual ranks are 111 versus 141; the external ranks are 44 versus 62. The positive statement is that the fitted both-leg family predicts better under these controls, not that added physiology alone has been isolated after perfectly matching capacity.

The focused challenge uses artificial missingness. It does not establish successful recovery of truly occluded anatomical ground truth in arbitrary videos. Its targets are estimated image coordinates, which can themselves be wrong. The laboratory experiment uses processed PiG angles and provides a different measurement check, not direct validation of the video estimates.

The normal GAVD sample remains 28 previously explored recordings, with unknown participant identities. The external normal sample is larger and participant-separated, but it includes one selected trial per eligible person. Neither dataset establishes broad clinical deployment or etiology-specific interpretation.

The preserved negative detection result is a useful boundary, not something to hide or repair by selecting a new weight from test labels. R discards information during bilateral averaging and diagonal-distance construction, and M and R are not equivalently calibrated. Those are research opportunities for a future frozen study, not evidence that all other-leg information is useless.

## What can be done now without changing the scientific result

Reorganize the manuscript around the positive completion comparison and its conditions of validity. Add the saved external interpolation ordering, the focused-task absolute benefit and interval, and an accessible explanation of the equal-width versus unequal-rank limitation. Keep every training policy, the original primary anomaly comparison, the failed timing adequacy and the small JEPA-probe results available.

Correct the reporting issues from the draft02 review: distinguish 1,000-draw detection intervals from 5,000-draw normal-benefit intervals; name the availability-plus-view comparator accurately; describe the boundary failure as one frame; identify U in the timing figure; and distinguish fitting normals from reserved reference normals in the workflow diagram.

The resulting paper can be positive about what was learned: a controlled, reproducible way to demonstrate useful completion and identify where its interpretation fails. It should remain modest about algorithmic novelty and make no claim of a successful clinical detector.
