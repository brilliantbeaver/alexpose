# When the other leg helps

**There is a defensible positive paper here. Its strongest result is useful gait completion under clearly defined observation conditions. It is not improved abnormal-gait detection.**

Read the [manuscript PDF](manuscript.pdf), [plain-language manuscript](manuscript.md), [claim boundaries](claims_and_boundaries.md), and [readiness assessment](submission_readiness.md). This is an alternate interpretation of the same frozen studies reported in [draft02](../draft02/README.md). No model, score, source, mask or primary endpoint was changed to produce a more favorable result.

## What worked

Imagine hiding a short part of the left ankle's movement. First predict it using the visible left leg. Then retain that prediction and allow a correction using the right leg. A second correction using only the left leg provides a stronger comparison than a simple local baseline alone.

The fitted bilateral predictor performs better in two useful settings:

| Completed comparison | Positive result | Important boundary |
| --- | --- | --- |
| Uniform-trained model on the declared GAVD distal-leg challenge, 28 normal recordings. | Error falls from 0.080769 to 0.065020, a 19.5% reduction relative to the own-leg augmentation. The absolute-benefit interval is [0.010721, 0.021300]. | The task hides measured pose coordinates together. It does not validate invisible anatomical ground truth or independent patients. |
| Independent laboratory angles, 128 held-out healthy participants. | Error falls from 201.038 to 190.584 squared degrees, a 5.2% reduction with conditional interval [3.6%, 6.7%]. The bilateral predictor gives lower error for 109 of 128 participants. | The fitted feature families have different numerical feature ranks. This is not a capacity-free information estimate or video-weight transfer. |
| Anatomical versus uniform training on the same fixed GAVD distal challenge. | Error falls by 7.7%, interval [6.4%, 9.0%]. All four fold means and 27 of 28 recordings improve. | On the unchanged uniform task, the 0.8% gain is inconclusive, with interval [−1.0%, 2.6%]. This is task specialization, not general mask superiority. |

The U-trained reference leads the completion story so that the paper does not simply choose the largest result among three policies. Direct interpolation is also weaker than the fitted laboratory predictors in every fold. The additional intervals and positive emphasis were developed after outcomes were known. They are descriptive analyses of saved predictions, not new prospective primary tests.

![Measured completion and task-specific masking effects.](figures/positive_completion.svg)

## Why this is more useful than a negative-only story

The experiments show that the movement predictor learned something useful under the declared model families. The missing step is converting that behavior into dependable anomaly evidence. Those are different research problems.

A useful contribution is to make that distinction measurable. Both branches predict the same coordinates or angles. The local solution is retained. Hidden inputs cannot return through derived features. Training masks are compared on fixed test banks. The same question is tested with independent laboratory participants. Finally, the study checks whether the model still predicts adequately when a timing edit reduces permitted context.

This provides a concrete lesson for anomaly research: **a model should first demonstrate useful prediction on the observations it actually receives.** A nonzero response to an edit is insufficient. In this study, the timing test passes its recording-count floor but fails prediction adequacy on sparse paired inputs. That failure identifies a boundary of the successful completion result.

The neurological motivation remains useful. Stroke, Parkinsons, cerebral palsy and muscle disease can change posture, movement size, timing and compensation differently. A stable abnormal pattern can remain predictable. The method therefore preserves local movement descriptors while investigating a separate relational signal. It does not identify lesions, dopamine state, muscle force or disease causes.

## What the paper must still say prominently

The fixed anomaly score remains worse than local movement alone. GAVD focal AUROC falls from 0.694 to 0.576, and laboratory AUROC falls from 0.958 to 0.892. The laboratory relational score alone reaches 0.694, so it contains some abnormality-ranking information, but the experiment does not establish incremental detection value beyond the strong local baseline.

Matching input width is not matching effective capacity. The laboratory own-leg and bilateral residual ranks are 44 and 62. The appropriate positive claim is that the declared bilateral family predicts better on held-out data. The stronger claim that every difference in modeling flexibility has been excluded is unsupported.

The focused GAVD challenge was declared before fitting, but emphasizing it now is an exploratory editorial decision. The original primary detection result remains in the abstract, results and conclusion. All three training arms and both evaluation banks remain reported. No favorable subgroup replaces the original denominator.

## The most promising next research direction

Build on the demonstrated completion result by asking when its evidence is dependable. Use normal development data to map prediction adequacy across visible-context patterns. Strengthen the own-leg feature family so effective flexibility is more closely matched. Then freeze a normal-only calibration that reflects each channel's reliability before evaluating untouched participants or trials.

This could become a model that preserves posture and movement amplitude, uses opposite-leg evidence when its normal prediction is supported, and reports when the available observations are insufficient. The present results motivate that design; they do not show it already works. A larger JEPA alone would not resolve the current calibration, support and comparison problems.

The strongest current submission is a careful positive measurement paper with explicit negative detection findings. Its novelty is moderate and empirical. It is more defensible than either presenting the whole study as a failure or claiming a successful neurological detector.

See the [paired completion audit](advisory/completion_audit_v1/README.md), [independent method interpretation](advisory/method_positive_review.md), [independent framing advice](advisory/positive_framing_review.md), and [review responses](response_to_reviews.md). The [evidence record](evidence.json) links every numerical claim to saved artifacts.
