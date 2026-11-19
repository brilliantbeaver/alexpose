# Independent review of presentation positions 16–30

Reviewed the rendered slides, deck specification and speaker notes on 15 November 2026. Positions refer to the presentation order, not the original content IDs. This reviewer did not write these speaker notes and made no changes to presentation content.

- Deck specification SHA256: `965d8a236d3ab7b91253e009833d1e816dbd840d3382ed35819a6b74502c8ef4`.
- Presentation PDF SHA256: `b121a5100080869010838d42ac1c356f378cf31b02fa4cdbf64253e365225505`.
- Visual scope: every rendered page from `review/render-16.png` through `review/render-30.png`, grouped for comparison, with additional full-size inspection of positions 21 and 26.
- Scientific checks: saved completion summaries, the declared study evidence and the aggregation implementation in `aaai_gait/iteration_scoring.py`.

## Overall assessment

The sequence is understandable for a general AI/ML audience. It introduces the controlled comparison, shows completion results, defines the anomaly scores, and then evaluates recording-level detection. The images have clear labels, readable tables and enough space; I found no clipped text or overlapping elements. The transitions follow the reordered presentation correctly. Three wording corrections should be made before the notes are considered final. None requires changing the measured results or refitting models.

## Important corrections

1. **Position 24, content ID 22, explanation: distinguish averaging errors from averaging predictions.** The notes currently say, “We first average repeated predictions of the same physical target.” The implementation computes squared losses and benefits before averaging repeated mask draws (`compress_target_rows`, lines 171–175). Averaging coordinate predictions first would define a different estimator. Replace this with: “We first average the prediction errors and benefits from repeated masks of the same target, then aggregate targets, windows and sequences into recording-level measurements.” The first visible card could similarly say “Average errors for one target” rather than “Average masks of one target.”

2. **Position 26, content ID 24, visual and evidence notes: describe the three rows actually displayed.** The plot contains two Combined-minus-M comparisons and one Anatomical-minus-Random comparison. The notes instead say “The three combination comparisons lie below zero” and refer to “four independent confirmations.” The laboratory contrast appears only in the spoken explanation, not in this plot. Change the visual paragraph to say “The two combination comparisons lie below zero.” Either move the laboratory result to its laboratory slide or explicitly introduce it as an additional result that is not plotted here. Preserve the warning that the two GAVD populations share recordings.

3. **Position 21, content ID 41, transition: specify what stays fixed in the masking comparison.** “With the both-leg predictor ... fixed” can imply that the same fitted correction weights are used for Anatomical and Random training. The correction models are refitted. Say “with the both-leg predictor design and evaluation targets fixed” or “using the same predictor family and evaluation targets.” The shared baseline is fixed, as position 17 correctly explains.

## Optional polish

- Position 21 would be more precise with “Own-leg MSE” and “Both-leg MSE” as table headings. Its footer should identify squared torso-scaled coordinate units for GAVD and squared degrees for the laboratory data. The present wording omits that the video error is also squared.
- Position 16's visual paragraph describes connections between methods and banks, whereas the current drawing is a table. “Read across each training-method row” would match the figure directly. Position 17 likewise refers to following branches that are not drawn on its three-card layout.
- Position 22's phrase “keeping the both-leg predictor and evaluation task defined” is vague. “Using the same predictor design and evaluation targets” would make the control easier to understand.

## Checks that passed

The 19.5% and 5.2% reductions compare Both-leg correction with Own-leg correction. The 7.7% reduction instead compares Anatomical with Random training of the Both-leg correction, within Grouped hiding. The saved summaries support these values and their denominators. The uncertain 0.8% scattered-hiding result remains visible, and the notes disclose the post-outcome, conditional nature of the completion intervals and unequal numerical ranks.

Position 23 correctly defines Cross-leg R as unusual benefit relative to separate normal references, including unusually high or low benefit. It does not mislabel R as raw prediction error or disease severity. The AUROC evaluation retains within-fold pair weights and paired resampling. The plotted intervals distinguish a supported ranking difference from an uncertain mask advantage. Position 29 keeps the baseline-error fusion visible without claiming a gain from its slightly higher point estimate. Position 30 explicitly identifies STG-NF as a bounded adaptation, explains missing-value imputation and the training budget, and avoids claiming superiority over the published method or normalizing flows generally.
