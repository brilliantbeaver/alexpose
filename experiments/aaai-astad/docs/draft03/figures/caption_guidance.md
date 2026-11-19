# Figure captions and reading guidance

## method

The workflow uses one gait window and the same hidden targets for three predictors. The cross-leg benefit compares both-leg correction with own-leg correction. The movement score comes directly from observed-gait descriptors. The cross-leg score measures how unusual the prediction benefit is relative to normal reference data; the combined score is their fixed average. A token contains one joint across four frames. An anatomical lower-leg group contains knee, ankle, heel and toe points on one side over a short time span. The diagram motivates measurements, not a claim to recover neurological causes.

## detection_difference

The combined score performs worse than the movement score in all four comparisons. Dots show Combined minus Movement AUROC; bars show the unchanged conditional 95% intervals. The first two comparisons use GAVD, the third uses the laboratory participants, and the fourth uses the age/sex-matched laboratory subset. Intervals condition on fitted models and the declared evaluation design.

## normal_prediction

All 28 held-out normal GAVD recording sources contribute. Each bar averages errors over the same hierarchy of targets, windows, sequences and recording sources; eight lower-leg joints receive equal weight. Scattered hiding removes individual tokens across the sequence. Grouped hiding scores nearby knee, ankle, heel and toe points hidden together on one side. The two panels test different missing-observation patterns, so their absolute errors should not be treated as a controlled comparison of training masks. Units are squared torso-scaled image coordinates.

## timing_limits

The experiment substitutes permitted other-leg samples from four frames earlier or later, preserving the original target and body reference. Four frames correspond to about 0.27 seconds at 15 Hz. The left panel compares the two corrections on the common unshifted observations available for each shift: both-leg prediction is worse before either perturbation. The right panel shows the change after shifting, with conditional 95% intervals. All values are unchanged; 27 held-out normal recordings contribute at each offset. Because unshifted prediction already fails to help, these responses do not establish a coordination mechanism.

## failure_cases

These are the unchanged high-error normal and low-error Parkinsons-folder examples selected by the saved review protocol. Curves show measured poses at samples 0, 15 and 31, with coordinates centered at the pelvis and scaled by torso size. Anatomical indices are mapped explicitly, and edges touching missing coordinates are omitted rather than filled. Viewpoint, amplitude and pose-estimation limits can affect error; these illustrations do not establish the cause of any individual score or verify a diagnosis.

## positive_completion

Left: the original mean errors for all four laboratory folds, each with 32 held-out healthy participants. Both-leg correction is compared with own-leg correction, giving a 5.2% mean reduction across participants; linear interpolation is the added descriptive benchmark. Right: the same both-leg predictor is trained with Anatomical or Random masking and evaluated on identical GAVD targets. Anatomical training reduces grouped-hiding error by 7.7%; the scattered-hiding difference is uncertain. Error bars are the unchanged conditional paired 95% intervals. These are post-outcome descriptive completion comparisons, not improved anomaly-detection results.
