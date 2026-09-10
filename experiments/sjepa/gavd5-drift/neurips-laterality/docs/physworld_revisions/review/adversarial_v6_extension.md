# Independent final extension review of paper v6

Reviewed 9 September 2026. The complete v6 manuscript and appendices were checked against the extension audit, independent numerical recomputation, historical README ledger and current figure previews. All twelve local-link occurrences in the manuscript resolved at the time of review, including the newly delivered compact training figure. This review requires no further model training.

The central science is now internally consistent. The latest absolute scores, source-bootstrap mask intervals and new trained-versus-initial contrasts agree with the saved predictions. Shapes, per-clip target-count pairing, variable-count loss reduction, full-input regularization, current reflection flags and clinical limitations match the implementation. The clamped reflection-loss equation now matches Notebook 09. Historical evidence is visibly separated from the directly audited latest grid, including the positive augmentation effect on token consistency.

## Corrections before v7

| Priority | Location | Remaining issue | Specific change |
|---|---|---|---|
| Necessary | §6, verification paragraph | “Independently reproduces the latest grid's 125,000 prediction rows” implies rerunning inference from checkpoints. This revision verified saved rows and recomputed aggregates; it did not generate new predictions. | Write “independently audits 125,000 saved prediction rows and recomputes 200 pooled score rows and three saved mask intervals.” Keep the separate grid-hash/history checks. |
| Necessary | Figure 2 caption | It refers to “right-hand intervals,” while the actual figure puts the intervals in the bottom panel C. | Refer to “Panel C” or “the lower panel.” Panel B at right is the correspondence count. |
| Necessary | Appendix C table | Inherited column labels call the error “feature RMSE,” although the values are coordinate error after decoding features. | Use “Coordinate RMSE from observed future features” and “Coordinate RMSE from predicted future features.” The parent agent has already identified this correction. |
| Minor | §4.3 / Appendix A | Main text says Appendix A gives the bootstrap seed, but Appendix A currently gives neither seed 812 nor 2,000 resamples explicitly. | Add one sentence identifying 2,000 paired source-video draws, NumPy bootstrap seed 812, and no retraining; link the retained `review/extension_recomputed_summary.json`. |

## Numerical and methodological checks

The latest summary is accurate: initial mean-motion R² 0.222544, trained-teacher range 0.100777–0.114209, initial mean R² 0.070827, and direct pose R² 0.034541. Initial and trained means use 625 clips/93 source videos for each seed. The five teacher-minus-initial intervals in Appendix A match the independent computation and are correctly described as exploratory marginal intervals conditional on fitted models. The historical Appendix G values match the preserved README rather than claiming newly restored raw artifacts.

The target keeps observed timestamps and common bilateral transition support; model inputs are separately interpolated, normalized and resized. The paper does not claim that this preparation preserves original timing or calibrated geometry. It states the full 64×33×3 model input, four-step patching, 16×33 token grid and 96 feature channels without conflating the synthetic 16-frame/16-channel reflection example.

Cross-entropy remains the correct description of the real-data masked objective. The shared gait-pooled VICReg branch receives full geometric views, and the teacher receives no gradients. Laterality enters through anatomical identities and supplied priors; the signed target enters only the readout. The latest grid has no reflection augmentation or explicit reflection penalty. The variable-count pathway correctly matches target counts within each paired clip while allowing different counts across clips.

Source separation and the limits of inner validation are accurately stated. The bootstrap resamples source videos jointly across clips and seeds, while fitted encoders and source partitions remain fixed. Repeated seeds and diagnostic rows do not create more independent recordings. The adaptive development history is visible, so no external preregistration or untouched confirmatory dataset is implied.

The correspondence interpretation is improved: contextualized teacher features may use information already available in the visible context, and their mismatch advantage does not establish recovery of withheld motion. This resolves the main earlier risk of overstating the positive diagnostic. The paper also preserves the possibility that another readout could access useful learned information.

## Figure and link consistency

The compact training figure is now present and readable. It shows the source groups, anatomical token grid, masked student, predictor, EMA teacher and loss, then the held-out readout. Its dotted reflection extension is labeled synthetic. The full-input regularizer is explained in the footnote and main methods. This is an appropriate main illustration; the larger diagram can remain supporting material.

Figure 2's plotted scores and mask intervals agree with the manuscript, and its dependence caveats are visible. The three panels are a useful presentation of the main finding. The seed-independent direct-pose row still displays repeated coincident seed markers; a single marker would be clearer, but the paper's explicit control-reuse statement prevents this from becoming a substantive inference error. The reflection schematic continues to identify its bodies and numbers as illustrative.

All manuscript local links currently resolve. Add the machine-readable interval output to Appendix A for a direct path to exact values. The paper does not yet have a verified final workshop-template page count; main-text word count and a two-figure layout express an eight-page intent only.

## Plain-English flow and final emphasis

The shorter main text now follows a coherent sequence from movement measurement to training intervention and then to predictor/readout disagreement. The earlier quantitative histories support that sequence from the appendices. The abstract remains strategic and avoids procedural numerical detail, while the main text carries the evidence needed to evaluate its claim.

A few terms would benefit from a brief explanation on first use. Define ridge as a linear regression readout with a weight penalty, and define equivariance as a predictable transformation of features when coordinates are transformed. In §4.1, “normalized token discrepancy q” would orient the reader before the historical q result. The methods otherwise explain the tensor and statistical notation adequately.

One dense sentence in §6 could be simplified. Instead of “The supported scope is post-development, within-GAVD performance on excluded source videos,” use “The evaluation holds out source videos within GAVD, which was used throughout development.” This states the same limitation in ordinary language. It does not imply that the held-out sources form a newly collected independent cohort.

The discussion gives a reasonable next experimental sequence using the existing checkpoints before revising training. Explicit reflection training remains one testable hypothesis after measurement and readout checks, rather than a claimed remedy. The paper should preserve this order and avoid adding more speculative mechanisms in the final version.

After the four concrete corrections above, no remaining extension-evidence error requires another experiment before delivering the seventh manuscript. A stronger empirical paper would still benefit from the proposed regularization and summary ablations, independent measurement validation and participant-separated confirmation; those are limitations of the current evidence rather than tasks accomplished by editorial revision.
