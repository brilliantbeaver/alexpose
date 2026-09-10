# Final adversarial review of paper_v6.md

Reviewed 9 September 2026. The complete main text and Appendices A–H were read. This review checks scientific claims, historical versus current evidence, notation, coordinate calibration, statistical interpretation, and errors introduced by condensing and assembling the paper. It does not claim a new training run or an independent audit of every model checkpoint.

V6 has a coherent and substantially more credible argument than the starting paper. It now includes the favorable historical reflection-augmentation effect on the specified token discrepancy, while retaining the inconclusive predictive contrast. Its main evidence is the newer, recomputable readout comparison. It explains that input preparation changes the measurement path, that the initial-encoder control retains learned pose extraction and supervised readout fitting, and that source separation does not turn repeated development on these videos into an untouched confirmatory study. The central interpretation is appropriately conditional.

The following specific changes should be made for v7.

## 1. Write the negative token-error value as a difference

Section 4.1 currently writes “augmented-minus-vanilla q = −0.00843.” The discrepancy q is nonnegative, so putting a negative number directly after `q =` is an avoidable mathematical ambiguity. Write

\[
\Delta q=q_{\mathrm{augmented}}-q_{\mathrm{vanilla}}
=-0.00843,
\]

with interval [−0.01020, −0.00687]. The favorable interpretation is a reduction in the specified discrepancy, not a negative discrepancy value. Continue to identify it as a historically recorded paired result whose original prediction/checkpoint chain is absent locally.

## 2. Restore the promised bootstrap seed in Appendix A

Section 4.3 says Appendix A supplies the bootstrap seed, but Appendix A currently does not give it. The material appeared in an earlier version and was lost during condensation.

Add that these revision-time exploratory contrasts use 2,000 paired source resamples, bootstrap seed 812, and percentile 95% intervals. Keep the marginal/no-familywise-adjustment qualification and the fixed-pipeline conditioning. Distinguish this seed from the original reflection protocol's bootstrap seed and from the retained latest-grid mask intervals.

## 3. Describe validation of saved prediction rows accurately

Section 6 says the present review “independently reproduces the latest grid's 125,000 prediction rows.” No encoder inference was rerun to regenerate those rows. The review validated their coverage and used them to recompute the pooled scores and intervals.

Replace that sentence with wording such as: “The present review validates the coverage of 125,000 saved prediction rows, recomputes 200 pooled score rows and the three recorded mask intervals, and checks grid-table hashes and 125 complete update histories.” This clearly reports what was done and remains consistent with the next sentence's restriction on checkpoint verification. Do not imply that CSV hash validation authenticates unavailable original checkpoints.

## 4. Distinguish online tokens, teacher tokens, and downstream features

Appendix F defines `D_b` and `E_b` using online-encoder tokens `Z_theta` for an additional training penalty. Appendix G then defines the original audit as `q = D/E`, although the historical audit used **target/teacher encoder tokens**. The same algebra applies, but the encoder and gradient context differ.

Define the original q directly using teacher tokens `Z_bar_theta`, or state that D and E are recomputed for the frozen target encoder in the historical audit. Appendix F should retain online tokens and its correctly displayed detached, clamped denominator. There is no need to change the implemented loss.

Appendix G's lower-case `z` also remains undefined in the parity paragraph. Define it as the frozen encoder's 960-dimensional bilateral feature summary used in the original parity comparison. Capital Z denotes the joint/time token array; y denotes the observed-coordinate measurement; a fitted readout g predicts y. The identities `y(Mx) = -y(x)` and `g(Mx) = -g(x)` concern different quantities. The latter guarantees a transformation law but not prediction accuracy.

The diagrams use P for the joint permutation whereas the paper uses S. Harmonizing these symbols, or explicitly identifying them, would avoid making readers search for a second operation.

## 5. Correct the synthetic forecasting table's quantity label

Appendix C still labels its columns “Observed-future feature RMSE” and “Predicted-future feature RMSE.” The displayed numbers are errors in decoded future coordinates, not errors in the latent features themselves. Its paragraph below the table already makes this distinction correctly.

Use “Decoded-coordinate RMSE from observed future features” and “Decoded-coordinate RMSE from predicted future features,” with normalized-coordinate units in the caption. Preserve the two-synthetic-test-clip scope, the 24 common endpoints, the four-update budget, and the explicit absence of real-data forecasting evidence. Those limitations make the example useful as a pipeline illustration without granting it substantive forecasting efficacy.

## 6. Add the exact archive coordinate convention to Appendix D

The main text correctly rejects calibrated metric 3D speed, but “depth scaled relative to crop width” remains incomplete. The retained extraction code uses

```text
x = (crop_x0 + landmark.x * crop_width) / image_width
y = (crop_y0 + landmark.y * crop_height) / image_height
z = landmark.z * crop_width / image_width
```

A short appendix sentence or equation should give this convention. The important limitation is not merely a generic absence of calibration: x and y are normalized by different image dimensions, and depth is inferred. Pelvis and body-scale normalization do not recover metric geometry. The rest of Appendix D's timestamp, visibility, interpolation, and pelvis-fallback account matches the inspected original implementation.

## 7. Use “comparison jobs” when counting the 125 encoders

Section 4.2 calls the latest grid “50 paired fold/seed jobs, containing 125 encoders.” Readers may naturally interpret a paired job as two encoders, making the arithmetic seem inconsistent. The grid combines 25 motion jobs with three conditions and 25 region jobs with two conditions.

Write “50 matched comparison jobs across folds and seeds: 25 motion jobs with three conditions and 25 region jobs with two, totaling 125 encoders.” Subsequent pairwise contrasts still legitimately compare each alternative with its family reference. This is a workload clarification, not a change in the inferential unit.

## 8. Keep a few definitions close to the claims they support

The main text describes four inner folds in the original audit and three “in the extensions.” Notebook 08 used a fixed readout penalty, and some later notebooks are demonstrations rather than the same evaluated procedure. Narrow the sentence to the original audit and the later comparative/motion readout protocols, with the notebook map providing the exact scope.

The main text says motion policies increase “target-motion enrichment,” but does not define the statistic. A short appendix definition should explain that it compares the selected-token motion score with the eligible-token score under the audit's implementation. It is neither a percentage improvement nor a calibrated biological motion measurement.

Appendix A says every contrast uses the same “clips and seed,” which could sound like a single-seed analysis. Say “the same clips and each matched seed, then averaging the five seed-specific scores.” The current methods section defines that estimand correctly.

## 9. Appendix assembly and duplication check

Appendix E contains the clinical/annotation census table and a scope paragraph. It does not accidentally duplicate the old full introduction, and its clinical distinctions remain appropriate. The table deliberately uses muscular-dystrophy literature as motivation while refusing to equate the broad myopathic annotation with Duchenne disease.

Appendix H contains the detailed absolute-score table once, along with the notebook map. The main text gives selected summary values and Figure 2, so the appendix table is a reasonable supporting detail rather than an accidental duplicate. Retain it, but a short caption can point to the main result section and avoid repeating the entire initial-control explanation. The zero seed SD for deterministic baseline rows is appropriately interpretable from the reuse paragraph; one explicit phrase that it does not indicate zero source uncertainty would make this foolproof.

Appendices A and G distinguish newly computed learning contrasts from historically retained original reflection and Notebook 12 results. That separation is important and should survive canonical copies of the manuscript. The historical five-decimal values are “recorded precision,” not recovered machine precision.

## 10. Final presentation and evidence limits

The section structure now follows motivation, a known geometric hypothesis, measurement checks, controlled changes, and a next experiment that could resolve the remaining utility question. The paper avoids inferring complete loss of information or complete feature collapse. Its no-equivalence and no-power-guarantee language is appropriate. The proposed use of newly reserved or external sources acknowledges adaptive development on the current cohort.

For the final abstract, a short mention that the historical reflection augmentation improved the specified token score without an established prediction gain would improve the success/failure balance. This is optional if the abstract's emphasis remains the newer completed evidence, but the main-text historical success should remain explicit.

A source-bootstrap interval is conditional on fitted models and on treating video sources as the sampling clusters; unidentified repeated people or related uploads can undermine independence beyond that level. V6 has the correct scope restriction. Do not turn its larger evaluation-row counts into a claim of a larger independent sample.

The current word count is not a verified workshop page count. A future NeurIPS-template build with the final figures and bibliography is still needed before claiming compliance with an eight-page limit. Likewise, inline authoritative links are useful working references, but submission preparation requires a consistent bibliography and verified anonymity. These are submission-preparation tasks, separate from completion of the user's Markdown revision request.

The current governance record remains unresolved. V6 correctly says that local manuscript revision does not establish permission to submit or release derived data. This final review does not alter that status.

## Final revision priority

The essential v7 corrections are the negative-difference notation, missing bootstrap seed, saved-row verification wording, online-versus-teacher token distinction, and decoded-coordinate table headers. The coordinate convention and comparison-job arithmetic are useful precision improvements. After these edits, the paper's empirical interpretation is coherent: the observed advantages and deficits are properties of specified geometry tests and readout procedures on held-out GAVD source videos, with historical and newly recomputed evidence clearly separated.
