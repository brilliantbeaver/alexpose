# Independent adversarial review of paper v5

Reviewed 9 September 2026 against the complete motion/region grid, current implementations, `docs/README.md`, and the rendered PNG previews for the three available figures. The entire manuscript, including Appendices A–D, was inspected. Earlier versions and reviews remain preserved.

The latest numerical results and Appendix A agree with the independent recomputation. The central table correctly distinguishes initial, online and teacher encoders, retains the direct-pose control, and reports source coverage and seed variation. Cross-entropy, the full-input gait-pooled regularizer, the 0.5 gait-derived motion budget, precision and source separation are now represented accurately. The draft is scientifically stronger than v3; its main remaining weaknesses are compression, a small equation mismatch and incomplete historical balance.

## Corrections required before the next version

| Item | Finding | Concrete correction |
|---|---|---|
| Reflection-loss denominator | The displayed equation adds `epsilon_floor` to the detached energy, but the implementation clamps that energy below at `1e-12`. The following sentence acknowledges the difference, leaving the mathematical method and code inconsistent. | Write the denominator as `max(stopgrad(energy), 10^-12)` or define a clamp operator. The distinction is small numerically for ordinary inputs but should be exact in a methodology paper. |
| Figure 1 asset | V5 references `figures/training_pipeline_compact.svg`, which was absent from the figure directory at review time. The expanded diagram exists and renders. | Deliver and inspect the compact SVG before claiming all figure links are ready. This may already be in progress with the figure author. |
| Synthetic RMSE labels | Appendix C headings say “Observed-future feature RMSE” and “Predicted-future feature RMSE,” although the values measure decoded coordinate error. | Use “Coordinate RMSE from observed future features” and “Coordinate RMSE from predicted future features.” Retain the synthetic/four-update/two-test-source scope. |
| Historical consistency effect | V5 reports training's unfavorable token effect and augmentation's uncertain predictive effect, but omits the supported augmentation improvement in token discrepancy. | Preserve the historical README result: augmented minus vanilla token error −0.00843, interval [−0.01020,−0.00687], alongside predictive ΔR² 0.00408, interval [−0.00556,0.01277]. This is a useful positive consistency result and avoids presenting the history as uniform failure. |
| Local evidence scope | §8 refers broadly to “the later prediction grids,” which can imply direct access to Notebook 12's raw grid. | Name the latest motion/region grid specifically. Notebook 12 has retained completion output and numerical summaries here, while `artifacts/comparative_masking` is absent. |

## Historical evidence from the README

The higher-precision historical values in `docs/README.md` are an additional retained documentary source. They can improve the appendix's accuracy without implying a new raw-data reconstruction:

| Historical contrast | Estimate | 95% source-bootstrap interval |
|---|---:|---:|
| Vanilla learned − initial token discrepancy | 0.03055 | [0.01576,0.04763] |
| Augmented − vanilla token discrepancy | −0.00843 | [−0.01020,−0.00687] |
| Native learned predictive utility | 0.05979 | [−0.02527,0.12571] |
| Native learned − initial prediction | −0.01798 | [−0.03851,0.00248] |
| Augmented − vanilla prediction | 0.00408 | [−0.00556,0.01277] |
| Constructed learned predictive utility | 0.04302 | [−0.04356,0.11283] |
| Constructed learned − initial prediction | −0.05874 | [−0.09549,−0.01740] |
| Native output antisymmetry error | 0.21548 | [0.19352,0.23635] |

The historical README describes a completed report and verification performed at the time. The current checkout lacks that full report chain. State both facts directly: the README preserves reported estimates and earlier verification, while this revision independently reconstructs the newest grid only. Do not convert the README's historical verification paragraph into a claim that those 100,000 original predictions were checked again today.

The augmentation finding helps the story. It shows improvement on the specified geometric score without a demonstrated predictive benefit, while trained variants still trail initialization under the token test. The constructed readout has exact output parity and an unfavorable learned-versus-initial predictive contrast. These comparisons reinforce the distinction between imposed consistency and useful representation learning.

## Figure and caption review

The three available PNGs render cleanly at full resolution and contain no obvious text overlap. The expanded pipeline clearly separates training, the EMA teacher, source-separated readout and the synthetic reflection-loss extension. It correctly notes that latest motion experiments use no reflection augmentation and that the signed target does not supervise the encoder. Its density makes it more suitable as the expanded appendix illustration once the compact main figure is present.

The results figure is the most effective main-text illustration. It displays per-seed initial/trained scores and paired mask intervals, while the correspondence panel includes the 375/375 trained versus 33/75 initial comparison and warns that the rows are repeated checks. Its interpretation matches the saved artifacts. One small improvement is to draw a single marker for the seed-independent direct-pose control; five duplicate jittered seed points add no evidence and may suggest independent baseline fits. The main text already explains control reuse.

The reflection figure identifies its skeletons and numerical target values as schematic. It usefully separates anatomical exchange, an odd readout and a constant-feature failure control. Its sentence that “a positive sign identifies greater left-side motion” could be more precise: it identifies a positive average left-minus-right median-speed contrast. Pair contrasts can have mixed signs and cancel, so the sign need not describe every limb or total physical work.

Figure 2's correspondence language should preserve the contextualized-teacher limitation. Teacher vectors come from the full clip; their matched-target advantage could depend on information already visible in context. This diagnostic does not prove recovery of withheld movement. The caption and §4.5 otherwise keep its dependency and scale limitations clear.

## Structural recommendation for an eight-page main paper

The planned v6 change is sensible: the newest directly audited experiment should occupy most of the empirical section. A compact structure can preserve the intellectual sequence without turning notebook chronology into the paper's table of contents.

1. Keep the main motivation, the coordinate-derived target and its reflection law. The clinical motivation can stay in two connected paragraphs covering different mechanisms; move the detailed condition-count table to a cohort appendix.
2. Keep the input/target distinction and source-group split in the main methods, together with compact Figure 1. Move the full tensor-shape table and detailed normalization conventions to Appendix D. State the fixed model-input shape and validity rule once in the main text.
3. Keep the completed JEPA loss, shared anatomical prior and matched controls in the main methods. Move the explicit synthetic reflection-loss equation, auxiliary forward-pass details and historical recipe constants to an appendix. A short main-text sentence and the figure can still explain precisely where the proposed loss would update the encoder.
4. Retain a brief measurement/preparation diagnostic and a short bridge from the historical reflection and target-eligibility tests to the newest question. Put most numerical 00–12 history into one evidence-status appendix table, including the positive augmentation-consistency effect.
5. Keep the latest initial/trained result table, the source-bootstrap comparison and Figure 2 in the main results. The figure already displays mask contrasts, so a second full main-text table of the same three contrasts may be unnecessary; exact estimates can remain in the caption or appendix.
6. Close the empirical arc with predictor/readout disagreement and the most consequential unresolved readout controls. Keep longer alternative-mechanism and synthetic-forecast material in Appendices B and C.

This is an eight-page intent, not a verified page count. The final rendered manuscript still needs checking with the workshop template, readable figure sizes and normal margins.

## Scientific assessment

The strongest supported claim is that the trained encoders have worse accessible laterality information under the tested summaries and ridge procedures, even when the predictor acquires sensitivity to matching clip targets. The new source intervals quantify that conditional disadvantage. Their marginal, fitted-model and exploratory status is clear. The mask contrasts continue to permit modest changes in either direction and should remain inconclusive comparisons.

The health discussion now handles important target limitations: the dataset labels do not identify affected side, shoulder landmarks do not directly measure arm swing, and average signed scores can cancel across pairs or people. The paper also avoids claiming multimodal sensing, calibrated 3D motion, clinical diagnosis or successful real-data forecasting. Those boundaries should survive the compression into v6.

The evidence does not yet distinguish the contribution of missingness, movement amplitude, temporal order, feature covariance and input-preparation mismatch. The proposed low-cost controls are appropriate. The discussion should continue to present explicit reflection training as one subsequently testable hypothesis, rather than the demonstrated remedy for the measured deficit.
