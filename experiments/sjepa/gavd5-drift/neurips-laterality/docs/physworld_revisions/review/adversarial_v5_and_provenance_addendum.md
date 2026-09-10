# Adversarial review of v5 and original-result provenance addendum

Reviewed 9 September 2026. The full `paper_v5.md`, the existing `docs/README.md`, the symmetry-loss implementation, the mean/motion feature implementation, and the reflection diagram were read. The current original-run artifact directory was checked again and still contains only cohort, inputs, splits, and the protocol snapshot.

This addendum corrects one incompleteness in the earlier core audit: the retained three-decimal figure JSON is **not the only surviving source of original-run aggregates**. The pre-existing `docs/README.md` contains five-decimal results, including a statistically distinguishable reduction in strict token error from reflection augmentation. The report/prediction/checkpoint chain remains absent in this checkout, so these are historical reported aggregates rather than results independently reproduced in this revision.

## 1. Material correction: reflection augmentation did improve the specified token score

The historical README's results table records the following comparison:

> Augmented minus vanilla token error: −0.00843 [−0.01020, −0.00687].

The source is `docs/README.md:45` as read before the root agent's planned canonical-review append. It identifies the original report row as `strict_representation_equivariance_source_bootstrap.csv`, `reflection_minus_vanilla_strict_equivariance`. The interval excludes zero. Since smaller q is better, this is a favorable effect on the **specified identity-channel token discrepancy**.

V5 reports the adverse effect of ordinary pretraining and the inconclusive predictive effect of reflection, but omits this favorable paired token effect. That omission makes the symmetry evidence unnecessarily one-sided. V6 should state the paired augmentation finding alongside the lack of established prediction gain:

“The historical report also records a reduction in strict token error from reflection augmentation, −0.00843 [−0.01020, −0.00687], while its predictive contrast remains inconclusive, +0.00408 [−0.00556, 0.01277] in R². Improved consistency under this supplied action did not establish a corresponding improvement in the movement readout.”

This does not establish intrinsic equivariance under arbitrary latent actions, successful clinical prediction, or superiority to initialization. The historical README says both trained variants remained worse than initialization on this action, and neither met the absolute acceptance criterion. The three-decimal figure JSON supports the direction of those learned-versus-initial comparisons; raw original prediction rows are unavailable for rechecking them now.

A concise mention in the abstract would improve the success/failure balance if space permits. The larger scientific point is a distinction between geometric consistency and target utility, supported by a favorable geometry-only comparison as well as unfavorable readout comparisons.

## 2. Higher-precision values recorded in the historical README

These are five-decimal **documented** values, not recovered machine-precision estimates. Do not label them full precision or pretend to recompute their intervals from the rounded figure JSON.

| Historical comparison | Estimate | 95% source-bootstrap interval |
|---|---:|---|
| Vanilla learned minus initial token error | +0.03055 | [0.01576, 0.04763] |
| Augmented minus vanilla token error | −0.00843 | [−0.01020, −0.00687] |
| Native learned predictive utility | 0.05979 | [−0.02527, 0.12571] |
| Native learned minus initial prediction | −0.01798 | [−0.03851, 0.00248] |
| Augmented minus vanilla prediction | +0.00408 | [−0.00556, 0.01277] |
| Constructed learned predictive utility | 0.04302 | [−0.04356, 0.11283] |
| Constructed learned minus initial prediction | −0.05874 | [−0.09549, −0.01740] |
| Native output antisymmetry error | 0.21548 | [0.19352, 0.23635] |

The five-decimal values agree with the rounded figure summary where both exist. The paired augmentation token-error interval cannot be inferred by subtracting the endpoints of other intervals; the historical directly reported paired comparison is the relevant source.

The README also explicitly states that the protocol was frozen internally after prior development, that no external preregistration is claimed, and that the 0.10 operational margins have no application-level calibration. Those qualifications should carry into the canonical review and final paper.

## 3. Separate historical verification from current verification

The pre-existing `docs/README.md:65` says read-only verification passed for 100,000 prediction rows, 50 jobs, and 16 lanes. That is a retained claim about an earlier manuscript revision. Its expected row count is arithmetically plausible: 625 clips × 5 seeds × 2 variants × 16 lanes = 100,000; outer-fold rotation supplies one held-out prediction per clip per seed/variant, not an additional factor of five in this total.

Current verification independently checked the retained cohort and split hashes, reproduced every split, and recomputed the input-agreement diagnostic. It **could not repeat that earlier original-run evaluation validation**, because the complete report, prediction CSVs, and checkpoints are not present. Both statements can be retained if their times and evidentiary roles are explicit. A newly added note above the old README result/verification sections should label them historical; a canonical review section can then report the current state and link the seven versions.

The README's earlier build instructions, manuscript title, YAML-header description, and eight/four-page PDF claims also describe an earlier manuscript state. They should not silently become validation claims for the seven new Markdown versions or new figures. Preserve useful history but label it accordingly.

The earlier core audit remains correct about the missing original artifacts and current verification limitations. Its statements that the only primary numeric source is the figure JSON are superseded by this addendum. The v1/v3 adversarial reviews remain records of what was known when those versions were reviewed; their chronology should not be retroactively rewritten.

## 4. Correct the displayed reflection-loss denominator

V5 displays an additive epsilon denominator and then says the implementation clamps it. The main equation should match the implementation instead of requiring the next sentence to contradict the displayed expression.

In `laterality_extensions/symmetry_learning.py`, `token_equivariance_loss` returns:

```python
(difference / energy.detach().clamp_min(1e-12)).mean()
```

Let `E_b` be the sum of squared energies of the aligned original and reflected tokens on common-valid support. The displayed denominator should be

\[
\max\{\operatorname{stopgrad}(E_b),10^{-12}\}.
\]

The code takes a per-sequence normalized residual and averages over the batch. Gradients flow through both original and reflected online-encoder outputs in the numerator; the denominator is detached. This is an additional two-pass online-encoder loss. The teacher remains outside that gradient route. V5 correctly labels the only retained efficacy demonstration as synthetic and says the loss does not exclude constant nonzero features.

## 5. Resolve the two meanings of “target” and the parity notation

The manuscript has teacher-feature prediction targets and the downstream signed target y. They enter different objectives. Its pipeline generally distinguishes them, but the parity paragraph leaves `z` undefined at the point of use. Explicitly define `z(x)` as the frozen encoder's original 960-dimensional bilateral summary in the historical parity experiment. Capital `Z` should remain the joint/time token array. The odd feature is a downstream two-pass construction; it does not redefine y and does not add an encoder training loss.

The exact readout relationship is `g(Mx) = -g(x)` for any fixed fitted weights with zero intercept and origin-preserving scaling. The target's `y(Mx) = -y(x)` is a separate algebraic fact about the observed-coordinate measurement. Satisfying the first does not show `g(x)` approximates `y(x)`.

The reflection illustration uses `g(x)=[h(x)-h(Mx)]/2`, whereas the paper's constructed feature uses division by sqrt(2). Both are valid odd projections, but they are different descriptions. Either mark the former as a general illustrative projection or show the actual zero-intercept linear readout on the sqrt(2)-scaled feature. Do not imply two differently normalized fitting procedures would produce identical ridge solutions at the same alpha. Harmonize the diagram's joint-permutation symbol `P` with the text's `S`, or explicitly state that they denote the same operation.

## 6. Remaining v5 precision issues

The central latest-grid readout and source-bootstrap descriptions now have appropriate scope. In particular, the distinction between seed variation and source uncertainty, separate random references for different mask budgets, support-feature and dimensionality changes, and adaptive reuse of the cohort are all explicit. The 960- versus 2,890-dimensional feature descriptions agree with `laterality_extensions/motion_readout.py`.

Several smaller refinements would improve the final version:

- “Coordinate depth scaled relative to crop width” is incomplete. The retained extraction stores `z = landmark.z × crop_width / image_width`; x is normalized by image width and y by image height. Appendix D should give this exact convention, since width/height anisotropy helps explain why the resulting norm is not calibrated 3D speed.
- Define the mask audit's enrichment statistic as a difference between selected and eligible token-motion scores, rather than letting its values look like percentages or physical units. Its small positive values describe the sampling manipulation, not validated biomechanical sensitivity.
- The retrospective narrative should avoid implying that the input-agreement diagnostic preceded the historical baseline training. V5 now says its question ordering is retrospective, which is an appropriate qualification.
- A constant nonzero **shared** token vector is the simple q-degeneracy example. An input-independent representation with different fixed vectors for left and right joints need not have zero q. “Can also have zero” is defensible, but “a shared constant token vector” makes the example exact.
- For the parity lane, say that the recorded learned-minus-initial comparison was adverse under that same wrapper. It does not prove no useful signal exists in either encoder or that the wrapper itself creates the predictive information.
- Main-text results and provenance should identify the original README aggregates and figure JSON as historical evidence, Notebook 12's summaries as historical context, and Notebook 18's saved prediction grid as the evidence newly recomputed in this revision.

## Recommended v6 action

Restore the favorable historical augmentation-to-q effect, fix the reflection-loss equation, define token/feature/measurement notation unambiguously, and update the original-result source note to include the README. Preserve the stronger v5 restrictions on clinical interpretation, coordinate geometry, real-data forecasting, and representation information. In the user's requested canonical `docs/README.md`, place current scores, critiques, suggestions, and remaining submission limitations in a clearly dated section, with earlier report and verification claims labeled as historical.
