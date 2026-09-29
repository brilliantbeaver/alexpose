# Independent adversarial review: methods and scientific narrative

Reviewed 25 September 2026. Scope: both new Markdown draft sources, checked against preparation, model, objective and evaluator code and the completed experiment interpretation. This is an independent read-only review of the drafts; the reviewer did not author or edit their prose. Equation and visual placeholders were still being built, so this review is not final rendered-layout approval.

## Overall judgment

The main claim is supported and appropriately bounded: restoration must be evaluated as movement measurement; direct coordinate training improves the source estimates; the original change objective damages trajectories across tested variants; a low-weight scalar control explains most mean repair; the intended JEPA advantage remains unestablished. The narrative successfully separates the completed three-stage study from planned confirmation. It handles the strongest negative evidence—the zero-response baseline, measurement penalties and teacher/encoder diagnostic distinction—rather than burying it.

No claim-reversing or severe scientific error remains after reconciling the family-count objection below. Several moderate changes will make the methods sufficiently explicit and prevent easy reviewer misunderstandings.

## Actionable moderate findings

1. **Explain the effect of physical mirroring on the edited anatomical side.** Full Data provenance and brief Data collection currently say right-knee edits are crossed with physical mirroring. `preparation.py:118–132` reflects the geometry and reorders joints with `BODY_SWAP`. The right-knee intervention is applied before that step, so the mirrored condition exchanges the anatomical side. Suggested full wording: “The unmirrored motion receives a right-knee edit; mirrored versions reflect the motion and exchange anatomical sides.” This also explains the scientific purpose of the mirror condition. A compact brief alternative is “knee edits with side-exchanged mirror variants.”

2. **Define the feature residual rather than referring to an undeclared transformation.** Full §4 describes `e_a,e_b` as errors “after the declared centering and temperature scaling,” but neither is declared in the new document. These choices alter the loss materially, and I-JEPA is not a citation for this local centered-cross-entropy implementation. A compact methods note or appendix should define `H(v)=v−D⁻¹Σ_d v_d`, `e_i=H[p_i/τ_s−(t_i−c)/τ_t]`, `τ_s=.1`, `τ_t=.06`, and the detached teacher/center. State that the original objective compares softmax feature distributions by cross-entropy and includes translated-view consistency/variance/covariance regularization. Explain centering as removing the mean across channels, and temperature as fixed numerical scaling. This can stay out of the brief.

3. **Make the control logic explicit.** Both drafts list initialized and shuffled-reference encoders but give little explanation of the latter. The full should state that initialized features hold the encoder at random initialization while training the same readout; shuffled-reference pretraining replaces the matching reference with a different window from the same person while preserving endpoint role and nuisance conditions. These test readout/input-path capacity and the value of the intended motion correspondence. In the brief, one short clause—“testing untrained-feature capacity and the value of correct reference pairing”—is enough. Source: `training.py:192–204` and fresh-readout setup.

4. **Fix the brief's metric definition and baseline.** Brief Methodology says waveform error is “the mean error”; it must say **mean absolute error**, since signed error can cancel. Brief Results should identify the 12.69° and18.57° starting values as **unchanged estimated poses**. The current reduction numbers are correct but require the reader to infer their comparator. The brief should also explain failure in a few words: nonfinite coordinates or degenerate limb segments can make an angular measurement unusable.

5. **Finish first-use explanations in the brief.** “Frozen encoder” means its parameters are held fixed while the readout learns; “endpoint” means a complete original or edited movement state, not a frame; “three fitted seeds” is less clear than “three random training seeds.” These can be explained through small replacements rather than extra paragraphs. The numerical example and sample definition otherwise satisfy the first-principles requirement well.

6. **Qualify the clipping statement to the audited fits.** Full Discussion currently says “JEPA gradients were clipped on nearly every pretraining update.” The verified result is the **six new JEPA pretraining fits in the response follow-up**, each clipped on1,999 or2,000 of2,000 updates. The available evidence should not imply that every historical/core JEPA run was audited identically. This is an optimization limitation, not an established cause of the result.

7. **Use the exact core method label.** Full Results says “Ordinary JEPA illustrates the tradeoff.” `ordinary_jepa` exists as a distinct implementation elsewhere but was not the core method here. Use **core reference-paired JEPA** or **core JEPA**. This was independently identified by the evidence-review agent.

8. **Keep the original loss-package attribution consistent in the abstract and brief.** Full Results correctly states that adding the original change objective also adds a geometry term. Abstract and brief say “scalar change objective,” which could imply that the core isolates the scalar term alone. Prefer “original change-supervised objective”; then let the full methods explain the scalar-plus-geometry package and the repair's matched geometry term. The repair is the evidence that specifically makes scalar weight informative.

## Family-count objection and reconciliation

Initial review challenged the brief's phrase “all eight tested model families” because the core contains five. This objection was sent immediately and then reconciled against the independent evidence audit: the three follow-up representation variants also show waveform worsening under the matched original-change readout:

- Delta JEPA:17.37697° →21.76481°.
- Endpoint JEPA:16.89883° →22.03110°.
- Coordinate-delta:16.22093° →21.08487°.

Therefore **eight is numerically supported across core plus follow-up**. It should not be reported as a factual error. The clearest wording is “all five core families and all three follow-up variants,” avoiding an implication of eight distinct architectures. The full may retain the narrower five-core statement if the brief labels its broader scope explicitly.

## Optional narrative improvements

- The full cohort table's planned confirmation row and its “later slim plan” explanation interrupt the completed-evidence story and may suggest a third observed population. Consider moving planned counts into a reproducibility appendix, retaining a short statement that confirmation has no completed result. This is a readability recommendation, not a correctness requirement: the existing caption does disclose candidate status.
- Define GAVD's role with “no training or evaluation contribution to these runs,” rather than only “contributes no evaluation,” to remove any implication of unsaid pretraining on real video.
- The claim that plausible poses can alter movement is motivating reasoning, not evidence of visual realism in these outputs. Preserve the present conditional wording; the transferred packet does not include raw reconstructions on which to base a perceptual-quality claim.

## Confirmed strengths and limits worth preserving

- The illustrative excursion example is numerically correct and explicitly distinguished from observed data.
- The method measures image-plane angles and controlled kinematic changes, with no clinical or anatomical3D claim.
- Paired states are correctly distinguished from adjacent video frames and temporal derivatives.
- Teacher targets use privileged synthetic reference poses. The drafts do not present this as label-free learning solely from observed video or as future-motion world modeling.
- The readout is separately fitted with a frozen encoder, and direct training is not described as equally optimized merely because update totals match.
- Primary contrasts are separated by metric and estimator population; the repair's person t interval is distinguished from crossed bootstrap intervals.
- No interval spanning zero is treated as equivalence.
- Failure contributions use the full denominator and are not confused with conditional means of successful predictions.
- The zero-response baseline is accompanied by conditional direct-model sensitivity, avoiding a false conclusion that all models learned nothing.
- Secondary endpoint-repair evidence is not substituted for the uncertain delta primary.
- The data-example limitation is explicit. The participant-aggregate plot is actual exported measurement data, but it cannot meet the ideal of a verified raw pose/frame restoration sequence. This remains an evidence gap, not something to repair with an unlabeled illustrative image.

## Pending validation outside this review

The final equation SVGs must match the documented formulas, including averaging over both legs and supported timestamps in the dense loss. The rendered brief must stay near two pages without shrinking text excessively. Figure values, labels, captions and source-population conditions require final visual and numerical checks after the build; image filenames were still placeholders during this review.
