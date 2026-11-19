<!-- iteration02-execution-status -->
> This is the preserved design or feasibility record. Consult the [implemented protocol](protocol.md), [verified results](results.md), and [external execution report](external_replication_results.md) for current execution status.

# Independent adversarial review of the next iteration

This review covers the research design in [05-iterate-02-INSTR.md](../../../../notes/aaai/05-iterate-02-INSTR.md), the completed draft01 results, and the relevant implementation defaults in `aaai_gait/context_prediction.py`. It evaluates a proposed experiment. It does not certify new training, notebook execution, external validation or improved performance.

The design is substantially stronger than repeating the earlier JEPA experiment. Its main question is now identifiable at the level of a fitted predictor: does visible opposite-leg context improve a fixed observable prediction beyond a competent own-leg predictor and an own-leg capacity control? A second experiment asks whether that predictor uses the temporal alignment of the additional context. These are useful and falsifiable questions. Neither is a direct measurement of neurological coupling or a disease cause.

There is no remaining conceptual blocker to implementing this bounded exploratory study. The specification corrections below are necessary parts of that conclusion. The coordinator accepted the final precision items on the timing estimand, source-vector aggregation, joint-specific aggregation and mask seed. These must remain in the implementation and protocol record.

## Consequential findings and corrections

| Severity | Problem identified during review | Required correction and present disposition |
| --- | --- | --- |
| High | Two independently fitted predictors can differ because one is poorly optimized. Earlier full-context errors were worse on normal development data. | Preserve the competent coordinate baseline `f`, compare `f+h(O)` with `f+g(O,C)`, and gate interpretation on source-grouped normal development. The instructions now require fixtures, headroom checks and a local fallback. |
| High | Zero-initializing `g` preserves `f`, but does not guarantee that it matches the separately fitted `f+h` on unseen sources. | State precisely what is nested, test zero-gate identity, and make no held-out dominance claim. The instructions now distinguish these properties and compare against `h` directly. |
| High | Additional parameters can masquerade as information from the opposite leg. Matching input dimensions is insufficient. | Retain an own-capacity correction, report effective feature rank and inductive bias, and include a calibrated base-to-own sham score. The instructions now require these controls. |
| High | A shifted pelvis-centered trajectory can import a shifted shared reference. Computing a limb vector from a hidden hip leaks information. | Use permitted donor joint-minus-hip vectors, the original scale and original permitted attachment hip. Hold the target, own leg and shared reference fixed. Restrict access before deriving features. The instructions now enforce this distinction. |
| High | Selecting architecture, timing offsets or fusion using the five reference normals would contaminate calibration. New folds also cannot erase previous exploration. | Choose only within fitting-normal grouped folds, reserve reference sources, refit comparators under the new roles, and call the redesigned study exploratory. All 28 candidate normals were previously explored. These boundaries are explicit. |
| High | Pooled out-of-fold rankings can create a method gain from fold scale differences, especially when relational learning fails in some folds. | Make the primary AUROC summary a weighted within-fold mean. Bootstrap within fold and binary-label stratum, retaining paired method scores and duplicate groups. Set an unavailable relational channel to zero, keep the fold, and retain local-motion ranking. These corrections are incorporated. |
| Medium | Intersecting original, donor and attachment visibility can leave very little opposite-leg evidence, even when the nominal window is long enough. | Recheck adequacy and context counts on the exact paired-support original inputs. A sparse-context null is uninformative, not evidence of timing independence. The instructions now include this check. |
| Medium | “Uniform masking” alone does not specify the estimand or eligibility. | Retain four-frame joint patches, 128 tokens per 32-frame window, a 0.5 mask ratio, 16 evaluation draws and the recorded seed. Use the corresponding 192-token parent grid for timing controls. These choices are incorporated, with seed 17 accepted as the retained default. |
| Medium | A change in a two-sided calibrated anomaly score is not the same as a change in prediction error. | Define the primary timing effect as shifted-context loss minus aligned-context loss, with fixed target and paired support, then aggregate within sources. Treat changes in R or S as secondary. The coordinator accepted this explicit formula. |
| Medium | Distance before aggregation and distance after aggregation are different detectors. Averaging all targets can also give prolific joints more weight. | For M, first obtain one descriptor vector per source, then fit and apply source-level imputation, scaling and covariance. For R, retain the joint axis through aggregation, then average four joint-specific source benefits per side. Both corrections were accepted. |

No correction requires changing a completed result or pretending that a new method has already succeeded. Preserve the original evidence and create a separate protocol identity.

## Scientific risks that remain after correcting the specification

**The task can be too easy locally.** A knee, heel and foot at the same time can nearly determine an ankle through local geometry. A small contralateral gain can therefore reflect a completion ceiling. The separately declared temporal-gap evaluation is a useful diagnostic, but it must not silently replace uniform-mask evaluation after an unfavorable result.

**The alignment edit is a diagnostic, not a patient simulation.** Donor limb vectors preserve observed internal geometry before attachment. The resulting combination need not be dynamically feasible, and finite crops do not exactly preserve marginal distributions. A whole-body temporal crop changes the target and cannot serve as an identical fixed-answer intervention. The revised instructions correctly distinguish these operations.

**Five reference normals do not provide stable clinical calibration.** Shrinkage and a fixed variance floor make a score computable, not reliably calibrated. Reference sensitivity, constituent errors and per-source values remain essential. A good AUROC cannot establish a 5% false-alarm guarantee or a deployment threshold.

**A conditional-prediction detector can miss regular abnormal movement.** Small steps, bilateral crouch or repeatable compensation can remain predictable. Keeping posture and excursion in M is therefore scientifically necessary. It also creates a demanding comparison: R must add value beyond M, its richer 36-feature counterpart, a sham benefit, constituent-error calibration and simple timing descriptors.

**Source diversity is not independent clinical validation.** The candidate 304-source population improves observed coverage, while the focal 90-source endpoint matches the requested conditions. Source IDs still denote recordings, normal data remain scarce and explored, and viewpoint, activity, assistance and person overlap remain potential explanations. Folder labels cannot identify neural mechanisms. The proposed external replication could strengthen the study, but it must retain participant separation and its own measurement semantics.

## Novelty and testability assessment

Normal-only gait prediction and reconstruction are established. [FSGait](https://openaccess.thecvf.com/content/ACCV2024/html/Duan_FSGait_Fine_Grained_Self-Supervised_Gait_Abnormality_Detection_ACCV_2024_paper.html) already separates posture and temporal modeling. [GenGait](https://arxiv.org/abs/2604.01997v2) uses normative masked reconstruction and joint inconsistency. [SeeKer](https://arxiv.org/abs/2506.18368) models conditional keypoint densities. [Beyond Marginals](https://arxiv.org/abs/2509.15033) distinguishes marginal, temporal and cross-variable anomaly structure. A residual connection, two score channels or a coordination diagram is therefore insufficient novelty.

The plausible contribution is a tested measurement procedure under incomplete skeletal observations: a fixed coordinate target, a retained competent baseline, an own-capacity comparison, explicit permitted inputs, and a timing diagnostic with source-level evidence. It becomes a stronger paper if these choices change the scientific conclusion or improve real-source detection beyond credible alternatives. It should not be called the first such method without a broader priority review.

The external dataset is a concrete opportunity rather than a promised result. [Van Criekinge and colleagues](https://www.nature.com/articles/s41597-023-02767-y) provide simultaneous biomechanical measurements from 138 able-bodied adults and 50 stroke survivors. A participant-separated replication can test the algorithmic contrast in a different measurement system. It cannot automatically validate transfer of video weights, BlazePose geometry or all four condition families.

## Conditions for the final evidence review

Before writing a positive method claim, verify the following outcomes from saved real-data artifacts:

1. The estimator recovers a known added-context dependency and avoids systematic benefit under a conditional-independence fixture, using separate generated training and test realizations.
2. Hidden-input changes cannot affect restricted predictions. Zero correction and zero offset satisfy their exact software identities.
3. The both-leg predictor improves held-out normal error beyond both the local baseline and the own-capacity control. Report failures and fallback folds rather than excluding them.
4. Timing responses use identical targets, fixed references and explicit common support. Report marginal drift, actual available context and the paired-support adequacy result.
5. S-versus-M is evaluated with the frozen within-fold estimand. The richer motion baseline, sham score, constituent-error fusion and simple timing comparators are reported even when unfavorable.
6. Focal and broad endpoints, source counts, reference sensitivity, coverage and limitations remain visible. The new endpoint is not presented as a confirmatory test following the old results.

If the predictor is adequate but R adds little detection value, that is an informative measurement result. If adequacy or timing support fails, the defensible contribution is a narrower audit or negative result. A larger encoder, a favorable seed or a different headline subset would not repair those failures.
