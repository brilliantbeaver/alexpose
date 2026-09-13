# Review, revisions and remaining risks

**Final integration: 13 September 2026.** Three research agents examined local evidence, gait models and world models independently. The work then moved through candidate selection, proposal drafting, cross-review, revisions and inspection of rendered figures. The later optical-flow request triggered a new literature search and replacement of the weakest candidate. This document records changes to the final seven, rather than treating review as an endorsement of untested results.

## What changed the recommendation

The latest source-learning-curve result remains `development_stop`. Its small positive increment is below the mismatched-skeleton control and far below the frozen target. It does not establish useful distillation, identify the cause of failure or demonstrate that a motion prior erases real events. The [evidence contract](execution-contract.md) preserves those boundaries.

Several initially attractive claims already have close precedents. CCH and ATLAS narrow the opening for predicting distillation benefit. PersonaBooth, MetaGait and STyMo narrow the opening for personal motion adaptation. Jacobian distillation and CAER narrow the opening for transferring responses. The final proposals identify the additional experiment needed to distinguish their contributions. The [literature ledger](literature.md) separates full-text reading, partial access, public file listings and checkpoints actually run. No proposed checkpoint was downloaded or executed for this portfolio.

Optical flow changed the portfolio materially. Generic pose-and-flow fusion was rejected as a novelty claim because MC-JEPA, H-MoRe, H-Flow and earlier articulated-motion work already cover much of that territory. The new P7 instead asks whether a verified limitation of joint-position input matters for forecasting, and whether a small amount of visible surface motion repairs it. It replaced the [constraint-conflict draft](reviews/rejected-constraint-conflicts.md), whose learned-model contribution was weaker than its classical geometry baseline.

The final order is **P1, P7, P2, P6, P5, P3, P4**. P7 ranks second for conceptual upside and a cheap disconfirmation test. Its natural-motion forecasting benefit is unproven. Neither the rank nor the five-point scores estimates an ICLR acceptance probability.

## Substantive revisions to every proposal

| Proposal | Objection that could invalidate the result | Revision in the final protocol |
| --- | --- | --- |
| [P1: Preserve real movement](../../../docs/studies/motion-preservation/protocol/proposal.md) | A method can preserve events by repairing less, distinguish a corruption generator, or choose a favorable test operating point. | Require exactly matched skeletal input tensors, an event-by-noise factorial with overlapping event and noise, strong flow/tracker gates, and calibration-locked repair strengths. Score preservation and achieved repair together. Add explicit uncertainty, a mixture ceiling and final projected-trajectory evaluation. Reserve a third event family after development. |
| [P2: Show competing motions](../proposals/motion-ambiguity.md) | Ordinary monocular ambiguity or failed optimization could be presented as a new discovery or proof of uniqueness. | Exclude trivial gauge changes, independently verify every witness, compare against geometric search at equal wall time, and use descriptor margins with independently bounded controls. New evidence can reject either or both candidates; search again and never infer uniqueness from failure. |
| [P3: Predict distillation value](../proposals/distillation-value.md) | An attractive teacher score can simply restate existing criteria, ignore selection cost or overfit a small teacher panel. | Predict actual student utility for target, rank and no-teacher choices. Include CCH, whitening, CCA and a short student pilot. Count extraction and selection costs, qualify the noisy best-candidate comparison, and limit two-family evidence to one directional transfer test. |
| [P4: Teach uncertain futures](../proposals/probabilistic-forecasting.md) | A branching toy or a nonsignificant mean gain could masquerade as new distributional information. | Require proper-score gains on real grouped motion and an interval-supported equivalence test for the mean claim. Specify scalar marginal scores, fixed scales and angular handling. Compare ordinary forward-KL and direct probabilistic forecasting; make no joint-distribution claim from marginal scores. |
| [P5: Transfer responses](../proposals/response-distillation.md) | Mapping every response to zero looks invariant, while a synthetic edit or future-selected pair can make evaluation easy. | Fix a training-fitted projection and scale, require nonzero held-out response variance and independent predictive value, and include existing response-transfer baselines. Match evaluation pairs using prefixes; evaluate all eligible unedited test motions as the primary endpoint. |
| [P6: Cross-activity memory](../proposals/cross-activity-prediction.md) | A personal code may encode body shape, source or recognizable style without improving an actual future. Eligible cross-activity people may be too few. | Count the complete participant/activity intersection before training. Compare the same query with matched donors, static support and equally rich raw-support models. Fix stochastic aggregation, audit pretraining overlap, and stop if the required independent people or residual predictive signal are absent. |
| [P7: Motion beyond keypoints](../proposals/motion-beyond-keypoints.md) | Ideal joint equality may fail after real conversion; synthetic twists may be trivial; flow guidance and compact-state claims are already crowded. | Check complete finite input tensors, retain aliases only as diagnostics, and require unedited-motion forecasts beyond extrapolation, extra landmarks, direct fusion and RGB mesh recovery. Use independent renderer correspondence, prefix-only extraction and common cutoff-selected points. Test H-Flow tolerances, count token bytes and total extraction cost, and qualify AMASS surfaces as model-derived references. |

The detailed reviews are [P1 and P2](reviews/adversarial-front-runners.md), [P3 through P6](reviews/adversarial-methods.md) and [the new P7](reviews/optical-flow-review.md). Earlier review wording records intermediate drafts; the linked final proposals and common execution contract govern the experiments.

## A second revision for clarity

Each proposal now opens with one measurable question and an everyday example, then explains the method and the comparison that can defeat it. The [shared glossary](execution-contract.md#a-short-reading-guide) defines technical terms before the protocols use them. Method names alone are not treated as contributions.

Several language changes prevent a reader from inferring too much. Flow is additional evidence computed from the same video, not another physical sensor. A kinematically admissible motion is not guaranteed to be physically executable. An ambiguity witness is a specific competing explanation, not a calibrated estimate of its prevalence. A held adaptation participant is not necessarily unseen during foundation pretraining. A synthetic movement edit is not a disease simulation.

The schedule now separates development decisions from untouched final evaluation. Forecasting restrictions include preprocessing, optical-flow image pairs and offline tracking, not just the network's nominal frame list. The seven budgets describe alternatives; the plan selects one flagship after two small shared pilots. The proposed stop thresholds are practical choices that must be checked against pilot variance, not power guarantees.

## Figure review and revision

There are 15 final SVGs: a ranking map and two diagrams per proposal. Reviewers inspected rendered PNGs, including both contact sheets and individual figures. They did not infer visual quality solely from SVG source. The generator also measures text bounds and detects text collisions; this is a complement to visual inspection, not an automated proof of clarity.

Scientific corrections were applied to the graphics as well as the prose:

- P1 labels observed, prior and retained trajectories; its experiment identifies matched tensors, calibration-locked repair and a development gate.
- P2 labels low/high bend conclusions, adds independently bounded controls and keeps the possibility of further explanations after a reveal.
- P3 labels target rank. P4 requires interval-supported mean equivalence. P5 requires equal-budget predictive benefit instead of an unsupported information-theoretic claim.
- P6 separates walking-derived memory from the query prefix with two inputs to the frozen forecaster. Its holdout wording refers to adaptation.
- P7 replaces an overly absolute textureless claim with ambiguity tests and explicitly labels AMASS surface targets as model-derived. The ranking map reflects the new optical-flow selection.

All figures are labeled as design schematics without experimental outcomes. The [generator](../render_figures.py), [layout checks](../figures/layout-checks.json), [initial visual review](reviews/visual-review.md), [methods visual review](reviews/visual-methods.md), [flow visual review](reviews/visual-flow.md) and [final visual check](reviews/visual-final.md) preserve the inspection trail.

## What was verified, and what remains open

The review checked saved local evidence, primary literature and release surfaces, document links, SVG structure and rendered layout. A small [rigid-chain algebra check](reviews/verify_alias_algebra.py) confirmed its stated ideal forward-kinematics construction. It did not run SMPL, inspect the actual S-JEPA input conversion, render optical flow or establish predictive usefulness. Its [output](reviews/alias_sanity_check.json) is a mathematical sanity check, not a new experimental result.

The most consequential unknowns remain whether the proposed failures occur with usable public priors, whether simple flow or geometry already solves them, and whether natural held-out motion supports a meaningful gain within the week. The first two days are designed to answer those questions. No training, HAIC job, patient evaluation or scientific success claim is part of this proposal deliverable. Machine-readable document checks are recorded in [validation.json](../validation.json).
