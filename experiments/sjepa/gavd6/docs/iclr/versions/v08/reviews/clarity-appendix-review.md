# Appendix prose clarity pass

Date: 2026-09-26. Edited only `appendix.tex` and this review record. The parent agent edits the main manuscript separately. Editing began after the parent confirmed that the complete baseline archive was saved.

## Snapshot and scope

- Original appendix snapshot: `/tmp/gavd6-v08-appendix-before-clarity-20260926.tex`.
- Original SHA-256: `69efbc79460f9f86d5556c5f29b5cdd462bacbecaf1d60c73007094d34720696`.
- Edited SHA-256: `103f0a0374621f283e2921ef1dde1d366e74836596bb0237d9e36c5400733061`.
- Full baseline archive retained by the parent: `history/v08-before-clarity-20260926.zip`.
- Whitespace-delimited word count: 4,478 → 4,554, an increase of 1.70%. No font, margin, section, table, figure, caption, or layout command changed.

The full appendix was read before editing. The pass separates operations from assumptions and interpretation, replaces compressed terminology with short explanations, and preserves the completed study's limits. It adds no experiment, hypothesis test, result, or evidential claim.

## Changed explanation blocks

| Location | Clarity change | Fidelity boundary retained |
|---|---|---|
| Opening implementation paragraph | Identifies saved configurations and training logs as the basis for executed settings. | Current defaults and tutorial fits do not establish study results. |
| Movement preparation | Separates candidate selection, local rotation, gating/tapering, mirroring, and camera fitting. | Geometry/filename checks do not certify clinical status; nominal edits do not equal projected changes. |
| Tokens and readout | Explains safe coordinates, artificial masking, coordinate corrections, and the student/online encoder. | All slots remain; missing positions receive absolute coordinates; frozen-encoder versus joint fitting stays explicit. |
| Coordinate reduction and feature queries | Gives the averaging order step by step and explains which tokens can be queried. | Validity/support rules, endpoint weighting, supervised teacher input, and S-JEPA/DINO versus I-JEPA distinction remain. |
| VICReg and paired auxiliary support | Separates the three regularizer components and describes common token eligibility directly. | All coefficients, variance conventions, covariance reduction, singleton exclusion, pair weighting, teacher evolution, and shuffled-control limits remain. |
| Angular support and readout calibration | Explains which reference frames qualify, then separates initial magnitude matching from subsequent optimization. | Predictions cannot remove support; calibration matches low scalar rather than coordinate loss; encoder/teacher freezing remains. |
| Optimization and feature calibration | Expands hierarchical sampling, EMA, gradient energy/RMS, and before-clipping interpretation. | Optimizer settings, schedule, coefficient, parameter count, seed, update order and equal-endpoint center reduction remain exactly specified. |
| Clipping interpretation | Separates ledger counts from conclusions about convergence or each loss term's influence. | No persistent-gradient or causal explanation is introduced; missing time series/retraining/sweep remain explicit. |
| Measurement and assignment | Explains eligibility and failed predictions before listing costs, and separates assignment support from angular support. | All thresholds, failure components, sign-distribution caveat, and missing component breakdown remain. |
| Population averaging and intervals | Explains each averaging level, why exported groups need unequal weights, and how crossed resampling differs from the repair interval. | The 14 people and three fitted seeds remain the units; paired methods, conditional person-t scope, and exploratory interval limits remain. |
| Provenance and selected condition groups | Replaces compressed terms such as “original estimand” with the population question being averaged. | Adaptive development reuse, no independent preregistration, fixed primaries, unadjusted analyses, and no confirmation remain explicit. |
| Response-magnitude limitation | Explains cancellation of opposite signs before describing what the aggregated CSV cannot recover. | No per-pair response bins/counts or individual distribution is inferred from averages. |
| Failure accounting | Defines success contribution over the whole weighted population, conditioning on success, and the effect on person weights. | The two conditional aggregation orders remain distinct; neither conditional error is added to unconditional failure cost. |
| Failure-cost contrast and sensitivity | Explains that 74.17% is a share of a mean contrast, while the small percentages are failure rates. | All numbers and costs remain exact; zero cost retains failures in the population and is not success-only evaluation. |
| Repair interpretation | Separates experiment setup, mean ratios, paired effects, seed variation, and noninferiority. | The delta primary is unchanged; endpoint secondary results do not replace it; waveform improvement does not establish preserved response accuracy. |
| Reproduction and notebook validation | Distinguishes rebuilding summaries, rerunning the study, matching notebook calculations to source, and executing every notebook. | Missing assets/checkpoints, interrupted full fixture run, incomplete remote notebook validation, and the separate source-run receipts remain. |
| Data validation and leakage | Replaces “lazy view” with loading training arrays on demand and explains what development-reference validation may read. | Development targets do not enter fitting/calibration losses; unknown aliases and upstream training overlap remain unverified. |
| Final population/inference paragraph | Explains unequal record counts and adaptive experiment choice directly. | Held conditions remain evaluations within the reused development cohort, not another independent confirmation cohort. |

## Fidelity checks

A mechanical comparison against the snapshot passed for all of the following, preserving both content and order:

- All 322 numeric tokens.
- All 152 inline-math spans.
- All nine display-math blocks.
- All 12 complete table/figure blocks, including captions.
- All 47 section/subsection, label, reference, and citation commands.

The semantic review checked support definitions, averaging denominators, optimizer/calibration scope, source versus notebook evidence, and adaptive-development limits. No method or conclusion was intentionally changed. The readout wording now expressly identifies the frozen object as the encoder, avoiding the shorthand “frozen readout” for a readout that is itself trained.

No training, notebook execution, new statistical analysis, or PDF build was performed during this pass. The parent should compile and check pagination using the simultaneous main-text edits; unchanged TeX structure and a 1.70% length increase limit but do not prove layout preservation.

## Independent consistency check of the rewritten main text

Read the complete rewritten `paper-v08.tex` against the appendix and the previously audited implementation. Main-source SHA-256 at this check: `477b2088350464b2db7836e512d36c968e024240c697ccbe10fb281fac3b52de`. The appendix still matches the edited hash recorded above.

The concrete abstract accurately defines the outcome as the between-motion change in bilateral knee excursion, reports the zero baseline and its paired interval with the correct sign, and leaves the delta-versus-endpoint contrast unresolved. It explicitly identifies synthetic references and complete-window restoration. Future world models appear as motivation; the introduction states that future-state prediction is outside the completed evaluation.

The methods preserve the percentile response definition, physical-mirror versus naming distinction, shared base plus endpoint/delta auxiliaries, pair-equal support reduction, compound original change objective, and dense-to-low-scalar calibration. Training/development separation, permitted validation reads, adaptive reuse, stage primaries, retrospective nulls, and the different crossed versus person-t intervals remain consistent with the appendix. Failure contributions and conditional success error keep their distinct denominators. The repair result is not promoted to demonstrated preservation of response accuracy, and neither initial gradient imbalance nor score decomposition is made causal.

Two minor precision clarifications were sent to the parent and are now resolved:

1. The sentence “Dense supervision changes which timestamps receive gradients” needed to identify the **dense angular term**. Both readout arms also have coordinate supervision across timestamps; the intended contrast concerns where the additional angular loss acts. The source now reads: “The dense angular term changes where its gradients act and what they target.”
2. The assignment definition now uses “summed Euclidean distances,” retaining the exact metric underlying the two-pixel margin.

These are wording clarifications, not changes to the results or a request for new experiments. No material mathematical or statistical discrepancy remains from the main-text simplification. No main-source edit was made by this reviewer.

## Final resolution and consistency check

The final check verified both corrections, the explicit definition of frozen encoders as held fixed during readout fitting, and the shortened introduction/discussion. The description of privileged reference supervision, full-window restoration, adaptive development reuse, unresolved primary contrasts, and absence of future-state or clinical validation remains intact. The appendix's change from “source package” to “version directory” accurately limits where the executable audit is supplied; no technical specification changed.

Final checked source hashes:

| Source | SHA-256 |
|---|---|
| `paper-v08.tex` | `00f2570880b6d8b76c03bd7b074f43b170280794fcabcaa477bf2f7e79b6f975` |
| `appendix.tex` | `13b06b4f8faadfdc362e624b49cb56356ad7288acd3fd86333290f4972408b12` |

Against the original clarity snapshot, the appendix still preserves all 322 numeric tokens, 152 inline-math spans, nine display equations, 12 table/figure blocks, and 47 section/label/reference/citation commands exactly and in order. Against the archived main-source baseline, all three display equations, both tables, and all five figure commands/captions are unchanged. Every original inline-math span is retained; the eight additional spans explain existing symbols, the zero predictor, and the person-t interval. A response-loss expression moved within its paragraph without changing. Main prose adds the explanatory percentile numbers and an explicit zero prediction, and removes duplicate or redundant numerical wording; no reported result or specification changed.

**Closed:** both minor wording findings are resolved. No open mathematical, statistical, or methodological consistency item remains in this review. This conclusion concerns the checked sources; it does not claim a new experiment or a PDF-layout inspection.
