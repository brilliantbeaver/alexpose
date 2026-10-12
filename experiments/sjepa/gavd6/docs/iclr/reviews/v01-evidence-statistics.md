# Version 01: independent evidence and statistical review

Reviewed artifact: `docs/iclr/paper-v01.tex` and `docs/iclr/scripts/build_figures.py`. Reviewer: an independent evidence/statistics agent. This assessment uses the 69 hash-verified empirical exports and independent reanalysis recorded in `evidence/independent-evidence-verification.json`; it does not claim to rerun training.

## Evidence auditor

The main empirical numbers are accurate at their displayed precision. The population counts, seven-row results table, primary estimates, response interaction, failure decomposition, repair means, and feature-probe values agree with the retained evidence. The discussion usefully retains the zero-response benchmark, direct baseline, compound readout objective, reference-input teacher limitation, absent independent confirmation, and optimization caveats.

The central conceptual diagram contains a material error. A pure exchange of the two measured excursions reverses A, but the actual experiment reflects three-dimensional geometry and projects it through a fixed camera. That transformation need not exchange projected excursions or reverse A. For a concrete counterexample in the exported condition-person table, person rub002, oblique camera, HRNet, clear/correct naming and nonheld states has reference (qL,qR) = (60.8181,45.5285) for original and (59.7821,48.9681) for mirrored. Both A values are negative. This is a condition mean, sufficient to disprove an exact universal sign-reversal claim. The figure must distinguish algebraic bilateral exchange from actual rerendering with recomputed references.

The introduction also defines laterality as anatomical movement identity, although the operational response and geometric diagnostic do not establish which anatomical limb changed. A signed change in right-minus-left excursion can result from either side or both. The abstract's opening should describe preserving a side-sensitive measurement; any broader motivation must be clearly separated from achieved validation.

| ID | Severity | Objection and evidence | Correction required for v02 | Residual limitation | Status at v01 |
|---|---|---|---|---|---|
| E01 | Major | Figure 1 and caption make fixed-camera physical reflection imply qL/qR exchange and exact A sign reversal. Contradicted by the actual reference-condition export. | Make panel A a pure measured-side exchange illustration; say actual mirrors are independently projected and their targets recomputed. | No direct equivariance experiment is completed. | Open |
| E02 | Major | Introduction defines laterality as anatomical identity, while ΔA does not identify the changed side and geometric failure has no independent anchor. | Define the operational side-sensitive projected target and diagnostic; avoid claiming changed-side identification. | Anatomical or clinical side accuracy needs independent references. | Open |
| E03 | Minor numerical; major provenance practice | Figure-source code hardcodes core CI [-0.643,1.980], differing from retained [-0.6402124309,1.9767092658]. The promised later exact insertion is absent. | Read all plotted effects/intervals directly from comparison JSON; record hashes. At minimum use the actual exported endpoints immediately. | Summary reconstruction remains distinct from training reproduction. | Open |
| E04 | Moderate | Geometric laterality paragraph gives percentages without 4-pixel reference separation, 2-pixel named/swapped margin, or precise category rule. | Give thresholds and explain that wrong, ambiguous, and missing pairs all fail, using a different denominator from angle eligibility. | Metric is a geometric diagnostic rather than anatomical classification. | Open |
| E05 | Moderate | Seven versions currently reference common figures. Overwriting a shared corrected diagram would change rebuilt v01. | Preserve version-specific figure assets and source. | None after immutable dependencies are retained. | Open |

## Statistical reviewer

The primary uncertainty is represented honestly: all three declared primary intervals include zero, repair's person-t interval is distinguished from crossed bootstraps, and the same 14 people are not counted as three populations. The failure decomposition is additive at the original denominator and is correctly distinguished from success-conditional error. The paper does not infer chance from 50% direction, equivalence from uncertainty, or causality from the 74% and 93% arithmetic fractions.

The new laterality strata were selected after the completed development results and must be called post hoc in the results paragraph. A broad statement that secondary comparisons are unadjusted is insufficient to disclose this selection. The paper should also state the unequal intervention-level weights when explaining reproducibility: nonheld/held means need 4:1 for endpoint metrics and 2:1 for response metrics. This matters even though the numbers printed in v01 are correct.

| ID | Severity | Objection and evidence | Correction for next version | Residual limitation | Status at v01 |
|---|---|---|---|---|---|
| S01 | Major | Laterality condition results are presented without disclosing their post hoc selection. | Label them exploratory development diagnostics selected after the main experiments; preserve all declared primary contrasts. | No multiplicity adjustment or independent confirmation. | Open |
| S02 | Moderate | The summary hierarchy omits unequal held/nonheld condition weights needed to reconstruct the new tables. | State 4:1 endpoint and 2:1 nonzero response weighting, or provide an explicit referenced reproducibility table. | Direction rates require their own eligibility treatment and cannot use these weights blindly. | Open |
| S03 | Moderate | The laterality emphasis could make direct's lower assignment/response/waveform errors read as general dominance; omitted per-leg excursion errors do not support that inference. | Include a brief counterexample: direct L/R excursion errors 17.944/17.187 degrees versus delta/scalar 15.697/15.293. | No universally dominant method is established. | Open |
| S04 | Substantial empirical limitation | Fourteen reused development people, only three seeds, no confirmation, limited convergence study, and selected secondary strata limit generalization. | Keep these limits plainly stated; do not increase evaluation scores merely through polishing. | Requires new people, seeds, prespecified hypotheses, and training controls. | Bounded, not resolved |

## Fixed rubric

The weighted score is the sum of score/10 times each percentage weight. Scores assess this version, not likely conference acceptance.

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | A useful controlled structured-output evaluation; incremental representation benefit is unresolved and the scale is limited. |
| Claim accuracy and evidence support | 20% | 6.5 | Main numerical claims are correct, but the physical-mirror claim and anatomical identity framing materially overreach. |
| Evaluation and statistical rigor | 15% | 5.0 | Careful denominator/interval treatment, with substantial unrepaired adaptive-development and power limitations. |
| Scientific insight and positioning | 15% | 6.0 | Strong distinction between scalar score, waveform, and failures; laterality interpretation needs correction. |
| Reproducibility | 10% | 7.0 | Detailed saved evidence and independent hash/summary checks, but hardcoded CI endpoints and incomplete raw artifacts. |
| Clarity and narrative | 10% | 7.0 | Mostly coherent scientific argument; laterality terminology and three sequential primary questions need tighter connection. |
| Figures | 5% | 6.0 | Quantitative panels are purposeful; the main transformation diagram currently misstates the experimental geometry. |
| Submission fit | 5% | 7.0 | Uses official format and bounded claim; this review has not independently verified final rendered page count or all current submission requirements. |
| **Weighted total** | **100%** | **62.0/100** | Two major conceptual corrections are achievable by revision; evaluation limitations need new evidence. |

Required v02 priorities are E01, E02, E03, E04, and S01. No additional experiment should be implied by revising those passages. The final version can remove material misstatements while still carrying a modest evaluation score because the underlying development evidence remains the same.
