# Assessment of the original paper and revision contract

Reviewed 9 September 2026. Target: ../paper.md. Scores are reviewer judgments on a 0–5 scale, not calibrated acceptance probabilities. A score of 1 indicates a major deficiency, 3 a credible but limited workshop contribution, and 5 an unusually strong treatment. Weighted score = sum(weight × score/5). Evidence quality and presentation are scored separately so editing cannot manufacture stronger science.

| Dimension | Weight | Original score | Reason |
|:--|--:|--:|:--|
| Physical World AI relevance | 15 | 3.0 | Articulated geometry and evaluation fit; dynamics, physical intervention and multimodal evidence are absent. |
| Contribution and novelty | 15 | 2.0 | Reflection audit is useful; generic geometry/JEPA claims overlook established methods. |
| Method description and controls | 15 | 3.5 | Source grouping and paired initializations are strong; tensor preparation and actual mask denominator are underspecified. |
| Statistical inference | 15 | 2.5 | Cluster uncertainty is appropriate; “fully powered null” and categorical conclusions overstate it. |
| Evidence completeness and traceability | 15 | 2.0 | Latest completed notebooks are omitted; original report/prediction chain cannot be reproduced from the retained summary alone. |
| Interpretation and scientific restraint | 10 | 2.0 | Identity-channel failure is generalized to equivariance; self-consistency oracle is used to localize failure. |
| Clarity and narrative | 10 | 3.0 | Definitions help, but repeated slogans and defensive phrasing crowd out the intellectual sequence. |
| Figures and reproducibility communication | 5 | 2.5 | Original refers to figures without embedding them; laterality entry points and evidence status need explicit depiction. |
| **Weighted total** | **100** | **51.5/100** | Promising evaluation study requiring substantial revision. |

## Findings that materially change the paper

**1. Reframe the contribution around the latest evidence.** The current draft omits the complete anatomical, motion-weighted, and connected-region comparisons. The latest finding is that trained predictors distinguish correct from mismatched clip features while all five trained arms underperform their matched initial encoders on the tested laterality endpoint. This is a stronger empirical organizing question than a universal slogan about symmetry.

**2. Replace categorical null language with estimates and scope.** “Fully powered null” requires a justified effect size and power or equivalence analysis that the package does not supply. The recorded primary learned-minus-initial interval includes zero; that is insufficient to conclude zero benefit. Failure to meet a success gate supports a failed criterion. It does not prove the complement of the scientific hypothesis.

**3. Restrict the equivariance claim.** The token test prescribes joint exchange and identity action on feature channels. General equivariance allows another channel transformation. The text should name the tested transformation each time it interprets q. The original learned q interval also crosses the .10 margin; failing an upper-bound gate does not establish that the entire confidence interval lies above it.

**4. Correct the information-loss inference.** Reconstructing the target from its own five components checks arithmetic. It neither tests the prepared encoder input nor identifies whether preparation, encoder learning, pooling, or regularization caused poor prediction. Notebook 07's target-recomputation diagnostic is agreement between measurements, not model accuracy or an information ceiling.

**5. Be exact about preservation.** The archive has variable T×33×4 pose/visibility entries. Input preparation produces 64×33×3 coordinates and 64×33 validity, then 16×33 tokens. Short-gap interpolation, pelvis normalization and resampling change values and time discretization. Preserve provenance, anatomical correspondence, mask alignment and each clip's target association; do not claim exact physical trajectories are preserved.

**6. Explain actual masking budgets.** The base .6 fraction is applied to eligible gait tokens, with shared batch feasibility; later .5 settings also refer to a smaller pool. Neither means 60% or 50% of all body tokens. Both sides remain explicit anatomical identities. Anatomical masks are not signed laterality supervision.

**7. Separate completed data from demonstrations.** Notebook 09's explicit reflection penalty and Notebook 14's future-feature decoder have synthetic executions. Whole trajectories and temporal gaps have coverage audits but no trained real comparison. These cannot be promoted into clinical or real forecasting results.

**8. Correct evidence provenance.** The original rounded aggregate values are retained in docs/figures/v21_figure_numbers.json, but this checkout does not contain their full original report/checkpoint/prediction chain. Existing source preparation and later comparison artifacts are different evidence. State the distinction and prioritize recomputable recent results.

**9. Make the medical motivation specific and bounded.** Stroke, Parkinson's, cerebral palsy and myopathy can affect different movement features. Published clinical work motivates measurement; the local annotations do not establish affected side, subtype, severity, or a condition-specific association with this target. The target averages signed contrasts and can cancel opposing asymmetries.

**10. Situate novelty and workshop relevance honestly.** S-JEPA, MAMP and recent SLiM already address relevant representation and masking methods. Geometry-aware evaluation offers a defensible workshop contribution. A convincing claim of physical dynamics would need observable future prediction; a multimodal claim needs additional independently measured sensors.

## Seven successive revision objectives

1. Repair scope and incorporate the current evidence.
2. Make input preservation, target correspondence, split boundaries and inference explicit.
3. Strengthen the within-experiment comparisons and explain positive and negative diagnostics.
4. Ground the health motivation and novelty in primary literature; make competing hypotheses falsifiable.
5. Resolve adversarial objections about measurement, parity, adaptive development and forecasting.
6. Edit into a focused workshop manuscript with provenance-backed figures and a concrete follow-up design.
7. Address final independent review, check all claims and links, and retain a candid readiness assessment.

Each paper is saved separately. A critique of each version records addressed issues and remaining work; earlier versions are retained as an audit trail. Later scores reward clearer evidence use and better argumentation, while limitations requiring new data remain unresolved.

## Requested review capability

The named codex:adversarial-review skill was not found in the available catalog, local skills, plugin cache, or repository. The available review-agent skill addresses code regressions and is not an appropriate substitute for a manuscript review. Independent research reviewers therefore provide an explicitly labeled adversarial-review fallback. No claim is made to have executed the unavailable named skill.

## Workshop constraints checked against the current call

The workshop includes articulated geometry and evaluation protocols. Its posted format allows an eight-page long paper or four-page extended abstract, excluding references and appendices, with double-blind review. The call lists 9 September 2026 for archival papers and 29 September–29 October for non-archival papers; a deadline time zone is not supplied there. The final Markdown is a manuscript source requiring pagination in the official style before submission. [Official call](https://physworld-org.github.io/physworld.github.io/cfp/)

## Preservation record

Original paper SHA-256: C1ED8B39F242ED5CC9A6E0A9935E3A4799879E85F5DC3E4B31288C35B9C8B210.
Original tutorial SHA-256: 1944000E5D139BE64E5DB53CDB7D286EE10A006E4E53D37A5CD6D1DE2927282C.
No experiment, original paper, tutorial, or governance record is edited by this revision package.

