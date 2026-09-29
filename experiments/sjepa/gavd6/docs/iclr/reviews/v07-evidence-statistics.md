# Version 07: final independent evidence and statistical review

Reviewed artifact: `paper-v07.tex`, SHA-256 `eb05eea1a00f6a1bdf4341916187eecf2f10485a017471102ffd2d40cf3bd6be`, with all changes from the independently reviewed v06 and the retained empirical evidence. The final check reverified all 69 copied evidence-file hashes and binds all five used figure PDFs in [the final verification record](../evidence/v07-final-evidence-check.json). The parent build verified nine main pages. This is an independent evidence/statistics review of the exported development results, not a training rerun or independent empirical validation.

## Evidence auditor

**No unresolved material evidence misstatement was found within the paper's stated scope.** The manuscript accurately distinguishes projected measurements from anatomical or clinical validity, reference-paired feature prediction from future world-model prediction, physical mirroring from input renaming, scalar response from side identity, shared coefficients from matched gradient influence, and descriptive results from confirmation.

The final calibration caveat is correct. From `outputs/iclr/jepa-response/diagnostics/loss-calibration.json`, source SHA-256 `3c00e2931bfa3a8cc7e4d94616774b81a066e60a0aabc3aa22a303a0266620d5`, the shared coefficient is 0.012646811176583523. Multiplying the retained component gradient RMS values by this coefficient yields endpoint/base 0.10000000000000002 and delta/base 0.000013037139634433077. These are **10% and 0.00130371396%**, respectively, so the paper's 10% and 0.0013% are correct. Independent recomputation from sums of squared gradient norms gives the same ratios. The paper confines the statement to initialization and does not claim the delta contribution remains small or caused the outcome. This resolves E11.

The revised abstract confines the 93% recovery statement to the delta feature-difference repair setting on ViTPose, where the independently calculated fraction is 92.98697%. It calls the laterality discrepancy post hoc and reports higher **mean** assignment failure, preserving the contrast's selected development status. The introduction now identifies the declared feature-response primary as the original change-supervised readout, with coordinate-only readouts secondary. These refinements improve precision without adding experimental support.

| Past objection | Severity when raised | Final correction or disposition | Residual limitation |
|---|---|---|---|
| E01: mirror implies projected sign reversal | Major | Figure/caption distinguish pure bilateral measurement exchange from separately reprojected 3D mirrors. | No direct equivariance test. |
| E02: scalar identifies changed anatomical side | Major | Signed target is explicitly insufficient to identify which leg changed; geometric diagnosis is bounded. | Anatomical side validation requires independent references. |
| E03: inaccurate/hardcoded plotted primary interval | Numerical minor; provenance moderate | Exact source JSON supplies effects/intervals, with hashes and correct inferential types. | No raw prediction or training reconstruction. |
| E04: assignment definition/denominator missing | Moderate | Four-pixel eligibility, two-pixel margin, and wrong/ambiguous/missing categories are specified. | No calibrated chance level or clinical diagnosis. |
| E05: shared mutable figure dependencies | Moderate | Earlier assets are preserved; final figure hashes are recorded. | Rebuilding must preserve those dependencies. |
| E06: probe population ambiguity | Minor | Correct 112-training-person diagnostic context remains in the audit; main text no longer makes a probe-based mechanism claim. | A fixed probe cannot establish all information availability. |
| E07: single-state forward path confused with paired fitting | Moderate | Figure distinguishes single-state computation, paired training losses, and single-state inference. | None within the depicted scope. |
| E08-E10: weak laterality visual focus and ambiguous labels | Editorial/minor | Verified naming plot, lower-score heading, and frozen-variant caption implemented. | Figure remains descriptive, post hoc evidence. |
| E11: shared coefficient suggests equal gradient influence | Moderate | Initial 10% versus 0.0013% difference is now explicit and temporally bounded. | Later influence and causal effects remain untested. |

All material objections to accuracy or reporting are addressed. The residual limitations are scientifically consequential and remain visible rather than being described as solved.

## Statistical reviewer

**No unresolved material statistical misstatement was found.** All three declared primary intervals include zero and are described as unresolved. The repair person-t interval is distinguished from crossed person/seed bootstrap intervals; means preserve the window/motion/person hierarchy; original primary contrasts are not replaced by selected favorable comparisons; and the same 14 development people are not counted as independent replications. The final caption now names each candidate, comparator, readout, endpoint, and extractor scope, preventing the three primary-effect rows from being read as a meta-analysis.

The independent final sweep confirms the eight waveform deteriorations, all core person- and seed-mean waveform deteriorations, direct response gains for 10 of 14 people, direct waveform/coordinate gains for all 14, zero-response superiority over all 16 learned variants under pooled response MAE, the 74.16509% failure fraction, and the 92.98697% low-scalar recovery fraction. The delta excess assignment interval is correctly transformed to 0.77 [0.17,1.53] percentage points and remains explicitly exploratory and unadjusted. The per-leg counterexample prevents a universal direct-method ranking.

| Past statistical objection | Final status | Remaining empirical requirement |
|---|---|---|
| S01: laterality selection was not disclosed | Resolved by explicit post hoc/exploratory labeling. | Freeze the laterality hypothesis before new confirmation. |
| S02: condition weighting was not reproducible | Resolved by 4:1 endpoint and 2:1 response weighting with executable audit. | Direction eligibility must remain separately handled. |
| S03: favorable direct ranking omitted a counterexample | Resolved by per-leg excursion results and failure costs. | Multi-objective utility needs a justified application-specific criterion. |
| S04: small adaptive development panel and limited controls | Bounded, not resolved. | More independent people and fitted seeds; separate confirmation; external trained baselines; optimization/budget studies. |
| S05: signed assignment interval could be inverted incorrectly | Passed independent negation, endpoint ordering, and percentage-point conversion. | No confirmatory claim is implied. |
| S06: abstract's aggregate statements needed a complete sweep | Passed checks against all available families and person/seed panels. | Export verification remains distinct from raw source validation. |

The final text appropriately leaves convergence, generalization, anatomical accuracy, and a uniquely identified feature-coupling or sparse-gradient mechanism unresolved. These limits justify maintaining a modest evaluation score even after reporting defects are corrected.

## Fixed rubric

Scores remain the same as v06. The final clarification closes an interpretive ambiguity, but does not cross a further scoring threshold or add data. A higher version number is not a reason for a higher score.

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | A useful bounded evaluation, with no established incremental feature-prediction advantage or external validation. |
| Claim accuracy and evidence support | 20% | 9.0 | No material misstatement found; exact exported evidence and scope support the claims, while upstream raw validation is unavailable. |
| Evaluation and statistical rigor | 15% | 5.0 | Fourteen reused people, three seeds, post hoc diagnostics, no confirmation, and limited external/optimization controls remain substantial weaknesses. |
| Scientific insight and positioning | 15% | 7.0 | Clear scalar/trajectory/naming distinction and gradient-calibration limitation; mechanisms and generality remain uncertain. |
| Reproducibility | 10% | 8.0 | Source hashes, editable figures, exact reductions, calibrations and interval procedures are retained; compact packet omits full raw assets. |
| Clarity and narrative | 10% | 8.5 | Laterality focus is coherent and primary/post hoc questions are separated; dense technical scope still demands careful reading. |
| Figures | 5% | 8.5 | Purposeful diagrams and verified vector plots with consistent units/encodings; no real reconstruction panel can be verified from local artifacts. |
| Submission fit | 5% | 8.0 | Official format and nine-page main text reported by build validation; no acceptance prediction is made. |
| **Weighted total** | **100%** | **72.75/100** | Same empirical and scoring scope as v06; no automatic final-version uplift. |

The final manuscript is acceptable as an accurately bounded development-study draft. Establishing the hoped-for method benefit, anatomical laterality recovery, external validity, or a causal optimization explanation requires new evidence rather than further wording changes.
