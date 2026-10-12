# Version 03: independent evidence and statistical review

Reviewed artifacts: `paper-v03.tex` and `scripts/build_figures_v03.py`, compared against the fully reviewed v02 and the original retained empirical exports. The figure builder now reads and hashes all three primary comparison JSON files. No new empirical evidence has been added.

## Evidence auditor

The added assignment counterexample is numerically correct: delta/scalar has 50.4567% failure against endpoint/scalar's 49.6879%, a 0.7688-percentage-point deterioration. Its scalar response point estimate is nevertheless lower. The per-leg caveat also matches the person table: direct/base left/right excursion errors are 17.944/17.187 degrees, versus delta/scalar 15.697/15.293. This corrects an overly broad possible reading of the direct baseline's ranking without changing any primary endpoint.

The figure code now selects the core and response crossed intervals and the repair person-t interval directly from their saved keys and records each source hash. The primary estimates and all printed tables remain correct. Previous corrections to physical-mirror geometry, side identity, full assignment definition, and version preservation remain intact.

| ID | Severity | Evidence and disposition | Remaining action or limit | Status at v03 |
|---|---|---|---|---|
| E01, E02, E04, E05 | Prior major/moderate | Correct physical projection, operational side claim, metric thresholds, and immutable figure references retained. | No independent anatomical validation or equivariance test. | Resolved in text/artifacts |
| E03 | Prior moderate | Builder reads all primary effects/intervals from comparison JSON and hashes them. | Raw experimental rerun remains unavailable. | Resolved |
| E06 | Minor | Probe section correctly says training-population, but leaves its 112-person count implicit. | In appendix name 112 training people, three probe folds, and encoder exposure to those people. | Open, bounded |
| E07 | Moderate diagram ambiguity | Model-flow B says readout fitting and deployment occur “one state at a time”; change and dense fitting require paired outputs in their losses. | Label the single-state forward path and state that training may couple two outputs while inference requires one observed state. | Open |

The appendix should supply actual support rules and constants rather than imply that reproducing aggregate tables reproduces rendered data or checkpoint inference. The current manuscript already makes that broader artifact limit clear.

## Statistical reviewer

The new paragraph explicitly calls the laterality analysis post hoc, exploratory, and unadjusted. This resolves S01. The added per-leg errors resolve S03 by making the multi-outcome nature of fidelity concrete. The paper preserves its original primary questions, reports unresolved intervals as unresolved, retains the zero-response benchmark, and does not treat a secondary endpoint repair as confirmation.

No material statistical misstatement was identified in v03. The main remaining reproducibility omission is the exact reaggregation rule for the new condition means. The final analysis uses 4:1 weights for nonheld/held endpoint metrics and 2:1 for response metrics, with person and seed pairing preserved. Direction accuracy cannot be assumed to follow the same fixed weights because its eligible denominator depends on the reference response.

| ID | Severity | Evidence and disposition | Remaining action or limit | Status at v03 |
|---|---|---|---|---|
| S01 | Prior major | Explicit post hoc development label now precedes assignment results. | New confirmation must freeze the hypothesis. | Resolved in reporting |
| S02 | Moderate reproducibility | Correct condition numbers have an independently verified executable reconstruction; weighting remains implicit in paper text. | Give endpoint 4:1 and response 2:1 weights and their state counts in the appendix. | Open |
| S03 | Prior moderate | Direct's per-leg counterexample is included accurately. | No universal best method established. | Resolved |
| S04 | Substantial empirical limitation | Same 14 development people, three seeds, adaptive stages and no confirmation. | Requires new experiments; no score gain warranted from prose. | Bounded, unchanged |

The correct “mean advantage” wording in the abstract improves precision over simply saying “improves,” given that the crossed interval includes harm. The reduced scalar weight recovers most of the point-estimate waveform improvement; this remains arithmetic rather than attribution of a unique training mechanism.

## Fixed rubric

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | Same useful controlled evaluation, without an established incremental representation benefit or external validation. |
| Claim accuracy and evidence support | 20% | 8.5 | Numerical claims accurate and post hoc/metric limitations clearer; minor diagram and probe-context ambiguity remain. |
| Evaluation and statistical rigor | 15% | 5.0 | Underlying population, seed count and adaptive-development limits unchanged. |
| Scientific insight and positioning | 15% | 6.5 | Complementary per-leg and assignment counterexamples sharpen the measurement argument; mechanism and generality remain uncertain. |
| Reproducibility | 10% | 7.5 | Primary intervals now source-driven and hashed; exact condition weighting and training-probe context still need documentation. |
| Clarity and narrative | 10% | 7.5 | Clear operational target and balanced results; laterality remains a late diagnostic and panel B is ambiguous. |
| Figures | 5% | 7.5 | Correct primary data provenance and corrected mirror example, with single-state training/inference distinction still needed. |
| Submission fit | 5% | 7.0 | Scope and template retained; independent page-layout verification remains outside this numerical review. |
| **Weighted total** | **100%** | **68.5/100** | Increase is limited to resolved reporting/provenance issues; empirical rigor is unchanged. |

Recommended v04 changes are E06, E07, and S02. A complete reproducibility appendix can close these bounded documentation gaps without changing the underlying scientific result.
