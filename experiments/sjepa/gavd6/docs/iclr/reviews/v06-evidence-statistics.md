# Version 06: independent evidence and statistical review

Reviewed artifact: `paper-v06.tex`, with comparison against v05, all retained primary result files, the new loss-calibration description, and a fresh numerical sweep. The sweep is recorded in `evidence/v06-numerical-check.json` and binds this manuscript's SHA-256. The parent reports nine main pages. No new experiment is counted.

## Evidence auditor

The abstract now foregrounds the laterality discrepancy without overstating it: the response advantage is a mean with an interval spanning zero, and assignment is explicitly post hoc. The exact assignment means, 50.4567% and 49.6879%, round to the stated 50.46% and 49.69%. The final abstract sentence leaves the declared primary comparisons unresolved, so the new emphasis does not convert a selected diagnostic into a confirmatory success.

The formerly ambiguous subsection heading and “both JEPA readouts” label are corrected. The feature-loss reduction now matches the implementation: both endpoints must query the joint/time token and have complete reference support for its four frames; token errors are averaged within supported pairs and pairs receive equal weight. The calibration equation matches the saved receipt and uses the same coefficient across seeds.

The exact calibration record reveals an important remaining interpretation detail. The common coefficient 0.012646811 makes the larger endpoint auxiliary's initial gradient RMS 10% of base, but the delta auxiliary's weighted initial RMS is only about 0.0013% of base. The base, delta, and endpoint unweighted RMS values are 0.06333581, 0.0000652906, and 0.50080459. This follows directly from the stated maximum-based formula, so v06 does not contain a false equation; however, a reader might interpret “matched endpoint supervision” as matched gradient influence. A concise explicit caveat is warranted. This is an initialization observation only, and neither persistence through training nor a causal explanation of performance follows. The independent calculation is saved in `evidence/calibration-gradient-audit.json`.

| ID | Severity | Evidence and disposition | Residual limitation | Status at v06 |
|---|---|---|---|---|
| E01-E08 | Earlier substantive/documentation objections | Corrected geometry, side scope, denominators, figure provenance, training/inference distinction, and naming visualization retained. | External validation and raw rerun unavailable. | Resolved/bounded |
| E09/E10 | Earlier minor wording | Heading now refers to a lower response score; naming caption identifies frozen JEPA variants. | None within the wording scope. | Resolved |
| E11 | Moderate interpretation | Shared coefficient calibrates the largest auxiliary to 10% of base; delta's initial contribution is much smaller. | State this asymmetry explicitly and keep it limited to initialization. Its performance consequences need further optimization evidence. | Open suggested clarification; formula itself correct |

## Statistical reviewer

The fresh sweep independently checks all eight mean waveform increases: 7.2076, 5.5248, 5.1001, 4.9881, 4.6321, 4.8639, 5.1323, and 4.3878 degrees in the order direct, coordinate, core JEPA, initialized, shuffled, coordinate delta, endpoint, and delta. It also verifies the stronger stated core pattern: each family's person-averaged increase is positive for all 14 people and each seed-averaged increase is positive for all three seeds. These are correlated descriptions, and the manuscript says so.

The sweep confirms direct's response improvement for 10 of 14 people and waveform/coordinate improvement for all 14, all 16 learned response scores above zero prediction, and the two arithmetic proportions: 92.98697% low-scalar recovery and 74.16509% failure contribution. All printed values and ranges agree at their displayed precision. The response, waveform and assignment intervals retain their different metric units and inferential roles.

| ID | Severity | Disposition | Residual limitation | Status at v06 |
|---|---|---|---|---|
| S01-S03/S05 | Earlier reporting and sign checks | Post hoc status, level weights, complementary metrics, and assignment-CI transformation remain correct. | Selected secondary evidence remains unadjusted. | Resolved |
| S04 | Substantial empirical limitation | Same reused 14 people, three seeds, limited optimization study, no independent confirmation or external trained baselines. | New experiments required. | Bounded, unchanged |
| S06 | Fresh verification | New sweep confirms every numerical pattern emphasized in abstract and results. | Checks exported statistics, not raw predictions or source training. | Passed |

The phrase “across eight model families” remains descriptive within the three sequential experiments and should always be read with the nearby warning that these are not eight replications. The new abstract retains the 14-person, three-seed development context, so its scope is adequate.

## Fixed rubric

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | Same bounded evaluation contribution and unresolved incremental representation benefit. |
| Claim accuracy and evidence support | 20% | 9.0 | Final numerical sweep passes; calibration influence needs one additional explicit qualification. |
| Evaluation and statistical rigor | 15% | 5.0 | Data, seeds, adaptive development, and baseline limitations are unchanged. |
| Scientific insight and positioning | 15% | 7.0 | Laterality/response discrepancy is explicit and carefully bounded; calibration and general mechanisms remain untested. |
| Reproducibility | 10% | 8.0 | Exact support reduction and calibration equation improve specificity, within the same aggregate-evidence artifact limit. |
| Clarity and narrative | 10% | 8.5 | Abstract and introduction now align with the laterality figure and distinguish declared and post hoc questions. |
| Figures | 5% | 8.5 | Accurate, accessible naming figure and corrected diagram remain; no raw trajectory example is available. |
| Submission fit | 5% | 8.0 | Nine-page main text and official formatting retained, with final visual QA handled separately. |
| **Weighted total** | **100%** | **72.75/100** | The small increase is for a clearer narrative; no statistical/evidence-scope inflation. |

For v07, add E11's initialization-gradient caveat and preserve the distinction between shared coefficient, common support, and matched optimization influence. No additional result needs to be claimed.
