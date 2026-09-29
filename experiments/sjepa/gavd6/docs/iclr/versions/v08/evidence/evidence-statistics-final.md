# Version 08: independent evidence and statistical review

Reviewed against v07's fixed rubric and the completed development exports. This reviewer authored the version-local numerical reanalysis but did not write the manuscript or figure builder. The review independently checked manuscript claims, plotting transformations, all seven generated numerical tables, retained primary-comparison JSONs, the calibration receipt, and training-ledger clipping counts. This is verification of the available experiment packet, not independent validation of raw motion, rendering, predictions, or training.

## Evidence auditor

**No unresolved material numerical or evidential misstatement was found in the reviewed v08 manuscript and appendix.** The revised contribution is a defensible controlled evaluation of what the fitted predictive procedures restore. It does not establish an improved representation method. The paper now makes zero response central, exposes the feature comparison's unequal initial auxiliary influence, distinguishes adaptive direct fitting from frozen readouts, and reports repair arms and failure accounting in interpretable units.

The main and supplemental numbers reproduce. The seven generated tables contain 96 result rows; every displayed numerical string matches the audited CSV at its printed precision. The eight plotted change-minus-coordinate waveform effects match an independent paired reconstruction. Figure inputs match their recorded hashes. Sixteen retained primary-comparison metric entries reproduce to an absolute tolerance of `1e-10`. `final-numerical-checks.json` records additional independent checks, and `manuscript-claims.json` binds claims and figures to their sources and review snapshot.

| Claim | Verified value or result | Evidence and interpretation |
|---|---|---|
| Direct-coordinate restoration versus unchanged observations | Response gain 5.143928 degrees, crossed interval [2.151695, 8.651396]; 10/14 people improve response, 14/14 improve waveform and NLE | Core `per-person.csv`; distinct paired outcomes, not a general restoration ranking |
| Zero-response benchmark | 5.810830 degrees, lower than all 16 pooled neural response means at cost 720; advantage over direct 1.733325 [0.528417, 3.149885] | Response `per-person.csv`; abstract's reversed effect/interval sign is correct; no invented waveform, coordinates, or naming score for zero prediction |
| Delta versus endpoint response | 0.373054 degrees, crossed interval [-1.110272, 1.760337] | Original response primary retained; uncertain benefit |
| Readout ordering | Coordinate-only delta 10.944775 versus endpoint 10.232657 degrees; interaction 1.085171 [-0.651744, 2.847543] | Reversed point estimates do not establish a population interaction |
| Initial auxiliary influence | Endpoint 10%; delta 0.001303714% of base gradient RMS after the common coefficient, before clipping | Calibration receipt SHA256 `3c00e2931bfa3a8cc7e4d94616774b81a066e60a0aabc3aa22a303a0266620d5`; all eight bound code hashes match |
| Clipping | 11,999/12,000 updates in six feature pretrains; coordinate-delta 0/6,000 | Combined-gradient ledger counts, with no trajectory or causal inference |
| Weighted response failures | Endpoint 0.563210%; delta 0.524783%; direct 0.343762% | Hierarchically weighted rates, not raw failed-pair fractions |
| Failure share of feature contrast | 0.276675/0.373054 = 74.16509% | Arithmetic score decomposition, distinct from failure frequency and causation |
| Nonnegative-cost bound | Delta successful contribution 6.209596 and endpoint 6.305974 exceed zero's 5.810830 | For these observed means and fixed weights, no nonnegative failure cost reverses either ranking; this is not a success-conditional comparison |
| Observation strata | Direct beats zero in both clear-image nominal-edit groups and loses in both occluded groups; delta/endpoint lose in all four | All groups and all 16 methods are retained; nominal edit levels are not mislabeled actual response-magnitude bins |
| Original package | All eight waveform means worsen; all 14 person means and all three seed means worsen in each of the five core families | Package adds scalar and geometry terms; no isolated scalar causation claim |
| Repair primary | Delta dense-minus-low gain 0.278856 degrees, person-t interval [-0.194914, 0.752625]; crossed sensitivity [-0.436648, 0.857563] | ViTPose, retained delta encoder, conditional fitted-seed primary; original-to-low gain 3.697397 degrees |
| Laterality discrepancy | Delta excess assignment failure 0.768812 percentage points [0.173766, 1.527897] | Exploratory, unadjusted combined wrong/ambiguous/missing rate with separate eligibility; no chance claim |

The new failure analysis adds a useful distinction absent from the earlier narrative. Direct's ranking against zero changes with the failure cost, while the two predictive change-readout variants already lose at the zero-cost lower bound. The manuscript restricts this observation to fitted outputs and preserves the original primary at 720 degrees. The four-cost sweep is complete and none of its delta–endpoint intervals excludes zero.

The figure redesign accurately separates different questions. Restoration shows all eight families and both objectives with a practical direct reference distinguished from frozen models. Reliability separates rates from additive contributions and sensitivity effects. Repair retains absolute means and paired intervals rather than relying on a 93% ratio. Laterality uses condition-specific points with no categorical connecting lines or invented component decomposition. The method schematic conveys single-state inference, privileged reference teaching, and the distinction between physical mirroring and input naming. No synthetic image is presented as an observed reconstruction.

## Statistical reviewer

**No unresolved material statistical misstatement was found within the retained-export scope.** The independent population remains 14 development people; the three fitted seeds are not multiplied into 42 independent participants. New intervals preserve the paired person-by-seed panel. Core/response and exploratory crossed intervals resample people and seeds independently while preserving method pairing; the repair primary retains its different person-t interval after seed averaging. Mean intervals are not used as substitutes for paired-effect intervals.

The original three primaries remain unresolved. The manuscript distinguishes declared primary from preregistered, secondary coordinate-readout contrasts from the primary, and post hoc laterality, observation strata, penalty sensitivity, and new figure intervals from confirmation. It reports the uncertain readout interaction instead of treating point-estimate reversal as proof of effect modification. A favorable endpoint repair secondary does not replace the delta primary, and response noninferiority is neither specified nor claimed.

The reanalysis preserves source weighting. Nonheld/held endpoint strata use 4:1 and nonzero response strata use 2:1; reaggregation reproduces the original pooled means. The same support and failure indicators are used across the fixed-cost sweep. Conditional successful error is explicitly `U/(1-f)`, which differs from both the unconditional successful contribution and the existing export's nested conditional mean. This prevents an incorrect additive decomposition or silent denominator change.

The true-magnitude analysis is appropriately bounded by data availability. The compact curve export averages signed reference responses before export; taking an absolute value afterward cannot recover individual magnitudes or magnitude bins. The paper reports all supported clear/occluded-by-nominal-edit and clear/occluded-by-estimator groups instead. Neither the new strata nor the 2,000 bootstrap draws increase the amount of independent evidence.

The remaining statistical limitations are substantial: repeated inspection of the same development people; only three seeds; adaptive sequential development; post hoc selected diagnostic questions; no multiplicity-adjusted confirmatory inference; no new protected-person results; and no broad optimization, external-baseline, or natural-observation comparison. These are clearly reported and cannot be repaired by wording or additional resampling.

## Objections and dispositions

| ID | Severity | Evidence or concern | Disposition in v08 | Residual status |
|---|---|---|---|---|
| V8-E01 | Major framing | Zero response 5.811 beats all 16 pooled learned variants | Central in abstract/results and complete inventory | Useful recovery remains conditional on observation group and task/cost |
| V8-E02 | Major interpretation | Shared coefficient masks 10% versus 0.0013% initial influence; almost universal clipping | Receipt, formula, code hashes, and ledger counts moved beside feature comparison | Matched-influence/convergence retraining unavailable |
| V8-E03 | Moderate scope | Earlier draft's abstract could imply repair improved all eight families | Final abstract restricts follow-up to two frozen encoders | No all-family repair evidence claimed |
| V8-S01 | Major denominator | Failure rate, failure contribution, and successful conditional error differ | Separate rates, additive contributions, explicit conditional formula, all-cost sweep | Penalties remain a chosen application-independent stress test |
| V8-S02 | Moderate selection | Magnitude binning from averaged signed responses would be invalid | Explicitly rejected; all supported nominal-edit/observation groups retained | Per-pair magnitude analysis requires missing artifacts |
| V8-S03 | Moderate inference | Mean intervals and different bootstrap/t procedures could be conflated | Self-contained captions identify pairing, sample, procedure, and status | Three seeds still yield coarse training-variation inference |
| V8-S04 | Major design | Same 14 people inspected across stages | Explicit development scope and unchanged primary statuses | Bounded, not empirically resolved |
| V8-E04 | Moderate comparison | Direct adapts encoder; other families freeze it | Table, plot groupings, and captions identify adaptation | No matched adaptation/exposure experiment |
| V8-E05 | Moderate mechanism | Scalar-plus-geometry package and dense repair could be overattributed | Package wording, fixed-term repair comparison, ratio subordinate to absolute effects | No isolated sparsity or coupling mechanism established |
| V8-R01 | Moderate reproducibility | Compact packet cannot support raw reconstruction or full training rerun | Recalculation and end-to-end reproduction explicitly separated | Raw assets, predictions, and checkpoints remain required |
| V8-E06 | Moderate generalization | Initial gradient or point-estimate reversal could imply general predictive failure | Initialization-only scope, uncertain interaction, fitted-procedure conclusion | General representation benefit remains unanswered |
| V8-S05 | Moderate laterality | Around-50% score or signed summary could imply chance/anatomical recovery | Explicit combined geometric assignment rule, post hoc status, per-leg counterexample | Anatomical/clinical and external validation absent |

Earlier v07 objections concerning mirror/sign conflation, scalar identification of the changed leg, inaccurate interval endpoints, missing assignment thresholds, mutable figure dependencies, probe population, single-state inference, post hoc disclosure, condition weighting, and initial gradient interpretation remain addressed. The newly requested scope and denominator objections are also addressed. No material reporting objection is being hidden by the rubric score.

## Fixed rubric, compared with v07

| Dimension | Weight | v07 /10 | v08 /10 | Concrete reason |
|---|---:|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | 6.5 | A clearer evaluation contribution supported by cost and observation-condition results; no new method advantage or external validation |
| Claim accuracy and evidence support | 20% | 9.0 | 9.0 | Accurate, auditable claims in both versions; raw-source validation still unavailable |
| Evaluation and statistical rigor | 15% | 5.0 | 5.5 | Complete supported strata, fixed-cost sensitivity, denominator clarity, and paired effect displays improve analysis; independent sample/design limitations unchanged |
| Scientific insight and positioning | 15% | 7.0 | 8.0 | Zero-baseline heterogeneity and the nonnegative-cost lower bound sharpen the empirical lesson beyond scalar information loss |
| Reproducibility | 10% | 8.0 | 8.5 | Executable exact comparison/weighting checks, full numerical tables, editable plots, and claim bindings improve reconstruction from exports |
| Clarity and narrative | 10% | 8.5 | 9.0 | Question-centered organization and consolidated limitations clarify the controlling comparisons |
| Figures | 5% | 8.5 | 9.0 | Substantive coordinated redesign, complete family/repair comparisons, separate rate/accounting panels, and paired uncertainty improve readability |
| Submission fit | 5% | 8.0 | 8.0 | Nine-page main text and official-format checks are retained; this review does not predict acceptance or certify unverified administrative requirements |
| **Weighted total** | **100%** | **72.75/100** | **77.25/100** | **+4.50 points for concrete analytical and presentation improvements, with no new independent experiment** |

The revision is a materially stronger, accurately bounded development-study manuscript. It remains limited as evidence for an incremental predictive-representation benefit, anatomical recovery, clinical utility, or a causal optimization explanation.
