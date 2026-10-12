# Version 02: independent evidence and statistical review

Reviewed artifacts: `paper-v02.tex`, `scripts/build_figures_v02.py`, and its separately retained `figures/v02/` assets. The evidence scope is unchanged: 14 reused development people and three fitted seeds, with no new training or confirmation. This reviewer independently verified the empirical tables and saved comparisons as recorded in `evidence/independent-evidence-verification.json`.

## Evidence auditor

The revised mirror caption distinguishes algebraic exchange of measured excursions from three-dimensional reflection followed by fixed-camera projection. This resolves v01 E01. The abstract now asks for retention of side-sensitive measurements, and the introduction states that the signed scalar does not identify the changed leg, resolving v01 E02 within the paper's bounded synthetic labeling scope. The definition of geometric assignment now gives the correct four-pixel reference separation, two-pixel named/swapped margin, and wrong/ambiguous/missing categories, resolving E04. Version-specific figures resolve E05. The displayed core interval now uses the correct endpoints; the remaining source-generation concern is traceability rather than a numerical discrepancy.

All primary estimates, baseline means, normalized coordinate errors, repair values, failure contributions, and laterality percentages remain accurate. The repair target now averages supported pairs explicitly, improving correspondence between equation and implementation. No new material numerical misstatement was found.

| ID | Severity | Evidence and objection | Required disposition | Residual limitation | Status at v02 |
|---|---|---|---|---|---|
| E01 | Major in v01 | Fixed-camera mirroring did not imply exact projected A sign reversal. | Corrected figure label/caption and prose distinguish the transformations. | No direct equivariance experiment. | Resolved |
| E02 | Major in v01 | Signed bilateral change did not establish changed-side identity. | Abstract and introduction now use a bounded measurement claim. | Anatomical validity still requires new data. | Resolved |
| E03 | Moderate remaining | Plot code still hardcodes comparison values rather than reading comparison JSON, despite corrected endpoints. | Load all plotted effects/intervals from source JSON and record their hashes in v03. | No raw-data rerun in this packet. | Numerically corrected; provenance open |
| E04 | Moderate in v01 | Geometric metric thresholds and denominator were unspecified. | Full thresholds and categories now included. | Geometric metric is not a calibrated anatomical classifier. | Resolved |
| E05 | Moderate in v01 | Shared figure edits could alter rebuilt earlier versions. | v02 uses separate figures and source. | Preserve this practice through v07. | Resolved |
| E06 | Minor | Probe section says 333 pairs and training population but not its 112 people; the person-separated probe fits could be mistaken for a separate evaluation cohort. | Name 112 training people and clarify that they were seen by the encoder, although held out from each probe fit. | One fixed ridge probe cannot establish all information availability. | Open |

## Statistical reviewer

The statistical interpretation continues to be sound on the primary questions: three uncertain contrasts remain uncertain; intervals are correctly distinguished; no equivalence or noninferiority is inferred; failures retain their original denominator; deterministic benchmark rows are not additional seeds; and sequential use of the same people is visible.

The new laterality paragraph still needs its post hoc status. The selected global-swap contrast may be compelling, but its selection followed examination of the entire development panel. This affects the strength of inference even when the descriptive percentages are correct. The general multiplicity sentence does not fully disclose that origin.

| ID | Severity | Evidence and objection | Required v03 correction | Residual limitation | Status at v02 |
|---|---|---|---|---|---|
| S01 | Major | Laterality/global-swap analyses are selected post hoc but presented without that label. | State they are exploratory development diagnostics selected after the main experiments. | Independent confirmation must freeze this hypothesis. | Open |
| S02 | Moderate | The printed condition means are right, but the text omits 4:1 endpoint versus 2:1 response nonheld/held weighting. | State weights and their reason, or identify the exact executable reconstruction. | Direction accuracy uses distinct eligibility and should not be recombined by these simple weights. | Open |
| S03 | Moderate | Per-leg excursion outcomes provide an important counterexample to a broad reading of direct's favorable ranking. | Add that direct L/R excursion errors are 17.94/17.19 degrees versus delta/scalar 15.70/15.29, while retaining all other outcomes. | No universal best method or scalar summary. | Open |
| S04 | Substantial empirical limitation | Fourteen reused people, three seeds, no confirmation, no broad training-budget/weight sweep. | Already bounded; do not relabel as solved by revision. | Requires new experiments. | Bounded, unchanged |

The suggested per-leg sentence is evidence-balanced reporting, not a request to replace the response or waveform primary endpoints. Likewise, a laterality figure should label descriptive means and avoid showing new significance claims without their exploratory provenance.

## Fixed rubric

| Dimension | Weight | Score /10 | Reason and remaining weakness |
|---|---:|---:|---|
| Conference relevance and contribution | 20% | 6.0 | Same limited but useful structured-output evaluation and unresolved feature-prediction advantage. |
| Claim accuracy and evidence support | 20% | 8.0 | Main conceptual misstatements corrected; remaining omissions concern post hoc status and completeness. |
| Evaluation and statistical rigor | 15% | 5.0 | Careful primary analysis, but the underlying repeated-development and precision limitations have not changed. |
| Scientific insight and positioning | 15% | 6.5 | Corrected geometric distinction improves the argument; causal mechanism remains unresolved. |
| Reproducibility | 10% | 7.0 | Version preservation and correct values help, but JSON-driven intervals and a condition-weight account are still needed. |
| Clarity and narrative | 10% | 7.5 | Clearer operational laterality and compound-loss wording; results could integrate laterality earlier. |
| Figures | 5% | 7.5 | Corrected transformation illustration; no new empirical figure or independent full-page render review in this assessment. |
| Submission fit | 5% | 7.0 | Same official format and bounded scientific scope; page/layout verification is delegated to the build review. |
| **Weighted total** | **100%** | **67.0/100** | Gain comes from specific corrected claims, not from the version number; evaluation score remains fixed. |

Priority for v03 is S01, followed by E03/S02 and the concise S03 counterexample. These revisions can improve auditability and balance without increasing empirical support for the method hypothesis.
