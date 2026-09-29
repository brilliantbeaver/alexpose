# Comparison selection for the simplified v08

This review proposes a smaller scientific comparison set without changing any measurements, fitted procedures, or source evidence. It reads the current main paper, appendix, saved primary-comparison JSONs, and version-local audited CSVs. The manuscript and data were not edited, and no new intervals were calculated. Selection is by scientific role and recorded question, not by statistical significance or favorable ordering.

## Main Table 1: exactly three rows

Use the columns **Question**, **Comparison**, and **Outcome**. The following wording keeps the table intelligible without run IDs, layer counts, or a model-family catalogue.

| Question | Comparison | Outcome |
| --- | --- | --- |
| Does learning to predict reference features improve restoration? | Core JEPA with a frozen encoder versus direct fitting of encoder and readout; both use the original change objective. | Pooled error in the change in gait asymmetry. |
| Does predicting the difference between paired features help? | Delta versus endpoint JEPA; both use frozen encoders and the original change readout. | Pooled error in the change in gait asymmetry. |
| Does denser change supervision improve knee trajectories? | Dense versus low-scalar readout for the same retained delta encoder. | Knee-angle trajectory error on ViTPose inputs. |

Suggested caption: “The three recorded primary comparisons. The original change objective adds asymmetry-change and short-segment penalties to coordinate fitting. All stages reuse 14 development people and three training seeds; later stages do not provide independent confirmation.” The surrounding methods should continue to explain that the first comparison changes encoder adaptation and coordinate-supervised exposure. Calling it an isolated test of representation architecture would overstate what was matched.

Do not place all secondary families back into a caption. A short subsequent sentence can name their purposes: initialized features check how much a fitted readout can obtain without encoder learning; shuffled references check correspondence; coordinate pretraining checks a reconstruction alternative. Their exact configurations belong in the technical supplement.

## Appendix C: questions and the reference checks that constrain them

Rename the section **“Primary comparisons and essential reference checks.”** Replace the 16-by-5 outcome inventory with the seven rows below, separated into three recorded primaries and four interpretive checks. A compact three-column layout can contain the comparison/outcome, reference-to-candidate mean errors, and paired gain with its interval. All gains below are reference error minus candidate error, so positive values favor the candidate. Values are in degrees.

| Comparison and scope | Reference → candidate means | Gain [95% interval] |
| --- | --- | --- |
| **Primary:** direct/change → core JEPA/change; pooled response | 10.75 → 10.07 | 0.69 [−0.64, 1.98], crossed |
| **Primary:** endpoint/change → delta/change; pooled response | 10.36 → 9.99 | 0.37 [−1.11, 1.76], crossed |
| **Primary:** delta/low scalar → delta/dense; ViTPose waveform | 19.47 → 19.19 | 0.28 [−0.19, 0.75], person-t |
| **Exploratory baseline:** unchanged observations → direct/coordinate; pooled response | 12.69 → 7.54 | 5.14 [2.15, 8.65], crossed |
| **Exploratory baseline:** zero response → direct/coordinate; pooled response | 5.81 → 7.54 | −1.73 [−3.15, −0.53], crossed |
| **Secondary readout check:** endpoint/coordinate → delta/coordinate; pooled response | 10.23 → 10.94 | −0.71 [−1.73, 0.18], crossed |
| **Exploratory weighting check:** delta/original change → delta/low scalar; ViTPose waveform | 23.17 → 19.47 | 3.70 [2.58, 4.81], person-t |

The six decimal sources for the three primary effects are respectively `0.688084 [−0.640212, 1.976709]`, `0.373054 [−1.110272, 1.760337]`, and `0.278856 [−0.194914, 0.752625]`. Preserve the recorded person-t interval for repair rather than accidentally taking the crossed sensitivity columns from `repair_effects.csv`. A short caption should state that response and waveform retain their original 720° and 180° failure costs; the first two primary intervals resample people and seeds, while the repair primary conditions on the three fitted seeds. No row combines these different outcomes into an overall model ranking.

The coordinate-only row matters because it prevents presenting the delta point estimate as a stable advantage across readouts. Its interval also crosses zero. The interaction remains uncertain, 1.09 [−0.65, 2.85]°, so the reversal must not become a claim that readout choice reliably changes the feature comparison. The low-scalar row matters because it separates the substantial weighting effect from the unresolved incremental dense-supervision effect. Endpoint's favorable secondary dense result cannot replace the delta primary; it can remain in the repair discussion and complete source table.

Retain this small **descriptive negative-control block** immediately below the table, either as two sentences or three compact comparison rows. Each number pair is comparator → core JEPA response error, with the same readout objective on both sides:

| Control question | Coordinate readout | Original change readout |
| --- | --- | --- |
| Learned features versus initialized frozen features | 9.43 → 10.69 | 10.16 → 10.07 |
| Matched references versus shuffled references | 9.35 → 10.69 | 11.75 → 10.07 |
| Feature prediction versus coordinate pretraining | 10.25 → 10.69 | 11.17 → 10.07 |

These are descriptive means without new paired-interval claims. Both objectives are shown because selecting only the original-change control values would conceal the opposite coordinate-only ordering. Initialized encoders still have trained readouts; they are not wholly untrained prediction systems. Shuffling substitutes another accepted window from the same person while preserving endpoint role and observation conditions; it is not the unperformed movement re-pairing control for the delta auxiliary. The coordinate-pretraining row avoids implying that predictive features were the only tested way to learn from references. Coordinate-delta remains visible in the all-family figure and full supplement; no favorable or unfavorable outcome is erased by omitting its internal configuration from this summary.

The seven-row table and three descriptive control rows are the recommended complete selection, not a menu from which to retain only favorable comparisons. If space is tight, turn the control block into prose rather than dropping its coordinate-only counterexamples. The existing all-family restoration figure already preserves the eight-family waveform deterioration result. Appendix C need not reproduce its 16 means for every outcome. The notebook-journey section can point to this primary table instead of repeating another three-row results table.

## Distinctions and safeguards that must survive

- **Direct/change and direct/coordinate have different roles.** The first is the original primary comparator; the second is the stronger practical restoration reference. Substituting the latter changes the primary question. Direct fitting updates its encoder, while the feature families fit readouts on held-fixed encoders.
- **Pretraining targets and readout objectives are separate choices.** “Coordinate pretraining” is not “coordinate-only readout.” Endpoint and delta name feature objectives, not gait measurements. Original change adds scalar and geometry penalties together; low scalar alters one retained term, whereas dense replaces the scalar term with a calibrated angular objective.
- **Pooled response and ViTPose waveform are different outcomes and populations of conditions.** Do not mix their means in one unnamed error column or rank them against one another. Keep 14 people and three fitted seeds, the original hierarchy, and reused development status visible.
- **Zero response supplies only a response estimate.** It has no restored waveform, joint coordinates, or assignment prediction. Retain the clear-versus-occluded exception in the main results; its pooled superiority does not hold for every observation group. Preserve all selected groups in the supplement, including unfavorable ones.
- **The numerical and anatomical checks answer different questions.** Response failure, geometric assignment failure, and per-person error deterioration have different meanings and eligibility rules. Simplification must retain the post hoc status of naming, the combined wrong/ambiguous/missing definition, and the absence of an established 50% chance line.
- **Selected presentation is not selected evidence.** State why these rows were selected, link the full inventories, retain uncertain intervals and counterexamples, and keep secondary/exploratory labels. Do not rename the three stages independent replications or present a retrospective directional null as preregistration.

## Complete material to retain in the source supplement

Preserve all current generated tables even if they are no longer included in the PDF: `tables/inventory.tex` (16 learned variants and five outcomes), `repair-inventory.tex` (all nine ViTPose procedures/references), `strata-levels.tex`, `strata-extractors.tex`, `failure-inventory.tex`, and `penalties.tex`. Provide one readable supplementary Markdown or PDF inventory alongside the CSV/TeX so inspection does not require running code. Keep the table builder and a pointer from the shortened appendix and package README.

Retain `evidence/method_summary.csv`, `restoration_means.csv`, `restoration_effects.csv`, `condition_person.csv`, `condition_summary.csv`, `condition_effects.csv`, `penalty_sensitivity_person.csv`, `penalty_sensitivity.csv`, `penalty_effects.csv`, `repair_means.csv`, `repair_effects.csv`, and the case-figure CSVs. Preserve definitions, hierarchical weights, hashes, and source mappings. The complete six deterministic core controls (`unchanged`, three filters, joint offset, joint affine) remain in the original core export even though only unchanged appears in the focused summary; do not describe the 16-neural-variant file as the entire completed control inventory.

Primary numerical authority remains `outputs/iclr/{walking-core,jepa-response}/evaluation/comparisons.json` and `outputs/iclr/readout-repair/development/evaluation/comparisons.json`. The coordinate-readout comparison and interaction are in `outputs/iclr/jepa-response/evaluation/readout-control-comparisons.json`. Control and practical-baseline means come from the core/response `per-person.csv` files and the audited version-local summaries. The zero comparison is the original-cost row of `evidence/penalty_effects.csv`; repair weighting is in `evidence/repair_effects.csv` using its person-t columns. Retain all original exports and their provenance unchanged. Compacting the PDF should not trigger re-estimation, resampling changes, or deletion of an unfavorable arm.

**Assessment:** this selection substantially reduces configuration detail while keeping all three recorded primary questions, practical benchmarks, negative controls, and the counterexamples needed for honest interpretation. The essential statistical limitations and numerical conclusions remain unchanged.


## Independent review of the implemented simplification

Reviewed on 2026-09-26. The editor used the three-question design table and a three-primary forest figure with contextual prose, while retaining every one of the eight core/response procedures under both readout objectives in the main restoration figure. I accept this alternative to the seven-row results table proposed above: it preserves the critical comparisons and negative-control counterexamples once in the default document, without pooling configurations into new means. The full new Appendix A–E was read, and both new summary figures were inspected.

**Numerical and inferential checks passed.** The new question figure selects the original source JSON's core/change versus direct/change response contrast, delta/change versus endpoint/change response contrast, and delta dense versus low-scalar ViTPose waveform contrast. Candidate/comparator identities, gains, and interval endpoints match those original fields exactly. The repair figure retains its recorded person-t interval, rather than replacing it with the crossed-bootstrap sensitivity. The figure visibly separates the response and waveform rows, labels their interval procedures, and reports no pooled effect. Its eight output-file hashes match the new figure provenance.

The Appendix C range **4.39–7.21°** is the correctly rounded minimum/maximum of all eight change-minus-coordinate waveform means: **4.38784059052076°** for delta and **7.207554868221944°** for direct fitting, from `evidence/restoration_effects.csv`. The zero, unchanged, and direct-coordinate means, the delta/endpoint coordinate-only reversal, and the **3.70° [2.58, 4.81]°** low-scalar improvement remain faithful. The uncertainty of the interaction and added dense-supervision benefit remains explicit. The visibility counterexample, failure-cost lower-bound argument, and all-participant profiles are retained, so simplification has not removed evidence that changes the conclusions.

**Controls remain properly bounded.** Appendix C calls initialized, shuffled-reference, and coordinate-pretrained variants diagnostic controls within the reused cohort, and claims no established feature-learning benefit from them. Both objectives remain visible in the main all-family figure, including the controls' coordinate-only ordering that would be hidden by selecting only their original-change scores. The text preserves the original core/change comparator, distinguishes direct coordinate fitting as the practical reference, and does not elevate endpoint's secondary repair result to the primary. The protocol summary retains person-level split inheritance, the distinct 112/14 fitting/development populations, fixed endpoint dimensions, reference-based eligibility, hierarchical weighting, and the limits of adaptive development reuse.

**Evidence preservation was verified byte-for-byte.** `supplement/technical-details.tex` equals `appendix.tex` from `history/v08-before-comparison-simplification-20260926.zip`. All seven archived numerical table files and all 13 archived audit CSVs are unchanged. The supplement README explains how to typeset the complete technical reference using the retained table and figure sources. Compacting the default PDF therefore does not discard the complete inventory or change its numerical authority. Final rebuilt bundle contents and PDF pagination belong to the editor's package checks.

Two small corrections found during review are closed: the main observation-strata paragraph now reads “and retain,” and Appendix C now calls the estimand a **paired difference in measurement error between fitted procedures**, avoiding confusion with the within-motion asymmetry response. No unresolved material numerical, selection, or inferential misstatement was found in the revised selection. No manuscript, source data, numerical export, or figure was edited by this review.

Source bindings at this review (later layout-only edits require refreshing these hashes):

- `paper-v08.tex`: `2ee2025c0c40da1a3508afdb24f895a9bcf0cd7a4a885469c714c7f14f2017af`
- `appendix.tex`: `65e8d0ebb34f4854e24cec7bcc1aa767fa0b32039107bef82b9e53d036a798f6`
- `supplement/technical-details.tex`: `13b06b4f8faadfdc362e624b49cb56356ad7288acd3fd86333290f4972408b12`
- `scripts/build_summary_figures.py`: `4b4de8b7d201cb1c33f07b24afdbaea6816df10777d7ad937ff8baaf0ab046a6`
- `figures/study-questions.pdf`: `d171802c5d8d9de9dbcf9c57758e827176d7fe0a2711d7ba6b6ea91ac1ff634d`
- `figures/protocol.pdf`: `855da0812a1d56029494a8c714ca799bac2562c05b6e4fd847f827c94729b3e0`
