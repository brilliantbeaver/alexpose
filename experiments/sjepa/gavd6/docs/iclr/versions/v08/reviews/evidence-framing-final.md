# Evidence/statistics review of the v08 framing extension

This review covers the revised motivation, retrospective hypothesis notation, notebook reconstruction appendix, tensor/split controls, and participant Figure 6. It supplements `evidence-statistics-final.md`; the earlier primary numerical audit remains applicable. The reviewer inspected the revised source, the methods and literature audits, executed configuration and ledger fields, token packing/decoding, training-pair selection, dataset validation, and the retained tutorial-validation receipt. No training or notebook execution was added.

## Numerical and figure findings

The new participant figure is faithful to the saved data. `case-figure-seeds.csv` contains 126 rows: three panels × 14 people × three fitted seeds. `case-figure-person.csv` contains 42 panel/person means. Every person is present in each panel, ordered by the same canonical identifier and displayed as P01–P14. The plotted seed rows, summaries, figure files, and provenance hashes agree.

Pooled unchanged-minus-direct gain is 5.143928 degrees, with 10/14 person means favoring direct. Clear-image zero-minus-direct gain is 1.913510 degrees, with 14/14 favoring direct. Occluded-image zero-minus-direct gain is -5.380161 degrees, with 1/14 favoring direct. The caption correctly identifies different comparators and axis ranges, the 720-degree failure cost, 2:1 response-stratum weighting, the retained hierarchy, and seed ranges rather than confidence intervals. It does not invent pose examples, clinical cases, or additional independent participants.

The newly restated primary-effect table uses the correct comparisons, directions, outcomes, and interval procedures. All three primary intervals include zero. The added coordinate-delta coefficient `3.79389230231159` matches the original calibration receipt. Training/development row counts of 315,840/55,800 are consistent with 1,645 × 192 and 155 × 360 records and the executed ledger; they do not replace 112/14 as person counts.

## Hypotheses and research sequence

The directional no-benefit notation is acceptable because both main text and appendix explicitly label it a retrospective formalization of recorded candidate/comparator/outcome choices. The source records do not establish that these mathematical null statements were originally preregistered. The manuscript retains the historical two-sided intervals, performs no new one-sided test, and asserts no clinically meaningful, equivalence, or noninferiority margin. All three configurations have a null `meaningful_margin_deg`, including repair's authoritative nested protocol.

The notebook narrative accurately describes a reconstruction path, not a sequence of 15 completed scientific experiments. It distinguishes graph–time source fits from planned alternative mask, learned-refiner, and movement re-pairing controls. The original core, response, and repair ledgers remain the sources of empirical findings. Repeated development inspection and the absence of independent confirmation remain explicit.

The retained refresh receipt reports targeted software checks, response/repair calculations, and the downloaded-evidence walkthrough, while marking full fixture execution blocked by disk exhaustion and HAIC validation false. The appendix reports that limited status rather than claiming that every current notebook completed. The methods audit additionally distinguishes current notebook hashes from retained executed copies; the manuscript makes no unsupported current-file execution claim.

## Data integrity and framing

The tensor contract matches the source: four consecutive frames of one joint become a 20-channel patch, 32 × 12 patches become 384 width-96 tokens, and the readout emits eight coordinates per token before restoring the 128 × 12 × 2 grid. Missing slots remain queries. Artificial hiding zeroes confidence as well as coordinates and usability; naturally missing coordinates are made safe while finite native confidence may remain. The manuscript distinguishes these cases and does not claim preservation of raw AMASS values or record counts.

Canonical split inheritance, duplicate-hash rejection, train-only pair selection, input-only normalization, inference-field allow-lists, and locked-confirmation rejection are supported implementation controls. The precise leakage statement is also correct: full-bundle validation can read development references, while optimization and coefficient-calibration losses select training pairs. The text does not claim those references are never opened. It also retains the limitations of unknown identity aliases, upstream estimator exposure, and development-driven later design choices.

The new clinical citations motivate measurement fidelity; they do not become evaluated clinical evidence for this system. The literature audit supports their bounded statements and records its primary-source access limits. The paper avoids labeling synthetic knee edits as disease simulation or assuming symmetry universally defines healthy gait. The world-model link remains motivation: privileged-reference, complete-window restoration is explicitly distinguished from future-state dynamics or planning. The final abstract's future-state limitation is consequential and should remain.

## Objections and disposition

| ID | Severity | Concern | Requested correction or accepted boundary | Status |
|---|---|---|---|---|
| F-E01 | Minor precision, important to laterality | Section 3.1 says joint identities persist through corruption, although naming corruption deliberately permutes anatomical contents of slots. | Say “joint-index slots” or “named joint slots” when describing tensor structure. References retain anatomical names; corrupted inputs need not. | Resolved in the final source: joint-slot order is distinguished from anatomical identity. |
| F-E02 | Moderate interpretive wording | Appendix participant prose says “failures to recover changes,” which could overstate higher response MAE as absence of response information. | Describe higher error than zero prediction for most people under occlusion; retain the explicit distinction from binary measurement failures. | Resolved in the final source: higher error than zero prediction is stated directly. |
| F-S01 | Major if unqualified | Retrospective null statements might be mistaken for original one-sided preregistered tests. | Main/appendix explicitly state retrospective notation, no new one-sided test, original two-sided intervals, no margins. | Addressed in reviewed source. |
| F-R01 | Moderate | A notebook sequence might imply completed alternative controls or fresh full execution. | Tables separate completed fits from teaching/planned calculations; retained validation limitations are explicit. | Addressed in reviewed source. |
| F-E03 | Major if claimed | World-model or disease motivation could imply tested dynamics, clinical labels, or clinical restoration. | Complete-window privileged-reference adaptation, no future-state model, no clinical examples, and no healthy/disease cohort claims are explicit. | Bounded; capabilities remain untested. |
| F-S02 | Moderate | Participant profiles could be selected examples or mistaken for independent seed observations. | All 14 people and three seeds are retained; paired differences and seed ranges are explicit. | Addressed; no sample-size inflation. |

No material numerical or statistical misstatement was found in the additions. Both wording corrections are now verified in the final source. They tighten already stated boundaries without changing a numerical conclusion. The evidence review has no unresolved material reporting objection within the saved-export scope.

The fixed rubric remains **77.25/100**, versus v07's 72.75. The stronger explanation and participant illustration are useful, but the prior v08 scores already credit clear presentation and strong figures. They do not warrant another automatic increase without new independent evidence. Fourteen adaptively reused development people, three seeds, unequal initial auxiliary influence, absent external fitted baselines, and missing confirmation remain the controlling empirical limitations.

`evidence/framing-claims.json` maps these added claims to relative source paths and the methods, literature, and case audits. `evidence/manuscript-claims.json` retains the earlier claim inventory and adds the participant figure; final manuscript/PDF hashes were refreshed after the reviewed build, together with all current source and figure bindings.

Final closure verified against the nine-main-page, 23-page compiled snapshot with PDF SHA256 `67c63826c7d396b3eafb1d6eec7c2379cef99eb5b139dc809aeecb2e24d96f83`. No manuscript edits were made by this reviewer.
