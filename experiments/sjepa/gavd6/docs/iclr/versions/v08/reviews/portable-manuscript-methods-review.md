# Full-appendix manuscript portability review

Date: 2026-09-26. Scope: `supplement/technical-details.tex` only. The parent agent owns the main manuscript, concise appendix, document builds, and packaging.

## Disposition

The full appendix now describes experiments, measurement, analysis, and limitations without assuming access to the author's workspace or instructional materials. No unresolved methods or numerical-fidelity issue was found in this source revision. This is an editorial revision, not a new experiment or an independent revalidation of the training results. Render review is handled separately.

## Source binding

- Baseline: `history/v08-before-local-reference-cleanup-20260926.zip`, member `supplement/technical-details.tex`.
- Baseline SHA-256: `13b06b4f8faadfdc362e624b49cb56356ad7288acd3fd86333290f4972408b12`.
- Revised full appendix SHA-256 after the redundant-table removal: `53c9b4b711f00d555bbddd561da51b04dc76568e5e0b671157b879ccddc7517e`.
- Earlier editorial revision, superseded: `c5e6e477d828f0e4fbcc561812f33d044f2b920150577153b552883e242d30db`.

## Conceptual changes

- The opening now identifies the three completed experiments and their development population. It no longer treats configurations, defaults, exports, instructional calculations, or notebook executions as the reader's starting point.
- Movement selection is described by motion names and projected geometry. Its inability to establish clinical status or natural gait is retained. “Local right-knee rotation” remains because it specifies the anatomical transformation, not a computer location.
- Masking describes overlap and truncation directly. It explicitly limits the completed comparison to graph–time masking.
- Calibration and optimization descriptions retain the initial-gradient measurements and clipping counts. A missing complete gradient trajectory, absence of matched-influence retraining, and absence of convergence sweeps remain limitations, expressed without package or audit-workflow language.
- Grouped summaries, response magnitudes, and failure accounting are described statistically. The distinction between whole-population success contribution, success-conditioned error, and conditioning within source families remains explicit.
- Reproducibility is expressed in terms of retained outcomes and missing research materials. Aggregated outcomes permit reanalysis but are insufficient to repeat the full study; missing motion/body-model assets, renders, pose inputs, per-example predictions, checkpoints, and original arrays remain stated.
- The notebook walkthrough was initially replaced by a scientific-questions table, then removed because its seven rows repeated the detailed methods, data controls, and primary-comparison table. The retained experimental-sequence prose and primary table state the three questions directly. Untested masks, external refiners, and movement re-pairing remain clearly unexecuted.
- Data controls use conventional population and study terminology. Person assignments, inherited variant membership, duplicate-motion separation, training-only fitting/calibration, observation-based normalization, excluded deployment inputs, and absence of locked-confirmation outcomes remain stated.
- Participant profiles describe the same people across panels without referring to an audit CSV or internal identifier storage.

The unrelated proposed **102-fit tutorial design** was deliberately removed from the completed-fit caption at the parent agent's explicit direction. It was never a completed experiment or reported result. All completed-fit counts are unchanged. The historical notebook identifiers and the statement about eight matching source hashes were likewise removed because they described instructional or software-audit organization, not experimental outcomes.

## Exact fidelity checks

Compared directly with the snapshot:

- All **152 inline mathematical spans** are byte-identical and remain in the same order.
- All **nine display-equation/align blocks** are byte-identical.
- All **32 label, reference, and citation commands** are byte-identical and remain in the same order.
- All **seven numerical-table and figure inclusion commands** are byte-identical.
- All **six included numerical table files** are byte-identical to the snapshot.
- The inline calibration, completed-fit, and primary-comparison numerical tables are byte-identical.
- The only changed numeric tokens are the intentionally removed unexecuted 102-fit count, removed notebook indices, and widths of the removed nonnumerical walkthrough table. No scientific result, experimental count, model dimension, optimization setting, loss coefficient, confidence interval, or threshold changed.
- The ten sections and five subsections remain. Whitespace-based word count decreases from 4,554 to 4,276; no font or margin setting changed.

A terminology scan, excluding nonrendering LaTeX inclusion commands, found no paths, file extensions, notebook references, workspace/directory/setup references, tutorial/fixture/disk/remote-execution language, exports, receipts, checkout references, or package-inventory descriptions. The remaining anatomical use of “local” is intentional.

## Scientific boundaries preserved

The appendix still states that teacher targets are projected references; inference uses observations and preserves tensor indexing; hidden values cannot affect normalization; calibration and losses use training data; development references may be inspected for consistency checks; unknown identity aliases and upstream estimator overlap cannot be ruled out; and no independent locked-confirmation evaluation exists. It retains the repeated development cohort, adaptive sequence of later experiments, retrospective null formalization, absence of independent preregistration, lack of multiplicity correction, unresolved primary effects, no noninferiority margin, and limited three-seed uncertainty.

Failure costs, reference-defined eligibility, failed predictions in denominators, 4:1 endpoint versus 2:1 response weighting, crossed bootstrap and person-*t* distinctions, and the limits of conditional summaries remain unchanged. Clinical validity, preserved response accuracy, causal attribution to the scalar term, convergence, and complete study reproduction are not newly asserted.

## Final coverage check after redundant-table removal

The parent removed the second table in the experimental-sequence section to avoid an orphaned final paragraph in the optional full-appendix rendering. That table had no label and contained no equations or numerical outcomes. Its removal is scientifically safe: the section retains the three-stage motivation, retrospective null formalization, primary comparisons and intervals, adaptive progression, and final paragraph identifying untested alternatives. Detailed experimental methods retain pair construction, masking, normalization, and objective support; completed-design and data-integrity sections retain failure accounting, inference, and shape/split controls. The complete variant inventory still includes direct, initialized, and shuffled-reference controls, and the methods still define the shuffled-reference operation. No completed control or limitation becomes undocumented. Exact mathematical, reference-command, and numerical-table comparisons above were rerun on the final bound source. No open methods finding remains.
