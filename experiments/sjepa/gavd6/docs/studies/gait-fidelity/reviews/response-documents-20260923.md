# Review of the focused movement-response documents

**Revision:** 23 September 2026. **Disposition:** accepted after scientific and readability corrections. This is a documentation review, not evidence of a successful source experiment.

[Current proposal](../README.md) · [Detailed protocol](../methods/jepa-response.md) · [Validation receipt](../records/response-documents-20260923.json)

## Scope and independent roles

The revision updates the active proposal, method pages, data specification, literature positioning and all eighteen vector diagrams for the three-variant JEPA response follow-up. Historical audits, reviews and implementation validation records remain dated snapshots. The implementation, experiment settings, notebooks and Slurm files were not changed by this documentation revision.

The primary editor wrote the proposal, data and literature updates, then checked the generated reading experience. A separate methods reviewer traced the scientific details to the implementation, revised the detailed protocols and independently reviewed the primary editor's claims and the figure author's rendered diagrams. A separate figure author rebuilt the illustrations and independently checked their captions against the written protocol. Reviewers examined one another's work before the final rebuild.

## Scientific findings and corrections

| Review finding | Correction |
| --- | --- |
| An unfamiliar reader could reach features and losses before understanding the measured movement. | Added a first-principles introduction: video, joint coordinates, trajectories, references, bilateral measurement, movement response, features and frozen readout. Definitions also accompany equations, with a terminology page for reference. |
| Matched reference support could be mistaken for numerically identical teacher features throughout training. | Explained that seed-specific initialization and target-construction rules match, while teachers evolve separately in each fit. The algebra compares losses on common tensors. |
| Older descriptions could imply that blur is an active observation condition. | Current data descriptions specify inherited obstruction and joint-naming errors. No new blur arm is claimed. |
| A small difference or narrow interval could be described as equivalence. | Explicitly require a scientifically meaningful equivalence margin for that claim; an uncertain difference remains inconclusive. |
| The signed-direction threshold could be read as selecting positive responses only. | Specify that the absolute reference-change magnitude must exceed the saved one-degree tolerance. Missing predictions remain incorrect. |
| Later-stage supervision and pretrained features could be conflated. | Show the existing movement-supervised readout, discarded predictor and observation residual path, and explain why initialized-feature and direct-training controls remain relevant. |
| Existing datasets could appear to enter the new child run automatically. | Separate exact parent-bundle reuse from GAVD already reported on HAIC, Mac-only audited clips and optional datasets not established as acquired. |
| The repository-wide ignore rule hid the data documentation from ordinary Git updates. | Added a directory-local exception for the five reviewed documentation/viewer files; experiment arrays remain excluded. |

The final protocol retains the common-bias limitation of the delta loss, independent observation normalization, matched endpoint exposure, all-parameter gradient calibration, person/seed aggregation and failure penalties. It makes no claim of clinical validity, unique anatomical-pairing causation or a general JEPA advantage. HAIC profiling remains necessary for source-runtime and GPU compatibility evidence.

## Figures and interactive explanations

All eighteen diagrams were independently reviewed on a contact sheet. The residual-loss diagram, architecture, coverage diagnostic and response example were also inspected at native resolution. Review found no remaining text overlaps, clipping or arithmetic errors. Automated geometry checks cover label bounds, panel containment, text collisions and declared connector paths; these checks supplement rendered review rather than replace it.

The review led to explicit shared encoder/predictor weights across movement endpoints, the complete calibration → pretraining → frozen encoder → readout sequence, a schematic heatmap legend and captions aligned to the figures' actual content. Scope ribbons distinguish the active follow-up, inherited design, supporting diagnostics and deferred extensions. The projected-angle response example separates signed bias from absolute error.

Two interactive examples illustrate arithmetic only. The first compares shared, opposite and zero feature errors. The second separates signed response bias, absolute response error and sensitivity to observation conditions. Their functions were checked against hand-computable cases; actual Chrome controls were exercised for both examples. The diagrams and controls contain no study results and cannot alter an experiment.

## Validation and remaining limits

The accompanying receipt records local links, generated HTML structure, figure hashes, browser checks and preservation hashes. All 170 implementation files identified by the earlier response validation and all 43 snapshotted historical/evidence files remained byte-identical. The earlier 141-test result and CPU fixture are retained as historical software validation; this documentation edit does not relabel them as a new test run or as H100 scientific evidence.

The local proposal and figure gallery work without a network connection. The deferred video viewer still requires its separately stored original videos; packaging its HTML does not package those videos. No HAIC jobs were launched, no remote files were transferred and no experimental outcomes were generated during this revision.
