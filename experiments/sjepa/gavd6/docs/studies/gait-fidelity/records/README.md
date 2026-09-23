# Organization records and historical snapshots

[Current proposal](../README.md) · [Review index](../reviews/README.md)

This folder preserves how the study was organized and which versions were checked. Read the current proposal and protocols for active choices.

| Record | Interpretation |
| --- | --- |
| [Transparent tutorial validation](transparent-tutorials-20260922.json) | Expanded mathematical lessons, complete kernel execution, independent review and exact prediction/metric comparison with the earlier fixture |
| [Notebook readiness after rename](notebook-readiness-20260922.json) | Fresh execution of all twelve notebooks from `notebooks/gait_fidelity`, local study tests, reconstruction and packaging preview; live HAIC authentication was unavailable |
| [Python package consolidation](package-consolidation-20260922.json) | Moves the implementation into the single `gait_fidelity` package, removes the import bridge and verifies packaging and execution; earlier receipts retain their historical paths |
| [Implementation validation](implementation-validation-20260922.json) | Source and workflow hashes, local tests, all twelve executed notebooks, reconstruction checks and untested HAIC requirements |
| [Tutorial execution receipt](tutorial-execution-20260922.json) | Per-notebook passing status and elapsed time for the fresh CPU software fixture; not source-study throughput |
| [Eight-H100 compute plan](compute-plan-20260921.json) | Proposed recipes, seeds, shared-pretraining dependencies, dated capacity and budget assumptions; not a runnable launcher |
| [Compute-plan validation](compute-validation-20260921.json) | Independent matrix review, budget/deadline arithmetic, links, new figure and preservation checks |
| [Organization map](organization-20260921.json) | Old and new file locations, with byte hashes immediately before and after moves |
| [HAIC availability validation](availability-validation-20260921.json) | Subsequent availability clarification, rebuilt paper, links and preservation checks; no live HAIC audit |
| [Paper organization validation](paper-validation-20260921.json) | Earlier link, image, interactive-calculation and preservation checks for the reorganized study |
| [Previous full synthesis](history/proposal-before-organization-20260921.md) | Historical proposal retained for technical depth and exploratory alternatives; links rebased to current files |
| [Initial validation](history/validation-20260919.json) | Original proposal receipt, preserved byte-for-byte |
| [Initial relocation](history/relocation-20260921.json) | Move into an independent research direction |
| [Direction rename](history/rename-20260921.json) | Shortening the direction name to Gait Fidelity |
| [Masking/video extension validation](history/extension-validation-20260921.json) | Earlier source-audit and twelve-figure receipt |

Historical JSON records contain the paths and hashes valid when they were written. Their content is deliberately preserved; the organization map explains subsequent moves. A historical hash mismatch caused by a documented later edit is not silently rewritten as a fresh validation.

On 22 September 2026, the notebook directory was renamed from `notebooks/gait-fidelity` to [`notebooks/gait_fidelity`](../../../../notebooks/gait_fidelity/README.md). Active documentation, notebook imports, generation and execution scripts, tests and the HAIC release packager now use the underscore spelling. Earlier validation records retain the original paths and hashes; notebook setup cells changed to import helpers from the new location.
