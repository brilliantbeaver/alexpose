# Manuscript and abstract development

[Study home](../README.md) · [Executed analyses](../results/README.md)

Start with [abstract version 10](abstract-v10-expanded-results.md), the latest retained draft, and its [review](abstract-review-v10.md). The [version history](abstract-versions.md) explains all ten versions and their evidence boundaries. [Version 09](abstract-v09-study-scope.md) preserves the earlier study-scope framing.

## Drafts, evidence and reviews

| Material | Entry points |
| --- | --- |
| Abstract drafts | [Version history and links to versions 01–10](abstract-versions.md) |
| Original rubric and evidence | [Evidence and rubric](abstract-20260919-evidence-and-rubric.md), [scores](abstract-20260919-scores.json), [score table](abstract-20260919-scores.csv) |
| Predecessor-study audits | [Original synthetic study](abstract-original-study-audit-20260919-v01.md), [laterality evidence](abstract-laterality-audit-20260919-v01.md) |
| Literature and positioning | [Literature audit](abstract-literature-positioning-20260919-v01.md) |
| Later draft reviews | [Version 09](abstract-review-v09.md), [version 10](abstract-review-v10.md), [version 10 review data](abstract-review-v10.json) |
| Shared conceptual figure | [SVG](images/abstract-training-pipeline-20260919-v01.svg), [PNG](images/abstract-training-pipeline-20260919-v01.png), [PDF](images/abstract-training-pipeline-20260919-v01.pdf), [generator](images/build-abstract-pipeline-20260919-v01.py) |

Version numbers and dated filenames are retained so existing drafts remain distinguishable. Earlier drafts describe the evidence available at the time and may contain statements corrected in later versions.

## Historical arithmetic and validation

The [exploratory arithmetic summary](abstract-20260919-exploratory-summary.json) and [validation receipt](abstract-20260919-validation.json) preserve the earlier rounded-table calculations and writing checks. Their recorded hashes refer to the files at the time of that audit. The [organization record](../development/organization-20260920.json) records subsequent path and link changes without replacing those receipts.

The early [recomputation script](abstract-20260919-recompute.py) and [validation script](abstract-20260919-validate.py) depend on the former results-bearing version of `proposal/README.md`. The current [proposal](../proposal/README.md) contains only study design, so it cannot reproduce that historical table or match its recorded hash. Older drafts' references to a source summary in the proposal refer to that former content. Use the retained exploratory summary for those historical ratios and the [results guide](../results/README.md) for the later diagnostic evidence; a full rerun of the early audit requires its original source revision.
