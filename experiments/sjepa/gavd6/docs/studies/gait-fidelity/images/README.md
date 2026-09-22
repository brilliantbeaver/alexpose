# Figures and visual inspection

[Current proposal](../README.md) · [Interactive paper](../proposal.html) · [Open all figures](gallery.html)

The gallery displays all sixteen editable vector diagrams. The proposal uses six of them to explain its main argument; the others support the detailed data and method protocols. Diagrams describe proposed experiments or explicitly illustrative calculations, rather than new experimental results.

| Concept | Figures |
| --- | --- |
| Research question and controlled comparisons | [01 · Research question](01-research-question.svg), [02 · Crossed design](02-crossed-design.svg), [16 · Response error](16-response-estimand.svg) |
| Data, references and permitted claims | [07 · Video evidence](07-video-evidence.svg), [09 · Source partitions](09-source-splits.svg), [13 · Data to claims](13-data-to-claims.svg) |
| Feature prediction and masking | [03 · Changing graph masks](03-changing-graph-masks.svg), [04 · Mask contract](04-mask-contract.svg), [05 · Coverage audit](05-coverage-audit.svg), [06 · Matched masking experiment](06-matched-mask-experiment.svg), [14 · Paired JEPA](14-paired-jepa-method.svg), [15 · Graph-time mask](15-graph-time-mask.svg) |
| Inspection, ambiguity and validation | [08 · Data review display](08-data-review-view.svg), [10 · Assignment uncertainty](10-assignment-uncertainty.svg), [11 · Transformation contracts](11-transformation-contracts.svg), [12 · Evidence workflow](12-evidence-workflow.svg) |

`previews/` contains native-size and 900-pixel PNG renders for each SVG. `local-video-review/` contains local thumbnails used by the [video viewer](../data/video-gallery.html); these are inspection frames, not reference annotations.

The JSON layout files record geometry and automated overlap checks. Independent reviewers also inspected rendered figures; their findings and accepted asset hashes are in the [review index](../reviews/README.md). See the [reproduction instructions](../scripts/README.md) to rebuild diagrams, previews and galleries. Builders preserve descriptive SVG titles and alternative text so that the diagrams remain understandable outside the paper.
