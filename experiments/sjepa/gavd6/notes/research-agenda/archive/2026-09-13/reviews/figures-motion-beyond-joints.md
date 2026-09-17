# Visual review of the optical-flow replacement

Reviewed the current Proposal 7 prose and actually opened these rendered images with `view_image`:

- [07 mechanism](../figures/previews/motion-beyond-joints-mechanism.png)
- [07 experiment](../figures/previews/motion-beyond-joints-experiment.png)
- [Portfolio ranking](../figures/previews/proposal-comparison.png)

This review concerns the new optical-flow proposal. The previous constraint-conflict diagrams are archived and are not the current Proposal 7.

## Layout

All three images pass layout inspection. No clipped text, overlapping labels, arrow collisions or crowded paragraphs were visible. The four-card mechanism and three-card experiment are legible at full size. Card margins, numbering and arrow spacing are consistent with the rest of the portfolio. The portfolio's seven rows align cleanly, including the longer Proposal 7 title.

The rotating-surface drawing communicates the distinction between fixed joint endpoints and moving material. The probe illustration does not imply a new physical sensor. Flow-token counts are readable. The budget comparison includes zero tokens in the experiment figure, and the experiment explicitly counts full extraction cost.

## Conceptual checks and requested wording corrections

The mechanism appropriately requires actual input equality, holds the final image fixed, tests untouched movement and demands an advantage over simple flow fusion. Its footer correctly states that flow comes from the same pixels and is neither independent ground truth nor a unique 3D explanation. The diagram is explicitly labeled as a schematic with no experimental outcomes.

Two wording corrections were requested from the figure author:

1. **“No texture: no evidence” is too absolute.** A textureless body can still change its silhouette or shading. The intended control is an observation where the chosen movement has no distinguishing image evidence. Suggested replacement: “Textureless ambiguity tests.”
2. **“Forecast real motion” and “Test untouched recordings” could imply measured surface truth.** AMASS contains fitted body-model motion; its rendered surface trajectories are model-derived references. Suggested replacements: “Forecast untouched motion” and “Test untouched AMASS motion,” with the model-derived-reference qualification preserved in the prose or a figure caption.

“Does joint guidance erase it?” is correctly posed as a hypothesis rather than a measured failure of H-MoRe. The stop rule also correctly rejects a synthetic-only effect and a result matched by ordinary fusion.

The portfolio now ranks the new Proposal 7 second and removes the old constraint reserve. That is the author's conditional prioritization, not established efficacy. The footer explicitly states that the ranking is not an acceptance probability and no proposed result has been measured. The new sequence, small P1 and P7 flow assays followed by a single flagship decision, is clearly communicated.

## Status

Layout passes. The two conceptual wording changes above should be rendered and checked before final delivery. No proposal text was edited during this visual review.
