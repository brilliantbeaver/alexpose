# Final visual verification

Actually inspected the rendered PNGs with `view_image` after the readability revisions:

- [Proposal 1 mechanism](../../figures/previews/01-mechanism.png)
- [Proposal 2 mechanism](../../figures/previews/02-mechanism.png)
- [Proposal 6 mechanism](../../figures/previews/06-mechanism.png)
- [Proposal 1 experiment](../../figures/previews/01-experiment.png)
- [Proposal 2 experiment](../../figures/previews/02-experiment.png)

**Verdict: all five pass. No further fix is required.** No clipped text, overlapping labels, crowded cards or misleading arrow intersections were visible.

Proposal 1 now labels the observed, prior and retained curves directly. The labels are separated from the lines, and the retained excursion makes the intended preservation behavior clear. Its experiment explicitly calls the threshold a development gate and includes matched observation tensors. The schematic footer prevents these curves and thresholds from being mistaken for measured outcomes.

Proposal 2 labels the candidate outputs “Low bend” and “High bend,” making their different conclusions concrete. Its experiment includes independently bounded controls. Both diagrams preserve the distinction between finding a checked alternative and proving uniqueness; unsuccessful search is not presented as evidence of uniqueness.

Proposal 6's “Support only” arrow runs above the cards from walking memory directly into the frozen forecaster. It does not pass through the query card or collide with the subtitle. The different-activity query enters separately, and the forecaster text explicitly receives both the walking memory and the same query prefix. No arrow suggests that query information was used to build the personal support memory.

This was a visual verification only. Proposal text and figure source files were not edited.
