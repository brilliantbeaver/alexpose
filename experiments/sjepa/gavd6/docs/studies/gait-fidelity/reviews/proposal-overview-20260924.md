# Full proposal and two-page overview review

**24 September 2026.** Reviewed the coordinated [full proposal](../proposal.html), its [Markdown source](../README.md), and the independently distilled [overview](../proposal-brief.html) / [two-page PDF](../proposal-brief.pdf). The [validation receipt](../records/proposal-overview-validation-20260924.json) identifies the final artifacts. This is a documentation and evidence-interpretation review, not validation of new model results.

## Independent work and adversarial review

Three independent agents worked in parallel, then reviewed the revised documents in further passes:

- **evidence_narrative** checked the research argument, equations, implementation, core CSVs, amended protocol, and the consistency of the full and brief versions. Its final adversarial pass found no remaining material scientific defect.
- **real_images_sources** verified primary-source connections to OpenCap and Stanford AmI, selected licensed photographs, preserved their provenance, and reviewed narrative clarity and the rendered two-page PDF.
- **research_visuals** built source-derived vector diagrams and the results chart, checked labels and geometry, corrected the older crossed-design figure, and independently inspected both rendered PDF pages.

The author integrated those findings and separately checked the source, browser rendering, mathematical notation, PDF pages, and cross-document claims. Reviews were used to revise the argument, not simply to approve its prose.

## Substantive issues resolved

| Issue raised | Revision |
| --- | --- |
| The experiment sequence could read as a protocol without a measurement motivation. | Both versions begin with distinguishing physical change from observation and processing error, then define the movement response and the scientific question. |
| The follow-up could appear to identify the cause of the core's failure. | State the pretraining hypothesis explicitly; explain the two-readout comparison while acknowledging that it cannot uniquely diagnose the core tradeoff. |
| A downstream null contrast was interpreted too broadly as lack of useful representation information. | Limit the conclusion to the unresolved response-error advantage over the initialized-encoder control. |
| The strongest positive result lacked nearby uncertainty and exploratory status. | Both versions give the exploratory 5.144° improvement and descriptive 95% interval [2.152°, 8.651°]. |
| The full proposal omitted useful experimental scope. | Restore interval length, frame rate, estimator/view counts, and the 15° edit withheld from training; distinguish commanded edits from projected responses. |
| Failures could be mistaken for excluded observations or the 720° cost for a physical angle. | Explain the all-attempted scoring and failure decomposition; label 720° as a fixed scoring penalty in the full version. |
| A schematic said core results were pending. | Change Figure 2 to “INHERITED DESIGN / SCHEMATIC,” retaining its illustrative status. |
| The method figure could imply identical learned features across separately trained arms. | Describe matched joint–time positions; the full text explains that teachers evolve separately. |
| “JEPA · coordinates” and the brief's third control were underspecified. | Clarify the coordinate-only readout label and name the coordinate-difference control in the brief. |
| Real photographs could imply Stanford deployment or study participation. | Attribute each directly, describe Rush's structured residential test, retain published privacy edits, and provide linked CC BY 4.0 licenses and provenance. |

The numerical review confirmed the declared core JEPA contrast of 0.688° [−0.640°, 1.977°], its 3.272° waveform deterioration, and the initialized-encoder contrast of 0.093° [−1.394°, 1.393°]. The new feature-difference experiment remains proposed; no follow-up success is asserted. Neither version claims clinical validity, future-state dynamics prediction, or adoption by a Stanford program.

## Rendering and consistency checks

The overview has exactly **Introduction, Methodology, Experiments, Results, Discussion**. Its PDF has two US Letter pages, 10.5-point body text, two vector research figures, and two credited contextual photographs. Both pages were rendered at 1500 pixels and independently inspected for legibility, overlap, clipping, photo proportions, and quantitative clarity. Final renders are retained as [page 1](../images/previews/proposal-brief-page-1.png) and [page 2](../images/previews/proposal-brief-page-2.png).

The full proposal renders 30 LaTeX expressions, including nine display equations, as offline SVGs. Neither scientific source uses the prohibited macro. The loss definitions, difference identity, stop-gradient notation, common coordinate scale, and gradient calibration were checked against the implementation. All symbols are explained in the main text or technical details.

Local Chromium checks at 1440 × 1000 and 390 × 844 found no page overflow or broken images. Longer equations scroll within their own containers on narrow screens. Figure enlargement, closing, both numerical explorers, and their expected arithmetic passed. The brief's five headings and images loaded correctly at both viewport sizes, with no console errors. The final local-link and artifact checks are recorded in the validation receipt.

## Remaining scientific limits

These edits use the completed core's aggregate export, which lacks raw predictions and full fit/cohort receipts. The 14-person development population is reused, projected angles are two-dimensional, and the movement edits are kinematic stress tests. The documents preserve those constraints and motivate independent reference-based validation. New scientific claims still require completed follow-up predictions and evaluation.
