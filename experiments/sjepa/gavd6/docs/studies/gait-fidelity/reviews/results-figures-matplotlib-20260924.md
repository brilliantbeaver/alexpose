# Adversarial review of the Matplotlib results graphs

**24 September 2026.** Reviewed the [full results graph](../images/research-core-results.svg), [compact graph](../images/research-core-results-compact.svg), their proposal captions, and their final browser/print rendering. The [builder](../scripts/build_results_figure.py) reads the retained core CSVs; no experiment outputs were changed.

## Design and review findings

The previous chart repeated five method/objective labels, ten three-decimal values, and cohort information inside the image. Its two axes used different scales. The replacement groups Direct and JEPA, connects their two restoration objectives, and uses a common 0–25° scale in both mean panels. Objective is encoded by color and marker shape. Direct labels use one decimal; exact values remain in the text and source CSVs. Dashed references show unchanged pose estimates.

The full version includes a separate paired-contrast panel. The compact version shows the same mean data at a 7.5 × 2.5-inch size, with uncertainty described in the adjacent prose. Cohort and methodological qualifications are in captions rather than a long in-figure footer.

Two independent agents reviewed the work in successive passes:

- **evidence_narrative** checked the source means, contrast identities, interval choice, rounding, and visual interpretation. It verified that the declared response contrast is −0.688° [−1.977°, +0.640°] and trajectory contrast +3.272° [+1.944°, +4.699°], when plotted as JEPA minus direct. These descriptive crossed person/seed intervals belong to paired differences, not individual method means. The response interval crosses zero. The final technical review passed.
- **research_visuals** inspected both figures and the rendered second page of the brief. It identified the risk of nearly overlapping JEPA response points on a same-height dumbbell; the implemented categorical vertical offsets keep the actual x values and their small difference intact. A subsequent review caught numbers masking the dashed baseline. Labels were moved away from the baseline, with a two-point adjustment where needed to clear a connector. Final independent inspection found no remaining clutter, clipping, or overlap requiring revision.

The author additionally removed unnecessary background grid lines so they cannot cross number glyphs. The final labels do not require white boxes to conceal references or connectors. Intentional intersections between data lines and reference lines remain part of the graph's meaning.

## Validation

The Matplotlib builder measures actual rendered text extents and checks every value label against the reference and connector paths. Both outputs pass with zero text/text overlaps, zero text outside the figure, and zero connector/reference intersections with value labels. Values and intervals are recorded with source hashes in [the build receipt](../records/results-figure-build.json).

Both final HTML proposals were inspected in local Chromium at 1440 × 1000. The SVGs render with readable labels and continuous lines. The regenerated brief remains exactly two pages, with vector figures; its final PDF pages were rendered at 1500 pixels and inspected. The compact plot fits without crowding the caption or discussion. The [validation receipt](../records/results-figure-validation-20260924.json) records the final assets and checks.

Captions identify completed core findings, 14 development participants, three seeds, unchanged-pose references, and the distinction between JEPA readout training and pretraining. The full caption explains the JEPA-minus-direct sign convention, which is the reverse of the positive-improvement convention in the original CSV. The proposed feature-difference follow-up remains unreported. The redesign changes presentation, not the study's conclusions.
