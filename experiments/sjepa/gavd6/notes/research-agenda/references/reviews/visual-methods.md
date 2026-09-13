# Visual review of proposals 3 through 6

Reviewed on 12 September 2026 using the actual rendered PNG files, not only SVG source inspection.

## Images inspected

Both `images/previews/contact-mechanism.png` and `images/previews/contact-experiment.png` were opened with the image tool. This covers all eight diagrams for proposals 3 through 6. Individual full-resolution checks then examined `03-mechanism.png`, `05-mechanism.png`, `04-experiment.png` and `06-experiment.png`.

The contact sheets establish consistent reading order and comparison across proposals. The individual checks establish that the smaller explanatory text remains readable and that the more conceptually delicate graphics do not imply an unsupported result.

## Layout findings

No visible text overlap, clipped label, crossing connector or crowded caption was found in the inspected diagrams. The panels have sufficient white space. Numbered steps and left-to-right arrows give a clear reading order. The separate experiment diagrams make the early stopping decisions easy to locate.

The small line plots are explicitly marked as design schematics, so they do not present invented experimental outcomes. In proposal 5, the paired feature arrows and difference symbol communicate response matching adequately without equations. The mechanism diagrams are especially useful when read before the longer experimental sections.

## Required wording alignment

Two diagram statements need to follow the adversarial revisions in the proposal text. These requests were sent to the root author, who owns the diagram generator.

1. **Proposal 4 experiment gate.** Replace the phrase about a negligible mean-error gain with an explicit interval-based equivalence requirement. Suggested compact text: “48-hour gate: proper-score gain with mean-error equivalence supported by an interval, on real grouped motion.” A small point estimate does not establish that the mean contains no extra information.
2. **Proposal 6 experiment, third panel.** Replace “Unseen person and activity” with “Person held out from adaptation.” The base checkpoint may have encountered related people or activities during pretraining. The text now separates this exposure audit from the adapter split.

One optional clarity improvement is to add the short label “Target rank” above the `0`, `8` and `32` circles in proposal 3. Without this label, a reader must use the caption below to decode the numbers. It is not a layout defect.

## Remaining communication limit

The graphics explain workflows and decision gates. They intentionally do not establish that the proposed methods beat the strong baselines. That distinction is supported by the proposal text and the schematic disclaimer. Proposal 4's distribution picture should be read as an illustration of a scalar marginal, not as a demonstrated joint model of all body parts; the revised text now states that scope explicitly.
