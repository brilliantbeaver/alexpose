# Independent visual review

Reviewed the actual rendered PNGs, not only SVG source:

- `images/previews/contact-mechanism.png` and `contact-experiment.png`, covering all fourteen proposal figures.
- Individual full-size previews of `01-mechanism`, `01-experiment`, `02-mechanism` and `02-experiment`.

Each individual preview is 1440 × 800. Pixel-bound checks on the four individually reviewed files place nonwhite content within approximately x=47–1393 and y=37–778, so no actual canvas clipping was found. Cards have adequate whitespace, text is legible, arrows do not run through labels, and the contact sheets make the repeated structure easy to compare. The issues below concern scientific meaning and conceptual clarity, not invented overlap defects.

## Required corrections

| File and area | Objection | Specific revision |
| --- | --- | --- |
| `images/02-mechanism.svg`, bottom teal message | It says an extra view rejects the alternative while retaining the true motion. Neither found candidate has to be the true one, and eliminating a pair does not establish uniqueness. This contradicts the revised protocol. | Replace with: “Reveal held-out evidence. Reject an inconsistent pair and search again; another explanation may remain.” |
| `images/01-experiment.svg`, first card | “Matched injected errors” does not show the new decisive event-by-noise factorial. A reader could infer that real motion and tracker errors appear only in separate clips. | Replace the three bullets with “Known reference motion,” “Real event + noise together,” and “Matched observation tensors.” If space permits, show a small 2 × 2 event/noise grid rather than another bullet list. |
| `images/01-experiment.svg`, second card and threshold box | “Same denoising quality” and “at matched error” leave test-time operating-point selection ambiguous. | Use “Calibration-locked repair level.” State that the 48-hour threshold applies to development people. The unseen event family belongs to the final test, not the 48-hour selection gate. |
| `images/02-experiment.svg`, final card | “Known 3D control cases” is too weak after the review exposed the near-threshold/noisy-3D problem. | Use “Independently bounded controls.” Replace “Reject false certainty” with “Reject pair; search again,” while keeping false-certainty evaluation in the text. |

## Clarity improvements

**`images/01-mechanism.svg`, cards 2 and 4:** The raw/prior/repaired curve identities are unlabelled. In card 2, the gray curve has the small bump and the blue curve is smooth, which makes it difficult to tell which is the observed event and which is the proposed repair. Add short curve labels and use the same mapping across both cards, for example blue “observed,” gray “prior,” and teal “retained.” The event bump should be attached to the observed curve before the gate. Retain the schematic disclaimer so this cannot be read as a measured result.

**`images/02-mechanism.svg`, card 4:** The boxes labeled A and B provide little explanation of the disputed finding. “Below threshold” and “Above threshold” would make the competing conclusions concrete without adding numerical results. Alternatively, put “low bend” and “high bend” beneath A/B. Avoid suggesting that the two different stick figures alone demonstrate identical projections; that property is established by the observation-fit checker.

**`images/06-mechanism.svg`, flow between cards 1–3:** A single chain from personal support through a different activity to motor memory can imply that the memory is inferred from the query activity. The method defines the personal memory from walking support only. A small two-input fork, with “walking memory” and “same query prefix” both entering a frozen forecaster, would show the information boundary more accurately. This is a conceptual improvement, not a layout defect.

**All experiment figures, heading:** The repeated heading “Proposal N: the experiment that can change the decision” is readable but long and less informative than the mechanism titles. Short concrete titles would improve scanning, such as “Repair noise without deleting movement” and “Find a checked competing explanation.” This is optional; current headings fit.

## Review conclusion

The vector layouts pass the clutter and overlap review at the inspected sizes. The four required corrections should be rendered again because they change scientific claims or the decisive experiment. The curve legend and two-input motor-memory flow would improve understanding without making the figures denser. After regeneration, preserve the distinction between conceptual diagrams and empirical results.
