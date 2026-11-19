# Independent review of slides 1 through 15

Reviewed the current deck specification, its structured notes, contact sheets 01 and 02, and enlarged renders for slides 3, 5, 8, 10, 11, 12, 13, 14 and 15. Compared the claims with `docs/draft03/manuscript.md` and `technical_details.md`. The reviewer did not author these notes. No deck content was edited.

Reviewed identities:

- `deck_spec.json`: `965d8a236d3ab7b91253e009833d1e816dbd840d3382ed35819a6b74502c8ef4`
- `tutorial.pdf`: `b121a5100080869010838d42ac1c356f378cf31b02fa4cdbf64253e365225505`
- `source/notes_01_15.json`: `267ce59f3cb511f87be6c20414e6ebf8a0d9629d7e4a2efdb945679dd0856a8c`

The material findings concern note accuracy and agreement with the visible slide. I found no incorrect population totals, changed scientific results, clinical diagnosis claims, clipped text or overlapping labels in this range.

## Corrections needed

1. **Slide 4, content ID 5, evidence paragraph: distinguish preserved timing from unchanged sample times.** The notes say the video representation “keeps the original sampling times.” The executed pipeline resamples at 15 Hz, as slide 9 and the manuscript correctly explain. Replace this with wording such as “preserves elapsed time on a shared 15 Hz sampling grid, without independently aligning each leg's walking cycle.” This preserves the intended scientific point without implying that original video samples are used unchanged.

2. **Several visual-reading paragraphs still describe a different diagram.** These passages will misdirect a presenter or a reader following the slides and notes together. The underlying explanations can remain, but the descriptions should identify the elements actually displayed:

| Slide position | Content ID | Mismatch | Concrete correction |
| --- | --- | --- | --- |
| 1 | 1 | The visual paragraph asks the audience to read “dataset summaries.” The slide instead shows walking measurements, controlled prediction and anomaly detection. | Describe those three stages; introduce the two dataset summaries on slide 5. |
| 4 | 5 | The visual paragraph describes a separate stable-compensation panel. The actual panels are Posture, Movement size and Shared timing. | Name those three displayed panels. Stable compensation can remain part of the explanatory paragraph. |
| 6 | 6 | The visual paragraph refers to “prior-work panels” about normal-gait modeling, dependence and anatomical masks. The slide only contains Predict observations, Predict features and Our main comparison. | Guide the reader through these displayed prediction targets. Keep the cited prior work in the evidence paragraph. |
| 9 | 9 | The visual paragraph describes consequences of longer intervals and the population surviving them. The actual panels show an overlapping window, a four-sample token and distinct target support. | Explain those displayed units and their distinction from independent evaluation cases. |
| 14 | 14 | The notes say teal is the structured core and blue is random filling. Every hidden cell in the actual figure is teal; blue filling is not drawn. | State that teal means hidden and pale means visible before validity checks. Alternatively, add a genuine two-color core/fill rendering while preserving the masks. |
| 15 | 15 | The notes ask the audience to multiply joint count by time-block length “across each row.” That anatomical-group table is not shown. The actual columns are Training mask, Hidden tokens and Chance per token. | Explain that all displayed rows have 64 hidden tokens and a 50% chance per token. Keep the eight-token-core derivation as spoken explanation rather than pointing to nonexistent columns. |

Slide 14's actual masks are valid illustrations: the implementation uses the declared time-by-joint layout, all three rendered grids contain 64 hidden cells, and Whole joint contains a complete eight-cell hidden joint row. The problem is the color explanation, not an incorrectly generated mask.

## Checks that passed

- Structured note fields match the text embedded in the current deck specification for all fifteen slides. Paragraphs use connected full sentences, and transitions follow the actual slide order, including the transition from slide 15 to the fixed evaluation tasks on slide 16.
- The main GAVD population has 90 recordings: 28 normal and 62 abnormal, with condition counts 10, 15, 10 and 27. The broader population has 304 recordings, including 28 normal and 276 abnormal. The separate five-class count is correctly 89.
- The laboratory count is correctly 178 participants, comprising 128 healthy and 50 stroke participants. The notes distinguish recording uploads from identified people and do not pool the datasets.
- The 32-sample span is correctly 31/15 seconds, about 2.07 seconds. One token is four samples of one joint; repeated masks and overlapping windows are not described as new participants.
- The 16-joint index set, eight lower-leg readouts and four-fold normal roles of 16 fitting, five reference and seven test recordings agree with the manuscript. Prior use of the normal recordings is disclosed.
- The own-leg and both-leg corrections predict the same hidden answer while sharing a fixed baseline. Equal feature width is explicitly described as a partial control, with unequal numerical rank and the shared body-reference qualification preserved.
- Clinical motivation remains conditional and non-diagnostic. Myopathic gait is connected to muscle disease rather than described as a single central neurological lesion. The main coordinate/angle experiment is correctly distinguished from the separate JEPA experiment.

## Optional improvements

- On slide 5, “Group composition” would describe the final table column more precisely than “Measurements,” since its entries are category counts rather than sensor measurements.
- On slide 13, append “with the body reference fixed” to the restricted-input explanation. Slides 10 and 12 already state this qualification, so this is a useful local reminder rather than a missing qualification across the deck.
- In the slide 10 notes, clarify that the 16-joint tensor defines masking and overall quality checks, while the restricted coordinate predictors use the permitted leg observations conditional on the shared body reference. This would prevent “additional input landmarks” from being read as unrestricted access to arm observations.

These revisions require no new experiment and no change to the reported scientific values.
