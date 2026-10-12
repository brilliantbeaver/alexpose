# Full-writeup review dispositions

28 September 2026. The independent review is preserved in `writeup-adversarial-review.md`; the second-pass closeout is `writeup-final-review.md`. The reviewer did not author the report and did not modify its prose.

| Objection | Revision |
|---|---|
| Forecasting audit could compare the same operation twice | Hold the observed prefix fixed, alter or remove every future frame, and require all forecast-time inputs and forecasts to remain unchanged with random states fixed. |
| VICReg translation wording could imply independent joint jitter | Specify one sampled 2D translation per view, with x/y components uniform in the recorded normalized interval. |
| CHOIS venue and stroke-paper author initials | Correct to ECCV 2024 and Padmanabhan, P. |
| Operational instructions note in bibliography | Remove the agent-process language; retain the prior report's role as background/style reference. |
| Ambiguous seed averaging in Figure 4 | State that three fitted seeds are averaged within each of fourteen people. |
| OpenCap Monocular stages compressed inaccurately | Separate optimization of WHAM estimates from constrained biomechanical modeling, simulation, and learning. |
| Evaluation support and failure triggers implicit | State shared reference support across source-window variants; retain nonfinite/short-segment predictions as failures. |
| External validity and retained assets insufficiently explicit | State no completed GAVD/natural-video evaluation and absence of complete trajectories/checkpoints in the retained packet. |

Additional production edits corrected exact S-JEPA and OpenCap Monocular reference titles, retained anonymous authorship for the source manuscript, explained all eight model families, and placed the restoration figure immediately after its first result paragraph. Visual inspection corrected missing norm-bar and superscript-minus glyphs. The six empirical figures preserve the previously audited estimates and intervals.

The reviewer rechecked the revised source and reported **no unresolved substantive blocker**. This is a prose/evidence review, not a new experimental replication. PDF page rendering and structural checks are separately recorded in `../qa/writeup/VALIDATION.md`.
