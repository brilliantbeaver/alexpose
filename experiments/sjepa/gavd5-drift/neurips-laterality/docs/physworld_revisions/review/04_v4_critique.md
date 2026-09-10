# Review of revision 4 → decisions for revision 5

The independent core and extension reviewers agree that the central numerical claim is now defensible, but identify remaining gaps in implementation precision, evidence status and the workshop argument.

| Review finding | Action in v5 |
|:--|:--|
| Abstract implies calibrated physical measurement | Use coordinate-derived movement contrast and initial-encoder readout. |
| Target collection and training tensor descriptions are ambiguous | Retain full grid, specify loss-selected valid targets and four three-coordinate observations. |
| Explicit laterality training is requested but only verbally described | Add the actual online-token penalty, denominator handling, gradient path, extra passes and synthetic-only status. |
| Notebook 12 sounds freshly reproduced | Identify its retained-summary status and unavailable raw directory. |
| Summary dimension/capacity confound remains incompletely explained | Define 960 versus 2,890 features and motivate support-only and capacity-controlled comparisons. |
| Latest recipe is obscured by historical settings | Add actual .5 budget, region extent, EMA, temperatures and BF16/FP32 scope. |
| Clinical and geometry concepts need visual explanation | Embed the compact training pipeline, latest results and schematic reflection figure. |
| World-model relevance lacks a decisive next test | Require joint improvement in observable utility and geometric behavior; specify pilot source exclusion and independent confirmation. |
| Synthetic forecasting deserves an honest failure illustration | Add an appendix with observed-versus-predicted future decoding, actual tiny scope and no real-data claim. |
| Reconstruction conventions omitted | Add timestamps, coordinate units, pelvis fallback, body scale and epsilon in an appendix. |

This version addresses the reviewers' substantive writing requests. Research requests that require new training or a clinical dataset become explicit future tests rather than invented completed results.

Reviewer score for v4: 75.0/100. The paper now needs editorial selection: its cumulative form contains too many tables and too much historical context for the intended main-paper space.

