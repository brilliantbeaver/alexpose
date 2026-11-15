# Review resolutions

The slide rewrite keeps the measured results and scientific comparisons intact. The reports in this folder identify the exact versions each independent reviewer assessed. Subsequent presentation builds have different hashes, which are recorded in the final build and validation reports.

| Review finding | Applied correction |
| --- | --- |
| Sampling times were described as unchanged. | The notes now explain the shared 15 Hz grid and distinguish it from independently aligning each leg to a walking cycle. |
| Several notes described panels, arrows or colors that were not displayed. | The visual-reading paragraphs now describe the actual cards, tables, mask colors and percentage-reduction bars. |
| Aggregation could be read as averaging coordinate predictions. | The slide and notes now state that prediction errors and benefits are averaged across repeated masks before higher-level aggregation. |
| The masking comparison could imply unchanged correction weights. | The notes distinguish fixed predictor design and evaluation targets from refitted correction weights. |
| The timing table omitted its training policy. | The visible introduction and notes identify Random U and the 27 normal test recordings. |
| Laboratory score adaptations were not explicit. | The notes and score guide distinguish 12 laboratory descriptors and 20 reference participants from 16 video descriptors and five reference recordings. |
| Reference scaling could be confused with a disease probability or a raw benefit. | The score guide explains both corrected predictors, the signed error benefit, normal mean and standard deviation, two-sided departures, and fixed averaging. It states that the scores are not disease probabilities. |
| Per-slide feedback created 43 live status regions. | Ordinary visible status labels now accompany one shared live region, which announces only the action taken. |
| Unselected control boundaries were faint. | Both themes now use control boundaries with measured contrast above 3:1 on their relevant backgrounds. |
| SVG text fell back to a serif font in the browser. | The light and dark SVGs now embed subsetted regular and bold fonts, with their license and hashes retained. |
| Existing accent text fell just below 4.5:1 on a pale background. | The active export darkens that text slightly while preserving the historical drawing helper and archived decks. |

All material findings were accepted. Optional refinements concerning the shared body reference, table headings, score arithmetic, shared left/right reference identities and within-fold AUROC were also applied. No criticism was rejected on scientific grounds. No model was retrained and no outcome was replaced during this presentation revision.

The final [browser report](../../review/viewer_validation.json) records actual interaction checks. The [visual review](../../visual_review.md) describes rendered-page inspection and its limits. These checks do not replace a full assistive-technology audit.
