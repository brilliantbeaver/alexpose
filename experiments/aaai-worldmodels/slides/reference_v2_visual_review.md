# V2 slide production and visual review

The visualization owner, `/root/v2_visuals`, completed this production review. This is an author check, not independent acceptance. The deck is ready for the coordinator's independent reviewer. No blocking visual issue remains in the checked export. A material change to the generator, displayed evidence, slide text, or page layout reopens the relevant review.

## Exact reviewed export

| Artifact | SHA-256 |
| --- | --- |
| `reference_v2.pdf` | `129c223109ac7468055543c19b9484a2f37767612f95f7e0d7cf7f8b448036f2` |
| `reference_v2_manifest.json` | `dd470e9763024f971d4054ca65bc07510b36f9d5d9a52204853121d0ba6952f6` |

The manifest pins the generator, five completed result-run inputs, observation audits, protocol/contracts, and generated slide assets. The additional export checks pin that manifest and PDF and record every inspected PNG identity in [reference_v2_delivery_checks.json](reference_v2_delivery_checks.json). The five result identities are the core, frozen external transfer, robustness, predeclared identity sensitivity, and separately identified post hoc mean-prior diagnostic. The original restart and iteration02 decks were preserved.

## Checks actually performed

The final generation command was run from the repository root:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_reference_v2_slides.py
```

The final generation completed in 19.7 seconds. It produced 40 pages with zero rendered text-container or plot-panel violations. It verified every file in each required completed run's hash manifest before using its results. [The numerical replay](reference_v2_numeric_checks.json) independently reconstructed the internal and external primary and matched-posture AUROC point comparisons from saved group rows, with maximum absolute discrepancy below `6e-17`. It also reconstructed all 38 normal eight-offset forecast means and four post hoc prior/reference means. Confidence intervals are taken from the verified result files rather than recomputed by the slide generator.

Every source PNG was inspected individually at its 2,000 by 1,125 export size during production. Changed pages were reopened after revision. The inspected content includes every cohort count, the frame boundary, joint map, fold diagram, real-mask panel, reference forecast, masking contrast, calibration diagram, internal endpoint, external endpoint, and sensitivity table. The final PNG and PDF contact sheets were also inspected. Every PDF page was rendered through PyMuPDF, and the final PDF versions of pages 19, 27, and 34 were inspected separately at full render size to check the central paired and ranking results.

The PDF export check used the existing `/tmp/medwm-docs-env/bin/python` utility environment with PyMuPDF. All 40 PDF pages have the intended 16:9 dimensions, contain selectable text, embed their fonts, and contain no raster image objects. All 40 SVGs contain editable text and vector shapes, with no embedded raster images. The exporter checked all 89 linked local evidence paths and compiled the HTML JavaScript with Node. These checks establish file validity, editable vector content, local link availability, and JavaScript syntax. They are recorded separately from visual and scientific review.

Browser runtime was not tested because this host exposes no browser automation tool and has no installed Playwright or Puppeteer. Keyboard navigation, responsive layout, and menu behavior therefore remain unverified in an actual browser. The deck provides the vector PDF as a standalone presentation.

## Issues found and resolved

| Severity | Evidence | Remedy and disposition |
| --- | --- | --- |
| Moderate | Some card headings and short-card bodies exceeded their declared text containers. | Text wrapping now uses rendered font measurements. Card body placement follows the actual heading extent. The final container check passes. |
| Moderate | Plot x-axis labels were too close to the full-sentence captions. | Chart panels were raised and their reserved bounds are now checked. The final charts have clear label/caption separation. |
| Moderate | A duplicate text legend crowded the lower mask-grid labels. | The duplicate legend was removed. The full caption defines all five observation states. |
| Moderate | Sampling or insufficiently marking a narrow empirical central interval can obscure the true zero-tail region. | The curve now uses exact sorted interval edges and midpoint values, includes midrank markers at the references, and points out the narrow zero-tail interval. |
| Minor | Log-scale minor tick labels crowded one another. | The reference-error panel uses five explicitly formatted decimal tick labels on its declared log scale. |
| Minor | Heel/ankle labels touched the schematic foot links. | The four relevant labels were moved away from joints and connecting lines. |
| Moderate | View-specific pooled AUROCs and the matched within-fold endpoint had different aggregation. | Each view row now names pooled scoring, and the caption separately identifies the matched within-fold contrast and retained population. |
| Moderate | Mean-pose controls were added after original outcomes were known. | The new comparisons use orange diamonds, a post hoc page label, a separate source identity, and an explicit outcome-exposure caption. They do not change the original detector selection or endpoint. |
| Moderate | Slide 9 used the overly broad term "normalization" in its future-isolation statement. | The table now names prefix normalization. The caption explicitly permits future-relative target normalization and evaluation support to change. |
| Moderate | Slides 15 and 16 initially colored all sixteen future joint rows as forecast queries. | Only the eight distal target rows are now navy. A distinct neutral state marks non-query future rows, the caption explains it, and assertions require exactly 64 queries. |
| Minor | The AUROC explanation omitted equal-score treatment despite coarse empirical ranks. | Slide 26 now gives half credit to tied scores. |

## Independent-review revision

The independent reviewer inspected all forty original pages and requested four corrections affecting pages 9, 15, 16, 25, and 26. The source generator was revised, all exports were regenerated, and those pages were reopened at full size. The refreshed PDF renders were checked separately. [The revision comparison](reference_v2_revision01_comparison.json) records the exact before/after PDF, manifest, generator, and PNG identities together with the original affected slide objects.

Only those five PNG pages and five slide objects changed. All empirical chart payloads, calibration residual values, and prefix mask patterns remain identical. The query diagrams now contain 64 navy joint-time cells and 64 neutral future cells for every policy. The exact empirical-step calculation retains its true zero-tail interval. The current export remains subject to the reviewer's exact-hash recheck.

The earlier per-SVG hashes and source JSON byte hash were not retained. Their comparison is explicitly unavailable, and no earlier identity was reconstructed or inferred. The original PDF/manifest/generator hashes, preserved source objects, and per-page PNG identities establish the scope of this revision.

## Scientific boundaries retained

The opening reports useful current-posture forecasting. The deck also retains the weaker richer-state forecast, every trained masking policy, geometry and motion controls, the stronger low-motion anomaly comparator, uncertainty conditional on saved fits, and frozen transfer with no target-domain fitting. M0 is explicitly an uncorrupted higher-information anchor; the other six policies share a 40% eligible-token removal budget. Future isolation refers to predictor inputs and outputs, while future-relative targets and scoring support may change for evaluation.

The internal 91 source videos and external 90 coded participants are not merged into one population. The deck states the small normal-source ceiling, unknown GAVD participant identities, label limitations, native-FPS and quality attrition differences, transfer covariate extrapolation, source-versus-participant averaging shift, and pre-score identity sensitivity. It does not claim neurological mechanisms, clinical diagnosis, disease progression, treatment response, or a detector advantage that the completed comparisons do not support.
