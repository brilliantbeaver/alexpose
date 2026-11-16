# Independent review of the restart tutorials and slides

The six executed restart notebooks and the 22-slide teaching deck pass this artifact review after the corrections listed below. Their reported numbers agree with `restart-dynamics-v1`. The review checks execution records, generated numerical content, rendered slide layout and interpretation. It does not independently validate clinical diagnoses or patient identity.

The reviewer did not author the notebook, execution or slide generators. The reviewer previously authored the data extraction adapter, so independence applies to these tutorial and visualization artifacts, not to the underlying data preparation. No parent-owned artifact was edited by the reviewer. Findings were returned to the author, who changed the generators, regenerated the slides, and re-executed the notebooks.

## Execution and numerical verification

All six notebooks completed in independent kernels, with five code cells each and 30 executed code cells in total. Every executed cell has an execution count, and there are no notebook error outputs. Each current source notebook matches its recorded source SHA-256. Each executed notebook matches its output SHA-256 and retains exactly the current source cells. All five recorded input identities match the actual cohort, preparation and model-result artifacts.

Notebook 11 records a fresh Ambient/MediaPipe call and 33 detected joints. Notebook 13 performs an actual ridge fit on 120 real normal windows and labels its error as training error. It does not present that didactic fit as held-out evidence. Notebook 13 also reconciles 2,818 prepared windows with 2,815 model windows and three common-support exclusions. Notebook 14 explains that the eight-sample error averages the nominal future samples from 0.1 through 0.8 seconds, rather than measuring only the final endpoint.

All seven plotted forecast estimates and their intervals, all five displayed paired contrasts, every displayed anomaly AUROC and interval, and the full model-cohort table match the saved result JSON. The 22 SVG hashes and the current generator hash agree with the slide manifest. The review's reproducible checks and exact final artifact identities are saved in `artifacts/restart_visual_review/check_artifacts.py` and `notebook_and_numeric_checks.json`.

The accepted numerical interpretation is that bilateral history reduces normal source-average forecast MSE by 33.5% relative to persistence, but the learned current-pose control has lower error: 0.07164 versus 0.08063. The static-input control is a learned future transition conditional on current pose, not a constant forecast. Shrinkage does not improve the primary bilateral fit. The excess-error AUROC of 0.793 is a separate exploratory annotation-ranking result. Neither it nor the matched timing-control difference establishes neurological causation or clinical utility.

## Findings and verified corrections

| Initial finding | Correction and verification |
|---|---|
| Several flow labels extended into adjacent boxes and arrows. Slide 15 was especially difficult to read. | The generator now uses shorter wraps and smaller flow-label text. The full contact sheet and individual slides 04, 05, 07, 09, 12, 15, 20 and 22 were reinspected. Labels fit their boxes. |
| The bilateral schematic used separate boxes whose labels still crossed borders after the first global fix. | Slide 11 now uses explicit wraps for own-leg history, opposite-leg history, predictor and target labels. Its corrected individual render was inspected. |
| Forecast and paired-effect chart ticks and axis labels overlapped the footer. | Chart axes were moved upward and axis-label font size was reduced. Individual slides 16 and 17 now have clear separation. |
| The static model's superior result and the unsuccessful shrinkage sensitivity were visible only indirectly. | Slide 16 now states that the current-pose learned transition forecasts best and gives the actual MSE values. Slide 13 explicitly states that shrinkage did not improve the unshrunk bilateral result. |
| The horizon label could be mistaken for error at the final 0.8-second endpoint. | The slide explanation and executed notebook 14 now describe the average across all eight nominal future samples. |
| The teaching sequence did not explicitly reconcile prepared and modeled window counts. | Executed notebook 13 prints 2,818 prepared windows, 2,815 model windows and three support exclusions. Notebook 14 also explains the exclusion. |
| The coordinator's subsequent PDF text-boundary check found that the slide 05 title extended beyond the page. | The author shortened it to “Full-duration processing expands movement coverage.” The reviewer inspected the corrected slide 05 render and reran all numerical and artifact-identity checks. The title now fits within the page. |

The complete 22-page contact sheet was inspected. Key individual pages were inspected at full resolution, including the previously failing flow diagrams, bilateral schematic, shrinkage explanation, forecast plot, paired contrasts and anomaly table. The corrected images show no remaining text collision in the inspected pages. The diagrams, tables and estimates are editable SVG content generated from the current source and run evidence.

## Scope and integration limits

The review inspected the generated PNG slide renders and SVG content, together with the combined PDF's recorded hash. It did not independently rasterize the combined PDF or test keyboard navigation in a browser. Those behaviors should not be described as separately validated by this review. The HTML and PDF use the same slide generator, but that shared provenance is not a substitute for a separate viewer test.

The final integration-link check confirmed that `docs/draft01/README.md` exists, closing the earlier pending submission-readiness link. Scientific limitations remain explicit: prior dataset exposure makes this exploratory work, sources are not verified patients, and fixed-fit source bootstrap intervals do not establish clinical transportability.

Result input SHA-256: `f14a6e533338c96da99b5085cb231b9de5cc47f7fe50a4c118225c42dc1fa1e8`.

The exact notebook record, slide-manifest and combined-PDF identities are retained in the machine-readable review record, so later material changes reopen the corresponding checks.

## Final artifact update

After the slide 05 title correction, the complete artifact checker passed again. Only the corrected slide 05 was visually reinspected in that pass. A final typography update then replaced PDF Type 3 fonts with embedded TrueType fonts. The artifact checker passed once more against the final manifest and PDF. The result input, slide numbers and story, and all notebook identities remain unchanged; unchanged slide layouts were not repeatedly reinspected.

| Artifact | Final SHA-256 |
|---|---|
| Slide manifest | `280d22eb0cb16a0a2af6e50b7d3d84c4c94c0dcf5c25833586fd69d40c8d1e0b` |
| Combined slide PDF | `c4362779107d9ce8827bcaee4fb83b4c8ecf9bd1b4a7d29a3a81c9fb6fdaaa80` |
| Notebook execution record | `63f4b29ba80d0c7c00205537c85712eb518a1e23553d8cebb05a47bd5b441498` |

The coordinator separately rasterized the combined 22-page slide PDF and final six-page manuscript. That distinct check is recorded in `docs/restart/pdf_validation.json`; it reports no off-page text and no Type 3 fonts for either PDF. The final slide PDF identity in that record matches the reviewer-pinned identity above. The PDF rasterization check is not attributed to this reviewer.
