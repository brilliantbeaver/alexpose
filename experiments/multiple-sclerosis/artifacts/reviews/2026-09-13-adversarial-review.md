# Adversarial review and revisions — 13 September 2026

This record accompanies the updated slide tutorial and `docs/08-0913-PROGRESS.MD`. The review concerns the accuracy and clarity of the documentation. It preserves the notebooks, checkpoints, and historical evaluation outputs.

## Review process

Independent agents inspected the notebook executions and implementation while the report and vector illustrations were drafted. Their findings are recorded in [the notebook evidence inventory](2026-09-13-notebook-evidence.md), [the methods review](2026-09-13-method-evidence.md), and [the independent report review](2026-09-13-report-review.md).

The installed `codex:adversarial-review` companion was also invoked with `--wait --scope working-tree`, explicitly focused on scientific reasoning, retained results, source separation, transformations, symmetry claims, uncertainty, and correspondence between code and evidence. Its review-only operation was kept separate from the author's revisions, which the user had requested in advance.

## Revisions from the evidence and report reviews

| Finding | Revision |
|---|---|
| The frozen 0.438 S-JEPA result predates the once-per-batch centering correction. | Added the distinction early in the report, next to results, in the protocol diagram, and in slide explanations. A five-fold evaluation of the current objective remains outstanding. |
| Notebook filenames and old introductions describe normal-only training and progressive VICReg. | Explained current executable behavior: fold-local, all-condition label-free pretraining; another stage from saved weights; labels in frozen probes. |
| Notebook 04 reloads weights without restoring its optimizer/center/schedules. | Described a warm start with restarted state, separating it from an uninterrupted 1,200-update run. |
| Notebook 05 retains a checkpoint error and no downstream plots. | Reported the failure, noted that checkpoints now exist, and omitted invented clusters or silhouette findings. |
| R1, 800+400-update probes, and live 500-update demonstration answer different comparisons. | Gave each its own scope and denominator. Verified that frozen and live RF fold-0 scores agree. |
| “Exact dataset shape preservation” would exceed what processing guarantees. | Distinguished within-frame angles/length ratios, lossless tensor rearrangement, changes from interpolation/padding, removal of normalized root travel, and unconstrained learned feature geometry. |
| Source separation does not establish participant independence. | Defined source IDs, retained clip/source/frame/window counts separately, and described clip-weighted evaluation and repeated development use. |
| Some control names misdescribe their inputs. | Explained duplicated retained-frame count and raw pixel distances; retained both classifiers' results; avoided treating strong controls as causal proof of camera shortcuts. |
| The strongest full-table point estimate is 0.703, not the RF branch's 0.667. | Corrected the comparison and reported the 0.036 difference without an unsupported significance claim. |
| RF and S-JEPA have different principal PD errors. | Corrected confusion interpretation and displayed both count matrices. |
| The baseline feature schema overstates the measurements actually populated. | Explained the 15 core feature slots, the current duplicated ankle-range assignment, and the absence of angle-specific attribution. Historical scores remain unchanged. |
| Effective rank is measured on a centered batch of 32. | Stated the ceiling of 31 and limited interpretation to evidence against constant measured representations. |
| A generic 1/3 chance macro-F1 and an unverified historical bootstrap interval would mislead. | Removed them from current findings and illustrated point estimates without fabricated confidence intervals. |
| The null hypothesis could be mistaken for a preregistered test or an information-theoretic claim. | Defined the operational macro-F1 comparison and stated that the organizing hypothesis is retrospective; no formal null test is claimed. |
| Some technical terms needed explanation. | Defined representation, visibility, effective rank, temperature, running center, teacher drift, frozen probe, macro-F1, equivariance, and ablation in context. |
| Figure labels were crowded or extended outside their SVG bounds. | Requested wrapping of long captions, repositioning of the reflection schematic, and precise control labels; browser inspection follows regeneration. |

## Scope of unresolved empirical work

The reviews identify a current feature-constructor defect, incomplete visualization execution, incomplete evaluation of the corrected centering objective, timing/padding limitations, and missing inferential analysis. The documentation records these accurately. A documentation revision cannot supply the outcomes of those future experiments, and none is described as completed.


## Codex adversarial pass 1

The [verbatim review](2026-09-13-codex-adversarial-pass1.md) returned `needs-attention` with two medium-severity figure findings. It independently confirmed the retained numerical results. The slide-source rewrite was not yet on disk at the time of this pass, so a final pass covers it separately.

1. **Phase alignment changed amplitude between panels.** The first schematic had equal-amplitude curves before alignment but unequal amplitudes afterward. Correct phase alignment cannot create an amplitude difference. The revision preserves the unequal right-side amplitude in both panels so the second panel reveals a difference already present in the first.
2. **Control labels overstated the measurements.** The plot used “Duration / acquisition” and “Body proportions” although its inputs were retained-frame count and raw median-pose pixel distances. The revision names those actual quantities and updates the figure description.

Both findings require regeneration of the source SVGs and all derived documents. They do not alter the saved experimental measurements.


## Codex adversarial pass 2

The [verbatim final review](2026-09-13-codex-adversarial-pass2.md) returned **`approve`**, with no material findings. It verified the two prior figure corrections, the complete 33-slide tutorial, cohort counts, distinctions among retained protocols, the earlier R1 objective, the failed visualization execution, and the stated inferential limits. Its PDF/PPTX text checks and DOCX content checks found no substantive loss; sampled PDF layouts were readable.

The export review also corrected PowerPoint's SVG extension namespace to Microsoft's specified `http://schemas.microsoft.com/office/drawing/2016/SVG/main` and validated that each SVG relationship points to the embedded vector asset. The specification is documented in [Microsoft's svgBlip definition](https://learn.microsoft.com/en-us/openspecs/office_standards/ms-odrawxml/2451f45e-5d77-4661-86d1-0a017fced779).

## Rendering and final-format verification

Browser checks cover all 33 Marp slide layouts and all 24 SVG text bounds, with no content outside their frames. Contact sheets were inspected, and crowded schematic captions were adjusted. The standalone HTML embeds its SVGs; the slide PDF preserves vectors and omits visible note annotations, while explanatory notes remain in the HTML presentation and PowerPoint.

The Word exporter verifies paragraph, heading, table, caption, equation, link, and embedded-image coverage against the Markdown. Its document has 14 main sections, 19 subheadings, 10 editable tables, 12 SVG figures with PNG fallbacks, and 9 native Office Math expressions. A Pages-rendered copy was inspected as a 21-page document.

The PowerPoint exporter verifies 33 slides and speaker-note pages, editable source text, native tables/cards, the SVG namespace, and byte-for-byte identity of all 24 embedded vector assets. A separate LibreOffice rendering checks all 33 pages. It identified excess spacing in the reproduction code block and awkward wrapping in the final citation pair; both export-specific issues were corrected, and inherited theme shadows were removed for a cleaner layout. Microsoft Word and Microsoft PowerPoint themselves are not installed in this workspace; these compatibility checks do not claim execution in those applications.

The notebooks, experiment caches, checkpoints, and saved prediction records were not changed. The substantive issues found in the documentation have been addressed; unresolved experiments remain explicitly identified as future work.
