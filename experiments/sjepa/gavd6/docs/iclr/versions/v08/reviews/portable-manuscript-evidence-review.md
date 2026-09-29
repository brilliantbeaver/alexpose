# Portable manuscript wording: independent evidence review

Reviewed on 2026-09-26 against `history/v08-before-local-reference-cleanup-20260926.zip`. This review covers the main manuscript, the concise appendix, the optional full technical appendix, and the rebuilt default PDF. Only this review file was written; manuscript, numerical, and figure files were not edited.

## Final assessment

**Approved.** Removing local directories, file names, notebook navigation, and operational vocabulary has preserved the scientific comparisons and evidence limits. The resulting prose explains the experiment without requiring knowledge of an author's workspace. No unresolved material numerical, methodological, authorship-text, or reproducibility-scope misstatement was found.

The broader cleanup was checked as well as literal paths. Terms such as saved exports, calibration receipts, internal artifacts, source bundles, and tutorial execution have been replaced with conventional descriptions of recorded comparisons, calibration measurements, study results, and untested controls. Scientific model settings, training logs, and the limits on complete reproduction remain where they explain the procedure. The removal does not claim that instructional calculations were additional experiments or that every ancillary analysis was executed successfully.

## Evidence integrity

- Every inline mathematical expression and displayed equation in the main paper and concise appendix is unchanged from the snapshot. Their scientific numerical tokens are unchanged; the concise appendix's removed notebook-navigation numbers were excluded from that comparison.
- The full technical appendix preserves all **152 inline math spans**, **nine displayed equation/align blocks**, and its label/reference/citation commands. Its scientific numerical tokens remain unchanged after excluding the removed notebook walkthrough and the explicitly unexecuted **102-fit instructional recipe**. The completed fit counts and their populations remain intact.
- All **seven numerical table files** and **13 audit CSVs** present in the snapshot are byte-identical. This editorial revision performs no re-estimation, new resampling, selection of a favorable arm, or change to plotted results.
- The three primary comparisons still use their original comparators and outcomes: core/change versus direct/change, delta/change versus endpoint/change, and delta dense versus low scalar on ViTPose waveform error. Their unresolved intervals, secondary/exploratory status, and distinct uncertainty procedures remain explicit.
- The limits that could change interpretation remain: repeated use of 14 development people and three fitted seeds; no independent confirmation; held estimator/edit conditions within that same cohort; training-only optimization/calibration with development-reference integrity checks; unequal initial auxiliary influence; no convergence or matched-influence experiment; and no clinical or natural-video validation.
- The zero-response comparison, clear-versus-occluded exception, separate geometric assignment support, failure-cost decomposition, and distinction between unconditional contributions and conditional successful errors are preserved. The aggregated-data limitation still prevents reconstruction of individual-pair true-response magnitudes. The full method/condition/penalty/repair inventories remain available in the technical reference.
- Reproducibility remains bounded accurately: participant-level summaries permit reaggregation and checks, while complete study reproduction additionally requires motion/body-model assets, renders, pose arrays, per-example predictions, and checkpoints not included in the available study materials. Removing packaging instructions did not turn this into a claim of a self-contained training reproduction.

## Authorship and AI disclosure

The first human-authorship paragraph is **byte-identical** to the snapshot. Its SHA-256 is `f68a4dd447c997f14ee2dbe5d3110c0596342b5461dd49da345a065f155e0221`. Human origin of ideas and initial drafts, final editing and verification, approval, and responsibility have not been reassigned or qualified by this revision.

The AI-assistance paragraph still describes research-code and analysis inspection, scientific and literature review, result analysis, drafting, plotting/build work, and subsequent human verification. Replacing “notebooks” with “analysis materials” preserves that contribution. One scope concern raised during review is closed: the final sentence now says **“No new study models were trained or participant data collected during this manuscript revision.”** This retains the original study-training boundary without broadening it into an assertion about every instructional computation.

## Readability and rendered-reference checks

The main manuscript and concise appendix pass a source-prose scan for local paths, file extensions, notebook references, directory/setup instructions, and the wider artifact/receipt/export/tutorial/packet/saved/bundle vocabulary. The full technical appendix passes the same scan after its scientific rewrite. Nonrendered LaTeX commands that load styles, tables, and figures remain necessary document dependencies and are not user-facing local references.

The rebuilt **16-page default PDF**, newer than its main and concise-appendix sources, was extracted with `pdftotext -layout` and scanned independently. It contains **zero matches** for the local-reference and wider operational-vocabulary patterns. The human-authorship paragraph and revised study-model disclosure are present in the rendered text. The optional full technical appendix was reviewed in its final source form; its separate typesetting check belongs to the editor's package validation.

## Final bindings

- `paper-v08.tex`: `85b08ea58f95be0384244aa3848bf23fb77a4ac5c64ae7f32e723d2d1a523e68`
- `appendix.tex`: `6dde88735f69940dbcb2ed69d8c686a1d8db2ff23588de7363c61279410c8e50`
- `supplement/technical-details.tex`: `53c9b4b711f00d555bbddd561da51b04dc76568e5e0b671157b879ccddc7517e`
- `paper-v08.pdf`: `cbcffb29e2825bf9103cc4a69297f3d48373bcaf66854eea7c7b6f79a946d5bc`
- `history/v08-before-local-reference-cleanup-20260926.zip`: `abf9ced226fcdffda3103d5009fb1a8c4d2193a34f0897fb5b55efc909db64b0`


## Optional appendix layout disposition

The final layout revision removes only the second, nonnumerical table in Section H, “Scientific questions and their evidence boundaries.” I checked its coverage against the surrounding technical reference. Pair construction/projection and observation-based normalization remain in A; fixed tensor slots, split inheritance, and fitting boundaries remain in I; reference support and uncertainty remain in B; initial gradient calibration and frozen readout fitting remain in A; and H retains all three primary questions and their results. Its closing paragraph explicitly retains the untested alternative masks, learned external refiners, and movement re-pairing, together with the absence of additional evidence or independent confirmation. No completed comparison or limitation is lost. All 152 inline math spans, nine display blocks, and the seven numerical table files remain unchanged. Approval is preserved; the supplement hash above binds this final layout revision.
