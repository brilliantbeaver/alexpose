# Delivery validation

Checked 28 September 2026. This record concerns the planning/figure package, not validation of a new model or clinical use.

## Evidence and numerical fidelity

- The supplied three-page HAI report was read as text and inspected visually, including its final schematic. The 16-page v08 paper, its relevant figures, appendix, technical methods, and numerical exports were inspected.
- All eight figure sets build successfully. Six are empirical redraws and two are explicitly conceptual diagrams.
- All empirical marks use retained v08 exports. The builder does not train models, estimate new confidence intervals, or change aggregation weights.
- Builder checks confirm all 16 trained pooled response scores exceed zero response; all three primary intervals include zero; score components add to the total; and participant counts favoring direct are 10/14 pooled, 14/14 clear, 1/14 occluded.
- The independent reviewer matched all 248 plotted values, including 80 interval/range rows, to the source exports. See `reviews/final-numerical-audit.json` and the final review for the audit scope.
- Every recorded input SHA-256 matches the current source. The source paper and its original assets were not modified.

## Figure and PDF inspection

- Inspected every figure at readable resolution and all figures in grayscale. Fixed text overflow in the proposed architecture and workflow; separated effect estimates from axis ticks; spaced participant panel titles; set participant axis ranges to include every seed value.
- Rendered all eight pages of the final PDF atlas with Poppler and inspected both four-page contact sheets. No text clipping, missing glyphs, overlapping labels, or missing empirical marks remain in the inspected final rendering.
- `pdfinfo` confirms eight atlas pages. `pdffonts` confirms embedded TrueType fonts with Unicode mappings. SVGs preserve editable text; PDFs preserve vector content; PNGs are 300 dpi.
- Figure 3 separates response and trajectory outcomes and their interval procedures. Figure 4 preserves unconditional score contributions. Figure 6 uses the full percentage scale without a chance line. Figure 7 labels seed ranges as non-CI ranges and includes all people. Figures 1 and 8 identify illustrative/proposed content.
- Full publication captions and image descriptions accompany all eight figures in the HTML/Markdown gallery.

## Documents and reproducibility

- Built readable HTML from the final Markdown sources with Pandoc. The displayed equation is native MathML rather than unparsed TeX. No remote rendering scripts are required.
- Checked every local `href`/`src`, internal anchor, the plan's MathML block, and all eight gallery alt texts. Results are in `structural-validation.json`.
- HTML structure and asset links were checked; this is not a cross-browser layout certification. PDF/PNG figures received visual inspection.
- Rebuild instructions specify the actual Python 3.12.9 workspace interpreter. Plotting library versions are recorded in `figures/provenance.json`.
- Independent initial, draft, and final reviews and the revision dispositions are retained. The plan preserves completed evidence, uncertainty, and future work as distinct categories.

## Intentional limits

The deliverable is a plan for the new narrative and a figure package. It does not claim to have written the final 3,200-3,600-word article, run new experiments, recovered unavailable checkpoints/reconstructions, validated clinical fall risk, or established a physics-aware world model. Those are either subsequent writing steps or explicitly proposed research.
