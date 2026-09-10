# Vector PDF assets

These PDFs reproduce the SVG illustrations for inclusion with LaTeX `\includegraphics`. Each PDF contains one page, searchable text, and vector paths. Inspection found no raster image objects. The compact figure was subsequently redesigned with explicit user authorization; the results illustration later received display-only numerical rounding, while the detailed pipeline and reflection illustrations remain unchanged. Four additional print layouts use labels of approximately 8 points at a 396-point manuscript width.

| Asset | Page size in PDF points | Vector path operations | Text-show operations |
|:--|--:|--:|--:|
| [Compact training pipeline](training_pipeline_compact.pdf) | 743.04 × 533.04 | 1,273 | 61 |
| [Detailed training pipeline](training_pipeline.pdf) | 1170.00 × 840.00 | 1,317 | 139 |
| [Reflection and target](reflection_and_target.pdf) | 1080.00 × 595.92 | 2,362 | 74 |
| [Learning results](learning_results.pdf) | 1155.12 × 775.92 | 2,576 | 82 |

Headless Chrome prints a temporary HTML wrapper with a zero-margin page matching the SVG's dimensions. Chromium rounds some page dimensions by less than 0.8 PDF points, with no visible effect. The conversion preserves the illustration's text, geometry, colors, and aspect ratio. Poppler independently rendered every resulting PDF at 96 dpi; those previews were inspected for clipping, glyph errors, and layout changes. The additional previews use the suffix `_pdf_preview.png`.

The print PDFs are [compact pipeline](training_pipeline_compact_print.pdf), [detailed pipeline](training_pipeline_print.pdf), [reflection](reflection_and_target_print.pdf), and [results](learning_results_print.pdf). Their page sizes are 743.04 × 533.04, 743.04 × 990.00, 743.04 × 593.04, and 743.04 × 720.00 PDF points. [physworld_print_pdf_validation.json](physworld_print_pdf_validation.json) records their vector and text checks; [compact_pdf_validation.json](compact_pdf_validation.json) records the current canonical compact PDF. [physworld_print_figures.json](physworld_print_figures.json) records the print-size font check and preserved-original hashes.

To reproduce using the available bundled runtime:

```powershell
& "$env:USERPROFILE/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe" C:/Users/alexm/dev/alexpose/experiments/sjepa/gavd5-drift/neurips-laterality/docs/figures/make_physworld_vector_pdfs.py
```

The script accepts `--browser` and `--pdftoppm` to select other local executables. Use `--print-variants` for the four print layouts, or `--assets NAME... --report-name NAME.json` for an explicit asset set. `--publish-preview` intentionally updates each selected asset's PNG from the independently rendered PDF; it is off by default. Temporary wrappers and browser profiles are confined to unique subdirectories of `tmp/pdfs` and removed when conversion finishes. The script does not edit manuscript or SVG sources. The original [physworld_pdf_validation.json](physworld_pdf_validation.json) remains a record of the earlier four-asset conversion; use the newer compact and print reports for the current redesign.

The 10 September results-only precision update refreshed both learning-results PDFs and PNG previews. Use [learning_precision_pdf_validation.json](learning_precision_pdf_validation.json) for their current hashes and vector checks. The full-precision figure ledger and plotted coordinates are unchanged; only the connected-region difference label shortened from −0.0087 to −0.009. Earlier validation files remain historical conversion records.

The validation report records page dimensions, searchable-text checks, fonts, vector and text operation counts, original-file hashes, and generated PDF hashes. The figure generator and standalone captions remain documented in [physworld_figures.md](physworld_figures.md).
