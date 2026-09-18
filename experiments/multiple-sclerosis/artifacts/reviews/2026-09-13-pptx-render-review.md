# PowerPoint export review — September 13, 2026

The 33-slide PowerPoint was generated from `slides/slides.md` using
`scripts/scripts_build_slides_pptx.py`. Its package contains 24 distinct SVG
illustrations, PNG fallbacks, editable titles and explanatory text, editable
cards and the summary table, reproduction commands, citation links, and 33
native speaker-note pages.

The exporter's checks verify ZIP integrity; parse every XML part; compare
source prose, cards, table cells, commands, and speaker notes with the saved
PowerPoint; check slide-shape bounds; and compare every embedded SVG with its
current source hash. SVG picture extensions use Microsoft's
`http://schemas.microsoft.com/office/drawing/2016/SVG/main` namespace and
resolve to valid image relationships. The build manifest is
`2026-09-13-pptx-validation.json`.

For visual verification, LibreOffice 26.2.6 rendered the PowerPoint to a
separate 33-page PDF. The renderer was used from a temporary, read-only disk
image whose SHA-256 matched the official HTTPS download metadata; it was not
installed as an application. Every slide was inspected in contact sheets,
with larger views of the title, conceptual cards, results, summary table,
reproduction instructions, and closing citations. Extracted PDF text had no
spans outside page boundaries.

Review found and corrected an SVG namespace error, an inherited oversized
blank paragraph in the code block, and an awkward citation line break.
The final render also verifies the restrained hyperlink colors and flat
shape styling. The reproduction commands fit within their panel and the
summary table's wrapped cells remain legible. The final source includes the
revised two-line caption in the schematic symmetry illustration.

This is an export and presentation review. Scientific claims and the
underlying retained results were reviewed separately; the export did not
run training or change experimental artifacts.
