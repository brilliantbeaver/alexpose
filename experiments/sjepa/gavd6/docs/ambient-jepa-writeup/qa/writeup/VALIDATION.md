# Full-writeup validation

28 September 2026. Deliverable: `../../output/pdf/jepa-gait-research-writeup.pdf`, with canonical prose in `../../writeup/WRITEUP.md` and a matching HTML version.

## Content and independent review

- Main narrative: approximately 3,203 whitespace-delimited words excluding captions and image alt text; methods/research companion: approximately 1,744 words on the same basis.
- Seventeen letter-size pages: main narrative on pages 1-11, technical companion on pages 12-16, references on page 17.
- Six main figures, two companion figures, and 22 references. The figure-number-to-asset mapping and final PDF/source hashes are in `build-manifest.json`.
- All seven requested extensions are covered, with independent measurements, simpler baselines, decision criteria, and a distinction between completed restoration and proposed prediction/generation/tool routing.
- Independent adversarial review checked the scientific framing, selected numerical claims, methodology, novelty, and literature details. Every requested correction was applied; the second-pass closeout reports no unresolved substantive blocker. The review is separate from visual QA and does not constitute experimental replication.

## Numerical provenance and structure

- All fourteen source hashes in the existing figure provenance still match. No figure data, uncertainty estimate, source manuscript, or plotted value was changed while writing the narrative.
- The earlier independent audit of 248 empirical marks and 80 intervals/ranges remains applicable to the unchanged figure assets.
- PDF checks confirm seventeen nonempty letter-size pages, exactly one caption for each of Figures 1-6 and A1-A2, eight vector figures, no raster images, and embedded/subset Unicode-mapped fonts throughout.
- HTML checks confirm eight local SVG assets with descriptive alt text, four native MathML equations, valid internal anchors and local links, and a single prose source shared with the PDF.
- The attempted browser inspection returned “No browser is available.” The HTML received structural, asset, link, and accessibility-text checks; it is not represented as browser-render verified. The PDF is the visually verified delivery format.

## Visual inspection and corrections

Every PDF page was rendered with Poppler at 110 dpi and inspected as a full-page image. Updated pages were re-rendered and reinspected after revisions; a final contact sheet confirms the page sequence. The mathematical norm bars and learning-rate exponent were inspected after font correction. The final PDF embeds the original seven-inch vector figures, preserving their labels and line art.

The inspection corrected missing norm-bar and superscript-minus glyphs, clarified the Figure 4 caption, moved Figure 2 next to its opening result, and kept the short zero-baseline interpretation intact across the page boundary. Final inspection found no clipping, overlapping text, detached captions, missing glyphs, or illegible figure labels. Section headings retain following text, page numbers and running headers are consistent, and the source-paper/reference links are preserved.

Color and grayscale figure checks from the planning package remain applicable because the figure artwork is unchanged. No unretained example poses or participant trajectories were invented. The conceptual figures are explicitly labeled as a completed workflow or proposed architecture.

## Rebuild scope

`../../scripts/build_writeup.py` builds HTML and PDF from the Markdown and embeds the existing vector artwork. Pinned Python dependencies are in `../../scripts/requirements-writeup.txt`; the package README records the exact workspace commands and macOS font dependency. This rebuild verifies the writeup and figures, not the original training pipeline.
