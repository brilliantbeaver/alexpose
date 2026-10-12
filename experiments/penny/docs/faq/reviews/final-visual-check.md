# Final visual and Word compatibility check

Reviewed on 4 October 2026. The final Word document was rendered with LibreOffice 26.8.0 and inspected as `docs/faq/render/faq.pdf`. This review concerns presentation and delivery integrity; it does not rerun the research experiments.

**Result: pass.** The final render contains 20 pages, all 45 FAQ questions, four readable tables, and all 15 numbered figures with their captions. No clipping, detached figure captions, overlapping diagram labels, or orphan section/question headings were found. Paragraphs can continue onto the next page; each question heading remains with its opening answer.

## Pages inspected

Every page was inspected using five contact sheets, with additional inspection of the dense table and figure pages. After the final definitions of transformer, batch, and random seed were added, the latest PDF was checked again for boundaries, caption placement, and question placement; pages 5, 10, and 13 were visually reinspected at full page size. The render remains 20 pages.

| Render pages | Items checked | Outcome |
| --- | --- | --- |
| 1–4 | Title and opening summary; reading-route table; Figure 1; revised null hypothesis; full notebook table on page 3; JEPA definitions | Readable. All notebook rows fit on page 3, with no truncated links or split rows. Question 3's retrospective qualification continues normally onto page 3. |
| 5–8 | Figures 2–4; encoder/predictor arrows and EMA label; loss/collapse labels; cohort table; full-frame coordinate explanation; extraction/normalization diagram | Readable. Cohort rows and figure captions are intact. The figure text retains its intended emphasis and alignment. |
| 9–12 | Figures 5–8; anatomy diagram; mask grid and eligibility caption; 384-value pooling; metric definitions; exact-split bar labels and caution | Readable. The mask caption correctly says the grid displays the ten eligible landmarks. Quantitative labels fit within the illustration. |
| 13–16 | Figures 9–12; source-split diagram; video-ID standard-deviation whiskers; grouped evaluation caveat; gait-probe table and its three-family illustration | Readable. The dense gait table on page 16 includes all six rows and all three feature families. Negative signs and R² labels render correctly. |
| 17–20 | Figures 13–15; mask-ablation counts; future-context diagram; full-clip normalization example; phase failure and latent-loss caveat; final evidence notes | Readable. Figure 15 stays with its caption on page 19. The last-page evidence and reference links remain legible. |

Additional boundary checks found no text outside the horizontal page margins or clipped by the page edge. The running footer sits in its intended footer area, below the body text. All 15 captions occur on the same pages as their figures. Figure headings, chart labels and explanatory notes were checked at the document's 6.4-inch illustration width. Earlier SVG inspections also verified that no illustration text exceeds its canvas.

## Vector media and compatibility

The DOCX archive contains 15 SVG media parts referenced through Office's `svgBlip` extension and 15 PNG fallback images. The figure relationships resolve to the correct media; each figure has descriptive alternative text. The document also preserves short valid bookmarks, relative repository hyperlinks, external web references, and a footer page-number field. All local hyperlink targets resolved in the accompanying workspace at review time.

LibreOffice's final PDF contains **zero raster image objects and 735 vector drawing paths**. Illustration labels are extractable as text, confirming that this renderer used the embedded vector artwork. The PNG media remain available for viewers that use the fallback instead. Microsoft Word itself was not available for a separate visual-render test; the verified package provides the Office SVG extension and fallback path, rather than assuming every Word version renders SVG identically.

## Reviewed artifact fingerprints

| Artifact | SHA-256 |
| --- | --- |
| `docs/faq.md` | `34f50e13e926aa13e9d0a5d26d94ae6a0d26c1ab99690f26cc83f6e84cfe1ce9` |
| `docs/faq.docx` | `111afdf436693fa55b59369b0072ed86e4a937bdc59b115bccb35dc155caf697` |
| Local inspection PDF | `3b4d971479e353d9b7260c5af1b6df7dde8d82607f5d6b68b543250ca1c35fa4` |

No rebuild or content change was required by this final visual review. Temporary PDF renders, page images and contact sheets remain in the ignored `docs/faq/render/` directory.
