# Draft03 author-side rendering review

The generator compiled the manuscript with the local pdfLaTeX/BibTeX tools and rendered every PDF page using PyMuPDF. The authoring agent inspected all six full-size page images and the contact sheet. This is author-side visual quality assurance, separate from the independent scientific review.

The final body occupies three pages. References begin on page four and the reproducibility appendix begins on page five. The six-page PDF is within the workshop's short-paper body range. Text-bound checks find no edge overflow. The final TeX log has no undefined citations or overfull boxes.

The first render exposed a dangling final sentence on a largely empty fourth body page, an appendix heading displaced by floating tables, and diagram labels extending outside their boxes. These were corrected by shortening prose, moving the source scatter to the appendix, keeping appendix tables in normal reading flow, and shortening diagram labels. Font sizes and manuscript margins were not reduced. The final equation is typeset as mathematics rather than literal caret notation.

The main figure labels, anomaly axes, conditional-interval caption, source counts, and normalized-coordinate units were checked against the evidence. The main narrative reads as complete connected paragraphs. The appendix retains all model arms, paired contrasts, original anomaly directions, and explicitly post hoc diagnostics.

Final inspected PDF SHA-256: `5d54db8359941171c3f739703aa68221e9f8125fc9e85be80d124ed3e61087d2`.

Author and affiliation fields remain placeholders. This is an internal draft, not a submitted manuscript.
