# Manuscript visual and format review

The revised manuscript was compiled with the unmodified official AAAI-27 submission style, rendered page by page, and checked on November 15, 2026. The [PDF](manuscript.pdf) has **seven pages total**, with the conclusion and references on page seven. It stays within the workshop's seven-content-page and two-reference-page limits.

The [machine-readable checks](rendered/checks.json) verify US Letter pages, no words outside the page bounds, embedded fonts, no Type 3 fonts, vector figures without raster images, unchanged official style files, defined citations, generic anonymous text, and references to both figures and both tables. The final LaTeX pass has no warnings, undefined references or overfull boxes. Nineteen primary references are cited.

All pages were reviewed in the [contact sheet](rendered/contact.png). The two figure pages were also inspected at full size. The method diagram has separated boxes and arrows, short labels and readable body text. The result graphic names each comparator, fixes the positive direction and makes the zero-crossing intervals visible. Table labels distinguish random latent heads from random coordinate heads. No text or connector overlaps were observed.

The visual review found and corrected a real production issue: the first SVG-to-PDF conversion introduced Type 3 fallback glyphs through the local Arial renderer. The method diagram is now generated with embedded TrueType text through Matplotlib. Its SVG remains editable, and both manuscript figures remain vector graphics. This changed rendering only, not any experimental value.

The [initial draft](revisions/01_initial/manuscript.pdf) is preserved as an iteration record and should not be submitted. Use the current `manuscript.pdf`, whose hash is recorded in both [build_record.json](build_record.json) and [rendered/checks.json](rendered/checks.json).

Reproduce the format check from `experiments/aaai`:

```bash
uv run python docs/draft01/build_manuscript.py
uv run python docs/draft01/check_manuscript.py
```

The checker uses installed Poppler tools. It checks visible text and PDF metadata conventions, not whether the scientific evidence is sufficient. The [readiness assessment](submission_readiness.md) keeps those questions separate. The broader 40-slide deck has its own [visual review](../../slides/visual_review.md).
