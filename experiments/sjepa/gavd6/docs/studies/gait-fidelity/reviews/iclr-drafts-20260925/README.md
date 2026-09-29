# Scientific-draft review and validation

Deliverables are new files beside the existing proposals:

- [Full scientific HTML](../../full-writeup.html), with editable Markdown and a 12-page browser-printed PDF.
- [Focused overview HTML](../../research-overview.html), with editable Markdown and an exactly two-page browser-printed PDF. Its body text is 11 pt, with seven requested sections, two vector figures, captions, and references included in the page count.

The central claim is bounded to the completed synthetic development protocol: restoration improves source estimates, but downstream change supervision and its weight materially affect trajectory fidelity; the tested JEPA modifications do not establish their intended primary advantages. Failure accounting and the zero-response benchmark limit stronger movement-recovery claims.

## Independent review and revisions

Three reviewers independently audited evidence, methods/narrative, and figures. Their notes are retained in this directory. Corrections incorporated into the final drafts include:

- Replacing planned follow-up language with the completed three-stage evidence, while retaining independent confirmation as uncompleted.
- Separating pooled-estimator results from ViTPose-only repair results, and stating the exact interval estimators and primary comparison arms.
- Correcting unsupported development raw-motion counts, core JEPA terminology, residual handling of missing coordinates, and the effect of physical mirroring on the edited anatomical side.
- Defining feature residual normalization, control purposes, waveform absolute error, privileged-reference supervision, and the audited gradient-clipping scope.
- Preserving the distinction between failure contributions and errors conditional on successful predictions; retaining the zero-response benchmark and teacher-versus-encoder caveat.
- Keeping tables with their captions, explicitly supplying SVG font fallbacks, aligning numeric columns, and containing tables on narrow screens.

The final rendered visual review covers all full-document pages, both overview pages, equations, the model schematic, and the empirical figures. The final consistency check verifies all 69 transferred evidence-file hashes, original-document preservation, principal means, population counts, every local link and image, the seven overview sections, and its two-page count. Browser checks found no script errors, broken images, or document-level horizontal overflow at desktop and mobile widths. See [final-validation.json](final-validation.json), [browser-checks.json](browser-checks.json), and [final rendered review](final-rendered-visual-review.md).

## Evidence boundary

The data example is a real exported participant-level response curve, chosen by sorted identifier and clearly labeled as an aggregate. Raw rendered frames, pose arrays, and per-example restorations are absent from the transfer, so a verified qualitative reconstruction panel remains unavailable. Protected-person confirmation and real-video/clinical reference validation are also absent. No new training or checkpoint/raw-data replay was performed for these drafts.

The local disk briefly prevented writing additional preview images. Obsolete task previews were removed, and final PDF inspection also used in-memory rendering. Final HTML/PDF files and the original proposals remained intact.

## Reproduction

From the repository root, use `.venv/bin/python docs/studies/gait-fidelity/scripts/build_iclr_draft_figures.py` for empirical assets, then `.venv/bin/python docs/studies/gait-fidelity/scripts/build_iclr_drafts.py` for the HTML and mathematical/architecture assets. The new Markdown files are the prose sources. The optional `render-check.cjs` uses an installed Playwright module to print the actual HTML/CSS and capture browser checks. Run `validate.py` after rendering to refresh the final validation receipt. Existing proposal builders are not invoked.
