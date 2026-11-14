# Revised WACV manuscript

The selected title is **Self-Supervised Skeleton Representations for Neurological Gait Analysis**. It is applied to the main paper and supplementary material, with both PDFs rebuilt and the Overleaf package refreshed. See the [title-update verification](wacv-v1-revision-assets/qa/title-update-2026-10-16/delivery-verification.json).

The complete revision implements the reviewed changes to the original wacv-v1.pdf. It uses the corrected October results throughout, preserves the earlier RF result as a historical comparison, and develops the positive finding about useful trained skeleton representations from a small video collection.

The follow-up accuracy review's one inference correction and three wording refinements are now implemented. The visibility result is described without asserting a demonstrated label association; training diagnostics describe their measured behavior; continuation is evaluated explicitly on validation results; and the proposed untrained-encoder comparison is framed as an estimate of predictive benefit. Numerical results and all 16 bibliography entries are unchanged. The PDF and Overleaf package were refreshed and checked; see the [follow-up review](wacv-v1-revision-assets/qa/wording-update-2026-10-16/followup-review.md) and [delivery verification](wacv-v1-revision-assets/qa/wording-update-2026-10-16/delivery-verification.json).

## Files

- [Main paper PDF](wacv-v1-revision.pdf) and [editable LaTeX](wacv-v1-revision.tex).
- [Supplementary PDF](wacv-v1-revision-supplementary.pdf) and [editable LaTeX](wacv-v1-revision-supplementary.tex).
- [Bibliography](wacv-v1-revision.bib), with 16 references used by the main paper.
- [Overleaf package](wacv-v1-revision-overleaf.zip), containing the manuscript sources, bibliography, official style files and all three figure PDFs.
- [Implementation and review record](wacv-v1-revision-assets/reviews/implementation-review.md).

The main paper has eight pages including references; the main text ends on page seven. The separate supplement has three pages. Both use the official WACV 2027 anonymous review style, without modified margins or font sizes. The original PDF is preserved.

## Using Overleaf

Upload the ZIP as a new project. Set the main document to wacv-v1-revision.tex; use XeLaTeX. To build the supplement, change the main document to wacv-v1-revision-supplementary.tex. Each source is fully expanded and can be edited directly; the earlier review fragments and local Python environment are not needed to compile it.

Both sources currently set \wacvPaperID to DRAFT. Replace that value with the assigned submission identifier before uploading the manuscript. The official style's applications option is distinct from selecting Vision4Neuro in the workshop's OpenReview form.

The authors must also supply their documented ethics-review or exemption status and the basis for using the source recordings. The available records do not establish these facts, so the revision makes no consent, approval or redistribution-permission claim. Nothing has been submitted.

For a local build with Tectonic, run from this directory:

    tectonic wacv-v1-revision.tex
    tectonic wacv-v1-revision-supplementary.tex

## Changes implemented

The abstract, introduction and interpretation lead with the small-data representation-learning result. The main comparison is source-weighted macro-F1 of 0.457 for S-JEPA and 0.325 for the corrected handcrafted RF. Mean pose remains prominent at 0.452, with its stronger clip-weighted result stated explicitly.

The dataset and source-partition tables, normalization equation and essential tokenization details are retained. The methods now clarify target-encoder input and initialization, stage selection, fixed-mask readout, actual source counts and the corrected RF features. The masking percentages use their correct denominators. The comparison does not claim an isolated causal benefit from pretraining, temporal order or anatomical masking.

The pipeline and landmark schematic were redrawn, and the results figure was regenerated from verified numerical values. It shows fold scores and the corrected confusion matrices with a shared scale. Detailed training settings, the DINO-style loss and the uncertainty analysis are in the separate supplement. The background is shortened and the two closest neurological-gait studies are included.

The main and supplementary manuscripts were compiled, checked for unresolved references and overflowing text, and inspected page by page. The package was also compiled after extraction into an isolated directory. Exact verification records are retained under wacv-v1-revision-assets/qa/.

The figure-generation script and its numerical inputs are retained in wacv-v1-revision-assets. They reproduce the graphics, not model training. The build_manuscript.py script is an author-workspace assembly helper tied to the archived review fragments; it is not required by the Overleaf package. Edit the expanded .tex files directly for subsequent manuscript changes.
