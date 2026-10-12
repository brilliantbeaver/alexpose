# WACV 2027 Vision4Neuro paper and evidence

The preferred manuscript is [v9 (PDF)](wacv2007-v9.pdf), with [editable LaTeX](wacv2007-v9.tex), its [bibliography snapshot](references-v9.bib), and a [tested Overleaf ZIP](wacv2007-v9-overleaf.zip). It is written for a general research audience and uses the official WACV 2027 review template. The originally requested seven drafts, **v2 through v8**, remain preserved, followed by the separately requested constructive v9 revision. The supplied v1 PDF and top-level notebooks are unchanged. The requested folder name, `wacv2007`, has also been preserved despite the 2027 venue year.

V9 foregrounds the observed equal-recording score advantage over the tested handcrafted RF and the usefulness of trained features for collection-label prediction. It reduces statistical jargon while retaining the weighting-dependent comparisons and corrected intervals. Exact evaluation details appear in [the v9 methods companion](evidence/v9-methods.md). The [final v9 review](reviews/v9-review.md) assigns **86.0/100**, with the increase confined to presentation.

For Overleaf, upload the ZIP as a new project, select `main.tex`, and use pdfLaTeX. The archive includes the official style files, bibliography, two vector figures and companion records. A fresh extraction compiled successfully with local Tectonic and produced text identical to the delivered PDF; no Overleaf-hosted build or submission was performed. [Package checks](qa/wacv2007-v9-overleaf/manifest.json) record the exact contents and ZIP hash.

The separate [Section 4.1 refinement](wacv2007-v1-section-4.1-refined.md) gives replacement text specifically for v1, with explanations of the masking corrections, a revised figure caption, and notebook evidence. A [PDF copy](wacv2007-v1-section-4.1-refined.pdf) is included.

## Drafts and reviews

| Draft | PDF | Editable source | Score /100 | Detailed critique |
|:--|:--|:--|--:|:--|
| Supplied original | [v1](wacv2007-v1.pdf) | Original PDF only | 46.0 | [Original review](evidence/original-draft-review.md) |
| Revision 1 | [v2](wacv2007-v2.pdf) | [LaTeX](wacv2007-v2.tex) | 74.2 | [Review](reviews/v2-review.md) |
| Revision 2 | [v3](wacv2007-v3.pdf) | [LaTeX](wacv2007-v3.tex) | 81.0 | [Review](reviews/v3-review.md) |
| Revision 3 | [v4](wacv2007-v4.pdf) | [LaTeX](wacv2007-v4.tex) | 83.1 | [Review](reviews/v4-review.md) |
| Revision 4 | [v5](wacv2007-v5.pdf) | [LaTeX](wacv2007-v5.tex) | 85.7 | [Review](reviews/v5-review.md) |
| Revision 5 | [v6](wacv2007-v6.pdf) | [LaTeX](wacv2007-v6.tex) | 85.7 | [Review](reviews/v6-review.md) |
| Revision 6 | [v7](wacv2007-v7.pdf) | [LaTeX](wacv2007-v7.tex) | 85.7 | [Review](reviews/v7-review.md) |
| Revision 7 | [v8](wacv2007-v8.pdf) | [LaTeX](wacv2007-v8.tex) | 85.7 | [Review](reviews/v8-review.md) |
| Requested follow-up, preferred | [v9](wacv2007-v9.pdf) | [LaTeX](wacv2007-v9.tex) | 86.0 | [Final review](reviews/v9-review.md) |

[Scores and revision responses](scores.md) explain the weights, category ratings and changes made after each critique. The rubric is derived from the [workshop call](https://cvpath-vision4neuro.github.io/wacv2027/vision4neuro.html) and WACV reviewer guidance; it is an editorial assessment, not an official numerical conference rubric or an acceptance probability. The evidence-category scores remain unchanged in v9 because clearer prose cannot supply missing experiments or clinical evidence. Earlier drafts retain their documented weaknesses so the revision history remains inspectable; use v9 for further paper preparation.

## Main corrections

The primary comparison follows the top-level notebooks and the associated September execution. Its saved predictions give S-JEPA source-weighted macro-F1 **0.456624**, rather than the supplied PDF's 0.468. One unsupported prediction change reconstructs the original rounded numbers, but the available files do not establish how that change entered the draft. The diagnostic changed only an in-memory copy; the saved predictions were preserved.

The paired intervals are different: S-JEPA minus mean pose is **0.004731 [−0.160496, 0.173999]**, and S-JEPA minus the historical RF is **0.045361 [−0.133447, 0.220114]**. Both use 10,000 label-stratified source resamples, seed 20261004, and the retained predictions. These are retrospective sensitivity intervals conditional on the fitted models, not estimates covering all training or study-design uncertainty. The duplicate interval in v1 is unsupported.

Section 4.1 now describes all 33 inputs, the twelve landmarks' influence on region weights, the correction for overlapping regions, the stopping rule and limited context guarantee. The coordinate chain is also consistent: MediaPipe image fractions are converted to pixels, cleaned and interpolated, then centered and scaled by body geometry for S-JEPA and mean pose. The RF reads the cleaned pixel coordinates. The later corrected RF result remains a separately identified verification; it does not replace the notebook-associated primary comparison.

All 57 original references were checked for bibliographic identity against primary records. No fabricated work or incompatible original author list was identified, but several citations did not support the claims attached to them. The final manuscript retains 13 relevant, verified references. Bibliographic existence and scientific support are documented separately.

## Evidence and reproducibility

| Record | Purpose |
|:--|:--|
| [Notebook audit](evidence/notebook-audit.md) | All seven notebooks, their saved outputs, and the implementation they call. |
| [Claim ledger](evidence/claim-ledger.md) | Links consequential claims to cells, source files and artifacts. |
| [Numerical consistency audit](evidence/results-numerical-consistency.md) | Original-draft discrepancy, paired bootstrap, counts, folds and RF root cause. |
| [Machine-readable results](evidence/results-audit.json) | Exact recomputed values; manuscript settings are under `original_draft_audit`, with separate later-run diagnostics clearly named. |
| [Full reference audit](evidence/reference-audit.md) | Identity and metadata checks for all 57 original references. |
| [Venue and retained claim support](evidence/venue-and-literature.md) | Requirements, rubric rationale, primary literature and claim boundaries. |
| [Reproducibility record](evidence/reproducibility.md) | Run identities, hashes, software versions, commands and replay limitations. |
| [Technical methods](evidence/method-details.md) | Exact equations accompanying the accessible manuscript description. |
| [Adversarial review responses](reviews/adversarial-disposition.md) | Read-only Codex challenges, including the v9 fresh-agent fallback; findings, resolutions and scope limits. |
| [Final independent checks](qa/final-v8-independent-checks.json) and [page/font review](reviews/v8-review.md) | Numerical assertions and source preservation in JSON; seven-page visual inspection and embedded-font checks in the review. |
| [Final v9 checks](qa/v9-independent-checks.json) and [review](reviews/v9-review.md) | Final numerical verification, seven-page inspection and input preservation. |
| [Delivery checks](qa/delivery-verification.json) | Revisions v2–v9, citations, page bounds and notebook preservation. |

The review process used separate research agents and fresh Codex sessions from the same model family. It does not claim human peer review or independence across model families. Rendered figures were inspected for readable labels, contrast and overlap; the preferred draft uses aligned boxes and straight connectors. [Figure review](qa/figure-design-review.md) records the revisions.

To rebuild v9 and its Overleaf package from the experiment directory:

```sh
python3 docs/wacv2007/scripts/build_papers.py 9
.venv/bin/python docs/wacv2007/scripts/verify_delivery.py
python3 docs/wacv2007/scripts/package_overleaf.py
```

The build requires Tectonic and Poppler. The numerical audit command and its dependencies are in the reproducibility record. The manuscript sources, audits and figure generators are provided; access to the retained project data is still required for data-dependent reproduction.

## Preparation status

The final manuscript has seven US-letter pages including references, with no unresolved citations or observed layout collisions. It fits the workshop's eight-page regular-paper body limit and retains an anonymous author block. `DRAFT` is a preparation placeholder, not an assigned submission number. No paper has been submitted.

The workshop call was rechecked on October 9, 2026: the regular-paper deadline is now extended to **October 16, 2026, 23:59 AoE**. See the linked call and [venue record](evidence/venue-and-literature.md) for the remaining instructions.

The study remains exploratory: diagnoses and participant identities are unverified, only one training seed is represented, and no controlled experiment isolates the benefit of temporal order or anatomical masking. The reflection mismatch is disclosed without an invented estimate of its effect. The supplied records do not establish consent for secondary health inference, redistribution rights or an ethics-review determination. Those questions require author verification before submission; revising the text cannot resolve them.
