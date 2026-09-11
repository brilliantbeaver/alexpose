# FMTS manuscript revisions

The final version is **[paper_v8.md](paper_v8.md)**, with its **[PDF](paper_v8.pdf)**, **[anonymous source package](paper_v8_overleaf.zip)** and **[anonymous numerical supplement](paper_v8_supplement.zip)**. It has four main-text pages, one page of references and one supporting appendix page in the official NeurIPS 2026 anonymous workshop style.

The [FMTS review record](../README.md#fmts-2026-review) contains the fixed rubric, grades for the original and Physical World AI V8, every revision's scores and critiques, seven title suggestions, claim-to-artifact provenance, primary-reference checks, final adversarial review and detailed proposed forecasting experiment. The final score is **83.50/100**, an author-side assessment rather than an acceptance prediction. Missing raw artifacts and unperformed controls remain limitations.

| Version | Main editorial change | Score /100 | Manuscript and reproducible packages |
|:--|:--|--:|:--|
| V1 | Establish the focused FMTS evaluation contribution | 75.25 | [Markdown](paper_v1.md), [PDF](paper_v1.pdf), [TeX](paper_v1.tex), [source](paper_v1_overleaf.zip), [supplement](paper_v1_supplement.zip) |
| V2 | Put all five paired effects and diagnostic denominators in the main text | 77.00 | [Markdown](paper_v2.md), [PDF](paper_v2.pdf), [TeX](paper_v2.tex), [source](paper_v2_overleaf.zip), [supplement](paper_v2_supplement.zip) |
| V3 | Clarify source holdout, provenance and conditional inference | 78.50 | [Markdown](paper_v3.md), [PDF](paper_v3.pdf), [TeX](paper_v3.tex), [source](paper_v3_overleaf.zip), [supplement](paper_v3_supplement.zip) |
| V4 | Explain temporal-order limits and synthesize related work | 80.25 | [Markdown](paper_v4.md), [PDF](paper_v4.pdf), [TeX](paper_v4.tex), [source](paper_v4_overleaf.zip), [supplement](paper_v4_supplement.zip) |
| V5 | Redesign vector figures for print size and show the full objective | 81.75 | [Markdown](paper_v5.md), [PDF](paper_v5.pdf), [TeX](paper_v5.tex), [source](paper_v5_overleaf.zip), [supplement](paper_v5_supplement.zip) |
| V6 | Complete Markdown references and separate evidence types concisely | 82.50 | [Markdown](paper_v6.md), [PDF](paper_v6.pdf), [TeX](paper_v6.tex), [source](paper_v6_overleaf.zip), [supplement](paper_v6_supplement.zip) |
| V7 | Reconcile precision and consolidate supporting sections | 82.50 | [Markdown](paper_v7.md), [PDF](paper_v7.pdf), [TeX](paper_v7.tex), [source](paper_v7_overleaf.zip), [supplement](paper_v7_supplement.zip) |
| V8 | Rewrite in clear connected prose and finish the diagram and one-page appendix | 83.50 | [Markdown](paper_v8.md), [PDF](paper_v8.pdf), [TeX](paper_v8.tex), [source](paper_v8_overleaf.zip), [supplement](paper_v8_supplement.zip) |

Every version has four main-text pages. V1–V7 have seven pages in total; V8 has six. Equal V6/V7 scores are deliberate: better precision does not create additional empirical evidence. Private assets under `assets/vN/` include the bibliography, official style, numerical record, vector generator, figures, rendered pages and manifest. V1–V4 retain their smaller figure lettering as part of the documented revision history; V5 addresses it. No FMTS manuscript depends on a mutable Physical World AI asset.

## Reproduce a manuscript

For the strongest version-preservation guarantee, extract that version's `_overleaf.zip` into a fresh directory and compile `main.tex` with Tectonic, or pdfLaTeX plus BibTeX. Each package was compiled in an isolated temporary directory; extracted text and page counts matched its delivered PDF. There are no external figure or bibliography dependencies, and no shell escape is needed. The unmodified official style is included.

Alternatively, from this directory, `tectonic paper_v8.tex` compiles the frozen V8 source using its private assets. Treat the existing manuscripts and private assets as snapshots; make a new numbered directory for subsequent revisions. The version-specific `assets/vN/build_version.py` retains the Markdown-to-TeX builder, while the frozen TeX/source ZIP is the authoritative typesetting source. Rebuilding may change PDF timestamps or binary hashes without changing the rendered document.

To verify the retained numerical arithmetic without additional dependencies:

```sh
python assets/v8/verify_summary.py assets/v8/numerical_evidence.json
```

To regenerate the figures, extract the numerical supplement into a fresh directory, install ReportLab, and run its `draw_figures.py --version 8 --assets .`. The supplement includes the matching generator and evidence. The [V8 training pipeline](assets/v8/figures/training_pipeline_compact.svg) and [result figure](assets/v8/figures/learning_results.svg) are editable SVGs with matching vector PDFs. All poses are schematic.

The supplements contain retained seed aggregates and recorded source-bootstrap intervals. They cannot reproduce source resampling or encoder inference without the missing prediction rows, poses and checkpoints. No new training was performed for these revisions. The [delivery checks](review/delivery_checks.json), [evidence check](review/evidence_check.json), [scores](review/scores.json), and [final review](review/review_v8_final.md) state what was verified.

## Submission scope

The FMTS CFP and official template were checked on 10 September 2026. Manuscript and numerical packages are anonymous and include substantive AI-assistance disclosure. They contain no videos, derived poses, author identities, acknowledgements, local execution paths or personal repository links. Public bibliographic author names and the official style's credits are retained normally. The project's institutional ethics, data-use and derived-pose release determinations remain unresolved; no external submission or data release has been made.
