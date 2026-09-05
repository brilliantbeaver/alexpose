# GenAI4Health draft package

Start with [the review and revision strategy](review_and_revision_strategy.md). The recommended submission is a position paper with an empirical audit, titled **Before Gait Models Inform Care: Evidence Boundaries for Predictive Health AI**.

## Drafts

- Paper: [PDF](genai4health_paper_draft.pdf), [editable LaTeX](genai4health_paper_draft.tex), [Markdown reader copy](genai4health_paper_draft.md).
- Companion extended abstract: [PDF](genai4health_extended_abstract.pdf), [editable LaTeX](genai4health_extended_abstract.tex), [Markdown reader copy](genai4health_extended_abstract.md).
- [Numerical supplement](numerical_supplement/README.md): reproduces the headline scores and source-weighting comparison without video data or third-party packages.
- Download bundles: [paper source and numerical files](genai4health_position_source.zip), [numerical supplement alone](genai4health_numerical_supplement.zip). The companion abstract is intentionally separate.
- [Figure assets](figures/) and [verified data tables](evidence/).

LaTeX is canonical; Markdown is generated from it. The final paper has five main-text pages, one reference page, and three appendix pages. The companion has two main-text pages and one reference page. See the [PDF verification record](review/pdf_qa/qa_summary.json).

The [workshop call](https://genai4health.github.io/2026-NeurIPS/) lists no separate extended-abstract track. Submit one chosen manuscript, not both variants. The package uses `\usepackage{neurips_2026}` without options. No external submission has been made.

## What changed

The drafts replace the previous performance/world-model emphasis with two verified observations: raw pose summaries have the higher observed readout score in one split, and a dominant upload changes the interpretation of clip-weighted coordinate similarity. They correct the implemented loss, annotation use, architecture, refitting and aggregation, diagnostic preprocessing, and boundaries of forecasting and clinical claims.

The source notebooks and BrainBodyFM files remain untouched. Historical classifier/laterality/consolidation/surprise values are excluded from current claims. Full methods limitations remain visible. Actual ethics/data-use determination and authorship must be settled by the authors; neither is invented here.

## Rebuild and verify

From this folder with Tectonic installed:

```powershell
tectonic genai4health_paper_draft.tex
tectonic genai4health_extended_abstract.tex
python numerical_supplement/verify.py
```

For the full local audit, from the experiment root using its existing environment:

```powershell
.venv/Scripts/python.exe neurips-genai4health/docs/reproducibility/verify_core_artifacts.py
.venv/Scripts/python.exe neurips-genai4health/docs/review/audit_advanced_artifacts.py --output neurips-genai4health/docs/review/advanced_artifact_checks.json
.venv/Scripts/python.exe neurips-genai4health/docs/reproducibility/build_evidence.py
.venv/Scripts/python.exe neurips-genai4health/docs/reproducibility/make_numerical_supplement.py
.venv/Scripts/python.exe neurips-genai4health/docs/reproducibility/render_and_check.py
uv run --no-project --with pypandoc-binary python neurips-genai4health/docs/reproducibility/make_reader_copies.py
```

`review/` and `reproducibility/notebook_text/` are internal working records and contain local provenance details; do not upload them as an anonymous supplement. The numerical supplement is prepared separately with source aliases and no local paths. Its aliases support anonymous author review, not a guarantee of patient anonymity. The source bundle is built from an explicit allowlist.
