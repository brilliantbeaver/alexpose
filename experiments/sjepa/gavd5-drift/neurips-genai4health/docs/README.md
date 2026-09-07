# GenAI4Health submission drafts

The recommended draft is **Before Health Agents Interpret Movement: Lessons from a Gait Representation Study**, a position paper with a numerical case study. Start with the [critical assessment and revision strategy](review_and_revision_strategy.md) for the argument, evidence selection, workshop fit, and remaining author decisions.

## Current documents

- Paper: [PDF](genai4health_paper_draft.pdf), [canonical LaTeX](genai4health_paper_draft.tex), [Markdown reader copy](genai4health_paper_draft.md).
- Companion extended abstract: [PDF](genai4health_extended_abstract.pdf), [canonical LaTeX](genai4health_extended_abstract.tex), [Markdown reader copy](genai4health_extended_abstract.md).
- [Submission metadata](submission_metadata.md): title, proposed keywords, TL;DR, and track.
- [Masking-paper assessment](masking_candidate_assessment.md): whether gait-informed masking can support a separate research paper and which experiments are missing.
- [New weighting figure](figures/weighting_comparison.svg): editable SVG, with vector PDF and PNG preview alongside.
- [Numerical supplement](numerical_supplement/README.md): recalculates classifier metrics and weighting averages from retained records.
- [Final verification](review/final_verification.md) and [current evidence checks](evidence/current_verification.json).

LaTeX is the editable source of truth; Markdown and PDF are generated from it. The paper has five main-text pages, one reference page, and two appendix pages. The companion has two main-text pages and one reference page. Both use the official anonymous NeurIPS 2026 style without options.

The [current workshop call](https://genai4health.github.io/2026-NeurIPS/) lists position, research, and demonstration tracks, with no separate extended-abstract track. The deadline is September 9, 2026, 11:59 PM AoE. The companion is a shorter alternative or synopsis; do not treat it as a second independent submission of the same work.

## What the paper says

The same recorded feature similarities average to 0.89 per clip and 0.70 per video because one video supplies 60 of the 64 clips. Neither value measures patient stability or retained predictive ability. Separately, simple pose summaries correctly classify 10 of 20 test videos, compared with six for learned features and six for landmark availability alone. That exploratory result has important procedural and sample-size limits.

The proposed contribution is to keep a movement summary's measurement, recording weights, model reference, and tested scope visible when a future health assistant interprets it. The evidence record and its proposed evaluation are not a deployed system or a demonstrated clinical benefit. Laterality findings, unexecuted forecasting, and historical repair claims are excluded.

## Rebuild

From the experiment root:

```sh
bash neurips-genai4health/docs/build_submission.sh
```

The build uses the existing Python environment with NumPy, Matplotlib, and Pillow, plus Pandoc, Tectonic, and Poppler tools. Set `GENAI4HEALTH_PYTHON` to another suitable Python executable if necessary. It verifies retained outputs, generates the vector figure, builds both PDFs, refreshes Markdown, checks page counts and rendering, and packages the local review files. It does not run notebook cells, train models, download videos, or submit anything.

The standalone numerical check needs only standard Python:

```sh
python3 neurips-genai4health/docs/numerical_supplement/verify.py
```

The full local evidence check also compares notebook predictions, annotation inventories, and quality-control ledgers:

```sh
.venv/bin/python -B neurips-genai4health/docs/reproducibility/build_evidence.py
```

Original fold checkpoints and cached embeddings are absent from this checkout. Current verification is therefore distinct from the earlier checkpoint-backed review. The historical builder is preserved as `reproducibility/build_evidence_from_training_artifacts.py`; it requires the original inputs and is not part of the default build. The earlier `evidence/verification_manifest.json`, advanced-artifact report, and unused figures are historical records, not current verification or submission graphics.

## Local review bundles and author decisions

[Paper source](genai4health_position_source.zip), [companion source](genai4health_companion_source.zip), and [numerical records](genai4health_numerical_supplement.zip) are packaged separately. The paper bundle includes only the new weighting figure. Internal reviews, notebook exports, and local provenance paths are excluded.

GAVD's MIT licensing for research reuse is acknowledged. The annotation license and the separately hosted videos' access conditions have different scopes. The user has not supplied a project-specific institutional determination; approval or exemption is not invented. Authors must settle ethics/data-use wording, authorship, related-submission overlap, and any release of the aliased numerical records before uploading. Alias replacement alone does not guarantee anonymity. Packaging is preparation for local review, not release authorization.

The BrainBodyFM paper and source notebooks are preserved. Its classification appendix overlaps this case study, even though the new argument and primary weighting analysis are distinct from its laterality contribution. No external submission has been made.
