# Submission readiness for draft01

The exact manuscript has passed independent scientific, numerical, and visual review. Submission still requires the confirmed author block and human approval.

The manuscript targets the Med-WM short-paper route of 2–4 body pages, excluding references and appendices. The build record reports the actual page boundaries. Do not submit a body outside the allowed range without revising it or deliberately switching to the 6–8 page full-paper route. The official workshop page lists November 20, 2026, Anywhere on Earth. Its OpenReview invitation deadline remains independently unverified.

## What the completed study supports

Bilateral history improved normal-source forecasting over persistence in this exploratory evaluation. The complete set of primary comparisons does not establish a timing-specific bilateral forecasting advantage. The secondary shrunk forecast also improves persistence, but does not improve on the unshrunk bilateral model.

The results come from actual pose inference and normal-only model fitting. The generator validates result hashes and independently recomputes all displayed mean effects, AUROC, and average precision. This is exploratory source-grouped evaluation following exposure to earlier GAVD outcomes. Confidence intervals condition on fixed cross-validation fits and cannot establish patient-level clinical generalization.

## Files and reproduction

Read `manuscript.md` or `manuscript.pdf`. Editable sources are `manuscript.tex`, `references.bib`, and `../../scripts/build_restart_manuscript.py`. Figures are editable SVG and vector PDF, with PNG previews. `evidence.json` identifies the inputs. `manuscript_initial.md` preserves the pre-results narrative. `revisions/01_initial_results_draft/` preserves the first empirical source draft before independent comments were incorporated; that version has no PDF because compiler dependencies were still being resolved.

From the repository root:

```bash
experiments/aaai-worldmodels/.venv/bin/python experiments/aaai-worldmodels/scripts/build_restart_manuscript.py --compile --render
```

The actual compiler is `pdfTeX 3.141592653-2.6-1.40.29 (TeX Live 2026)`, invoked at `/tmp/medwm-tools/TinyTeX/bin/universal-darwin/pdflatex`. BibTeX comes from the same portable TinyTeX installation, and `/tmp/medwm-docs-env/bin/python` performs PDF rendering. Exact commands are recorded in `build_record.json`; command-line options can point to equivalent tools on another machine. The official AAAI style requires pdfTeX, so Tectonic is unsuitable. The style and bibliography files are copied without modification from the existing author-kit copy, with hashes in `author_kit_provenance.json`.

## Systemic critiques and concrete improvements

| Issue | Current consequence | Concrete next step |
| --- | --- | --- |
| Unknown participant identity | Source separation does not prove patient separation. | Verify repeated participants across uploads and obtain an external participant-disjoint cohort. |
| Few normal sources | Many windows cannot establish broad normative coverage. | Add independent normal participants across sites, ages, views, and clinically measured walking conditions. |
| Observational 2D measurement | Predictions do not identify neurology, anatomy, or treatment response. | Validate pose measurements against synchronized clinical reference data and pair movement with clinical assessments. |
| Historical outcome exposure | The new study is exploratory even with nested source folds. | Freeze this protocol and evaluate a genuinely external cohort once. |
| Conditional uncertainty | Source bootstrap does not include full retraining variability. | Repeat outer partitions or bootstrap the complete training pipeline when the independent-source budget permits. |
| Identity and confidence gates | Filtering can select easier sources and retain undetected identity errors. | Review a blinded source-stratified sample of accepted and rejected sequences and document discrepancies. |
| Biological interpretation | Predictive dependence may reflect posture, view, or correlated pose error. | Add controlled view and speed strata, validated gait events, and matched sensor measurements. |
| Workshop scope | This is an observation transition model without actions or disease progression. | Frame it as a tested prerequisite for medical world models; add longitudinal and intervention data before stronger claims. |
| Authorship and submission metadata | Human author details and approval may still be pending. | Confirm the actual author block, affiliations, acknowledgments, AI-use disclosure, and portal metadata before submission. |

## Final review gates

The artifact-specific independent review is recorded under `reviews/`, with fixes in `response_to_reviews.md`. Readiness is marked accepted only when `reviews/acceptance.json` matches the exact PDF, Markdown, and LaTeX hashes. Otherwise a new or changed manuscript needs review. No public submission is performed by the build script.
