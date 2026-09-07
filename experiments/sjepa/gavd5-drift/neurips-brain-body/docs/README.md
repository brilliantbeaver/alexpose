# Brain and Body documents

The current short paper integrates two studies: a classification pilot from the Brain and Body notebooks and the completed real-data laterality v2.1 experiment. Start with the [submission PDF](bbfm2026_paper_draft.pdf) for the argument, or the [explanatory paper](explanatory_paper.pdf) for the step-by-step methods, numerical evidence, vector illustrations, and workshop assessment.

Study A contains 639 sequences from 97 sources, with a retained fold-0/seed-42 execution on 20 test sources. Its current checkpoint and evaluation bundles are absent from this checkout. Study B contains 625 sequences from 93 sources across five outer folds, five seeds, and two variants. Its 50 trained checkpoints and 100,000 prediction rows were inspected, and independent in-memory recomputation of the two primary bootstrap tables reproduced the saved estimates within $10^{-16}$. This checks recorded artifacts and calculations, without independently retraining either study. The cohorts overlap and should not be pooled or described as independent replication.

The [readiness guide](neurips-brain-body.md) and explanatory Section 13 assess the revised integrated paper at 72/100 using an internal weighted rubric; the historical uncorrected pilot-only draft scored 48/100. These are readiness judgments, not acceptance probabilities. Submission and release remain blocked by three unresolved reviews in the [laterality governance record](../../neurips-laterality/governance/status.json). The weighted score does not override that gate.

## Contents

- [Short submission: Markdown](bbfm2026_paper_draft.md), [generated LaTeX](bbfm2026_paper_draft.tex), and [PDF](bbfm2026_paper_draft.pdf).
- [Explanatory paper: Markdown](explanatory_paper.md), [generated LaTeX](explanatory_paper.tex), and [PDF](explanatory_paper.pdf).
- [Submission-readiness guide: Markdown](neurips-brain-body.md), [generated LaTeX](neurips-brain-body.tex), and [PDF](neurips-brain-body.pdf).
- [Editable SVGs and vector PDFs](figures), with [tutorial provenance](figures/tutorial_figure_provenance.json) and [laterality submission-figure provenance](figures/submission_laterality_provenance.json).
- Dated availability and acquisition audits, which are snapshots rather than live availability guarantees.

Markdown is the editable source of truth for all three documents. The short manuscript has been revised to distinguish the two studies and correct earlier pilot-method descriptions. The September 5 build review confirms five main-text pages, two appendix pages, and one reference page, using the shared NeurIPS 2026 double-blind workshop style. All eight pages were visually inspected, and PDF metadata contains no author identity. The embedded laterality graphic is vector-only, with labels of at least 9.3 pt at the official 5.5-inch text width. These checks do not resolve the project reviews or verify submission-portal declarations.

## Rebuild

From the repository root, rebuild the short paper and explanatory companion with:

```sh
bash neurips-brain-body/docs/build_submission.sh
bash neurips-brain-body/docs/build_explanatory_paper.sh
```

The builds use the repository's `.venv/bin/python` with NumPy and Matplotlib, Pandoc, and Tectonic. Set `BRAIN_BODY_PYTHON` to an absolute Python executable path if needed. Generated LaTeX should be rebuilt from Markdown.

To rebuild the readiness guide, run these commands from `neurips-brain-body/docs`:

```sh
pandoc neurips-brain-body.md \
  --from=markdown --to=latex --standalone --shift-heading-level-by=-1 \
  --template=tutorial_template.tex --output=neurips-brain-body.tex
tectonic --keep-logs neurips-brain-body.tex
```

The tutorial figure generator checks retained notebook outputs and source hashes without executing the notebooks. The submission figure generator reads the laterality result artifacts. Both render editable SVGs and vector PDFs; the LaTeX documents embed the PDFs. Regenerating these figures verifies their correspondence to saved evidence and does not substitute for training or recovery of the missing pilot bundles.

The illustrations contain aggregate counts, results, and schematics rather than individual video frames or trajectories. Source-linked underlying records remain subject to the documented governance and release decisions.
