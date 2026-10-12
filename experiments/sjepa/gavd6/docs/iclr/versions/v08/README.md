# Version 08

**[Read the revised paper](paper-v08.pdf)** — nine main pages, two pages of disclosure/references, and five appendix pages (16 total). The title is **Evaluating JEPA-Inspired Motion Representations through Geometry and Gait Asymmetry**. Versions 01–07 are preserved. The earlier v08 fixed-rubric synthesis was **79.33/100**, versus v07’s **75.75/100**; that score predates this framing revision and is not an acceptance probability.

The central contribution is an empirical account of how feature objectives, readout supervision, failure costs, and geometric naming affect evidence of faithful movement restoration. The zero-response benchmark is now central. The manuscript leaves the incremental predictive-feature benefit unresolved and does not generalize this result to predictive representations as a whole.

The in-place framing revision explicitly connects self-supervised learning, JEPA,
and world-model research to the tested restoration task. It adds primary clinical
motivation, a sequence of experimental questions and hypotheses, concrete split and tensor-integrity
controls, and an all-participant illustration of lower and higher response error.
The prior v08 artifacts are preserved in [a history snapshot](history/v08-before-jepa-framing-20260926.zip).
See [framing decisions](FRAMING-PLAN.md) and the revision record for review dispositions.

The latest clarity revision explains the asymmetry-change outcome before reporting
its errors, defines the zero-response baseline, and distinguishes trajectory
accuracy from response accuracy. The introduction follows the experimental
questions in order, and the methods and appendix explain the weighting and
uncertainty calculations. The [evidence check](reviews/clarity-evidence-review.md)
and [methods/appendix review](reviews/clarity-appendix-review.md) record the
source and notebook checks. World models remain future motivation, with one
mention each in the abstract, introduction, and discussion.

The latest comparison revision reduces Table 1 to the three declared scientific
questions and the appendix from twelve pages to five. Two new vector figures
show the data/evaluation contract and the three primary paired effects. The
single main waveform graphic retains all eight fitted procedures under both
readout objectives. Repeated inventories move to the optional
[full technical reference](supplement/README.md), included in both ZIPs; every
numerical table and audit CSV is retained in the portable source bundle.
[Comparison selection](reviews/comparison-selection-review.md) and
[methods review](reviews/appendix-simplification-review.md) document the checks.

Figure 1 now combines original vector gait/keypoint illustrations with the
experimental workflow. It shows the controlled pair, the endpoint-or-delta
training choice, and independent restoration followed by gait-measurement
scoring. Training references and evaluation references have separate roles.
The keypoints are explicitly illustrative, and the shortened caption links to
the exact feature losses. The [design record](evidence/figure1-design-provenance.json)
documents these choices. All eight figures include editable SVGs and vector PDFs.

The latest manuscript cleanup removes local directory/file references, notebook
names and IDs, and machine execution details from the main text and both appendix
versions. The scientific experimental sequence and reproducibility boundaries
remain explicit. Independent [methods](reviews/portable-manuscript-methods-review.md)
and [evidence](reviews/portable-manuscript-evidence-review.md) reviews verify the
cleanup. The [current visual check](qa/portable-manuscript-review.json) and
[clean rebuild](qa/portable-manuscript-rebuild.json) bind this revision. Relative
paths in the bundle READMEs are retained for compiling
the Overleaf project and rebuilding figures; they are not manuscript content.

## Deliverables

- [Main LaTeX source](paper-v08.tex), [appendix source](appendix.tex), and [bibliography](references.bib).
- [Overleaf upload ZIP](paper-v08-overleaf.zip): a single `main.tex` plus all typesetting assets. Select **XeLaTeX** after upload; see [upload instructions](OVERLEAF.md) and the [clean-build check](qa/overleaf-package.json).
- [Portable paper source and editable figures](paper-v08-source.zip): compiles independently and rebuilds figures/tables from the included audited data. The complete upstream audit uses the repository exports described below.
- [Revision and independent-review synthesis](REVISION-RECORD.md), [initial scientific argument](CONTRIBUTION.md), and [figure specification](FIGURE-SPEC.md).
- [Framing and integrity claim map](evidence/framing-claims.json), [new participant-figure audit](reviews/case-figure-evidence.md), and [current guideline verification](reviews/framing-literature-and-guidelines.md).
- [Evidence audit](evidence/audit.md), [claim-to-artifact map](evidence/manuscript-claims.json), [analysis provenance](evidence/provenance.json), and all selected groups in `evidence/*.csv`.
- [Earlier framing-stage Codex adversarial review](reviews/codex-adversarial-review.md) and [disposition](reviews/codex-adversarial-disposition.md): approve, no material findings.
- [Prior framing methods review](reviews/methods-framing-final.md), [evidence review](reviews/evidence-framing-final.md), and [writing/figure review](reviews/editor-framing-final.md). Earlier v08 assessments remain in the same review directory.
- [Visual precedents](reviews/visual-precedents.md), [verified ICLR rules](reviews/submission-rules.md), [structural checks](qa/validation.json), and [render review](qa/portable-manuscript-review.json).
- Eight figure sets in `figures/`: vector PDF, editable SVG, color PNG, grayscale PNG; shared styling and builders in `scripts/`.

## Rebuild from the repository

From the repository root:

```sh
.venv/bin/python docs/iclr/versions/v08/scripts/analyze_evidence.py
.venv/bin/python docs/iclr/versions/v08/scripts/build_figures.py
.venv/bin/python docs/iclr/versions/v08/scripts/build_tables.py
.venv/bin/python docs/iclr/versions/v08/scripts/build_case_figure.py
.venv/bin/python docs/iclr/versions/v08/scripts/build_summary_figures.py
```

Then compile from this version directory:

```sh
tectonic paper-v08.tex --outdir . --keep-logs --keep-intermediates
pdftoppm -r 110 -png paper-v08.pdf qa/final-page
```

Run `.venv/bin/python docs/iclr/versions/v08/scripts/validate_package.py` from the repository root. These commands modify only v08 analysis/build artifacts; they launch no training. The Python package versions used are recorded in `requirements.txt`. Tectonic can download missing TeX dependencies on first use. Official style files are unchanged.

To regenerate the separate Overleaf upload ZIP, run
`python3 docs/iclr/versions/v08/scripts/build_overleaf_archive.py` from the repository root.

## Evidence and submission boundaries

All new stratification, fixed-cost sensitivity, and expanded plotting intervals are exploratory analyses of completed exports. There are still 14 development people, three fitted seeds, adaptive development reuse, no protected-person confirmation, no natural-video anatomical reference study, and no matched-influence/convergence sweep or trained external-refiner comparison. Per-pair true response magnitudes, separate wrong/ambiguous/missing assignment components, and verified reconstructed-pose examples cannot be recovered from the compact packet.

The local folder is a complete, technically checked manuscript package, not evidence of likely acceptance or an actual submission. The portable source archive excludes machine-specific ledgers and local review work records; the full local audit links those records in the repository. The human authors have confirmed that they originated the ideas, wrote the initial drafts, completed the final edits and verification of the text, reported results and references, and approved the manuscript. The AI-use disclosure also records subsequent AI assistance. Submission metadata, eligibility, and submission-form disclosure remain author responsibilities. No submission or communication to others was performed.
