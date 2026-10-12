# ICLR manuscript revisions

**Current revision: [Version 08 paper](versions/v08/paper-v08.pdf), [Overleaf upload ZIP](versions/v08/paper-v08-overleaf.zip), [source archive](versions/v08/paper-v08-source.zip), and [complete package index](versions/v08/README.md).**

The title is **Evaluating JEPA-Inspired Motion Representations through Geometry and Gait Asymmetry**. Version 08 contains nine main pages, disclosure/references on pages 10–11, and an implementation/results appendix on pages 12–23. Earlier versions are preserved.

Version 08 centers the zero-response comparison and the unresolved feature-prediction benefit, adds complete exploratory observation/cost analyses, distinguishes joint training from frozen readouts, and substantially rebuilds all six figures. The [revision record](versions/v08/REVISION-RECORD.md) explains the changes, independent objections and dispositions, and the earlier fixed-rubric increase from **75.75 to 79.33/100** (before the in-place framing revision). This score is not an acceptance probability.

## Version index

|Version|Paper|Editable source|Main pages|Fixed-rubric synthesis /100|
|---|---|---|---:|---:|
|[v01](versions/v01/README.md)|[PDF](versions/v01/paper-v01-final.pdf)|[LaTeX](versions/v01/paper-v01-final.tex)|9|65.75|
|[v02](versions/v02/README.md)|[PDF](versions/v02/paper-v02-final.pdf)|[LaTeX](versions/v02/paper-v02-final.tex)|9|69.00|
|[v03](versions/v03/README.md)|[PDF](versions/v03/paper-v03-final.pdf)|[LaTeX](versions/v03/paper-v03-final.tex)|9|72.00|
|[v04](versions/v04/README.md)|[PDF](versions/v04/paper-v04-final.pdf)|[LaTeX](versions/v04/paper-v04-final.tex)|9|74.58|
|[v05](versions/v05/README.md)|[PDF](versions/v05/paper-v05-final.pdf)|[LaTeX](versions/v05/paper-v05-final.tex)|9|75.42|
|[v06](versions/v06/README.md)|[PDF](versions/v06/paper-v06-final.pdf)|[LaTeX](versions/v06/paper-v06-final.tex)|9|75.75|
|[v07](versions/v07/README.md)|[PDF](versions/v07/paper-v07-final.pdf)|[LaTeX](versions/v07/paper-v07-final.tex)|9|75.75|
|[v08](versions/v08/README.md)|[PDF](versions/v08/paper-v08.pdf)|[LaTeX](versions/v08/paper-v08.tex)|9|79.33|

These are successive manuscripts based on the same completed study, not independent experiments. Versions 01–07 each have 11 total pages; version 08 adds a substantial appendix. The historical `-final` suffix describes typesetting within that version. Version 08 is the current scientific recommendation.

## Evidence, review, and reproduction

- [Version 08 evidence audit](versions/v08/evidence/audit.md), [claim-to-artifact map](versions/v08/evidence/manuscript-claims.json), and [numerical checks](versions/v08/evidence/final-numerical-checks.json).
- [Version 08 independent reviews and score synthesis](versions/v08/REVISION-RECORD.md), [structural validation](versions/v08/qa/validation.json), and [visual review](versions/v08/qa/clarity-review.json).
- [Version 08 build commands and source package](versions/v08/README.md). Its portable archive regenerates figures/tables and compiles the paper from included audited data; the complete upstream audit uses the retained repository exports.
- [Historical integrated review record](REVIEW-RECORD.md): all 21 v01–v07 panel records, original scores and dispositions.
- [Historical claim ledger](CLAIM-LEDGER.md), [exact reproduction contract](evidence/reproducibility-methods.md), and [v01–v07 build instructions](REPRODUCE.md).
- [Relocation map](organization-map.json): historical moved-file paths and hashes. The v08 validator confirms all 212 recorded files retain their original bytes.

`versions/v08` owns its source, bibliography, unchanged official template, figures, analyses, reviews, tables, scripts, and QA. The shared top-level `figures`, `reviews`, `evidence`, `scripts`, and `template` directories remain the historical v01–v07 package. Historical review text may use pre-relocation paths; resolve them through the relocation map.

## Remaining scientific and submission boundaries

The evidence still comprises 14 repeatedly inspected development people and three fitted seeds. There is no completed independent-person confirmation, natural-video anatomical-reference study, matched-influence/convergence sweep, or trained external-refiner comparison. Raw frames, reconstructed poses, and full checkpoints are incomplete in the compact packet. Editing and reaggregation do not remove these limits.

The current official ICLR 2027 requirements were reverified for v08; see its [submission-rule record](versions/v08/reviews/submission-rules.md). No paper was submitted and no message was sent to others. Human authors have confirmed completion of final editing, verification, and manuscript approval; the paper records these contributions alongside AI assistance. Administrative eligibility, submission metadata, and form disclosure remain separate. Local working audit records contain machine-specific provenance and are not automatically an anonymous public supplement; the portable paper-source archive excludes those ledgers and review work records.
