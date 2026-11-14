# Original-paper revision guide

This review addresses `docs/wacv2027-v1/wacv-v1.pdf` and the current notebooks. It was prepared on 15 October 2026 without changing the original paper, notebooks, model code or existing Overleaf package. All new review outputs are here, as requested.

Start with [the prioritized revision map](01-priority-revisions.md), then use [the Overleaf replacement passages](02-overleaf-replacements.md). The latter supplies exact locations, full replacement sections, a corrected results table and figure, and supplementary text. Each passage is also available as a separate [LaTeX fragment](latex/).

The main correction is the Random Forest baseline: the draft reports 0.411 from an earlier extractor with a duplicated ankle feature; the corrected execution scores 0.325. S-JEPA remains at 0.457. The revision foregrounds useful trained representations from 24 training source videos per fold, while retaining mean pose at 0.452 and the missing controls needed to identify why the representation works.

- [Evidence and submission notes](03-evidence-and-submission.md): verified results, reproducible checking command and claim boundaries.
- [Review disposition](04-review-disposition.md): adversarial findings and their resolution.
- [Notebook audit](evidence/notebook-methods-audit.md), [results audit](evidence/results-audit.md), and [paper/venue audit](evidence/paper-and-venue-audit.md): detailed supporting evidence.

The live workshop call and author kit were checked. The regular-paper deadline is 16 October 2026 at 23:59 AoE; consult the linked [venue requirements](evidence/paper-and-venue-audit.md#venue-requirements-and-fit) for the submission route and format. Nothing has been submitted.

The review files contain internal provenance and are for author use. Upload only the revised anonymous manuscript and chosen supplementary material, after rebuilding and checking the complete Overleaf project.
