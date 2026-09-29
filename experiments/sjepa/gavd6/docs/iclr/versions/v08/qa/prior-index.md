# ICLR manuscript: seven reviewed versions

**Start with [Version 7 PDF](versions/v07/paper-v07-final.pdf) or its [editable LaTeX source](versions/v07/paper-v07-final.tex).** The title is **Evaluating Feature Prediction for 2D Pose Trajectory Restoration with Paired Synthetic Supervision**.

Each delivered PDF has exactly **nine main-text pages**, followed by the mandatory AI disclosure and references (11 pages total). The papers use the official 2027 style, embedded Times-family text, vector figures, and verified primary-paper citations. Original manuscript files outside this new `docs/iclr` folder were left untouched.

The strongest supported framing is a controlled evaluation of side-sensitive movement restoration: a lower signed-response error can coexist with worse geometric left–right assignment and angular trajectories. The tested feature-prediction variants do not establish an incremental advantage on their declared primary comparisons. The final text also exposes the unequal initial gradient influence hidden by a shared auxiliary coefficient.

## Folder layout

- `versions/v01/` … `versions/v07/`: the source/PDF pair from review and its final-typeset companion, sorted by revision; compiler logs live in each version's `logs/` folder.
- `versions/initial/`: the preliminary unnumbered source previously stored under `scripts/`.
- `figures/`, `references.bib`, `template/`: shared manuscript inputs, including retained version-specific figures.
- `reviews/`, `evidence/`: independent review rounds, scores, and supporting audits.
- `qa/v01/` … `qa/v07/`: page previews and contact sheets, grouped by revision. Final validation and visual-review summaries remain at `qa/`.
- `scripts/`: figure builders, review aggregation, and artifact validation.

[Relocation map](organization-map.json) records every moved file and its original SHA-256. Historical review text and evidence receipts retain the paths used when the reviews were written; resolve those paths through this map. Manuscript sources, PDFs, reviews, evidence, figures, and compiler logs retain their original contents. See [REPRODUCE.md](REPRODUCE.md) for compiling the relocated sources against the shared inputs.

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

These are seven successive scientific manuscripts, not seven independent studies. The `-final` suffix identifies the final typesetting of each historical version; **only version 7 is the final scientific recommendation**. Earlier versions deliberately retain the scientific weaknesses documented in their reviews. Sources/PDFs without that suffix preserve the original review-stage renders; the final companions add correct font encoding, complete citations, and a shorter equivalent table caption without changing results.

## Reviews and evidence

- [Integrated ICLR review record](REVIEW-RECORD.md): all 21 independent panel records, four requested perspectives in every round, scores, objection severity/evidence/corrections, and residual limits.
- [Claim ledger](CLAIM-LEDGER.md): source paths for the paper’s quantitative and methodological statements.
- [Exact reproduction contract](evidence/reproducibility-methods.md) and [reproduction commands](REPRODUCE.md).
- [Independent evidence audit](reviews/evidence-initial.md), including 69 transferred-file hash checks and 16 reconstructed saved comparisons.
- [Literature and submission review](reviews/literature-initial.md), with primary papers and official implementation links.
- [Final validation](qa/final-validation.json) and [visual review](qa/final-visual-review.md).
- [Machine-readable scores](reviews/scores.csv), [full scoring details](reviews/scores.json), and editable figure builders under `scripts/`.

## Interpretation of the final score

**75.75/100** is the arithmetic synthesis of the independent reviewers’ fixed-rubric scores, not an acceptance probability. Version 7 deliberately has the same score as version 6. The final adversarial passes found no unresolved material misstatement within the exported-development-evidence scope.

The largest remaining weakness is empirical: 14 repeatedly used development people, three seeds, no completed independent-person confirmation, no natural-video anatomical reference evaluation, and limited optimization/external-baseline coverage. The evidence/statistical reviewer therefore keeps evaluation rigor at 5/10. New wording cannot remove those limits.

## Submission boundary

Formatting follows the [official ICLR 2027 author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines), and the disclosure reflects the [author AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors). These files are local manuscript artifacts; no submission or communication to others was made. The human authors must verify scientific content, prior abstract eligibility, author metadata, disclosure, and any anonymous supplementary release before submission. Working audit records contain local paths and should not be uploaded as an anonymous supplement without preparation.
