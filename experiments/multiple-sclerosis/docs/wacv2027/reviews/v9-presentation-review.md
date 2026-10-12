# Independent presentation and citation review of v9

Reviewed 2026-10-09: the complete `wacv2007-v9.tex`, all seven rendered PDF pages, the compiled bibliography, `references-v9.bib`, the new `evidence/v9-methods.md`, the existing methods and reproducibility records, and the build output. The workshop requirements were rechecked earlier in this revision against the live call and official author guide. No manuscript, figure, result, or notebook was edited during this review.

**Presentation pass.** The revision gives the work a more constructive and accessible emphasis while retaining the evidence needed to judge it. Its defensible positive result is that the implemented, label-free encoder and supervised classifier produce features usable for collection-label prediction on held-out source IDs, with a higher observed source-weighted score than the particular handcrafted RF procedure tested here. The manuscript does not establish that training caused this usefulness, that temporal order improved prediction, or that the features measure neurological impairment. It states those distinctions explicitly.

## Rubric

This is the same task-specific workshop-grounded rubric used for previous drafts, not an official conference score or an acceptance probability. Positive tone does not supply new experimental evidence, so the score remains at the established plateau.

| Category | Weight | Rating /5 | Weighted points | Assessment |
|---|---:|---:|---:|---|
| Workshop relevance and practical contribution | 15 | 4.0 | 12.0 | The application and its limited-data setting are clearer. Markerless behavioral analysis, evaluation and reproducibility fit the track; no new clinical measure or learning method is claimed. |
| Empirical correctness and uncertainty | 25 | 4.7 | 23.5 | Scores and their direction remain correct. The main paper preserves the recording/clip distinction, four-of-five-fold mean-pose result, and source-resampling ranges; exact details survive in the companion. |
| Evaluation design and generalization scope | 20 | 3.8 | 15.2 | The positive interpretation remains confined to held-out source IDs and the tested procedures. Unknown participant overlap, one seed, and absent causal controls remain limits. |
| Reproducibility and methodological completeness | 15 | 4.5 | 13.5 | The main explanation is easier to follow and points to exact evaluation formulas and the retained audit. It does not supply new acquisition metadata or exact neural replay guarantees. |
| Clinical interpretation, provenance and ethics | 10 | 4.0 | 8.0 | Supplied labels are explicitly unverified. The discussion retains the unknown consent, rights and ethics status and excludes individual clinical use. |
| Literature, clarity and WACV presentation | 15 | 4.5 | 13.5 | All 13 retained citations render correctly. The positive framing is readable, tables and figures are clear, and the seven-page review PDF follows the retained official format. |
| **Total** | **100** | | **85.7 / 100** | **Presentation pass; no forced score increase.** |

## Framing and accessibility

The title and abstract now emphasize what was implemented and measured. The opening sentence explains the purpose of learned features without making a novelty claim. Training size is concrete: 45–58 clips from 24 source recordings per outer split, consistent with the partition table. The encoder's label-free training is distinguished from the separate classifier's use of collection labels. The abstract also identifies the unverified labels and the RF correction, so its favorable comparison is not presented as a validated diagnostic result.

“Observed improvement” is explicitly attached to the equal-recording score and the evaluated RF. In this setting it describes the measured difference between two complete procedures, rather than a causal effect of training. The Discussion reinforces that interpretation by stating that feature construction, classifier choice and stage selection differ, and by defining useful features as information supporting prediction of the supplied labels. The untrained-encoder and frame-order comparisons remain unperformed follow-up experiments. This is a defensible way to make the contribution positive without inventing superiority over a tuned clinical gait system.

The simpler scoring account works for an educated researcher outside machine learning. It explains what F1 balances, the meaning of a perfect score, and its difference from accuracy. The thirteen-clips-at-one-thirteenth example gives the source-weighting choice a concrete interpretation. Calling the resampling output the middle 95% of recalculated differences avoids relying on technical vocabulary alone, while the main Results still say that models are held fixed and the ranges do not establish performance in a new clinical population. The companion preserves replacement sampling, the exact seed, label-specific counts, paired draws, occurrence weights, percentile definition, and the assumptions that the simplified text omits.

The Results do not hide the countervailing comparisons. They explicitly report that mean pose wins in four of five test groups and under equal-clip weighting, that the historical RF is slightly ahead of S-JEPA before rounding on clip-weighted F1, and that mean pose and RF each classify 37 clips correctly versus 36 for S-JEPA. Both paired source-resampling ranges span zero. The historical feature defect and its separately reproduced correction retain distinct descriptions and scores. No new learning curve, statistical significance, clinical gain or independent-seed replication is implied.

The reflection/mask correspondence issue remains stated in the Discussion, including the fact that a target physical joint can remain visible in the opposite-side input slot and that its classification effect was not measured. The manuscript does not silently convert this run into a corrected masking experiment. Its anatomical rule is still a design hypothesis without an ablation.

## Two optional wording refinements

Neither point requires a new experiment or a negative abstract. They would make the positive summary harder to overread:

1. **Include a brief weighting qualifier in the abstract's closing interpretation.** The abstract correctly says that the observed RF improvement uses equal-source scoring, but the reader sees no reminder that this ordering does not hold under equal-clip scoring. A short clause such as “the ordering depends on whether recordings or clips receive equal weight” would preserve the positive result while summarizing an important observation. An interval or a detailed limitations paragraph is unnecessary there.
2. **Prefer “auditable” to an unqualified “reproducible” in the contribution paragraph.** The sentence “The documented processing and evaluation also make the comparison reproducible” is reasonable for the saved-prediction audit, but can sound like a guarantee of end-to-end neural replay. The companion correctly records nondeterministic execution and incomplete historical source reconstruction. “Make the comparison auditable” communicates the demonstrated strength precisely, especially when the Overleaf ZIP will contain manuscript sources rather than the video collection and training environment.

These are small editorial refinements within the current scoring band. No further figure redesign or new statistical language is requested.

## Citation retention and claim scope

The source contains exactly **13 unique citation keys**, and all 13 appear as compiled bibliography items. The v9 bibliography file is byte-for-byte identical to the verified v8 file. The rendered list has the correct S-JEPA authors, Mohamed Abdelfattah and Alexandre Alahi, the correct ECCV 2024 entry, the BlazePose arXiv identifier, and the previously checked full author lists, titles, venues, years and pagination. No citation was dropped, fabricated or silently replaced during the positive rewrite.

The associated claims remain appropriate. The clinical studies motivate movement measurement and its validation without validating this collection's labels. I-JEPA, V-JEPA and S-JEPA support the predictive-learning lineage; SkeletonMAE and MotionBERT support the broader skeleton/motion literature. SkeletonMAE remains correctly described as graph-based reconstruction. The selection/evaluation separation statement is supported by the retained cross-validation literature. No cited study is used to justify this particular twelve-landmark weighting scheme as clinically proven.

## Full rendered-PDF inspection

All seven pages were inspected, including the new title and abstract, two-column text flow, all four tables, both figures, the shortened statistical account, and the complete references. No clipped text, intersecting panels, overlapping captions, missing glyph, or unreadable figure annotation was observed. Figure 1 has matching rectangular panels within each row and straight connectors. Figure 2 pairs the two methods within each fold through straight horizontal segments and uses consistent confusion-matrix geometry. Fold identifiers remain 0–4 in both the figure and partition table.

The manuscript body and bibliography fit in seven US-letter pages, within the regular-paper limit of eight body pages excluding references. It retains the official WACV 2027 review style, line numbers, anonymous author block and draft ID. The draft ID is a preparation placeholder, not an assigned submission identifier. The build reports no undefined citation/reference or overfull box. The existing `lineno.sty` UTF-8 warning remains a build-package warning with no observed damaged manuscript character.

PDF metadata lists the LaTeX/hyperref creator and PDF producer without an author identity. The manuscript, v9 methods record, learning-method record and reproducibility record contain no checked absolute local-user path (`/Users/`, `pmui`, or `alexpose`). This inspection covers those files; it is not a claim that every archived audit file belongs in an anonymous submission.

## Reviewed file identities

| File | SHA-256 |
|---|---|
| `wacv2007-v9.tex` | `d09bc20708543cafa8d76f22efb3f78d479719a507ce8a23ab711bf1afc5ba43` |
| `wacv2007-v9.pdf` | `bb66d7115a3008d51e0496a717d5079c904e17fc097e1d1b33d8360b09010212` |
| `evidence/v9-methods.md` | `528dc47fb0bf91a19c25126416af99e07affb9fd9511e2a342ab6eaf5116f8d4` |

If the optional wording changes are applied, record the resulting build as a follow-up rather than pretending these hashes identify the changed files. The Overleaf ZIP should be separately checked for complete compile dependencies and faithful manuscript text; it has not been reviewed in this document.


## Host finalization note

After this presentation review, the final text explicitly names the duplicated RF feature, narrows the reproducibility claim, shortens the abstract opening and removes a secondary interpretation of training-batch errors. The source/PDF identities recorded above refer to this review's earlier snapshot. The final seven-page source and PDF were independently rechecked in `reviews/v9-review.md` and `qa/v9-independent-checks.json`. The exact final differences are retained in `qa/v9-final-editorial-diff.txt`; no result, citation or figure changed.
